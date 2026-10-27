"""Whole-scene device comparison, repetition and FP64 sensitivity evidence."""

import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np

from .rigid_cloth import SceneRunFailure, load_run, provenance, run_scene, save_run


def _position_difference(reference, candidate):
    output = {}
    np.testing.assert_allclose(
        reference.arrays["time"], candidate.arrays["time"], atol=2e-6, rtol=0
    )
    for family, name in (("rigid", "sphere"), ("particle", "cloth")):
        a = reference.arrays[family + "_position"].astype(np.float64)
        b = candidate.arrays[family + "_position"].astype(np.float64)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("trajectory position data must have matching finite shapes")
        output["max_" + name + "_position_difference_m"] = float(
            np.max(np.linalg.norm(a - b, axis=-1))
        )
    return output


def compare_backend_runs(reference, candidate):
    """Compare the same colored experiment, allowing only arithmetic/backend changes."""
    if reference.config["backend"] != "numpy_colored" or candidate.config["backend"] not in (
        "interpret",
        "cpu_cpp",
        "cuda",
    ):
        raise ValueError("expected colored NumPy reference and explicit device backend")
    if reference.config["dtype"] != "float32" or candidate.config["dtype"] != "float32":
        raise ValueError("matched device comparison requires FP32 storage")
    omit = {"backend", "projection_arithmetic"}
    if {k: v for k, v in reference.config.items() if k not in omit} != {
        k: v for k, v in candidate.config.items() if k not in omit
    }:
        raise ValueError("physical configurations and iteration controls must match")
    if candidate.metrics["execution_backend"] != f"tyx:{candidate.config['backend']}":
        raise ValueError("record does not prove the requested backend executed")
    np.testing.assert_array_equal(reference.arrays["time"], candidate.arrays["time"])
    difference = _position_difference(reference, candidate)
    if max(difference.values()) > 2e-3:
        raise AssertionError("matched FP32 position difference exceeds 2e-3 m")
    return difference


def _device_environment(mode):
    result = {"tyx_version": importlib.metadata.version("tyx"), "mode": mode}
    if mode == "cuda":
        try:
            output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            result["nvidia_smi"] = (
                output.stdout.strip() if output.returncode == 0 else "unavailable"
            )
        except (OSError, subprocess.TimeoutExpired):
            result["nvidia_smi"] = "unavailable"
    return result


def run_backend_suite(
    destination,
    *,
    frames=240,
    dt=1 / 120,
    mode="cuda",
    tolerance=1e-6,
    iterations=200,
    cache_root=None,
    fp64_reference=None,
    progress=None,
):
    if mode not in ("interpret", "cpu_cpp", "cuda"):
        raise ValueError("backend suite requires an explicit tyx mode")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    source = provenance()
    runs = {}
    try:
        environment = _device_environment(mode)
        controls = dict(
            frames=frames,
            dt=dt,
            tolerance=tolerance,
            iterations=iterations,
            dtype=np.float32,
            cache_root=cache_root,
        )
        for name, backend in (
            ("numpy_colored", "numpy_colored"),
            ("device", mode),
            ("repeat", mode),
        ):
            callback = (
                None
                if progress is None
                else lambda frame, report, name=name: progress(name, frame, report)
            )
            run = run_scene(**controls, mode=backend, progress=callback)
            save_run(run, destination / name, source=source)
            runs[name] = run
        reference, candidate, repeat = runs["numpy_colored"], runs["device"], runs["repeat"]
        comparison = compare_backend_runs(reference, candidate)
        for name in candidate.arrays:
            np.testing.assert_array_equal(candidate.arrays[name], repeat.arrays[name])
        if candidate.metrics["trajectory_sha256"] != repeat.metrics["trajectory_sha256"]:
            raise SceneRunFailure("device repetition digest differs")
        if min(run.metrics["contact_substeps"] for run in runs.values()) < 1:
            raise SceneRunFailure("backend suite did not exercise contact in every run")
        if fp64_reference is None:
            fp64 = run_scene(frames=frames, dt=dt)
            save_run(fp64, destination / "fp64_reference", source=source)
            fp64_source = {"generated_in_suite": True}
        else:
            fp64 = load_run(fp64_reference)
            fp64_source = json.loads(
                (Path(fp64_reference) / "run.json").read_text(encoding="utf-8")
            )["provenance"]
        physical = (
            "scene",
            "rows",
            "columns",
            "spacing_m",
            "density_kg_m2",
            "structural_compliance_m_N",
            "shear_compliance_m_N",
            "cloth_origin_m",
            "pinned",
            "sphere_mass_kg",
            "sphere_radius_m",
            "sphere_position_m",
            "sphere_velocity_mps",
            "gravity_mps2",
            "contact",
            "frames",
            "requested_frame_dt",
            "substeps",
            "interaction",
        )
        if fp64.config["dtype"] != "float64" or any(
            fp64.config[k] != candidate.config[k] for k in physical
        ):
            raise ValueError("FP64 reference must match the physical scene and requested horizon")
        record = {
            "status": "complete",
            "provenance": source,
            "device_environment": environment,
            "matched_fp32": comparison,
            "device_repetition_identical": True,
            "fp64_difference": _position_difference(fp64, candidate),
            "fp64_reference": {
                "config": fp64.config,
                "metrics": fp64.metrics,
                "provenance": fp64_source,
            },
            "runs": {
                name: {"config": run.config, "metrics": run.metrics} for name, run in runs.items()
            },
        }
        (destination / "suite.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return record
    except Exception as error:
        (destination / "failure.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(error),
                    "completed_runs": list(runs),
                    "provenance": source,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
