"""Serial material and full-step measurements at one, 24 and 192 tetrahedra."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from phymeo import (
    ParticleState,
    SceneInputs,
    SceneState,
    StepInputs,
    TetrahedralModel,
    make_tetrahedral_runtime,
    tetrahedral_block,
)
from phymeo.experiments.rigid_cloth import provenance
from phymeo.materials.stvk import elastic_energy, elastic_forces


def stats(values):
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}


def models():
    unit = TetrahedralModel(
        np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]]),
        np.array([[0, 1, 2, 3]]),
        6.0,
        1000.0,
        0.3,
        np.array([True, False, False, False]),
        np.zeros(3),
    )
    return [
        unit,
        *(
            tetrahedral_block((0.0, 0.0, 0.0), (0.1, 0.1, 0.1), shape=shape)[0]
            for shape in ((3, 2, 3), (5, 3, 5))
        ),
    ]


def measure(model):
    position = model.rest_position.copy()
    position[:, 1] *= 0.99
    native = ParticleState(position, np.zeros_like(position))
    top = model.rest_position[:, 1] == model.rest_position[:, 1].max()
    force = np.zeros_like(position)
    force[top, 1] = -0.05 / np.count_nonzero(top)
    state, inputs = SceneState({"soft": native}), SceneInputs({"soft": StepInputs(force)})
    start = perf_counter_ns()
    runtime = make_tetrahedral_runtime(model)
    build_ms = (perf_counter_ns() - start) / 1e6
    operations = {
        "forces": lambda: elastic_forces(model, native.position),
        "energy": lambda: elastic_energy(model, native.position),
        "step": lambda: runtime.step(state, inputs, 1 / 4000),
    }
    results = {}
    with runtime:
        for name, function in operations.items():

            def call():
                value = function()
                if name == "step" and not value.report.accepted:
                    raise RuntimeError(value.report.reason)
                return value

            start = perf_counter_ns()
            call()
            first_ms = (perf_counter_ns() - start) / 1e6
            for _ in range(2):
                for _ in range(100):
                    call()
            repeats, phases, quality = [], {}, []
            for repeat in range(3):
                times = []
                for _ in range(100):
                    start = perf_counter_ns()
                    value = call()
                    times.append((perf_counter_ns() - start) / 1e6)
                    if name == "step":
                        report = value.report.substeps[0]
                        d = report.diagnostics
                        quality.append(
                            [
                                d["integration_residual"],
                                d["min_det_f"],
                                d["min_singular_value"],
                                d["max_singular_value"],
                            ]
                        )
                        for phase, duration in report.timings.items():
                            phases.setdefault(phase, []).append(duration * 1000)
                repeats.append({"repeat": repeat + 1, "calls": 100, **stats(times)})
            results[name] = {
                "first_call_ms": first_ms,
                "repeats": repeats,
                "phases": {key: stats(values) for key, values in phases.items()},
            }
            if quality:
                q = np.array(quality)
                results[name]["quality"] = {
                    "accepted_steps": len(q),
                    "max_integration_residual": float(q[:, 0].max()),
                    "min_det_f": float(q[:, 1].min()),
                    "min_singular_value": float(q[:, 2].min()),
                    "max_singular_value": float(q[:, 3].max()),
                }
    return {
        "nodes": model.particle_count,
        "tetrahedra": len(model.tetrahedra),
        "fixed_nodes": int(model.fixed.sum()),
        "volume": float(model.volume.sum()),
        "density": model.density,
        "young_modulus": model.young_modulus,
        "poisson_ratio": model.poisson_ratio,
        "runtime_construction_ms": build_ms,
        "operations": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    report = {
        "status": "running",
        "stage": "P4",
        "provenance": provenance(),
        "measurement_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "processor": platform.processor(),
        "workloads": [],
        "protocol": {
            "dtype": "float64",
            "dt": 1 / 4000,
            "substeps": 1,
            "serial": True,
            "initial_strain": "uniform y compression .99; same initial snapshot each call",
            "load": "total -.05 N along Y on maximum-y nodes",
            "warmup_batches": 2,
            "warmup_calls_per_batch": 100,
            "repeats": 3,
            "calls_per_repeat": 100,
        },
        "unmeasured": [
            "process import and isolated cold cache",
            "peak memory/allocation counts",
            "GPU/tyx/Warp",
            "independent stability boundary or timestep convergence",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for model in models():
            report["workloads"].append(measure(model))
        report.update(status="complete", failures=0)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(
        json.dumps(
            {str(w["tetrahedra"]): w["operations"]["step"] for w in report["workloads"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
