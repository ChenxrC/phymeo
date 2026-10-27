"""Serial controlled-pair measurements, with quality gates and public tyx counters."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from phymeo import (
    CoupledInputs,
    CoupledState,
    ParticleModel,
    ParticleState,
    RigidBodyModel,
    RigidBodyState,
    SphereShapes,
    make_coupled_runtime,
)
from phymeo.experiments.rigid_cloth import provenance


def statistics(values):
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}


def hardware():
    result = {"processor": platform.processor(), "logical_cpus": os.cpu_count()}
    try:
        result["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError) as error:
        result["gpu_query_error"] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("numpy", "cuda"), required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    if args.output.exists() or min(args.samples, args.repeats, args.warmups) < 1:
        parser.error("output must be new and all sample counts positive")
    cache = args.cache_root or Path("artifacts/runtime-measure-cache") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-%f"
    )
    if cache.exists():
        parser.error("use a new cache directory to identify the cold first step")
    dtype = np.float32 if args.mode == "cuda" else np.float64
    tolerance = 1e-6 if args.mode == "cuda" else 1e-7
    model_start = time.perf_counter_ns()
    pm = ParticleModel(np.ones(1, dtype), np.zeros(3, dtype))
    bm = RigidBodyModel(np.array([0.5], dtype), np.full((1, 3), 2, dtype), np.zeros(3, dtype))
    state = CoupledState(
        ParticleState(np.array([[0, 1, 0]], dtype), np.array([[0, -1, 0]], dtype)),
        RigidBodyState(
            np.zeros((1, 3), dtype),
            np.array([[0, 0, 0, 1]], dtype),
            np.zeros((1, 3), dtype),
            np.zeros((1, 3), dtype),
        ),
    )
    shapes, inputs = SphereShapes(np.ones(1, dtype)), CoupledInputs.zeros(1, 1, dtype=dtype)
    model_ms = (time.perf_counter_ns() - model_start) / 1e6
    source = provenance()
    tx = None
    dependency = None
    if args.mode == "cuda":
        import tyx as tx

        distribution = importlib.metadata.distribution("tyx-compute")
        dependency = {
            "version": distribution.version,
            "source": json.loads(distribution.read_text("direct_url.json")),
        }
    report = {
        "status": "running",
        "provenance": source,
        "hardware": hardware(),
        "tyx": dependency,
        "workload": {
            "mode": args.mode,
            "dtype": np.dtype(dtype).name,
            "particles": 1,
            "bodies": 1,
            "distance_constraints": 0,
            "potential_contacts": 1,
            "dt": 0.1,
            "substeps": 4,
            "iterations": 200,
            "tolerance": tolerance,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "samples_per_repeat": args.samples,
            "state_policy": "reset to same closing-contact initial state on every call; not an evolving trajectory",
        },
        "cache": {
            "path": str(cache),
            "initial_state": "new directory; process/toolchain/OS caches may already be warm",
        },
        "model_build_ms": model_ms,
        "results": [],
        "failures": 0,
        "limits": [
            "phase timers are host wall time and retain validation; not kernel-only profiling",
            "full step includes synchronized workspace close and adapter conversion",
            "peak host/device memory and allocation timing unmeasured",
            "phase remainder includes outer ownership/identity/report work, not a measured pure-copy category",
            "no Warp comparison; NumPy FP64 and tyx FP32 are not a matched-arithmetic speedup comparison",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    creation = time.perf_counter_ns()
    try:
        with make_coupled_runtime(
            pm, bm, None, shapes, mode=args.mode, tolerance=tolerance, cache_root=cache
        ) as solver:
            report["runtime_construction_ms"] = (time.perf_counter_ns() - creation) / 1e6

            def sample():
                start = time.perf_counter_ns()
                result = solver.step(state, inputs, 0.1)
                milliseconds = (time.perf_counter_ns() - start) / 1e6
                if not result.report.accepted:
                    raise RuntimeError(result.report.reason)
                p, b = result.state.particles, result.state.rigid
                momentum_error = float(
                    np.max(
                        np.abs(
                            p.velocity.astype(float) + 2 * b.velocity.astype(float) - [[0, -1, 0]]
                        )
                    )
                )
                com_error = float(
                    np.max(
                        np.abs(
                            p.position.astype(float) + 2 * b.position.astype(float) - [[0, 0.9, 0]]
                        )
                    )
                )
                gap_error = float(
                    abs(np.linalg.norm(p.position.astype(float) - b.position.astype(float)) - 1)
                )
                if max(momentum_error, com_error, gap_error) > 2e-6:
                    raise AssertionError("independent closed-pair physics failed")
                phases = {}
                for substep in solver.last_scene_result.report.substeps:
                    for name, seconds in substep.timings.items():
                        phases[name] = phases.get(name, 0.0) + seconds * 1000
                phases["outer_remainder"] = milliseconds - sum(phases.values())
                return milliseconds, phases, (momentum_error, com_error, gap_error), result.report

            report["first_step_ms"] = sample()[0]
            for _ in range(args.warmups):
                sample()
            for repetition in range(args.repeats):
                before = None if tx is None else tx.runtime_stats()
                elapsed, phase_rows, quality = [], {}, np.zeros(3)
                residual, position_iterations, velocity_iterations = 0.0, 0, 0
                for _ in range(args.samples):
                    duration, phases, errors, outcome = sample()
                    elapsed.append(duration)
                    for name, value in phases.items():
                        phase_rows.setdefault(name, []).append(value)
                    quality = np.maximum(quality, errors)
                    residual = max(residual, outcome.max_residual)
                    position_iterations += outcome.iterations
                    velocity_iterations += sum(r.velocity_iterations for r in outcome.substeps)
                counters = None if tx is None else asdict(tx.runtime_stats() - before)
                row = {
                    "repeat": repetition + 1,
                    "accepted": args.samples,
                    **statistics(elapsed),
                    "total_ms": sum(elapsed),
                    "phases": {
                        k: {**statistics(v), "total_ms": sum(v)} for k, v in phase_rows.items()
                    },
                    "max_momentum_error_kg_m_s": quality[0],
                    "max_weighted_position_error_m": quality[1],
                    "max_contact_gap_error_m": quality[2],
                    "max_residual": residual,
                    "position_iterations": position_iterations,
                    "velocity_iterations": velocity_iterations,
                    "tyx_counters": counters,
                }
                report["results"].append(row)
                print(json.dumps({"repeat": repetition + 1, **statistics(elapsed)}), flush=True)
        report["status"] = "passed"
    except BaseException as error:
        report["status"], report["failures"], report["error"] = "failed", 1, str(error)
        raise
    finally:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
