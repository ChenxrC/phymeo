"""Serial dense-reference P3 measurements; correctness checks remain enabled."""

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from phymeo import RobotInputs, SceneInputs, SceneState, make_robot_runtime, pendulum_chain
from phymeo.dynamics.robot_dynamics import bias_forces, mass_matrix
from phymeo.dynamics.robot_kinematics import forward_kinematics
from phymeo.experiments.rigid_cloth import provenance


def stats(values):
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    source, workloads = provenance(), []
    for count in (1, 2, 8):
        start = perf_counter_ns()
        model, native = pendulum_chain(count)
        runtime = make_robot_runtime(model)
        construct_ms = (perf_counter_ns() - start) / 1e6
        state, inputs = (
            SceneState({"robot": native}),
            SceneInputs({"robot": RobotInputs.zeros(count)}),
        )
        operations = {
            "fk": lambda: forward_kinematics(model, native.q),
            "mass": lambda: mass_matrix(model, native.q),
            "bias": lambda: bias_forces(model, native),
            "step": lambda: runtime.step(state, inputs, 1 / 2000),
        }
        results = {}
        try:
            for name, function in operations.items():

                def call():
                    try:
                        value = function()
                        if name == "step" and not value.report.accepted:
                            raise RuntimeError(value.report.reason)
                        return value
                    except Exception as error:
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        with args.output.open("x", encoding="utf-8") as stream:
                            json.dump(
                                {
                                    "status": "failed",
                                    "provenance": source,
                                    "dof": count,
                                    "operation": name,
                                    "error": str(error),
                                    "completed_workloads": workloads,
                                    "completed_operations": results,
                                },
                                stream,
                                indent=2,
                            )
                        raise

                start = perf_counter_ns()
                call()
                first_ms = (perf_counter_ns() - start) / 1e6
                for _ in range(2):
                    for _ in range(100):
                        call()
                repeats, phases, residuals = [], {}, []
                for repeat in range(3):
                    times = []
                    for _ in range(100):
                        start = perf_counter_ns()
                        value = call()
                        times.append((perf_counter_ns() - start) / 1e6)
                        if name == "step":
                            row = value.report.substeps[0]
                            residuals.append(row.diagnostics["equation_residual"])
                            for phase, duration in row.timings.items():
                                phases.setdefault(phase, []).append(duration * 1000)
                    repeats.append({"repeat": repeat + 1, "calls": 100, **stats(times)})
                results[name] = {
                    "first_call_ms": first_ms,
                    "repeats": repeats,
                    "phase_timings": {key: stats(values) for key, values in phases.items()},
                    "accepted_steps": 300 if name == "step" else None,
                    "max_equation_residual": max(residuals) if residuals else None,
                }
        finally:
            runtime.close()
        workloads.append(
            {
                "dof": count,
                "build_ms": construct_ms,
                "q": native.q.tolist(),
                "v": native.v.tolist(),
                "operations": results,
            }
        )
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    report = {
        "status": "complete",
        "failed_calls": 0,
        "stage": "P3",
        "provenance": source,
        "measurement_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "hardware": {
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "blas_config": config.getvalue(),
            "thread_environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "protocol": {
            "serial": True,
            "warmup_batches": 2,
            "warmup_calls_per_batch": 100,
            "repeats": 3,
            "calls_per_repeat": 100,
            "dt": 1 / 2000,
            "substeps": 1,
            "dtype": "float64",
            "step_input": "same initial snapshot on each call, zero applied loads",
            "quality": "full production checks and diagnostics, no constraints, one sweep per stage",
        },
        "workloads": workloads,
        "limitations": [
            "fresh process import latency unmeasured; first call follows construction and earlier operation measurements",
            "peak memory and allocation counts unmeasured",
            "phase timings exclude outer snapshot and report construction; total step includes them",
            "no optimized CRBA/ABA, GPU, Warp or MuJoCo speed comparison",
            "no concurrent benchmark/development process intentionally launched; background OS scheduling uncontrolled",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {str(row["dof"]): row["operations"]["step"]["repeats"] for row in workloads}, indent=2
        )
    )


if __name__ == "__main__":
    main()
