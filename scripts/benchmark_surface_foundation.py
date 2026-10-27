"""Serial tiny operator observations, including host return and correctness checks.

This is not a simulation-step benchmark or a CPU/GPU/Warp speedup comparison.
"""

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from phymeo import (
    BackendContext,
    ParticleModel,
    ParticlePointPort,
    ParticleState,
    PointPairOperator,
    RobotInputs,
    RobotPointPort,
    RobotState,
    StepInputs,
    StepSettings,
    TyxClothParticipant,
    TyxRobotParticipant,
    WeightedPointPort,
    closest_point_triangle,
    closest_segment_segment,
    pendulum_chain,
)


def observe(operation, samples, repeats):
    start = perf_counter()
    operation()
    first = perf_counter() - start
    for _ in range(2):
        operation()
    batches = []
    for _ in range(repeats):
        elapsed = []
        for _ in range(samples):
            start = perf_counter()
            operation()
            elapsed.append(perf_counter() - start)
        batches.append(
            {
                "seconds": elapsed,
                "median": float(np.median(elapsed)),
                "p95": float(np.percentile(elapsed, 95)),
            }
        )
    return {
        "first_call_seconds": first,
        "warmups": 2,
        "batches": batches,
        "accepted_measured_calls": samples * repeats,
        "failed_calls": 0,
    }


def port_case(kind, mode, samples, repeats):
    start = perf_counter()
    ws = None
    if kind == "nodes":
        model = ParticleModel(np.array([1.0, 2, 0]), np.zeros(3))
        state = ParticleState(np.eye(3), np.zeros((3, 3)))
        b = np.array([[0.5, 0.5, 0], [0, 0.25, 0.75]])
        expected_response = 0.375
        expected_velocity = np.array([[0.05, 0, 0], [0.05, 0, 0], [0, 0, 0]])
        indices, local = np.arange(3), None
        factory = TyxClothParticipant(model, None, mode) if mode != "numpy" else None
        inputs = StepInputs.zeros(3)
        base = ParticlePointPort(model, state, indices) if mode == "numpy" else None
    else:
        model, _ = pendulum_chain(2)
        model = replace(model, gravity=np.zeros(3))
        state = RobotState(np.zeros(2), np.zeros(2))
        b = np.array([[1.0, 0], [0.25, 0.75]])
        j = np.array([-0.375, -0.375])
        mass = np.array([[2 / 3, 5 / 24], [5 / 24, 1 / 12]])
        expected_velocity = np.linalg.solve(mass, j * 0.1)
        expected_response = float(j @ expected_velocity) / 0.1
        indices, local = np.arange(2), np.array([[0.0, -0.5, 0], [0.0, -0.5, 0]])
        factory = TyxRobotParticipant(model, mode) if mode != "numpy" else None
        inputs = RobotInputs.zeros(2)
        base = RobotPointPort(model, state, indices, local) if mode == "numpy" else None
    try:
        if mode != "numpy":
            ws = factory.create_workspace(
                "body",
                state,
                inputs,
                0.01,
                StepSettings(tolerance=2e-6),
                BackendContext("tyx:" + mode, np.float64),
            )
            ws.predict()
            base = ws.point_port(indices, local)
        port = WeightedPointPort(base, b)
        pair = PointPairOperator(port, np.array([[0, 1]]))
        construction = perf_counter() - start
        p = np.array([[0.1, 0, 0]])

        def operation():
            response = pair.response(p)
            candidate = pair.apply_impulses(p)
            velocity = candidate.velocity if kind == "nodes" else candidate.v
            np.testing.assert_allclose(response, p * expected_response, atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(velocity, expected_velocity, atol=2e-5, rtol=2e-5)

        before = dict(ws.diagnostics()["execution"]) if ws is not None else None
        result = observe(operation, samples, repeats)
        after = dict(ws.diagnostics()["execution"]) if ws is not None else None
        if ws is not None and after["completed_launches"] <= before["completed_launches"]:
            raise RuntimeError("no actual backend execution during measurement")
        result.update(
            kind=kind,
            mode=mode,
            setup_and_predict_seconds=construction,
            execution_before=before,
            execution_after=after,
        )
        return result
    finally:
        if ws is not None:
            ws.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("numpy", "interpret", "cpu_cpp", "cuda"), required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1:
        parser.error("samples/repeats must be positive")
    root = Path(__file__).resolve().parents[1]
    hashes = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "src/phymeo").rglob("*.py"))
    }
    hashes["scripts/benchmark_surface_foundation.py"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    result = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "working_tree": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "source_hashes": hashes,
        "scope": "host FP64 maps; base backend explicit; no time step",
        "protocol": "one first call, 2 warmups, serial repeats; response+candidate+host assertions",
        "samples": args.samples,
        "repeats": args.repeats,
        "not_measured": [
            "cold process/JIT",
            "bytes",
            "peak memory",
            "GPU kernel-only time",
            "Warp",
        ],
    }
    result["ports"] = [
        port_case(kind, args.mode, args.samples, args.repeats) for kind in ("nodes", "robot")
    ]
    if args.mode == "numpy":

        def geometry():
            face = closest_point_triangle(
                np.array([0.5, 0.5, 1.0]), np.array([[0.0, 0, 0], [2.0, 0, 0], [0.0, 2, 0]])
            )
            edge = closest_segment_segment(
                np.array([[-1.0, 0, 0], [1.0, 0, 0]]), np.array([[0.0, -1, 1], [0.0, 1, 1]])
            )
            if abs(face.distance - 1) > 1e-12 or abs(edge.distance - 1) > 1e-12:
                raise RuntimeError("geometry analytic check failed")

        result["geometry"] = observe(geometry, args.samples, args.repeats)
    for name, digest in hashes.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("source changed during observation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved complete serial observation to {args.output}")


if __name__ == "__main__":
    main()
