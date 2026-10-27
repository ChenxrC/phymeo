"""Serial host control-law observations; not a robot time-step benchmark."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from benchmark_surface_foundation import observe

from phymeo import RobotState, pendulum_chain, solve_bounded_pd
from phymeo.dynamics.robot_dynamics import bias_forces, mass_matrix
from phymeo.experiments.rigid_cloth import provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, _ = pendulum_chain(2)
    state = RobotState(np.array([0.2, -0.3]), np.array([-0.1, 0.2]))
    mass, bias = mass_matrix(model, state.q), bias_forces(model, state)
    target, kp, kd, h = np.array([0.25, -0.2]), np.full(2, 6.0), np.full(2, 0.4), 0.001
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "src/phymeo/control").glob("*.py")) + [
        Path(__file__),
        root / "scripts/benchmark_surface_foundation.py",
    ]
    hashes = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    data = {
        "scope": "host FP64 held-effort law, frozen two-DOF M/b; no actual state step or device timing",
        "source": provenance(),
        "source_hashes": hashes,
        "cases": {},
        "not_measured": [
            "device integration",
            "contact",
            "controller history",
            "Warp",
            "cold process",
            "memory",
        ],
    }
    for name, limit in [("saturated", 0.1), ("unsaturated", 2.0)]:
        iterations = []
        residuals = []

        def operation():
            result = solve_bounded_pd(
                mass, bias, state.q, state.v, target, np.zeros(2), kp, kd, np.full(2, limit), h
            )
            np.testing.assert_allclose(
                result.effort,
                np.clip(
                    kp * (target - result.predicted_configuration) - kd * result.predicted_velocity,
                    -limit,
                    limit,
                ),
                atol=1e-10,
                rtol=0,
            )
            np.testing.assert_allclose(
                mass @ ((result.predicted_velocity - state.v) / h) + bias,
                result.effort,
                atol=1e-10,
                rtol=0,
            )
            iterations.append(result.iterations)
            residuals.append(result.residual_effort)

        record = observe(operation, 100, 3)
        record.update(
            iterations_min_max=[min(iterations), max(iterations)],
            max_effort_residual=max(residuals),
        )
        data["cases"][name] = record
    for name, value in hashes.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != value:
            raise RuntimeError("source changed during observation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved bounded-control observations to", args.output)


if __name__ == "__main__":
    main()
