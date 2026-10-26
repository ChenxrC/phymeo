"""A small hanging stretch/shear patch settles against a frictionless y-plane."""

import argparse
import json

import numpy as np

from phymeo import Plane, StepInputs, XPBDSolver, cloth_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=400, help="XPBD budget per step")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    dtype = np.dtype(args.dtype)
    dt, pins = 0.02, [0, 3]
    patch = cloth_grid(
        4,
        4,
        spacing=0.15,
        density=0.5,
        structural_compliance=1e-3,
        shear_compliance=1e-3,
        pinned=pins,
        origin=(0.0, 0.3, 0.0),
        dtype=dtype,
    )
    solver = XPBDSolver(
        patch.constraints, plane=Plane(), iterations=args.iterations, tolerance=1e-6
    )
    current, inputs = patch.initial_state.copy(), StepInputs.zeros(16, dtype=dtype)
    iterations, max_residual = [], 0.0
    for frame in range(args.steps):
        result = solver.step(patch.model, current, inputs, dt)
        if not result.report.accepted:
            raise RuntimeError(f"Frame {frame}: {result.report}")
        current = result.state
        np.testing.assert_array_equal(current.position[pins], patch.initial_state.position[pins])
        if np.min(current.position[:, 1]) < 0.0:
            raise AssertionError("Plane penetration")
        iterations.append(result.report.iterations)
        max_residual = max(max_residual, result.report.max_residual)
    print(
        json.dumps(
            {
                "steps": args.steps,
                "dt": dt,
                "particles": 16,
                "triangles": len(patch.triangles),
                "edges": len(patch.constraints.indices),
                "physical_mass_kg": float(np.sum(patch.vertex_mass)),
                "minimum_height_m": float(np.min(current.position[:, 1])),
                "max_normalized_residual": max_residual,
                "max_iterations": max(iterations),
                "mean_iterations": float(np.mean(iterations)),
                "dtype": args.dtype,
                "backend": "numpy",
                "model": "stretch_shear_without_bending_or_self_collision",
            }
        )
    )


if __name__ == "__main__":
    main()
