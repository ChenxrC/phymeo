"""Headless compliant chain; rejects a frame if its constitutive solve fails."""

import json

import numpy as np

from phymeo import DistanceConstraints, ParticleModel, ParticleState, StepInputs, XPBDSolver


def main():
    count, dt, steps = 8, 0.01, 200
    inverse_mass = np.ones(count)
    inverse_mass[0] = 0.0
    model = ParticleModel(inverse_mass)
    position = np.zeros((count, 3))
    position[:, 1] = 1.5 - np.arange(count) * 0.15
    state = ParticleState(position, np.zeros_like(position))
    edges = np.column_stack([np.arange(count - 1), np.arange(1, count)])
    constraints = DistanceConstraints(edges, np.full(count - 1, 0.15), np.full(count - 1, 1e-6))
    solver = XPBDSolver(constraints, iterations=300, tolerance=1e-8)
    inputs = StepInputs.zeros(count)
    max_residual, max_iterations = 0.0, 0
    for _ in range(steps):
        result = solver.step(model, state, inputs, dt)
        if not result.report.accepted:
            raise RuntimeError(result.report)
        max_residual = max(max_residual, result.report.max_residual)
        max_iterations = max(max_iterations, result.report.iterations)
        state = result.state
    np.testing.assert_array_equal(state.position[0], position[0])
    print(
        json.dumps(
            {
                "steps": steps,
                "particles": count,
                "dt": dt,
                "tip_height": float(state.position[-1, 1]),
                "max_normalized_residual": max_residual,
                "max_iterations": max_iterations,
                "final_geometric_error_m": result.report.max_constraint_error,
                "backend": "numpy",
            }
        )
    )


if __name__ == "__main__":
    main()
