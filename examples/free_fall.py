"""Run a headless free fall and check the exact discrete Euler trajectory."""

import json

import numpy as np

from phymeo import ParticleModel, ParticleState, SemiImplicitSolver, StepInputs


def main():
    dt, steps, height = 0.01, 100, 10.0
    model = ParticleModel(np.ones(1))
    state = ParticleState(np.array([[0.0, height, 0.0]]), np.zeros((1, 3)))
    inputs = StepInputs.zeros(1)
    solver = SemiImplicitSolver()
    for _ in range(steps):
        result = solver.step(model, state, inputs, dt)
        if not result.report.accepted:
            raise RuntimeError(result.report.reason)
        state = result.state
    expected_y = height + model.gravity[1] * dt**2 * steps * (steps + 1) / 2
    error = abs(float(state.position[0, 1]) - expected_y)
    if error > 1e-12:
        raise AssertionError(f"Discrete trajectory error: {error}")
    print(
        json.dumps(
            {
                "steps": steps,
                "dt": dt,
                "height": float(state.position[0, 1]),
                "discrete_position_error": error,
                "backend": "numpy",
            }
        )
    )


if __name__ == "__main__":
    main()
