"""A low-budget XPBD attempt falls back once to the same physical problem."""

import json
from dataclasses import asdict

import numpy as np

from phymeo import (
    DistanceConstraints,
    ParticleModel,
    ParticleState,
    StepInputs,
    Stepper,
    XPBDSolver,
)


def main():
    constraints = DistanceConstraints(np.array([[0, 1], [1, 2]]), np.ones(2), np.zeros(2))
    model = ParticleModel(np.array([0.0, 1.0, 1.0]), np.zeros(3))
    state = ParticleState(
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]), np.zeros((3, 3))
    )
    stepper = Stepper(
        XPBDSolver(constraints, iterations=1, tolerance=1e-9),
        XPBDSolver(constraints, iterations=100, tolerance=1e-9),
        required_features={"distance"},
    )
    result = stepper.step(model, state, StepInputs.zeros(3), 0.1)
    if not result.report.accepted or not result.report.fallback_used:
        raise RuntimeError("Expected one successful mathematical fallback")
    print(
        json.dumps(
            {"positions_x": result.state.position[:, 0].tolist(), "report": asdict(result.report)}
        )
    )


if __name__ == "__main__":
    main()
