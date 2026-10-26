"""Run with an isolated, non-editable wheel environment: python -I this_file.py."""

import sys
from pathlib import Path

import numpy as np

import phymeo


def main():
    installed = Path(phymeo.__file__).resolve()
    if not installed.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Expected a wheel inside {sys.prefix}, imported {installed}")
    model = phymeo.ParticleModel(np.array([0.0, 1.0]))
    state = phymeo.ParticleState(np.zeros((2, 3)), np.zeros((2, 3)))
    inputs = phymeo.StepInputs.zeros(model.particle_count)
    if state.particle_count != 2 or inputs.force.shape != (2, 3):
        raise RuntimeError("Installed data contract failed")
    free = phymeo.Stepper(phymeo.SemiImplicitSolver()).step(model, state, inputs, 0.1)
    if not free.report.accepted:
        raise RuntimeError("Installed free solver failed")
    np.testing.assert_allclose(free.state.position[1, 1], -0.0981)
    constraints = phymeo.DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.zeros(1))
    state.position[1, 0] = 2.0
    constrained = phymeo.Stepper(phymeo.XPBDSolver(constraints)).step(
        phymeo.ParticleModel(np.array([0.0, 1.0]), np.zeros(3)), state, inputs, 0.1
    )
    if not constrained.report.accepted:
        raise RuntimeError("Installed constrained solver failed")
    np.testing.assert_allclose(constrained.state.position[1], [1.0, 0.0, 0.0])
    rigid = phymeo.RigidSemiImplicitSolver().step(
        phymeo.RigidBodyModel(np.ones(1), np.ones((1, 3)), np.zeros(3)),
        phymeo.RigidBodyState(
            np.zeros((1, 3)),
            np.array([[0.0, 0.0, 0.0, 1.0]]),
            np.zeros((1, 3)),
            np.array([[0.0, 0.0, 2.0]]),
        ),
        phymeo.RigidBodyInputs.zeros(1),
        0.5,
    )
    if not rigid.report.accepted:
        raise RuntimeError("Installed rigid solver failed")
    np.testing.assert_allclose(rigid.state.orientation, [[0.0, 0.0, np.sin(0.5), np.cos(0.5)]])
    if {"tyx", "torch", "warp"} & set(sys.modules):
        raise RuntimeError("The base wheel imported an optional runtime")
    print(f"Installed-wheel smoke passed: {installed}")


if __name__ == "__main__":
    main()
