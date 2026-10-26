"""Headless rigid rotation: exact spherical reference and anisotropic drift report."""

import json

import numpy as np

from phymeo import RigidBodyInputs, RigidBodyModel, RigidBodyState, RigidSemiImplicitSolver
from phymeo.dynamics.rotation import quaternion_matrix


def kinetic_energy(model, state):
    rotation = quaternion_matrix(state.orientation)
    omega_body = np.einsum("nji,nj->ni", rotation, state.angular_velocity)
    return 0.5 * np.sum(model.inertia_diagonal * omega_body**2, axis=1)


def main():
    dt, steps = 0.001, 1000
    model = RigidBodyModel(np.ones(2), np.array([[2.0, 2.0, 2.0], [1.0, 2.0, 3.0]]), np.zeros(3))
    state = RigidBodyState(
        np.zeros((2, 3)),
        np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        np.zeros((2, 3)),
        np.array([[0.0, 2.0, 0.0], [1.0, 2.0, 3.0]]),
    )
    initial_energy = kinetic_energy(model, state)
    solver, inputs = RigidSemiImplicitSolver(), RigidBodyInputs.zeros(2)
    for _ in range(steps):
        result = solver.step(model, state, inputs, dt)
        if not result.report.accepted:
            raise RuntimeError(result.report.reason)
        state = result.state
    expected = np.array([0.0, np.sin(steps * dt), 0.0, np.cos(steps * dt)])
    error = float(np.max(np.abs(state.orientation[0] - expected)))
    if error > 1e-12:
        raise AssertionError(f"Spherical rotation error: {error}")
    print(
        json.dumps(
            {
                "steps": steps,
                "dt": dt,
                "spherical_quaternion_error": error,
                "unit_quaternion_error": float(
                    np.max(np.abs(np.linalg.norm(state.orientation, axis=1) - 1.0))
                ),
                "anisotropic_energy_relative_drift": float(
                    kinetic_energy(model, state)[1] / initial_energy[1] - 1.0
                ),
                "backend": "numpy",
                "integrator": "first_order",
            }
        )
    )


if __name__ == "__main__":
    main()
