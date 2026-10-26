"""Audit an off-centre particle/rigid impulse exchange and point response."""

import argparse
import json

import numpy as np

from phymeo import ParticleModel, ParticleRigidPorts, ParticleState, RigidBodyModel, RigidBodyState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    dtype = np.dtype(parser.parse_args().dtype)
    pm = ParticleModel(np.array([0.5], dtype), np.zeros(3))
    ps = ParticleState(np.array([[1, 0.5, -0.25]], dtype), np.array([[0.2, -0.3, 0.4]], dtype))
    bm = RigidBodyModel(np.array([0.25], dtype), np.array([[1, 2, 3]], dtype), np.zeros(3))
    angle = np.pi / 3
    bs = RigidBodyState(
        np.array([[0.1, -0.2, 0.3]], dtype),
        np.array([[0, 0, np.sin(angle / 2), np.cos(angle / 2)]], dtype),
        np.array([[-0.1, 0.2, 0.3]], dtype),
        np.array([[0.2, -0.3, 0.4]], dtype),
    )
    port = ParticleRigidPorts(np.array([0]), np.array([0]))
    impulse = np.array([[0.4, -0.7, 0.2]], dtype)
    effective = port.effective_inverse_mass(pm, ps, bm, bs)
    result = port.apply_impulses(pm, ps, bm, bs, impulse)
    if not result.accepted:
        raise RuntimeError(result.reason)
    # Independent z-axis rotation for angular momentum, not the engine helper.
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    inertia_world = rotation @ np.diag([1, 2, 3]) @ rotation.T

    def momentum(particles, body):
        p_linear, b_linear = 2 * particles.velocity.astype(float), 4 * body.velocity.astype(float)
        linear = p_linear + b_linear
        angular = (
            np.cross(particles.position.astype(float), p_linear)
            + np.cross(body.position.astype(float), b_linear)
            + body.angular_velocity.astype(float) @ inertia_world.T
        )
        return linear, angular

    linear_before, angular_before = momentum(ps, bs)
    linear_after, angular_after = momentum(result.particle_state, result.rigid_state)
    arm = ps.position.astype(float) - bs.position.astype(float)
    before = (
        ps.velocity.astype(float)
        - bs.velocity.astype(float)
        - np.cross(bs.angular_velocity.astype(float), arm)
    )
    after = (
        result.particle_state.velocity.astype(float)
        - result.rigid_state.velocity.astype(float)
        - np.cross(result.rigid_state.angular_velocity.astype(float), arm)
    )
    measured = after - before
    predicted = np.einsum("nij,nj->ni", effective, impulse.astype(float))
    tolerance = 8e-7 if dtype == np.float32 else 5e-14
    np.testing.assert_allclose(linear_after, linear_before, atol=tolerance, rtol=0)
    np.testing.assert_allclose(angular_after, angular_before, atol=tolerance, rtol=0)
    np.testing.assert_allclose(measured, predicted, atol=tolerance, rtol=0)
    np.testing.assert_array_equal(result.particle_state.position, ps.position)
    np.testing.assert_array_equal(result.rigid_state.orientation, bs.orientation)
    print(
        json.dumps(
            {
                "dtype": dtype.name,
                "impulse_kg_m_per_s": impulse.tolist(),
                "linear_momentum_error": float(np.linalg.norm(linear_after - linear_before)),
                "angular_momentum_error": float(np.linalg.norm(angular_after - angular_before)),
                "relative_point_velocity_response_error": float(
                    np.linalg.norm(measured - predicted)
                ),
                "rigid_angular_impulse_kg_m2_per_s": result.ledger.rigid_angular.tolist(),
                "scope": "instantaneous same-point exchange; no contact detection, constraint impulse solve or time integration",
            }
        )
    )


if __name__ == "__main__":
    main()
