"""Independent discrete trajectories for one massive free root with an offset body frame."""

import numpy as np
import pytest

from phymeo.core.rigid import RigidBodyInputs, RigidBodyModel
from phymeo.core.robot import RobotModel
from phymeo.dynamics.floating_robot import (
    FloatingRobotModel,
    FloatingRobotState,
    floating_kinematics,
)
from phymeo.solvers.free_root import step_free_root


def model(gravity=(0.0, 0, 0), children=None):
    rigid = RigidBodyModel(
        np.array([0.5]), np.array([[0.25, 0.375, 0.5]]), np.asarray(gravity, dtype=float)
    )
    return FloatingRobotModel(rigid, np.array([0.125, 0, 0]), children)


def state(velocity=(0, 0.25, 0, 0, 0, 2)):
    return FloatingRobotState(np.array([0.0, 0, 0, 0, 0, 0, 1]), np.asarray(velocity, dtype=float))


def test_force_free_com_drifts_and_body_origin_curves_over_sixteen_steps():
    m = model()
    previous = state()
    for k in range(1, 17):
        result = step_free_root(m, previous, RigidBodyInputs.zeros(1), 1 / 32)
        assert result.report.accepted
        t = k / 32
        s = result.state
        np.testing.assert_allclose(s.v, [0, 0.25, 0, 0, 0, 2], rtol=0, atol=2e-14)
        np.testing.assert_allclose(
            s.q[:3],
            [0.125 * (1 - np.cos(2 * t)), t / 4 - 0.125 * np.sin(2 * t), 0],
            rtol=0,
            atol=2e-14,
        )
        np.testing.assert_allclose(s.q[3:], [0, 0, np.sin(t), np.cos(t)], rtol=0, atol=2e-14)
        data = floating_kinematics(m, s)
        np.testing.assert_allclose(data.com_position, [[0.125, t / 4, 0]], rtol=0, atol=2e-14)
        np.testing.assert_allclose(
            data.origin_velocity,
            [[0.25 * np.sin(2 * t), 0.25 * (1 - np.cos(2 * t)), 0]],
            rtol=0,
            atol=2e-14,
        )
        previous = s


def test_external_com_force_torque_and_gravity_drive_actual_public_states():
    m = model((0, 0, -1))
    load = RigidBodyInputs(np.array([[4.0, -2, 0]]), np.array([[0.0, 0, 1]]))
    s = state(np.zeros(6))
    dt = 1 / 16
    for k in range(1, 9):
        result = step_free_root(m, s, load, dt)
        assert result.report.accepted
        s = result.state
        np.testing.assert_allclose(s.v, k * dt * np.array([2, -1, -1, 0, 0, 2]), atol=2e-14)
        sum_h2 = dt**2 * k * (k + 1) / 2
        angle = 2 * sum_h2
        center = np.array([0.125, 0, 0]) + sum_h2 * np.array([2, -1, -1])
        np.testing.assert_allclose(
            s.q[:3], center - 0.125 * np.array([np.cos(angle), np.sin(angle), 0]), atol=2e-14
        )
        np.testing.assert_allclose(
            s.q[3:], [0, 0, np.sin(angle / 2), np.cos(angle / 2)], atol=2e-14
        )
    np.testing.assert_array_equal(load.force, [[4, -2, 0]])


def test_nonprincipal_spin_uses_body_inertia_and_world_gyro_acceleration():
    # Cyclic quaternion maps body axes (x,y,z) to world (y,z,x).
    m = model()
    s = FloatingRobotState(
        np.array([0.25, -0.5, 0.125, 0.5, 0.5, 0.5, 0.5]), np.array([0.0, 0, 0, 3, 1, 2])
    )
    dt = 1 / 64
    # Body omega=(1,2,3). Torque-free Euler rates=(-3,2,-1/2).
    result = step_free_root(m, s, RigidBodyInputs.zeros(1), dt)
    assert result.report.accepted
    np.testing.assert_allclose(
        result.state.v[3:], [3 - 0.5 * dt, 1 - 3 * dt, 2 + 2 * dt], atol=2e-14
    )
    np.testing.assert_allclose(
        floating_kinematics(m, result.state).com_position, [[0.25, -0.375, 0.125]], atol=2e-14
    )


def test_rejection_keeps_exact_input_and_retry_matches_a_clean_step():
    m = model()
    s = state()
    overflow = RigidBodyInputs(np.array([[1.0e308, 0, 0]]), np.zeros((1, 3)))
    rejected = step_free_root(m, s, overflow, 16.0)
    assert not rejected.report.accepted
    assert rejected.report.reason
    np.testing.assert_array_equal(rejected.state.q, s.q)
    np.testing.assert_array_equal(rejected.state.v, s.v)
    inputs = RigidBodyInputs.zeros(1)
    retry = step_free_root(m, rejected.state, inputs, 1 / 32)
    clean = step_free_root(m, s, inputs, 1 / 32)
    np.testing.assert_array_equal(retry.state.q, clean.state.q)
    np.testing.assert_array_equal(retry.state.v, clean.state.v)
    with pytest.raises(ValueError):
        step_free_root(m, s, inputs, -1)


def test_single_body_solver_declares_coupled_tree_out_of_scope():
    children = RobotModel(
        np.array([-1]),
        ("revolute",),
        np.zeros((1, 3)),
        np.array([[0.0, 0, 0, 1]]),
        np.array([[0.0, 0, 1]]),
        np.ones(1),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )
    s = FloatingRobotState(np.r_[state().q, 0.0], np.r_[state().v, 0.0])
    with pytest.raises(ValueError, match="(single|children|root-only)"):
        step_free_root(model(children=children), s, RigidBodyInputs.zeros(1), 1 / 32)


def test_stationary_eccentric_origin_preserves_a_small_absolute_coordinate():
    m = FloatingRobotModel(model().root, np.array([float(2**54), 0, 0]))
    s = FloatingRobotState(np.array([0.125, -0.375, 0.25, 0, 0, 0, 1]), np.zeros(6))
    result = step_free_root(m, s, RigidBodyInputs.zeros(1), 1 / 32)
    assert result.report.accepted
    np.testing.assert_array_equal(result.state.q, s.q)
