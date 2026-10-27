"""Physical load, power, ownership and rejection through the existing PointPort."""

import importlib
from dataclasses import dataclass

import numpy as np
import pytest
from test_robot_dynamics import planar_mass
from test_robot_kinematics import robot_model

from phymeo.core.capabilities import SolverFailure
from phymeo.core.model import ParticleModel
from phymeo.core.robot import RobotState
from phymeo.core.state import ParticleState
from phymeo.coupling.ports import ParticlePointPort, exchange_impulses


def port_api():
    return importlib.import_module("phymeo.coupling.robot_port").RobotPointPort


def rod_port(points=None, indices=None):
    points = np.array([[0.0, -0.5, 0]]) if points is None else points
    indices = np.zeros(len(points), dtype=int) if indices is None else indices
    return port_api()(robot_model(1), RobotState(np.zeros(1), np.zeros(1)), indices, points)


def test_rod_endpoint_impulse_has_analytic_effective_mass_and_keeps_q():
    port = rod_port()
    impulse = np.array([[2.0, 0, 0]])
    np.testing.assert_allclose(port.positions(), [[0, -0.5, 0]], atol=1e-12)
    np.testing.assert_allclose(port.response(impulse), [[6, 0, 0]], atol=1e-12)
    candidate = port.apply_impulses(impulse)
    np.testing.assert_allclose(candidate.v, [12], atol=1e-12)
    np.testing.assert_array_equal(candidate.q, [0])
    np.testing.assert_array_equal(port.snapshot().v, [0])
    np.testing.assert_array_equal(impulse, [[2, 0, 0]])


def test_shared_ancestor_response_matches_analytic_two_rod_matrix():
    port = port_api()(
        robot_model(),
        RobotState(np.zeros(2), np.zeros(2)),
        [0, 1],
        np.array([[0.0, -0.5, 0], [0.0, -0.5, 0]]),
    )
    # Endpoints at y=-.5/-1, their x Jacobian rows are [.5,0] / [1,.5].
    jacobian = np.array([[0.5, 0], [1.0, 0.5]])
    force = np.array([2.0, 0])
    delta = np.linalg.solve(planar_mass(np.zeros(2)), jacobian.T @ force)
    expected = np.zeros((2, 3))
    expected[:, 0] = jacobian @ delta
    assert abs(expected[1, 0]) > 0.1
    np.testing.assert_allclose(
        port.response(np.array([[2.0, 0, 0], [0.0, 0, 0]])), expected, atol=1e-12
    )
    np.testing.assert_allclose(
        port.apply_impulses(np.array([[2.0, 0, 0], [0.0, 0, 0]])).v, delta, atol=1e-12
    )


def test_same_rod_opposite_attachments_have_opposite_cross_response():
    port = rod_port(np.array([[0.0, -0.5, 0], [0.0, 0.5, 0]]))
    np.testing.assert_allclose(
        port.response(np.array([[2.0, 0, 0], [0.0, 0, 0]])), [[6, 0, 0], [-6, 0, 0]], atol=1e-12
    )


def test_duplicate_points_sum_generalized_load_with_cancellation_preserved():
    port = rod_port(np.tile([0.0, -0.5, 0.0], (3, 1)))
    impulse = np.array([[1e16, 0, 0], [1.0, 0, 0], [-1e16, 0, 0]])
    np.testing.assert_allclose(port.apply_impulses(impulse).v, [6], atol=1e-12)
    np.testing.assert_allclose(port.response(impulse), [[3, 0, 0]] * 3, atol=1e-12)


def test_port_velocity_and_generalized_impulse_satisfy_power_and_energy():
    state = RobotState(np.array([0.3]), np.array([0.4]))
    port = port_api()(robot_model(1), state, [0], np.array([[0.0, -0.5, 0]]))
    impulse = np.array([[0.2, -0.3, 0.1]])
    jacobian = np.array([0.5 * np.cos(0.3), 0.5 * np.sin(0.3), 0])
    np.testing.assert_allclose(port.velocities(), [jacobian * 0.4], atol=1e-12)
    generalized = jacobian @ impulse[0]
    after = port.apply_impulses(impulse)
    assert (1 / 12) * (after.v[0] - state.v[0]) == pytest.approx(generalized, abs=1e-12)
    power = np.sum(impulse * port.velocities())
    assert power == pytest.approx(generalized * state.v[0], abs=1e-12)
    delta_energy = (after.v[0] ** 2 - state.v[0] ** 2) / 24
    assert delta_energy == pytest.approx(
        power + np.sum(impulse * port.response(impulse)) / 2, abs=1e-12
    )


def test_prismatic_particle_exchange_is_same_point_equal_opposite_and_once():
    model = robot_model(
        1, joint_types=("prismatic",), joint_axis=np.array([[0.0, 1, 0]]), mass=np.array([0.5])
    )
    state = RobotState(np.array([0.2]), np.array([-0.1]))
    robot = port_api()(model, state, [0], np.array([[0.3, 0, 0]]))
    particle = ParticlePointPort(
        ParticleModel(np.array([2.0]), np.zeros(3)),
        ParticleState(np.array([[0.3, 0.2, 0.0]]), np.zeros((1, 3))),
        [0],
    )
    result = exchange_impulses(particle, robot, np.array([[0.0, 0.4, 0]]))
    np.testing.assert_allclose(result.states["particles"].velocity, [[0, 0.8, 0]], atol=1e-12)
    np.testing.assert_allclose(result.states["robot"].v, [-0.9], atol=1e-12)
    np.testing.assert_array_equal(result.impulses["particles"], -result.impulses["robot"])
    np.testing.assert_array_equal(result.points["particles"], result.points["robot"])
    np.testing.assert_array_equal(state.v, [-0.1])


def test_port_owns_model_state_and_attachments_and_all_returned_arrays():
    model = robot_model(1)
    state = RobotState(np.zeros(1), np.zeros(1))
    indices, local = np.array([0]), np.array([[0.0, -0.5, 0]])
    port = port_api()(model, state, indices, local, domain="arm")
    indices[:] = 42
    local[:] = 42
    state.q[:] = 42
    model.mass.flags.writeable = True
    model.mass[:] = 42
    port.snapshot().v[:] = 42
    port.positions()[:] = 42
    port.response(np.array([[2.0, 0, 0]]))[:] = 42
    assert port.domain == "arm"
    assert port.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(port.positions(), [[0, -0.5, 0]], atol=1e-12)
    np.testing.assert_allclose(port.response(np.array([[2.0, 0, 0]])), [[6, 0, 0]], atol=1e-12)


def test_empty_port_returns_owned_unchanged_state():
    port = rod_port(np.empty((0, 3)), np.array([], dtype=int))
    assert port.positions().shape == (0, 3)
    assert port.response(np.empty((0, 3))).shape == (0, 3)
    np.testing.assert_array_equal(port.apply_impulses(np.empty((0, 3))).v, [0])


@pytest.mark.parametrize(
    "indices,points",
    [
        ([-1], np.zeros((1, 3))),
        ([1], np.zeros((1, 3))),
        ([True], np.zeros((1, 3))),
        ([0], np.zeros((2, 3))),
        ([0], np.zeros((1, 3), np.float32)),
    ],
)
def test_port_rejects_bad_attachment_configuration(indices, points):
    with pytest.raises((TypeError, ValueError)):
        rod_port(points, indices)


@pytest.mark.parametrize(
    "impulses", [np.zeros((1, 3), np.float32), np.zeros((2, 3)), np.array([[np.nan, 0, 0]])]
)
def test_port_rejects_bad_impulses_without_changing_state(impulses):
    port = rod_port()
    with pytest.raises((TypeError, ValueError)):
        port.apply_impulses(impulses)
    np.testing.assert_array_equal(port.snapshot().v, [0])


def test_port_candidate_cannot_change_q_shape_or_hide_history():
    port = rod_port()
    for candidate in (RobotState(np.ones(1), np.zeros(1)), RobotState(np.zeros(2), np.zeros(2))):
        with pytest.raises(ValueError):
            port.validate_candidate(candidate)

    @dataclass
    class HiddenState(RobotState):
        warm: np.ndarray

    with pytest.raises(TypeError, match="unregistered"):
        port.validate_candidate(HiddenState(np.zeros(1), np.zeros(1), np.zeros(1)))


def test_unrepresentable_point_response_is_failure_not_a_nonfinite_state():
    port = rod_port()
    with pytest.raises(SolverFailure):
        port.apply_impulses(np.array([[1e308, 0, 0]]))
    np.testing.assert_array_equal(port.snapshot().v, [0])
