"""Analytic mechanics and independently differentiated energy for robot loads."""

import importlib
from dataclasses import replace

import numpy as np
import pytest
from test_robot_kinematics import branched_model, robot_model

from phymeo.core.capabilities import SolverFailure
from phymeo.core.robot import RobotInputs, RobotState
from phymeo.dynamics.robot_kinematics import forward_kinematics


def dynamics_api():
    return importlib.import_module("phymeo.dynamics.robot_dynamics")


def planar_mass(q):
    c = np.cos(q[1])
    return np.array([[5 / 12 + c / 4, 1 / 12 + c / 8], [1 / 12 + c / 8, 1 / 12]])


def planar_bias(q, v):
    h = np.sin(q[1]) / 8
    return np.array(
        [
            -h * (2 * v[0] * v[1] + v[1] ** 2)
            + 9.81 * (0.75 * np.sin(q[0]) + 0.25 * np.sin(q.sum())),
            h * v[0] ** 2 + 9.81 * 0.25 * np.sin(q.sum()),
        ]
    )


@pytest.mark.parametrize("q", [-0.7, 0.0, 0.4])
def test_single_rod_matches_analytic_inertia_gravity_and_drive(q):
    api, model = dynamics_api(), robot_model(1)
    state = RobotState(np.array([q]), np.array([0.3]))
    loads = RobotInputs.zeros(1)
    loads.effort[:] = 0.2
    np.testing.assert_allclose(api.mass_matrix(model, state.q), [[1 / 12]], atol=1e-12)
    np.testing.assert_allclose(api.bias_forces(model, state), [2.4525 * np.sin(q)], atol=1e-12)
    expected = (0.2 - 2.4525 * np.sin(q)) * 12
    np.testing.assert_allclose(api.forward_dynamics(model, state, loads), [expected], atol=1e-12)


@pytest.mark.parametrize(
    "q,v", [([0, 0], [0, 0]), ([0.3, -0.5], [0.7, 0.2]), ([-0.6, 0.4], [-0.2, 0.5])]
)
def test_two_rods_match_independent_planar_mass_coriolis_and_gravity(q, v):
    api, model = dynamics_api(), robot_model()
    state = RobotState(np.array(q, dtype=float), np.array(v, dtype=float))
    acceleration = np.array([0.2, -0.3])
    mass, bias = planar_mass(state.q), planar_bias(state.q, state.v)
    np.testing.assert_allclose(api.mass_matrix(model, state.q), mass, atol=1e-12)
    np.testing.assert_allclose(api.bias_forces(model, state), bias, atol=1e-12)
    np.testing.assert_allclose(
        api.inverse_dynamics(model, state, acceleration), mass @ acceleration + bias, atol=1e-12
    )
    loads = RobotInputs(np.array([0.1, -0.2]), np.zeros((2, 3)), np.zeros((2, 3)))
    np.testing.assert_allclose(
        api.forward_dynamics(model, state, loads),
        np.linalg.solve(mass, loads.effort - bias),
        atol=1e-12,
    )


def polar_model():
    return robot_model(
        joint_types=("revolute", "prismatic"),
        joint_position=np.zeros((2, 3)),
        joint_axis=np.array([[0.0, 0, 1], [1.0, 0, 0]]),
        com=np.zeros((2, 3)),
        mass=np.array([2.0, 3.0]),
        inertia_diagonal=np.ones((2, 3)),
        gravity=np.zeros(3),
    )


def test_moving_prismatic_axis_has_coriolis_and_centrifugal_loads():
    api, model = dynamics_api(), polar_model()
    theta, radius, omega, radial_speed = 0.4, 0.7, 0.6, -0.2
    state = RobotState(np.array([theta, radius]), np.array([omega, radial_speed]))
    # Polar-coordinate Lagrangian: T = (2+3*r*r)*omega**2/2 + 3*rdot**2/2.
    mass = np.diag([2 + 3 * radius**2, 3])
    bias = np.array([6 * radius * omega * radial_speed, -3 * radius * omega**2])
    np.testing.assert_allclose(api.mass_matrix(model, state.q), mass, atol=1e-12)
    np.testing.assert_allclose(api.bias_forces(model, state), bias, atol=1e-12)
    acceleration = np.array([0.2, 0.3])
    np.testing.assert_allclose(
        api.inverse_dynamics(model, state, acceleration), mass @ acceleration + bias, atol=1e-12
    )


def test_world_com_force_and_torque_project_with_power_duality():
    api, model = dynamics_api(), polar_model()
    state = RobotState(np.array([0.0, 0.7]), np.array([0.6, -0.2]))
    loads = RobotInputs(
        np.array([0.1, 0.2]),
        np.array([[9.0, 8, 0], [2.0, 3, 0]]),
        np.array([[0.0, 0, 0.4], [0.0, 0, -0.1]]),
    )
    expected = [0.1 + 0.7 * 3 + 0.4 - 0.1, 0.2 + 2]
    actual = api.generalized_forces(model, state, loads)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    world_power = 0.1 * 0.6 + 0.2 * -0.2 + 2 * -0.2 + 3 * (0.7 * 0.6) + (0.4 - 0.1) * 0.6
    assert actual @ state.v == pytest.approx(world_power, abs=1e-12)


def geometry_energy(model, q, v):
    # Velocities from differences of world transforms, no tested mass/Jacobian.
    eps = 1e-6
    g = forward_kinematics(model, q)
    minus, plus = (forward_kinematics(model, q + sign * eps * v) for sign in (-1, 1))
    linear = (plus.com_position - minus.com_position) / (2 * eps)
    rdot = (plus.rotation - minus.rotation) / (2 * eps)
    kinetic = 0.0
    for i in range(model.dof_count):
        skew = g.rotation[i].T @ rdot[i]
        omega = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        kinetic += (
            model.mass[i] * (linear[i] @ linear[i]) + (model.inertia_diagonal[i] * omega) @ omega
        ) / 2
    return kinetic


def test_branched_rotated_tree_mass_and_gravity_match_independent_energy():
    api, model = dynamics_api(), branched_model()
    q, v = np.array([0.3, -0.2, 0.4, 0.1]), np.array([-0.1, 0.2, 0.3, -0.4])
    actual = v @ api.mass_matrix(model, q) @ v / 2
    assert actual == pytest.approx(geometry_energy(model, q, v), rel=2e-8, abs=1e-10)
    eps, gradient = 1e-6, np.zeros(4)
    for i in range(4):
        delta = np.eye(4)[i] * eps
        potentials = [
            -np.sum(
                model.mass[:, None]
                * forward_kinematics(model, q + sign * delta).com_position
                * model.gravity
            )
            for sign in (-1, 1)
        ]
        gradient[i] = (potentials[1] - potentials[0]) / (2 * eps)
    np.testing.assert_allclose(
        api.bias_forces(model, RobotState(q, np.zeros(4))), gradient, atol=2e-8
    )


def test_seeded_tree_equations_are_spd_and_finite_without_mutation():
    api, model = dynamics_api(), branched_model()
    rng = np.random.default_rng(20261027)
    for _ in range(32):
        q, v = rng.uniform(-0.75, 0.75, 4), rng.uniform(-0.5, 0.5, 4)
        state = RobotState(q, v)
        loads = RobotInputs(
            rng.uniform(-0.5, 0.5, 4),
            rng.uniform(-0.5, 0.5, (4, 3)),
            rng.uniform(-0.5, 0.5, (4, 3)),
        )
        mass = api.mass_matrix(model, q)
        np.testing.assert_allclose(mass, mass.T, atol=1e-12, rtol=0)
        np.linalg.cholesky(mass)
        acc = rng.uniform(-0.5, 0.5, 4)
        bias = api.bias_forces(model, state)
        np.testing.assert_allclose(
            api.inverse_dynamics(model, state, acc) - bias, mass @ acc, atol=1e-10, rtol=1e-10
        )
        solved = api.forward_dynamics(model, state, loads)
        np.testing.assert_allclose(
            mass @ solved,
            api.generalized_forces(model, state, loads) - bias,
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_array_equal(state.q, q)
        np.testing.assert_array_equal(state.v, v)


@pytest.mark.parametrize("bad", [np.zeros(2, np.float32), np.zeros(3), np.array([np.nan, 0.0])])
def test_bad_acceleration_is_rejected(bad):
    api, model = dynamics_api(), robot_model()
    with pytest.raises((TypeError, ValueError)):
        api.inverse_dynamics(model, RobotState(np.zeros(2), np.zeros(2)), bad)


@pytest.mark.parametrize("field", ["q", "v", "effort", "force", "torque"])
def test_mutated_bad_state_or_load_is_rejected(field):
    api, model = dynamics_api(), robot_model()
    state, loads = RobotState(np.zeros(2), np.zeros(2)), RobotInputs.zeros(2)
    getattr(state if field in ("q", "v") else loads, field).flat[0] = np.inf
    with pytest.raises((TypeError, ValueError)):
        api.forward_dynamics(model, state, loads)


def test_unrepresentable_mass_and_inertial_load_are_failures():
    api, model = dynamics_api(), robot_model(1)
    huge = replace(model, mass=np.array([1e308]), com=np.array([[0.0, -10, 0]]))
    with pytest.raises(SolverFailure):
        api.mass_matrix(huge, np.zeros(1))
    with pytest.raises(SolverFailure):
        api.bias_forces(model, RobotState(np.zeros(1), np.array([1e308])))


def test_loss_of_positive_definiteness_is_not_regularized():
    api = dynamics_api()
    model = robot_model(
        com=np.zeros((2, 3)),
        joint_position=np.zeros((2, 3)),
        inertia_diagonal=np.array([[1e-300] * 3, [1e300] * 3]),
    )
    with pytest.raises(SolverFailure):
        api.forward_dynamics(model, RobotState(np.zeros(2), np.zeros(2)), RobotInputs.zeros(2))


@pytest.mark.parametrize("bad_solution", [np.array([np.inf]), np.array([0.0])])
def test_linear_solver_output_must_be_finite_and_satisfy_equation(monkeypatch, bad_solution):
    api, model = dynamics_api(), robot_model(1)
    monkeypatch.setattr(np.linalg, "solve", lambda *args, **kwargs: bad_solution.copy())
    loads = RobotInputs.zeros(1)
    loads.effort[:] = 1
    with pytest.raises(SolverFailure):
        api.forward_dynamics(model, RobotState(np.zeros(1), np.zeros(1)), loads)
