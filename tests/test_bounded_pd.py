"""Independent clipped-control equations, active-set oracle and actual dynamics."""

import importlib
from itertools import product

import numpy as np
import pytest

from phymeo import SolverFailure


def solve(**overrides):
    args = dict(
        mass=np.array([[2.0]]),
        bias=np.array([0.3]),
        q=np.array([0.2]),
        velocity=np.array([-0.1]),
        target=np.array([0.8]),
        target_velocity=np.array([0.1]),
        kp=np.array([4.0]),
        kd=np.array([0.5]),
        effort_limit=np.array([10.0]),
        h=0.02,
    )
    args.update(overrides)
    return importlib.import_module("phymeo.control.bounded_pd").solve_bounded_pd(**args)


def test_scalar_implicit_equation_has_closed_form():
    result = solve()
    a = 4 * (0.8 - 0.2) + 0.5 * 0.1
    d = 0.02 * 4 + 0.5
    u = (2 * (-0.1) + 0.02 * (a - 0.3)) / (2 + 0.02 * d)
    assert result.effort[0] == pytest.approx(a - d * u, abs=1e-12)
    assert result.predicted_velocity[0] == pytest.approx(u, abs=1e-12)
    assert result.predicted_configuration[0] == pytest.approx(0.2 + 0.02 * u, abs=1e-12)
    assert result.residual_effort <= 1e-10 and result.equation_residual <= 1e-12


def test_bounded_control_public_exports():
    import phymeo

    module = importlib.import_module("phymeo.control.bounded_pd")
    assert phymeo.solve_bounded_pd is module.solve_bounded_pd
    assert phymeo.BoundedPDResult is module.BoundedPDResult


def test_installed_bounded_control_smoke():
    from test_articulated_measurement import script

    result = script("smoke_installed").check_bounded_control()
    np.testing.assert_allclose(result.effort, [0.25], atol=1e-12)


@pytest.mark.parametrize("target,sign", [(100.0, 1), (-100.0, -1)])
def test_effort_saturates_inside_coupled_solution(target, sign):
    result = solve(target=np.array([target]), effort_limit=np.array([0.2]))
    assert result.effort[0] == sign * 0.2
    assert result.predicted_velocity[0] == pytest.approx(-0.1 + 0.02 * (sign * 0.2 - 0.3) / 2)


def test_zero_gain_and_zero_limit_are_well_defined():
    result = solve(
        kp=np.zeros(1), kd=np.zeros(1), feedforward=np.array([2.0]), effort_limit=np.ones(1)
    )
    np.testing.assert_array_equal(result.effort, [1.0])
    result = solve(effort_limit=np.zeros(1))
    np.testing.assert_array_equal(result.effort, [0.0])
    assert result.predicted_velocity[0] == pytest.approx(-0.103)


def active_set_oracle(mass, bias, q, velocity, target, target_velocity, kp, kd, limit, h, ff):
    # Enumerate physical actuator modes, then solve the original affine law.
    n = len(q)
    mobility = np.linalg.solve(mass, np.eye(n))
    free = velocity - h * mobility @ bias
    a = kp * (target - q) + kd * target_velocity + ff
    d = h * kp + kd
    for mode in product((-1, 0, 1), repeat=n):
        mode = np.array(mode)
        fixed = mode != 0
        active = ~fixed
        effort = mode * limit
        if np.any(active):
            equation = np.eye(sum(active)) + h * d[active, None] * mobility[np.ix_(active, active)]
            rhs = a[active] - d[active] * (
                free[active] + h * mobility[np.ix_(active, fixed)] @ effort[fixed]
            )
            effort[active] = np.linalg.solve(equation, rhs)
        u = free + h * mobility @ effort
        if np.max(np.abs(effort - np.clip(a - d * u, -limit, limit))) < 1e-10:
            return effort, u
    raise AssertionError("oracle did not find a valid actuator mode")


def test_coupled_mass_matches_independent_active_set_oracle():
    rng = np.random.default_rng(812)
    for index in range(32):
        factors = rng.normal(size=(2, 2))
        mass = factors @ factors.T + np.eye(2) * 0.3
        bias, q, v, target, vt, ff = rng.normal(size=(6, 2))
        kp = rng.uniform(0, 30, 2)
        kd = rng.uniform(0, 3, 2)
        if index % 3 == 0:
            kp[0] = kd[0] = 0
        limit = rng.uniform(0.1, 2, 2)
        h = 0.02
        expected, u = active_set_oracle(mass, bias, q, v, target, vt, kp, kd, limit, h, ff)
        result = solve(
            mass=mass,
            bias=bias,
            q=q,
            velocity=v,
            target=target,
            target_velocity=vt,
            kp=kp,
            kd=kd,
            effort_limit=limit,
            h=h,
            feedforward=ff,
        )
        np.testing.assert_allclose(result.effort, expected, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(result.predicted_velocity, u, atol=1e-11, rtol=1e-11)
        assert np.all(np.abs(result.effort) <= limit)


def test_light_inertia_scalar_control_dissipates_zero_target_energy():
    q, v = np.array([0.01]), np.array([0.0])
    previous = 0.5 * 2 * q[0] ** 2
    for _ in range(20):
        result = solve(
            mass=np.array([[1e-6]]),
            bias=np.zeros(1),
            q=q,
            velocity=v,
            target=np.zeros(1),
            target_velocity=np.zeros(1),
            kp=np.array([2.0]),
            kd=np.array([0.06]),
            effort_limit=np.array([1.0]),
            h=0.001,
        )
        q, v = result.predicted_configuration, result.predicted_velocity
        energy = 0.5 * 1e-6 * v[0] ** 2 + 0.5 * 2 * q[0] ** 2
        assert energy <= previous + 1e-16
        previous = energy


def test_result_owns_data_and_cannot_change_inputs():
    target = np.array([0.8])
    result = solve(target=target)
    target[:] = 50
    for array in (result.effort, result.predicted_velocity, result.predicted_configuration):
        assert not array.flags.writeable
    assert result.predicted_configuration[0] < 0.3


@pytest.mark.parametrize(
    "overrides",
    [
        dict(mass=np.eye(1, dtype=np.float32)),
        dict(mass=np.ones((1, 2))),
        dict(q=np.ones(2)),
        dict(bias=np.array([np.nan])),
        dict(kp=np.array([-1.0])),
        dict(kd=np.array([-1.0])),
        dict(effort_limit=np.array([-1.0])),
        dict(h=0),
        dict(h=np.inf),
        dict(max_iterations=True),
        dict(max_iterations=0),
        dict(effort_tolerance=0),
        dict(mass_tolerance=0),
        dict(feedforward=np.ones(2)),
        dict(target_velocity=np.array([np.inf])),
    ],
)
def test_invalid_control_inputs_rejected(overrides):
    with pytest.raises((TypeError, ValueError)):
        solve(**overrides)


@pytest.mark.parametrize("mass", [np.array([[0.0]]), np.array([[-1.0]])])
def test_nonpositive_mass_rejected(mass):
    with pytest.raises(SolverFailure):
        solve(mass=mass)


def test_nonrepresentable_control_is_explicit_failure():
    with pytest.raises(SolverFailure):
        solve(kp=np.array([1e308]), target=np.array([1e308]))


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_rounded_prediction_must_satisfy_original_state_based_control_law(sign):
    q = sign * 1e8
    with pytest.raises(SolverFailure, match="prediction|control law"):
        solve(
            mass=np.ones((1, 1)),
            bias=np.zeros(1),
            q=np.array([q]),
            velocity=np.zeros(1),
            target=np.array([q + sign * 1e-8]),
            target_velocity=np.zeros(1),
            kp=np.array([1e8]),
            kd=np.zeros(1),
            effort_limit=np.array([10.0]),
            h=0.001,
        )


def test_budget_is_not_silent_approximate_success():
    with pytest.raises(SolverFailure, match="budget|converg"):
        solve(
            mass=np.array([[1.0, 0.9], [0.9, 1.0]]),
            bias=np.zeros(2),
            q=np.zeros(2),
            velocity=np.zeros(2),
            target=np.array([1.0, -1.0]),
            target_velocity=np.zeros(2),
            kp=np.full(2, 1000.0),
            kd=np.full(2, 100.0),
            effort_limit=np.full(2, 1e6),
            h=0.1,
            max_iterations=1,
        )


@pytest.mark.parametrize(
    "matrix",
    [np.array([[1.0, 0.1], [0.2, 1.0]]), np.diag([1.0, 1e-14]), np.array([[1.0, 2.0], [2.0, 1.0]])],
)
def test_asymmetric_indefinite_or_ill_conditioned_mass_refused(matrix):
    with pytest.raises(SolverFailure):
        solve(
            mass=matrix,
            bias=np.zeros(2),
            q=np.zeros(2),
            velocity=np.zeros(2),
            target=np.ones(2),
            target_velocity=np.zeros(2),
            kp=np.ones(2),
            kd=np.ones(2),
            effort_limit=np.ones(2),
        )


@pytest.mark.parametrize("limit", [0.1, 2.0])
def test_numpy_participant_uses_effort_and_owns_actual_prediction(limit):
    from test_robot_kinematics import robot_model

    from phymeo import BackendContext, RobotInputs, RobotParticipant, RobotState, StepSettings
    from phymeo.dynamics.robot_dynamics import bias_forces, mass_matrix

    model = robot_model()
    state = RobotState(np.array([0.2, -0.3]), np.array([-0.1, 0.2]))
    target = np.array([0.25, -0.2])
    kp = np.full(2, 6.0)
    kd = np.full(2, 0.4)
    h = 0.001
    mass, bias = mass_matrix(model, state.q), bias_forces(model, state)
    command = solve(
        mass=mass,
        bias=bias,
        q=state.q,
        velocity=state.v,
        target=target,
        target_velocity=np.zeros(2),
        kp=kp,
        kd=kd,
        effort_limit=np.full(2, limit),
        h=h,
    )
    ws = RobotParticipant(model).create_workspace(
        "arm",
        state,
        RobotInputs(command.effort, np.zeros((2, 3)), np.zeros((2, 3))),
        h,
        StepSettings(),
        BackendContext(),
    )
    try:
        ws.predict()
        expected = np.clip(kp * (target - ws.configuration()) - kd * ws.velocity(), -limit, limit)
        np.testing.assert_allclose(command.effort, expected, atol=1e-10, rtol=0)
        np.testing.assert_allclose(
            mass @ ((ws.velocity() - state.v) / h) + bias, command.effort, atol=1e-12
        )
        np.testing.assert_array_equal(state.q, [0.2, -0.3])
    finally:
        ws.close()
