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
