"""Independent KKT and energy checks for the local anisotropic friction problem."""

import importlib

import numpy as np
import pytest

from phymeo import SolverFailure


def solve(k, v, normal=2.0, **kwargs):
    options = dict(mu_static=0.6, mu_dynamic=0.5)
    options.update(kwargs)
    return importlib.import_module("phymeo.constraints.friction").solve_friction_disk(
        k, v, normal, **options
    )


def check_energy(result, k, v):
    change = float(v @ result.impulse + 0.5 * result.impulse @ k @ result.impulse)
    assert result.energy_change == pytest.approx(change, abs=1e-12, rel=1e-10)
    assert change <= 1e-12
    np.testing.assert_allclose(result.velocity_after, v + k @ result.impulse, atol=1e-12)


def test_sticking_cancels_velocity_with_full_response():
    k = np.array([[2.0, 0.5], [0.5, 1.0]])
    expected = np.array([-0.2, 0.3])
    v = -k @ expected
    result = solve(k, v)
    assert result.regime == "sticking"
    assert result.iterations == 0
    assert result.lagrange_multiplier == 0
    np.testing.assert_allclose(result.impulse, expected, atol=1e-12)
    np.testing.assert_allclose(result.velocity_after, 0, atol=1e-12)
    check_energy(result, k, v)


@pytest.mark.parametrize(
    "k,t,eta",
    [
        (np.diag([1.0, 4]), np.array([-0.6, -0.8]), 2.0),
        (np.array([[2.0, 0.6], [0.6, 1.0]]), np.array([-0.8, -0.6]), 0.7),
    ],
)
def test_sliding_known_anisotropic_kkt_solution(k, t, eta):
    v = -(k + eta * np.eye(2)) @ t
    result = solve(k, v, mu_static=0.5, mu_dynamic=0.5)
    assert result.regime == "sliding"
    np.testing.assert_allclose(result.impulse, t, atol=2e-10, rtol=2e-10)
    assert result.lagrange_multiplier == pytest.approx(eta, abs=2e-9)
    assert 0 < result.iterations <= 96
    np.testing.assert_allclose(
        k @ result.impulse + v + result.lagrange_multiplier * result.impulse, 0, atol=1e-10
    )
    # A radial truncation of the unconstrained answer is not this KKT solution.
    radial = np.linalg.solve(k, -v)
    radial /= np.linalg.norm(radial)
    assert np.linalg.norm(radial - result.impulse) > 0.01
    check_energy(result, k, v)


def test_static_dynamic_switch_and_static_boundary():
    k = np.eye(2)
    assert solve(k, np.array([1.2, 0.0])).regime == "sticking"
    result = solve(k, np.array([1.21, 0.0]))
    assert result.regime == "sliding"
    np.testing.assert_allclose(result.impulse, [-1, 0], atol=1e-10)
    np.testing.assert_allclose(result.velocity_after, [0.21, 0], atol=1e-10)


@pytest.mark.parametrize("v", [np.zeros(2), np.array([2.0, -1.0])])
def test_zero_normal_load_is_open_even_when_velocity_is_zero(v):
    result = solve(np.eye(2), v, normal=0)
    assert result.regime == "open"
    assert result.lagrange_multiplier is None
    np.testing.assert_array_equal(result.impulse, np.zeros(2))
    np.testing.assert_array_equal(result.velocity_after, v)
    assert result.energy_change == 0


def test_zero_dynamic_radius_does_not_claim_sticking():
    v = np.array([2.0, -1.0])
    result = solve(np.eye(2), v, mu_dynamic=0)
    assert result.regime == "sliding"
    assert result.lagrange_multiplier is None
    np.testing.assert_array_equal(result.impulse, np.zeros(2))
    np.testing.assert_array_equal(result.velocity_after, v)


def test_tangent_basis_rotation_preserves_physics():
    k, v = np.diag([1.0, 4]), np.array([1.8, 4.8])
    angle = 0.37
    q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    result, rotated = solve(k, v), solve(q @ k @ q.T, q @ v)
    np.testing.assert_allclose(rotated.impulse, q @ result.impulse, atol=2e-10)
    np.testing.assert_allclose(rotated.velocity_after, q @ result.velocity_after, atol=2e-10)
    assert rotated.energy_change == pytest.approx(result.energy_change, abs=1e-10)


def test_random_spd_disk_kkt_energy_and_stick_velocity():
    rng = np.random.default_rng(167)
    for _ in range(64):
        a = rng.normal(size=(2, 2))
        k = a.T @ a + np.eye(2) * 0.1
        v = rng.normal(size=2) * 2
        result = solve(k, v)
        if result.regime == "sticking":
            assert np.linalg.norm(result.impulse) <= 1.2
            np.testing.assert_allclose(result.velocity_after, 0, atol=1e-10)
        else:
            assert np.linalg.norm(result.impulse) == pytest.approx(1.0, abs=2e-10)
            np.testing.assert_allclose(
                result.velocity_after, -result.lagrange_multiplier * result.impulse, atol=2e-9
            )
        check_energy(result, k, v)


def test_inputs_owned_and_outputs_readonly():
    k, v = np.eye(2), np.array([0.1, 0.2])
    result = solve(k, v)
    np.testing.assert_array_equal(k, np.eye(2))
    np.testing.assert_array_equal(v, [0.1, 0.2])
    k[:] = v[:] = 5
    np.testing.assert_allclose(result.impulse, [-0.1, -0.2], atol=1e-12)
    for value in (result.impulse, result.velocity_after):
        assert value.dtype == np.float64
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value[:] = 1


@pytest.mark.parametrize(
    "k,v",
    [
        (np.eye(2, dtype=np.float32), np.ones(2)),
        (np.eye(2), np.ones(2, np.float32)),
        (np.ones((2, 3)), np.ones(2)),
        (np.eye(2), np.ones((1, 2))),
        (np.array([[1.0, 0.2], [0.3, 1.0]]), np.ones(2)),
        (np.full((2, 2), np.nan), np.ones(2)),
        (np.eye(2), np.array([np.inf, 0])),
    ],
)
def test_invalid_arrays_rejected(k, v):
    with pytest.raises((TypeError, ValueError)):
        solve(k, v)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(normal=-1),
        dict(normal=np.inf),
        dict(normal=True),
        dict(mu_static=-1),
        dict(mu_dynamic=-1),
        dict(mu_dynamic=0.7),
        dict(mu_static=np.nan),
        dict(tolerance=0),
        dict(tolerance=0.1),
        dict(max_iterations=0),
        dict(max_iterations=True),
        dict(max_iterations=2.5),
    ],
)
def test_invalid_scalar_settings_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        solve(np.eye(2), np.ones(2), **kwargs)


@pytest.mark.parametrize(
    "k", [np.zeros((2, 2)), np.diag([1.0, 0]), np.diag([1.0, -1]), np.diag([1.0, 1e-13])]
)
def test_singular_indefinite_or_ill_conditioned_response_fails(k):
    with pytest.raises(SolverFailure):
        solve(k, np.ones(2))


def test_iteration_budget_cannot_return_unconverged_impulse():
    with pytest.raises(SolverFailure, match="converg"):
        solve(np.diag([1.0, 4]), np.array([1.8, 4.8]), max_iterations=1)


def test_unrepresentable_impulse_or_disk_radius_fails():
    with pytest.raises(SolverFailure):
        solve(np.eye(2) * 1e-300, np.ones(2) * 1e100)
    with pytest.raises(SolverFailure):
        solve(np.eye(2), np.ones(2), normal=1e308, mu_static=1e308, mu_dynamic=1e308)
