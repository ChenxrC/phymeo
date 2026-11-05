"""Independent convex return-map oracles; no geometry tracker or scene adoption."""

import importlib

import numpy as np
import pytest

from phymeo.core.capabilities import SolverFailure
from phymeo.solvers._material_stick_law import return_traction


def api():
    return importlib.import_module("phymeo.solvers.coupled_tangent")


def solve(old, free, W, h, normal, stiffness, mu, **kwargs):
    return api().solve_coupled_tangent(
        np.asarray(old, dtype=float), np.asarray(free, dtype=float),
        np.asarray(W, dtype=float), h, np.asarray(normal, dtype=float),
        stiffness_n_per_m=np.asarray(stiffness, dtype=float),
        friction=np.asarray(mu, dtype=float), **kwargs,
    )


def energy_identity(result, old, stiffness):
    old_energy = .5 * np.asarray(stiffness) * np.sum(np.asarray(old)**2, axis=1)
    balance = (result.work_j + result.stored_energy_j - old_energy
               + result.plastic_dissipation_j + result.implicit_dissipation_j
               + result.released_energy_j)
    np.testing.assert_allclose(balance, 0, atol=2e-12, rtol=0)
    np.testing.assert_allclose(result.energy_residual_j, balance, atol=2e-12, rtol=0)
    assert np.all(result.plastic_dissipation_j >= -2e-12)
    assert np.all(result.implicit_dissipation_j >= 0)


def test_redundant_sites_keep_full_cross_mobility():
    # Two stations on one 1 kg prismatic body: both rows see both forces.
    # A_x=.01*[[2,1],[1,2]], rhs=(.03,.03), so each f=-1 N.
    W = np.zeros((4, 4))
    W[np.ix_([0, 2], [0, 2])] = 1
    result = solve(np.zeros((2, 2)), [[.03, 0], [.03, 0]], W, .1,
                   [4, 4], [100, 100], [.5, .5])
    np.testing.assert_allclose(result.force_n, [[-1, 0], [-1, 0]], atol=1e-11, rtol=0)
    np.testing.assert_allclose(result.predicted_displacement_m, [[.01, 0], [.01, 0]], atol=1e-12)
    np.testing.assert_allclose(result.stored_energy_j, [.005, .005], atol=1e-12)
    assert result.regime == ("stick", "stick") and result.residual <= 1e-10
    energy_identity(result, np.zeros((2, 2)), [100, 100])


@pytest.mark.parametrize("free", [[.001, 0], [.03, .04], [-.03, .01]])
def test_single_isotropic_station_retains_existing_scalar_return(free):
    old = np.array([.001, -.002])
    expected, sliding = return_traction(old, np.array(free), .5, .01, 2, None, 1000, .5, .5)
    result = solve([old], [free], .5 * np.eye(2), .01, [2], [1000], [.5])
    np.testing.assert_allclose(result.force_n[0], expected, atol=1e-10, rtol=1e-10)
    assert result.regime == (("slide" if sliding else "stick"),)
    energy_identity(result, [old], [1000])


def test_anisotropic_point_response_has_hand_solved_force_and_work():
    # W comes from m=1, Iz=1 and r=(1,-1,0) for tangent x/y.
    result = solve([[0, 0]], [[1/64, -1/32]], [[2, 1], [1, 2]],
                   1/8, [4], [64], [.5])
    np.testing.assert_allclose(result.force_n, [[-5/8, 7/8]], atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.elastic_displacement_m, [[5/512, -7/512]], atol=1e-12)
    np.testing.assert_allclose(result.work_j, [-37/2048], atol=1e-12)
    np.testing.assert_allclose(result.stored_energy_j, [37/4096], atol=1e-12)
    energy_identity(result, [[0, 0]], [64])


def yielded_data():
    G = np.array([[1, 0, 0], [0, 1, 0], [.5, .25, 1], [.2, -.5, .3]])
    W = G @ G.T
    stiffness, h = np.array([40., 100.]), .2
    exact = np.array([[-.6, -.8], [.4, 0]])
    A = np.diag(np.repeat(1/stiffness, 2)) + h*h*W
    # KKT gradient=-eta*f at the first disk, zero at the interior second disk.
    gradient = np.r_[-.3*exact[0], [0, 0]]
    free = (gradient - A @ exact.ravel()).reshape(2, 2)
    return W, stiffness, h, exact, free


def test_yielded_station_keeps_other_station_cross_blocks_and_global_kkt():
    W, stiffness, h, exact, free = yielded_data()
    result = solve(np.zeros((2, 2)), free, W, h, [2, 1], stiffness, [.5, .5])
    np.testing.assert_allclose(result.force_n, exact, atol=2e-9, rtol=0)
    assert result.regime == ("slide", "stick") and result.residual <= 1e-10
    np.testing.assert_allclose(result.plastic_increment_m[0], -.3*exact[0], atol=2e-10)
    energy_identity(result, np.zeros((2, 2)), stiffness)


def test_station_permutation_and_rotated_tangent_coordinates_are_covariant():
    W, stiffness, h, exact, free = yielded_data()
    R = np.array([[0., -1], [1, 0]])
    p, flat = [1, 0], [2, 3, 0, 1]
    B = np.kron(np.eye(2), R)
    result = solve(np.zeros((2, 2)), free[p] @ R.T,
                   B @ W[np.ix_(flat, flat)] @ B.T, h,
                   [1, 2], stiffness[p], [.5, .5])
    np.testing.assert_allclose(result.force_n, exact[p] @ R.T, atol=2e-9, rtol=0)
    energy_identity(result, np.zeros((2, 2)), stiffness[p])


def test_load_decrease_returns_stored_energy_without_forbidding_positive_work():
    result = solve([[.02, 0]], [[0, 0]], np.eye(2), .1, [1], [100], [.5])
    np.testing.assert_allclose(result.force_n, [[-.5, 0]], atol=1e-12)
    np.testing.assert_allclose(result.predicted_displacement_m, [[-.005, 0]], atol=1e-12)
    np.testing.assert_allclose(result.plastic_increment_m, [[.01, 0]], atol=1e-12)
    np.testing.assert_allclose(result.work_j, [.0025], atol=1e-12)
    np.testing.assert_allclose(result.plastic_dissipation_j, [.005], atol=1e-12)
    np.testing.assert_allclose(result.implicit_dissipation_j, [.01125], atol=1e-12)
    energy_identity(result, [[.02, 0]], [100])


@pytest.mark.parametrize("normal,mu,regime", [(0, .5, "open"), (4, 0, "frictionless")])
def test_zero_capacity_releases_old_spring_once_and_distinguishes_open(normal, mu, regime):
    old = np.array([[.02, 0.]])
    result = solve(old, [[.01, 0]], np.eye(2), .1, [normal], [100], [mu])
    assert result.regime == (regime,)
    np.testing.assert_array_equal(result.force_n, [[0, 0]])
    np.testing.assert_array_equal(result.elastic_displacement_m, [[0, 0]])
    np.testing.assert_allclose(result.released_energy_j, [.02], atol=1e-14)
    np.testing.assert_array_equal(result.implicit_dissipation_j, [0])
    energy_identity(result, old, [100])
    again = solve(result.elastic_displacement_m, [[.01, 0]], np.eye(2), .1,
                  [normal], [100], [mu])
    np.testing.assert_array_equal(again.released_energy_j, [0])


def test_failed_convergence_does_not_mutate_inputs_or_report_success():
    W, stiffness, h, _, free = yielded_data()
    old = np.zeros((2, 2))
    copies = [x.copy() for x in (old, free, W, stiffness)]
    with pytest.raises(SolverFailure, match="converg|residual|iteration"):
        solve(old, free, W, h, [2, 1], stiffness, [.5, .5], tolerance=1e-14, max_iterations=1)
    for actual, before in zip((old, free, W, stiffness), copies, strict=True):
        np.testing.assert_array_equal(actual, before)


@pytest.mark.parametrize("change", [
    {"W": [[1, .01], [0, 1]]}, {"W": [[1, 0], [0, -.01]]},
    {"W": [[1, 0], [0, float("nan")]]}, {"stiffness": [0]},
    {"mu": [-1]}, {"normal": [-1]}, {"h": 0},
    {"tolerance": 0}, {"max_iterations": 4097},
])
def test_invalid_physical_or_resource_inputs_fail_closed(change):
    options = dict(old=[[0, 0]], free=[[.01, 0]], W=np.eye(2), h=.1,
                   normal=[2], stiffness=[100], mu=[.5])
    options.update(change)
    with pytest.raises((ValueError, TypeError)):
        solve(**options)


def test_results_own_immutable_arrays_and_motion_tolerance_is_explicit():
    old = np.zeros((1, 2))
    result = solve(old, [[.01, 0]], np.eye(2), .1, [2], [100], [.5])
    old[:] = 99
    for name in ("force_n", "predicted_displacement_m", "elastic_displacement_m",
                 "plastic_increment_m", "work_j", "stored_energy_j"):
        with pytest.raises(ValueError):
            getattr(result, name).setflags(write=True)
    good = result.verify_motion(result.predicted_displacement_m, tolerance_m=1e-12)
    assert good.accepted and good.maximum_error_m == 0
    with pytest.raises(TypeError):
        result.verify_motion(result.predicted_displacement_m)
    for bound in (0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            result.verify_motion(result.predicted_displacement_m, tolerance_m=bound)
    bad = result.verify_motion(result.predicted_displacement_m + [[.001, 0]], tolerance_m=1e-12)
    assert not bad.accepted and bad.maximum_error_m == pytest.approx(.001)
    np.testing.assert_allclose(bad.discrepancy_m, [[.001, 0]], atol=1e-15)


def test_station_capacity_is_bounded_without_silently_dropping_rows():
    with pytest.raises(ValueError, match="128|capacity|station"):
        solve(np.zeros((129, 2)), np.zeros((129, 2)), np.zeros((258, 258)),
              .1, np.ones(129), np.ones(129), np.ones(129))
