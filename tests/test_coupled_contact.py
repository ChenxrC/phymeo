"""Analytic unilateral contact response, release and shared-body interactions."""

import importlib

import numpy as np
import pytest

from phymeo import SolverFailure, SphereShapes, particle_sphere_geometry


def api():
    return importlib.import_module("phymeo.constraints.particle_sphere")


def position_problem(dtype=np.float64, weights=(0.5, 0.25)):
    return (
        np.array([[0.5, 0, 0]], dtype=dtype),
        np.zeros((1, 3), dtype=dtype),
        np.array([weights[0]], dtype=dtype),
        np.array([weights[1]], dtype=dtype),
        SphereShapes(np.ones(1, dtype=dtype)),
        np.zeros(1),
    )


@pytest.mark.parametrize("dtype,atol", [(np.float64, 1e-12), (np.float32, 1e-7)])
def test_position_projection_uses_both_masses_and_records_opposite_loads(dtype, atol):
    p, b, wp, wb, shapes, multipliers = position_problem(dtype)
    transfer = api().project_particle_spheres(p, b, wp, wb, shapes, multipliers)
    np.testing.assert_allclose(p, [[5 / 6, 0, 0]], rtol=0, atol=atol)
    np.testing.assert_allclose(b, [[-1 / 6, 0, 0]], rtol=0, atol=atol)
    np.testing.assert_allclose(multipliers, [2 / 3], rtol=0, atol=1e-15)
    np.testing.assert_allclose(2 * p + 4 * b, [[1, 0, 0]], atol=3 * atol, rtol=0)
    np.testing.assert_allclose(transfer.particle, [[2 / 3, 0, 0]], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(transfer.rigid, -transfer.particle)
    assert transfer.kind == "position"
    assert not transfer.particle.flags.writeable and not transfer.rigid.flags.writeable
    assert transfer.particle.dtype == np.float64
    assert api().particle_sphere_residual(p, b, wp, wb, shapes, multipliers) <= atol


def test_documented_keyword_arguments_select_the_physical_buffers():
    p, b, wp, wb, shapes, multipliers = position_problem()
    arguments = dict(
        particle_position=p,
        rigid_position=b,
        particle_inverse_mass=wp,
        rigid_inverse_mass=wb,
        spheres=shapes,
        multipliers=multipliers,
    )
    api().project_particle_spheres(**arguments)
    assert api().particle_sphere_residual(**arguments) < 1e-12
    pv, bv, wp, wb, geometry, impulses = velocity_problem()
    arguments = dict(
        particle_velocity=pv,
        rigid_velocity=bv,
        particle_inverse_mass=wp,
        rigid_inverse_mass=wb,
        geometry=geometry,
        impulses=impulses,
    )
    api().project_particle_sphere_velocities(**arguments)
    assert api().particle_sphere_velocity_residual(**arguments) == 0


@pytest.mark.parametrize("weights,want_p,want_b", [((1, 0), 1, 0), ((0, 1), 0.5, -0.5)])
def test_fixed_object_receives_load_but_does_not_move(weights, want_p, want_b):
    p, b, wp, wb, shapes, multipliers = position_problem(weights=weights)
    transfer = api().project_particle_spheres(p, b, wp, wb, shapes, multipliers)
    np.testing.assert_array_equal(p, [[want_p, 0, 0]])
    np.testing.assert_array_equal(b, [[want_b, 0, 0]])
    np.testing.assert_array_equal(transfer.particle, [[0.5, 0, 0]])
    np.testing.assert_array_equal(transfer.rigid, [[-0.5, 0, 0]])


def test_inconsistent_fixed_fixed_contact_keeps_positive_residual():
    args = position_problem(weights=(0, 0))
    api().project_particle_spheres(*args)
    np.testing.assert_array_equal(args[0], [[0.5, 0, 0]])
    np.testing.assert_array_equal(args[1], [[0, 0, 0]])
    assert api().particle_sphere_residual(*args) == 0.5


def test_zero_multiplier_does_not_attract_a_separated_particle():
    args = position_problem()
    args[0][:] = [2, 0, 0]
    transfer = api().project_particle_spheres(*args)
    np.testing.assert_array_equal(args[0], [[2, 0, 0]])
    np.testing.assert_array_equal(args[1], [[0, 0, 0]])
    np.testing.assert_array_equal(transfer.particle, [[0, 0, 0]])
    assert api().particle_sphere_residual(*args) == 0


def test_contact_multiplier_releases_partially_then_fully():
    p, b, wp, wb, shapes, multipliers = position_problem(weights=(1, 1))
    api().project_particle_spheres(p, b, wp, wb, shapes, multipliers)
    p[0, 0] += 0.4  # Another constraint changes separation between sweeps.
    assert api().particle_sphere_residual(p, b, wp, wb, shapes, multipliers) > 0.3
    api().project_particle_spheres(p, b, wp, wb, shapes, multipliers)
    np.testing.assert_allclose(multipliers, [0.05], atol=1e-15)
    np.testing.assert_allclose(p - b, [[1, 0, 0]], atol=1e-15)
    p[0, 0] += 1
    api().project_particle_spheres(p, b, wp, wb, shapes, multipliers)
    np.testing.assert_array_equal(multipliers, [0])
    np.testing.assert_allclose(p - b, [[1.9, 0, 0]], atol=1e-15)
    assert api().particle_sphere_residual(p, b, wp, wb, shapes, multipliers) == 0


def test_shared_sphere_motion_creates_contact_for_an_earlier_pair():
    p, b = np.array([[1.2, 0, 0], [-0.5, 0, 0]]), np.zeros((1, 3))
    wp, wb, shapes, multipliers = np.ones(2), np.ones(1), SphereShapes(np.ones(1)), np.zeros(2)
    args = p, b, wp, wb, shapes, multipliers
    api().project_particle_spheres(*args)
    assert api().particle_sphere_residual(*args) > 0.04
    for _ in range(30):
        api().project_particle_spheres(*args)
    assert api().particle_sphere_residual(*args) < 1e-12
    assert multipliers[0] > 0
    np.testing.assert_allclose(p.sum(axis=0) + b[0], [0.7, 0, 0], atol=1e-12)
    assert np.min(np.linalg.norm(p - b, axis=1) - 1) >= -1e-12


def test_center_degeneracy_has_finite_equal_opposite_response():
    args = position_problem(weights=(1, 1))
    args[0][:] = 0
    api().project_particle_spheres(*args)
    np.testing.assert_array_equal(args[0], [[0.5, 0, 0]])
    np.testing.assert_array_equal(args[1], [[-0.5, 0, 0]])
    assert api().particle_sphere_residual(*args) == 0


def test_very_small_positive_inverse_masses_are_not_treated_as_fixed():
    args = position_problem(weights=(1e-300, 1e-300))
    api().project_particle_spheres(*args)
    np.testing.assert_allclose(args[0], [[0.75, 0, 0]], rtol=0, atol=1e-15)
    np.testing.assert_allclose(args[1], [[-0.25, 0, 0]], rtol=0, atol=1e-15)
    assert api().particle_sphere_residual(*args) < 1e-15


def test_complementarity_residual_does_not_cancel_a_small_gap_against_huge_lambda():
    p, b, wp, wb, shapes, multipliers = position_problem(weights=(1, 0))
    p[0, 0] = 1.125
    multipliers[0] = 1e100
    assert api().particle_sphere_residual(p, b, wp, wb, shapes, multipliers) == 0.125


def test_positive_residual_normalization_never_underflows_to_zero():
    tiny = np.nextafter(0.0, 1.0)
    args = (
        np.zeros((1, 3)),
        np.zeros((1, 3)),
        np.ones(1),
        np.zeros(1),
        SphereShapes(np.array([tiny])),
        np.zeros(1),
    )
    assert api().particle_sphere_residual(*args, reference_length=1e308) > 0


def velocity_problem(dtype=np.float64):
    geometry = particle_sphere_geometry(
        np.array([[1, 0, 0]], dtype=dtype),
        np.zeros((1, 3), dtype=dtype),
        SphereShapes(np.ones(1, dtype=dtype)),
    )
    return (
        np.array([[-3, 2, 4]], dtype=dtype),
        np.array([[0, -2, -1]], dtype=dtype),
        np.array([0.5], dtype=dtype),
        np.array([0.25], dtype=dtype),
        geometry,
        np.zeros(1),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_plastic_normal_collision_conserves_momentum_and_keeps_tangent(dtype):
    pv, bv, wp, wb, geometry, impulses = velocity_problem(dtype)
    before_p, before_b = pv.copy(), bv.copy()
    initial_momentum = 2 * before_p + 4 * before_b
    initial_energy = np.sum(before_p**2) + 2 * np.sum(before_b**2)
    transfer = api().project_particle_sphere_velocities(pv, bv, wp, wb, geometry, impulses)
    np.testing.assert_array_equal(pv, [[-1, 2, 4]])
    np.testing.assert_array_equal(bv, [[-1, -2, -1]])
    np.testing.assert_array_equal(impulses, [4])
    np.testing.assert_allclose(2 * pv + 4 * bv, initial_momentum, atol=1e-10, rtol=0)
    assert np.sum(pv**2) + 2 * np.sum(bv**2) <= initial_energy + 1e-10
    np.testing.assert_array_equal(transfer.particle, [[4, 0, 0]])
    np.testing.assert_array_equal(transfer.rigid, [[-4, 0, 0]])
    assert transfer.kind == "velocity"
    assert api().particle_sphere_velocity_residual(pv, bv, wp, wb, geometry, impulses) == 0


def test_normal_contact_load_has_zero_torque_about_sphere_center():
    p, b = np.array([[0.6, 0.8, 0]]), np.zeros((1, 3))
    geometry = particle_sphere_geometry(p, b, SphereShapes(np.ones(1)))
    pv, bv = np.array([[-0.6, -0.8, 0]]), np.zeros((1, 3))
    transfer = api().project_particle_sphere_velocities(
        pv, bv, np.ones(1), np.ones(1), geometry, np.zeros(1)
    )
    np.testing.assert_allclose(np.cross(p - b, transfer.rigid), 0, atol=1e-15)
    np.testing.assert_allclose(pv, [[-0.3, -0.4, 0]], atol=1e-15)


def test_velocity_impulse_can_release_and_does_not_pull_a_separating_point():
    pv, bv, wp, wb, geometry, impulses = velocity_problem()
    wp[:], wb[:] = 1, 1
    pv[:], bv[:] = [-2, 0, 0], 0
    args = pv, bv, wp, wb, geometry, impulses
    api().project_particle_sphere_velocities(*args)
    pv[0, 0] += 1
    assert api().particle_sphere_velocity_residual(*args) == 1
    api().project_particle_sphere_velocities(*args)
    np.testing.assert_array_equal(impulses, [0.5])
    pv[0, 0] += 4
    api().project_particle_sphere_velocities(*args)
    np.testing.assert_array_equal(impulses, [0])
    np.testing.assert_array_equal(pv, [[3, 0, 0]])
    np.testing.assert_array_equal(bv, [[0, 0, 0]])
    assert api().particle_sphere_velocity_residual(*args) == 0


def test_shared_body_velocity_solve_releases_earlier_contact():
    geometry = particle_sphere_geometry(
        np.array([[1.0, 0, 0], [1, 0, 0]]), np.zeros((1, 3)), SphereShapes(np.ones(1))
    )
    pv, bv = np.array([[-1.0, 0, 0], [-4, 0, 0]]), np.zeros((1, 3))
    args = pv, bv, np.ones(2), np.ones(1), geometry, np.zeros(2)
    api().project_particle_sphere_velocities(*args)
    assert api().particle_sphere_velocity_residual(*args) > 0
    for _ in range(10):
        api().project_particle_sphere_velocities(*args)
    np.testing.assert_allclose(pv, [[-1, 0, 0], [-2, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(bv, [[-2, 0, 0]], atol=1e-12)
    np.testing.assert_array_equal(args[-1], [0, 2])
    assert api().particle_sphere_velocity_residual(*args) < 1e-12


def test_velocity_contact_band_is_explicit_and_does_not_extend_by_default():
    geometry = particle_sphere_geometry(
        np.array([[1.125, 0, 0]]), np.zeros((1, 3)), SphereShapes(np.ones(1))
    )
    pv, bv = np.array([[-1.0, 0, 0]]), np.zeros((1, 3))
    args = pv, bv, np.ones(1), np.zeros(1), geometry, np.zeros(1)
    api().project_particle_sphere_velocities(*args)
    np.testing.assert_array_equal(pv, [[-1, 0, 0]])
    api().project_particle_sphere_velocities(*args, contact_tolerance=0.125)
    np.testing.assert_array_equal(pv, [[0, 0, 0]])


def test_fixed_fixed_closing_velocity_is_not_accepted():
    args = list(velocity_problem())
    args[2][:], args[3][:] = 0, 0
    api().project_particle_sphere_velocities(*args)
    assert api().particle_sphere_velocity_residual(*args) == 3


def test_inactive_contact_releases_its_previously_applied_impulse():
    geometry = particle_sphere_geometry(
        np.array([[2.0, 0, 0]]), np.zeros((1, 3)), SphereShapes(np.ones(1))
    )
    pv, bv = np.array([[0.25, 0, 0]]), np.array([[-0.25, 0, 0]])
    args = pv, bv, np.ones(1), np.ones(1), geometry, np.array([0.25])
    assert api().particle_sphere_velocity_residual(*args) == 0.5
    transfer = api().project_particle_sphere_velocities(*args)
    np.testing.assert_array_equal(pv, [[0, 0, 0]])
    np.testing.assert_array_equal(bv, [[0, 0, 0]])
    np.testing.assert_array_equal(args[-1], [0])
    np.testing.assert_array_equal(transfer.particle, [[-0.25, 0, 0]])
    assert api().particle_sphere_velocity_residual(*args) == 0


@pytest.mark.parametrize("np_count,nb_count", [(0, 0), (0, 2), (2, 0)])
def test_empty_system_sweeps_and_residuals(np_count, nb_count):
    p, b, wp, wb = (
        np.zeros((np_count, 3)),
        np.zeros((nb_count, 3)),
        np.ones(np_count),
        np.ones(nb_count),
    )
    shapes, multipliers = SphereShapes(np.ones(nb_count)), np.zeros(0)
    transfer = api().project_particle_spheres(p, b, wp, wb, shapes, multipliers)
    assert transfer.particle.shape == p.shape and transfer.rigid.shape == b.shape
    assert api().particle_sphere_residual(p, b, wp, wb, shapes, multipliers) == 0
    geometry = particle_sphere_geometry(p, b, shapes)
    api().project_particle_sphere_velocities(p, b, wp, wb, geometry, multipliers)
    assert api().particle_sphere_velocity_residual(p, b, wp, wb, geometry, multipliers) == 0


@pytest.mark.parametrize(
    "mutation",
    ["negative_mass", "negative_lambda", "readonly", "lambda_dtype", "shape", "dtype", "alias"],
)
def test_invalid_workspaces_are_rejected_before_mutation(mutation):
    args = list(position_problem())
    if mutation == "negative_mass":
        args[2][0] = -1
    elif mutation == "negative_lambda":
        args[5][0] = -1
    elif mutation == "readonly":
        args[0].flags.writeable = False
    elif mutation == "lambda_dtype":
        args[5] = args[5].astype(np.float32)
    elif mutation == "shape":
        args[2] = np.ones(2)
    elif mutation == "dtype":
        args[1] = args[1].astype(np.float32)
    elif mutation == "alias":
        args[1] = args[0]
    before = [value.copy() for value in args if isinstance(value, np.ndarray)]
    with pytest.raises((TypeError, ValueError)):
        api().project_particle_spheres(*args)
    for actual, saved in zip((v for v in args if isinstance(v, np.ndarray)), before):
        np.testing.assert_array_equal(actual, saved)


def test_unrepresentable_multiplier_is_not_silent_success():
    args = position_problem(weights=(np.nextafter(0.0, 1.0), 0))
    with pytest.raises(SolverFailure):
        api().project_particle_spheres(*args)


@pytest.mark.parametrize("value", [0, -1, np.inf, True])
def test_residual_reference_length_must_be_positive_finite_real(value):
    with pytest.raises(ValueError):
        api().particle_sphere_residual(*position_problem(), reference_length=value)


def test_inactive_pair_does_not_compute_an_unused_overflowing_relative_velocity():
    geometry = particle_sphere_geometry(
        np.array([[2.0, 0, 0]]), np.zeros((1, 3)), SphereShapes(np.ones(1))
    )
    large = np.finfo(np.float64).max
    pv, bv = np.array([[large, 0, 0]]), np.array([[-large, 0, 0]])
    args = pv, bv, np.ones(1), np.ones(1), geometry, np.zeros(1)
    api().project_particle_sphere_velocities(*args)
    assert api().particle_sphere_velocity_residual(*args) == 0
    np.testing.assert_array_equal(pv, [[large, 0, 0]])
    np.testing.assert_array_equal(bv, [[-large, 0, 0]])


def test_normal_velocity_residual_preserves_small_term_between_cancelling_terms():
    geometry = particle_sphere_geometry(
        np.ones((1, 3)), np.zeros((1, 3)), SphereShapes(np.array([np.sqrt(3)]))
    )
    pv, bv = np.array([[1e16, -1, -1e16]]), np.zeros((1, 3))
    # The first/third products cancel exactly, leaving the middle component.
    residual = api().particle_sphere_velocity_residual(
        pv, bv, np.ones(1), np.ones(1), geometry, np.zeros(1), contact_tolerance=1e-12
    )
    assert residual == pytest.approx(1 / np.sqrt(3), abs=1e-15)


def test_inexact_subnormal_normal_speed_is_rejected_instead_of_zero():
    geometry = particle_sphere_geometry(
        np.ones((1, 3)), np.zeros((1, 3)), SphereShapes(np.array([np.sqrt(3)]))
    )
    tiny = np.nextafter(0.0, 1.0)
    pv, bv = np.array([[-tiny, 0, 0]]), np.zeros((1, 3))
    with pytest.raises(SolverFailure):
        api().particle_sphere_velocity_residual(
            pv, bv, np.ones(1), np.ones(1), geometry, np.zeros(1), contact_tolerance=1e-12
        )
