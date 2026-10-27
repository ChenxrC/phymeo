"""Analytic geometry, stable pair identity and caller ownership."""

import numpy as np
import pytest

from phymeo import SolverFailure, SphereShapes


def evaluate(particles, bodies, radii):
    from phymeo import particle_sphere_geometry

    return particle_sphere_geometry(particles, bodies, SphereShapes(radii))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_signed_gap_and_outward_normal(dtype):
    result = evaluate(
        np.array([[0.5, 0, 0], [0, 2, 0], [-1, 0, 0], [0, 0, 0]], dtype=dtype),
        np.zeros((1, 3), dtype=dtype),
        np.array([1], dtype=dtype),
    )
    np.testing.assert_array_equal(result.gap, [-0.5, 1, 0, -1])
    np.testing.assert_array_equal(result.normal, [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(result.degenerate, [False, False, False, True])
    assert result.normal.dtype == result.gap.dtype == np.float64


def test_all_pairs_include_separation_with_particle_major_identity():
    result = evaluate(
        np.array([[3.0, 0, 0], [0, 4, 0]]),
        np.array([[0.0, 0, 0], [3, 4, 0]]),
        np.array([1.0, 2]),
    )
    np.testing.assert_array_equal(result.particle_indices, [0, 0, 1, 1])
    np.testing.assert_array_equal(result.body_indices, [0, 1, 0, 1])
    np.testing.assert_array_equal(result.gap, [2, 2, 3, 1])
    np.testing.assert_array_equal(result.normal, [[1, 0, 0], [0, -1, 0], [0, 1, 0], [-1, 0, 0]])


def test_non_axis_normal_has_analytic_direction_and_unit_length():
    result = evaluate(np.array([[4.0, 6, 3]]), np.array([[1.0, 2, 3]]), np.array([2.0]))
    np.testing.assert_array_equal(result.gap, [3])
    np.testing.assert_allclose(result.normal, [[0.6, 0.8, 0]], rtol=0, atol=1e-15)


@pytest.mark.parametrize("particle_count,body_count", [(0, 0), (0, 2), (2, 0)])
def test_empty_families_have_consistent_zero_row_arrays(particle_count, body_count):
    result = evaluate(
        np.zeros((particle_count, 3)), np.zeros((body_count, 3)), np.ones(body_count)
    )
    assert result.normal.shape == (0, 3)
    for name in ("particle_indices", "body_indices", "gap", "degenerate"):
        assert getattr(result, name).shape == (0,)


def test_result_does_not_alias_input_and_arrays_are_readonly():
    particles, bodies, radii = np.array([[2.0, 0, 0]]), np.zeros((1, 3)), np.ones(1)
    result = evaluate(particles, bodies, radii)
    np.testing.assert_array_equal(particles, [[2, 0, 0]])
    particles[:] = 9
    bodies[:] = 7
    radii[:] = 3
    np.testing.assert_array_equal(result.gap, [1])
    np.testing.assert_array_equal(result.normal, [[1, 0, 0]])
    for name in ("particle_indices", "body_indices", "normal", "gap", "degenerate"):
        array = getattr(result, name)
        assert not array.flags.writeable
        assert array.flags.c_contiguous
        for original in (particles, bodies, radii):
            assert not np.shares_memory(array, original)


def test_negative_stride_input_has_same_geometry():
    result = evaluate(np.array([[1.0, 0, 0], [2, 0, 0]])[::-1], np.zeros((1, 3)), np.ones(1))
    np.testing.assert_array_equal(result.gap, [1, 0])


@pytest.mark.parametrize(
    "particles,bodies,radii,error",
    [
        (np.ones((1, 2)), np.zeros((1, 3)), np.ones(1), ValueError),
        (np.ones((1, 3)), np.zeros(3), np.ones(1), ValueError),
        (np.ones((1, 3), dtype=int), np.zeros((1, 3)), np.ones(1), TypeError),
        (np.array([[np.nan, 0, 0]]), np.zeros((1, 3)), np.ones(1), ValueError),
        (np.ones((1, 3)), np.array([[np.inf, 0, 0]]), np.ones(1), ValueError),
        (np.ones((1, 3), dtype=np.float32), np.zeros((1, 3)), np.ones(1), ValueError),
        (np.ones((1, 3)), np.zeros((1, 3)), np.ones(1, dtype=np.float32), ValueError),
        (np.ones((1, 3)), np.zeros((2, 3)), np.ones(1), ValueError),
    ],
)
def test_invalid_geometry_inputs_are_rejected(particles, bodies, radii, error):
    with pytest.raises(error):
        evaluate(particles, bodies, radii)


def test_spheres_parameter_must_be_explicit_geometry():
    from phymeo.geometry.particle_sphere import particle_sphere_geometry

    with pytest.raises(TypeError):
        particle_sphere_geometry(np.ones((1, 3)), np.zeros((1, 3)), np.ones(1))


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_finite_norm_does_not_square_into_underflow_or_overflow(scale):
    result = evaluate(np.array([[3 * scale, 4 * scale, 0]]), np.zeros((1, 3)), np.array([scale]))
    np.testing.assert_allclose(result.gap, [4 * scale], rtol=3e-15, atol=0)
    np.testing.assert_allclose(result.normal, [[0.6, 0.8, 0]], rtol=0, atol=1e-15)


def test_smallest_axis_separation_is_not_misclassified_as_coincident():
    tiny = np.nextafter(0.0, 1.0)
    result = evaluate(np.array([[0.0, tiny, 0]]), np.zeros((1, 3)), np.array([tiny]))
    np.testing.assert_array_equal(result.normal, [[0, 1, 0]])
    np.testing.assert_array_equal(result.degenerate, [False])
    np.testing.assert_array_equal(result.gap, [0])


def test_inexact_subnormal_distance_cannot_silently_become_zero_gap():
    tiny = np.nextafter(0.0, 1.0)
    # The true distance is sqrt(2)*tiny, so the true gap is positive. Rounding
    # that distance to tiny would falsely report tangency; reject lost evidence.
    with pytest.raises(SolverFailure):
        evaluate(np.array([[tiny, tiny, 0]]), np.zeros((1, 3)), np.array([tiny]))


@pytest.mark.parametrize("kind", ["subtraction", "norm"])
def test_unrepresentable_geometry_is_a_solver_failure(kind):
    large = np.finfo(np.float64).max
    particles = np.array([[large, 0, 0]]) if kind == "subtraction" else np.array([[large, large, 0]])
    bodies = np.array([[-large, 0, 0]]) if kind == "subtraction" else np.zeros((1, 3))
    with pytest.raises(SolverFailure):
        evaluate(particles, bodies, np.ones(1))


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("particle_indices", np.array([-1]), ValueError),
        ("body_indices", np.array([0.0]), TypeError),
        ("body_indices", np.array([2**63], dtype=np.uint64), ValueError),
        ("normal", np.zeros((1, 3)), ValueError),
        ("gap", np.array([np.inf]), ValueError),
        ("gap", np.ones(2), ValueError),
        ("degenerate", np.array([0]), TypeError),
    ],
)
def test_geometry_record_rejects_invalid_rows(field, value, error):
    from phymeo.geometry.particle_sphere import ParticleSphereGeometry

    arguments = dict(
        particle_indices=np.array([0]), body_indices=np.array([0]),
        normal=np.array([[1.0, 0, 0]]), gap=np.array([0.0]), degenerate=np.array([False]),
    )
    arguments[field] = value
    with pytest.raises(error):
        ParticleSphereGeometry(**arguments)


def test_direct_geometry_record_owns_copies():
    from phymeo.geometry.particle_sphere import ParticleSphereGeometry

    arguments = dict(
        particle_indices=np.array([0]), body_indices=np.array([0]),
        normal=np.array([[1.0, 0, 0]]), gap=np.array([0.0]), degenerate=np.array([False]),
    )
    result = ParticleSphereGeometry(**arguments)
    for name, value in arguments.items():
        assert value.flags.writeable
        assert not np.shares_memory(value, getattr(result, name))
        assert not getattr(result, name).flags.writeable
