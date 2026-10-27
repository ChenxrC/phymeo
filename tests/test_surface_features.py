"""Analytic geometry, convex optimality and physical barycentric reconstruction."""

import importlib

import numpy as np
import pytest

from phymeo import SolverFailure


def api():
    return importlib.import_module("phymeo.geometry.surface_features")


def test_geometry_public_exports():
    import phymeo

    assert phymeo.ClosestFeaturePair is api().ClosestFeaturePair
    assert phymeo.closest_point_triangle is api().closest_point_triangle
    assert phymeo.closest_segment_segment is api().closest_segment_segment


def test_triangle_feature_weights_drive_physical_impulse_and_power():
    from phymeo import ParticleModel, ParticlePointPort, ParticleState, WeightedPointPort

    tri = triangle()
    contact = api().closest_point_triangle(np.array([0.5, 0.5, 0.0]), tri)
    state = ParticleState(tri, np.array([[1.0, -1, 0], [2.0, 0, 1], [-1.0, 2, 0]]))
    base = ParticlePointPort(ParticleModel(np.array([1.0, 2, 4]), np.zeros(3)), state, [0, 1, 2])
    port = WeightedPointPort(base, contact.second_weights[None])
    np.testing.assert_allclose(port.positions(), [contact.first], atol=1e-12)
    impulse = np.array([[0.0, 0, 2.0]])
    # Barycentric weights at (.5,.5) in the 2m right triangle are .5,.25,.25.
    candidate = port.apply_impulses(impulse)
    np.testing.assert_allclose(
        candidate.velocity - state.velocity, [[0, 0, 1], [0, 0, 1], [0, 0, 2]], atol=1e-12
    )
    assert np.sum(impulse * port.velocities()) == pytest.approx(
        np.sum(np.array([[0, 0, 1], [0, 0, 0.5], [0, 0, 0.5]]) * state.velocity), abs=1e-12
    )
    np.testing.assert_array_equal(candidate.position, state.position)


def triangle():
    return np.array([[0.0, 0, 0], [2.0, 0, 0], [0.0, 2, 0]])


def assert_convex_optimal(result, first, second, atol=1e-12):
    for weights in (result.first_weights, result.second_weights):
        assert np.all(weights >= 0)
        assert np.sum(weights) == pytest.approx(1, abs=1e-14)
    np.testing.assert_allclose(result.first_weights @ first, result.first, atol=atol)
    np.testing.assert_allclose(result.second_weights @ second, result.second, atol=atol)
    # First-order variational inequalities are sufficient for this convex objective.
    difference = result.first - result.second
    assert np.min((first - result.first) @ difference) >= -atol
    assert np.max((second - result.second) @ difference) <= atol
    assert result.distance == pytest.approx(np.hypot.reduce(difference), abs=atol)


@pytest.mark.parametrize(
    "p,expected",
    [
        ([0.5, 0.5, 3.0], [0.5, 0.5, 0]),
        ([0.5, 0.5, 0.0], [0.5, 0.5, 0]),
        ([0.5, -1, 1.0], [0.5, 0, 0]),
        ([-1, 0.5, 1.0], [0, 0.5, 0]),
        ([2.0, 2, 1.0], [1, 1, 0]),
        ([-1.0, -1, 2], [0, 0, 0]),
        ([3.0, -1, 2], [2, 0, 0]),
        ([-1.0, 3, 2], [0, 2, 0]),
    ],
)
def test_point_triangle_face_edges_vertices(p, expected):
    p = np.array(p)
    result = api().closest_point_triangle(p, triangle())
    np.testing.assert_allclose(result.second, expected, atol=1e-12)
    np.testing.assert_array_equal(result.first, p)
    assert_convex_optimal(result, p[None], triangle())


def test_triangle_rigid_transform_permutation_and_repeat():
    p = np.array([0.4, 0.7, 0.8])
    # Orthogonal matrix with no axis-aligned triangle plane.
    q, _ = np.linalg.qr(np.array([[1.0, 2, 3], [-2.0, 3, 1], [4.0, 1, -1]]))
    shift = np.array([4.0, -2, 1])
    transformed = triangle() @ q.T + shift
    point = p @ q.T + shift
    result = api().closest_point_triangle(point, transformed)
    np.testing.assert_allclose(result.second, np.array([0.4, 0.7, 0]) @ q.T + shift, atol=1e-12)
    assert result.distance == pytest.approx(0.8, abs=1e-12)
    assert_convex_optimal(result, point[None], transformed)
    swapped = api().closest_point_triangle(point, transformed[[2, 0, 1]])
    np.testing.assert_allclose(swapped.second, result.second, atol=1e-12)
    repeated = api().closest_point_triangle(point, transformed)
    np.testing.assert_array_equal(repeated.second_weights, result.second_weights)


@pytest.mark.parametrize("scale", [1e-120, 1e120])
def test_triangle_scale_does_not_square_world_coordinates(scale):
    result = api().closest_point_triangle(np.array([0.5, 0.5, 3.0]) * scale, triangle() * scale)
    np.testing.assert_allclose(result.second / scale, [0.5, 0.5, 0], atol=1e-12)
    assert result.distance / scale == pytest.approx(3.0, abs=1e-12)


def test_feature_result_owns_readonly_arrays_and_input_is_unchanged():
    p, tri = np.array([0.5, 0.5, 3.0]), triangle()
    before_p, before_tri = p.copy(), tri.copy()
    result = api().closest_point_triangle(p, tri)
    np.testing.assert_array_equal(p, before_p)
    np.testing.assert_array_equal(tri, before_tri)
    p[:] = tri[:] = 100
    np.testing.assert_array_equal(result.first, before_p)
    for array in (result.first, result.second, result.first_weights, result.second_weights):
        assert array.dtype == np.float64
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[:] = 9


@pytest.mark.parametrize(
    "p,tri",
    [
        (np.ones(3, np.float32), triangle()),
        (np.ones((1, 3)), triangle()),
        (np.ones(3), np.ones((2, 3))),
        (np.ones(3), triangle().astype(np.float32)),
        (np.array([0.0, np.nan, 0]), triangle()),
        (np.ones(3), np.full((3, 3), np.inf)),
        (np.ones(3), np.zeros((3, 3))),
        (np.ones(3), np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])),
        (np.ones(3), np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1e-13, 0]])),
    ],
)
def test_invalid_or_degenerate_triangle_rejected(p, tri):
    with pytest.raises((TypeError, ValueError)):
        api().closest_point_triangle(p, tri)


def test_unrepresentable_geometry_difference_is_solver_failure():
    p = np.full(3, -1e308)
    tri = np.array([[1e308, 0, 0], [0, 1e308, 0], [0, 0, 1e308]])
    with pytest.raises(SolverFailure):
        api().closest_point_triangle(p, tri)


@pytest.mark.parametrize(
    "second,distance",
    [
        ([[0.0, -1, 0], [0.0, 1, 0]], 0.0),
        ([[0.0, -1, 1], [0.0, 1, 1]], 1.0),
        ([[-0.5, 1, 0], [0.5, 1, 0]], 1.0),
        ([[0.5, 0, 0], [2.0, 0, 0]], 0.0),
        ([[2.0, 1, 0], [3.0, 2, 0]], np.sqrt(2)),
        ([[2.0, 0, 0], [3.0, 0, 0]], 1.0),
        ([[1.0, 0, 0], [-1.0, 0, 0]], 0.0),
    ],
)
def test_segment_interior_boundary_parallel_and_overlap(second, distance):
    first = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
    second = np.array(second)
    result = api().closest_segment_segment(first, second)
    assert result.distance == pytest.approx(distance, abs=1e-12)
    assert_convex_optimal(result, first, second)
    swapped = api().closest_segment_segment(second, first)
    assert swapped.distance == pytest.approx(distance, abs=1e-12)
    repeated = api().closest_segment_segment(first, second)
    np.testing.assert_array_equal(repeated.first_weights, result.first_weights)
    np.testing.assert_array_equal(repeated.second_weights, result.second_weights)


def test_segment_rigid_transform_and_unique_weights():
    first = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
    second = np.array([[0.0, -1, 1], [0.0, 3, 1]])
    q, _ = np.linalg.qr(np.array([[1.0, 2, 3], [-2.0, 3, 1], [4.0, 1, -1]]))
    shift = np.array([4.0, -2, 1])
    result = api().closest_segment_segment(first @ q.T + shift, second @ q.T + shift)
    np.testing.assert_allclose(result.first, shift, atol=1e-12)
    np.testing.assert_allclose(result.second, np.array([0.0, 0, 1]) @ q.T + shift, atol=1e-12)
    np.testing.assert_allclose(result.first_weights, [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(result.second_weights, [0.75, 0.25], atol=1e-12)
    assert result.distance == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize("scale", [1e-120, 1e120])
def test_segment_normalization_preserves_tiny_and_large_scale(scale):
    a = np.array([[-1.0, 0, 0], [1.0, 0, 0]]) * scale
    b = np.array([[0.0, -1, 1], [0.0, 1, 1]]) * scale
    result = api().closest_segment_segment(a, b)
    assert result.distance / scale == pytest.approx(1, abs=1e-12)
    np.testing.assert_allclose(result.first_weights, [0.5, 0.5], atol=1e-12)


def test_segments_random_convex_optimality_independent_of_candidate_enumeration():
    rng = np.random.default_rng(741)
    for _ in range(64):
        a, b = rng.normal(size=(2, 2, 3))
        result = api().closest_segment_segment(a, b)
        assert_convex_optimal(result, a, b, atol=2e-12)


@pytest.mark.parametrize(
    "bad", [np.ones((2, 3), np.float32), np.ones((1, 3)), np.full((2, 3), np.nan), np.zeros((2, 3))]
)
def test_invalid_or_zero_length_segment_is_rejected(bad):
    valid = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
    with pytest.raises((TypeError, ValueError)):
        api().closest_segment_segment(valid, bad)
    with pytest.raises((TypeError, ValueError)):
        api().closest_segment_segment(bad, valid)


def test_nonzero_near_parallel_conditioning_is_not_silently_accepted():
    a = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
    b = np.array([[-1.0, 1, 0], [1.0, 1 + 1e-13, 0]])
    with pytest.raises(SolverFailure, match="parallel"):
        api().closest_segment_segment(a, b)


def test_segment_nonrepresentable_difference_and_ownership():
    a = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
    b = np.array([[0.0, -1, 1], [0.0, 1, 1]])
    result = api().closest_segment_segment(a, b)
    saved = result.second.copy()
    a[:] = b[:] = 40
    np.testing.assert_array_equal(result.second, saved)
    assert not result.first_weights.flags.writeable
    assert not result.second_weights.flags.writeable
    a = np.array([[-1e308, 0, 0], [1e308, 0, 0]])
    with pytest.raises(SolverFailure):
        api().closest_segment_segment(a, b)


def test_large_normal_separation_does_not_hide_unique_triangle_minimum():
    tri = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
    p = np.array([0.5, 2.0, 1e10])
    result = api().closest_point_triangle(p, tri)
    np.testing.assert_allclose(result.second, [0, 1, 0], atol=1e-12)
    assert np.max((tri - result.second) @ (p - result.second)) <= 1e-12


def test_large_normal_separation_does_not_hide_unique_segment_minimum():
    a = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    b = np.array([[0.5, 2, 1e8], [0.5, 3, 1e8]])
    result = api().closest_segment_segment(a, b)
    np.testing.assert_allclose(result.first, [0.5, 0, 0], atol=1e-12)
    np.testing.assert_allclose(result.second, b[0], atol=1e-12)
    assert_convex_optimal(result, a, b, atol=1e-12)
