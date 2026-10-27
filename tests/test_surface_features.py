"""Analytic geometry, convex optimality and physical barycentric reconstruction."""

import importlib

import numpy as np
import pytest

from phymeo import SolverFailure


def api():
    return importlib.import_module("phymeo.geometry.surface_features")


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
