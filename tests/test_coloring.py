"""Edge scheduling must cover constraints without simultaneous vertex writers."""

import importlib

import numpy as np
import pytest


def color_edges(indices):
    module_name = "phymeo.constraints.coloring"
    assert importlib.util.find_spec(module_name) is not None, "edge coloring is not implemented"
    return importlib.import_module(module_name).color_edges(indices)


def assert_valid(edges, result):
    np.testing.assert_array_equal(np.sort(result.order), np.arange(len(edges)))
    assert result.order.dtype == result.offsets.dtype == np.int64
    assert result.offsets[0] == 0 and result.offsets[-1] == len(edges)
    assert not result.order.flags.writeable and not result.offsets.flags.writeable
    for begin, end in zip(result.offsets[:-1], result.offsets[1:]):
        vertices = edges[result.order[begin:end]].ravel()
        assert len(vertices) == len(np.unique(vertices))
        assert end > begin


def test_greedy_schedule_is_stable_and_separates_duplicate_edges():
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3], [0, 1]])
    result = color_edges(edges)
    assert_valid(edges, result)
    np.testing.assert_array_equal(result.order, [0, 2, 1, 3, 4, 5, 6])
    np.testing.assert_array_equal(result.offsets, [0, 2, 4, 6, 7])
    replay = color_edges(edges)
    np.testing.assert_array_equal(replay.order, result.order)
    edges[:] = 0
    np.testing.assert_array_equal(result.order, [0, 2, 1, 3, 4, 5, 6])


def test_random_graph_has_no_shared_writers_per_color():
    rng = np.random.default_rng(394)
    edges = np.array([rng.choice(50, 2, replace=False) for _ in range(300)])
    assert_valid(edges, color_edges(edges))


def test_sparse_vertex_ids_do_not_require_vertex_sized_allocation():
    edges = np.array([[0, 10**12], [10**12, 3], [3, 4]], dtype=np.int64)
    assert_valid(edges, color_edges(edges))


def test_empty_graph_has_no_colors():
    result = color_edges(np.empty((0, 2), dtype=np.int64))
    assert_valid(np.empty((0, 2), dtype=np.int64), result)
    np.testing.assert_array_equal(result.offsets, [0])


@pytest.mark.parametrize(
    "edges",
    [
        np.array([0, 1]),
        np.array([[0, 1, 2]]),
        np.array([[0.0, 1.0]]),
        np.array([[False, True]]),
        np.array([[-1, 0]]),
        np.array([[1, 1]]),
        np.array([[0, 2**63]], dtype=np.uint64),
    ],
)
def test_invalid_edges_are_rejected(edges):
    with pytest.raises((ValueError, TypeError)):
        color_edges(edges)
