"""Deterministic edge scheduling with no shared endpoint within each color."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class EdgeColoring:
    """Owned read-only edge permutation and exclusive color boundaries."""

    order: np.ndarray
    offsets: np.ndarray


def color_edges(indices) -> EdgeColoring:
    """Stable greedy coloring; sparse vertex IDs need no dense vertex allocation.

    Duplicate constraints remain separate edges, in different colors. The result
    is deterministic for an input order, not a minimum-color optimization.
    """
    edges = np.asarray(indices)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("indices must have shape (m, 2)")
    if edges.dtype.kind not in "iu":
        raise TypeError("indices must be integers")
    if np.any(edges < 0) or np.any(edges > np.iinfo(np.int64).max):
        raise ValueError("indices must be non-negative int64 values")
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("an edge must connect two different particles")
    used, groups = {}, []
    for edge, (first, second) in enumerate(edges):
        first, second = int(first), int(second)
        first_colors = used.setdefault(first, set())
        second_colors = used.setdefault(second, set())
        color = 0
        while color in first_colors or color in second_colors:
            color += 1
        if color == len(groups):
            groups.append([])
        groups[color].append(edge)
        first_colors.add(color)
        second_colors.add(color)
    order = np.array([edge for group in groups for edge in group], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum([len(group) for group in groups], dtype=np.int64)))
    order.flags.writeable = offsets.flags.writeable = False
    return EdgeColoring(order, offsets)
