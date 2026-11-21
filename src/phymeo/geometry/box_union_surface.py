"""Exterior triangles of axis-aligned boxes in one explicitly shared frame.

Bounds are finite (n, 2, 3) lower/upper coordinates, not oriented-box centers
and sizes. No transforms, tolerance welding, dilation, or topological repairs
are inferred. Every selected exposed coordinate-grid quad becomes two outward
triangles; coplanar neighbors share node indices and never overlap in area.
"""

from itertools import product
from numbers import Integral

import numpy as np


def axis_aligned_box_union_mesh(bounds, *, face_directions):
    """Return float64 vertices and integer triangles of selected union faces.

    ``face_directions`` explicitly selects unique (axis, sign) pairs: axes are
    0/1/2 in the bounds' common frame and signs are -1/+1. For lateral sides in
    a frame whose vertical axis is z, select ((0,-1),(0,1),(1,-1),(1,1)).
    Boxes may overlap or repeat. Input order does not affect output ordering.
    The caller owns coordinate-frame and source-orientation validation.
    """
    try:
        boxes = np.asarray(bounds)
    except (TypeError, ValueError) as exc:
        raise ValueError("bounds must be finite nonempty (n, 2, 3) lower/upper boxes") from exc
    if (
        boxes.ndim != 3
        or boxes.shape[1:] != (2, 3)
        or not len(boxes)
        or boxes.dtype.kind not in "fiu"
    ):
        raise ValueError("bounds must be finite nonempty (n, 2, 3) lower/upper boxes")
    boxes = boxes.astype(np.float64, copy=True)
    if not np.isfinite(boxes).all() or np.any(boxes[:, 0] >= boxes[:, 1]):
        raise ValueError("bounds must have finite strictly positive extent on each axis")
    if not isinstance(face_directions, (tuple, list)) or not face_directions:
        raise ValueError("face_directions must be nonempty unique (axis, sign) pairs")
    selected = set()
    for direction in face_directions:
        if (
            not isinstance(direction, (tuple, list))
            or len(direction) != 2
            or any(
                isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in direction
            )
            or direction[0] not in (0, 1, 2)
            or direction[1] not in (-1, 1)
        ):
            raise ValueError("face_directions must contain integer axes 0..2 and signs -1/+1")
        key = (int(direction[0]), int(direction[1]))
        if key in selected:
            raise ValueError("face_directions must not repeat a direction")
        selected.add(key)

    axes = [np.unique(boxes[:, :, axis]) for axis in range(3)]
    solid = {}
    for cell in product(*(range(len(axis) - 1) for axis in axes)):
        lo = np.array([axes[a][i] for a, i in enumerate(cell)])
        hi = np.array([axes[a][i + 1] for a, i in enumerate(cell)])
        # Full interval containment is exact for this coordinate partition.
        # A rounded midpoint can instead fill a genuine one-ULP empty gap.
        if np.any(np.all((boxes[:, 0] <= lo) & (hi <= boxes[:, 1]), axis=1)):
            solid[cell] = (lo, hi)
    nodes, indices, faces = [], {}, []
    for cell, (lo, hi) in solid.items():
        for axis, side in product(range(3), (-1, 1)):
            if (axis, side) not in selected:
                continue
            neighbor = list(cell)
            neighbor[axis] += side
            if tuple(neighbor) in solid:
                continue
            tangents = [i for i in range(3) if i != axis]
            rectangle = []
            for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                point = lo.copy()
                point[axis] = hi[axis] if side > 0 else lo[axis]
                point[tangents] = (
                    hi[tangents[0]] if u else lo[tangents[0]],
                    hi[tangents[1]] if v else lo[tangents[1]],
                )
                key = tuple(point)
                if key not in indices:
                    indices[key] = len(nodes)
                    nodes.append(point)
                rectangle.append(indices[key])
            a, b, c, d = rectangle
            if np.cross(nodes[b] - nodes[a], nodes[c] - nodes[a])[axis] * side < 0:
                b, d = d, b
            faces.extend(((a, b, c), (a, c, d)))
    return np.asarray(nodes, dtype=np.float64), tuple(faces)
