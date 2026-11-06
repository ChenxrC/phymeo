"""Conservative original-face selection for a validated translating closed mesh.

Only candidate indices are supplied here. Whole-solid topology, containment,
furthest vertices and the precise material/feature query remain in MeshRoundQuery.
The callback may use HOST or a genuine device BVH; this helper never substitutes
its own traversal and never falls back to all faces after an invalid result.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction as F
from math import isfinite
from numbers import Integral, Real

import numpy as np

from ..constraints.collision_shapes import MAX_MESH_TRIANGLES, CollisionKey, CollisionShape
from ..core.capabilities import SolverFailure
from ._closed_mesh import ClosedMesh
from ._prism_pair_exact import sqrt_bounds

_F32_MAX = F(float(np.finfo(np.float32).max))
_F32_TINY = F(float(np.finfo(np.float32).tiny))


@dataclass(frozen=True)
class MeshRoundFaceBinding:
    """Stable face-table identity within a workspace bound to one immutable model."""

    key: CollisionKey
    revision: int
    vertex_ids: tuple[int, ...]
    faces: tuple[tuple[int, int, int], ...]


def _rational(value):
    if isinstance(value, F):
        return value
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return F(int(value))
    if (
        isinstance(value, Real)
        and not isinstance(value, (bool, np.bool_))
        and isfinite(float(value))
    ):
        return F(float(value))
    raise ValueError("mesh-round candidate coordinates must be finite real values")


def _point(value):
    try:
        point = tuple(_rational(x) for x in value)
    except TypeError as error:
        raise ValueError("mesh-round candidate point must have three coordinates") from error
    if len(point) != 3:
        raise ValueError("mesh-round candidate point must have three coordinates")
    return point


def _radius(value):
    radius = _rational(value)
    if radius <= 0:
        raise ValueError("mesh-round candidate radius must be positive")
    return radius


def _outward(value, upper):
    """Same finite/flush-to-zero envelope contract as the public broad phase."""
    if abs(value) > _F32_MAX:
        raise ValueError("mesh-round candidate bound exceeds finite FP32 storage")
    if 0 < abs(value) < _F32_TINY:
        if upper:
            return np.finfo(np.float32).tiny if value > 0 else np.float32(0)
        return -np.finfo(np.float32).tiny if value < 0 else np.float32(0)
    rounded = np.float32(float(value))
    exact = F(float(rounded))
    if (upper and exact < value) or (not upper and exact > value):
        rounded = np.nextafter(rounded, np.float32(np.inf if upper else -np.inf))
    return rounded


def _boxes(rows):
    data = np.array(
        [[_outward(value, i >= 3) for i, value in enumerate(row)] for row in rows], dtype=np.float32
    )
    # The callback receives owned immutable bytes, not aliases into prepared
    # geometry or a mutable buffer left over from an earlier query.
    return np.frombuffer(data.tobytes(), dtype=np.float32).reshape(-1, 6)


def _axis_square(point, start, end):
    axis = tuple(b - a for a, b in zip(start, end, strict=True))
    length2 = sum(v * v for v in axis)
    t = (
        sum((p - a) * v for p, a, v in zip(point, start, axis, strict=True)) / length2
        if length2
        else F(0)
    )
    t = max(F(0), min(F(1), t))
    return sum((p - a - t * v) ** 2 for p, a, v in zip(point, start, axis, strict=True))


class MeshRoundFaceSelector:
    """One current prepared mesh and one callback; no cross-query geometry cache.

    query_boxes(binding, leaves, queries) receives outward FP32 arrays of shape
    (F,6) and (1,6), with original directed face order. It must return one list
    of original face indices. At most 2F entries (including duplicate indices)
    are accepted, canonicalized to a sorted tuple without silently truncating.
    """

    def __init__(self, shape, prepared_mesh, query_boxes):
        if not isinstance(shape, CollisionShape) or shape.kind != "mesh" or shape.radius:
            raise ValueError("mesh-round face selector requires an unrounded mesh registration")
        if not isinstance(prepared_mesh, ClosedMesh) or not callable(query_boxes):
            raise TypeError("mesh-round face selector requires a prepared closed mesh and callback")
        faces = tuple(f.ids for f in prepared_mesh.facets)
        if (
            not 4 <= len(faces) <= MAX_MESH_TRIANGLES
            or faces != shape.triangles
            or tuple(prepared_mesh.points) != shape.vertex_ids
        ):
            raise ValueError("mesh-round candidate material binding does not match prepared mesh")
        points = {i: _point(p) for i, p in prepared_mesh.points.items()}
        rows = []
        for facet in prepared_mesh.facets:
            xyz = tuple(points[i] for i in facet.ids)
            low = tuple(min(p[k] for p in xyz) for k in range(3))
            high = tuple(max(p[k] for p in xyz) for k in range(3))
            if (
                facet.shape.key != shape.key
                or facet.shape.vertex_ids != facet.ids
                or tuple(_point(p) for p in facet.xyz) != xyz
                or _point(facet.lo) != low
                or _point(facet.hi) != high
            ):
                raise ValueError(
                    "mesh-round candidate face coordinates do not match material binding"
                )
            rows.append((*low, *high))
        self.binding = MeshRoundFaceBinding(shape.key, shape.revision, shape.vertex_ids, faces)
        self._points = points
        self._leaves = _boxes(rows)
        self._query_boxes = query_boxes

    def _query(self, endpoints, padding, *, require_nonempty):
        low = tuple(min(p[k] for p in endpoints) - padding for k in range(3))
        high = tuple(max(p[k] for p in endpoints) + padding for k in range(3))
        query = _boxes(((*low, *high),))
        rows = self._query_boxes(self.binding, self._leaves, query)
        count = len(self.binding.faces)
        if (
            not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], list)
            or len(rows[0]) > 2 * count
        ):
            raise SolverFailure("mesh_round_invalid_face_candidates")
        if any(
            isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or not 0 <= i < count
            for i in rows[0]
        ):
            raise SolverFailure("mesh_round_invalid_face_candidates")
        result = tuple(sorted({int(i) for i in rows[0]}))
        if require_nonempty and not result:
            raise SolverFailure("mesh_round_empty_face_candidates")
        return result

    def sample(self, start, end, radius, vertex_distance2):
        """Select every global nearest tie and every face at distance <= radius.

        U² is the minimum of the complete current vertex-distance map. Its
        minimizing vertex is rechecked against the actual finite axis, proving
        U is a real upper bound even at this internal call boundary. Since every
        mesh vertex belongs to a finite face, nearest surface distance <= U.
        sqrt_bounds supplies a rational upper endpoint before FP32 enclosure.
        """
        start, end, radius = _point(start), _point(end), _radius(radius)
        if not isinstance(vertex_distance2, Mapping) or set(vertex_distance2) != set(self._points):
            raise ValueError("mesh-round sample needs the complete vertex-distance map")
        values = {i: _rational(value) for i, value in vertex_distance2.items()}
        if any(value < 0 for value in values.values()):
            raise ValueError("mesh-round vertex squared distances must be nonnegative")
        node = min(values, key=lambda i: (values[i], i))
        upper2 = values[node]
        if upper2 != _axis_square(self._points[node], start, end):
            raise ValueError("mesh-round sample vertex distance does not match current finite axis")
        padding = max(radius, sqrt_bounds(upper2, 80)[1])
        return self._query((start, end), padding, require_nonempty=True)

    def translation(self, start, end, drift, radius):
        """Select radius-near faces for the entire affine finite-axis path.

        This can return an empty tuple. It proves only absence of a boundary
        event in that swept box; the caller must retain its full initial solid
        relation and must not infer absence of containment from this result.
        """
        start, end, drift, radius = _point(start), _point(end), _point(drift), _radius(radius)
        endpoints = (
            start,
            end,
            tuple(a + d for a, d in zip(start, drift, strict=True)),
            tuple(b + d for b, d in zip(end, drift, strict=True)),
        )
        return self._query(endpoints, radius, require_nonempty=False)
