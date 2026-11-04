"""Exact material incidence and affine footprint intervals on one oriented translating sheet."""

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import hypot
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionModel
from ..core.capabilities import SolverFailure


def _xyz(values):
    a = np.asarray(values)
    if a.shape != (3,) or a.dtype.kind not in "ifu" or not np.isfinite(a).all():
        raise ValueError("expected three finite spatial coordinates")
    return tuple(Fraction(float(x)) for x in a)


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _det(a, b):
    return a[0] * b[1] - a[1] * b[0]


@dataclass(frozen=True)
class _Face:
    key: CollisionKey
    ids: tuple
    origin: tuple
    u: tuple
    v: tuple
    det: Fraction

    def weights(self, p):
        d = _sub(p, self.origin)
        b, c = _det(d, self.v) / self.det, _det(self.u, d) / self.det
        return 1 - b - c, b, c


@dataclass(frozen=True)
class MaterialFaceInterval:
    face: CollisionKey
    t0: float
    t1: float
    entry_weights: tuple
    exit_weights: tuple
    entry_local_m: tuple
    exit_local_m: tuple


def _bounds(w0, w1):
    lo, hi = Fraction(0), Fraction(1)
    for a, b in zip(w0, w1, strict=True):
        if a < 0 and b < 0:
            return None
        if a < 0:
            lo = max(lo, -a / (b - a))
        elif b < 0:
            hi = min(hi, a / (a - b))
    return (lo, hi) if lo <= hi else None


@dataclass(frozen=True, init=False, eq=False)
class TranslatingMaterialPatch:
    """An oriented material chart fixed in its translating frame, with one common translation DOF.

    Faces have one world and material body. Only shared material node IDs create adjacency.
    The supplied coordinates are offsets from the translating frame's origin (its mass center).
    """

    vertices: Mapping
    normal: tuple
    material_body: tuple

    def __init__(self, vertices, model):
        if not isinstance(model, CollisionModel) or not model.shapes or model.excluded:
            raise ValueError("expected a nonempty explicit triangle patch without excluded pairs")
        if not isinstance(vertices, Mapping) or any(
            isinstance(i, bool) or not isinstance(i, Integral) for i in vertices
        ):
            raise ValueError("patch vertices must be keyed by material node IDs")
        points = {int(i): _xyz(value) for i, value in vertices.items()}
        faces = sorted(model.shapes, key=lambda shape: shape.key)
        body = (faces[0].key.world, faces[0].key.body)
        if any(
            s.kind != "triangle" or s.radius or (s.key.world, s.key.body) != body for s in faces
        ):
            raise ValueError(
                "a translating patch requires unrounded triangles of one material body"
            )
        if set(points) != {i for s in faces for i in s.vertex_ids}:
            raise ValueError("registered vertices must exactly cover the material patch")
        a, b, c = (points[i] for i in faces[0].vertex_ids)
        nf = _cross(_sub(b, a), _sub(c, a))
        scale = max(map(abs, nf))
        if scale == 0 or any(_dot(nf, _sub(p, a)) != 0 for p in points.values()):
            raise ValueError("material patch must be nondegenerate and exactly coplanar")
        axis = max(range(3), key=lambda i: abs(nf[i]))
        keep = tuple(i for i in range(3) if i != axis)

        def project(p):
            return tuple(p[i] for i in keep)

        local = {k: project(p) for k, p in points.items()}
        rows = []
        edges = {}
        tri_ids = set()
        for s in faces:
            i, j, k = s.vertex_ids
            canonical = tuple(sorted(s.vertex_ids))
            if canonical in tri_ids:
                raise ValueError("duplicate material triangle")
            tri_ids.add(canonical)
            a0, b0, c0 = points[i], points[j], points[k]
            if _dot(_cross(_sub(b0, a0), _sub(c0, a0)), nf) <= 0:
                raise ValueError("flipped or degenerate material face")
            for x, y in ((i, j), (j, k), (k, i)):
                edge = tuple(sorted((x, y)))
                owners = edges.setdefault(edge, [])
                if len(owners) >= 2 or (x, y) in owners:
                    raise ValueError("nonmanifold or inconsistently oriented shared material edge")
                owners.append((x, y))
            u, v = _sub(local[j], local[i]), _sub(local[k], local[i])
            rows.append(_Face(s.key, s.vertex_ids, local[i], u, v, _det(u, v)))
        scaled = tuple(float(x / scale) for x in nf)
        length = hypot(*scaled)
        object.__setattr__(self, "normal", tuple(v / length for v in scaled))
        object.__setattr__(self, "material_body", body)
        object.__setattr__(
            self, "vertices", MappingProxyType({i: tuple(map(float, p)) for i, p in points.items()})
        )
        object.__setattr__(self, "_rows", tuple(rows))
        object.__setattr__(self, "_origin", a)
        object.__setattr__(self, "_nf", nf)
        object.__setattr__(self, "_normal_sq", _dot(nf, nf))
        object.__setattr__(self, "_normal_scale", scale)
        object.__setattr__(self, "_scaled_length", length)
        object.__setattr__(self, "_axes", keep)

    def _projection(self, position, translation):
        local = _sub(_xyz(position), _xyz(translation))
        height = _dot(self._nf, _sub(local, self._origin))
        projected = tuple(
            x - height * n / self._normal_sq for x, n in zip(local, self._nf, strict=True)
        )
        return projected, float(height / self._normal_scale) / self._scaled_length

    def gap(self, position, translation):
        return self._projection(position, translation)[1]

    def trace(self, start, end, translation0=(0.0, 0.0, 0.0), translation1=(0.0, 0.0, 0.0)):
        """Return up to two strictly ordered intervals, with exact incidence checked before rounding."""
        p0, _ = self._projection(start, translation0)
        p1, _ = self._projection(end, translation1)
        xy0, xy1 = (tuple(p[i] for i in self._axes) for p in (p0, p1))
        spans = []
        for face in self._rows:
            w0, w1 = face.weights(xy0), face.weights(xy1)
            interval = _bounds(w0, w1)
            if interval is None:
                continue
            lo, hi = interval
            if (lo == 0 and min(w0) == 0) or (hi == 1 and min(w1) == 0):
                raise SolverFailure("a loaded endpoint lies on a patch edge or vertex")
            if lo == hi or any(a == b == 0 for a, b in zip(w0, w1, strict=True)):
                raise SolverFailure("path touches a patch vertex or follows an edge")

            def weights(t):
                return tuple(a + t * (b - a) for a, b in zip(w0, w1, strict=True))

            spans.append((lo, hi, face, weights(lo), weights(hi)))
        spans.sort(key=lambda row: (row[0], row[1], row[2].key))
        if not spans or spans[0][0] != 0 or spans[-1][1] != 1:
            raise SolverFailure("loaded path extends outside the finite patch")
        if len(spans) > 2:
            raise SolverFailure("loaded path has multiple edges or overlapping material faces")
        if len(spans) == 2:
            left, right = spans
            if left[1] != right[0]:
                raise SolverFailure("loaded face intervals overlap or have a gap")
            x = {i: w for i, w in zip(left[2].ids, left[4], strict=True) if w}
            y = {i: w for i, w in zip(right[2].ids, right[3], strict=True) if w}
            if len(x) != 2 or len(y) != 2:
                raise SolverFailure("transfer through a material vertex is ambiguous")
            if x != y:
                raise SolverFailure("geometrically adjacent faces have disconnected material edges")

        def at(t):
            return tuple(float(a + t * (b - a)) for a, b in zip(p0, p1, strict=True))

        segments = []
        for lo, hi, face, w0, w1 in spans:
            if float(lo) >= float(hi):
                raise SolverFailure("material interval below output precision")
            segments.append(
                MaterialFaceInterval(
                    face.key,
                    float(lo),
                    float(hi),
                    tuple(sorted(zip(face.ids, map(float, w0), strict=True))),
                    tuple(sorted(zip(face.ids, map(float, w1), strict=True))),
                    at(lo),
                    at(hi),
                )
            )
        return tuple(segments)
