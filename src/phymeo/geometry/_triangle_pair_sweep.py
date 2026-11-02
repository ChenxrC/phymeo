"""Zero-thickness triangle pairs on affine paths, with same-body material identity."""

from dataclasses import dataclass
from fractions import Fraction as F

import numpy as np

from ..core.capabilities import SolverFailure
from ._round_sweep import _axis_features, _unit, _xyz
from ._sweep_polynomial import (
    Roots,
    affine_vectors,
    cross,
    difference,
    dot,
    face_coordinates,
    first_zero_entry,
    mul,
    split_projection_roots,
    sub,
)
from .collision_witness import CollisionWitness, MaterialContactPoint

_EDGES = ((0, 1), (1, 2), (2, 0))
_ZERO_VECTOR = np.array((F(0), F(0), F(0)), dtype=object)


@dataclass(frozen=True)
class _TouchFeature:
    equation: tuple
    limits: tuple
    edge_pair: bool = False

    def entry(self, proposal):
        return first_zero_entry(self.equation, self.limits, proposal)

    def packed(self):
        # Cubic coplanarity equation, then up to three quartic material gates.
        row = [0.0] * 19
        for i, p in enumerate((self.equation, *self.limits)):
            offset = 0 if i == 0 else 4 + (i - 1) * 5
            scale = max(map(abs, p), default=0) or 1
            row[offset : offset + len(p)] = [float(c / scale) for c in p]
        return row


class _FacePath:
    def __init__(self, shape, motion):
        self.ids = shape.vertex_ids
        self.start = np.array([[F(x) for x in p] for p in motion.start], dtype=object)
        end = motion.start if motion.end is None else motion.end
        self.step = np.array([[F(x) for x in p] for p in end], dtype=object) - self.start
        self.vertices = affine_vectors(self.start, self.step)
        p, q, r = self.vertices
        self.u, self.v = difference(q, p), difference(r, p)
        self.normal = cross(self.u, self.v)
        self.area2 = dot(self.normal, self.normal)
        self.valid = bool(
            self.area2 and self.area2[0] != 0 and not Roots(self.area2).count(F(0), F(1))
        )

    def vertex_face(self, vertex):
        h, limits = face_coordinates(
            difference(self.vertices[0], vertex), self.u, self.v, self.normal, self.area2
        )
        return _TouchFeature(h, limits)


def _edge_entries(p, p1, q, q1):
    u, v, w = difference(p1, p), difference(q1, q), difference(q, p)
    normal = cross(u, v)
    height = dot(w, normal)
    if len(height) < 2:
        # Persistently coplanar interior crossings persist locally. Their first
        # contact from separation is on an endpoint, covered by vertex/face.
        return
    remaining = Roots(height).chain[0]
    s, t = cross(w, v), cross(w, u)
    for axis, s_num, t_num in zip(normal, s, t, strict=True):
        if len(remaining) < 2:
            break
        usable, remaining = split_projection_roots(remaining, axis)
        if len(usable) > 1:
            # At each retained coplanarity root axis != 0. Each product tests a
            # projected parameter in [0,1], independent of the sign of axis.
            limits = (mul(s_num, sub(axis, s_num)), mul(t_num, sub(axis, t_num)))
            yield _TouchFeature(usable, limits, True)
    # Roots left singular in all three charts have parallel axes. If those
    # closed segments intersect, at least one endpoint belongs to the other.


def _cross3(a, b):
    return np.array([a[i] * b[j] - a[j] * b[i] for i, j in ((1, 2), (2, 0), (0, 1))], dtype=object)


class _Pose:
    def __init__(self, face, time):
        self.ids = face.ids
        self.p = face.start + time * face.step
        self.u, self.v = self.p[1] - self.p[0], self.p[2] - self.p[0]
        self.n = _cross3(self.u, self.v)
        self.den = self.n @ self.n

    def projected(self, point):
        rel = point - self.p[0]
        y = (_cross3(rel, self.v) @ self.n) / self.den
        z = (_cross3(self.u, rel) @ self.n) / self.den
        w = (1 - y - z, y, z)
        return w if min(w) >= 0 else None

    def address(self, weights):
        return tuple(sorted(zip(self.ids, weights, strict=True)))


def _on_edge(i, j, t):
    w = [0, 0, 0]
    w[i], w[j] = 1 - t, t
    return tuple(w)


def _on_vertex(i):
    return tuple(int(k == i) for k in range(3))


def _vertex_candidates(source, target, *, disjoint=False):
    for i, p in enumerate(source.p):
        if disjoint and source.ids[i] in target.ids:
            continue
        target_w = target.projected(p)
        if target_w is not None:
            yield _on_vertex(i), target_w
        if disjoint:
            # A vertex opposite a face can be nearest to its boundary rather
            # than to the supporting plane, in particular at an off-grid event.
            for k, later in _EDGES:
                d = target.p[later] - target.p[k]
                s = min(1, max(0, ((p - target.p[k]) @ d) / (d @ d)))
                yield _on_vertex(i), _on_edge(k, later, s)


def _face_candidates(source, target):
    """Vertex projections and transverse edge/face intersections at one pose."""
    yield from _vertex_candidates(source, target)
    heights = [(p - target.p[0]) @ target.n for p in source.p]
    for i, j in _EDGES:
        hi, hj = heights[i], heights[j]
        if hi * hj >= 0:
            continue  # Endpoints on the plane were considered above.
        t = hi / (hi - hj)
        target_w = target.projected(source.p[i] + t * (source.p[j] - source.p[i]))
        if target_w is not None:
            yield _on_edge(i, j, t), target_w


def _edge_candidates(a, b, *, disjoint=False):
    for i, j in _EDGES:
        u = a.p[j] - a.p[i]
        for b_start, b_end in _EDGES:
            if disjoint and (
                a.ids[i] in (b.ids[b_start], b.ids[b_end])
                or a.ids[j] in (b.ids[b_start], b.ids[b_end])
            ):
                continue
            v = b.p[b_end] - b.p[b_start]
            features = _axis_features(u, v, a.p[i] - b.p[b_start], _ZERO_VECTOR, F(0))
            minima = [(f.c, f.s[0], f.t[0]) for f in features]
            best = min(q for q, _, _ in minima)
            for q, s, t in minima:
                if q == best:
                    yield _on_edge(i, j, s), _on_edge(b_start, b_end, t)


def _as_weights(weights):
    return tuple((node, float(w)) for node, w in weights)


def _material_points(points):
    return tuple(
        MaterialContactPoint(_xyz(points[key]), *map(_as_weights, key)) for key in sorted(points)
    )


def _center_witness(points, shared):
    # The material maps are affine. Averaging all intersection generators gives
    # an interior witness off the shared simplex whenever any generator is off it.
    generators = points | shared
    count = len(generators)
    point = _xyz(sum(generators.values()) / count)
    template = next(iter(generators))
    weights = []
    for side in (0, 1):
        weights.append(
            tuple(
                (node, float(sum((key[side][i][1] for key in generators), F(0)) / count))
                for i, (node, _) in enumerate(template[side])
            )
        )
    return CollisionWitness(
        0.0,
        None,
        point,
        point,
        *weights,
        source="host-affine-triangle-pair",
        intersection_points=_material_points(points),
        shared_material_points=_material_points(shared),
    )


def _record(a, b, candidates, *, incident, source="host-affine-triangle-pair"):
    best = None
    points = {}
    shared = {}
    for wa, wb in candidates:
        pa = np.asarray(wa, dtype=object) @ a.p
        pb = np.asarray(wb, dtype=object) @ b.p
        delta = pb - pa
        d2 = delta @ delta
        if best is not None and d2 > best[0][0]:
            continue
        addresses = (a.address(wa), b.address(wb))
        if not d2:
            if incident:
                first, second = (
                    tuple((node, w) for node, w in material if w) for material in addresses
                )
                if first == second:
                    shared[addresses] = pa
                    continue
            points[addresses] = pa
        key = (d2, sum(bool(x) for x in (*wa, *wb)), addresses)
        if best is None or key < best[0]:
            best = (key, pa, pb, delta)
    if shared:
        if points:
            return _center_witness(points, shared), F(0)
        return CollisionWitness(
            0.0, None, None, None, (), (), source, shared_material_points=_material_points(shared)
        ), F(0)
    key, pa, pb, delta = best
    normal, distance = _unit(delta)
    return CollisionWitness(
        distance,
        None if normal is None else _xyz(normal),
        _xyz(pa),
        _xyz(pb),
        *map(_as_weights, key[2]),
        source=source,
        intersection_points=_material_points(points),
    ), key[0]


class _TrianglePair:
    def __init__(self, a, b, ma, mb, *, sweep=True):
        self.a, self.b = _FacePath(a, ma), _FacePath(b, mb)
        self.shared = (
            set(self.a.ids).intersection(self.b.ids) if a.key.body == b.key.body else set()
        )
        self.features = ()
        if sweep and self.a.valid and self.b.valid and (ma.moving or mb.moving):
            # A vertex's own face and edges with a common material endpoint
            # coincide by construction. Keep the rest of each triangle pair.
            features = [
                self.b.vertex_face(p)
                for node, p in zip(self.a.ids, self.a.vertices, strict=True)
                if node not in self.shared
            ]
            features.extend(
                self.a.vertex_face(q)
                for node, q in zip(self.b.ids, self.b.vertices, strict=True)
                if node not in self.shared
            )
            for i, j in _EDGES:
                for k, later in _EDGES:
                    if self.shared and (
                        {self.a.ids[i], self.a.ids[j]} & {self.b.ids[k], self.b.ids[later]}
                    ):
                        continue
                    features.extend(
                        _edge_entries(
                            self.a.vertices[i],
                            self.a.vertices[j],
                            self.b.vertices[k],
                            self.b.vertices[later],
                        )
                    )
            self.features = tuple(features)

    def sample(self, time):
        if not self.a.valid or not self.b.valid:
            raise SolverFailure("triangle_degenerate_on_path")
        a, b = _Pose(self.a, time), _Pose(self.b, time)
        candidates = list(_face_candidates(a, b))
        candidates.extend((wa, wb) for wb, wa in _face_candidates(b, a))
        candidates.extend(_edge_candidates(a, b))
        return _record(a, b, candidates, incident=bool(self.shared))

    def event_witness(self, time):
        """Closest disjoint feature pair at the reported sample of an incident hit."""
        a, b = _Pose(self.a, time), _Pose(self.b, time)
        candidates = list(_vertex_candidates(a, b, disjoint=True))
        candidates.extend((wa, wb) for wb, wa in _vertex_candidates(b, a, disjoint=True))
        candidates.extend(_edge_candidates(a, b, disjoint=True))
        witness, _ = _record(
            a, b, candidates, incident=False, source="host-incident-event-features"
        )
        return witness


def triangle_pair_query(a, b, ma, mb):
    if a.kind != b.kind or a.kind != "triangle" or a.radius or b.radius:
        return None
    return _TrianglePair(a, b, ma, mb)
