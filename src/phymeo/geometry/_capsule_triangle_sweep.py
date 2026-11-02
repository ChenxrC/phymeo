"""Distance and first contact between a translating capsule and an affine triangle."""

from dataclasses import dataclass
from fractions import Fraction as F

import numpy as np

from ..constraints.collision_shapes import CollisionMotion, CollisionShape
from ..core.capabilities import SolverFailure
from ._round_sweep import _axis_features, _translation, _unit, _vector, _weights, _xyz
from ._sphere_triangle_sweep import _SphereFace, pack_surface_conditions
from ._sweep_polynomial import (
    Roots,
    add,
    affine_vectors,
    cross,
    difference,
    dot,
    first_entry,
    first_zero_entry,
    mul,
    split_projection_roots,
    sub,
)
from ._triangle_pair_sweep import _EDGES, _ZERO_VECTOR, _FacePath, _on_edge, _Pose
from .collision_witness import CapsuleAxisPoint, CollisionWitness


@dataclass(frozen=True)
class _ShaftFeature:
    conditions: tuple
    # Only interior edge contacts need a nonparallel radius-equality root.
    equation: tuple | None = None

    @property
    def capsule_feature(self):
        return "vertex" if self.equation is None else "edge"

    def entry(self, proposal):
        if self.equation is None:
            return first_entry(self.conditions, proposal)
        return first_zero_entry(self.equation, self.conditions[1:], proposal)

    def packed(self):
        return pack_surface_conditions(self.conditions)


def _shaft_edge(u, p, q, r2):
    # p and q are relative to the translating first axis endpoint.
    edge = difference(q, p)
    n = cross(u, edge)
    n2 = dot(n, n)
    height = dot(p, n)
    clearance = sub(mul(r2, n2), mul(height, height))
    if len(clearance) < 2:
        # Persistent tangency starts at a parameter boundary (or at a
        # parallel pose). A cap or a cloth vertex then realizes the distance.
        return None
    regular, _ = split_projection_roots(Roots(clearance).chain[0], n2)
    if len(regular) < 2:
        return None
    # At a regular root n2>0. Segment parameters have numerators s and t.
    # The products implement their closed unit intervals in one gate each.
    s = dot(cross(p, edge), n)
    t = dot(cross(p, u), n)
    return _ShaftFeature((clearance, mul(s, sub(n2, s)), mul(t, sub(n2, t))), regular)


def _sample_candidates(start, axis, tri):
    """Generate the extreme minimizers, including a piercing of the face interior."""
    for s in (F(0), F(1)):
        w = tri.projected(start + s * axis)
        if w is not None:
            yield s, w
    across = axis @ tri.n
    if across:
        s = ((tri.p[0] - start) @ tri.n) / across
        if 0 <= s <= 1:
            w = tri.projected(start + s * axis)
            if w is not None:
                yield s, w
    for i, j in _EDGES:
        edge = tri.p[j] - tri.p[i]
        for f in _axis_features(axis, edge, start - tri.p[i], _ZERO_VECTOR, F(0)):
            # These static feature intervals contain t=0. Their affine
            # coefficients give the closest endpoint or both segment interiors.
            yield f.s[0], _on_edge(i, j, f.t[0])


class _CapsuleFace:
    def __init__(self, a, b, ma, mb):
        self.face_first = a.kind == "triangle"
        face, cap, mf, mc = (a, b, ma, mb) if self.face_first else (b, a, mb, ma)
        self.cap = cap
        self.face = _FacePath(face, mf)
        self.start = _vector(mc.start[0])
        self.axis = _vector(mc.start[1]) - self.start
        self.speed = _translation(mc)
        self.radius = F(cap.radius)
        self.features = ()
        if not self.face.valid or not (ma.moving or mb.moving):
            return
        uu = self.axis @ self.axis
        features = []
        for i in range(2 if uu else 1):
            tip = CollisionShape(cap.key, "sphere", (cap.vertex_ids[i],), cap.radius)
            mp = CollisionMotion(
                mc.start[i : i + 1], None if mc.end is None else mc.end[i : i + 1], mc.turn
            )
            features.extend(_SphereFace(face, tip, mf, mp).features)
        if uu:
            vertices = affine_vectors(self.face.start - self.start, self.face.step - self.speed)
            u = tuple((x,) for x in self.axis)
            r2 = (self.radius**2,)
            for p in vertices:
                along = dot(p, u)
                clearance = add(mul((uu,), sub(r2, dot(p, p))), mul(along, along))
                features.append(_ShaftFeature((clearance, along, sub((uu,), along))))
            for i, j in _EDGES:
                edge = _shaft_edge(u, vertices[i], vertices[j], r2)
                if edge is not None:
                    features.append(edge)
        self.features = tuple(features)

    def sample(self, time):
        if not self.face.valid:
            raise SolverFailure("triangle_degenerate_on_path")
        start = self.start + time * self.speed
        tri = _Pose(self.face, time)
        best = None
        ties = []
        for s, w in _sample_candidates(start, self.axis, tri):
            delta = np.asarray(w, dtype=object) @ tri.p - (start + s * self.axis)
            d2 = delta @ delta
            if best is None or d2 < best:
                best = d2
                ties = []
            if d2 == best:
                ties.append((s, w))
        low, high = min(ties), max(ties)
        if not any(self.axis):
            low, high = (F(0), low[1]), (F(1), low[1])

        def endpoint(s, w):
            return CapsuleAxisPoint(
                float(s),
                _xyz(start + s * self.axis),
                _xyz(np.asarray(w, dtype=object) @ tri.p),
                tuple(sorted(zip(tri.ids, map(float, w), strict=True))),
            )

        # The closest-pair set is convex and the displacement is unique.
        # Interpolating the two extremes recovers the face weights at its center.
        s = (F(low[0]) + high[0]) / 2
        w = np.array([(F(a) + b) / 2 for a, b in zip(low[1], high[1], strict=True)], dtype=object)
        axis_point = start + s * self.axis
        face_point = w @ tri.p
        signed = best - self.radius**2
        direction, distance = _unit(face_point - axis_point)
        gap = float(signed / F(distance + self.cap.radius))
        face_pos = _xyz(face_point)
        cap_pos = normal = None
        if direction is not None:
            normal = _xyz(-direction if self.face_first else direction)
            cap_pos = (
                face_pos
                if not signed
                else _xyz(np.asarray(axis_point, float) + self.cap.radius * direction)
            )
        cap_weights = _weights(self.cap, s)
        face_weights = tuple(sorted(zip(tri.ids, map(float, w), strict=True)))
        if self.face_first:
            sides = face_pos, cap_pos, face_weights, cap_weights
        else:
            sides = cap_pos, face_pos, cap_weights, face_weights
        return CollisionWitness(
            gap,
            normal,
            *sides,
            "host-affine-capsule-face",
            capsule_axis_span=(endpoint(*low), endpoint(*high)),
        ), signed


def capsule_face_query(a, b, ma, mb):
    if {a.kind, b.kind} != {"capsule", "triangle"}:
        return None
    face, mc = (a, mb) if a.kind == "triangle" else (b, ma)
    if face.radius or (mc.turn is not None and mc.turn.angle != 0):
        return None
    return _CapsuleFace(a, b, ma, mb)
