"""Continuous queries between a translating sphere and a two-sided affine triangle."""

from dataclasses import dataclass
from fractions import Fraction as F

import numpy as np

from ..core.capabilities import SolverFailure
from ._round_sweep import _translation, _unit, _vector, _xyz
from ._sweep_polynomial import (
    Roots,
    add,
    affine_vectors,
    cross,
    difference,
    dot,
    face_coordinates,
    first_entry,
    mul,
    sub,
    value,
)
from .collision_witness import CollisionWitness


@dataclass(frozen=True)
class _SurfaceFeature:
    conditions: tuple
    weights: tuple
    denominator: tuple

    def entry(self, proposal):
        return first_entry(self.conditions, proposal)

    def packed(self):
        # Four inequalities of degree at most six, each independently scaled.
        row = [0.0] * 28
        for i, poly in enumerate(self.conditions):
            scale = max(map(abs, poly), default=0) or 1
            row[7 * i : 7 * i + len(poly)] = [float(c / scale) for c in poly]
        return row

    def barycentric(self, t):
        den = value(self.denominator, t)
        numerators = tuple(value(p, t) for p in self.weights)
        return tuple(F(x, den) for x in numerators) if min(numerators) >= 0 else None


class _SphereFace:
    def __init__(self, a, b, ma, mb):
        self.face_first = a.kind == "triangle"
        self.face, self.ball = (a, b) if self.face_first else (b, a)
        mf, ms = (ma, mb) if self.face_first else (mb, ma)
        self.center = _vector(ms.start[0])
        self.speed = _translation(ms)
        self.radius = F(self.ball.radius)
        self.relative = np.array([_vector(p) - self.center for p in mf.start], dtype=object)
        end = mf.start if mf.end is None else mf.end
        self.rate = np.array(
            [_vector(y) - _vector(x) - self.speed for x, y in zip(mf.start, end, strict=True)]
        )
        vertices = affine_vectors(self.relative, self.rate)
        u, v = difference(vertices[1], vertices[0]), difference(vertices[2], vertices[0])
        normal = cross(u, v)
        area2 = dot(normal, normal)
        self.features = ()
        if not area2 or area2[0] == 0 or Roots(area2).count(F(0), F(1)):
            return
        r2 = (self.radius * self.radius,)
        features = []
        for i, p in enumerate(vertices):
            weights = tuple((1,) if j == i else () for j in range(3))
            features.append(_SurfaceFeature((sub(r2, dot(p, p)),), weights, (1,)))
        for i, j in ((0, 1), (1, 2), (2, 0)):
            p = vertices[i]
            edge = difference(vertices[j], p)
            den = dot(edge, edge)
            along = sub((), dot(p, edge))
            remain = sub(den, along)
            weights = [()] * 3
            weights[i], weights[j] = remain, along
            # Multiply the squared-distance inequality by positive |edge|².
            clearance = add(mul(sub(r2, dot(p, p)), den), mul(along, along))
            features.append(_SurfaceFeature((clearance, remain, along), tuple(weights), den))
        height, weights = face_coordinates(vertices[0], u, v, normal, area2)
        clearance = sub(mul(r2, area2), mul(height, height))
        features.append(_SurfaceFeature((clearance, *weights), weights, area2))
        self.features = tuple(features)

    def sample(self, t):
        if not self.features:
            raise SolverFailure("triangle_degenerate_on_path")
        vertices = self.relative + t * self.rate
        candidates = []
        for feature in self.features:
            weights = feature.barycentric(t)
            if weights is not None:
                point = np.asarray(weights, dtype=object) @ vertices
                candidates.append((point @ point, point, weights))
        dist2, point, weights = min(candidates, key=lambda c: c[0])
        signed = dist2 - self.radius * self.radius
        direction, distance = _unit(-point)
        gap = float(signed / F(distance + float(self.radius)))
        center = self.center + t * self.speed
        face_point = _xyz(center + point)
        ball_point = (
            None
            if direction is None
            else _xyz(np.asarray(center, float) - self.ball.radius * direction)
        )
        normal = None if direction is None else _xyz(direction if self.face_first else -direction)
        material = tuple(sorted(zip(self.face.vertex_ids, map(float, weights), strict=True)))
        ball_material = ((self.ball.vertex_ids[0], 1.0),)
        first, second = (face_point, ball_point) if self.face_first else (ball_point, face_point)
        wa, wb = (material, ball_material) if self.face_first else (ball_material, material)
        return CollisionWitness(
            gap, normal, first, second, wa, wb, "host-affine-sphere-face"
        ), signed


def sphere_face_query(a, b, ma, mb):
    if {a.kind, b.kind} != {"sphere", "triangle"} or not (ma.moving or mb.moving):
        return None
    face, ms = (a, mb) if a.kind == "triangle" else (b, ma)
    if face.radius != 0 or (ms.turn is not None and ms.turn.angle != 0):
        return None
    return _SphereFace(a, b, ma, mb)
