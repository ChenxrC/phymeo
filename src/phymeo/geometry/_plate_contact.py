"""A finite convex polygon plus a ball, against affine material points and triangles."""

from fractions import Fraction as F

import numpy as np

from ..constraints.collision_shapes import CollisionMotion, CollisionShape
from ..core.capabilities import SolverFailure
from ._capsule_triangle_sweep import _CapsuleFace
from ._convex_hull import MaterialChart, plane_axes
from ._prism_clip import intersect
from ._round_sweep import _translation, _unit, _vector, _xyz, sweep_result
from ._sphere_triangle_sweep import _SphereFace
from ._tetrahedral_surface import _TriangleMotion
from ._triangle_pair_sweep import _cross3, _edge_candidates, _face_candidates, _FacePath, _Pose
from .collision_witness import (
    CollisionWitness,
    ConvexContactPatch,
    ConvexContactPoint,
    PlateContactPose,
    PlateContactTrace,
)


class RoundedPlate:
    def __init__(self, shape, path):
        self.shape, self.path, self.step = shape, path, _translation(path)
        self.ids = shape.vertex_ids
        self.points = {i: _vector(p) for i, p in zip(self.ids, path.start, strict=True)}
        a, b, c = (self.points[i] for i in self.ids[:3])
        normal = _cross3(b - a, c - a)
        if not any(normal) or any((p - a) @ normal for p in self.points.values()):
            raise ValueError("plate_not_nondegenerate_planar_polygon")
        self.normal = normal
        i, j = plane_axes(normal)
        chart = MaterialChart({(p[i], p[j]): k for k, p in self.points.items()})
        ring = tuple(chart.ids[p] for p in chart.ring)
        anchor = self.ids.index(ring[0])
        ids = self.ids[anchor:] + self.ids[:anchor]
        if ids != ring and ids != (ring[0], *ring[:0:-1]):
            raise ValueError("plate_not_ordered_strictly_convex")
        self.fan = []
        for b, c in zip(ids[1:-1], ids[2:], strict=True):
            v = (ids[0], b, c)
            tri = CollisionShape(shape.key, "triangle", v)
            points = np.array([self.points[k] for k in v], dtype=object)
            motion = _TriangleMotion(points, points + self.step, path.moving)
            self.fan.append((tri, motion, _FacePath(tri, motion)))


def _closest_triangles(first, second):
    candidates = list(_face_candidates(first, second))
    candidates.extend((w, v) for v, w in _face_candidates(second, first))
    candidates.extend(_edge_candidates(first, second))
    for wa, wb in candidates:
        pa = np.asarray(wa, dtype=object) @ first.p
        pb = np.asarray(wb, dtype=object) @ second.p
        delta = pb - pa
        yield delta @ delta, tuple(pa), tuple(pb), first.address(wa), second.address(wb)


class PlateSurfaceQuery:
    def __init__(self, plate, cloth, mc, plate_first):
        self.plate, self.cloth, self.plate_first = plate, cloth, plate_first
        self.radius = F(plate.shape.radius)
        self.start = np.array([_vector(p) for p in mc.start], dtype=object)
        end = self.start if mc.end is None else np.array([_vector(p) for p in mc.end], dtype=object)
        self.rate = end - self.start
        self.face = _FacePath(cloth, mc) if cloth.kind == "triangle" else None
        self.points = []
        features = []
        for i, node in enumerate(cloth.vertex_ids):
            ball = CollisionShape(cloth.key, "sphere", (node,), plate.shape.radius)
            vertex = CollisionMotion(
                mc.start[i : i + 1], None if mc.end is None else mc.end[i : i + 1]
            )
            for triangle, motion, _ in plate.fan:
                query = _SphereFace(triangle, ball, motion, vertex)
                features.extend(query.features)
                if self.face is None:
                    self.points.append(query)
        if self.face is not None:
            for i, k in enumerate(plate.ids):
                j = (i + 1) % len(plate.ids)
                edge = CollisionShape(
                    plate.shape.key, "capsule", (k, plate.ids[j]), plate.shape.radius
                )
                mp = plate.path
                path = CollisionMotion(
                    mp.start[[i, j]], None if mp.end is None else mp.end[[i, j]], mp.turn
                )
                features.extend(_CapsuleFace(edge, cloth, path, mc).features)
        self.features = tuple(dict.fromkeys(features))
        self.samples = {}

    def _closest(self, t):
        if self.face is not None:
            if not self.face.valid:
                raise SolverFailure("triangle_degenerate_on_path")
            cloth = _Pose(self.face, t)
            return min(
                c for _, _, fan in self.plate.fan for c in _closest_triangles(_Pose(fan, t), cloth)
            )
        candidates = []
        for query in self.points:
            center = query.center + t * query.speed
            rel = query.relative + t * query.rate
            for feature in query.features:
                w = feature.barycentric(t)
                if w is not None:
                    offset = np.asarray(w, dtype=object) @ rel
                    candidates.append(
                        (
                            offset @ offset,
                            tuple(center + offset),
                            tuple(center),
                            tuple(sorted(zip(query.face.vertex_ids, w, strict=True))),
                            ((self.cloth.vertex_ids[0], 1),),
                        )
                    )
        return min(candidates)

    def _patch(self, t, delta, direction):
        plate = {i: p + t * self.plate.step for i, p in self.plate.points.items()}
        cloth = dict(zip(self.cloth.vertex_ids, self.start + t * self.rate, strict=True))
        amax = max(p @ delta for p in plate.values())
        bmin = min(p @ delta for p in cloth.values())
        a = {k: p for k, p in plate.items() if p @ delta == amax}
        b = {k: p for k, p in cloth.items() if p @ delta == bmin}
        i, j = plane_axes(delta)
        ca = MaterialChart({(p[i], p[j]): k for k, p in a.items()})
        cb = MaterialChart({(p[i] - delta[i], p[j] - delta[j]): k for k, p in b.items()})
        projected = intersect(ca.ring, cb.ring)
        if not projected:
            raise SolverFailure("plate_closest_region_unresolved")
        vertices, points = [], []
        for q in projected:
            wa, wb = ca.weights(q), cb.weights(q)
            pa = sum(w * a[k] for k, w in wa)
            pb = sum(w * b[k] for k, w in wb)
            vertices.append(pa)
            near = _xyz(pa + _vector(float(self.radius) * direction))
            far = _xyz(pb)
            wa = tuple((k, float(w)) for k, w in wa)
            wb = tuple((k, float(w)) for k, w in wb)
            sides = (near, far, wa, wb) if self.plate_first else (far, near, wb, wa)
            points.append(ConvexContactPoint(*sides))
        area = sum(
            (
                _cross3(vertices[k] - vertices[0], vertices[k + 1] - vertices[0])
                for k in range(1, len(vertices) - 1)
            ),
            np.array((F(0),) * 3, dtype=object),
        )
        if (area @ delta < 0) == self.plate_first and len(vertices) > 2:
            points.reverse()
        normal = _xyz(direction if self.plate_first else -direction)
        first, second = (a, b) if self.plate_first else (b, a)
        patch = ConvexContactPatch(
            normal,
            tuple(sorted(first)),
            tuple(sorted(second)),
            min(len(points) - 1, 2),
            _unit(area)[1] / 2,
            tuple(points),
        )
        if len(a) == len(plate):
            feature = "front" if delta @ self.plate.normal > 0 else "back"
        else:
            feature = "rim" if len(a) > 1 else "corner"
        return patch, feature

    def pose(self, t):
        if t in self.samples:
            return self.samples[t]
        d2, pa, pb, wa, wb = self._closest(t)
        delta = np.array(pb, dtype=object) - pa
        direction, distance = _unit(delta)
        signed = d2 - self.radius * self.radius
        gap = float(signed / F(distance + float(self.radius)))
        if direction is None:
            patch, feature = None, "middle"
            wa = tuple((i, float(w)) for i, w in wa if w)
            wb = tuple((i, float(w)) for i, w in wb if w)
            sides = (None, _xyz(pb), wa, wb) if self.plate_first else (_xyz(pb), None, wb, wa)
            witness = CollisionWitness(gap, None, *sides, "host-rounded-plate")
        else:
            patch, feature = self._patch(t, delta, direction)
            p = patch.points[0]
            witness = CollisionWitness(
                gap,
                patch.normal,
                p.first_point,
                p.second_point,
                p.first_weights,
                p.second_weights,
                "host-rounded-plate",
            )
        relation = "overlap" if signed < 0 else "touching" if signed == 0 else "separated"
        pose = PlateContactPose(relation, feature, witness, patch)
        self.samples[t] = pose, signed
        return pose, signed

    def sample(self, t):
        pose, signed = self.pose(t)
        return pose.witness, signed

    def result(self, moving, seeds):
        start, _ = self.pose(F(0))
        if not moving:
            return (
                start.relation,
                "rounded_plate",
                start.witness,
                None,
                PlateContactTrace(start, start, None),
            )
        witness, sweep = sweep_result(self, seeds)
        entry = None if sweep.witness_time is None else self.pose(F(sweep.witness_time))[0]
        end, _ = self.pose(F(1))
        return (
            ("hit" if entry else "separated"),
            "rounded_plate_sweep",
            witness,
            sweep,
            PlateContactTrace(start, end, entry),
        )
