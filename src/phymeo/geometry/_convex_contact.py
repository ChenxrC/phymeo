"""Translating convex polyhedra: SAT intervals, contact patches, and material witnesses."""

from fractions import Fraction as F
from itertools import product
from math import ceil, floor

import numpy as np

from ..core.capabilities import SolverFailure
from ._convex_hull import MaterialChart, direction_key, plane_axes, sparse_weights
from ._prism_clip import intersect
from ._round_sweep import _GRID, _axis_features, _unit, _xyz
from ._triangle_pair_sweep import _ZERO_VECTOR, _cross3
from .collision_witness import (
    CollisionSweep,
    CollisionWitness,
    ConvexContactPatch,
    ConvexContactPoint,
    ConvexContactPose,
    ConvexHullTrace,
    MaterialContactPoint,
)

_MAX_AXES = 4096
_SOURCE = "host-translating-convex-hulls"


def _public(weights):
    return tuple((i, float(w)) for i, w in weights)


def _projection(shape, d):
    values = [(i, shape.points[i] @ d) for i in shape.vertices]
    lo, hi = min(x for _, x in values), max(x for _, x in values)
    return lo, hi, tuple(i for i, x in values if x == lo), tuple(i for i, x in values if x == hi)


class _Axis:
    def __init__(self, d, a, b):
        self.vector = np.array(d, dtype=object)
        self.norm2 = self.vector @ self.vector
        al, ah, alo, ahi = _projection(a, self.vector)
        bl, bh, blo, bhi = _projection(b, self.vector)
        self.forward, self.backward = bl - ah, al - bh
        self.support = ((ahi, blo), (alo, bhi))
        self.speed = (b.step - a.step) @ self.vector

    def interval(self):
        if not self.speed:
            return (F(0), F(1)) if max(self.forward, self.backward) <= 0 else None
        t, u = sorted((-self.forward / self.speed, self.backward / self.speed))
        lo, hi = max(F(0), t), min(F(1), u)
        return (lo, hi) if lo <= hi else None

    def gaps(self, t):
        return self.forward + self.speed * t, self.backward - self.speed * t


def _support_chart(shape, support, t, origin, d, norm2, cols):
    coords = {}
    i, j = cols
    for node in support:
        x = shape.points[node] + t * shape.step - origin
        y = x - ((x @ d) / norm2) * d
        coords[y[i], y[j]] = node
    return MaterialChart(coords)


def _patch(a, b, t, d, fa, fb):
    nn = d @ d
    origin = a.points[fa[0]] + t * a.step
    cols = plane_axes(d)
    ca, cb = (
        _support_chart(shape, ids, t, origin, d, nn, cols) for shape, ids in ((a, fa), (b, fb))
    )
    polygon = intersect(ca.ring, cb.ring)
    if not polygon:
        return None
    points = []
    vertices = []
    for xy in polygon:
        wa, wb = ca.weights(xy), cb.weights(xy)
        if wa is None or wb is None:
            raise SolverFailure("convex_material_chart_unresolved")
        pa, pb = a.point(wa, t), b.point(wb, t)
        vertices.append(pa)
        points.append(ConvexContactPoint(_xyz(pa), _xyz(pb), _public(wa), _public(wb)))
    vector_area = sum(
        (
            _cross3(vertices[i] - vertices[0], vertices[i + 1] - vertices[0])
            for i in range(1, len(vertices) - 1)
        ),
        _ZERO_VECTOR.copy(),
    )
    if vector_area @ d < 0:
        points.reverse()
    anchor = min(
        range(len(points)), key=lambda i: (points[i].first_weights, points[i].second_weights)
    )
    points = points[anchor:] + points[:anchor]
    return ConvexContactPatch(
        _xyz(_unit(d)[0]), fa, fb, min(len(points) - 1, 2), _unit(vector_area)[1] / 2, tuple(points)
    )


def _closest(a, b, t):
    """Finite vertex/face and true-edge/edge candidates for a disjoint pair."""
    best = None

    def keep(d2, wa, wb):
        nonlocal best
        key = d2, wa, wb
        if best is None or key < best:
            best = key

    for shape, target, reverse in ((a, b, False), (b, a, True)):
        relative = t * (shape.step - target.step)
        for node in shape.vertices:
            x = shape.points[node] + relative
            for face in target.faces:
                height = x @ face.normal - face.plane
                nn = face.normal @ face.normal
                d2 = height * height / nn
                if best is not None and d2 > best[0]:
                    continue
                projected = x - (height / nn) * face.normal
                i, j = face.cols
                w = face.chart.weights((projected[i], projected[j]))
                if w is not None:
                    v = ((node, 1),)
                    keep(d2, w, v) if reverse else keep(d2, v, w)
    for (i, j), (k, m) in product(a.edges, b.edges):
        u, v = a.points[j] - a.points[i], b.points[m] - b.points[k]
        w = a.points[i] - b.points[k] + t * (a.step - b.step)
        for f in _axis_features(u, v, w, _ZERO_VECTOR, F(0)):
            if best is None or f.c <= best[0]:
                s, r = f.s[0], f.t[0]
                keep(f.c, sparse_weights((i, j), (1 - s, s)), sparse_weights((k, m), (1 - r, r)))
    _, wa, wb = best
    pa, pb = a.point(wa, t), b.point(wb, t)
    normal, dist = _unit(pb - pa)
    if normal is None:
        raise SolverFailure("convex_disjoint_witness_unresolved")
    return CollisionWitness(
        dist, _xyz(normal), _xyz(pa), _xyz(pb), _public(wa), _public(wb), _SOURCE
    )


class HullPairQuery:
    features = ()

    def __init__(self, a, b):
        self.a, self.b = a, b
        axes = {direction_key(face.normal) for shape in (a, b) for face in shape.faces}
        for u, v in product(a.edge_rays, b.edge_rays):
            ray = direction_key(_cross3(u, v))
            if ray is not None:
                axes.add(ray)
                if len(axes) > _MAX_AXES:
                    raise SolverFailure("convex_axis_capacity")
        self.axes = tuple(_Axis(d, a, b) for d in sorted(axes))

    def device_inputs(self):
        # One moving frame and uniform scale keep the FP32 projection proposals
        # finite. HOST uses the original stored coordinates for final decisions.
        origin = self.a.points[self.a.vertices[0]]
        points = [shape.points[i] - origin for shape in (self.a, self.b) for i in shape.vertices]
        points.append(self.b.step - self.a.step)
        scale = max(abs(x) for p in points for x in p) or 1
        return (
            np.array([[float(x / scale) for x in p] for p in points], dtype=np.float32),
            len(self.a.vertices),
            len(self.b.vertices),
            np.array([axis.vector for axis in self.axes], dtype=np.float32),
        )

    def _window(self, proposals):
        order = range(len(self.axes))
        if proposals is not None:
            # Examine claimed misses and shortest time windows first. Only exact
            # projection intervals can eliminate a pair or determine contact time.
            order = sorted(
                order,
                key=lambda i: -1 if proposals[i] is None else proposals[i][1] - proposals[i][0],
            )
        lo, hi = F(0), F(1)
        for i in order:
            interval = self.axes[i].interval()
            if interval is None:
                return None
            lo, hi = max(lo, interval[0]), min(hi, interval[1])
            if lo > hi:
                return None
        return lo, hi

    def sample(self, t):
        best = None
        ties = []
        for axis in self.axes:
            for side, gap in enumerate(axis.gaps(t)):
                # g*abs(g)/|d|² preserves the order of signed normalized gaps.
                score = gap * abs(gap) / axis.norm2
                if best is None or score > best:
                    best, ties = score, []
                if score == best:
                    d = axis.vector if side == 0 else -axis.vector
                    ties.append((gap, d, *axis.support[side]))
        patches = []
        if best > 0:
            for gap, d, fa, fb in ties:
                patch = _patch(self.a, self.b, t, d, fa, fb)
                if patch is not None:
                    p = patch.points[0]
                    w = CollisionWitness(
                        float(gap) / _unit(d)[1],
                        patch.normal,
                        p.first_point,
                        p.second_point,
                        p.first_weights,
                        p.second_weights,
                        _SOURCE,
                    )
                    return ConvexContactPose("separated", w, ())
            return ConvexContactPose("separated", _closest(self.a, self.b, t), ())
        for _, d, fa, fb in ties:
            patch = _patch(self.a, self.b, t, d, fa, fb)
            if patch is None:
                raise SolverFailure("convex_support_clip_unresolved")
            patches.append(patch)
        patch = patches[0]
        point = patch.points[0]
        gap, d, *_ = ties[0]
        touching = best == 0
        points = (
            tuple(
                MaterialContactPoint(p.first_point, p.first_weights, p.second_weights)
                for p in patch.points
            )
            if touching
            else ()
        )
        w = CollisionWitness(
            float(gap) / _unit(d)[1],
            patch.normal if len(patches) == 1 else None,
            point.first_point,
            point.second_point,
            point.first_weights,
            point.second_weights,
            _SOURCE,
            intersection_points=points,
        )
        return ConvexContactPose("touching" if touching else "overlap", w, tuple(patches))

    def result(self, moving, proposals):
        samples = {}

        def pose(t):
            if t not in samples:
                samples[t] = self.sample(t)
            return samples[t]

        start = pose(F(0))
        end = pose(F(1)) if moving else start
        window = self._window(proposals)
        entry = exit_pose = first_ratio = last_ratio = first_t = witness_time = witness = None
        if window is not None:
            lo, hi = window
            first_ratio, last_ratio = (lo.numerator, lo.denominator), (hi.numerator, hi.denominator)
            entry = pose(lo) if moving else start
            exit_pose = pose(hi) if moving else start
            first_t = (floor(lo * _GRID) / _GRID, ceil(lo * _GRID) / _GRID)
            witness_time = sum(first_t) / 2
            witness = pose(F(witness_time)).witness if moving else start.witness
        trace = ConvexHullTrace(
            start, end, first_ratio, last_ratio, entry, exit_pose, len(self.axes)
        )
        sweep = None
        if moving:
            sweep = CollisionSweep(
                start.witness.gap_m, end.witness.gap_m, first_t, witness_time, witness
            )
        status = ("hit" if window is not None else "separated") if moving else end.relation
        return (
            status,
            "convex_sat_interval" if moving else "convex_support_region",
            end.witness,
            sweep,
            trace,
        )
