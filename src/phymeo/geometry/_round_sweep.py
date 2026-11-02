"""Closed translating round primitives and fixed-normal moving half-spaces.

Each closest feature of two translated segments has affine parameters on a
closed time interval. Its squared clearance is a convex quadratic there.
Device root proposals narrow a checked time search on the stored coordinates.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor, isfinite

import numpy as np

from .collision_witness import CollisionSweep, CollisionWitness

_GRID = 1 << 40
_ZERO = Fraction(0)
_ONE = Fraction(1)
_HALF = Fraction(1, 2)


def _vector(point):
    return np.array([Fraction(float(v)) for v in point], dtype=object)


def _xyz(point):
    return tuple(map(float, point))


def _unit(delta):
    scale = max(map(abs, delta))
    if not scale:
        return None, 0.0
    direction = np.asarray(delta / scale, dtype=float)
    magnitude = float(np.hypot.reduce(direction))
    return direction / magnitude, float(scale) * magnitude


def _weights(shape, t):
    if len(shape.vertex_ids) == 1:
        return ((shape.vertex_ids[0], 1.0),)
    return tuple(sorted(zip(shape.vertex_ids, (float(1 - t), float(t)), strict=True)))


def _translation(path):
    if path.turn is not None:
        return _vector(path.turn.pivot_end) - _vector(path.turn.pivot_start)
    return (
        _vector(path.end[0]) - _vector(path.start[0])
        if path.end is not None
        else _vector((0, 0, 0))
    )


@dataclass(frozen=True)
class _Feature:
    lo: Fraction
    hi: Fraction
    a: Fraction
    b: Fraction
    c: Fraction
    s: tuple = (0, 0)
    t: tuple = (0, 0)

    def value(self, t):
        return (self.a * t + self.b) * t + self.c

    def packed(self):
        scale = max(abs(self.a), abs(self.b), abs(self.c)) or 1
        return tuple(map(float, (self.lo, self.hi, self.a / scale, self.b / scale, self.c / scale)))

    def entry(self, proposal):
        """Return the earliest intersected dyadic cell, identically for all backends."""
        if self.value(self.lo) <= 0:
            return floor(self.lo * _GRID), ceil(self.lo * _GRID)
        minimum = self.hi if not self.a else max(self.lo, min(self.hi, -self.b / (2 * self.a)))
        at_min = self.value(minimum)
        if at_min > 0:
            return None
        if at_min == 0:
            return floor(minimum * _GRID), ceil(minimum * _GRID)

        left, right = floor(self.lo * _GRID), ceil(minimum * _GRID)

        def reached(k):
            t = Fraction(k, _GRID)
            return t >= minimum or (t >= self.lo and self.value(t) <= 0)

        # Use the actual tyx approximation as a pivot; it may miss a tangency,
        # or hit on the rounded polynomial only, so it cannot certify exclusion.
        if proposal is not None and isfinite(proposal) and proposal >= 0:
            seed = round(proposal * _GRID)
            if left < seed < right:
                if reached(seed):
                    right = seed
                else:
                    left = seed
        while right - left > 1:
            mid = (left + right) // 2
            if reached(mid):
                right = mid
            else:
                left = mid
        t = Fraction(right, _GRID)
        if t <= minimum and self.value(t) == 0:
            return right, right
        return left, right


def _axis_features(u, v, w, velocity, radius):
    uu, uv, vv = u @ u, u @ v, v @ v
    uw, vw = u @ w, v @ w
    ud, vd = u @ velocity, v @ velocity
    s_ends = (0, 1) if uu else (_HALF,)
    t_ends = (0, 1) if vv else (_HALF,)
    candidates = [((s, 0), (t, 0)) for s in s_ends for t in t_ends]
    if vv:
        candidates.extend(((s, 0), ((vw + s * uv) / vv, vd / vv)) for s in s_ends)
    if uu:
        candidates.extend((((t * uv - uw) / uu, -ud / uu), (t, 0)) for t in t_ends)
    det = uu * vv - uv * uv
    if det:
        candidates.append(
            (
                ((uv * vw - vv * uw) / det, (uv * vd - vv * ud) / det),
                ((uu * vw - uv * uw) / det, (uu * vd - uv * ud) / det),
            )
        )
    for s, t in candidates:
        lo, hi = _ZERO, _ONE
        for initial, rate in (s, t):
            if rate:
                times = sorted((-initial / rate, (1 - initial) / rate))
                lo, hi = max(lo, times[0]), min(hi, times[1])
            elif not 0 <= initial <= 1:
                hi = -1
        if lo <= hi:
            start = w + s[0] * u - t[0] * v
            step = velocity + s[1] * u - t[1] * v
            yield _Feature(
                lo, hi, step @ step, 2 * (start @ step), start @ start - radius * radius, s, t
            )


class _AxisPair:
    def __init__(self, a, b, ma, mb):
        self.a, self.b = a, b
        self.p, self.q = _vector(ma.start[0]), _vector(mb.start[0])
        self.u, self.v = _vector(ma.start[-1]) - self.p, _vector(mb.start[-1]) - self.q
        self.dp, self.dq = _translation(ma), _translation(mb)
        self.radius = Fraction(a.radius) + Fraction(b.radius)
        self.features = tuple(
            _axis_features(self.u, self.v, self.p - self.q, self.dp - self.dq, self.radius)
        )

    def sample(self, time):
        active = [
            (f.value(time), (f.s[0] + f.s[1] * time, f.t[0] + f.t[1] * time))
            for f in self.features
            if f.lo <= time <= f.hi
        ]
        minimum = min(q for q, _ in active)
        params = sorted(st for q, st in active if q == minimum)
        s, t = params[0]
        first = self.p + time * self.dp + s * self.u
        second = self.q + time * self.dq + t * self.v
        direction, distance = _unit(second - first)
        # Rationalize the gap to keep the sign at a near tangent.
        gap = float(minimum / Fraction(distance + float(self.radius)))
        normal = pa = pb = None
        if direction is not None:
            normal = _xyz(direction)
            pa = _xyz(np.asarray(first, float) + self.a.radius * direction)
            pb = _xyz(np.asarray(second, float) - self.b.radius * direction)
        span = None
        if any(self.u) and any(self.v):
            span = tuple(tuple(map(float, st)) for st in (params[0], params[-1]))
        return CollisionWitness(
            gap,
            normal,
            pa,
            pb,
            _weights(self.a, s),
            _weights(self.b, t),
            "host-rational-axis",
            axis_parameter_span=span,
        ), minimum


class _HalfSpacePair:
    def __init__(self, a, b, ma, mb):
        self.plane_first = a.kind == "plane"
        self.plane, self.round = (a, b) if self.plane_first else (b, a)
        mp, mr = (ma, mb) if self.plane_first else (mb, ma)
        self.anchor, self.dp = _vector(mp.start[0]), _translation(mp)
        self.axis = tuple(_vector(p) for p in mr.start)
        self.dr = _translation(mr)
        self.normal = _vector(self.plane.normal)
        self.nn = self.normal @ self.normal
        self.unit, _ = _unit(self.normal)
        self.radius = Fraction(self.round.radius)
        slope = self.normal @ (self.dr - self.dp)
        features = []
        for p in self.axis:
            initial = self.normal @ (p - self.anchor)
            # s <= r|n| is the union of s <= 0 and s^2 <= r^2 (n·n).
            features.append(_Feature(_ZERO, _ONE, _ZERO, slope, initial))
            features.append(
                _Feature(
                    _ZERO,
                    _ONE,
                    slope * slope,
                    2 * initial * slope,
                    initial * initial - self.radius * self.radius * self.nn,
                )
            )
        self.features = tuple(features)

    def sample(self, time):
        anchor = self.anchor + time * self.dp
        points = [p + time * self.dr for p in self.axis]
        projection, index = min((self.normal @ (p - anchor), i) for i, p in enumerate(points))
        center = points[index]
        offset = center - anchor - (projection / self.nn) * self.normal
        height2 = projection * projection / self.nn
        height = float(np.hypot.reduce(np.asarray((projection / self.nn) * self.normal, float)))
        signed = height2 - self.radius * self.radius if projection >= 0 else -1
        gap = (
            float(signed / Fraction(height + float(self.radius)))
            if projection >= 0
            else -height - float(self.radius)
        )
        plane_point = _xyz(anchor + offset)
        round_point = _xyz(np.asarray(center, float) - float(self.radius) * self.unit)
        wp, wr = _weights(self.plane, 0), _weights(self.round, index)
        if self.plane_first:
            witness = CollisionWitness(
                gap,
                _xyz(self.unit),
                plane_point,
                round_point,
                wp,
                wr,
                "host-halfspace",
                first_offset=_xyz(offset),
            )
        else:
            witness = CollisionWitness(
                gap,
                _xyz(-self.unit),
                round_point,
                plane_point,
                wr,
                wp,
                "host-halfspace",
                second_offset=_xyz(offset),
            )
        return witness, signed


def round_query(a, b, ma, mb):
    if any(p.turn is not None and p.turn.angle != 0 for p in (ma, mb)):
        return None
    if a.kind in ("sphere", "capsule") and b.kind in ("sphere", "capsule"):
        return _AxisPair(a, b, ma, mb)
    if "plane" in (a.kind, b.kind) and (
        a.kind in ("sphere", "capsule") or b.kind in ("sphere", "capsule")
    ):
        return _HalfSpacePair(a, b, ma, mb)
    return None


def sweep_result(query, proposals):
    start, signed_start = query.sample(_ZERO)
    end, _ = query.sample(_ONE)
    if signed_start <= 0:
        return end, CollisionSweep(start.gap_m, end.gap_m, (0.0, 0.0), 0.0, start)
    entries = [
        bound
        for feature, seed in zip(query.features, proposals, strict=True)
        if (bound := feature.entry(seed)) is not None
    ]
    if not entries:
        return end, CollisionSweep(start.gap_m, end.gap_m, None, None, None)
    # The minimum of the lower/upper bounds encloses the earliest feature event.
    lo = min(x[0] for x in entries) / _GRID
    hi = min(x[1] for x in entries) / _GRID
    sample_time = (lo + hi) / 2
    witness, _ = query.sample(Fraction(sample_time))
    return end, CollisionSweep(start.gap_m, end.gap_m, (lo, hi), sample_time, witness)
