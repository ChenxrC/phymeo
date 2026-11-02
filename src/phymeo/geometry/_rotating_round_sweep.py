"""A rigidly rotating capsule against a sphere with an affine center path.

Two rational half-turn charts cover an entire revolution. The cap and shaft
conditions use the same bounded polynomial feature queries as affine surfaces.
"""

from dataclasses import dataclass
from fractions import Fraction as F
from math import ceil, cos, floor, isqrt, sin

import numpy as np

from ..core.capabilities import SolverFailure
from ._rotation_angle import _PI_LO, _AngleChart, sin_cos
from ._round_sweep import _GRID, _translation, _unit, _vector, _weights, _xyz, sweep_result
from ._sphere_triangle_sweep import pack_surface_conditions
from ._sweep_polynomial import add, dot, first_entry, mul, sub
from .collision_witness import CollisionSweep, CollisionWitness


@dataclass(frozen=True)
class _ArcFeature:
    chart: _AngleChart
    conditions: tuple
    arc_chart = True

    def packed(self):
        return pack_surface_conditions(self.conditions)

    def entry(self, proposal):
        bounds = first_entry(self.conditions, proposal)
        if bounds is None:
            return None
        lo = self.chart.time(bounds[0], False)
        hi = self.chart.time(bounds[1], True)
        if lo > 1:
            return None
        if hi > 1 or hi - lo > F(1, 100000):
            raise SolverFailure("rotation_time_resolution")
        return floor(lo * _GRID), ceil(hi * _GRID)


def _chart_features(center, edge, segment, radius, axis, axis2, chart):
    # Inverse rotation of the ball by the quaternion (1-s, -axis*s).
    one = (F(1), F(-1))
    s = (F(0), F(1))
    ww, ss = mul(one, one), mul(s, s)
    aa = mul((axis2,), ss)
    den = add(ww, aa)
    cross = np.cross(axis, center)
    along = axis @ center
    relative = tuple(
        sub(
            add(
                add(mul((c,), sub(ww, aa)), mul((2 * a * along,), ss)), mul((-2 * x,), mul(one, s))
            ),
            mul((p,), den),
        )
        for c, a, x, p in zip(center, axis, cross, segment, strict=True)
    )
    radius2 = mul((radius * radius,), mul(den, den))
    clearance = sub(radius2, dot(relative, relative))
    yield _ArcFeature(chart, (clearance,))
    edge2 = edge @ edge
    if edge2:
        other = tuple(sub(p, mul((e,), den)) for p, e in zip(relative, edge, strict=True))
        yield _ArcFeature(chart, (sub(radius2, dot(other, other)),))
        along = dot(relative, tuple((e,) for e in edge))
        yield _ArcFeature(
            chart,
            (
                add(mul((edge2,), clearance), mul(along, along)),
                along,
                sub(mul((edge2,), den), along),
            ),
        )


class RotatingCapsuleQuery:
    def __init__(self, cap, ball, mc, mb, cap_first):
        turn = mc.turn
        self.cap, self.ball, self.cap_first = cap, ball, cap_first
        self.pivot = _vector(turn.pivot_start)
        self.travel = _translation(mc)
        self.drift = _translation(mb) - self.travel
        self.center = _vector(mb.start[0]) - self.pivot
        self.segment = _vector(mc.start[0]) - self.pivot
        self.edge = _vector(mc.start[1]) - _vector(mc.start[0])
        self.axis = _vector(turn.axis)
        self.angle = turn.angle
        self.radius = F(cap.radius) + F(ball.radius)
        axis2 = self.axis @ self.axis
        n = isqrt((axis2.numerator << 192) // axis2.denominator)
        unit = 1 << 96
        length = (
            F(n, unit),
            F(n if n * n * axis2.denominator == axis2.numerator * unit * unit else n + 1, unit),
        )
        axis = self.axis if self.angle > 0 else -self.axis
        self.axis_length = length
        # This frozen ball encloses the actual relative center for every time.
        ball_center = self.center + self.drift / 2
        outer_radius = self.radius + sum(abs(x) for x in self.drift) / 2
        features = []
        for half in range(2):
            if half and F(abs(self.angle)) < _PI_LO:
                break
            chart = _AngleChart(half, length, F(abs(self.angle)))
            features.extend(
                _chart_features(
                    ball_center, self.edge, self.segment, outer_radius, axis, axis2, chart
                )
            )
            ball_center = 2 * axis * (axis @ ball_center) / axis2 - ball_center
        self.features = tuple(features)

    def sample(self, time):
        # The witness uses the original pose, finite segment and material IDs.
        # Its distance classification at the initial pose uses exact stored data.
        def rotated(v):
            if time == 0:
                return v
            axis = np.asarray(self.axis, float)
            axis /= np.hypot.reduce(axis)
            phase = float(time) * self.angle
            x = np.asarray(v, float)
            y = (
                cos(phase) * x
                + sin(phase) * np.cross(axis, x)
                + (1 - cos(phase)) * axis * (axis @ x)
            )
            return _vector(y)

        center = self.center + time * self.drift
        origin, edge = rotated(self.segment), rotated(self.edge)
        den = edge @ edge
        u = max(0, min(1, (center - origin) @ edge / den)) if den else F(1, 2)
        closest = origin + u * edge
        delta = center - closest
        direction, distance = _unit(delta)
        signed = delta @ delta - self.radius * self.radius
        gap = float(signed / F(distance + float(self.radius)))
        pa = pb = normal = None
        if direction is not None:
            base = self.pivot + time * self.travel
            pa = _xyz(base + closest + _vector(self.cap.radius * direction))
            pb = _xyz(base + center - _vector(self.ball.radius * direction))
            normal = _xyz(direction if self.cap_first else -direction)
        wa, wb = _weights(self.cap, u), _weights(self.ball, 0)
        if not self.cap_first:
            pa, pb, wa, wb = pb, pa, wb, wa
        return CollisionWitness(gap, normal, pa, pb, wa, wb, "host-rational-rotation-chart"), signed

    def _midpoint_distance(self, t):
        # Rotate the original affine ball center into the stationary capsule frame.
        center = self.center + t * self.drift
        sine, cosine, error = sin_cos(-t * F(self.angle))
        axis = self.axis
        parallel = axis * (axis @ center) / (axis @ axis)
        perp = center - parallel
        cross = np.cross(axis, center)
        low, high = self.axis_length
        point = parallel + cosine * perp + (sine / high) * cross
        point_error = error * sum(abs(x) for x in perp) + (
            error / low + abs(sine) * (1 / low - 1 / high)
        ) * sum(abs(x) for x in cross)
        delta = point - self.segment
        den = self.edge @ self.edge
        if den:
            delta -= max(0, min(1, (self.edge @ delta) / den)) * self.edge
        return delta @ delta, point_error

    def sweep(self, proposals):
        if not any(self.drift):
            return sweep_result(self, proposals)
        start, initial = self.sample(F(0))
        end, _ = self.sample(F(1))

        def result(bracket):
            if bracket is None:
                return end, CollisionSweep(start.gap_m, end.gap_m, None, None, None)
            lo, hi = map(float, bracket)
            at = (lo + hi) / 2
            witness, _ = self.sample(F(at))
            return end, CollisionSweep(start.gap_m, end.gap_m, (lo, hi), at, witness)

        if initial <= 0:
            return result((0, 0))
        # An inflated co-moving sphere can exclude an entire path or its prefix.
        # A hit of that sphere alone never certifies actual contact.
        earliest = 1
        found = False
        for f, seed in zip(self.features, proposals, strict=True):
            try:
                bound = f.entry(seed)
            except SolverFailure:
                bound = (0, _GRID)
            if bound is not None:
                found = True
                earliest = min(earliest, F(bound[0], _GRID))
        if not found:
            return result(None)
        speed = sum(abs(x) for x in self.drift) + abs(F(self.angle)) * max(
            sum(abs(x) for x in q) for q in (self.center, self.center + self.drift)
        )
        # DFS in time order. All cells before the earliest pending leaf have a
        # separating lower bound. A positive result needs an occupied sample.
        stack = [(earliest, F(1))]
        pending = hit = None
        for _ in range(4096):
            if not stack:
                if pending is not None:
                    raise SolverFailure("rotation_contact_unresolved")
                return result(None)
            lo, hi = stack.pop()
            lower = lo if pending is None else pending
            if hit is not None and hit - lower <= F(1, 1 << 27):
                return result((lower, hit))
            if pending is not None and lo - pending > F(1, 1 << 27):
                raise SolverFailure("rotation_contact_unresolved")
            mid = (lo + hi) / 2
            d2, error = self._midpoint_distance(mid)
            if d2 > (self.radius + error + speed * (hi - lo) / 2) ** 2:
                continue
            if self.radius >= error and d2 <= (self.radius - error) ** 2:
                hit = mid if hit is None else min(hit, mid)
                if hit - lower <= F(1, 1 << 27):
                    return result((lower, hit))
                stack.append((lo, mid))
            elif hi - lo <= F(1, 1 << 36):
                if pending is None:
                    pending = lo
            else:
                stack.extend(((mid, hi), (lo, mid)))
        raise SolverFailure("rotation_cell_capacity")


def rotating_round_query(a, b, ma, mb):
    if {a.kind, b.kind} != {"capsule", "sphere"}:
        return None
    cap_first = a.kind == "capsule"
    cap, ball, mc, ms = (a, b, ma, mb) if cap_first else (b, a, mb, ma)
    if mc.turn is None or not mc.turn.angle or (ms.turn is not None and ms.turn.angle):
        return None
    return RotatingCapsuleQuery(cap, ball, mc, ms, cap_first)
