"""World screw composition and certified center-distance CCD for a pair of finite spheres."""

from dataclasses import dataclass
from fractions import Fraction as F
from math import isfinite, ulp

import numpy as np

from ..constraints.collision_shapes import CollisionMotion, RigidTurn
from ..core.capabilities import SolverFailure
from ._bounded_contact import first_bounded_contact
from ._prism_pair_exact import sqrt_bounds
from ._rotation_angle import axis_length_bounds, rotate_enclosures
from ._round_sweep import _translation, _unit, _vector, _xyz
from .collision_witness import CollisionSweep, CollisionWitness

_ZERO, _ONE = F(0), F(1)
_WORLD_TOL = F(1, 10**6)


@dataclass(frozen=True, eq=False, kw_only=True)
class TwoHingeMotion(CollisionMotion):
    """Two nested rotations expressed at the start pose plus optional outer pivot translation.

    The inner pivot and axis rotate with the outer body. Both angles are the signed,
    unwrapped changes on the normalized interval. start stores registered world vertices.
    """

    outer: RigidTurn
    inner: RigidTurn

    def __post_init__(self):
        super().__post_init__()
        if self.end is not None or self.turn is not None:
            raise ValueError("a two-hinge motion already supplies its whole trajectory")
        if not all(isinstance(turn, RigidTurn) and turn.angle for turn in (self.outer, self.inner)):
            raise ValueError("two active, finite hinge turns are required")
        if self.inner.pivot_start != self.inner.pivot_end:
            raise ValueError("the inner pivot must be stationary in the outer frame")

    @property
    def moving(self):
        return True

    def positions_at(self, time):
        """Read all registered world vertices at one normalized time."""
        t = F(time)
        if not _ZERO <= t <= _ONE:
            raise ValueError("two-hinge sample time must be in [0, 1]")
        return np.array([_xyz(_PointPath(self, p).sample(t)[0]) for p in self.start])


def two_hinge_bounds(path, shape):
    """Conservative original-vertex boxes, before broad-phase margin is applied."""
    pivot = _vector(path.outer.pivot_start)
    end = _vector(path.outer.pivot_end)
    elbow = _vector(path.inner.pivot_start)
    span = sum(abs(x) for x in elbow - pivot) + F(shape.radius)
    if shape.curved is not None:
        # Both half-height/radius and all ellipsoid semiaxes fit inside this length.
        span += sum(F(x) for x in shape.curved.dimensions_m)
    for point in path.start:
        reach = span + sum(abs(x) for x in _vector(point) - elbow)
        yield np.minimum(pivot, end) - reach, np.maximum(pivot, end) + reach


class _PointPath:
    """Apply the distal rotation first. Each true rotation preserves its input error norm."""

    def __init__(self, path, point):
        point = _vector(point)
        self.turns = []
        if isinstance(path, TwoHingeMotion):
            ordered = (path.inner, path.outer)
        elif path.turn is not None and path.turn.angle:
            ordered = (path.turn,)
        else:
            ordered = ()
        self.travel = (
            _vector(ordered[-1].pivot_end) - _vector(ordered[-1].pivot_start)
            if ordered
            else _translation(path)
        )
        speed = sqrt_bounds(self.travel @ self.travel, 80)[1]
        reach = _ZERO
        for turn in ordered:
            pivot = _vector(turn.pivot_start)
            arm = point - pivot
            reach += sqrt_bounds(arm @ arm, 80)[1]
            angle = F(turn.angle)
            axis = _vector(turn.axis)
            self.turns.append((arm, axis, axis_length_bounds(axis), angle))
            speed += abs(angle) * reach
            point = pivot
        self.origin = point
        self.speed = speed

    def sample(self, t):
        offset = _vector((0, 0, 0))
        error = _ZERO
        for arm, axis, length, angle in self.turns:
            ((offset, extra),) = rotate_enclosures((arm + offset,), axis, length, angle * t)
            error += extra
        return self.origin + t * self.travel + offset, error


def _query(a, b, ma, mb):
    left = _PointPath(ma, ma.start[0])
    right = _PointPath(mb, mb.start[0])
    if any(abs(turn[-1]) > 32 for p in (left, right) for turn in p.turns):
        raise SolverFailure("two_hinge_sphere_angle_capacity")
    radius = F(a.radius) + F(b.radius)
    speed = left.speed + right.speed
    cache = {}

    def centers(t):
        if t not in cache:
            if len(cache) >= 512:
                raise SolverFailure("two_hinge_sphere_sample_capacity")
            ca, ea = left.sample(t)
            cb, eb = right.sample(t)
            delta = cb - ca
            cache[t] = ca, cb, delta, delta @ delta, ea + eb
        return cache[t]

    def distance(t):
        return centers(t)[3:]

    def witness(t):
        ca, cb, delta, d2, error = centers(t)
        if error and max(_ZERO, radius - error) ** 2 <= d2 <= (radius + error) ** 2:
            raise SolverFailure("two_hinge_sphere_pose_relation_unresolved")
        lower, upper = sqrt_bounds(d2, 96)
        gap = float((d2 - radius * radius) / ((lower + upper) / 2 + radius))
        n = pa = pb = oa = ob = None
        # When the centers coincide, occupation is known but the separating normal is not.
        if lower > error and (2 * error * radius / lower + error <= _WORLD_TOL / 2):
            direction, _ = _unit(delta)
            n = _xyz(direction)
            oa, ob = _xyz(a.radius * direction), _xyz(-b.radius * direction)
            pa, pb = _xyz(ca + _vector(oa)), _xyz(cb + _vector(ob))
            if max(ulp(x) for x in (*pa, *pb, *oa, *ob)) * 8 > _WORLD_TOL:
                raise SolverFailure("two_hinge_sphere_world_resolution")
        if not isfinite(gap) or (d2 != radius * radius and not gap):
            raise SolverFailure("two_hinge_sphere_gap_resolution")
        return CollisionWitness(
            gap,
            n,
            pa,
            pb,
            ((a.vertex_ids[0], 1.0),),
            ((b.vertex_ids[0], 1.0),),
            "host-two-hinge-spheres",
            first_offset=oa,
            second_offset=ob,
        )

    first, last = witness(_ZERO), witness(_ONE)
    if centers(_ZERO)[3] <= radius * radius:
        bracket = (_ZERO, _ZERO)
    else:
        bracket = first_bounded_contact([(_ZERO, _ONE)], distance, speed, radius)
    if bracket is None:
        if last.gap_m <= 0:
            raise SolverFailure("two_hinge_sphere_terminal_inconsistent")
        return (
            "separated",
            "two_hinge_spheres_swept_clear",
            last,
            CollisionSweep(first.gap_m, last.gap_m, None, None, None),
        )
    lo, hi = bracket
    if speed * (hi - lo) > _WORLD_TOL:
        raise SolverFailure("two_hinge_sphere_first_witness_resolution")
    w = first if not hi else last if hi == _ONE else witness(hi)
    if w.gap_m > 0:
        raise SolverFailure("two_hinge_sphere_occupied_sample_unresolved")
    return (
        "hit",
        "two_hinge_spheres_first_contact",
        last,
        CollisionSweep(first.gap_m, last.gap_m, (float(lo), float(hi)), float(hi), w),
    )


def serial_sphere_contact(a, b, ma, mb):
    if a.kind != "sphere" or b.kind != "sphere":
        return "unsupported", "two_hinge_shape_pair", None, None
    try:
        return _query(a, b, ma, mb)
    except (SolverFailure, OverflowError, FloatingPointError) as exc:
        return "unresolved", str(exc), None, None
