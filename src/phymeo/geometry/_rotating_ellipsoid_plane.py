"""Centered rigid ellipsoid rotation against a fixed oriented half-space."""

from dataclasses import replace
from fractions import Fraction as F
from math import ceil, floor, isfinite, pi, ulp

import numpy as np

from ..core.capabilities import SolverFailure
from ._curved_solids import CurvedEvent, _rotation, _static
from ._rotation_angle import _AngleChart, axis_length_bounds, sin_cos
from ._round_sweep import _GRID, _vector
from ._sweep_polynomial import Roots, add, dot, mul, sub, value
from ._triangle_pair_sweep import _cross3
from .collision_witness import CollisionSweep

_MAX_REFINEMENTS = 144
_MAX_TIME_WIDTH = F(1, 10**7)
_MAX_WITNESS_ERROR_M = F(1, 10**7)


def _contact_polynomial(normal, height, axis, initial, dims):
    """For positive height: >=0 iff the ellipsoid reaches the plane.

    The inverse quaternion (1-s, -s*axis) transports the fixed normal into
    the original frame. The strictly positive squared denominator clears it.
    """
    w, s = (F(1), F(-1)), (F(0), F(1))
    ww, ss, ws = mul(w, w), mul(s, s), mul(w, s)
    aa = mul((axis @ axis,), ss)
    denominator = add(ww, aa)
    along = axis @ normal
    cross = _cross3(axis, normal)
    world = [
        sub(add(mul((v,), sub(ww, aa)), mul((2 * u * along,), ss)), mul((2 * c,), ws))
        for v, u, c in zip(normal, axis, cross, strict=True)
    ]
    local = []
    for k, dimension in enumerate(dims):
        coordinate = ()
        for j in range(3):
            coordinate = add(coordinate, mul((initial[j, k],), world[j]))
        local.append(mul((dimension,), coordinate))
    return sub(dot(local, local), mul((height * height,), mul(denominator, denominator)))


def _extrema(poly, chart):
    """Find contacts on the actual angle prefix of the full rational half-turn."""
    zero, one = F(0), F(1)
    if not poly:
        return (0.0, 0.0), (1.0, 1.0)
    roots = Roots(poly)
    events = []
    # time 1 lies strictly between the last accepted chart root and the
    # first root past the stored terminal angle (float(pi) < mathematical pi).
    left, right = zero, one

    def isolate(lo, hi, count, depth):
        nonlocal left, right
        if not count:
            return
        if depth >= _MAX_REFINEMENTS:
            raise SolverFailure("ellipsoid_arc_root_isolation_limit")
        if count == 1 and value(poly, hi) == 0:
            lo = hi  # Exact chart root, including an isolated repeated root.
        # _AngleChart takes an index in units of 1/_GRID; exact rational
        # subindices permit extra refinement when a root nearly meets time 1.
        time_lo = chart.time(lo * _GRID, False)
        if time_lo > 1:
            right = min(right, lo)
            return
        if count == 1 and hi - lo <= F(1, _GRID):
            time_hi = chart.time(hi * _GRID, True)
            if time_hi <= 1 and time_hi - time_lo <= _MAX_TIME_WIDTH:
                bracket = (floor(time_lo * _GRID) / _GRID, ceil(time_hi * _GRID) / _GRID)
                if events and events[-1][1] >= bracket[0]:
                    raise SolverFailure("ellipsoid_arc_roots_not_separated")
                events.append(bracket)
                left = max(left, hi)
                return
            if hi == lo:
                raise SolverFailure("ellipsoid_arc_endpoint_order_unresolved")
        mid = (lo + hi) / 2
        lower_count = roots.count(lo, mid)
        isolate(lo, mid, lower_count, depth + 1)
        isolate(mid, hi, count - lower_count, depth + 1)

    isolate(zero, one, roots.count(zero, one), 0)
    # All chart locations between these two isolating bands have the same sign.
    # This avoids rebuilding an inexact endpoint and mistaking it for a root.
    if left >= right:
        raise SolverFailure("ellipsoid_arc_terminal_sign_unresolved")
    terminal_sign = value(poly, (left + right) / 2)
    if not terminal_sign:
        raise SolverFailure("ellipsoid_arc_terminal_sign_unresolved")
    first = (0.0, 0.0) if value(poly, zero) >= 0 else (events[0] if events else None)
    last = (1.0, 1.0) if terminal_sign > 0 else (events[-1] if events else first)
    return first, last


def _frame_at(initial, axis, length, phase, dims):
    if not phase:
        return initial
    sine, cosine, error = sin_cos(phase / 2)
    low, high = length
    v = axis * (sine / high)
    # Normalize the rational quaternion implicitly. This gives a genuinely
    # orthogonal frame, including for a non-dyadic sampled time and oblique axis.
    vv = v @ v
    norm2 = vv + cosine * cosine
    axis_l1 = sum(abs(x) for x in axis)
    quat_error = error * (1 + axis_l1 / low) + abs(sine) * axis_l1 * (high - low) / (low * high)
    # Normalizing a quaternion within delta of unit costs at most 2*delta.
    # The rotated unit direction error is at most 4*delta. For Q=R*diag(d)
    # the support point is -Q * normalize(Q.T@n) for a unit plane normal.
    extent = max(dims)
    point_error = 4 * quat_error * extent * (1 + 2 * extent / min(dims))
    if point_error > _MAX_WITNESS_ERROR_M:
        raise SolverFailure("ellipsoid_arc_witness_resolution")
    return np.column_stack(
        [
            ((cosine * cosine - vv) * col + 2 * v * (v @ col) + 2 * cosine * _cross3(v, col))
            / norm2
            for col in initial.T
        ]
    )


def rotating_ellipsoid_plane(curved, plane, mc, mp, curved_first):
    turn = mc.turn
    if (
        mp.moving
        or turn is None
        or not 0 < abs(turn.angle) <= pi
        or turn.pivot_start != turn.pivot_end
        or tuple(mc.start[0]) != turn.pivot_start
    ):
        return "unsupported", "ellipsoid_arc_motion_domain", None, None, None
    try:
        initial = _rotation(curved.curved, mc)
        dims = _vector(curved.curved.dimensions_m)
        normal = _vector(plane.normal)
        height = (_vector(mc.start[0]) - _vector(mp.start[0])) @ normal
        signed_angle = F(turn.angle)
        axis = _vector(turn.axis)
        length = axis_length_bounds(axis)
        if height <= 0:
            first, last = (0.0, 0.0), (1.0, 1.0)
        else:
            oriented = axis if signed_angle > 0 else -axis
            poly = _contact_polynomial(normal, height, oriented, initial, dims)
            first, last = _extrema(poly, _AngleChart(0, length, abs(signed_angle)))

        def sample(time):
            frame = _frame_at(initial, axis, length, signed_angle * F(time), dims)
            _, _, w, trace = _static(curved, plane, mc, mp, curved_first, frame=frame)
            # Keep roundoff in world-space reconstruction within the same metre
            # budget. Very large world offsets can otherwise hide the surface gap.
            coords = (*w.first_point, *w.second_point, *w.first_offset, *w.second_offset, w.gap_m)
            if any(not isfinite(x) or ulp(x) > float(_MAX_WITNESS_ERROR_M) for x in coords):
                raise SolverFailure("ellipsoid_arc_world_resolution")
            return replace(w, source="host-ellipsoid-angle-chart"), trace

        start, _ = sample(0.0)
        end, trace = sample(1.0)
        if first is None:
            if last is not None or end.gap_m < 0:
                raise SolverFailure("ellipsoid_arc_endpoint_sign_unresolved")
            sweep = CollisionSweep(start.gap_m, end.gap_m, None, None, None)
            return "separated", "ellipsoid_arc_separated", end, trace, sweep

        def event(bracket):
            time = (bracket[0] + bracket[1]) / 2
            w = start if time == 0 else end if time == 1 else sample(time)[0]
            return CurvedEvent(bracket, time, w)

        if (
            last is None
            or ((last == (1.0, 1.0)) and end.gap_m > 0)
            or (last != (1.0, 1.0) and end.gap_m < 0)
        ):
            raise SolverFailure("ellipsoid_arc_endpoint_sign_unresolved")
        entry = event(first)
        final = entry if first == last else event(last)
        trace = replace(trace, first_contact=entry, last_contact=final)
        sweep = CollisionSweep(
            start.gap_m, end.gap_m, entry.interval, entry.witness_time, entry.witness
        )
        return "hit", "ellipsoid_arc_contact", end, trace, sweep
    except (SolverFailure, OverflowError, FloatingPointError) as exc:
        return "unresolved", str(exc), None, None, None
