"""Whole deforming-prism CCD using exact samples and rational curvature bounds."""

from dataclasses import dataclass
from fractions import Fraction as F

import numpy as np

from ..core.scene import fingerprint
from ..solvers._contact_replay_audit import sqrt_bounds
from ._prism_exact import cross, dot, outward_float, rational, sub
from .prism_patch_manifold import exact_geometry, triangle_prism_patch_manifold, validate_prismatic
from .robot_patch_sweep import _positive_integer


def area_polynomial(start, end):
    a = tuple(tuple(rational(v) for v in p) for p in start)
    dx = tuple(sub(tuple(rational(v) for v in p), q) for p, q in zip(end, a, strict=True))
    e1, e2 = sub(a[1], a[0]), sub(a[2], a[0])
    d1, d2 = sub(dx[1], dx[0]), sub(dx[2], dx[0])
    return (
        cross(e1, e2),
        tuple(x + y for x, y in zip(cross(e1, d2), cross(d1, e2), strict=True)),
        cross(d1, d2),
    )


def polynomial_range(c, start, end):
    def at(t):
        return c[0] + t * c[1] + t * t * c[2]

    values = [at(start), at(end)]
    if c[2]:
        t = -c[1] / (2 * c[2])
        if start <= t <= end:
            values.append(at(t))
    return min(values), max(values)


@dataclass(frozen=True)
class PrismSweepSegment:
    start: F
    end: F
    gap_lower_m: float
    exact_gap_lower: F
    direction: tuple


@dataclass(frozen=True)
class PrismPatchSweepResult:
    status: str
    input_fingerprint: str
    segments: tuple
    checked_intervals: int
    witness_time: F | None
    reason: str
    area_coefficients: tuple


def candidate_fingerprint(model, q0, q1, collider, x0, x1, rho, budget, max_intervals, max_depth):
    def token(values):
        a = np.asarray(values, dtype=object)
        return (a.shape, tuple((rational(v).numerator, rational(v).denominator) for v in a.flat))

    return fingerprint(
        (
            model,
            token(q0),
            token(q1),
            collider,
            token(x0),
            token(x1),
            token([rho, budget]),
            max_intervals,
            max_depth,
        )
    )


def triangle_prism_patch_sweep(
    model,
    q0,
    q1,
    collider,
    triangle_start,
    triangle_end,
    *,
    half_thickness_m,
    penetration_budget_m,
    max_intervals=4096,
    max_depth=32,
):
    validate_prismatic(model)
    max_intervals = _positive_integer(max_intervals, "max_intervals")
    max_depth = _positive_integer(max_depth, "max_depth")
    rho, budget = rational(half_thickness_m), rational(penetration_budget_m)
    if rho <= 0 or budget < 0:
        raise ValueError("positive thickness and nonnegative budget required")
    x0 = np.array([[rational(v) for v in p] for p in triangle_start], object)
    x1 = np.array([[rational(v) for v in p] for p in triangle_end], object)
    q0 = tuple(rational(v) for v in q0)
    q1 = tuple(rational(v) for v in q1)

    sig = candidate_fingerprint(
        model, q0, q1, collider, x0, x1, rho, budget, max_intervals, max_depth
    )
    coeff = area_polynomial(x0, x1)

    def area(t):
        return tuple(a + b * t + c * t * t for a, b, c in zip(*coeff, strict=True))

    def at(t):
        x = (1 - t) * x0 + t * x1
        q = tuple(a + (b - a) * t for a, b in zip(q0, q1, strict=True))
        return x, q

    def curvature(start, end):
        mid = (start + end) / 2
        direction = area(mid)
        norm2 = sum(v * v for v in direction)
        if not norm2:
            return None
        projection = tuple(sum(a * b for a, b in zip(c, direction, strict=True)) for c in coeff)
        low = polynomial_range(projection, start, end)[0]
        if low <= 0:
            return None
        alow = low / sqrt_bounds(norm2)[1]
        # A constant normal is proved algebraically, including strict sign over the interval.
        if all(not any(cross(c, direction)) for c in coeff):
            return F(0)
        c1 = max(
            sqrt_bounds(sum((b + 2 * c * t) ** 2 for b, c in zip(coeff[1], coeff[2], strict=True)))[
                1
            ]
            for t in (start, end)
        )
        c2 = sqrt_bounds(sum(4 * c * c for c in coeff[2]))[1]
        return c2 / alow + 3 * c1 * c1 / (alow * alow)

    cache = {}

    def geometry(t):
        if t not in cache:
            x, q = at(t)
            cache[t] = exact_geometry(model, q, collider, x, rho)
        return cache[t]

    def lower(t, d):
        _, _, av, bv, _ = geometry(t)
        ap = [dot(d, p).bounds()[0] for p in av]
        bp = [dot(d, p).bounds()[1] for p in bv]
        return min(ap) - max(bp)

    segments = []
    checked = 0
    pending = [(F(0), F(1), 0)]
    while pending:
        start, end, depth = pending.pop()
        checked += 1
        mid = (start + end) / 2
        if checked > max_intervals:
            return PrismPatchSweepResult(
                "unresolved",
                sig,
                tuple(segments),
                checked - 1,
                None,
                "interval budget exhausted",
                coeff,
            )
        try:
            for t in (start, mid, end):
                if not any(area(t)):
                    return PrismPatchSweepResult(
                        "unresolved", sig, tuple(segments), checked, t, "area degeneracy", coeff
                    )
                x, q = at(t)
                m = triangle_prism_patch_manifold(
                    model,
                    q,
                    collider,
                    x,
                    half_thickness_m=half_thickness_m,
                    geometric_budget_m=penetration_budget_m,
                )
                if m.exact_gap_bounds[1] < -budget:
                    return PrismPatchSweepResult(
                        "violation",
                        sig,
                        tuple(segments),
                        checked,
                        t,
                        "complete SAT penetration witness",
                        coeff,
                    )
            curve = curvature(start, end)
            directions = set()
            for t in (start, mid, end):
                for axis in geometry(t)[4]:
                    # Candidate axes may be approximated; the fixed rational direction proof is exact.
                    d = tuple(rational(float(v)) for v in axis.direction)
                    if any(d):
                        directions.add(d)
            found = None
            if curve is not None:
                for d in sorted(directions):
                    nlo, nhi = sqrt_bounds(sum(v * v for v in d))
                    low = (
                        min(lower(start, d), lower(end, d))
                        - rho * nhi * curve * (end - start) ** 2 / 8
                    )
                    if low + budget * nlo >= 0:
                        gb = low / nhi if low >= 0 else low / nlo
                        if found is None or gb > found[0]:
                            found = (gb, d)
            if found is not None:
                segments.append(
                    PrismSweepSegment(
                        start, end, outward_float((found[0], found[0]))[0], found[0], found[1]
                    )
                )
                continue
        except (ValueError, OverflowError, ZeroDivisionError) as error:
            return PrismPatchSweepResult(
                "unresolved", sig, tuple(segments), checked, mid, str(error), coeff
            )
        if depth >= max_depth:
            return PrismPatchSweepResult(
                "unresolved", sig, tuple(segments), checked, None, "depth budget exhausted", coeff
            )
        pending.extend(((mid, end, depth + 1), (start, mid, depth + 1)))
    return PrismPatchSweepResult(
        "certified", sig, tuple(segments), checked, None, "whole convex prism covered", coeff
    )
