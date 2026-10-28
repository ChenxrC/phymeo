"""Host exact certificates for declared affine convex-feature trajectories.

FP64 endpoints denote exact stored real values. Closest-point computations
only propose convex weights; all proof and witness arithmetic is rational.
This does not certify curved FK motion, return TOI, or resolve contact.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from numbers import Integral

import numpy as np

from ..core._arrays import float_array
from ..core.capabilities import SolverFailure
from .surface_features import closest_point_triangle, closest_segment_segment

_METHOD = "affine-convex-separation-v1"


@dataclass(frozen=True)
class FeatureSweepResult:
    """Immutable partition; manual construction is not a geometric certificate."""

    status: str
    checked_intervals: int
    certified_intervals: tuple
    unresolved_intervals: tuple
    witness_time: float | None = None
    method: str = _METHOD

    def __post_init__(self):
        if type(self.status) is not str or type(self.method) is not str:
            raise TypeError("status and method must be Python str")
        if self.status not in ("certified", "violation", "unresolved") or self.method != _METHOD:
            raise ValueError("unsupported sweep status or method")
        if type(self.checked_intervals) is not int:
            raise TypeError("checked_intervals must be a Python int")
        if self.checked_intervals < 1:
            raise ValueError("checked_intervals must be positive")
        for group in (self.certified_intervals, self.unresolved_intervals):
            if type(group) is not tuple or any(type(pair) is not tuple for pair in group):
                raise TypeError("intervals must be tuples of tuples")
            previous = -1.0
            for pair in group:
                if len(pair) != 2 or any(type(t) is not float for t in pair):
                    raise TypeError("interval endpoints must be Python floats")
                lower, upper = pair
                if not (0 <= lower < upper <= 1) or lower < previous:
                    raise ValueError("intervals must be ordered, disjoint and within [0,1]")
                previous = upper
        intervals = sorted(self.certified_intervals + self.unresolved_intervals)
        if not intervals or intervals[0][0] != 0 or intervals[-1][1] != 1:
            raise ValueError("partition must cover [0,1]")
        if any(a[1] != b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError("partition must have no gaps or overlaps")
        if (self.status == "certified") != (not self.unresolved_intervals):
            raise ValueError("only certified status has no unresolved intervals")
        if self.status == "violation":
            if type(self.witness_time) is not float:
                raise TypeError("violation requires a Python float witness_time")
            if not any(
                lower <= self.witness_time <= upper for lower, upper in self.unresolved_intervals
            ):
                raise ValueError("witness must belong to an unresolved interval")
        elif self.witness_time is not None:
            raise ValueError("only violation has a witness")


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def _geometry(value, name):
    snapshot = float_array(value, name, ndim=2, columns=3)
    if snapshot.dtype != np.float64:
        raise TypeError(f"{name} must use float64 storage")
    return snapshot


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _at(start, end, t):
    return tuple(
        tuple(x + t * (y - x) for x, y in zip(a, b, strict=True))
        for a, b in zip(start, end, strict=True)
    )


def _closest_weights(a, b):
    rounded_a, rounded_b = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    try:
        if len(a) == 1:
            candidate = closest_point_triangle(rounded_a[0], rounded_b)
            return candidate.first_weights, candidate.second_weights
        if len(b) == 1:
            candidate = closest_point_triangle(rounded_b[0], rounded_a)
            return candidate.second_weights, candidate.first_weights
        candidate = closest_segment_segment(rounded_a, rounded_b)
        return candidate.first_weights, candidate.second_weights
    except SolverFailure:
        return None
    except ValueError as error:
        if str(error) not in ("degenerate triangle", "zero extent geometry", "zero length segment"):
            raise
        return None


def _weighted(vertices, weights):
    weights = tuple(Fraction(float(w)) for w in weights)
    total = sum(weights)
    if any(w < 0 for w in weights) or total <= 0:
        raise ValueError("closest candidate must have feasible convex weights")
    return tuple(
        sum(w * p[k] for w, p in zip(weights, vertices, strict=True)) / total for k in range(3)
    )


def _interval(a0, a1, b0, b1, lower, upper, clearance_squared):
    endpoints = ((_at(a0, a1, lower), _at(b0, b1, lower)), (_at(a0, a1, upper), _at(b0, b1, upper)))

    def separates(d):
        norm = _dot(d, d)
        if norm == 0:
            return False
        # Each synchronous relative projection is affine in time, so its
        # endpoint minimum bounds every convex pair throughout this interval.
        # Cauchy then proves distance >= minimum / |d| without a rounded root.
        minimum = min(_dot(d, _sub(a, b)) for aa, bb in endpoints for a in aa for b in bb)
        return minimum >= 0 and minimum * minimum >= clearance_squared * norm

    for t in (lower, (lower + upper) / 2, upper):
        aa, bb = _at(a0, a1, t), _at(b0, b1, t)
        directions = [_sub(a, b) for a in aa for b in bb]
        weights = _closest_weights(aa, bb)
        if weights is not None:
            directions.append(_sub(_weighted(aa, weights[0]), _weighted(bb, weights[1])))
        for d in directions:
            if _dot(d, d) < clearance_squared:
                return "violation", float(t)
            if separates(d):
                return "certified", None
    for k in range(3):
        for sign in (-1, 1):
            if separates(tuple(Fraction(sign if j == k else 0) for j in range(3))):
                return "certified", None
    return "unresolved", None


def linear_feature_sweep(
    first_start,
    first_end,
    second_start,
    second_end,
    *,
    clearance_m,
    max_intervals=128,
    max_depth=20,
) -> FeatureSweepResult:
    """Bounded host proof of positive clearance over normalized affine time.

    Supported vertex counts are (1,3), (3,1), and (2,2), with finite FP64
    (N,3) arrays. A violation is an exact feasible witness, never a TOI.
    An unresolved result must not be accepted as safe by an integration layer.

    clearance_m is a positive built-in float in metres, including subnormals.
    max_intervals counts popped intervals (positive non-bool Integral);
    max_depth is a non-bool Integral in [0,52]. Subdivision is left-first DFS.
    Both interval tuples partition [0,1], including unvisited pending leaves
    after an early violation or exhausted budget. Equality is certified when
    a rational separating direction proves it. Degenerate features are valid
    and may be unresolved; unknown errors propagate. Inputs are owned copies.
    """
    arrays = tuple(
        _geometry(v, name)
        for v, name in zip(
            (first_start, first_end, second_start, second_end),
            ("first_start", "first_end", "second_start", "second_end"),
            strict=True,
        )
    )
    if arrays[0].shape != arrays[1].shape or arrays[2].shape != arrays[3].shape:
        raise ValueError("same feature endpoints must have matching shapes")
    if (len(arrays[0]), len(arrays[2])) not in ((1, 3), (3, 1), (2, 2)):
        raise ValueError("only point/triangle and segment/segment combinations are supported")
    if type(clearance_m) is not float:
        raise TypeError("clearance_m must be a Python float")
    if not isfinite(clearance_m) or clearance_m <= 0:
        raise ValueError("clearance_m must be finite and positive")
    for value, name, minimum, maximum in (
        (max_intervals, "max_intervals", 1, None),
        (max_depth, "max_depth", 0, 52),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer, not bool")
        if value < minimum or (maximum is not None and value > maximum):
            raise ValueError(f"{name} is outside its supported range")
    exact = tuple(tuple(tuple(Fraction(float(x)) for x in row) for row in v) for v in arrays)
    clearance_squared = Fraction(clearance_m) ** 2
    pending = [(Fraction(0), Fraction(1), 0)]
    certified, unresolved = [], []
    checked, witness = 0, None
    while pending and checked < max_intervals:
        lower, upper, depth = pending.pop()
        checked += 1
        status, time = _interval(*exact, lower, upper, clearance_squared)
        if status == "certified":
            certified.append((float(lower), float(upper)))
        elif status == "violation":
            witness = time
            unresolved.append((float(lower), float(upper)))
            break
        elif depth < max_depth and checked < max_intervals:
            middle = (lower + upper) / 2
            pending.extend(((middle, upper, depth + 1), (lower, middle, depth + 1)))
        else:
            unresolved.append((float(lower), float(upper)))
    unresolved.extend((float(lower), float(upper)) for lower, upper, _ in pending)
    status = "violation" if witness is not None else "unresolved" if unresolved else "certified"
    return FeatureSweepResult(
        status, checked, tuple(sorted(certified)), tuple(sorted(unresolved)), witness
    )
