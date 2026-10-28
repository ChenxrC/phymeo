"""Host FP64 distance certificates for point/triangle and segment/segment hulls.

Weights supply a feasible witness, not a claim of closest-point optimality.
All geometry arithmetic uses exact rationals of the stored binary64 inputs.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from struct import pack, unpack

import numpy as np

from ..core._arrays import float_array
from ..core.capabilities import SolverFailure

_METHOD = "convex-kkt-distance-v1"
_MAX_FINITE_BITS = 0x7FEFFFFFFFFFFFFF


@dataclass(frozen=True)
class FeatureDistanceBounds:
    """Scalar interval in metres; arbitrary construction does not certify geometry.

    Trusted bounds must be recomputed with :func:`feature_distance_bounds`
    against the actual current geometry and feasible witness weights.
    """

    lower_m: float
    upper_m: float
    method: str = _METHOD

    def __post_init__(self):
        for name in ("lower_m", "upper_m"):
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be a Python float")
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.lower_m > self.upper_m:
            raise ValueError("lower_m must not exceed upper_m")
        if self.method != _METHOD:
            raise ValueError(f"method must be {_METHOD}")


def _fp64_array(value, name, *, ndim, columns=None):
    result = float_array(value, name, ndim=ndim, columns=columns)
    if result.dtype != np.float64:
        raise TypeError(f"{name} must use float64 storage")
    return result


def _witness(vertices, weights, name):
    exact = [tuple(Fraction(float(x)) for x in row) for row in vertices]
    weights = [Fraction(float(w)) for w in weights]
    if any(w < 0 for w in weights) or not (total := sum(weights)):
        raise ValueError(f"{name} must be non-negative with positive exact sum")
    point = tuple(sum(w * row[k] for w, row in zip(weights, exact)) / total for k in range(3))
    return exact, point


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _float_at(bits):
    return unpack(">d", pack(">Q", bits))[0]


def _sqrt_bracket(squared):
    """Adjacent directed binary64 roots, using at most 63 exact comparisons.

    Non-negative finite binary64 bit patterns have numerical order. Searching
    this finite integer range avoids rounded square roots, overflow and flush
    to zero, including for rational values below the least subnormal square.
    """
    maximum = Fraction(_float_at(_MAX_FINITE_BITS))
    if squared > maximum * maximum:
        raise SolverFailure("distance upper bound is not representable as finite FP64")
    low, high = 0, _MAX_FINITE_BITS + 1
    while high - low > 1:
        middle = (low + high) // 2
        value = Fraction(_float_at(middle))
        if value * value <= squared:
            low = middle
        else:
            high = middle
    lower = _float_at(low)
    if Fraction(lower) ** 2 == squared:
        return lower, lower
    return lower, _float_at(low + 1)


def feature_distance_bounds(
    first, second, *, first_weights, second_weights
) -> FeatureDistanceBounds:
    """Certify the static distance between supported closed convex features.

    Shapes are (1,3)/(3,3), the reversed pair, or (2,3)/(2,3).
    Geometry and witness arrays require finite FP64 storage. Weights must be
    non-negative with a positive exact sum; normalization is exact, even when
    their ordinary FP64 sum would overflow. Repeated/degenerate vertices are
    allowed. Loose witnesses may give a zero lower bound without collision.

    Bounds enclose sqrt(Q) and sqrt(S) with adjacent directed FP64 values,
    where S is witness separation squared and Q subtracts its convex KKT gap.
    An upper bound outside finite FP64 raises SolverFailure. This host-only
    certificate supplies no thickness, contact normal, time of impact or CCD.
    """
    first = _fp64_array(first, "first", ndim=2, columns=3)
    second = _fp64_array(second, "second", ndim=2, columns=3)
    if (len(first), len(second)) not in ((1, 3), (3, 1), (2, 2)):
        raise ValueError("only point/triangle and segment/segment combinations are supported")
    wa = _fp64_array(first_weights, "first_weights", ndim=1)
    wb = _fp64_array(second_weights, "second_weights", ndim=1)
    if wa.shape != (len(first),) or wb.shape != (len(second),):
        raise ValueError("weights must have one entry per feature vertex")
    a_vertices, a = _witness(first, wa, "first_weights")
    b_vertices, b = _witness(second, wb, "second_weights")
    d = tuple(x - y for x, y in zip(a, b))
    squared = _dot(d, d)
    gap_a = max(Fraction(0), *(-_dot(d, tuple(x - y for x, y in zip(v, a))) for v in a_vertices))
    gap_b = max(Fraction(0), *(_dot(d, tuple(x - y for x, y in zip(v, b))) for v in b_vertices))
    q = max(Fraction(0), squared - 2 * (gap_a + gap_b))
    # Check the witness upper range first, even if the lower certificate is zero.
    _, upper = _sqrt_bracket(squared)
    lower, _ = _sqrt_bracket(q)
    return FeatureDistanceBounds(lower, upper)
