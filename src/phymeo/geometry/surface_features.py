"""Static convex-feature distances; not a collider, normal selector or CCD test."""

from dataclasses import dataclass, field
from functools import wraps

import numpy as np

from ..core._arrays import float_array
from ..core.capabilities import SolverFailure


def _numerical(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
                return function(*args, **kwargs)
        except (FloatingPointError, OverflowError) as error:
            raise SolverFailure(f"surface geometry is not representable: {error}") from error

    return checked


def _input(value, name, shape):
    value = float_array(value, name, ndim=len(shape))
    if value.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must use float64 storage")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return value


def _norm(value):
    result = float(np.hypot.reduce(value))
    if not np.isfinite(result):
        raise SolverFailure("non-finite surface distance")
    return result


@dataclass(frozen=True, eq=False)
class ClosestFeaturePair:
    first: np.ndarray
    second: np.ndarray
    first_weights: np.ndarray
    second_weights: np.ndarray
    distance: float = field(init=False)

    @_numerical
    def __post_init__(self):
        for name in ("first", "second", "first_weights", "second_weights"):
            value = float_array(getattr(self, name), name, ndim=1)
            if value.dtype != np.dtype(np.float64):
                raise TypeError("feature results must use float64 storage")
            if name in ("first", "second"):
                if value.shape != (3,):
                    raise ValueError("world points must have three coordinates")
            elif np.any(value < 0) or abs(float(np.sum(value)) - 1) > 1e-12:
                raise ValueError("feature weights must be convex")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, "distance", _norm(self.first - self.second))


def _scaled(points):
    origin = points[0].copy()
    delta = points - origin
    scale = float(np.max(np.abs(delta)))
    if scale == 0:
        raise ValueError("zero extent geometry")
    result = delta / scale
    if not np.isfinite(result).all():
        raise SolverFailure("surface normalization is not representable")
    return origin, scale, result


def _segment_parameter(point, first, second):
    edge = second - first
    length = float(edge @ edge)
    if length == 0:
        raise ValueError("zero length segment")
    return float(np.clip(((point - first) @ edge) / length, 0, 1))


@_numerical
def closest_point_triangle(point, triangle):
    """Closest point on a nondegenerate closed triangle, including its boundary."""
    point = _input(point, "point", (3,))
    triangle = _input(triangle, "triangle", (3, 3))
    origin, scale, values = _scaled(np.vstack((triangle, point)))
    tri, p = values[:3], values[3]
    edge1, edge2 = tri[1] - tri[0], tri[2] - tri[0]
    cross = np.cross(edge1, edge2)
    area = _norm(cross)
    longest_squared = max(float(edge @ edge) for edge in (edge1, edge2, edge2 - edge1))
    if area <= 1e-12 * longest_squared:
        raise ValueError("degenerate triangle")

    # Orthogonal plane coordinates avoid cancellation in a Gram determinant.
    axis1 = edge1 / _norm(edge1)
    axis2 = np.cross(cross / area, axis1)
    t = float((p @ axis2) / (edge2 @ axis2))
    s = float((p @ axis1 - t * (edge2 @ axis1)) / _norm(edge1))
    face = np.array([1 - s - t, s, t])
    candidates = []
    if np.all(face >= 0):
        candidates.append((face, face @ tri))
    for first, second in ((0, 1), (1, 2), (2, 0)):
        parameter = _segment_parameter(p, tri[first], tri[second])
        weights = np.zeros(3)
        weights[first], weights[second] = 1 - parameter, parameter
        candidates.append((weights, weights @ tri))
    weights, closest = min(candidates, key=lambda item: _norm(p - item[1]))
    return ClosestFeaturePair(point, origin + scale * closest, np.ones(1), weights)


@_numerical
def closest_segment_segment(first, second):
    """Closest pair on two nonzero closed segments, with explicit conditioning limits."""
    first = _input(first, "first segment", (2, 3))
    second = _input(second, "second segment", (2, 3))
    origin, scale, values = _scaled(np.vstack((first, second)))
    a, b = values[:2], values[2:]
    u, v = a[1] - a[0], b[1] - b[0]
    length_u, length_v = _norm(u), _norm(v)
    if length_u == 0 or length_v == 0:
        raise ValueError("zero length segment")
    cross = np.cross(u, v)
    sine_scale = _norm(cross)
    candidates = []
    if sine_scale != 0:
        if sine_scale <= 1e-12 * length_u * length_v:
            raise SolverFailure("near-parallel segment conditioning is unsupported")
        # Cross-product form avoids subtracting nearly equal Gram products.
        denominator = sine_scale * sine_scale
        offset = b[0] - a[0]
        s = float(np.cross(offset, v) @ cross / denominator)
        t = float(np.cross(offset, u) @ cross / denominator)
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append((s, t))
    # For parallel lines a minimum is attained on this rectangle's boundary.
    for s in (0.0, 1.0):
        candidates.append((s, _segment_parameter(a[0] + s * u, b[0], b[1])))
    for t in (0.0, 1.0):
        candidates.append((_segment_parameter(b[0] + t * v, a[0], a[1]), t))
    s, t = min(candidates, key=lambda st: _norm(a[0] + st[0] * u - b[0] - st[1] * v))
    return ClosestFeaturePair(
        origin + scale * (a[0] + s * u),
        origin + scale * (b[0] + t * v),
        np.array([1 - s, s]),
        np.array([1 - t, t]),
    )
