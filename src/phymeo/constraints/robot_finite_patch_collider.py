"""Owned, exactly validated, canonical finite convex robot-local patches."""

from dataclasses import dataclass, field
from fractions import Fraction as F
from numbers import Integral

import numpy as np

from ..core._arrays import float_array
from ..core.scene import _name


def _dot(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), F(0))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


@dataclass(frozen=True, eq=False)
class RobotFinitePatchCollider:
    key: str
    link_index: int
    local_vertices: np.ndarray
    radius_m: float = field(kw_only=True)

    def __post_init__(self):
        if type(self.key) is not str:
            raise TypeError("collider key must be a builtin str")
        _name(self.key, "collider key")
        if isinstance(self.link_index, (bool, np.bool_)) or not isinstance(
            self.link_index, Integral
        ):
            raise TypeError("link_index must be a nonnegative integer")
        if self.link_index < 0:
            raise ValueError("link_index must be nonnegative")
        if type(self.radius_m) is not float or not np.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("radius_m must be a finite positive builtin float")
        v = float_array(self.local_vertices, "local_vertices", ndim=2, columns=3)
        if v.dtype != np.float64 or len(v) < 3:
            raise ValueError("local_vertices require at least three FP64 vertices")
        points = [tuple(F(float(x)) for x in row) for row in v]
        if len(set(points)) != len(points):
            raise ValueError("patch has duplicate vertices")
        normal = _cross(_sub(points[1], points[0]), _sub(points[2], points[0]))
        if not any(normal) or any(_dot(normal, _sub(p, points[0])) for p in points):
            raise ValueError("patch must be nondegenerate and exactly coplanar")
        for i, a in enumerate(points):
            edge = _sub(points[(i + 1) % len(points)], a)
            if any(
                _dot(_cross(edge, _sub(p, a)), normal) <= 0
                for j, p in enumerate(points)
                if j not in (i, (i + 1) % len(points))
            ):
                raise ValueError("patch boundary must be strictly convex and nonintersecting")
        if next(x for x in normal if x) < 0:
            points.reverse()
        start = min(range(len(points)), key=points.__getitem__)
        points = points[start:] + points[:start]
        owned = np.array(points, dtype=np.float64)
        owned.flags.writeable = False
        object.__setattr__(self, "local_vertices", owned)
        object.__setattr__(self, "link_index", int(self.link_index))
