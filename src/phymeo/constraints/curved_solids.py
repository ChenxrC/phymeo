"""Analytic solid dimensions and fixed orientation for public collision shapes."""

from dataclasses import dataclass
from math import hypot

from ..core.validation import finite_real


@dataclass(frozen=True)
class CurvedSolid:
    """A centered Z-axis cylinder/cone (radius, half-height) or ellipsoid (a, b, c).

    The cone apex lies at local +half-height. Lengths are in metres. The unit
    xyzw quaternion maps local vectors to world coordinates for a direct query.
    If CollisionMotion supplies a starting body orientation, it maps asset-local
    vectors into that body frame. Motion carries the world center separately.
    """

    kind: str
    dimensions_m: tuple[float, ...]
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def __post_init__(self):
        if self.kind not in ("cylinder", "cone", "ellipsoid"):
            raise ValueError("expected a cylinder, cone or ellipsoid")
        try:
            dims = tuple(finite_real(v, "curved dimension") for v in self.dimensions_m)
            q = tuple(finite_real(v, "curved orientation") for v in self.orientation_xyzw)
        except TypeError as exc:
            raise ValueError("dimensions and orientation must be finite vectors") from exc
        if len(dims) != (3 if self.kind == "ellipsoid" else 2) or min(dims) <= 0:
            raise ValueError("curved solid requires its positive principal dimensions")
        if len(q) != 4 or abs(hypot(*q) - 1.0) > 1e-12:
            raise ValueError("curved solid orientation must be a unit xyzw quaternion")
        object.__setattr__(self, "dimensions_m", dims)
        object.__setattr__(self, "orientation_xyzw", q)
