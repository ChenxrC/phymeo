"""Named body-local sites, independent of a solver's index space."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BodySite:
    name: str
    body: str
    position: tuple = (0.0, 0.0, 0.0)
    rotation_xyzw: tuple = (0.0, 0.0, 0.0, 1.0)

    def __post_init__(self):
        if (
            not isinstance(self.name, str)
            or not self.name
            or not isinstance(self.body, str)
            or not self.body
        ):
            raise ValueError("site name and body must be nonempty public names")
        for field, width in (("position", 3), ("rotation_xyzw", 4)):
            value = np.asarray(getattr(self, field), dtype=np.float64)
            if (
                value.shape != (width,)
                or not np.isfinite(value).all()
                or np.any(np.abs(value) > np.finfo(np.float32).max)
            ):
                raise ValueError(
                    f"site {field} requires finite float32-representable shape ({width},)"
                )
            object.__setattr__(self, field, tuple(float(v) for v in value))
        if abs(np.linalg.norm(self.rotation_xyzw) - 1.0) > 1e-6:
            raise ValueError("site rotation must be a unit XYZW quaternion")
