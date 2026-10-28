"""Owned infinite robot-local planes for automatic normal contact."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..core.scene import _name
from .friction import _norm
from .particle_plane import _values


@dataclass(frozen=True, eq=False)
class RobotPlaneCollider:
    key: str
    link_index: int
    local_point: np.ndarray
    local_normal: np.ndarray
    radius_m: float = 0.0
    friction: float = 0.0

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
        object.__setattr__(self, "link_index", int(self.link_index))
        for name in ("radius_m", "friction"):
            value = getattr(self, name)
            if type(value) is not float or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative builtin float")
        if self.friction != 0:
            raise ValueError("dynamic normal contact requires friction=0")
        for name in ("local_point", "local_normal"):
            value = _values(getattr(self, name), name, (3,))
            if name == "local_normal":
                length = _norm(value)
                if length == 0:
                    raise ValueError("plane normal must be nonzero")
                value /= length
                if not np.isfinite(value).all() or _norm(value) == 0:
                    raise ValueError("plane normal is not representable")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
