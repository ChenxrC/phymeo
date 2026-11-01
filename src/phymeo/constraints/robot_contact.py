"""Owned joint stops and body shapes; separate from inertial RobotModel data."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..core.robot import RobotState, robot_array
from ..core.validation import finite_real


def _owned(value, name, ndim, shape=None):
    array = robot_array(value, name, ndim=ndim)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class JointLimits:
    """Finite scalar bounds, spring effort/coordinate and damping effort/speed.

    Coordinates are radians for revolute joints and metres for prismatic joints.
    The physical stop is compliant. evaluate reports its force and penetration;
    it never clips a configuration or velocity.
    """

    lower: np.ndarray
    upper: np.ndarray
    stiffness: float
    damping: float = 0.0

    def __post_init__(self):
        lower = _owned(self.lower, "lower", 1)
        upper = _owned(self.upper, "upper", 1, lower.shape)
        if not len(lower) or np.any(lower >= upper):
            raise ValueError("joint lower bounds must be below upper bounds")
        for name, value in (("lower", lower), ("upper", upper)):
            object.__setattr__(self, name, value)
        for name in ("stiffness", "damping"):
            value = finite_real(getattr(self, name), name)
            if value < 0 or (name == "stiffness" and value == 0):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)

    def evaluate(self, state):
        if not isinstance(state, RobotState) or state.q.shape != self.lower.shape:
            raise ValueError("joint state does not match limits")
        if not np.isfinite(state.q).all() or not np.isfinite(state.v).all():
            raise ValueError("nonfinite joint state")
        lo = np.maximum(self.lower - state.q, 0.0)
        hi = np.maximum(state.q - self.upper, 0.0)
        effort = self.stiffness * (lo - hi)
        effort += self.damping * (
            (lo > 0) * np.maximum(-state.v, 0.0) - (hi > 0) * np.maximum(state.v, 0.0)
        )
        return dict(
            effort=effort,
            violation=lo + hi,
            potential_j=float(self.stiffness * (lo @ lo + hi @ hi) / 2),
        )


@dataclass(frozen=True, eq=False)
class LinkCapsule:
    """A local segment swept by a ball, attached to one massive link."""

    key: str
    domain: str
    link_index: int
    start: np.ndarray
    end: np.ndarray
    radius_m: float

    def __post_init__(self):
        if not all(isinstance(v, str) and v for v in (self.key, self.domain)):
            raise ValueError("capsule needs a key and robot domain")
        if (
            isinstance(self.link_index, (bool, np.bool_))
            or not isinstance(self.link_index, Integral)
            or self.link_index < 0
        ):
            raise ValueError("invalid capsule link index")
        for name in ("start", "end"):
            object.__setattr__(self, name, _owned(getattr(self, name), name, 1, (3,)))
        radius = finite_real(self.radius_m, "radius_m")
        if radius <= 0:
            raise ValueError("capsule radius must be positive")
        object.__setattr__(self, "radius_m", radius)


@dataclass(frozen=True, eq=False)
class CollisionPlane:
    """Stationary infinite half-space normal.dot(x) >= offset_m."""

    key: str
    normal: np.ndarray
    offset_m: float

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("plane needs a key")
        normal = _owned(self.normal, "normal", 1, (3,))
        if abs(np.linalg.norm(normal) - 1.0) > 1e-12:
            raise ValueError("plane normal must be unit length")
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "offset_m", finite_real(self.offset_m, "offset_m"))


@dataclass(frozen=True)
class RobotContactMaterial:
    """Normal N/m and Ns/m; dimensionless friction; linear slip regularization."""

    stiffness: float = 2000.0
    damping: float = 8.0
    friction: float = 0.3
    slip_speed_m_s: float = 0.01

    def __post_init__(self):
        for name in ("stiffness", "damping", "friction", "slip_speed_m_s"):
            value = finite_real(getattr(self, name), name)
            if value < 0 or (name in ("stiffness", "slip_speed_m_s") and value == 0):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)
