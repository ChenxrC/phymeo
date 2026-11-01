"""Owned joint stops and body shapes; separate from inertial RobotModel data."""

from dataclasses import dataclass

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
