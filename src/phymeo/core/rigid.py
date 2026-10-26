"""Owned rigid-body data with explicit centre-of-mass and rotation conventions."""

from dataclasses import dataclass, field

import numpy as np

from ._arrays import float_array
from .model import ParticleModel
from .state import ParticleState, StepInputs


@dataclass(frozen=True, eq=False)
class RigidBodyModel:
    """Mass and body-frame principal inertia [kg m^2] about the centre of mass.

    Zero inverse mass fixes both translation and rotation. Dynamic bodies require
    positive principal moments satisfying the physical inertia triangle inequalities.
    """

    inverse_mass: np.ndarray
    inertia_diagonal: np.ndarray
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, -9.81, 0.0]))

    def __post_init__(self):
        translation = ParticleModel(self.inverse_mass, self.gravity)
        inertia = float_array(self.inertia_diagonal, "inertia_diagonal", ndim=2, columns=3)
        if inertia.shape[0] != translation.particle_count:
            raise ValueError("inertia_diagonal must contain one row per body")
        if inertia.dtype != translation.inverse_mass.dtype:
            raise ValueError("mass and inertia must have the same dtype")
        if np.any(inertia < 0):
            raise ValueError("inertia must be non-negative")
        dynamic = translation.inverse_mass > 0
        if np.any(inertia[dynamic] <= 0):
            raise ValueError("dynamic principal inertia must be positive")
        moments = inertia[dynamic].astype(np.float64)
        # Normalize before sums, so finite large inertia values cannot overflow the check.
        scaled = moments / np.max(moments, axis=1, keepdims=True)
        allowance = 8 * np.finfo(inertia.dtype).eps
        if np.any(2 * np.max(scaled, axis=1) > np.sum(scaled, axis=1) + allowance):
            raise ValueError("principal inertia violates triangle inequalities")
        inertia.flags.writeable = False
        object.__setattr__(self, "inverse_mass", translation.inverse_mass)
        object.__setattr__(self, "gravity", translation.gravity)
        object.__setattr__(self, "inertia_diagonal", inertia)

    @property
    def body_count(self):
        return len(self.inverse_mass)


@dataclass(eq=False)
class RigidBodyState:
    """COM position, body-to-world xyzw orientation, world linear/angular velocity."""

    position: np.ndarray
    orientation: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray

    def __post_init__(self):
        translation = ParticleState(self.position, self.velocity)
        orientation = float_array(self.orientation, "orientation", ndim=2, columns=4)
        angular = float_array(self.angular_velocity, "angular_velocity", ndim=2, columns=3)
        if (
            orientation.shape[0] != translation.particle_count
            or angular.shape != translation.position.shape
        ):
            raise ValueError("all rigid state arrays must have the same body count")
        if (
            orientation.dtype != translation.position.dtype
            or angular.dtype != translation.position.dtype
        ):
            raise ValueError("all rigid state arrays must have the same dtype")
        norm = np.hypot.reduce(orientation.astype(np.float64), axis=1)
        if np.any(np.abs(norm - 1.0) > 1e-6):
            raise ValueError("orientation must contain unit quaternions in xyzw order")
        self.position, self.velocity = translation.position, translation.velocity
        self.orientation, self.angular_velocity = orientation, angular

    @property
    def body_count(self):
        return len(self.position)

    def copy(self) -> "RigidBodyState":
        return RigidBodyState(self.position, self.orientation, self.velocity, self.angular_velocity)


@dataclass(eq=False)
class RigidBodyInputs:
    """World external force [N] and torque [N m] about each body's centre of mass."""

    force: np.ndarray
    torque: np.ndarray

    def __post_init__(self):
        force = StepInputs(self.force).force
        torque = float_array(self.torque, "torque", ndim=2, columns=3)
        if torque.shape != force.shape or torque.dtype != force.dtype:
            raise ValueError("force and torque must have identical shape and dtype")
        self.force, self.torque = force, torque

    @classmethod
    def zeros(cls, count: int, *, dtype=np.float64) -> "RigidBodyInputs":
        force = StepInputs.zeros(count, dtype=dtype).force
        return cls(force, np.zeros_like(force))

    def copy(self) -> "RigidBodyInputs":
        return RigidBodyInputs(self.force, self.torque)
