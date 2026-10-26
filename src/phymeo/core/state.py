"""Owned, mutable simulation state; solvers must leave input states unchanged."""

from dataclasses import dataclass

import numpy as np

from ._arrays import float_array


@dataclass(eq=False)
class ParticleState:
    """Position [m] and velocity [m/s], both stored as contiguous (n, 3)."""

    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self):
        position = float_array(self.position, "position", ndim=2, columns=3)
        velocity = float_array(self.velocity, "velocity", ndim=2, columns=3)
        if position.shape != velocity.shape:
            raise ValueError("position and velocity must have the same shape")
        if position.dtype != velocity.dtype:
            raise ValueError("position and velocity must have the same dtype")
        self.position = position
        self.velocity = velocity

    @property
    def particle_count(self) -> int:
        return self.position.shape[0]

    def copy(self) -> "ParticleState":
        return ParticleState(self.position, self.velocity)


@dataclass(eq=False)
class StepInputs:
    """External force [N], excluding gravity, on each particle."""

    force: np.ndarray

    def __post_init__(self):
        self.force = float_array(self.force, "force", ndim=2, columns=3)

    @classmethod
    def zeros(cls, count: int, *, dtype=np.float64) -> "StepInputs":
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
            raise ValueError("count must be a non-negative integer")
        if count < 0:
            raise ValueError("count must be a non-negative integer")
        return cls(np.zeros((count, 3), dtype=dtype))
