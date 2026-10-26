"""Static particle parameters in SI units and a right-handed, y-up frame."""

from dataclasses import dataclass, field

import numpy as np

from ._arrays import float_array


@dataclass(frozen=True, eq=False)
class ParticleModel:
    """Particle inverse masses (zero fixes a particle) and world gravity."""

    inverse_mass: np.ndarray
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, -9.81, 0.0]))

    def __post_init__(self):
        inverse_mass = float_array(self.inverse_mass, "inverse_mass", ndim=1)
        if np.any(inverse_mass < 0):
            raise ValueError("inverse_mass must be non-negative")
        gravity = float_array(self.gravity, "gravity", ndim=1)
        if gravity.shape != (3,):
            raise ValueError("gravity must have shape (3,)")
        gravity = gravity.astype(inverse_mass.dtype, copy=True)
        if not np.all(np.isfinite(gravity)):
            raise ValueError("gravity is not representable in the model precision")
        inverse_mass.flags.writeable = False
        gravity.flags.writeable = False
        object.__setattr__(self, "inverse_mass", inverse_mass)
        object.__setattr__(self, "gravity", gravity)

    @property
    def particle_count(self) -> int:
        return self.inverse_mass.shape[0]
