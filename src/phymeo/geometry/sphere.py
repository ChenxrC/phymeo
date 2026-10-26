"""Spherical collision geometry centered at each rigid body's centre of mass."""

from dataclasses import dataclass
from numbers import Real

import numpy as np

from ..core._arrays import float_array
from ..core.capabilities import ProblemSignature


@dataclass(frozen=True, eq=False)
class SphereShapes:
    """Owned positive radii [m]; this geometry does not prescribe body inertia."""

    radii: np.ndarray

    def __post_init__(self):
        radii = float_array(self.radii, "radii", ndim=1)
        if np.any(radii <= 0):
            raise ValueError("sphere radii must be positive")
        radii.flags.writeable = False
        object.__setattr__(self, "radii", radii)


@dataclass(frozen=True)
class SpherePlaneSignature(ProblemSignature):
    """Complete identity of the limited plastic sphere/plane contact problem."""

    semantics: str = "rigid-sphere-plane-e0-v1"
    plane: tuple[float, str] | None = (0.0, "sphere-y-up-frictionless-e0-v1")
    radii: tuple[float, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        if (
            self.semantics != "rigid-sphere-plane-e0-v1"
            or self.distance
            or self.plane is None
            or self.plane[1] != "sphere-y-up-frictionless-e0-v1"
        ):
            raise ValueError("unsupported sphere/plane contact identity")
        radii = []
        for radius in self.radii:
            if isinstance(radius, (bool, np.bool_)) or not isinstance(radius, Real):
                raise ValueError("sphere radii must be finite positive real scalars")
            try:
                radius = float(radius)
            except OverflowError as error:
                raise ValueError("sphere radius is not representable") from error
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError("sphere radii must be finite and positive")
            radii.append(radius)
        object.__setattr__(self, "radii", tuple(radii))
