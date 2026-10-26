"""Owned exchange loads and an atomic two-state impulse outcome."""

from dataclasses import dataclass

import numpy as np

from ._arrays import float_array
from .rigid import RigidBodyState
from .state import ParticleState


@dataclass(frozen=True, eq=False)
class ImpulseLedger:
    """Assigned world loads; fixed supports absorb them without acquiring velocity.

    Linear impulses are kg m/s; rigid angular impulses about each COM are kg m^2/s.
    """

    particle: np.ndarray
    rigid_linear: np.ndarray
    rigid_angular: np.ndarray

    def __post_init__(self):
        for name in ("particle", "rigid_linear", "rigid_angular"):
            value = float_array(getattr(self, name), name, ndim=2, columns=3).astype(np.float64)
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if self.rigid_linear.shape != self.rigid_angular.shape:
            raise ValueError("rigid linear and angular impulse counts must match")


@dataclass(frozen=True)
class ImpulseExchangeResult:
    """Outcome container; apply_impulses supplies owned snapshots of both states.

    Direct dataclass construction retains the state references supplied by its caller.
    """

    particle_state: ParticleState
    rigid_state: RigidBodyState
    accepted: bool
    ledger: ImpulseLedger | None = None
    reason: str | None = None
