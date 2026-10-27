"""Velocity-first translation, independent of rotation and constraint algorithms."""

import numpy as np

from ..core.model import ParticleModel
from ..core.rigid import RigidBodyModel, RigidBodyState
from ..core.state import ParticleState


def predict(
    model: ParticleModel | RigidBodyModel,
    candidate: ParticleState | RigidBodyState,
    force: np.ndarray,
    h: np.floating,
) -> bool:
    """Update a candidate; reject non-finite values or lossy intermediate underflow."""
    dynamic = model.inverse_mass > 0
    # Do not evaluate forces on fixed particles: even unused arithmetic may overflow.
    try:
        with np.errstate(over="ignore", invalid="ignore", under="raise"):
            acceleration = model.gravity + model.inverse_mass[dynamic, None] * force[dynamic]
            candidate.velocity[dynamic] += h * acceleration
            candidate.position[dynamic] += h * candidate.velocity[dynamic]
    except FloatingPointError:
        # A later h can amplify a lost nonzero force into a finite displacement.
        # The caller owns rollback; never publish a false stationary success.
        return False
    candidate.velocity[~dynamic] = 0
    return bool(np.isfinite(candidate.position).all() and np.isfinite(candidate.velocity).all())
