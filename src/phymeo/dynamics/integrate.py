"""Velocity-first particle integration, independent of constraint algorithms."""

import numpy as np

from ..core.model import ParticleModel
from ..core.state import ParticleState


def predict(
    model: ParticleModel, candidate: ParticleState, force: np.ndarray, h: np.floating
) -> bool:
    """Update an owned candidate in place; return whether every value remains finite."""
    dynamic = model.inverse_mass > 0
    # Do not evaluate forces on fixed particles: even unused arithmetic may overflow.
    with np.errstate(over="ignore", invalid="ignore"):
        acceleration = model.gravity + model.inverse_mass[dynamic, None] * force[dynamic]
        candidate.velocity[dynamic] += h * acceleration
        candidate.position[dynamic] += h * candidate.velocity[dynamic]
    candidate.velocity[~dynamic] = 0
    return bool(np.isfinite(candidate.position).all() and np.isfinite(candidate.velocity).all())
