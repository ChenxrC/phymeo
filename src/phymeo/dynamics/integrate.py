"""Velocity-first particle integration, independent of constraint algorithms."""

from numbers import Real

import numpy as np

from ..core.model import ParticleModel
from ..core.state import ParticleState, StepInputs


def validate_step(
    model: ParticleModel, state: ParticleState, inputs: StepInputs, dt: float
) -> tuple[ParticleState, np.ndarray, np.floating]:
    """Validate mutable input buffers and return owned data for one attempted step."""
    if isinstance(dt, (bool, np.bool_)) or not isinstance(dt, Real):
        raise ValueError("dt must be a finite positive real scalar")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a finite positive real scalar")
    candidate = state.copy()
    force = StepInputs(inputs.force).force
    if model.particle_count != candidate.particle_count or force.shape != candidate.position.shape:
        raise ValueError("model, state and force must have the same particle count")
    dtype = candidate.position.dtype
    if model.inverse_mass.dtype != dtype or force.dtype != dtype:
        raise ValueError("model, state and force must have the same floating-point dtype")
    with np.errstate(over="ignore", under="ignore"):
        h = dtype.type(dt)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("dt is not representable in the state precision")
    return candidate, force, h


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
