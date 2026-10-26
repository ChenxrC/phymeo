"""Validation shared by compute backends, numerical solvers and runtime."""

from numbers import Real

import numpy as np

from .model import ParticleModel
from .state import ParticleState, StepInputs


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
