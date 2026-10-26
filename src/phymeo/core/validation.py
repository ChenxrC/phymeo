"""Validation shared by compute backends, numerical solvers and runtime."""

from numbers import Real

import numpy as np

from .model import ParticleModel
from .rigid import RigidBodyInputs, RigidBodyModel, RigidBodyState
from .state import ParticleState, StepInputs


def validate_dt(dt, dtype):
    """Return a positive finite scalar in the requested state precision."""
    if isinstance(dt, (bool, np.bool_)) or not isinstance(dt, Real):
        raise ValueError("dt must be a finite positive real scalar")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a finite positive real scalar")
    with np.errstate(over="ignore", under="ignore"):
        h = np.dtype(dtype).type(dt)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("dt is not representable in the state precision")
    return h


def validate_step(
    model: ParticleModel, state: ParticleState, inputs: StepInputs, dt: float
) -> tuple[ParticleState, np.ndarray, np.floating]:
    """Validate mutable input buffers and return owned data for one attempted step."""
    candidate = state.copy()
    force = StepInputs(inputs.force).force
    if model.particle_count != candidate.particle_count or force.shape != candidate.position.shape:
        raise ValueError("model, state and force must have the same particle count")
    dtype = candidate.position.dtype
    if model.inverse_mass.dtype != dtype or force.dtype != dtype:
        raise ValueError("model, state and force must have the same floating-point dtype")
    return candidate, force, validate_dt(dt, dtype)


def validate_rigid_step(
    model: RigidBodyModel, state: RigidBodyState, inputs: RigidBodyInputs, dt: float
):
    candidate, owned_inputs = state.copy(), inputs.copy()
    if (
        model.body_count != candidate.body_count
        or owned_inputs.force.shape != candidate.position.shape
    ):
        raise ValueError("model, state and inputs must have the same body count")
    dtype = candidate.position.dtype
    if model.inverse_mass.dtype != dtype or owned_inputs.force.dtype != dtype:
        raise ValueError("model, state and inputs must have the same dtype")
    return candidate, owned_inputs, validate_dt(dt, dtype)
