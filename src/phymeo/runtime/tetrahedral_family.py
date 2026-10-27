"""Local solid family with constitutive-domain and fixed-anchor candidate checks."""

import numpy as np

from ..core.state import ParticleState, StepInputs
from ..core.tetrahedral import TetrahedralModel, tetra_array
from ..materials.stvk import deformation_gradient
from .registry import StateFamily


def _state(model, state):
    if not isinstance(model, TetrahedralModel) or not isinstance(state, ParticleState):
        raise TypeError("expected TetrahedralModel and ParticleState")
    for name in ("position", "velocity"):
        value = tetra_array(getattr(state, name), name, ndim=2, columns=3)
        if value.shape != model.rest_position.shape:
            raise ValueError(f"solid {name} must match the model node count")
    if np.any(state.velocity[model.fixed] != 0):
        raise ValueError("fixed solid nodes require zero velocity")
    deformation_gradient(model, state.position)


def validate_tetrahedral_step(model, state, inputs):
    _state(model, state)
    if not isinstance(inputs, StepInputs):
        raise TypeError("solid requires StepInputs")
    force = tetra_array(inputs.force, "force", ndim=2, columns=3)
    if force.shape != model.rest_position.shape:
        raise ValueError("solid force must match the model node count")


def _invariants(model, before, after):
    _state(model, after)
    if not np.array_equal(before.position[model.fixed], after.position[model.fixed]):
        raise ValueError("candidate changed a fixed solid configuration")


def tetrahedral_family():
    return StateFamily(
        "tetrahedra",
        TetrahedralModel,
        ParticleState,
        StepInputs,
        ("position", "velocity"),
        ("force",),
        "position",
        validate_tetrahedral_step,
        _invariants,
    )
