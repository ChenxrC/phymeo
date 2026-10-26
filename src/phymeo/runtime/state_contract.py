"""Explicit state-family contracts used by the transactional runtime."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..core.model import ParticleModel
from ..core.rigid import RigidBodyInputs, RigidBodyModel, RigidBodyState
from ..core.state import ParticleState, StepInputs
from ..core.validation import validate_rigid_step, validate_step


@dataclass(frozen=True)
class StateContract:
    """Private dispatch metadata; each family preserves all of its physical fields."""

    kind: str
    model_type: type
    state_type: type
    inputs_type: type
    validator: Callable
    configuration_fields: tuple[str, ...]
    velocity_fields: tuple[str, ...]
    input_fields: tuple[str, ...]

    def _copy_state(self, state):
        # Extension providers may override copy() with a shallow/cache copy.
        # Reconstruct the known family so validation and ownership stay ours.
        return self.state_type(
            **{name: getattr(state, name) for name in self.configuration_fields + self.velocity_fields}
        )

    def snapshot(self, model, state, inputs, dt):
        for name, value, expected in (
            ("model", model, self.model_type),
            ("state", state, self.state_type),
            ("inputs", inputs, self.inputs_type),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{name} type must be {expected.__name__} for {self.kind}")
        canonical_state = self._copy_state(state)
        canonical_inputs = self.inputs_type(
            **{name: getattr(inputs, name) for name in self.input_fields}
        )
        snapshot, owned_inputs, h = self.validator(model, canonical_state, canonical_inputs, dt)
        if self.kind == "particles":
            owned_inputs = StepInputs(owned_inputs)  # legacy validator returns the owned force
        return snapshot, owned_inputs, h

    def candidate(self, result_state, snapshot, model):
        if not isinstance(result_state, self.state_type):
            return None, "invalid state type"
        try:
            candidate = self._copy_state(result_state)
        except (TypeError, ValueError) as error:
            return None, f"invalid candidate: {error}"
        # Constructors validate matching counts/dtypes of every array, quaternion
        # unit length for rigid states, and finite entries before these comparisons.
        if candidate.position.shape != snapshot.position.shape:
            return None, "candidate state shape changed"
        if candidate.position.dtype != snapshot.position.dtype:
            return None, "candidate state dtype changed"
        fixed = model.inverse_mass == 0
        for name in self.configuration_fields:
            if not np.array_equal(getattr(candidate, name)[fixed], getattr(snapshot, name)[fixed]):
                return None, "candidate violates fixed-state configuration invariants"
        for name in self.velocity_fields:
            if np.any(getattr(candidate, name)[fixed] != 0):
                return None, "candidate violates fixed-state velocity invariants"
        return candidate, None


_PARTICLES = StateContract(
    "particles",
    ParticleModel,
    ParticleState,
    StepInputs,
    validate_step,
    ("position",),
    ("velocity",),
    ("force",),
)
_RIGID = StateContract(
    "rigid_bodies",
    RigidBodyModel,
    RigidBodyState,
    RigidBodyInputs,
    validate_rigid_step,
    ("position", "orientation"),
    ("velocity", "angular_velocity"),
    ("force", "torque"),
)


def state_contract(kind):
    if isinstance(kind, str):
        if kind == "particles":
            return _PARTICLES
        if kind == "rigid_bodies":
            return _RIGID
    raise ValueError("unsupported state capability")
