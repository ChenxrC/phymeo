"""Local registration and owned snapshots without assumptions about state field names."""

from dataclasses import dataclass, fields, is_dataclass
from typing import Callable

import numpy as np

from ..core.model import ParticleModel
from ..core.rigid import RigidBodyInputs, RigidBodyModel, RigidBodyState
from ..core.scene import BackendContext, HistoryField, SceneInputs, SceneModel, SceneState, _name
from ..core.state import ParticleState, StepInputs
from ..core.validation import validate_rigid_step, validate_step


def _copy_fields(value, expected, names):
    if not isinstance(value, expected):
        raise TypeError(f"expected {expected.__name__}")
    declared = {field.name for field in fields(value)} if is_dataclass(value) else set()
    declared.update(getattr(value, "__dict__", {}))
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        declared.update((slots,) if isinstance(slots, str) else slots)
    extra = declared - set(names) - {"__dict__", "__weakref__"}
    if extra:
        raise TypeError(
            f"unregistered state/input fields: {sorted(extra)}; register a complete StateFamily"
        )
    copied = {}
    for name in names:
        array = np.asarray(getattr(value, name))
        if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError(f"{name} must be a floating-point state/input array")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
        copied[name] = np.array(array, copy=True, order="C")
    return expected(**copied)


def _fixed_invariants(configuration_fields, velocity_fields):
    def check(model, before, after):
        fixed = model.inverse_mass == 0
        for name in configuration_fields:
            if not np.array_equal(getattr(after, name)[fixed], getattr(before, name)[fixed]):
                raise ValueError("candidate changed a fixed configuration")
        for name in velocity_fields:
            if np.any(getattr(after, name)[fixed] != 0):
                raise ValueError("candidate changed a fixed velocity")

    return check


@dataclass(frozen=True)
class StateFamily:
    kind: str
    model_type: type
    state_type: type
    inputs_type: type
    state_fields: tuple[str, ...]
    input_fields: tuple[str, ...]
    dtype_field: str
    validator: Callable
    invariant: Callable | None = None

    def __post_init__(self):
        _name(self.kind, "family")
        if not all(
            isinstance(t, type) for t in (self.model_type, self.state_type, self.inputs_type)
        ):
            raise TypeError("family types must be classes")
        if not callable(self.validator) or (
            self.invariant is not None and not callable(self.invariant)
        ):
            raise TypeError("family validation must be callable")
        for attr, cls in (("state_fields", self.state_type), ("input_fields", self.inputs_type)):
            names = tuple(getattr(self, attr))
            if not names or len(names) != len(set(names)):
                raise ValueError("registered fields must be nonempty and unique")
            for name in names:
                _name(name, "field")
            if is_dataclass(cls) and set(names) != {f.name for f in fields(cls)}:
                raise ValueError("register every state/input dataclass field")
            object.__setattr__(self, attr, names)
        if self.dtype_field not in self.state_fields:
            raise ValueError("dtype_field must be a registered state field")

    def copy_state(self, state, context):
        state = _copy_fields(state, self.state_type, self.state_fields)
        if any(getattr(state, f).dtype != context.dtype for f in self.state_fields):
            raise ValueError("state precision does not match backend context")
        return state

    def snapshot(self, model, state, inputs, context):
        if not isinstance(model, self.model_type):
            raise TypeError(f"expected model {self.model_type.__name__}")
        owned = self.copy_state(state, context)
        forces = _copy_fields(inputs, self.inputs_type, self.input_fields)
        if any(getattr(forces, f).dtype != context.dtype for f in self.input_fields):
            raise ValueError("input precision does not match backend context")
        self.validator(model, owned, forces)
        return owned, forces

    def candidate(self, model, before, after, context):
        candidate = self.copy_state(after, context)
        for name in self.state_fields:
            a, b = getattr(before, name), getattr(candidate, name)
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"candidate changed {name} shape or precision")
        if self.invariant is not None:
            self.invariant(model, before, candidate)
        return candidate


def _particle_validator(model, state, inputs):
    validate_step(model, state, inputs, 1.0)


def _rigid_validator(model, state, inputs):
    validate_rigid_step(model, state, inputs, 1.0)


def copy_history(history, schema):
    schema = tuple(schema)
    if any(not isinstance(f, HistoryField) for f in schema):
        raise TypeError("history schema must contain HistoryField values")
    names = {f.name for f in schema}
    if len(names) != len(schema) or set(history) != names:
        raise ValueError("history fields do not match the declared schema")
    owned = {}
    for entry in schema:
        value = np.asarray(history[entry.name])
        if value.shape != entry.shape or value.dtype != entry.dtype:
            raise ValueError(f"history {entry.name} shape or precision changed")
        if not np.isfinite(value).all():
            raise ValueError(f"history {entry.name} contains non-finite values")
        owned[entry.name] = np.array(value, copy=True, order="C")
    return owned


class StateRegistry:
    def __init__(self, *, defaults=True):
        self._families = {}
        if defaults:
            self.register(
                StateFamily(
                    "particles",
                    ParticleModel,
                    ParticleState,
                    StepInputs,
                    ("position", "velocity"),
                    ("force",),
                    "position",
                    _particle_validator,
                    _fixed_invariants(("position",), ("velocity",)),
                )
            )
            self.register(
                StateFamily(
                    "rigid_bodies",
                    RigidBodyModel,
                    RigidBodyState,
                    RigidBodyInputs,
                    ("position", "orientation", "velocity", "angular_velocity"),
                    ("force", "torque"),
                    "position",
                    _rigid_validator,
                    _fixed_invariants(
                        ("position", "orientation"), ("velocity", "angular_velocity")
                    ),
                )
            )

    def register(self, family):
        if not isinstance(family, StateFamily):
            raise TypeError("register a StateFamily")
        if family.kind in self._families:
            raise ValueError(f"duplicate state family: {family.kind}")
        self._families[family.kind] = family

    def get(self, kind):
        try:
            return self._families[kind]
        except (KeyError, TypeError) as error:
            raise ValueError(f"unregistered state family: {kind}") from error

    def copy(self):
        result = StateRegistry(defaults=False)
        result._families = dict(self._families)
        return result

    @staticmethod
    def _check_domains(scene, state, context):
        if not isinstance(scene, SceneModel) or not isinstance(state, SceneState):
            raise TypeError("expected SceneModel and SceneState")
        if not isinstance(context, BackendContext):
            raise TypeError("expected BackendContext")
        scene.assert_unchanged()
        names = {d.name for d in scene.domains}
        if set(state.domains) != names:
            raise ValueError("state domains do not match scene")
        return names

    def snapshot(self, scene, state, inputs, context, history_fields=()):
        names = self._check_domains(scene, state, context)
        if not isinstance(inputs, SceneInputs):
            raise TypeError("expected SceneInputs")
        if set(inputs.domains) != names:
            raise ValueError("input domains do not match scene")
        states, forces = {}, {}
        for domain in scene.domains:
            states[domain.name], forces[domain.name] = self.get(domain.family).snapshot(
                domain.model,
                state.domains[domain.name],
                inputs.domains[domain.name],
                context,
            )
        return SceneState(states, copy_history(state.history, history_fields)), SceneInputs(forces)

    def clone_state(self, scene, state, context, history_fields=()):
        self._check_domains(scene, state, context)
        states = {
            d.name: self.get(d.family).copy_state(state.domains[d.name], context)
            for d in scene.domains
        }
        return SceneState(states, copy_history(state.history, history_fields))

    def validate_candidate(self, scene, before, after, context, history_fields=()):
        self._check_domains(scene, before, context)
        self._check_domains(scene, after, context)
        states = {
            d.name: self.get(d.family).candidate(
                d.model,
                before.domains[d.name],
                after.domains[d.name],
                context,
            )
            for d in scene.domains
        }
        return SceneState(states, copy_history(after.history, history_fields))
