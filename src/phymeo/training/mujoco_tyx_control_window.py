"""Owned evidence for bounded, endpoint-semantics policy control windows.

This module advances no physics and imports neither Torch nor a runtime. A
record preserves public observations; it is not a mid-window checkpoint.
"""

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from hashlib import sha256
from numbers import Real
from types import MappingProxyType

import numpy as np

_SEMANTICS = {
    "schema": "mujoco-tyx-control-window-v1",
    "action_hold": "native_channels",
    "reward": "endpoint",
    "done": "endpoint",
    "horizon_units": "policy_steps",
}


def _integer(value, name, minimum=0, maximum=None):
    if type(value) is not int:
        raise TypeError(name + " must be a builtin integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(name + " outside the bounded domain")
    return value


def _owned(value):
    """Clone detached public records, preserving their exact dataclass types."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays cannot be control-window evidence")
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if is_dataclass(value) and not isinstance(value, type):
        if not value.__dataclass_params__.frozen:
            raise TypeError("control-window dataclass evidence must be frozen")
        clone = copy.copy(value)
        for item in fields(value):
            object.__setattr__(clone, item.name, _owned(getattr(value, item.name)))
        return clone
    if isinstance(value, Mapping):
        return MappingProxyType({key: _owned(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_owned(item) for item in value)
    if value is None or isinstance(value, (str, bytes, int, float, bool, np.generic)):
        return value
    raise TypeError("unsupported detached control-window evidence type: " + type(value).__name__)


@dataclass(frozen=True)
class MuJoCoTyxControlWindowSpec:
    """One native action held for 1..8 physical steps; endpoint reward/done."""

    physics_steps: int = 1
    sha256: str = field(init=False)

    def __post_init__(self):
        _integer(self.physics_steps, "physics_steps", 1, 8)
        raw = json.dumps(
            {**_SEMANTICS, "physics_steps": self.physics_steps},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        object.__setattr__(self, "sha256", sha256(raw).hexdigest())

    def binding(self, h):
        if isinstance(h, (bool, np.bool_)) or not isinstance(h, Real):
            raise TypeError("physical timestep must be a real number")
        h = float(h)
        if not math.isfinite(h) or h <= 0 or not math.isfinite(h * self.physics_steps):
            raise ValueError("physical timestep must be finite and positive")
        return {
            **_SEMANTICS,
            "physics_steps": self.physics_steps,
            "physics_dt_s": h,
            "policy_dt_s": h * self.physics_steps,
        }


@dataclass(frozen=True)
class MuJoCoTyxSubstepRecord:
    """One actual request, including success with subsequently failed diagnostics.

    ``accepted_worlds`` records the physical return independently from snapshot
    availability. No missing state is synthesized or obtained by repeating a step.
    """

    ordinal: int
    entry_revision: int
    exit_revision: int
    accepted_worlds: Mapping
    states: Mapping | None
    diagnostics_complete: bool
    error: str | None = None

    def __post_init__(self):
        _integer(self.ordinal, "substep ordinal", 0, 7)
        _integer(self.entry_revision, "entry revision")
        _integer(self.exit_revision, "exit revision", self.entry_revision)
        if not isinstance(self.accepted_worlds, Mapping):
            raise TypeError("accepted_worlds must be a mapping")
        accepted = dict(self.accepted_worlds)
        for key, value in accepted.items():
            _integer(key, "world key")
            if type(value) is not bool:
                raise TypeError("world acceptance must be a builtin bool")
        if type(self.diagnostics_complete) is not bool:
            raise TypeError("diagnostics_complete must be a builtin bool")
        if self.states is not None:
            if not isinstance(self.states, Mapping) or set(self.states) != set(accepted):
                raise ValueError("state/acceptance world membership differs")
        elif self.diagnostics_complete:
            raise ValueError("complete diagnostics require actual public states")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("diagnostic error must be text or None")
        object.__setattr__(self, "accepted_worlds", MappingProxyType(accepted))
        object.__setattr__(self, "states", _owned(self.states))


@dataclass(frozen=True)
class MuJoCoTyxControlWindowRecord:
    """Bounded accepted prefix, not an atomic transaction over multiple steps."""

    spec_sha256: str
    physics_steps: int
    attempted: int
    completed: int
    substeps: tuple
    action_receipt: object
    phase: str
    failed_substep: int | None = None

    def __post_init__(self):
        expected = MuJoCoTyxControlWindowSpec(self.physics_steps).sha256
        if self.spec_sha256 != expected:
            raise ValueError("control-window specification hash mismatch")
        _integer(self.attempted, "attempted", 0, self.physics_steps)
        _integer(self.completed, "completed", 0, self.attempted)
        substeps = tuple(self.substeps)
        if len(substeps) != self.attempted:
            raise ValueError("attempt count must equal the actual prefix length")
        for ordinal, item in enumerate(substeps):
            if type(item) is not MuJoCoTyxSubstepRecord or item.ordinal != ordinal:
                raise ValueError("substeps must be complete ordered records starting at zero")
        completed = sum(
            bool(item.accepted_worlds) and all(item.accepted_worlds.values()) for item in substeps
        )
        if completed != self.completed:
            raise ValueError("completed count must follow actual whole-batch acceptance")
        if not isinstance(self.phase, str) or not self.phase:
            raise ValueError("control-window phase must be nonempty text")
        if self.failed_substep is not None:
            _integer(self.failed_substep, "failed substep", 0, self.attempted - 1)
        # Substep records already own all nested public evidence. Keep those
        # immutable objects rather than copying their potentially large arrays twice.
        object.__setattr__(self, "substeps", substeps)
        object.__setattr__(self, "action_receipt", _owned(self.action_receipt))
