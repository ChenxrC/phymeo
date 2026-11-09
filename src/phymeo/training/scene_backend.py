"""Borrow real HOST scenes through the vector-environment batch boundary.

A logical world owns one original public scene. Physics and its publication
remain in that scene; this adapter never writes configuration or velocity.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from numbers import Integral
from threading import RLock
from types import MappingProxyType

import numpy as np

from ..core.coupled import CoupledInputs
from .resident_stepper import PhysicsStepFailure


@dataclass(frozen=True)
class _ArrayValue:
    dtype: str
    shape: tuple
    data: bytes


@dataclass(frozen=True)
class _RecordValue:
    kind: type
    values: tuple


@dataclass(frozen=True)
class _MappingValue:
    values: tuple


def _pack(value):
    """Own immutable values; invoke no simulation or scene callback."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise TypeError("scene controls require real numeric arrays")
        return _ArrayValue(value.dtype.str, value.shape, value.tobytes(order="C"))
    if is_dataclass(value) and not isinstance(value, type):
        return _RecordValue(
            type(value),
            tuple(
                (field.name, _pack(getattr(value, field.name)))
                for field in fields(value)
                if field.init
            ),
        )
    if isinstance(value, Mapping):
        return _MappingValue(tuple((key, _pack(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_pack(item) for item in value)
    if value is None or isinstance(value, (str, bytes, bool, int, float, np.generic)):
        return value
    raise TypeError("scene controls must contain public value records and numeric arrays")


def _unpack(value):
    if isinstance(value, _ArrayValue):
        return np.frombuffer(value.data, dtype=np.dtype(value.dtype)).reshape(value.shape)
    if isinstance(value, _RecordValue):
        return value.kind(**{name: _unpack(item) for name, item in value.values})
    if isinstance(value, _MappingValue):
        return MappingProxyType({key: _unpack(item) for key, item in value.values})
    if isinstance(value, tuple):
        return tuple(_unpack(item) for item in value)
    return value


@dataclass(frozen=True, init=False)
class SceneControl:
    """Exact detached actions and external inputs, held until explicitly replaced.

    Properties materialize fresh public command/input records. Mutating such a
    returned input, or the caller's original mapping, cannot alter this packet.
    Actions are original RobotScene or MultiRobotScene command mappings.
    """

    _actions: object
    _external: object

    def __init__(self, actions, external=None):
        if not isinstance(actions, Mapping):
            raise TypeError("scene actions must be an original robot command mapping")
        if external is not None and not isinstance(external, CoupledInputs):
            raise TypeError("external must be original public CoupledInputs or None")
        object.__setattr__(self, "_actions", _pack(actions))
        object.__setattr__(self, "_external", _pack(external))

    @property
    def actions(self):
        return _unpack(self._actions)

    @property
    def external(self):
        return _unpack(self._external)


def _keys(values):
    result = tuple(values)
    if any(
        isinstance(key, (bool, np.bool_)) or not isinstance(key, Integral) or key < 0
        for key in result
    ):
        raise ValueError("logical world keys must be nonnegative integers")
    result = tuple(map(int, result))
    if len(set(result)) != len(result):
        raise ValueError("logical world keys must be unique")
    return result


def _stamp(reading):
    time = float(reading.time_s)
    counter = getattr(reading, "step_number", None)
    if not np.isfinite(time) or time < 0:
        raise ValueError("scene observation needs a finite nonnegative publication time")
    return time, counter


def _cleanup(scenes, primary=None):
    errors = []
    for scene in scenes:
        try:
            scene.close()
        except Exception as error:
            errors.append(error)
    if primary is not None:
        for error in errors:
            primary.add_note("replacement scene cleanup failed: " + repr(error))
    elif errors:
        raise RuntimeError("one or more owned scenes could not close") from errors[0]


class SceneBatchBackend:
    """Own independent original scenes; ResidentVectorEnv borrows this backend.

    factory(world) transfers a fresh initial scene to this object. It may retain
    public context for task callbacks, but must not advance/close that scene.
    Rebuilding a selected world resets its physical initial state through the
    factory. Logical keys are not rewritten into the scene's physical owner IDs.
    """

    def __init__(self, world_keys, factory):
        keys = _keys(world_keys)
        if not keys or not callable(factory):
            raise ValueError("nonempty world keys and a callable scene factory are required")
        self.world_keys, self._factory = keys, factory
        self._lock, self._closed, self._controls = RLock(), False, None
        self._scenes, self._retired = {}, []
        self._scenes = self._prepare(keys)

    def _require_open(self):
        if self._closed:
            raise RuntimeError("scene batch backend is closed")

    def _selected(self, keys):
        selected = self.world_keys if keys is None else _keys(keys)
        if not set(selected) <= set(self.world_keys):
            raise ValueError("unknown logical world key")
        return selected

    def _prepare(self, selected):
        prepared = {}
        existing = {id(scene) for scene in (*self._scenes.values(), *self._retired)}
        try:
            for key in selected:
                scene = self._factory(key)
                if id(scene) in existing or any(scene is other for other in prepared.values()):
                    raise ValueError("factory must transfer a fresh independent scene")
                prepared[key] = scene
                if any(
                    not callable(getattr(scene, name, None))
                    for name in ("observe", "step", "close")
                ):
                    raise TypeError("factory must return an original public scene")
                time, counter = _stamp(scene.observe())
                if time != 0 or counter not in (None, 0):
                    raise ValueError("factory must return the scene's original initial publication")
        except Exception as error:
            # Aliases of existing owners never enter prepared, and are not closed.
            _cleanup(
                tuple(
                    scene for scene in prepared.values() if callable(getattr(scene, "close", None))
                ),
                error,
            )
            raise
        return prepared

    def snapshot(self, keys=None):
        with self._lock:
            self._require_open()
            return MappingProxyType(
                {key: self._scenes[key].observe() for key in self._selected(keys)}
            )

    def set_controls(self, controls):
        with self._lock:
            self._require_open()
            if not isinstance(controls, Mapping) or set(controls) != set(self.world_keys):
                raise ValueError("controls must cover exactly all logical worlds")
            if any(not isinstance(controls[key], SceneControl) for key in self.world_keys):
                raise TypeError("every logical world needs an immutable SceneControl")
            self._controls = MappingProxyType({key: controls[key] for key in self.world_keys})

    def step(self, h):
        with self._lock:
            self._require_open()
            if isinstance(h, (bool, np.bool_)) or not np.isfinite(h) or h <= 0:
                raise ValueError("step duration must be positive and finite")
            if self._controls is None:
                raise RuntimeError("set complete scene controls before stepping")
            accepted = {key: False for key in self.world_keys}
            for key in self.world_keys:
                scene, control = self._scenes[key], self._controls[key]
                before = None
                entered = False
                try:
                    before = _stamp(scene.observe())
                    actions, external = control.actions, control.external
                    entered = True
                    scene.step(h, actions=actions, external=external)
                    accepted[key] = True
                except Exception as error:
                    known = True
                    if entered:
                        try:
                            after = _stamp(scene.observe())
                            if after == before:
                                accepted[key] = False
                            elif after[0] > before[0] and (
                                before[1] is None or after[1] == before[1] + 1
                            ):
                                accepted[key] = True
                            else:
                                known = False
                        except Exception:
                            known = False
                    raise PhysicsStepFailure(
                        f"scene world {key} failed; accepted publications are retained",
                        accepted_worlds=accepted,
                        acceptance_known=known,
                    ) from error
            return MappingProxyType(accepted)

    def reset(self, keys=None):
        with self._lock:
            self._require_open()
            selected = self._selected(keys)
            if not selected:
                return
            prepared = self._prepare(selected)
            old = [self._scenes[key] for key in selected]
            self._scenes = {**self._scenes, **prepared}
            self._controls = None
            # Publication is complete. If cleanup fails, retain ownership so close
            # can retry; never pretend the new scenes were rolled back.
            for scene in old:
                try:
                    scene.close()
                except Exception:
                    self._retired.append(scene)
            if self._retired:
                raise RuntimeError("reset installed initial scenes but old scene cleanup failed")

    def close(self):
        with self._lock:
            if self._closed and not self._retired:
                return
            self._closed = True
            owned = [*self._scenes.values(), *self._retired]
            self._scenes, self._retired, self._controls = {}, [], None
            errors = []
            for scene in owned:
                try:
                    scene.close()
                except Exception as error:
                    self._retired.append(scene)
                    errors.append(error)
            if errors:
                raise RuntimeError("owned scene cleanup failed; close may be retried") from errors[
                    0
                ]

    def __enter__(self):
        with self._lock:
            self._require_open()
            return self

    def __exit__(self, kind, error, traceback):
        try:
            self.close()
        except Exception as cleanup_error:
            if error is None:
                raise
            error.add_note("scene backend close failed: " + repr(cleanup_error))
        return False
