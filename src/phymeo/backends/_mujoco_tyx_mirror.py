"""Owned compact HOST state for TYX; native MjData is detached observer scratch.

State ordering is the pinned public mjtState bit order, with each component's
size obtained through mj_stateSize. No native arena, pointer or solver is stored.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

MIRROR_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "qacc",
    "qfrc_actuator",
    "qfrc_constraint",
    "qfrc_applied",
    "xfrc_applied",
)
_STATE_FIELDS = (
    ("mjSTATE_TIME", "time"),
    ("mjSTATE_QPOS", "qpos"),
    ("mjSTATE_QVEL", "qvel"),
    ("mjSTATE_ACT", "act"),
    ("mjSTATE_WARMSTART", "qacc_warmstart"),
    ("mjSTATE_CTRL", "ctrl"),
    ("mjSTATE_QFRC_APPLIED", "qfrc_applied"),
    ("mjSTATE_XFRC_APPLIED", "xfrc_applied"),
    ("mjSTATE_EQ_ACTIVE", "eq_active"),
    ("mjSTATE_MOCAP_POS", "mocap_pos"),
    ("mjSTATE_MOCAP_QUAT", "mocap_quat"),
    ("mjSTATE_USERDATA", "userdata"),
    ("mjSTATE_PLUGIN", "plugin_state"),
)


def _require(condition, message):
    if not condition:
        raise ValueError("MuJoCo TYX mirror: " + message)


def _owned(value):
    return np.frombuffer(value.tobytes(order="C"), dtype=np.float64).reshape(value.shape)


@dataclass(frozen=True, init=False, slots=True, eq=False)
class NativeMirror:
    """Immutable complete integration packet plus explicit published channels."""

    _layout: object
    cpu_state: np.ndarray
    _values: object

    @property
    def time(self):
        return float(self.cpu_state[self._layout.slices["time"]][0])

    def __getattr__(self, name):
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        layout = object.__getattribute__(self, "_layout")
        if name in layout.slices:
            return self.cpu_state[layout.slices[name]].reshape(layout.state_shapes[name])
        raise AttributeError(name)

    def __copy__(self):
        return self._layout.from_parts(self.cpu_state, self._values)

    def changed(self, **updates):
        """Prepare owned new data; never mutate an accepted record."""
        _require(set(updates) <= set(MIRROR_FIELDS) | {"time"}, "unknown changed field")
        cpu = self.cpu_state.copy()
        values = dict(self._values)
        for name, value in updates.items():
            shape = () if name == "time" else self._layout.mirror_shapes[name]
            value = np.asarray(value, dtype=np.float64)
            _require(value.shape == shape and np.isfinite(value).all(), "changed field: " + name)
            if name in self._layout.slices:
                cpu[self._layout.slices[name]] = value.reshape(-1)
            if name != "time":
                values[name] = value
        return self._layout.from_parts(cpu, values)


class NativeStateLayout:
    """Model-bound public native state codec; no data arena is retained."""

    def __init__(self, mj, model):
        _require(str(mj.__version__) == "3.3.7", "pinned MuJoCo 3.3.7 required")
        self.mj, self.model = mj, model
        self.signature = mj.mjtState.mjSTATE_INTEGRATION
        self.size = int(mj.mj_stateSize(model, self.signature))
        slices = {}
        position = 0
        for flag, name in _STATE_FIELDS:
            bit = getattr(mj.mjtState, flag)
            _require(int(self.signature) & int(bit), "complete integration signature")
            size = int(mj.mj_stateSize(model, bit))
            _require(size >= 0, "negative native state component")
            slices[name] = slice(position, position + size)
            position += size
        _require(position == self.size, "native state component layout")
        self.slices = MappingProxyType(slices)
        shapes = {name: (part.stop - part.start,) for name, part in slices.items()}
        shapes.update(
            xfrc_applied=(model.nbody, 6), mocap_pos=(model.nmocap, 3), mocap_quat=(model.nmocap, 4)
        )
        self.state_shapes = MappingProxyType(shapes)
        self.mirror_shapes = MappingProxyType(
            {
                "qpos": (model.nq,),
                "qvel": (model.nv,),
                "act": (model.na,),
                "ctrl": (model.nu,),
                "qacc": (model.nv,),
                "qfrc_actuator": (model.nv,),
                "qfrc_constraint": (model.nv,),
                "qfrc_applied": (model.nv,),
                "xfrc_applied": (model.nbody, 6),
            }
        )

    def from_parts(self, cpu_state, values):
        """Validate the existing checkpoint representation, then own all bytes."""
        _require(
            isinstance(cpu_state, np.ndarray)
            and cpu_state.dtype == np.float64
            and cpu_state.shape == (self.size,)
            and np.isfinite(cpu_state).all(),
            "CPU integration-state dimensions/finite",
        )
        _require(isinstance(values, Mapping) and set(values) == set(MIRROR_FIELDS), "mirror schema")
        cpu = _owned(cpu_state)
        eq = cpu[self.slices["eq_active"]]
        _require(np.all((eq == 0) | (eq == 1)), "native equality activation encoding")
        prepared = {}
        for name in MIRROR_FIELDS:
            value = values[name]
            _require(
                isinstance(value, np.ndarray)
                and value.dtype == np.float64
                and value.shape == self.mirror_shapes[name]
                and np.isfinite(value).all(),
                "mirror dimensions/finite: " + name,
            )
            if name in self.slices:
                state = cpu[self.slices[name]].reshape(value.shape)
                _require(state.tobytes() == value.tobytes(), "CPU state/mirror binding: " + name)
                prepared[name] = state
            else:
                prepared[name] = _owned(value)
        result = object.__new__(NativeMirror)
        object.__setattr__(result, "_layout", self)
        object.__setattr__(result, "cpu_state", cpu)
        object.__setattr__(result, "_values", MappingProxyType(prepared))
        return result

    def capture(self, native):
        _require(native.model is self.model, "native source model identity")
        cpu = np.empty(self.size, dtype=np.float64)
        self.mj.mj_getState(self.model, native, cpu, self.signature)
        return self.from_parts(cpu, {name: getattr(native, name) for name in MIRROR_FIELDS})

    def materialize(self, mirror, native):
        _require(type(mirror) is NativeMirror and mirror._layout is self, "mirror layout owner")
        _require(native.model is self.model, "observer model identity")
        # A failed/previous world may have left arbitrary derived scratch. Native
        # reset clears it; full saved inputs, including warmstart, are then restored.
        self.mj.mj_resetData(self.model, native)
        self.mj.mj_setState(self.model, native, mirror.cpu_state.copy(), self.signature)
        for name in MIRROR_FIELDS:
            getattr(native, name)[:] = getattr(mirror, name)
        return native
