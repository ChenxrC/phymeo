"""Exact, bounded MuJoCo-TYX accepted-state checkpoints.

This format is a checksum-protected trusted application artifact, not an
authentication mechanism. It contains no executable code, pickle or native
MjData. Physics continuation is bound to the original model/backend/layout.
Application, Env, PolicyRunner, RNG and policy history are separate owners.
"""

import base64
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType

import numpy as np

MAGIC = b"PHYMEOTYXCP1\n"
MAX_BYTES = 128 * 1024 * 1024
MAX_WORLDS = 256
MAX_ARRAY_ITEMS = 1_048_576
_DTYPES = {"<f4", "<f8", "<i4", "<i8", "|b1"}


def _require(condition, message):
    if not condition:
        raise ValueError("MuJoCo TYX checkpoint: " + message)


def _records():
    # Lazy, explicit whitelist. Serialized names never cause dynamic imports.
    from ._mujoco_tyx_runtime import (
        MuJoCoTyxContact,
        MuJoCoTyxDynamics,
        MuJoCoTyxNormalContact,
        MuJoCoTyxStepReceipt,
    )
    from .mujoco_runtime import MuJoCoControl, MuJoCoInitialState
    from .mujoco_tyx._constraint_publication import MuJoCoTyxConstraintRow

    classes = (
        MuJoCoControl,
        MuJoCoInitialState,
        MuJoCoTyxContact,
        MuJoCoTyxDynamics,
        MuJoCoTyxNormalContact,
        MuJoCoTyxStepReceipt,
        MuJoCoTyxConstraintRow,
    )
    return {cls.__name__: cls for cls in classes}


def _encode(value, depth=0):
    _require(depth <= 32, "nested payload limit")
    if isinstance(value, np.ndarray):
        _require(
            value.dtype.str in _DTYPES and value.ndim <= 4 and value.size <= MAX_ARRAY_ITEMS,
            "unsupported array dtype or extent",
        )
        _require(np.isfinite(value).all(), "nonfinite array")
        return {
            "$array": value.dtype.str,
            "shape": list(value.shape),
            "data": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
        }
    if is_dataclass(value):
        _require(type(value).__name__ in _records(), "unsupported record")
        return {
            "$record": type(value).__name__,
            "fields": {f.name: _encode(getattr(value, f.name), depth + 1) for f in fields(value)},
        }
    if isinstance(value, Mapping):
        _require(
            all(type(key) is str and not key.startswith("$") for key in value),
            "mapping keys must be ordinary strings",
        )
        return {key: _encode(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item, depth + 1) for item in value]
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        _require(math.isfinite(value), "nonfinite scalar")
        return value
    raise ValueError("unsupported checkpoint value")


def _decode(value, depth=0):
    _require(depth <= 32, "nested payload limit")
    if type(value) is list:
        return tuple(_decode(item, depth + 1) for item in value)
    if type(value) is dict:
        if "$array" in value:
            _require(set(value) == {"$array", "shape", "data"}, "array fields")
            dtype, shape, encoded = value["$array"], value["shape"], value["data"]
            _require(dtype in _DTYPES and type(shape) is list and len(shape) <= 4, "array layout")
            size = 1
            for dimension in shape:
                _require(
                    type(dimension) is int and 0 <= dimension <= MAX_ARRAY_ITEMS, "array dimension"
                )
                size *= dimension
                _require(size <= MAX_ARRAY_ITEMS, "array item limit")
            length = size * np.dtype(dtype).itemsize
            _require(
                type(encoded) is str and len(encoded) == 4 * ((length + 2) // 3),
                "encoded array byte count",
            )
            raw = base64.b64decode(encoded, validate=True)
            _require(len(raw) == length, "array byte count")
            result = np.frombuffer(raw, dtype=dtype).reshape(tuple(shape))
            _require(np.isfinite(result).all(), "nonfinite array")
            return result
        if "$record" in value:
            _require(set(value) == {"$record", "fields"}, "record fields")
            cls = _records().get(value["$record"])
            _require(cls is not None and type(value["fields"]) is dict, "unknown record")
            _require(set(value["fields"]) == {f.name for f in fields(cls)}, "record schema")
            return cls(**{key: _decode(item, depth + 1) for key, item in value["fields"].items()})
        _require(
            all(type(key) is str and not key.startswith("$") for key in value), "mapping schema"
        )
        return MappingProxyType({key: _decode(item, depth + 1) for key, item in value.items()})
    _require(value is None or type(value) in (bool, int, float, str), "scalar type")
    if type(value) is float:
        _require(math.isfinite(value), "nonfinite scalar")
    return value


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _bad_number(value):
    raise ValueError("nonfinite checkpoint JSON: " + value)


@dataclass(frozen=True, init=False)
class MuJoCoTyxCheckpoint:
    """Immutable owned bytes, with explicit source binding and original clock bits.

    Use runtime.checkpoint(), then restore into an already initialized compatible
    runtime. This is not a MuJoCoInitialState and does not normalize quaternions.
    Serialization preserves all captured arrays bit-for-bit. It is intended for
    trusted application storage; a SHA256 is corruption detection, not a signature.
    """

    _bytes: bytes

    def __init__(self, raw):
        _require(isinstance(raw, (bytes, bytearray, memoryview)), "expected bytes")
        _require(
            (raw.nbytes if isinstance(raw, memoryview) else len(raw)) <= MAX_BYTES,
            "serialized byte limit",
        )
        owned = bytes(raw)
        _require(owned.startswith(MAGIC) and len(owned) > len(MAGIC) + 65, "format header")
        digest, separator, body = owned[len(MAGIC) :].partition(b"\n")
        _require(
            separator and digest == hashlib.sha256(body).hexdigest().encode("ascii"),
            "payload checksum",
        )
        object.__setattr__(self, "_bytes", owned)

    @classmethod
    def from_bytes(cls, raw):
        return cls(raw)

    def to_bytes(self):
        return self._bytes

    @property
    def sha256(self):
        return hashlib.sha256(self._bytes).hexdigest()

    def _json(self):
        body = self._bytes[len(MAGIC) + 65 :]
        value = json.loads(body, object_pairs_hook=_unique, parse_constant=_bad_number)
        _require(
            type(value) is dict and set(value) == {"binding", "revision", "state"},
            "top-level schema",
        )
        _require(type(value["revision"]) is int and value["revision"] >= 0, "revision")
        return value

    @property
    def source_revision(self):
        return self._json()["revision"]

    @property
    def binding(self):
        return _decode(self._json()["binding"])

    def _read(self, expected_binding):
        value = self._json()
        # Reject a wrong model/mode/capacity/world order before decoding arrays.
        _require(value["binding"] == _encode(expected_binding), "source/backend/layout binding")
        return _decode(value["state"])


def _make_checkpoint(binding, revision, state):
    value = {"binding": _encode(binding), "revision": revision, "state": _encode(state)}
    body = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
    return MuJoCoTyxCheckpoint(
        MAGIC + hashlib.sha256(body).hexdigest().encode("ascii") + b"\n" + body
    )


def _model_binding(runtime):
    source, plan, model = runtime._compilation.source, runtime._plan, runtime._model
    _require(len(runtime.world_keys) <= MAX_WORLDS, "at most 256 checkpoint worlds")
    return {
        "schema": "mujoco-tyx-exact-checkpoint-v1",
        "primary_sha256": hashlib.sha256(source.primary_xml).hexdigest(),
        "archive_sha256": hashlib.sha256(source.archive_bytes).hexdigest(),
        "model_sha256": hashlib.sha256(source.model_binary).hexdigest(),
        "mujoco_version": source.mujoco_version,
        "mode": runtime._mode,
        "implementation": "tyx",
        "world_keys": runtime.world_keys,
        "dimensions": tuple(
            int(getattr(model, key)) for key in ("nq", "nv", "nu", "na", "nbody", "ngeom")
        ),
        "kind": plan.kind,
        "domain": runtime.capabilities["domain"],
        "cpu_state_size": int(
            runtime._mj.mj_stateSize(model, runtime._mj.mjtState.mjSTATE_INTEGRATION)
        ),
        "timestep": runtime.timestep,
        "capacities": (runtime._contact_capacity, runtime._constraint_capacity),
        "layout": (max(8, plan.nq), max(8, plan.nv), 2),
        "state_precision": "fp32-raw-bank; compensated-clock-high-minus-low",
        "solver_history": "cold-per-step; no-integrator-warmstart-or-contact-history",
    }


def _accepted_bank(device):
    # Read the accepted bank, never download()'s possibly rejected candidate.
    with device.tx.use_runtime(device.runtime):
        device.runtime.sync()
        device.counts["synchronizations"] += 1
        return tuple(
            device._get(buffers[device.active]) for buffers in (device.q, device.v, device.time)
        )


def _verify_bank(device, packed):
    actual = _accepted_bank(device)
    for got, expected in zip(actual, packed, strict=True):
        _require(
            got.dtype == expected.dtype
            and got.shape == expected.shape
            and got.tobytes() == expected.tobytes(),
            "replacement bank readback differs",
        )


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


def capture(runtime):
    binding = _model_binding(runtime)
    packed = _accepted_bank(runtime._device)
    for got, shape in zip(
        packed,
        (
            (len(runtime.world_keys), binding["layout"][0]),
            (len(runtime.world_keys), binding["layout"][1]),
            (len(runtime.world_keys), 2),
        ),
        strict=True,
    ):
        _require(
            got.dtype == np.float32 and got.shape == shape and np.isfinite(got).all(),
            "accepted bank layout",
        )
    worlds = []
    for index, key in enumerate(runtime.world_keys):
        data = runtime._data[key]
        _require(
            packed[0][index, : runtime._plan.nq].astype(np.float64).tobytes()
            == data.qpos.tobytes(),
            "accepted q mirror differs",
        )
        _require(
            packed[1][index, : runtime._plan.nv].astype(np.float64).tobytes()
            == data.qvel.tobytes(),
            "accepted v mirror differs",
        )
        _require(
            float(packed[2][index, 0]) - float(packed[2][index, 1]) == data.time,
            "accepted clock mirror differs",
        )
        cpu_state = np.empty(binding["cpu_state_size"], dtype=np.float64)
        runtime._mj.mj_getState(
            runtime._model, data, cpu_state, runtime._mj.mjtState.mjSTATE_INTEGRATION
        )
        worlds.append(
            {
                "cpu_state": cpu_state,
                "key": key,
                "tick": runtime._ticks[key],
                "time": float(data.time),
                "mirror": {name: getattr(data, name).copy() for name in MIRROR_FIELDS},
                "control": runtime._controls[key],
                "initial": runtime._initial[key],
                "receipt": runtime._receipts[key],
            }
        )
    return _make_checkpoint(
        binding,
        runtime._checkpoint_revision,
        {"q": packed[0], "v": packed[1], "clock": packed[2], "worlds": worlds},
    )


def restore(runtime, checkpoint, expected_revision):
    from ._mujoco_tyx_runtime import MuJoCoTyxStepReceipt
    from .mujoco_runtime import MuJoCoControl, MuJoCoInitialState

    _require(
        type(expected_revision) is int and expected_revision == runtime._checkpoint_revision,
        "stale destination revision",
    )
    _require(type(checkpoint) is MuJoCoTyxCheckpoint, "expected MuJoCoTyxCheckpoint")
    binding = _model_binding(runtime)
    saved = checkpoint._read(binding)
    _require(
        isinstance(saved, Mapping) and set(saved) == {"q", "v", "clock", "worlds"}, "state fields"
    )
    nw = len(runtime.world_keys)
    packed = (saved["q"], saved["v"], saved["clock"])
    for value, shape in zip(
        packed, ((nw, binding["layout"][0]), (nw, binding["layout"][1]), (nw, 2)), strict=True
    ):
        _require(
            isinstance(value, np.ndarray)
            and value.dtype == np.float32
            and value.shape == shape
            and np.isfinite(value).all(),
            "raw bank dimensions",
        )
    _require(type(saved["worlds"]) is tuple and len(saved["worlds"]) == nw, "world count")
    data, controls, ticks, receipts, initial = {}, {}, {}, {}, {}
    for index, (key, item) in enumerate(zip(runtime.world_keys, saved["worlds"], strict=True)):
        _require(
            isinstance(item, Mapping)
            and set(item)
            == {"key", "tick", "time", "mirror", "control", "initial", "receipt", "cpu_state"}
            and type(item["key"]) is int
            and item["key"] == key,
            "world identity",
        )
        tick, clock = item["tick"], item["time"]
        _require(type(tick) is int and 0 <= tick <= np.iinfo(np.int32).max, "tick range")
        _require(type(clock) is float and math.isfinite(clock) and clock >= 0, "world time")
        _require(
            clock == float(packed[2][index, 0]) - float(packed[2][index, 1]),
            "clock component binding",
        )
        _require(
            type(item["control"]) is MuJoCoControl and type(item["initial"]) is MuJoCoInitialState,
            "control/reset record type",
        )
        control = runtime._validated_control(item["control"])
        for value in (control.ctrl, control.qfrc_applied, control.xfrc_applied):
            _require(np.isfinite(value.astype(np.float32)).all(), "control cannot be stored")
        # Only original reset declarations pass the unchanged FP64 initial gate.
        # The accepted checkpoint q is assigned below without this gate/repair.
        runtime._initial_data(item["initial"])
        candidate = runtime._mj.MjData(runtime._model)
        cpu_state = item["cpu_state"]
        _require(
            isinstance(cpu_state, np.ndarray)
            and cpu_state.dtype == np.float64
            and cpu_state.shape == (binding["cpu_state_size"],)
            and np.isfinite(cpu_state).all(),
            "CPU observer integration-state dimensions",
        )
        runtime._mj.mj_setState(
            runtime._model, candidate, cpu_state, runtime._mj.mjtState.mjSTATE_INTEGRATION
        )
        mirror = item["mirror"]
        _require(isinstance(mirror, Mapping) and set(mirror) == set(MIRROR_FIELDS), "mirror schema")
        for name in MIRROR_FIELDS:
            value, destination = mirror[name], getattr(candidate, name)
            _require(
                isinstance(value, np.ndarray)
                and value.dtype == np.float64
                and value.shape == destination.shape
                and np.isfinite(value).all(),
                "mirror dimensions: " + name,
            )
            destination[:] = value
        # The CPU integration packet and explicit published channels must agree.
        check = runtime._mj.MjData(runtime._model)
        runtime._mj.mj_setState(
            runtime._model, check, cpu_state, runtime._mj.mjtState.mjSTATE_INTEGRATION
        )
        for name in ("qpos", "qvel", "act", "ctrl", "qfrc_applied", "xfrc_applied"):
            _require(
                getattr(check, name).tobytes() == getattr(candidate, name).tobytes(),
                "CPU observer state/mirror binding: " + name,
            )
        _require(candidate.time == clock, "CPU observer clock binding")
        for actual, values, width in (
            (candidate.qpos, packed[0], runtime._plan.nq),
            (candidate.qvel, packed[1], runtime._plan.nv),
        ):
            _require(
                actual.tobytes() == values[index, :width].astype(np.float64).tobytes(),
                "raw state/mirror binding",
            )
        runtime._finite_state(candidate)
        receipt = item["receipt"]
        _require(
            (receipt is None and tick == 0)
            or (
                type(receipt) is MuJoCoTyxStepReceipt
                and receipt.world == key
                and receipt.tick == tick
                and receipt.output_time == clock
                and receipt.backend == "mujoco-tyx-" + runtime._mode
            ),
            "last receipt binding",
        )
        data[key], controls[key], ticks[key], receipts[key], initial[key] = (
            candidate,
            control,
            tick,
            receipt,
            item["initial"],
        )
    replacement = None
    try:
        replacement = runtime._build_device(data, packed=packed)
        _verify_bank(replacement, packed)
        # All observers are prepared on copies before the single publication.
        # mj_forward cannot supply q/v or physics to the restored device.
        import copy

        for candidate in data.values():
            observed = copy.copy(candidate)
            runtime._mj.mj_forward(runtime._model, observed)
            runtime._finite_state(observed)
    except BaseException:
        if replacement is not None:
            try:
                replacement.close()
            except Exception:
                pass
        raise
    old = runtime._device
    # Operational resource/launch counters are monotonic, not simulated state.
    for key, value in old.counts.items():
        replacement.counts[key] += value
    runtime._device = replacement
    runtime._data, runtime._controls, runtime._ticks, runtime._receipts, runtime._initial = (
        data,
        controls,
        ticks,
        receipts,
        initial,
    )
    runtime._checkpoint_revision += 1
    # A cleanup error cannot turn a completed atomic publication into a false
    # refusal. Retain the old owner for close() to retry releasing its resources.
    try:
        old.close()
    except Exception:
        runtime._checkpoint_retired.append(old)
