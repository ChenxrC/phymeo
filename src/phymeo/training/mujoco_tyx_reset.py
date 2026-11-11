"""Versioned scalar q/v reset plans; no global RNG or physical state writes.

Sampling is HOST SHA256 counter mapping followed by explicit FP64 arithmetic and
one FP32 rounding. Physics admission and publication belong to reset_states().
This module neither changes the model nor performs a physical step.
"""

import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

import numpy as np

from phymeo.backends.mujoco_runtime import MuJoCoInitialState

SCHEMA = "mujoco-tyx-random-reset-v1"
ALGORITHM = "sha256-le53-f64-mul-add-rn32-v1"
MAX_INDEX = 2**64 - 1


def _uint(value, bits, name):
    if type(value) is not int or not 0 <= value < 2**bits:
        raise ValueError(name + " must be an unsigned " + str(bits) + "-bit integer")
    return value


def _world(value):
    return _uint(value, 31, "world key")


def encoded_reset_key(namespace, seed, world, index, channel, native_index):
    """Public byte protocol. No world-set/order, model hash or caller RNG enters."""
    if type(namespace) is not str:
        raise TypeError("stream namespace must be a string")
    encoded = namespace.encode("utf-8")
    if not 1 <= len(encoded) <= 128:
        raise ValueError("stream namespace must encode 1..128 UTF8 bytes")
    if channel not in ("q", "v"):
        raise ValueError("reset channel must be q or v")
    return (
        b"phymeo-reset-v1\0"
        + struct.pack("<H", len(encoded))
        + encoded
        + struct.pack("<QIQ", _uint(seed, 64, "seed"), _world(world),
                      _uint(index, 64, "reset index"))
        + channel.encode("ascii")
        + struct.pack("<I", _uint(native_index, 32, "native index"))
    )


def _frozen(value, dtype):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _source_binding(compilation):
    source = compilation.source
    return {
        "model_sha256": sha256(source.model_binary).hexdigest(),
        "archive_sha256": sha256(source.archive_bytes).hexdigest(),
        "mujoco_version": source.mujoco_version,
    }


def _bounds(indices, bounds, legal, name):
    values = tuple(indices)
    if (len(values) > 35 or any(type(i) is not int or i not in legal for i in values)
            or len(set(values)) != len(values)):
        raise ValueError(name + " requires distinct supported scalar hinge/slide addresses")
    array = np.asarray(bounds, dtype=np.float64)
    if not values and array.size == 0:
        array = array.reshape(0, 2)
    if array.shape != (len(values), 2) or not np.isfinite(array).all():
        raise ValueError(name + " requires finite [lower,upper] per native address")
    if np.any(array[:, 0] > array[:, 1]):
        raise ValueError(name + " bounds are reversed")
    with np.errstate(over="ignore", invalid="ignore"):
        if not np.isfinite(array.astype(np.float32)).all():
            raise ValueError(name + " bounds must fit finite FP32")
    return values, _frozen(array, np.float64)


@dataclass(frozen=True)
class MuJoCoTyxResetSample:
    plan_sha256: str
    world_keys: tuple
    indices: object
    requested: object
    next_indices: object


@dataclass(frozen=True, init=False)
class MuJoCoTyxResetPlan:
    """Immutable anchors and scalar-coordinate uniform bounds in native units."""

    world_keys: tuple
    anchors: object
    q_indices: tuple
    q_bounds: np.ndarray
    v_indices: tuple
    v_bounds: np.ndarray
    seed: int
    namespace: str
    binding: object
    sha256: str

    def __init__(self, compilation, anchors, *, q_indices, q_bounds, v_indices,
                 v_bounds, seed=6511, namespace="N5"):
        if not isinstance(anchors, Mapping) or not 1 <= len(anchors) <= 256:
            raise ValueError("reset anchors require 1..256 stable worlds")
        keys = tuple(_world(key) for key in anchors)
        encoded_reset_key(namespace, seed, keys[0], 0, "q", 0)
        model = compilation.fresh_native_model()
        nq, nv, na = int(model.nq), int(model.nv), int(model.na)
        if not 1 <= nq <= 36 or not 1 <= nv <= 35:
            raise ValueError("reset plan exceeds current native q/v layout")
        scalar = [j for j in range(int(model.njnt)) if int(model.jnt_type[j]) in (2, 3)]
        q, qb = _bounds(q_indices, q_bounds, {int(model.jnt_qposadr[j]) for j in scalar}, "q")
        v, vb = _bounds(v_indices, v_bounds, {int(model.jnt_dofadr[j]) for j in scalar}, "v")
        if not q and not v:
            raise ValueError("random reset requires at least one scalar coordinate")
        owned = {}
        for key in keys:
            state = anchors[key]
            if type(state) is not MuJoCoInitialState or state.time != 0:
                raise ValueError("reset anchor requires MuJoCoInitialState at time zero")
            arrays = []
            for name, size, default in (
                ("q", nq, model.qpos0), ("v", nv, np.zeros(nv)), ("act", na, np.zeros(na))
            ):
                value = getattr(state, name)
                array = np.asarray(default if value is None else value, np.float64)
                if array.shape != (size,) or not np.isfinite(array).all():
                    raise ValueError("reset anchor " + name + " shape/finite values")
                with np.errstate(over="ignore", invalid="ignore"):
                    if not np.isfinite(array.astype(np.float32)).all():
                        raise ValueError("reset anchor cannot be stored in FP32")
                arrays.append(array)
            # Preserve full legal anchor precision, in particular free quaternion.
            owned[key] = MuJoCoInitialState(arrays[0], arrays[1], act=arrays[2])
        identity = {
            "schema": SCHEMA, "algorithm": ALGORITHM,
            **_source_binding(compilation),
            "world_keys": keys, "nq": nq, "nv": nv, "na": na,
            "q_indices": q, "v_indices": v, "seed": seed, "namespace": namespace,
            "joint_type": tuple(int(i) for i in model.jnt_type),
            "joint_qadr": tuple(int(i) for i in model.jnt_qposadr),
            "joint_dadr": tuple(int(i) for i in model.jnt_dofadr),
        }
        digest = sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
        digest.update(qb.tobytes())
        digest.update(vb.tobytes())
        for state in owned.values():
            for name in ("q", "v", "act"):
                digest.update(getattr(state, name).tobytes())
        for name, value in (
            ("world_keys", keys), ("anchors", MappingProxyType(owned)),
            ("q_indices", q), ("q_bounds", qb), ("v_indices", v), ("v_bounds", vb),
            ("seed", seed), ("namespace", namespace),
            ("binding", MappingProxyType(identity)), ("sha256", digest.hexdigest()),
        ):
            object.__setattr__(self, name, value)

    def validate_runtime(self, runtime):
        """Only public immutable source/layout/order metadata; no physical readback."""
        if tuple(runtime.world_keys) != self.world_keys:
            raise ValueError("random reset world order binding differs")
        if not callable(getattr(runtime, "reset_states", None)):
            raise TypeError("runtime requires public reset_states")
        for key, value in _source_binding(runtime.compilation).items():
            if value != self.binding[key]:
                raise ValueError("random reset source binding differs")
        for name in ("nq", "nv", "na"):
            # na may be omitted by older generic capability dictionaries.
            if name in runtime.capabilities and runtime.capabilities[name] != self.binding[name]:
                raise ValueError("random reset native layout binding differs")

    def validate_indices(self, counters):
        if not isinstance(counters, Mapping) or set(counters) != set(self.world_keys):
            raise ValueError("reset counters must cover exact stable worlds")
        if any(type(key) is not int for key in counters):
            raise ValueError("reset counter key type")
        return MappingProxyType({
            key: _uint(counters[key], 64, "next reset index") for key in self.world_keys
        })

    def prepare(self, selected, counters):
        """Pure candidate sampling: no counters advance until the caller commits."""
        supplied = tuple(selected)
        if any(type(key) is not int for key in supplied):
            raise ValueError("selected world keys must be integers")
        if len(set(supplied)) != len(supplied) or not set(supplied) <= set(self.world_keys):
            raise ValueError("selected reset worlds must be unique plan members")
        before = self.validate_indices(counters)
        keys = tuple(key for key in self.world_keys if key in supplied)
        after = dict(before)
        requested, indices = {}, {}
        for key in keys:
            index = before[key]
            if index == MAX_INDEX:
                raise OverflowError("reset counter cannot advance beyond u64")
            arrays = {name: getattr(self.anchors[key], name).copy() for name in ("q", "v")}
            for name in ("q", "v"):
                for native, limits in zip(
                    getattr(self, name + "_indices"), getattr(self, name + "_bounds"), strict=True
                ):
                    raw = sha256(encoded_reset_key(
                        self.namespace, self.seed, key, index, name, native
                    )).digest()
                    z = int.from_bytes(raw[:8], "little") >> 11
                    unit = z * 2.0 ** -53
                    delta = float(limits[1]) - float(limits[0])
                    scaled = unit * delta
                    value = np.float32(float(limits[0]) + scaled)
                    if not np.isfinite(value) or (
                        value != 0 and abs(float(value)) < float(np.finfo(np.float32).tiny)
                    ):
                        raise ValueError("sample cannot be stored as finite normal-or-zero FP32")
                    arrays[name][native] = float(value)
            requested[key] = MuJoCoInitialState(
                arrays["q"], arrays["v"], act=self.anchors[key].act
            )
            indices[key], after[key] = index, index + 1
        return MuJoCoTyxResetSample(
            self.sha256, keys, MappingProxyType(indices),
            MappingProxyType(requested), MappingProxyType(after),
        )

    def training_binding(self):
        return {"schema": SCHEMA, "algorithm": ALGORITHM, "plan_sha256": self.sha256,
                "seed": self.seed, "namespace": self.namespace}

    def validate_training_state(self, state):
        """Validate saved stream state before any public physics CAS."""
        if type(state) is not dict or set(state) != {"plan_sha256", "next_indices", "last_sample"}:
            raise ValueError("random reset state schema")
        if state["plan_sha256"] != self.sha256:
            raise ValueError("random reset plan binding differs")
        counters = self.validate_indices(state["next_indices"])
        last = state["last_sample"]
        if last is not None:
            if type(last) is not dict or set(last) != {"world_keys", "indices"}:
                raise ValueError("random reset last sample schema")
            keys = last["world_keys"]
            if type(keys) is not tuple or not keys or any(type(k) is not int for k in keys):
                raise ValueError("random reset last sample world layout")
            if keys != tuple(key for key in self.world_keys if key in keys):
                raise ValueError("random reset last sample order")
            indices = last["indices"]
            if (not isinstance(indices, Mapping) or set(indices) != set(keys)
                    or any(type(key) is not int for key in indices)):
                raise ValueError("random reset last sample indices")
            copied = {}
            for key in keys:
                value = _uint(indices[key], 64, "last reset index")
                if value + 1 != counters[key]:
                    raise ValueError("random reset last sample not bound to accepted counter")
                copied[key] = value
            last = {"world_keys": keys, "indices": copied}
        elif any(counters.values()):
            raise ValueError("nonzero reset counters require a last commit identity")
        return counters, last
