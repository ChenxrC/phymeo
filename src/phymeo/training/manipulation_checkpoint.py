"""Bounded, non-pickle composition of manipulation metadata and exact physics.

This is a transport/integrity boundary. The Env owns semantic validation of
goal/history/cursor and commits metadata only after physical restore succeeds.
"""

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..backends.mujoco_heterogeneous_checkpoint import (
    MAX_BYTES as MAX_PHYSICS_BYTES,
)
from ..backends.mujoco_heterogeneous_checkpoint import (
    MuJoCoTyxHeterogeneousCheckpoint,
)

MAGIC = b"PHYMEOMANIP1\n"
HEADER = struct.Struct("<IQ")
MAX_METADATA_BYTES = 1024 * 1024
MAX_ITEMS = 65536
MAX_DEPTH = 32
MAX_BYTES = len(MAGIC) + 32 + HEADER.size + MAX_METADATA_BYTES + MAX_PHYSICS_BYTES
SCHEMA = "phymeo-manipulation-env-checkpoint-v1"


def _plain(value, budget=None, depth=0):
    if budget is None:
        budget = [MAX_ITEMS, MAX_METADATA_BYTES]
    budget[0] -= 1
    if budget[0] < 0 or depth > MAX_DEPTH:
        raise ValueError("manipulation checkpoint metadata depth/items")
    if isinstance(value, Mapping):
        if any(type(k) is not str for k in value):
            raise TypeError("manipulation checkpoint metadata keys must be strings")
        for key in value:
            budget[1] -= len(key)
        if budget[1] < 0:
            raise ValueError("manipulation checkpoint metadata string size")
        return {k: _plain(v, budget, depth + 1) for k, v in value.items()}
    if type(value) in (tuple, list):
        return [_plain(v, budget, depth + 1) for v in value]
    if type(value) is int:
        if not -(1 << 63) <= value < (1 << 63):
            raise ValueError("manipulation checkpoint metadata integer range")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("manipulation checkpoint nonfinite metadata")
        return value
    if type(value) is str:
        budget[1] -= len(value)
        if budget[1] < 0:
            raise ValueError("manipulation checkpoint metadata string size")
        return value
    if value is None or type(value) is bool:
        return value
    raise TypeError("manipulation checkpoint requires plain JSON metadata")


def _freeze(value):
    if type(value) is dict:
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if type(value) is list:
        return tuple(_freeze(v) for v in value)
    return value


def _unique(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate manipulation checkpoint key")
        output[key] = value
    return output


def _invalid_constant(value):
    raise ValueError("manipulation checkpoint nonfinite metadata")


def _decode(raw):
    offset = len(MAGIC)
    if len(raw) < offset + 32 + HEADER.size or not raw.startswith(MAGIC):
        raise ValueError("manipulation checkpoint header")
    digest = raw[offset : offset + 32]
    payload = memoryview(raw)[offset + 32 :]
    metadata_size, physics_size = HEADER.unpack_from(payload)
    if not 0 < metadata_size <= MAX_METADATA_BYTES or not 0 < physics_size <= MAX_PHYSICS_BYTES:
        raise ValueError("manipulation checkpoint declared size")
    if len(payload) != HEADER.size + metadata_size + physics_size:
        raise ValueError("manipulation checkpoint length/trailing bytes")
    if hashlib.sha256(payload).digest() != digest:
        raise ValueError("manipulation checkpoint checksum")
    end = HEADER.size + metadata_size
    try:
        metadata = json.loads(
            bytes(payload[HEADER.size : end]),
            object_pairs_hook=_unique,
            parse_constant=_invalid_constant,
        )
    except RecursionError as error:
        raise ValueError("manipulation checkpoint metadata depth") from error
    metadata = _plain(metadata)
    if type(metadata) is not dict or set(metadata) != {"schema", "binding", "state"}:
        raise ValueError("manipulation checkpoint envelope schema")
    if (
        metadata["schema"] != SCHEMA
        or type(metadata["binding"]) is not dict
        or type(metadata["state"]) is not dict
    ):
        raise ValueError("manipulation checkpoint schema/binding/state")
    physics = MuJoCoTyxHeterogeneousCheckpoint.from_bytes(payload[end:])
    # Reuse the bounded leaf validator, never a runtime, solver or state setter.
    physics._read(physics.binding)
    return metadata, physics


@dataclass(frozen=True, init=False)
class ManipulationEnvCheckpoint:
    """Owned immutable bytes; corruption checks do not certify task success."""

    _bytes: bytes

    def __init__(self, raw):
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise TypeError("manipulation checkpoint requires bytes")
        size = raw.nbytes if isinstance(raw, memoryview) else len(raw)
        if size > MAX_BYTES:
            raise ValueError("manipulation checkpoint exceeds bounded size")
        owned = bytes(raw)
        _decode(owned)
        object.__setattr__(self, "_bytes", owned)

    @classmethod
    def from_bytes(cls, raw):
        return cls(raw)

    def to_bytes(self):
        return self._bytes

    @property
    def sha256(self):
        return hashlib.sha256(self._bytes).hexdigest()

    @property
    def binding(self):
        return _freeze(_decode(self._bytes)[0]["binding"])

    def read(self, binding):
        """Return an original typed physics capsule and detached JSON metadata."""
        metadata, physics = _decode(self._bytes)
        if metadata["binding"] != _plain(binding):
            raise ValueError("manipulation checkpoint source/action/world binding")
        return physics, metadata["state"]


def capture_manipulation_checkpoint(binding, physics, state):
    if type(physics) is not MuJoCoTyxHeterogeneousCheckpoint:
        raise TypeError("manipulation checkpoint requires original typed physics")
    if not isinstance(binding, Mapping) or not isinstance(state, Mapping):
        raise TypeError("manipulation checkpoint binding/state require mappings")
    metadata = json.dumps(
        _plain(dict(schema=SCHEMA, binding=binding, state=state)),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    original = physics.to_bytes()
    if not 0 < len(metadata) <= MAX_METADATA_BYTES:
        raise ValueError("manipulation checkpoint metadata size")
    if not 0 < len(original) <= MAX_PHYSICS_BYTES:
        raise ValueError("manipulation checkpoint physics size")
    payload = HEADER.pack(len(metadata), len(original)) + metadata + original
    return ManipulationEnvCheckpoint(MAGIC + hashlib.sha256(payload).digest() + payload)
