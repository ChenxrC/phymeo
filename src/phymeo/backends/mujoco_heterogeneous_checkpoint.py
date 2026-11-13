"""Bounded heterogeneous capsules composed of original exact TYX checkpoints."""

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .mujoco_checkpoint import MuJoCoTyxCheckpoint

MAGIC = b"PHYMEOTYXHET1\n"
MAX_BYTES = 128 * 1024 * 1024
MAX_BUCKETS = 32


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or type(value) in (str, int, bool, float):
        return value
    raise TypeError("heterogeneous checkpoint metadata must be plain data")


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _unique(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate heterogeneous checkpoint key")
        output[key] = value
    return output


def _invalid_constant(value):
    raise ValueError("nonfinite heterogeneous checkpoint metadata")


def _json(value):
    return json.dumps(
        _plain(value), sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()


@dataclass(frozen=True, init=False)
class MuJoCoTyxHeterogeneousCheckpoint:
    """Trusted application bytes; each leaf keeps its original physics schema."""

    _bytes: bytes

    def __init__(self, raw):
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise TypeError("heterogeneous checkpoint requires bytes")
        size = raw.nbytes if isinstance(raw, memoryview) else len(raw)
        if size > MAX_BYTES:
            raise ValueError("heterogeneous checkpoint exceeds 128 MiB")
        owned = bytes(raw)
        if not owned.startswith(MAGIC):
            raise ValueError("heterogeneous checkpoint header")
        digest, separator, body = owned[len(MAGIC) :].partition(b"\n")
        if not separator or digest != hashlib.sha256(body).hexdigest().encode():
            raise ValueError("heterogeneous checkpoint checksum")
        object.__setattr__(self, "_bytes", owned)
        self._payload()

    def _payload(self):
        body = self._bytes[len(MAGIC) + 65 :]
        value = json.loads(body, object_pairs_hook=_unique, parse_constant=_invalid_constant)
        if type(value) is not dict or set(value) != {
            "binding",
            "revision",
            "leaves",
            "publication",
        }:
            raise ValueError("heterogeneous checkpoint schema")
        if type(value["revision"]) is not int or value["revision"] < 0:
            raise ValueError("heterogeneous checkpoint revision")
        if type(value["binding"]) is not dict:
            raise ValueError("heterogeneous checkpoint binding")
        leaves = value["leaves"]
        if type(leaves) is not list or not 1 <= len(leaves) <= MAX_BUCKETS:
            raise ValueError("heterogeneous checkpoint leaf count")
        for leaf in leaves:
            if (
                type(leaf) is not dict
                or set(leaf) != {"sha256", "bytes"}
                or type(leaf["bytes"]) is not str
            ):
                raise ValueError("heterogeneous checkpoint leaf schema")
        publication = value["publication"]
        if publication is not None:
            if type(publication) is not dict or set(publication) != {
                "schema",
                "revision_before",
                "revision_after",
                "bucket_revisions",
                "world_keys",
            }:
                raise ValueError("heterogeneous checkpoint publication schema")
            if publication["schema"] != "mujoco-tyx-heterogeneous-step-v1":
                raise ValueError("heterogeneous checkpoint publication kind")
            before, after = publication["revision_before"], publication["revision_after"]
            if (
                type(before) is not int
                or type(after) is not int
                or before < 0
                or after != before + 1
                or after > value["revision"]
            ):
                raise ValueError("heterogeneous checkpoint publication revision")
            if publication["world_keys"] != value["binding"].get("world_keys"):
                raise ValueError("heterogeneous checkpoint publication world binding")
            revisions = publication["bucket_revisions"]
            if type(revisions) is not list or len(revisions) != len(leaves):
                raise ValueError("heterogeneous checkpoint publication bucket count")
            if any(
                type(row) is not list
                or len(row) != 2
                or any(type(x) is not int or x < 0 for x in row)
                or row[1] != row[0] + 1
                for row in revisions
            ):
                raise ValueError("heterogeneous checkpoint publication bucket revision")
        return value

    @classmethod
    def from_bytes(cls, raw):
        return cls(raw)

    def to_bytes(self):
        return self._bytes

    @property
    def sha256(self):
        return hashlib.sha256(self._bytes).hexdigest()

    @property
    def source_revision(self):
        return self._payload()["revision"]

    @property
    def binding(self):
        return _freeze(self._payload()["binding"])

    def _read(self, expected_binding):
        value = self._payload()
        if value["binding"] != _plain(expected_binding):
            raise ValueError("heterogeneous checkpoint source/world/layout binding")
        leaves = []
        for item in value["leaves"]:
            raw = base64.b64decode(item["bytes"], validate=True)
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("heterogeneous checkpoint leaf checksum")
            leaves.append(MuJoCoTyxCheckpoint.from_bytes(raw))
        return tuple(leaves), _freeze(value["publication"])


def _capture(binding, revision, leaves, publication):
    if not 1 <= len(leaves) <= MAX_BUCKETS:
        raise ValueError("heterogeneous checkpoint leaf count")
    rows, total = [], 0
    for leaf in leaves:
        if type(leaf) is not MuJoCoTyxCheckpoint:
            raise TypeError("heterogeneous checkpoint requires original typed leaves")
        raw = leaf.to_bytes()
        total += 4 * ((len(raw) + 2) // 3)
        if total > MAX_BYTES:
            raise ValueError("heterogeneous checkpoint encoded leaves exceed size limit")
        rows.append(
            {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": base64.b64encode(raw).decode("ascii"),
            }
        )
    body = _json(dict(binding=binding, revision=revision, leaves=rows, publication=publication))
    if len(MAGIC) + 65 + len(body) > MAX_BYTES:
        raise ValueError("heterogeneous checkpoint exceeds size limit")
    return MuJoCoTyxHeterogeneousCheckpoint(
        MAGIC + hashlib.sha256(body).hexdigest().encode() + b"\n" + body
    )
