"""HOST codec boundaries with one real initial interpret physics capsule."""

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest
from test_mujoco_tyx_heterogeneous_io import declarations

from phymeo.backends import mujoco_heterogeneous_checkpoint as physics_codec
from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch
from phymeo.training import manipulation_checkpoint as codec


@pytest.fixture(scope="module")
def physics():
    specs, worlds = declarations((41,))
    with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="interpret") as batch:
        return batch.checkpoint()


def wire(metadata, physics_bytes):
    payload = codec.HEADER.pack(len(metadata), len(physics_bytes)) + metadata + physics_bytes
    return codec.MAGIC + hashlib.sha256(payload).digest() + payload


def envelope(state=None):
    return {
        "schema": codec.SCHEMA,
        "binding": {"world_keys": [41]},
        "state": {"episode": 0} if state is None else state,
    }


def encoded(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def test_roundtrip_detaches_metadata_and_owns_bytes(physics):
    binding, state = {"world_keys": (41,), "source": "原始模型"}, {"episode": 3, "history": [1, 2]}
    capsule = codec.capture_manipulation_checkpoint(binding, physics, state)
    restored = codec.ManipulationEnvCheckpoint.from_bytes(memoryview(capsule.to_bytes()))
    assert restored.to_bytes() == capsule.to_bytes()
    assert restored.sha256 == hashlib.sha256(capsule.to_bytes()).hexdigest()
    original, read = restored.read(binding)
    assert original.to_bytes() == physics.to_bytes() and read == state
    state["history"].append(99)
    read["history"].append(88)
    assert restored.read(binding)[1]["history"] == [1, 2]
    with pytest.raises(TypeError):
        restored.binding["world_keys"] = (7,)
    with pytest.raises(FrozenInstanceError):
        restored._bytes = b"changed"
    raw = bytearray(capsule.to_bytes())
    owned = codec.ManipulationEnvCheckpoint.from_bytes(raw)
    raw[-1] ^= 1
    assert owned.to_bytes() == capsule.to_bytes()
    with pytest.raises(ValueError, match="binding"):
        restored.read({"world_keys": [7], "source": "原始模型"})


@pytest.mark.parametrize(
    "case",
    (
        "header",
        "digest",
        "short",
        "trailing",
        "declared",
        "metadata",
        "duplicate",
        "nan",
        "physics",
        "schema",
        "extra",
        "depth",
        "items",
    ),
)
def test_corruption_and_limits_reject_without_physics_execution(physics, case):
    meta, physical = encoded(envelope()), physics.to_bytes()
    raw = wire(meta, physical)
    if case == "header":
        raw = b"X" + raw[1:]
    elif case == "digest":
        raw = raw[: len(codec.MAGIC)] + bytes(32) + raw[len(codec.MAGIC) + 32 :]
    elif case == "short":
        raw = raw[:-1]
    elif case == "trailing":
        raw += b"extra"
    elif case == "declared":
        payload = codec.HEADER.pack(codec.MAX_METADATA_BYTES + 1, len(physical)) + meta + physical
        raw = codec.MAGIC + hashlib.sha256(payload).digest() + payload
    elif case == "metadata":
        raw = wire(b"{", physical)
    elif case == "duplicate":
        meta = ('{"schema":"' + codec.SCHEMA + '","binding":{"x":1,"x":2},"state":{}}').encode()
        raw = wire(meta, physical)
    elif case == "nan":
        meta = ('{"schema":"' + codec.SCHEMA + '","binding":{},"state":{"x":NaN}}').encode()
        raw = wire(meta, physical)
    elif case == "physics":
        raw = wire(meta, b"not an original physics capsule")
    elif case == "schema":
        raw = wire(encoded({**envelope(), "schema": "unknown"}), physical)
    elif case == "extra":
        raw = wire(encoded({**envelope(), "extra": "not allowed"}), physical)
    elif case == "depth":
        deep = 0
        for _ in range(codec.MAX_DEPTH + 1):
            deep = [deep]
        raw = wire(encoded(envelope({"deep": deep})), physical)
    elif case == "items":
        raw = wire(encoded(envelope({"items": [0] * codec.MAX_ITEMS})), physical)
    with pytest.raises((ValueError, TypeError)):
        codec.ManipulationEnvCheckpoint.from_bytes(raw)


@pytest.mark.parametrize(
    "bad",
    (
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": 1 << 63},
        {1: "non-string key"},
        {"x": object()},
    ),
)
def test_capture_rejects_nonplain_metadata(physics, bad):
    with pytest.raises((ValueError, TypeError)):
        codec.capture_manipulation_checkpoint({}, physics, bad)


def test_metadata_size_is_bounded_and_typed_physics_is_required(physics):
    with pytest.raises(ValueError, match="size"):
        codec.capture_manipulation_checkpoint({}, physics, {"huge": "x" * codec.MAX_METADATA_BYTES})
    with pytest.raises(TypeError, match="typed physics"):
        codec.capture_manipulation_checkpoint({}, physics.to_bytes(), {})
    with pytest.raises(TypeError, match="mappings"):
        codec.capture_manipulation_checkpoint({}, physics, [])
    with pytest.raises(TypeError, match="bytes"):
        codec.ManipulationEnvCheckpoint.from_bytes("not bytes")


def test_original_leaf_integrity_is_checked_inside_valid_outer_envelope(physics):
    body = physics.to_bytes()[len(physics_codec.MAGIC) + 65 :]
    payload = json.loads(body)
    payload["leaves"][0]["sha256"] = "0" * 64
    changed = encoded(payload)
    raw = physics_codec.MAGIC + hashlib.sha256(changed).hexdigest().encode() + b"\n" + changed
    shallow = physics_codec.MuJoCoTyxHeterogeneousCheckpoint.from_bytes(raw)
    with pytest.raises(ValueError, match="leaf checksum"):
        codec.capture_manipulation_checkpoint({}, shallow, {})


def test_repeated_long_strings_reject_before_json_allocation(physics, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unbounded metadata reached JSON serialization")

    monkeypatch.setattr(codec.json, "dumps", forbidden)
    text = "x" * (codec.MAX_METADATA_BYTES // 2)
    with pytest.raises(ValueError, match="string size"):
        codec.capture_manipulation_checkpoint({}, physics, {"repeated": [text, text, text]})


@pytest.mark.parametrize("expected", ({"world": True}, {"world": 1.0}))
def test_binding_rejects_equal_numeric_values_of_different_json_types(physics, expected):
    capsule = codec.capture_manipulation_checkpoint({"world": 1}, physics, {})
    with pytest.raises(ValueError, match="binding"):
        capsule.read(expected)
