"""HOST-only entry codec boundaries; authored without project execution.

These synthetic seven-field records exercise layout and bounded serialization,
not accepted physics ticks, receipts, larger GPU worlds, or new solver capacity.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from phymeo.backends import mujoco_checkpoint as capsule
from phymeo.backends import mujoco_entry_checkpoint as entry
from phymeo.backends.mujoco_tyx._entry_storage import entry_layout

CHUNK_SCHEMA = "mujoco-tyx-entry-chunks-v1"
FIELDS = {
    "entry_q",
    "entry_v",
    "entry_act",
    "entry_clock",
    "entry_meta",
    "entry_contact_ids",
    "entry_contact_data",
}


def sample(world_count, contact_capacity):
    owner = SimpleNamespace(
        _plan=SimpleNamespace(kind=0, nq=1, nv=1, pair=None),
        world_keys=tuple(range(world_count)),
        _contact_capacity=contact_capacity,
        _constraint_capacity=256,
    )
    layout = entry_layout(owner._plan, world_count, (contact_capacity, 256))
    values = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in layout.items()}
    assert set(values) == FIELDS
    for index, value in enumerate(values.values(), start=1):
        value.flat[0], value.flat[-1] = index, -index
    data = values["entry_contact_data"].reshape(-1)
    if data.size > capsule.MAX_ARRAY_ITEMS:
        # Distinct boundary markers expose reordering, dropped tails and bad reshapes.
        data[capsule.MAX_ARRAY_ITEMS - 1] = 1.25
        data[capsule.MAX_ARRAY_ITEMS] = -2.5
        data[-1] = 3.75
    return owner, values


def assert_same(actual, expected):
    assert set(actual) == set(expected) == FIELDS
    for name in FIELDS:
        assert actual[name].shape == expected[name].shape, name
        assert actual[name].dtype == expected[name].dtype, name
        assert actual[name].tobytes() == expected[name].tobytes(), name


@pytest.mark.parametrize("world_count,contact_capacity", ((1, 64), (1, 65536), (256, 256)))
def test_entry_chunks_roundtrip_through_original_bounded_checkpoint_codec(
    world_count,
    contact_capacity,
):
    assert capsule.MAX_ARRAY_ITEMS == 1_048_576
    assert capsule.MAX_BYTES == 128 * 1024 * 1024
    owner, values = sample(world_count, contact_capacity)
    packed = entry._pack_entry(values)
    assert set(packed) == FIELDS
    assert values["entry_contact_data"].shape == (world_count, contact_capacity, 20)
    assert values["entry_contact_ids"].shape == (world_count, contact_capacity, 8)
    for name, original in values.items():
        if original.size <= capsule.MAX_ARRAY_ITEMS:
            assert packed[name] is original  # Existing small-array representation is unchanged.
        else:
            envelope = packed[name]
            assert set(envelope) == {"schema", "shape", "dtype", "chunks"}
            assert envelope["schema"] == CHUNK_SCHEMA
            assert type(envelope["shape"]) is tuple and envelope["shape"] == original.shape
            assert envelope["dtype"] == original.dtype.str
            chunks = envelope["chunks"]
            assert type(chunks) is tuple
            assert len(chunks) == 2
            assert tuple(chunk.size for chunk in chunks) == (1_048_576, 262_144)
            flat = original.reshape(-1)
            for index, chunk in enumerate(chunks):
                assert chunk.ndim == 1 and chunk.dtype == original.dtype
                assert 0 < chunk.size <= capsule.MAX_ARRAY_ITEMS
                start = index * capsule.MAX_ARRAY_ITEMS
                assert chunk.tobytes() == flat[start : start + chunk.size].tobytes()
            # Generic checkpoint arrays retain their original cap; only this
            # entry extension supplies independently bounded chunks.
            with pytest.raises(ValueError, match="array"):
                capsule._encode(original)
    binding = {
        "schema": "entry-codec-host-test",
        "worlds": world_count,
        "capacity": contact_capacity,
    }
    cp = capsule._make_checkpoint(binding, 17, {"accepted_entry": packed})
    wire = cp.to_bytes()
    assert len(wire) <= capsule.MAX_BYTES
    decoded_capsule = capsule.MuJoCoTyxCheckpoint.from_bytes(wire)
    assert decoded_capsule.source_revision == 17
    saved = decoded_capsule._read(binding)["accepted_entry"]
    assert_same(entry._unpack_entry(owner, saved), values)
    assert_same(entry._unpack_entry(owner, packed), values)


def test_malformed_large_chunk_envelopes_reject_before_concatenation(monkeypatch):
    owner, values = sample(1, 65536)
    packed = entry._pack_entry(values)
    envelope = packed["entry_contact_data"]
    chunks = envelope["chunks"]
    nonfinite = chunks[-1].copy()
    nonfinite[-1] = np.nan
    # One large record is reused; malformed blocks are views except the one
    # finite-data negative. This does not construct any device or native model.
    malformed = {
        "schema": envelope | {"schema": "mujoco-tyx-entry-chunks-v0"},
        "shape": envelope | {"shape": (1, 32768, 40)},
        "shape-type": envelope | {"shape": (True, 65536, 20)},
        "dtype": envelope | {"dtype": np.dtype(np.float64).str},
        "missing-field": {key: value for key, value in envelope.items() if key != "dtype"},
        "extra-field": envelope | {"extra": 0},
        "missing-chunk": envelope | {"chunks": chunks[:-1]},
        "extra-chunk": envelope | {"chunks": (*chunks, np.empty(0, np.float32))},
        "chunks-type": envelope | {"chunks": list(chunks)},
        "first-length": envelope | {"chunks": (chunks[0][:-1], chunks[1])},
        "tail-length": envelope | {"chunks": (chunks[0], chunks[1][:-1])},
        "chunk-rank": envelope | {"chunks": (chunks[0].reshape(1024, 1024), chunks[1])},
        "chunk-dtype": envelope | {"chunks": (chunks[0], chunks[1].view(np.int32))},
        "nonfinite-tail": envelope | {"chunks": (chunks[0], nonfinite)},
    }
    with monkeypatch.context() as patch:
        patch.setattr(
            entry.np,
            "concatenate",
            lambda *a, **kw: pytest.fail("malformed chunk record reached concatenation"),
        )
        for label, bad in malformed.items():
            with pytest.raises(ValueError, match="entry"):
                entry._unpack_entry(owner, packed | {"entry_contact_data": bad})
            assert envelope["schema"] == CHUNK_SCHEMA, label
    assert_same(entry._unpack_entry(owner, packed), values)


def test_small_chunk_envelopes_and_invalid_seven_field_layouts_reject(monkeypatch):
    owner, values = sample(1, 4)
    packed = entry._pack_entry(values)
    small = values["entry_q"]
    small_envelope = {
        "schema": CHUNK_SCHEMA,
        "shape": small.shape,
        "dtype": small.dtype.str,
        "chunks": (small.reshape(-1),),
    }
    nonfinite = values["entry_clock"].copy()
    nonfinite[0, 0] = np.inf
    malformed = {
        "small-envelope": packed | {"entry_q": small_envelope},
        "missing-field": {key: value for key, value in packed.items() if key != "entry_act"},
        "extra-field": packed | {"unexpected": np.zeros(1, np.float32)},
        "small-shape": packed | {"entry_q": small.reshape(2, 4)},
        "small-dtype": packed | {"entry_q": small.astype(np.float64)},
        "meta-dtype": packed | {"entry_meta": values["entry_meta"].astype(np.float32)},
        "nonfinite-small": packed | {"entry_clock": nonfinite},
    }
    with monkeypatch.context() as patch:
        patch.setattr(
            entry.np,
            "concatenate",
            lambda *a, **kw: pytest.fail("small or malformed record must not concatenate chunks"),
        )
        assert_same(entry._unpack_entry(owner, packed), values)
        for label, bad in malformed.items():
            with pytest.raises(ValueError, match="entry"):
                entry._unpack_entry(owner, bad)
            assert set(packed) == FIELDS, label
