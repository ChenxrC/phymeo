"""Native column equivalence, escaped-view leases and bounded reuse contracts."""

import copy
import gc
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from phymeo.backends.mujoco_contact_pool import (
    ContactPoolCapacityError,
    MuJoCoContactPool,
    PooledMuJoCoContacts,
    retained_contact_bytes,
)
from phymeo.backends.mujoco_recording import CONTACT_ROW_BYTES, contacts_to_columns
from phymeo.backends.mujoco_runtime import MuJoCoRuntime, _contacts
from phymeo.training.resident_stepper import PhysicsStepFailure

XML = """<mujoco><option timestep="0.001"/>
<worldbody><geom type="plane" size="1 1 .1"/>
<body pos="0 0 .099"><freejoint/><geom type="sphere" size=".1" mass=".5"/>
</body></worldbody></mujoco>"""


@pytest.fixture
def native():
    mj = pytest.importorskip("mujoco")
    m = mj.MjModel.from_xml_string(XML)
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    assert d.ncon == 1
    return mj, m, d


def same(a, b):
    assert a.count == b.count and a.arrays.keys() == b.arrays.keys()
    for name in a.arrays:
        x, y = a.arrays[name], b.arrays[name]
        assert x.dtype == y.dtype and x.shape == y.shape
        assert x.tobytes() == y.tobytes(), name


def test_native_all_fields_bit_exact_including_hessian_nan_payload(native):
    mj, m, d = native
    d.contact.H.view(np.uint64)[0, 0] = 0x7FF8000000000007
    # Scalars have the same permissive semantics as the legacy runtime.
    d.contact.mu[0] = np.nan
    pool = MuJoCoContactPool(block_rows=4)
    packet = pool.capture_native(mj, m, d)
    same(packet, contacts_to_columns(_contacts(mj, m, d)))
    assert packet[0].position.tobytes() == packet.arrays["position"][0].tobytes()
    assert packet[-1].index == 0 and len(packet[:]) == 1
    with pytest.raises(IndexError):
        packet[1]
    with pytest.raises(TypeError):
        packet[True]


@pytest.mark.parametrize(
    "escape",
    [
        lambda x: x,
        lambda x: x[:],
        lambda x: np.asarray(x),
        lambda x: x.view(np.ndarray),
        lambda x: memoryview(x),
    ],
)
def test_escaped_view_pins_slab_and_prevents_reuse(native, escape):
    mj, m, d = native
    pool = MuJoCoContactPool(max_bytes=CONTACT_ROW_BYTES, block_rows=1)
    packet = pool.capture_native(mj, m, d)
    array = packet.arrays["distance"]
    with pytest.raises(ValueError):
        array.flags.writeable = True
    held = escape(array)
    expected = held.tobytes()
    del packet, array
    pool.flush()
    gc.collect()
    with pytest.raises(ContactPoolCapacityError):
        pool.capture_native(mj, m, d)
    assert held.tobytes() == expected
    del held
    gc.collect()
    d.contact.dist[0] = -0.02
    new = pool.capture_native(mj, m, d)
    assert new.arrays["distance"][0] == -0.02
    assert pool.stats["reuses"] == 1 and pool.stats["allocations"] == 1


def test_nonoverlapping_suballocations_and_reuse_plateau(native):
    mj, m, d = native
    pool = MuJoCoContactPool(max_bytes=CONTACT_ROW_BYTES * 4, block_rows=2)
    first = pool.capture_native(mj, m, d)
    expected = first.arrays["distance"].tobytes()
    for i in range(100):
        d.contact.dist[0] = -float(i)
        other = pool.capture_native(mj, m, d)
        assert first.arrays["distance"].tobytes() == expected
        del other
    assert pool.stats["allocations"] == 2 and pool.stats["reuses"] > 10
    assert pool.stats["peak_bytes"] <= CONTACT_ROW_BYTES * 4


def test_writer_charge_counts_whole_shared_slab_once(native):
    mj, m, d = native
    pool = MuJoCoContactPool(block_rows=8)
    first = pool.capture_native(mj, m, d)
    second = pool.capture_native(mj, m, d)
    assert retained_contact_bytes([first, second, first]) == CONTACT_ROW_BYTES * 8
    detached = first.detach()
    assert retained_contact_bytes([detached]) == CONTACT_ROW_BYTES
    assert retained_contact_bytes([first, detached]) == CONTACT_ROW_BYTES * 9


def test_detach_and_close_preserve_exported_arrays(native):
    mj, m, d = native
    pool = MuJoCoContactPool(max_bytes=CONTACT_ROW_BYTES, block_rows=1)
    packet = pool.capture_native(mj, m, d)
    detached = packet.detach()
    pool.close()
    mj.mj_step(m, d)
    same(packet, detached)
    del packet
    gc.collect()
    assert pool.stats["allocated_bytes"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        pool.capture_native(mj, m, d)


def test_delayed_writer_owns_columns_after_producer_drops_packet(native):
    mj, m, d = native
    pool = MuJoCoContactPool(max_bytes=CONTACT_ROW_BYTES * 2, block_rows=1)
    packet = pool.capture_native(mj, m, d)
    expected = packet.arrays["position"].tobytes()
    gate = Event()

    def consume(array):
        assert gate.wait(5)
        return array.tobytes()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(consume, packet.arrays["position"])
        del packet
        pool.flush()
        for _ in range(20):
            other = pool.capture_native(mj, m, d)
            del other
        gate.set()
        assert future.result() == expected
    assert pool.stats["allocations"] == 2


@pytest.mark.parametrize("field", ["pos", "friction", "solref", "frame"])
def test_invalid_native_array_not_published_and_retry_uses_same_reservation(native, field):
    mj, m, d = native
    pool = MuJoCoContactPool(max_bytes=CONTACT_ROW_BYTES, block_rows=1)
    source = getattr(d.contact, field)
    original = source.copy()
    source.flat[0] = np.inf
    with pytest.raises(RuntimeError, match="nonfinite"):
        pool.capture_native(mj, m, d)
    assert pool.stats["captures"] == pool.stats["rows"] == 0
    source[:] = original
    same(pool.capture_native(mj, m, d), contacts_to_columns(_contacts(mj, m, d)))
    assert pool.stats["allocations"] == 1


def test_schema_validation_and_empty_capture(native):
    mj, m, d = native
    pool = MuJoCoContactPool()
    fields = {
        name: getattr(d.contact, name)
        for name in (
            "geom",
            "flex",
            "elem",
            "vert",
            "dim",
            "dist",
            "pos",
            "frame",
            "includemargin",
            "friction",
            "solref",
            "solreffriction",
            "solimp",
            "mu",
            "H",
            "exclude",
            "efc_address",
        )
    }
    fields["pos"] = np.zeros((1, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="schema"):
        pool.capture_native(mj, m, SimpleNamespace(ncon=1, contact=SimpleNamespace(**fields)))
    empty = pool.capture_native(mj, m, SimpleNamespace(ncon=0))
    assert len(empty) == 0 and empty.nbytes == 0
    assert len(empty.arrays) == 19 and empty.arrays["frame"].shape == (0, 3, 3)


def test_runtime_default_and_opt_in_preserve_physics_phases_reset_and_clone(native):
    with (
        MuJoCoRuntime.from_xml(XML) as old,
        MuJoCoRuntime.from_xml(XML, contact_storage="columns", contact_block_rows=8) as new,
    ):
        assert old.contact_pool_stats is None
        retained = new.snapshot()[0]
        frozen = retained.contacts.detach()
        for _ in range(4):
            old.step(0.001)
            new.step(0.001)
            a, b = old.snapshot()[0], new.snapshot()[0]
            np.testing.assert_array_equal(a.q, b.q)
            np.testing.assert_array_equal(a.v, b.v)
            assert isinstance(a.contacts, tuple)
            assert isinstance(b.contacts, PooledMuJoCoContacts)
            # Same-state independent legacy capture covers all raw fields.
            same(
                b.last_step.contacts,
                contacts_to_columns(_contacts(new._mj, new._model, new._data[0])),
            )
            endpoint = copy.copy(new._data[0])
            new._mj.mj_forward(new._model, endpoint)
            same(b.contacts, contacts_to_columns(_contacts(new._mj, new._model, endpoint)))
            assert b.contact_phase != b.last_step.contact_phase
        clone = new.clone()
        assert clone._contact_pool is not new._contact_pool
        new.reset()
        same(retained.contacts, frozen)
    with clone:
        clone.step(0.001)
        assert isinstance(clone.snapshot()[0].contacts, PooledMuJoCoContacts)
    same(retained.contacts, frozen)


def test_capacity_failure_retains_existing_partial_publication_contract(native):
    with MuJoCoRuntime.from_xml(
        XML,
        world_keys=(2, 9),
        contact_storage="columns",
        contact_pool_bytes=CONTACT_ROW_BYTES,
        contact_block_rows=1,
    ) as runtime:
        with pytest.raises(PhysicsStepFailure) as exc:
            runtime.step(0.001)
        assert exc.value.accepted_worlds == {2: True, 9: False}
        assert isinstance(exc.value.__cause__, ContactPoolCapacityError)
        assert runtime._ticks == {2: 1, 9: 0}
        assert runtime._data[9].time == 0 and runtime._data[2].time == 0.001
        assert runtime._poisoned


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_bytes": 1},
        {"max_bytes": True},
        {"block_rows": 0},
        {"block_rows": False},
    ],
)
def test_pool_configuration_rejects_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        MuJoCoContactPool(**kwargs)


def test_gpu_cannot_silently_accept_cpu_column_mode():
    with pytest.raises(ValueError, match="CPU-only"):
        MuJoCoRuntime(object(), mode="cuda", contact_storage="columns")
