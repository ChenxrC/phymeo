"""Lossless contact export, ownership, budgets, and real native snapshots."""

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace

import numpy as np
import pytest

from phymeo.backends.mujoco_recording import (
    CONTACT_COLUMN_SCHEMA,
    CONTACT_ROW_BYTES,
    contacts_to_columns,
)
from phymeo.backends.mujoco_runtime import MuJoCoContact, MuJoCoRuntime


def sample_contact(index=0):
    def array(shape):
        return np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 17

    return MuJoCoContact(
        index=index,
        geometry_ids=(1, 2),
        flex_ids=(-1, -1),
        element_ids=(-1, -1),
        vertex_ids=(-1, -1),
        dimension=3,
        distance=-0.0002,
        position=array((3,)),
        frame=array((3, 3)),
        include_margin=0.001,
        friction=array((5,)),
        solref=array((2,)),
        solref_friction=array((2,)),
        solimp=array((5,)),
        regularized_mu=0.006324555320336759,
        cone_hessian=array((36,)),
        excluded=0,
        constraint_address=0,
        wrench_contact_frame=array((6,)),
    )


def assert_contact_equal(actual, expected):
    for field in fields(MuJoCoContact):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        assert type(a) is type(b), field.name
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and a.shape == b.shape
            assert a.tobytes() == b.tobytes(), field.name
        else:
            assert a == b, field.name


def test_all_fields_roundtrip_without_deepcopy(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("recursive deepcopy must not be used")

    records = tuple(sample_contact(i) for i in range(7))
    monkeypatch.setattr(copy, "deepcopy", forbidden)
    packet = contacts_to_columns(records)
    assert packet.schema == CONTACT_COLUMN_SCHEMA
    assert packet.count == 7
    assert set(packet.arrays) == {f.name for f in fields(MuJoCoContact)}
    assert packet.nbytes == 7 * CONTACT_ROW_BYTES
    for i, record in enumerate(records):
        assert_contact_equal(packet.contact(i), record)


def test_bytes_are_detached_and_cannot_be_made_writeable():
    original = sample_contact()
    packet = contacts_to_columns([original])
    expected = packet.arrays["cone_hessian"].tobytes()
    original.cone_hessian[:] = -9
    assert packet.arrays["cone_hessian"].tobytes() == expected
    with pytest.raises(TypeError):
        packet.arrays["new"] = np.empty(0)
    for array in packet.arrays.values():
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flags.writeable = True
    with pytest.raises(ValueError):
        packet.contact(0).cone_hessian[0] = 1


def test_nan_payload_signed_zero_and_empty_contacts_are_retained():
    hessian = np.zeros(36, dtype=np.float64)
    hessian.view(np.uint64)[:3] = [0x7FF8000000001234, 0x8000000000000000, 0x7FF0000000000000]
    packet = contacts_to_columns([replace(sample_contact(), cone_hessian=hessian)])
    assert packet.contact(0).cone_hessian.tobytes() == hessian.tobytes()
    empty = contacts_to_columns(())
    assert empty.count == empty.nbytes == 0
    assert set(empty.arrays) == set(packet.arrays)
    for name, values in empty.arrays.items():
        assert values.shape == (0, *packet.arrays[name].shape[1:])


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="positive integer"):
        contacts_to_columns((), max_bytes=budget)


def test_budget_rejects_before_output_allocation(monkeypatch):
    record = sample_contact()

    def forbidden(*args, **kwargs):
        raise AssertionError("over-budget input must be rejected before allocating")

    monkeypatch.setattr(np, "empty", forbidden)
    with pytest.raises(ValueError, match="exceeds max_bytes"):
        contacts_to_columns([record], max_bytes=CONTACT_ROW_BYTES - 1)


def test_exact_budget_is_accepted_and_no_rows_are_filtered():
    records = [sample_contact(10), replace(sample_contact(99), excluded=1, constraint_address=-1)]
    packet = contacts_to_columns(records, max_bytes=2 * CONTACT_ROW_BYTES)
    assert packet.arrays["index"].tolist() == [10, 99]
    assert packet.arrays["excluded"].tolist() == [0, 1]
    assert packet.arrays["constraint_address"].tolist() == [0, -1]


@pytest.mark.parametrize(
    "bad", [np.zeros(36, np.float32), np.zeros((6, 6)), np.zeros(36, np.int64)]
)
def test_wrong_shape_or_dtype_is_rejected_without_coercion(bad):
    record = replace(sample_contact(), cone_hessian=bad)
    with pytest.raises(ValueError, match="cone_hessian"):
        contacts_to_columns([record])


@pytest.mark.parametrize("index", [-1, 1, True, 0.5])
def test_invalid_row_index(index):
    packet = contacts_to_columns([sample_contact()])
    with pytest.raises((TypeError, IndexError)):
        packet.contact(index)


def test_input_types_are_explicit():
    with pytest.raises(TypeError, match="finite sequence"):
        contacts_to_columns(iter([sample_contact()]))
    with pytest.raises(TypeError, match="MuJoCoContact"):
        contacts_to_columns([{}])


def test_concurrent_export_has_no_shared_scratch():
    records = tuple(sample_contact(i) for i in range(32))
    with ThreadPoolExecutor(max_workers=4) as pool:
        packets = list(pool.map(lambda _: contacts_to_columns(records), range(16)))
    for packet in packets:
        for name, array in packet.arrays.items():
            assert array.tobytes() == packets[0].arrays[name].tobytes()
            if packet is not packets[0]:
                assert not np.shares_memory(array, packets[0].arrays[name])


def test_real_solver_and_endpoint_exports_survive_step_reset_and_close(tmp_path):
    pytest.importorskip("mujoco")
    xml = """<mujoco><option timestep="0.001"/><worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .099"><freejoint/>
        <geom type="sphere" size=".1" mass=".5"/>
      </body></worldbody></mujoco>"""
    runtime = MuJoCoRuntime.from_xml(xml)
    try:
        runtime.step(0.001)
        state = runtime.snapshot()[0]
        assert state.contacts and state.last_step.contacts
        assert state.contact_phase != state.last_step.contact_phase
        solver = contacts_to_columns(state.last_step.contacts)
        endpoint = contacts_to_columns(state.contacts)
        for packet, source in [(solver, state.last_step.contacts), (endpoint, state.contacts)]:
            for index, contact in enumerate(source):
                assert_contact_equal(packet.contact(index), contact)
        expected = {name: values.tobytes() for name, values in endpoint.arrays.items()}
        runtime.step(0.001)
        runtime.reset()
    finally:
        runtime.close()
    del state, runtime
    file = tmp_path / "endpoint.npz"
    np.savez_compressed(file, **endpoint.arrays)
    with np.load(file, allow_pickle=False) as loaded:
        assert set(loaded.files) == set(expected)
        for name, values in loaded.items():
            assert values.tobytes() == expected[name]
            assert endpoint.arrays[name].tobytes() == expected[name]
