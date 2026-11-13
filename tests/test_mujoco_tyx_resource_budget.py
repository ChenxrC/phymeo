"""Exact registered payload and failure ownership; not a total-memory benchmark."""

import gc
import weakref
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from phymeo.backends.mujoco_resource_budget import MuJoCoTyxResourceBudget
from phymeo.backends.mujoco_tyx._allocation_manifest import AllocationEntry, allocation_manifest
from phymeo.backends.mujoco_tyx._resources import DeviceResources, quarantined_resources

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
ROUTES = ("base", "tree", "floating", "normal", "mixed", "activation-csr")


def gallery():
    from test_mujoco_tyx_heterogeneous_resources import runner

    return runner()


def declaration(route):
    import mujoco as mj

    from phymeo.backends.mujoco_tyx._model import compile_model
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    if route == "base":
        xml = """<mujoco><option timestep=".001"/><worldbody><body><joint type="slide" axis="0 0 1"/><geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/></body></worldbody></mujoco>"""
    elif route in ("tree", "floating"):
        tool = gallery()
        xml = tool.cases.xml(tool.cases.CASES[0 if route == "tree" else 16])
    elif route == "normal":
        from test_mujoco_tyx_normal_contacts import _two_feet

        xml = _two_feet()
    elif route == "mixed":
        from test_mujoco_tyx_constraints import slider_xml

        xml = slider_xml(dim=3, dry=0.1, limits=True)
    else:
        from test_mujoco_tyx_free_ball_activation import case

        xml = case.source("no-contact-early")
    model = mj.MjModel.from_xml_string(xml)
    plan = compile_model(mj, model)
    return mj, model, plan, compile_mujoco_xml(xml, require_asset=False)


def named_buffer(device, name):
    if "." not in name:
        return getattr(device, name)
    root, index = name.split(".")
    return getattr(device, root)[int(index)]


def device_arrays(device, mode):
    array_type = np.ndarray if mode == "interpret" else device.tx.CudaArray
    found = {}
    for value in vars(device).values():
        entries = value if isinstance(value, tuple) else (value,)
        for item in entries:
            if isinstance(item, array_type):
                found[id(item)] = item
    return found


@pytest.mark.parametrize("bad", (-1, True, 1.0, None))
def test_budget_rejects_noninteger_or_negative_limits(bad):
    with pytest.raises(ValueError):
        MuJoCoTyxResourceBudget(bad)


def test_budget_reservations_admission_and_immutable_snapshots():
    budget = MuJoCoTyxResourceBudget(20)
    zero = budget.snapshot()
    with budget.admit(20):
        assert budget.snapshot()["current_bytes"] == budget.snapshot()["reserved_bytes"] == 0
        owner = budget.reserve(20)
    assert budget.snapshot()["reserved_bytes"] == 20
    with pytest.raises(MemoryError):
        budget.reserve(1)
    owner.allocated("first", 8)
    owner.allocated("second", 12)
    owner.finish()
    snapshot = budget.snapshot()
    assert (snapshot["current_bytes"], snapshot["peak_bytes"], snapshot["reserved_bytes"]) == (
        20,
        20,
        0,
    )
    assert zero["current_bytes"] == 0
    with pytest.raises(TypeError):
        snapshot["current_bytes"] = 0
    with pytest.raises(TypeError):
        snapshot["owners"][0]["allocations"][0]["state"] = "RELEASED"
    owner.released("first")
    assert budget.snapshot()["current_bytes"] == 12
    with budget.admit(8):
        assert budget.snapshot()["current_bytes"] == 12
    with pytest.raises(MemoryError):
        with budget.admit(9):
            pytest.fail("one-byte excess admitted")
    owner.unknown("second")
    assert budget.snapshot()["unknown_retained_bytes"] == budget.snapshot()["current_bytes"] == 12
    with pytest.raises(RuntimeError):
        owner.released("second")
    assert budget.snapshot()["total_device_peak_bytes"] is None
    assert budget.snapshot()["host_peak_bytes"] is None


@pytest.mark.parametrize("when", ("before", "after"))
def test_unknown_free_remains_charged_and_strongly_owned(when):
    budget = MuJoCoTyxResourceBudget(20)
    calls = []

    class Runtime:
        def close(self):
            calls.append("runtime")

    class Buffer:
        def __init__(self, name, fail):
            self.name, self.fail, self.freed = name, fail, False

        def free(self):
            calls.append(self.name)
            if self.fail and when == "before":
                raise RuntimeError("before opaque effect")
            self.freed = True
            if self.fail:
                raise RuntimeError("after opaque effect")

    def exercise():
        token = budget.reserve(20)
        ledger = DeviceResources(token)
        runtime, failed, sibling = Runtime(), Buffer("failed", True), Buffer("sibling", False)
        ledger.set_runtime(runtime)
        for name, owner, count in (("failed", failed, 2), ("sibling", sibling, 3)):
            ledger.describe_buffer(AllocationEntry(name, (count,), "<f4", "scratch"))
            ledger.add_buffer(owner)
        ledger.finish_reservation()
        for unused in range(2):
            with pytest.raises(BaseExceptionGroup, match="UNKNOWN_RELEASE"):
                ledger.close()
        assert calls.count("failed") == calls.count("sibling") == 1
        assert "runtime" not in calls and sibling.freed
        assert failed.freed is (when == "after")
        assert ledger in quarantined_resources()
        return weakref.ref(runtime), weakref.ref(failed), weakref.ref(ledger)

    refs = exercise()
    gc.collect()
    assert all(ref() is not None for ref in refs)
    assert budget.snapshot()["current_bytes"] == 8
    assert budget.snapshot()["unknown_retained_bytes"] == 8
    assert budget.snapshot()["reserved_bytes"] == 0
    assert budget.snapshot()["peak_bytes"] == 20


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("mode", MODES)
def test_manifest_names_shapes_aliases_match_every_actual_owned_buffer(route, mode, tmp_path):
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime

    mj, model, plan, compiled = declaration(route)
    manifest = allocation_manifest(plan, 2, model.nbody, (64, 256))
    entries = {entry.name: entry for entry in manifest.entries}
    assert len(entries) == len(manifest.entries)
    assert manifest.payload_bytes == sum(
        np.dtype(entry.dtype).itemsize * int(np.prod(entry.shape)) for entry in entries.values()
    )
    for bank in (0, 1):
        assert entries[f"q.{bank}"].shape == (2, max(8, model.nq))
        assert entries[f"v.{bank}"].shape == (2, max(8, model.nv))
        assert entries[f"time.{bank}"].shape == (2, 2)
    assert entries["controls"].shape == (2, max(1, model.nu))
    assert entries["body_force"].shape == (2, model.nbody, 6)
    if route == "base":
        assert model.nu == 0 and entries["controls"].shape == (2, 1)
    if route == "tree":
        assert dict(manifest.aliases) == {
            "transmission_offsets": "parents",
            "transmission_dofs": "parents",
            "transmission_moment": "gravity",
            "transmission_fields": "body_force",
        }
    if route == "normal":
        slots = len(plan.slot_ids)
        assert entries["dual_matrix"].shape == entries["face_factor"].shape == (2, slots, slots)
        assert entries["extended_work"].shape == (2, 64, 3)
    if route == "mixed":
        assert entries["efc_data"].shape == (2, 256, 14)
        assert entries["contact_jacobian"].shape == (2, 64, 6, 8)
        assert entries["face_matrix"].shape == (2, 8, 8)
        assert not any(entry.shape == (2, 256, 256) for entry in entries.values())
    if route == "activation-csr":
        assert entries["act.0"].shape == entries["act.1"].shape == (2, model.na)
        assert {
            "activation_ids",
            "activation_values",
            "activation_derivative",
            "integration_derivative",
            "actuator_fields",
            "transmission_offsets",
            "transmission_dofs",
            "transmission_moment",
            "transmission_fields",
        } <= entries.keys()
        assert not manifest.aliases
    with pytest.raises((FrozenInstanceError, AttributeError)):
        manifest.entries = ()
    budget = MuJoCoTyxResourceBudget(manifest.payload_bytes)
    runtime = MuJoCoRuntime(compiled, world_keys=(3, 29), mode=mode, resource_budget=budget)
    try:
        device = runtime._device
        actual_ids = set()
        for name, entry in entries.items():
            array = named_buffer(device, name)
            actual_ids.add(id(array))
            dtype = array.dtype if mode == "interpret" else array.host_dtype
            assert tuple(array.shape) == entry.shape and np.dtype(dtype) == np.dtype(entry.dtype)
            assert array.nbytes == entry.nbytes
        assert len(actual_ids) == len(entries)
        assert actual_ids == set(device_arrays(device, mode))
        for alias, target in manifest.aliases:
            assert named_buffer(device, alias) is named_buffer(device, target)
        if mode == "cuda":
            assert device.counts["device_allocations"] == len(entries)
            assert device.counts["static_uploaded_bytes"] == manifest.payload_bytes
        snapshot = runtime.resource_snapshot()
        assert snapshot["allocation"]["manifest_payload_bytes"] == manifest.payload_bytes
        assert budget.snapshot()["current_bytes"] == (
            manifest.payload_bytes if mode == "cuda" else 0
        )
        assert budget.snapshot()["reserved_bytes"] == 0
        tool = gallery()
        (tmp_path / "manifest.json").write_text(
            __import__("json").dumps(tool.encode(dict(manifest=manifest, actual=snapshot))),
            encoding="utf-8",
        )
    finally:
        runtime.close()
    assert budget.snapshot()["current_bytes"] == budget.snapshot()["reserved_bytes"] == 0


def test_zero_contact_and_efc_capacity_still_describe_real_padding():
    _, model, plan, _ = declaration("mixed")
    manifest = allocation_manifest(plan, 1, model.nbody, (0, 0))
    entries = {entry.name: entry for entry in manifest.entries}
    assert entries["efc_data"].shape == (1, 1, 14)
    assert entries["contact_jacobian"].shape == (1, 1, 6, 8)
    assert entries["working"].shape == (1, 1)
    assert entries["contact_map"].shape == (1, 1, 3)


@pytest.mark.cuda
def test_one_byte_constructor_refusal_precedes_runtime_or_array_allocation(monkeypatch):
    import tyx as tx

    from phymeo.backends.mujoco_runtime import MuJoCoRuntime

    _, model, plan, compiled = declaration("tree")
    size = allocation_manifest(plan, 1, model.nbody, (64, 256)).payload_bytes
    budget = MuJoCoTyxResourceBudget(size - 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("no context or arrays may precede full budget admission")

    monkeypatch.setattr(tx, "Runtime", forbidden)
    monkeypatch.setattr(tx, "cuda_array", forbidden)
    with pytest.raises(MemoryError, match="payload budget"):
        MuJoCoRuntime(compiled, mode="cuda", resource_budget=budget)
    assert budget.snapshot()["current_bytes"] == budget.snapshot()["reserved_bytes"] == 0


@pytest.mark.cuda
@pytest.mark.parametrize("operation", ("reset", "restore"))
def test_facade_peak_exact_one_less_keeps_old_banks_and_retries(operation, monkeypatch, tmp_path):
    from phymeo.backends.mujoco_tyx._model import compile_model

    tool = gallery()
    decl = tool.declarations((0, 8))
    sizes = [
        allocation_manifest(
            compile_model(decl["mj"], model), 1, model.nbody, (64, 256)
        ).payload_bytes
        for model in decl["models"].values()
    ]
    extra = (
        2 * sizes[0]
        if operation == "reset"
        else max(2 * sizes[0], sizes[0] + 2 * sizes[1], sum(sizes))
    )
    budget = MuJoCoTyxResourceBudget(sum(sizes) + extra)
    with tool.Session(
        "cuda", 12, tmp_path / operation, indices=(0, 8), resource_budget=budget
    ) as run:
        run.advance(0)
        checkpoint = run.batch.checkpoint()
        before = run.batch.integration_snapshot()
        receipt, banks = (
            run.batch.last_step,
            tuple(owner._device.active for owner in run.batch._buckets),
        )
        refs = {key: __import__("copy").copy(data) for key, data in run.refs.items()}
        blocker = budget.reserve(1)  # Competing owner uses exactly the last available byte.
        try:
            with monkeypatch.context() as patch:

                def forbidden(*args, **kwargs):
                    raise AssertionError("peak refusal must precede clone/retirement")

                patch.setattr(run.batch, "_clone_bucket", forbidden)
                with pytest.raises(MemoryError, match="payload budget"):
                    if operation == "reset":
                        run.batch.reset((run.keys[0],))
                    else:
                        run.batch.restore(
                            checkpoint, expected_revision=run.batch.checkpoint_revision
                        )
            assert run.batch.checkpoint().to_bytes() == checkpoint.to_bytes()
            assert run.batch.last_step is receipt
            tool.exact(
                run.batch.integration_snapshot(), before, "budget refusal retains complete state"
            )
            assert tuple(owner._device.active for owner in run.batch._buckets) == banks
            assert budget.snapshot()["current_bytes"] == sum(sizes)
            run.emit(
                "one-byte-peak-refusal", operation=operation, state=before, budget=budget.snapshot()
            )
        finally:
            blocker.finish()
        if operation == "reset":
            run.batch.reset((run.keys[0],))
            run.refs[run.keys[0]] = __import__("copy").copy(run.initial_refs[run.keys[0]])
        else:
            run.batch.restore(checkpoint, expected_revision=run.batch.checkpoint_revision)
            run.refs = refs
        run.emit("exact-peak-success", operation=operation, budget=budget.snapshot())
        assert budget.snapshot()["peak_bytes"] == sum(sizes) + extra
        assert budget.snapshot()["current_bytes"] == sum(sizes)
        run.advance(1)
        run.mark_pass()


@pytest.mark.cuda
def test_real_free_then_error_retains_charge_without_retry(monkeypatch):
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime

    _, model, plan, compiled = declaration("base")
    size = allocation_manifest(plan, 1, model.nbody, (64, 256)).payload_bytes
    budget = MuJoCoTyxResourceBudget(size)
    runtime = MuJoCoRuntime(compiled, mode="cuda", resource_budget=budget)
    ledger = runtime._device.resources
    original = ledger._buffers[-1]
    entry = next(
        item
        for item in runtime._device.allocation_manifest.entries
        if item.name == original.allocation_name
    )
    calls = []

    def broken():
        original.callback()
        calls.append("actually-freed")
        raise RuntimeError("bounded post-free accounting seam")

    ledger._buffers[-1] = replace(original, callback=broken)
    for unused in range(2):
        with pytest.raises(BaseExceptionGroup):
            runtime.close()
    assert calls == ["actually-freed"]
    assert ledger.state == "UNKNOWN_RELEASE" and ledger.runtime_pending
    assert (
        budget.snapshot()["current_bytes"]
        == budget.snapshot()["unknown_retained_bytes"]
        == entry.nbytes
    )
    assert budget.snapshot()["reserved_bytes"] == 0
    assert ledger in quarantined_resources()
