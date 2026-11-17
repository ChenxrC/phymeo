"""Public candidate field updates and managed scalar commands on actual CUDA."""

import numpy as np
import pytest
from test_native_assembly import template
from test_native_assembly import urdf as urdf_fixture
from test_physolve_runtime import payload as payload_fixture

urdf = urdf_fixture
payload = payload_fixture


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_command_update_state_forward_and_masked_reset(urdf, payload, nworld, monkeypatch):
    compiled = template(urdf)
    assert compiled.scalar_controls == (("arm::slide", 0, 2, 0.5, 0.375, 0.125),)
    with compiled.bind(nworld).create_runtime(payload=payload) as runtime:
        storage = runtime.storage_binding()
        before = runtime.snapshot()
        targets = np.linspace(0.2, 0.4, nworld, dtype=np.float32)[:, None]
        velocities = np.full((nworld, 1), 0.3, np.float32)
        old_bank = runtime.active_bank
        ticket = runtime.begin_execution()
        bank = runtime.writable_bank
        storage.borrow("target_position", bank, write=True).assign(targets, stream=storage.stream)
        storage.borrow("target_velocity", bank, write=True).assign(
            velocities, stream=storage.stream
        )
        with pytest.raises(RuntimeError, match="candidate"):
            storage.borrow("ctrl", bank, write=True)

        def forbidden(*args):
            raise AssertionError("commands must not run forward")

        with monkeypatch.context() as patch:
            patch.setattr(runtime._native, "_forward_runtime", forbidden)
            runtime.enqueue_update(ticket, compose_controls=True)
            runtime.complete_execution(ticket)
        assert runtime.active_bank == old_bank
        runtime.accept_execution(ticket)
        after = runtime.snapshot()
        for name in before:
            if name != "ctrl":
                np.testing.assert_array_equal(after[name], before[name])
        np.testing.assert_allclose(after["ctrl"], 2 * targets + 0.5 * velocities, atol=1e-6, rtol=0)
        with pytest.raises(ValueError, match="managed"):
            runtime.step(ctrl=targets)
        runtime.step()
        stepped = runtime.snapshot()
        mask = np.arange(nworld) % 2 == 0
        runtime.reset(mask)
        reset = runtime.snapshot()
        for name in before:
            np.testing.assert_array_equal(reset[name][~mask], stepped[name][~mask])
        target_read = storage.borrow("target_position", runtime.active_bank).numpy()
        np.testing.assert_array_equal(target_read[mask], 0.375)
        np.testing.assert_array_equal(target_read[~mask], targets[~mask])
        ticket = runtime.begin_execution()
        bank = runtime.writable_bank
        storage.borrow("qpos", bank, write=True).assign(
            np.full((nworld, 1), 0.2, np.float32), stream=storage.stream
        )
        forward = runtime._native._forward_runtime
        calls = []

        def counted(handle):
            calls.append(handle)
            return forward(handle)

        with monkeypatch.context() as patch:
            patch.setattr(runtime._native, "_forward_runtime", counted)
            runtime.enqueue_update(ticket, refresh_state=True)
            runtime.complete_execution(ticket)
            runtime.accept_execution(ticket)
        assert len(calls) == 1
        final = runtime.snapshot()
        np.testing.assert_array_equal(final["qpos"], np.float32(0.2))
        np.testing.assert_array_equal(final["time"], reset["time"])


@pytest.mark.cuda
def test_failed_composition_retains_accepted_control_and_recovers(urdf, payload):
    with template(urdf).bind(1).create_runtime(payload=payload) as runtime:
        storage = runtime.storage_binding()
        before = runtime.snapshot()
        bank = runtime.active_bank
        ticket = runtime.begin_execution()
        storage.borrow("target_position", runtime.writable_bank, write=True).assign(
            np.full((1, 1), np.finfo(np.float32).max, np.float32), stream=storage.stream
        )
        runtime.enqueue_update(ticket, compose_controls=True)
        with pytest.raises(RuntimeError, match="finite"):
            runtime.complete_execution(ticket)
        assert runtime.active_bank == bank
        runtime.discard_execution(ticket)
        for name, value in runtime.snapshot().items():
            np.testing.assert_array_equal(value, before[name])
        runtime.step()


@pytest.mark.cuda
def test_command_update_recovers_after_discarded_capacity_failure(urdf, payload):
    with template(urdf).bind(1).create_runtime(payload=payload) as runtime:
        before = runtime.snapshot()
        stream = runtime.storage_binding().stream
        ticket = runtime.begin_execution()
        bank = runtime._banks[runtime.writable_bank]
        # Inject a completed failing candidate; do not run a native kernel with
        # deliberately invalid counters. All three counts must be restored.
        for name, limit in (
            ("nacon", bank.naconmax),
            ("ncollision", bank.naconmax),
            ("nefc", bank.njmax),
        ):
            array = getattr(bank, name)
            array.assign(np.full(array.shape, limit + 1, np.int32), stream=stream)
        runtime._validation.enqueue(runtime.writable_bank, stream)
        runtime._pending["phase"] = "enqueued"
        with pytest.raises(RuntimeError, match="capacity"):
            runtime.complete_execution(ticket)
        runtime.discard_execution(ticket)
        ticket = runtime.begin_execution()
        runtime.enqueue_update(ticket, compose_controls=True)
        runtime.complete_execution(ticket)
        runtime.accept_execution(ticket)
        for name, value in runtime.snapshot().items():
            np.testing.assert_allclose(value, before[name], atol=1e-6, rtol=0)
