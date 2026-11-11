"""Real CUDA execution contract; no Newton/Warp and no snapshot-based executor."""

import numpy as np
import pytest
from test_physolve_runtime import FREE, SLIDER
from test_physolve_runtime import payload as payload_fixture

from phymeo import PhySolveRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

payload = payload_fixture


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_public_storage_and_delayed_publication(payload, nworld):
    with PhySolveRuntime(compile_mujoco_xml(SLIDER, require_asset=False),
                        payload=payload, nworld=nworld) as runtime:
        storage = runtime.storage_binding()
        assert storage is runtime.storage_binding()
        assert storage.contract_version == 1
        assert storage.fields["xpos"].shape == (nworld, 2, 3)
        assert storage.fields["xquat"].shape == (nworld, 2, 4)
        assert "not COM" in storage.fields["xpos"].frame
        assert storage.fields["qfrc_applied"].sample == "persistent external command"
        q = tuple(storage.borrow("qpos", bank) for bank in (0, 1))
        assert q[0].ptr != q[1].ptr
        old = runtime.active_bank
        initial = runtime.snapshot()
        ticket = runtime.begin_execution()
        bank = runtime.writable_bank
        assert bank == 1 - old
        with pytest.raises(RuntimeError, match="pending"):
            runtime.begin_execution()
        with pytest.raises(RuntimeError, match="completed"):
            runtime.accept_execution(ticket)
        with pytest.raises(RuntimeError, match="candidate"):
            storage.borrow("qfrc_applied", old, write=True)
        with pytest.raises(RuntimeError, match="candidate"):
            storage.borrow("qacc", bank, write=True)
        force = np.arange(1, nworld + 1, dtype=np.float32)[:, None] / nworld
        storage.borrow("qfrc_applied", bank, write=True).assign(force, stream=storage.stream)
        runtime.enqueue_step(ticket, steps=4)
        assert runtime.active_bank == old
        runtime.complete_execution(ticket)
        assert runtime.active_bank == old
        for name, value in runtime.snapshot().items():
            np.testing.assert_array_equal(value, initial[name])
        assert runtime.execution_costs()["validation_d2h_bytes"] == 4
        runtime.accept_execution(ticket)
        assert runtime.active_bank == bank
        after = runtime.snapshot()
        np.testing.assert_allclose(after["qpos"][:, 0], force[:, 0] / 2 * .002**2 * 10,
                                   atol=1e-6, rtol=0)
        np.testing.assert_array_equal(after["qfrc_applied"], force)
        with pytest.raises(RuntimeError, match="stale"):
            runtime.accept_execution(ticket)
        ticket = runtime.begin_execution()
        runtime.enqueue_step(ticket)
        runtime.complete_execution(ticket)
        runtime.discard_execution(ticket)
        for name, value in runtime.snapshot().items():
            np.testing.assert_array_equal(value, after[name])
        assert storage.borrow("qpos", 0) is q[0]
    with pytest.raises(RuntimeError, match="stale"):
        storage.borrow("qpos", 0)


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_compact_validation_no_full_readback_and_masked_reset(payload, nworld, monkeypatch):
    import tyx as tx
    with PhySolveRuntime(compile_mujoco_xml(FREE, require_asset=False),
                        payload=payload, nworld=nworld) as runtime:
        storage = runtime.storage_binding()
        initial = runtime.snapshot()
        def forbidden(*args, **kwargs):
            raise AssertionError("full host snapshot or device allocation in prepared execution")
        with monkeypatch.context() as patch:
            patch.setattr(tx.Array, "numpy", forbidden)
            patch.setattr(tx, "empty", forbidden)
            for _ in range(4):
                ticket = runtime.begin_execution()
                runtime.enqueue_step(ticket, steps=4)
                runtime.complete_execution(ticket)
                runtime.accept_execution(ticket)
        after = runtime.snapshot()
        np.testing.assert_allclose(after["qpos"][:, 2], 1 - 9.81 * .002**2 * 136,
                                   atol=1e-6, rtol=0)
        assert runtime.execution_costs()["validation_readbacks"] == 4
        mask = np.arange(nworld) % 2 == 0
        runtime.reset(mask)
        reset = runtime.snapshot()
        for name in initial:
            np.testing.assert_array_equal(reset[name][mask], initial[name][mask])
            np.testing.assert_array_equal(reset[name][~mask], after[name][~mask])
        assert storage is runtime.storage_binding()


@pytest.mark.cuda
def test_nonfinite_latches_across_substeps_and_recovers(payload, monkeypatch):
    with PhySolveRuntime(compile_mujoco_xml(SLIDER, require_asset=False), payload=payload) as runtime:
        before = runtime.snapshot()
        original = runtime._native._step_runtime
        calls = 0
        def inject(handle):
            nonlocal calls
            original(handle)
            calls += 1
            if calls == 1:
                # Fault injection in an output, not a change to physical inputs/tolerances.
                runtime._banks[1 - runtime.active_bank].qacc.fill_(float("nan"), stream=runtime._stream)
        with monkeypatch.context() as patch:
            patch.setattr(runtime._native, "_step_runtime", inject)
            ticket = runtime.begin_execution()
            runtime.enqueue_step(ticket, steps=4)
            with pytest.raises(RuntimeError, match="non-finite.*qacc"):
                runtime.complete_execution(ticket)
            with pytest.raises(RuntimeError, match="completed"):
                runtime.accept_execution(ticket)
            runtime.discard_execution(ticket)
        for name, value in runtime.snapshot().items():
            np.testing.assert_array_equal(value, before[name])
        runtime.step()
        assert runtime.snapshot()["time"][0] > 0


@pytest.mark.cuda
def test_enqueue_failure_and_foreign_ticket_cannot_publish(payload, monkeypatch):
    record = compile_mujoco_xml(SLIDER, require_asset=False)
    with PhySolveRuntime(record, payload=payload) as runtime, PhySolveRuntime(record, payload=payload) as peer:
        ticket = runtime.begin_execution()
        other = peer.begin_execution()
        with pytest.raises(RuntimeError, match="foreign"):
            runtime.enqueue_step(other)
        peer.discard_execution(other)
        before = runtime.snapshot()
        def failed(handle):
            raise RuntimeError("injected native enqueue failure")
        with monkeypatch.context() as patch:
            patch.setattr(runtime._native, "_step_runtime", failed)
            with pytest.raises(RuntimeError, match="enqueue failure"):
                runtime.enqueue_step(ticket)
        with pytest.raises(RuntimeError, match="enqueued"):
            runtime.complete_execution(ticket)
        with pytest.raises(RuntimeError, match="completed"):
            runtime.accept_execution(ticket)
        runtime.discard_execution(ticket)
        for name, value in runtime.snapshot().items():
            np.testing.assert_array_equal(value, before[name])
        runtime.step()


@pytest.mark.cuda
def test_device_capacity_failure_and_close_retry(payload, monkeypatch):
    runtime = PhySolveRuntime(compile_mujoco_xml(SLIDER, require_asset=False), payload=payload)
    storage = runtime.storage_binding()
    try:
        ticket = runtime.begin_execution()
        index = runtime.writable_bank
        runtime.enqueue_forward(ticket)
        # Fault injection after forward, then re-run the exact prepared validator.
        runtime._banks[index].nefc.fill_(runtime.njmax + 1, stream=storage.stream)
        runtime._validation.enqueue(index, storage.stream)
        with pytest.raises(RuntimeError, match="capacity"):
            runtime.complete_execution(ticket)
        runtime.discard_execution(ticket)
        runtime.forward()
        with monkeypatch.context() as patch:
            def failed():
                raise RuntimeError("injected completion failure")
            patch.setattr(runtime, "_complete", failed)
            with pytest.raises(RuntimeError, match="completion failure"):
                runtime.close()
            with pytest.raises(RuntimeError, match="stale"):
                storage.require_valid()
    finally:
        runtime.close()


@pytest.mark.cuda
@pytest.mark.parametrize("failure", ["clear", "copy"])
def test_preparation_failure_recovers_without_publishing(payload, monkeypatch, failure):
    import tyx as tx

    with PhySolveRuntime(compile_mujoco_xml(FREE, require_asset=False), payload=payload) as runtime:
        before = runtime.snapshot()
        with monkeypatch.context() as patch:
            def rejected(*args, **kwargs):
                raise RuntimeError("injected preparation failure")
            if failure == "clear":
                patch.setattr(runtime._validation, "clear", rejected)
            else:
                assign = tx.Array.assign
                calls = 0
                def failed_copy(self, *args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 3:
                        rejected()
                    return assign(self, *args, **kwargs)
                patch.setattr(tx.Array, "assign", failed_copy)
            with pytest.raises(RuntimeError, match="preparation failure"):
                runtime.forward()
        assert runtime.pending_execution is None
        for name, value in runtime.snapshot().items():
            np.testing.assert_array_equal(value, before[name])
        runtime.step()


@pytest.mark.cuda
def test_preparation_cleanup_failure_retains_owner_and_primary_error(payload, monkeypatch):
    with PhySolveRuntime(compile_mujoco_xml(FREE, require_asset=False), payload=payload) as runtime:
        old = runtime.active_bank
        with monkeypatch.context() as patch:
            def preparation(*args):
                raise ValueError("injected prepare")
            def completion():
                raise RuntimeError("injected cleanup")
            patch.setattr(runtime._validation, "clear", preparation)
            patch.setattr(runtime, "_complete", completion)
            with pytest.raises(ValueError, match="injected prepare") as captured:
                runtime.forward()
            assert "injected cleanup" in str(captured.value.__cause__)
            assert runtime.pending_execution is not None
            assert runtime.active_bank == old
        runtime.discard_execution(runtime.pending_execution)
        runtime.forward()


@pytest.mark.cuda
def test_validation_warmup_failure_retains_launched_resources(payload, monkeypatch):
    from phymeo.backends._physolve_validation import PreparedValidation

    read = PreparedValidation.read
    native = payload.load()
    with monkeypatch.context() as patch:
        def failed_read(self, stream):
            read(self, stream)
            raise RuntimeError("injected validation warmup")
        def failed_close(*args):
            raise RuntimeError("injected native close")
        patch.setattr(PreparedValidation, "read", failed_read)
        patch.setattr(native, "_close_runtime", failed_close)
        with pytest.raises(RuntimeError, match="validation warmup") as captured:
            PhySolveRuntime(compile_mujoco_xml(FREE, require_asset=False), payload=payload)
        retained = captured.value.runtime
        assert retained._validation is not None
        assert retained._validation.flags.ptr
        assert retained._banks and retained._stream is not None
    retained.close()
    assert retained._closed and retained._validation is None
