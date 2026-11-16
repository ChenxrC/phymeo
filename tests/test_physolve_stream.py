"""External stream ordering and lifecycle against the actual frozen native Host."""
import numpy as np
import pytest
from test_physolve_runtime import SLIDER, payload

from phymeo import PhySolveRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml


def make_runtime(payload, nworld=1):
    return PhySolveRuntime(compile_mujoco_xml(SLIDER, require_asset=False),
                          payload=payload, nworld=nworld, body_observations=True,
                          contact_observations=True)


@pytest.mark.cuda
@pytest.mark.parametrize('nworld', [1, 8, 64])
def test_external_stream_control_step_forward_reset(payload, nworld, monkeypatch):
    import tyx as tx
    external = tx.CudaStream()
    runtime = make_runtime(payload, nworld)
    try:
        storage = runtime.storage_binding()
        original = storage.stream
        fields = [storage.borrow('qpos', i) for i in (0, 1)]
        for handle in runtime._handles:
            runtime._native._set_runtime_counters(handle, True, True)
        assert runtime.bind_stream(int(external.handle), owner=external)
        borrowed = storage.stream
        assert borrowed is not external and borrowed.handle == int(external.handle)
        assert not runtime.bind_stream(int(external.handle), owner=external)
        assert sum(runtime._native._runtime_counters(h)['rebind_calls'] for h in runtime._handles) == 2
        assert storage.stream is borrowed
        assert [storage.borrow('qpos', i) for i in (0, 1)] == fields
        for _ in range(3):
            ticket = runtime.begin_execution()
            target = storage.borrow('qfrc_applied', runtime.writable_bank, write=True)
            target.assign(np.full((nworld, 1), 2., np.float32), stream=external)
            runtime.enqueue_step(ticket, steps=4)
            runtime.complete_execution(ticket)
            runtime.accept_execution(ticket)
        after = runtime.snapshot()
        np.testing.assert_allclose(after['qvel'], .024, atol=1e-6, rtol=0)
        runtime.forward()
        np.testing.assert_array_equal(runtime.snapshot()['qpos'], after['qpos'])
        mask = np.arange(nworld) % 2 == 0
        runtime.reset(mask)
        reset = runtime.snapshot()
        np.testing.assert_array_equal(reset['qvel'][mask], 0.)
        np.testing.assert_array_equal(reset['qvel'][~mask], after['qvel'][~mask])
        assert runtime.bind_stream()
        assert storage.stream is original
        assert runtime.execution_costs()['stream_rebindings'] == 2
        external.validate_context()
        runtime.step()
        runtime.bind_stream(int(external.handle), owner=external)
        counters = [runtime._native._runtime_counters(h) for h in runtime._handles]
        assert sum(row['rebind_calls'] for row in counters) == 6
        print('B4B_NATIVE_STREAM_COST', nworld, counters, flush=True)
        runtime.close()
        external.validate_context()
        external.synchronize()
    finally:
        runtime.close()
        external.destroy()


@pytest.mark.cuda
def test_stream_prepare_failure_rolls_back_both_banks(payload, monkeypatch):
    import tyx as tx
    external = tx.CudaStream()
    runtime = make_runtime(payload)
    try:
        storage = runtime.storage_binding()
        original = storage.stream
        before = runtime.snapshot()
        native = runtime._native
        rebind = native._rebind_runtime
        calls = []
        def fail_second(handle, completed):
            calls.append(handle)
            if len(calls) == 2:
                raise RuntimeError('injected second bank preparation failure')
            return rebind(handle, completed)
        with monkeypatch.context() as patch:
            patch.setattr(native, '_rebind_runtime', fail_second)
            with pytest.raises(RuntimeError, match='second bank'):
                runtime.bind_stream(int(external.handle), owner=external)
        assert len(calls) == 4
        assert storage.stream is original
        for name, values in runtime.snapshot().items():
            np.testing.assert_array_equal(values, before[name])
        runtime.step(qfrc_applied=np.ones((1, 1), np.float32))
        assert runtime.snapshot()['qvel'][0, 0] > 0
        runtime.bind_stream(int(external.handle), owner=external)
    finally:
        runtime.close()
        external.validate_context()
        external.destroy()


@pytest.mark.cuda
def test_stream_pending_capture_validation_and_close_retry(payload, monkeypatch):
    import tyx as tx
    external = tx.CudaStream()
    runtime = make_runtime(payload)
    try:
        ticket = runtime.begin_execution()
        with pytest.raises(RuntimeError, match='pending'):
            runtime.bind_stream(int(external.handle), owner=external)
        runtime.discard_execution(ticket)
        scratch = tx.zeros(1, tx.f32, device='cuda:0')
        original = runtime.storage_binding().stream
        for capturing in (original, external):
            with tx.Graph(stream=capturing, mode='cuda_stream') as graph:
                scratch.zero_(stream=capturing)
                with pytest.raises(RuntimeError, match='captur'):
                    runtime.bind_stream(int(external.handle), owner=external)
            graph.destroy()
        assert runtime.storage_binding().stream is original
        with pytest.raises(ValueError, match='lifetime owner'):
            runtime.bind_stream(int(external.handle))
        with monkeypatch.context() as patch:
            patch.setattr(tx.CudaStream, 'validate_context', lambda s: (_ for _ in ()).throw(ValueError('foreign context')))
            with pytest.raises(ValueError, match='foreign context'):
                runtime.bind_stream(int(external.handle), owner=external)
        runtime.step()
        runtime.bind_stream(int(external.handle), owner=external)
        with monkeypatch.context() as patch:
            patch.setattr(runtime._native, '_close_runtime', lambda *a: (_ for _ in ()).throw(RuntimeError('close failed')))
            with pytest.raises(RuntimeError, match='close failed'):
                runtime.close()
        external.validate_context()
        runtime.close()
        external.validate_context()
    finally:
        runtime.close()
        external.destroy()


@pytest.mark.cuda
@pytest.mark.parametrize('failure', ['retire', 'rollback'])
def test_stream_cleanup_failure_retains_owners_until_close(payload, monkeypatch, failure):
    import tyx as tx
    external = tx.CudaStream()
    runtime = make_runtime(payload)
    try:
        native = runtime._native
        rebind = native._rebind_runtime
        calls = []
        def fail_rebind(handle, completed):
            calls.append(handle)
            if len(calls) >= 2:
                raise RuntimeError('injected rollback failure')
            return rebind(handle, completed)
        with monkeypatch.context() as patch:
            if failure == 'rollback':
                patch.setattr(native, '_rebind_runtime', fail_rebind)
            else:
                patch.setattr(native, '_retire_runtime', lambda *a: (_ for _ in ()).throw(RuntimeError('injected retirement failure')))
            with pytest.raises(RuntimeError) as error:
                runtime.bind_stream(int(external.handle), owner=external)
        assert error.value.stream_bindings_changed
        assert len(runtime._stream_retained) == 2
        with pytest.raises(RuntimeError, match='closing'):
            runtime.step()
        runtime.close()
        runtime.close()
        external.validate_context()
    finally:
        runtime.close()
        external.destroy()
