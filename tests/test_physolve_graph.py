"""Fixed real CUDA Graph qualification over the frozen PhySolve Host/CUBIN."""
import numpy as np
import pytest
from test_physolve_runtime import CONTACT, SLIDER, payload

from phymeo import PhySolveRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml


@pytest.mark.cuda
@pytest.mark.parametrize('nworld', [1, 8, 64])
@pytest.mark.parametrize('model', [SLIDER, CONTACT], ids=['force', 'contact'])
def test_fixed_banks_direct_graph_reset_stream_and_gc(payload, nworld, model, monkeypatch):
    import gc
    import tyx as tx
    record = compile_mujoco_xml(model.replace('.002', '.01'), require_asset=False)
    with PhySolveRuntime(record, payload=payload, nworld=nworld, body_observations=True,
                        contact_observations=True) as direct, \
         PhySolveRuntime(record, payload=payload, nworld=nworld, body_observations=True,
                        contact_observations=True) as captured:
        initial = captured.snapshot()
        assert captured.prepare_step_graph()
        assert not captured.prepare_step_graph()
        for name, value in captured.snapshot().items():
            np.testing.assert_array_equal(value, initial[name])
        assert captured.execution_costs()['body_observation_publications'] == 0
        gc.collect()
        def advance(runtime, mode):
            ticket = runtime.begin_execution()
            if model == SLIDER:
                storage = runtime.storage_binding()
                storage.borrow('qfrc_applied', runtime.writable_bank, write=True).assign(
                    np.full((nworld, 1), 2., np.float32), stream=storage.stream)
            runtime.enqueue_step(ticket, steps=4, execution_mode=mode)
            runtime.complete_execution(ticket)
            runtime.accept_execution(ticket)
        for _ in range(3):
            advance(direct, 'direct')
            with monkeypatch.context() as patch:
                patch.setattr(captured._native, '_step_runtime', lambda *a: (_ for _ in ()).throw(AssertionError('direct fallback')))
                patch.setattr(tx, 'empty', lambda *a, **k: (_ for _ in ()).throw(AssertionError('replay allocation')))
                advance(captured, 'cuda_graph')
            for field in ('qpos', 'qvel', 'time'):
                np.testing.assert_allclose(captured.snapshot()[field], direct.snapshot()[field], atol=1e-6, rtol=0)
            left, right = captured.storage_binding(), direct.storage_binding()
            for field in left.fields:
                if field.startswith('body_') or field in ('contact_count_by_world', 'contact_sample_time'):
                    actual = left.borrow(field, captured.active_bank).numpy(stream=left.stream)
                    expected = right.borrow(field, direct.active_bank).numpy(stream=right.stream)
                    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
            if model == CONTACT:
                def contacts(runtime):
                    binding = runtime.storage_binding()
                    def read(name):
                        return binding.borrow(name, runtime.active_bank).numpy(stream=binding.stream)
                    count = int(read('contact_count')[0])
                    order = np.argsort(read('contact_world')[:count, 0])
                    return {name: read(name)[:count][order] for name in
                            ('contact_geom', 'contact_dim', 'contact_position',
                             'contact_distance', 'contact_wrench', 'contact_frame')}
                actual, expected = contacts(captured), contacts(direct)
                for field in actual:
                    np.testing.assert_allclose(actual[field], expected[field], atol=1e-6, rtol=0)
        assert captured.execution_costs()['step_graph_replays'] == 3
        if model == SLIDER:
            np.testing.assert_allclose(captured.snapshot()['qvel'], .12, atol=1e-6, rtol=0)
        mask = np.arange(nworld) % 2 == 0
        direct.reset(mask)
        captured.reset(mask)
        advance(direct, 'direct')
        advance(captured, 'cuda_graph')
        for field in ('qpos', 'qvel', 'time'):
            np.testing.assert_allclose(captured.snapshot()[field], direct.snapshot()[field], atol=1e-6, rtol=0)
        external = tx.CudaStream()
        try:
            captured.bind_stream(int(external.handle), owner=external)
            assert not captured.step_graph_ready
            captured.prepare_step_graph()
            advance(captured, 'cuda_graph')
            assert captured.execution_costs()['step_graph_captures'] == 4
            captured.bind_stream()
        finally:
            external.destroy()


@pytest.mark.cuda
def test_native_capture_does_not_advance_before_replay(payload):
    import tyx as tx
    record = compile_mujoco_xml(CONTACT.replace('.002', '.01'), require_asset=False)
    with PhySolveRuntime(record, payload=payload, body_observations=True,
                        contact_observations=True) as runtime:
        storage = runtime.storage_binding()
        ticket = runtime.begin_execution()
        bank = runtime.writable_bank
        before = storage.borrow('qpos', bank).numpy(stream=storage.stream)
        graph = tx.Graph(stream=storage.stream, mode='cuda_stream')
        try:
            with graph:
                runtime.enqueue_step(ticket, steps=4)
            np.testing.assert_array_equal(storage.borrow('qpos', bank).numpy(stream=storage.stream), before)
            graph.replay()
            runtime.complete_execution(ticket)
            runtime.accept_execution(ticket)
            np.testing.assert_allclose(runtime.snapshot()['time'], .04, atol=1e-7, rtol=0)
        finally:
            storage.stream.synchronize()
            graph.destroy()


@pytest.mark.cuda
def test_failed_capture_and_graph_close_retry_keep_native_owners(payload, monkeypatch):
    import tyx as tx
    record = compile_mujoco_xml(SLIDER.replace('.002', '.01'), require_asset=False)
    runtime = PhySolveRuntime(record, payload=payload, body_observations=True,
                             contact_observations=True)
    try:
        initial = runtime.snapshot()
        step = runtime._native._step_runtime
        calls = []
        def fail_second_bank(handle):
            calls.append(handle)
            if len(calls) == 5:
                raise RuntimeError('injected second Graph capture failure')
            return step(handle)
        with monkeypatch.context() as patch:
            patch.setattr(runtime._native, '_step_runtime', fail_second_bank)
            with pytest.raises(RuntimeError, match='second Graph capture failure'):
                runtime.prepare_step_graph()
        assert not runtime.step_graph_ready
        assert not runtime.storage_binding().stream.is_capturing
        for name, value in runtime.snapshot().items():
            np.testing.assert_array_equal(value, initial[name])
        runtime.prepare_step_graph()
        native_close = runtime._native._close_runtime
        closed = []
        def track_close(*args):
            closed.append(args)
            return native_close(*args)
        with monkeypatch.context() as patch:
            patch.setattr(runtime._native, '_close_runtime', track_close)
            patch.setattr(tx.Graph, 'destroy', lambda *a: (_ for _ in ()).throw(RuntimeError('Graph close failed')))
            with pytest.raises(RuntimeError, match='Graph close failed'):
                runtime.close()
        assert not closed
        assert all(handle is not None for handle in runtime._handles)
        runtime.close()
    finally:
        runtime.close()


@pytest.mark.cuda
def test_graph_stream_rollback_restores_fixed_execution(payload, monkeypatch):
    import tyx as tx
    from test_physolve_stream import make_runtime
    external = tx.CudaStream()
    with make_runtime(payload) as runtime:
        try:
            runtime.prepare_step_graph()
            original = runtime.storage_binding().stream
            rebind = runtime._native._rebind_runtime
            calls = []
            def fail_second(handle, completed):
                calls.append(handle)
                if len(calls) == 2:
                    raise RuntimeError('injected native rebind failure')
                return rebind(handle, completed)
            with monkeypatch.context() as patch:
                patch.setattr(runtime._native, '_rebind_runtime', fail_second)
                with pytest.raises(RuntimeError, match='native rebind failure'):
                    runtime.bind_stream(int(external.handle), owner=external)
            assert runtime.storage_binding().stream is original
            assert runtime.step_graph_ready
            ticket = runtime.begin_execution()
            runtime.enqueue_step(ticket, steps=4, execution_mode='cuda_graph')
            runtime.complete_execution(ticket)
            runtime.accept_execution(ticket)
            np.testing.assert_allclose(runtime.snapshot()['time'], .008, atol=1e-7, rtol=0)
        finally:
            external.destroy()


@pytest.mark.cuda
def test_graph_error_blocks_accept_and_discard_recovers(payload):
    from test_physolve_stream import make_runtime
    with make_runtime(payload, 8) as runtime:
        runtime.prepare_step_graph()
        before = runtime.snapshot()
        storage = runtime.storage_binding()
        ticket = runtime.begin_execution()
        runtime.enqueue_step(ticket, steps=4, execution_mode='cuda_graph')
        # Fault injection at the supported compact-error boundary. Borrowed
        # native inputs require caller-validated finite data; deliberately
        # feeding NaN into the CUBIN is not a recoverable input contract.
        storage.borrow('execution_error', runtime.observation_bank).fill_(2, stream=storage.stream)
        with pytest.raises(RuntimeError, match='non-finite'):
            runtime.complete_execution(ticket)
        with pytest.raises(RuntimeError, match='completed'):
            runtime.accept_execution(ticket)
        runtime.discard_execution(ticket)
        for name, value in runtime.snapshot().items():
            np.testing.assert_array_equal(value, before[name])
        ticket = runtime.begin_execution()
        runtime.enqueue_step(ticket, steps=4, execution_mode='cuda_graph')
        runtime.complete_execution(ticket)
        runtime.accept_execution(ticket)
        assert runtime.snapshot()['time'][0] > 0


@pytest.mark.cuda
def test_graph_rebinding_memory_samples(payload):
    import gc
    import tyx as tx
    from cuda.bindings import driver
    record = compile_mujoco_xml(CONTACT.replace('.002', '.01'), require_asset=False)
    external = tx.CudaStream()
    samples = []
    try:
        with PhySolveRuntime(record, payload=payload, nworld=8, body_observations=True,
                            contact_observations=True) as runtime:
            for iteration in range(8):
                runtime.bind_stream(int(external.handle) if iteration % 2 else 0, owner=external)
                runtime.prepare_step_graph()
                ticket = runtime.begin_execution()
                runtime.enqueue_step(ticket, steps=4, execution_mode='cuda_graph')
                runtime.complete_execution(ticket)
                runtime.accept_execution(ticket)
                gc.collect()
                status, free, total = driver.cuMemGetInfo()
                assert int(status) == 0
                samples.append(dict(iteration=iteration, free_bytes=free, total_bytes=total,
                                    graph_captures=runtime.execution_costs()['step_graph_captures']))
        external.validate_context()
    finally:
        external.destroy()
    # Observational device-wide memory samples; other applications share this GPU.
    print('B5A_GRAPH_MEMORY', samples, flush=True)
