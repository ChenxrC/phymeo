"""Actual native solve-time IMU/F-T prerequisites, with independent SI references."""

import numpy as np
import pytest
from test_physolve_runtime import FREE
from test_physolve_runtime import payload as payload_fixture

from phymeo import PhySolveRuntime
from phymeo.backends.physolve_initial import NativeInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

payload = payload_fixture
HINGE = '''<mujoco><option timestep=".01" gravity="0 0 -9.81"/>
<worldbody><body pos="0 0 1"><joint type="hinge" axis="0 0 1"/>
<inertial pos=".2 0 0" mass="2" diaginertia=".1 .1 .1"/>
<geom type="sphere" size=".1" contype="0" conaffinity="0"/>
</body></worldbody></mujoco>'''


def read(runtime):
    storage = runtime.storage_binding()
    return {name: storage.borrow(name, runtime.observation_bank).numpy(stream=storage.stream).copy()
            for name in storage.fields if name.startswith('measured_') or name == 'measurement_sample_time'}


@pytest.mark.cuda
@pytest.mark.parametrize('nworld', [1, 8, 64])
def test_body_measurements_physical_reference_and_retained_time(payload, nworld, monkeypatch):
    import mujoco as mj
    import tyx as tx

    record = compile_mujoco_xml(HINGE, require_asset=False)
    cpu = record.fresh_native_model()
    data = mj.MjData(cpu)
    data.qvel[:] = 2
    data.qfrc_applied[:] = 1
    initial = NativeInitialState.from_data(record, data)
    alpha = 1 / (.1 + 2*.2**2)
    with PhySolveRuntime(record, payload=payload, nworld=nworld, initial_state=initial,
                         body_observations=True, body_measurements=True) as runtime:
        storage = runtime.storage_binding()
        assert runtime.body_measurement_contract_version == 1
        for name, field in storage.fields.items():
            if name.startswith('measured_'):
                assert not field.writable and field.index_space == 'native_body'
        base = read(runtime)
        np.testing.assert_allclose(base['measured_body_specific_acceleration'][:, 1],
            np.tile([-.8, .2*alpha, 9.81], (nworld, 1)), atol=1e-6, rtol=0)
        np.testing.assert_allclose(base['measured_body_internal_wrench'][:, 1, 2], 19.62, atol=1e-6, rtol=0)
        runtime.prepare_step_graph()
        with monkeypatch.context() as patch:
            patch.setattr(tx, 'empty', lambda *a, **k: (_ for _ in ()).throw(AssertionError('hot allocation')))
            patch.setattr(tx.Array, 'numpy', lambda *a, **k: (_ for _ in ()).throw(AssertionError('hot snapshot')))
            ticket = runtime.begin_execution()
            runtime.enqueue_step(ticket, steps=4, execution_mode='cuda_graph')
            runtime.complete_execution(ticket)
            runtime.accept_execution(ticket)
        after = read(runtime)
        # Last solve occurs at .03, before the fourth semi-implicit Euler integration.
        theta = .01 * (3*2 + .01*alpha*6)
        omega = 2 + .03*alpha
        point = .2*np.array([np.cos(theta), np.sin(theta), 0])
        expected = np.cross([0, 0, alpha], point) - omega**2*point + [0, 0, 9.81]
        np.testing.assert_allclose(after['measurement_sample_time'], .03, atol=1e-7, rtol=0)
        np.testing.assert_allclose(after['measured_body_specific_acceleration'][:, 1],
                                   np.tile(expected, (nworld, 1)), atol=1e-6, rtol=0)
        np.testing.assert_allclose(after['measured_body_com_twist'][:, 1, 5], omega, atol=1e-6, rtol=0)
        # Forward samples the endpoint without advancing physical time.
        runtime.forward()
        observed = read(runtime)
        np.testing.assert_allclose(observed['measurement_sample_time'], .04, atol=1e-7, rtol=0)
        assert runtime.execution_costs()['body_measurement_evaluations'] == 2
        ticket = runtime.begin_execution()
        storage.borrow('qfrc_applied', runtime.writable_bank, write=True).zero_(stream=storage.stream)
        runtime.enqueue_update(ticket)
        runtime.complete_execution(ticket)
        runtime.accept_execution(ticket)
        for name, value in read(runtime).items():
            np.testing.assert_array_equal(value, observed[name])
        selected = np.arange(nworld) % 2 == 0
        runtime.reset(selected)
        for name, value in read(runtime).items():
            np.testing.assert_array_equal(value[~selected], observed[name][~selected])
            np.testing.assert_allclose(value[selected], base[name][selected], atol=1e-6, rtol=0)
        runtime.reset()
        for _ in range(4):
            runtime.step()
        for name, value in read(runtime).items():
            np.testing.assert_allclose(value, after[name], atol=1e-6, rtol=0)
        print('B6C_MEASUREMENT', nworld, runtime.execution_costs())


@pytest.mark.cuda
@pytest.mark.parametrize('nworld', [1, 8, 64])
def test_freefall_specific_force_and_reaction_are_zero(payload, nworld):
    record = compile_mujoco_xml(FREE, require_asset=False)
    with PhySolveRuntime(record, payload=payload, nworld=nworld, body_measurements=True) as runtime:
        for _ in range(3):
            runtime.step()
        fields = read(runtime)
        np.testing.assert_allclose(fields['measured_body_specific_acceleration'][:, 1], 0, atol=1e-6, rtol=0)
        np.testing.assert_allclose(fields['measured_body_internal_wrench'][:, 1], 0, atol=1e-6, rtol=0)


@pytest.mark.cuda
def test_measurement_failure_rejects_publication_and_recovers(payload, monkeypatch):
    import tyx as tx
    from physolve_tyx import PostConstraintWorkspace

    record = compile_mujoco_xml(HINGE, require_asset=False)
    with PhySolveRuntime(record, payload=payload, body_measurements=True) as runtime:
        initial = read(runtime)
        evaluate = PostConstraintWorkspace.evaluate
        def poisoned(workspace, index):
            evaluate(workspace, index)
            workspace.cacc.view(tx.f32).fill_(float('nan'))
        with monkeypatch.context() as patch:
            patch.setattr(PostConstraintWorkspace, 'evaluate', poisoned)
            ticket = runtime.begin_execution()
            runtime.enqueue_forward(ticket)
            with pytest.raises(RuntimeError, match='non-finite.*body measurement'):
                runtime.complete_execution(ticket)
            with pytest.raises(RuntimeError, match='completed'):
                runtime.accept_execution(ticket)
            runtime.discard_execution(ticket)
        for name, value in read(runtime).items():
            np.testing.assert_array_equal(value, initial[name])
        runtime.forward()
        for name, value in read(runtime).items():
            np.testing.assert_allclose(value, initial[name], atol=1e-6, rtol=0)
        binding = runtime.storage_binding()
    with pytest.raises(RuntimeError, match='stale|closed'):
        binding.require_valid()


@pytest.mark.cuda
def test_measurements_disabled_preserves_default_cost(payload):
    record = compile_mujoco_xml(HINGE, require_asset=False)
    with PhySolveRuntime(record, payload=payload) as runtime:
        assert read(runtime) == {}
        assert runtime.execution_costs()['body_measurement_device_bytes'] == 0
        runtime.step()
        assert runtime.execution_costs()['body_measurement_evaluations'] == 0


@pytest.mark.cuda
def test_measurement_prepare_failure_retains_owner_for_cleanup_retry(payload, monkeypatch):
    from phymeo.backends._physolve_measurements import PreparedMeasurements

    record = compile_mujoco_xml(HINGE, require_asset=False)
    prepare = PreparedMeasurements.prepare
    native = payload.load()

    def fail_after_submit(owner, stream, mask):
        prepare(owner, stream, mask)
        raise RuntimeError('measurement prepare injection')

    with monkeypatch.context() as patch:
        patch.setattr(PreparedMeasurements, 'prepare', fail_after_submit)
        patch.setattr(native, '_close_runtime',
                      lambda *a: (_ for _ in ()).throw(RuntimeError('cleanup injection')))
        with pytest.raises(RuntimeError, match='measurement prepare injection') as caught:
            PhySolveRuntime(record, payload=payload, body_measurements=True)
        retained = caught.value.runtime
        assert retained._body_measurements is not None
        assert retained._body_measurements.raw[0].ptr
    retained.close()
    assert retained._body_measurements is None


@pytest.mark.cuda
def test_measurements_external_stream_selected_update_and_close_retry(payload, monkeypatch):
    import gc

    import tyx as tx

    record = compile_mujoco_xml(HINGE, require_asset=False)
    external = tx.CudaStream()
    try:
        runtime = PhySolveRuntime(record, payload=payload, nworld=8, body_measurements=True)
        try:
            runtime.bind_stream(int(external.handle), owner=external)
            runtime.prepare_step_graph()
            gc.collect()
            ticket = runtime.begin_execution()
            runtime.enqueue_step(ticket, steps=4, execution_mode='cuda_graph')
            runtime.complete_execution(ticket)
            runtime.accept_execution(ticket)
            before = read(runtime)
            storage = runtime.storage_binding()
            mask = tx.Array(np.arange(8) % 2 == 0, device='cuda:0')
            values = np.zeros((8, 1), np.float32)
            values[::2] = 2
            ticket = runtime.begin_execution()
            storage.borrow('qvel', runtime.writable_bank, write=True).assign(values, stream=storage.stream)
            runtime.enqueue_update(ticket, refresh_state=True, selected_worlds=mask)
            runtime.complete_execution(ticket)
            runtime.accept_execution(ticket)
            observed = read(runtime)
            for name in observed:
                np.testing.assert_array_equal(observed[name][1::2], before[name][1::2])
            np.testing.assert_allclose(observed['measured_body_com_twist'][::2, 1, 5], 2, atol=1e-6, rtol=0)
            runtime.bind_stream(0)
            runtime.forward()
            owner = runtime._body_measurements
            with monkeypatch.context() as patch:
                patch.setattr(runtime._native, '_close_runtime',
                              lambda *a: (_ for _ in ()).throw(RuntimeError('measurement close injection')))
                with pytest.raises(RuntimeError, match='measurement close injection'):
                    runtime.close()
            assert runtime._body_measurements is owner and owner.raw[0].ptr
            runtime.close()
            assert runtime._body_measurements is None
            with pytest.raises(RuntimeError, match='stale|closed'):
                storage.require_valid()
        finally:
            runtime.close()
    finally:
        external.destroy()
