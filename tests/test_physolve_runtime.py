"""Actual frozen native Host; optional explicit local payload, never a mock solver."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from phymeo import PhySolvePayload, PhySolveRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.builders.mujoco_compiler import compile_mujoco


FREE = '''<mujoco><option timestep=".002" integrator="Euler"/>
<worldbody><body pos="0 0 1"><freejoint/>
<geom type="sphere" size=".1" mass="2"/></body></worldbody></mujoco>'''
SLIDER = '''<mujoco><option timestep=".002" gravity="0 0 0"/>
<worldbody><body><joint type="slide" axis="1 0 0"/>
<geom type="sphere" size=".1" mass="2"/></body></worldbody></mujoco>'''
CONTACT = '''<mujoco><option timestep=".002" iterations="10" ls_iterations="10"/>
<worldbody><geom name="ground" type="plane" size="0 0 1"/>
<body pos="0 0 .08"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
</worldbody></mujoco>'''


@pytest.fixture(scope="module")
def payload():
    path = os.environ.get("PHYMEO_PHYSOLVE_SOURCES")
    if not path:
        pytest.skip("explicit PHYMEO_PHYSOLVE_SOURCES required for native qualification")
    root = Path(path).resolve().parent
    sources = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    frozen = json.loads((root / "evidence/payload-baseline.json").read_text(encoding="utf-8-sig"))
    return PhySolvePayload(sources["native_host"], sources["cubins"],
                          frozen["native_host"]["sha256"],
                          tuple((row["name"], row["sha256"]) for row in frozen["cubins"]))


def test_runtime_requires_compilation_and_explicit_payload():
    with pytest.raises(TypeError, match="compilation|Compilation"):
        PhySolveRuntime(None, payload=None)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("dtype_name", ["f32", "i32", "vec3f"])
def test_frontend_metadata_and_batch_view_ownership(payload, device, dtype_name):
    from importlib.metadata import version

    if version("physolve-tyx-frontend") == "0.0.0a7":
        pytest.skip("immutable frontend specs are introduced by the a8/TYX 0.2 pair")
    import dataclasses
    import gc
    import weakref
    import tyx as tx
    from physolve_tyx.specs import array_spec, batch_view, is_array_spec

    dtype = getattr(tx, dtype_name)
    spec = array_spec(("*", "nbody"), dtype)
    assert spec.shape == ("*", "nbody") and spec.dtype == dtype
    assert is_array_spec(spec) and is_array_spec(tx.array(dtype=dtype))
    assert not is_array_spec(int)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.shape = (9,)
    owner = tx.zeros((2, 2), dtype=dtype, device=device)
    values = owner.numpy()
    values[0] = 3
    values[1] = 7
    owner.assign(values)
    view = batch_view(owner)
    ref = weakref.ref(owner.storage)
    assert view.ptr == owner.ptr and view.dtype == owner.dtype
    assert view.shape == owner.shape and view.strides == (0, owner.strides[1])
    assert view._is_batched
    np.testing.assert_array_equal(owner.numpy(), values)
    del owner
    gc.collect()
    assert ref() is not None
    np.testing.assert_array_equal(view.numpy(), np.repeat(values[:1], 2, axis=0))
    del view
    gc.collect()
    assert ref() is None


@pytest.mark.parametrize('nworld', [1, 8, 64])
@pytest.mark.parametrize('graph', [False, True])
def test_follower_servo_affine_commands_parameters_reset_graph(payload, tmp_path, nworld, graph, monkeypatch):
    from dataclasses import replace
    from test_native_mimic import asset, compile_asset
    template = compile_asset(replace(asset(tmp_path), prepare_follower_drives=True))
    with template.bind(nworld).create_runtime(payload=payload) as runtime:
        if graph:
            runtime.prepare_step_graph(steps=4)
        storage = runtime.storage_binding()
        assert storage.borrow('target_position', runtime.active_bank).shape == (nworld, 1)
        captures = runtime.execution_costs()['step_graph_captures']
        runtime.update_scalar_drive_parameters({'arm::follower': {'kp':7., 'kd':2., 'effort_limit':0.}})
        ticket = runtime.begin_execution()
        storage.borrow('target_position', runtime.writable_bank, write=True).assign(
            np.full((nworld,1),.4,np.float32),stream=storage.stream)
        storage.borrow('target_velocity', runtime.writable_bank, write=True).assign(
            np.full((nworld,1),.5,np.float32),stream=storage.stream)
        runtime.enqueue_update(ticket,compose_controls=True)
        runtime.complete_execution(ticket)
        runtime.accept_execution(ticket)
        # follower qtarget=-2*.4+.1, vtarget=-2*.5; no independent command slot.
        np.testing.assert_allclose(runtime.snapshot()['ctrl'][:,1],-6.9,rtol=0,atol=1e-6)
        before=runtime.snapshot()
        parameters=runtime.scalar_drive_parameters()
        with pytest.raises(RuntimeError, match='non-finite PhySolve ctrl'):
            runtime.update_scalar_drive_parameters({'arm::follower': {'kp':3e38, 'kd':3e38}})
        assert runtime.scalar_drive_parameters() == parameters
        for name, values in before.items():
            np.testing.assert_array_equal(runtime.snapshot()[name], values)
        with monkeypatch.context() as patch:
            def fail(*args,**kwargs):
                raise RuntimeError('injected follower publication failure')
            patch.setattr(runtime,'enqueue_update',fail)
            with pytest.raises(RuntimeError,match='injected'):
                runtime.update_scalar_drive_parameters({'arm::follower':{'kp':9.}})
        assert runtime.scalar_drive_parameters()==parameters
        np.testing.assert_array_equal(runtime.snapshot()['ctrl'],before['ctrl'])
        ticket=runtime.begin_execution()
        runtime.enqueue_step(ticket,steps=4,execution_mode='cuda_graph' if graph else 'direct')
        runtime.complete_execution(ticket)
        runtime.accept_execution(ticket)
        before=runtime.snapshot()
        mask=np.arange(nworld)%2==0
        runtime.reset(mask)
        after=runtime.snapshot()
        np.testing.assert_allclose(after['ctrl'][mask,1],-2.1,rtol=0,atol=1e-6)
        for field in ('qpos','qvel','ctrl','qfrc_actuator'):
            np.testing.assert_array_equal(after[field][~mask],before[field][~mask])
        runtime.update_scalar_drive_parameters({'arm::follower':{'kp':0.,'kd':0.}})
        np.testing.assert_array_equal(runtime.snapshot()['ctrl'][:,1],0.)
        runtime.reset()
        np.testing.assert_array_equal(runtime.snapshot()['ctrl'][:,1],0.)
        assert runtime.execution_costs()['step_graph_captures']==captures


def _drive_binding(tmp_path, nworld, target=.1):
    from phymeo.builders.native_assembly import NativeAssetInstance, NativeJointInput, compile_native_template
    source = tmp_path / 'drive.urdf'
    source.write_text('''<robot name="slider"><link name="base"/><link name="link">
      <inertial><mass value="2"/><inertia ixx=".1" iyy=".1" izz=".1" ixy="0" ixz="0" iyz="0"/></inertial>
      <collision><geometry><sphere radius=".1"/></geometry></collision></link>
      <joint name="slide" type="prismatic"><parent link="base"/><child link="link"/>
      <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="0" velocity="2"/></joint></robot>''')
    asset = NativeAssetInstance('arm', str(source), drive_mode='scalar_pd',
        joint_defaults=(('kp', 10.), ('kd', 0.), ('armature', 0.), ('frictionloss', 0.)),
        joints=(NativeJointInput('slide', target_position=target),))
    return compile_native_template((asset,), timestep=.002, gravity=(0, 0, 0)).bind(nworld)


@pytest.mark.parametrize('nworld', [1, 8, 64])
@pytest.mark.parametrize('graph', [False, True])
def test_scalar_drive_parameter_update_rollback_reset_and_graph(payload, tmp_path, nworld, graph, monkeypatch):
    binding = _drive_binding(tmp_path, nworld)
    with binding.create_runtime(payload=payload) as runtime:
        if graph:
            runtime.prepare_step_graph(steps=4)
        baseline = runtime.execution_costs()
        runtime.update_scalar_drive_parameters({'arm::slide': {'kp': 80., 'kd': 2., 'effort_limit': 1.}})
        np.testing.assert_allclose(runtime.snapshot()['qfrc_actuator'], 1., rtol=0, atol=1e-6)
        runtime.update_scalar_drive_parameters({'arm::slide': {'effort_limit': 0.}})
        np.testing.assert_allclose(runtime.snapshot()['qfrc_actuator'], 8., rtol=0, atol=1e-6)
        metadata = runtime.scalar_drive_parameters()
        snapshot = runtime.snapshot()
        with monkeypatch.context() as patch:
            def reject(*args, **kwargs):
                raise RuntimeError('injected parameter forward failure')
            patch.setattr(runtime, 'enqueue_update', reject)
            with pytest.raises(RuntimeError, match='injected'):
                runtime.update_scalar_drive_parameters({'arm::slide': {'kp': 20.}})
        assert runtime.scalar_drive_parameters() == metadata
        for name in ('qpos', 'qvel', 'time', 'qfrc_actuator'):
            np.testing.assert_array_equal(runtime.snapshot()[name], snapshot[name])
        ticket = runtime.begin_execution()
        with pytest.raises(RuntimeError, match='pending'):
            runtime.update_scalar_drive_parameters({'arm::slide': {'kp': 30.}})
        runtime.enqueue_step(ticket, steps=4, execution_mode='cuda_graph' if graph else 'direct')
        runtime.complete_execution(ticket)
        runtime.accept_execution(ticket)
        assert np.all(runtime.snapshot()['qvel'] > 0.)
        mask = np.arange(nworld) % 2 == 0
        before = runtime.snapshot()
        runtime.reset(mask)
        after = runtime.snapshot()
        np.testing.assert_array_equal(after['qpos'][mask], 0.)
        np.testing.assert_array_equal(after['qpos'][~mask], before['qpos'][~mask])
        np.testing.assert_allclose(after['qfrc_actuator'][mask], 8., rtol=0, atol=1e-6)
        assert runtime.scalar_drive_parameters() == metadata
        costs = runtime.execution_costs()
        assert costs['drive_parameter_updates'] == 2
        assert costs['drive_parameter_rollbacks'] == 1
        assert costs['step_graph_captures'] == baseline['step_graph_captures']
        assert costs['step_graph_replays'] == int(graph)


def test_scalar_drive_parameter_capture_and_derived_overflow_admission(payload, tmp_path, monkeypatch):
    import tyx as tx

    with _drive_binding(tmp_path, 8, target=4.).create_runtime(payload=payload) as runtime:
        expected = runtime.scalar_drive_parameters()
        storage = runtime.storage_binding()
        graph = tx.Graph(stream=storage.stream, mode='cuda_stream')
        try:
            with graph:
                with pytest.raises(RuntimeError, match='capture'):
                    runtime.update_scalar_drive_parameters({'arm::slide': {'kp': 20.}})
        finally:
            graph.destroy()
        assert runtime.scalar_drive_parameters() == expected
        for target in (4., 0.):  # second case isolates the reset baseline
            ticket = runtime.begin_execution()
            storage.borrow('target_position', runtime.writable_bank, write=True).assign(
                np.full((8, 1), target, np.float32), stream=storage.stream)
            runtime.enqueue_update(ticket, compose_controls=True)
            runtime.complete_execution(ticket)
            runtime.accept_execution(ticket)
            before = runtime.snapshot()
            with monkeypatch.context() as patch:
                def forbidden(*args):
                    raise AssertionError('invalid derived ctrl reached native forward')
                patch.setattr(runtime._native, '_forward_runtime', forbidden)
                with pytest.raises(RuntimeError, match='non-finite PhySolve ctrl'):
                    runtime.update_scalar_drive_parameters({'arm::slide': {'kp': 3e38}})
            assert runtime.scalar_drive_parameters() == expected
            for name, values in before.items():
                np.testing.assert_array_equal(runtime.snapshot()[name], values)
        runtime.update_scalar_drive_parameters({'arm::slide': {'kp': 20.}})
        runtime.reset()
        np.testing.assert_allclose(runtime.snapshot()['ctrl'], 80., rtol=0, atol=1e-6)


@pytest.mark.parametrize('nworld', [1, 8, 64])
@pytest.mark.parametrize('graph', [False, True])
def test_staged_updates_share_candidate_masks_reset_and_step(payload, tmp_path, nworld, graph, monkeypatch):
    import tyx as tx
    with _drive_binding(tmp_path, nworld).create_runtime(payload=payload) as runtime:
        if graph:
            runtime.prepare_step_graph(steps=4)
        runtime.step()
        before = runtime.snapshot()
        storage = runtime.storage_binding()
        selected = np.zeros(nworld, np.bool_)
        selected[-1] = True
        mask = tx.Array(selected, device=storage.device)
        costs = runtime.execution_costs()
        original_forward = runtime._native._forward_runtime
        calls = []
        def forward(handle):
            calls.append(handle)
            return original_forward(handle)
        monkeypatch.setattr(runtime._native, '_forward_runtime', forward)
        ticket = runtime.begin_execution()
        bank = runtime.writable_bank
        storage.borrow('target_position', bank, write=True).assign(
            np.full((nworld, 1), .6, np.float32), stream=storage.stream)
        runtime.enqueue_update(ticket, compose_controls=True, defer_refresh=True)
        # Only state-edit worlds may refresh solve-time observations.
        q = before['qpos'].copy()
        q[-1, 0] = .1
        storage.borrow('qpos', bank, write=True).assign(q, stream=storage.stream)
        runtime.enqueue_update(ticket, refresh_state=True, selected_worlds=mask, defer_refresh=True)
        mask.zero_(stream=storage.stream)  # The union must not borrow this last caller mask.
        assert runtime.writable_bank == bank and runtime.active_bank != bank
        assert not calls
        assert runtime.execution_costs()['validation_readbacks'] == costs['validation_readbacks']
        runtime.complete_execution(ticket)
        runtime.accept_execution(ticket)
        assert len(calls) == 1
        after = runtime.snapshot()
        np.testing.assert_array_equal(after['qpos'], q)
        for field in ('qacc', 'qfrc_actuator', 'qfrc_constraint'):
            np.testing.assert_array_equal(after[field][~selected], before[field][~selected])
        np.testing.assert_allclose(after['qfrc_actuator'][selected], 5., rtol=0, atol=1e-6)
        # Ordered reset restores selected integration, then a later edit wins.
        ticket = runtime.begin_execution()
        runtime.enqueue_reset(ticket, selected_worlds=mask, defer_refresh=True)
        mask.assign(selected, stream=storage.stream)
        runtime.enqueue_reset(ticket, selected_worlds=mask, defer_refresh=True)
        target = np.full((nworld, 1), .4, np.float32)
        storage.borrow('target_position', runtime.writable_bank, write=True).assign(target, stream=storage.stream)
        runtime.enqueue_update(ticket, compose_controls=True, defer_refresh=True)
        calls.clear()
        runtime.enqueue_step(ticket, steps=4, execution_mode='cuda_graph' if graph else 'direct')
        assert not calls  # step consumes the composed controls without an extra forward
        runtime.complete_execution(ticket)
        runtime.accept_execution(ticket)
        assert np.all(runtime.snapshot()['qvel'] > 0)
        assert runtime.execution_costs()['staged_update_materializations'] == 1


@pytest.mark.parametrize('nworld', [1, 8, 64])
@pytest.mark.parametrize('graph', [False, True])
def test_command_continuation_preserves_candidate_observations(payload, tmp_path, nworld, graph, monkeypatch):
    import tyx as tx
    binding = _drive_binding(tmp_path, nworld)
    with binding.create_runtime(payload=payload) as runtime, binding.create_runtime(payload=payload) as reference:
        if graph:
            runtime.prepare_step_graph(steps=4)
            reference.prepare_step_graph(steps=4)
        mode, steps = ('cuda_graph', 4) if graph else ('direct', 1)
        storage = runtime.storage_binding()
        target = np.full((nworld, 1), .4, np.float32)
        before = runtime.execution_costs()
        ticket = runtime.begin_execution()
        bank = runtime.writable_bank
        def forbidden(*args, **kwargs):
            raise AssertionError('continued command allocated, copied baseline or forwarded')
        with monkeypatch.context() as patch:
            patch.setattr(runtime, '_candidate', forbidden)
            patch.setattr(runtime._native, '_forward_runtime', forbidden)
            for _ in range(4):
                runtime.prepare_command_update(ticket)
                storage.borrow('target_position', bank, write=True).assign(target, stream=storage.stream)
                runtime.enqueue_update(ticket, compose_controls=True, defer_refresh=True)
                with patch.context() as hot:
                    hot.setattr(tx, 'empty', forbidden)
                    runtime.enqueue_step(ticket, steps=steps, execution_mode=mode)
            assert runtime.execution_costs()['validation_readbacks'] == before['validation_readbacks']
            observed = storage.borrow('qfrc_actuator', runtime.observation_bank).numpy(stream=storage.stream)
            runtime.prepare_command_update(ticket)
            storage.borrow('target_position', bank, write=True).assign(target, stream=storage.stream)
            runtime.enqueue_update(ticket, compose_controls=True, defer_refresh=True)
            runtime.flush_updates(ticket)
            np.testing.assert_array_equal(storage.borrow('qfrc_actuator', runtime.observation_bank).numpy(stream=storage.stream), observed)
        runtime.complete_execution(ticket)
        runtime.accept_execution(ticket)
        for _ in range(4):
            other = reference.begin_execution()
            refstorage = reference.storage_binding()
            refstorage.borrow('target_position', reference.writable_bank, write=True).assign(target, stream=refstorage.stream)
            reference.enqueue_update(other, compose_controls=True, defer_refresh=True)
            reference.enqueue_step(other, steps=steps, execution_mode=mode)
            reference.complete_execution(other)
            reference.accept_execution(other)
        actual, expected = runtime.snapshot(), reference.snapshot()
        for name in ('qpos', 'qvel', 'qacc', 'time', 'qfrc_actuator'):
            np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=1e-6)
        costs = runtime.execution_costs()
        assert costs['contact_measurement_copies'] == before['contact_measurement_copies']
        assert costs['body_measurement_copies'] == before['body_measurement_copies']
        # Refusing state/reset after materialization must leave a completable ticket.
        ticket = runtime.begin_execution()
        runtime.enqueue_step(ticket, steps=steps, execution_mode=mode)
        runtime.prepare_command_update(ticket)
        with pytest.raises(ValueError, match='completion'):
            runtime.enqueue_reset(ticket, defer_refresh=True)
        runtime.enqueue_step(ticket, steps=steps, execution_mode=mode)
        runtime.complete_execution(ticket)
        runtime.accept_execution(ticket)


def test_staged_updates_preparation_failure_keeps_launched_mask_owner(payload, monkeypatch):
    from phymeo.backends._physolve_updates import PreparedUpdates
    native = payload.load()
    with monkeypatch.context() as patch:
        def prepare(plan, stream):
            plan.worlds.zero_(stream=stream)
            raise RuntimeError('injected staged update initialization')
        def close(*args):
            raise RuntimeError('injected native close')
        patch.setattr(PreparedUpdates, 'prepare', prepare)
        patch.setattr(native, '_close_runtime', close)
        with pytest.raises(RuntimeError, match='staged update initialization') as caught:
            PhySolveRuntime(compile_mujoco_xml(FREE, require_asset=False), payload=payload)
        retained = caught.value.runtime
        assert retained._updates is not None and retained._updates.worlds.ptr
        assert retained._stream is not None and retained._banks
    retained.close()
    assert retained._closed and retained._updates is None


@pytest.mark.parametrize('nworld', [1, 8, 64])
def test_capacity_metadata_is_owned_and_requires_no_device_work(payload, nworld, monkeypatch):
    import tyx as tx
    from dataclasses import fields
    from tyx.cuda_stream import CudaStream

    record = compile_mujoco_xml(SLIDER, require_asset=False)
    with PhySolveRuntime(record, payload=payload, nworld=nworld, nconmax=32, njmax=96) as runtime:
        expected = sum(getattr(group, field.name).storage.nbytes
                       for bank in runtime._banks for group in (bank.contact, bank.efc)
                       for field in fields(group))
        def forbidden(*args, **kwargs):
            raise AssertionError('capacity metadata performed device work')
        with monkeypatch.context() as patch:
            patch.setattr(tx.Array, 'numpy', forbidden)
            patch.setattr(tx.Array, 'copy_to_host', forbidden)
            patch.setattr(CudaStream, 'synchronize', forbidden)
            patch.setattr(tx, 'empty', forbidden)
            report = runtime.capacity_info()
            assert report['partial_buffer_bytes'] == expected
            assert len(report['banks']) == 2
            assert [b['naconmax'] for b in report['banks']] == [32*nworld]*2
            report['banks'][0]['array_bytes'].clear()
            assert runtime.capacity_info()['banks'][0]['array_bytes']
        runtime.step()
        runtime.reset()
        assert runtime.capacity_info()['partial_buffer_bytes'] == expected
    with pytest.raises(RuntimeError, match='closed'):
        runtime.capacity_info()


def test_configuration_readback_is_owned_idle_only_and_capture_safe(payload):
    import tyx as tx

    record = compile_mujoco_xml(SLIDER, require_asset=False)
    runtime = PhySolveRuntime(record, payload=payload, nworld=8, nconmax=32, njmax=96)
    try:
        expected = runtime.configuration_info()
        assert expected['bank_capacities'] == [(8, 256, 96)] * 2
        assert expected['nworld'] == 8 and expected['device'] == 'cuda:0'
        changed = runtime.configuration_info()
        changed['device_gravity'][0][0] = 123
        changed['gravity'][0] = 123
        assert runtime.configuration_info() == expected
        ticket = runtime.begin_execution()
        with pytest.raises(RuntimeError, match='pending'):
            runtime.configuration_info()
        runtime.discard_execution(ticket)
        stream = runtime.storage_binding().stream
        graph = tx.Graph(stream=stream, mode='cuda_stream')
        try:
            with graph:
                with pytest.raises(RuntimeError, match='capture'):
                    runtime.configuration_info()
        finally:
            graph.destroy()
        assert runtime.configuration_info() == expected
    finally:
        runtime.close()
    with pytest.raises(RuntimeError, match='closed|closing'):
        runtime.configuration_info()


def test_payload_installed_dependency_contract(tmp_path, monkeypatch):
    import hashlib
    from phymeo.backends import physolve_payload as module

    versions = {"mujoco": "3.8.0", "physolve-tyx-frontend": "0.0.0a7",
                "tyx-compute": "0.1.0a0"}
    monkeypatch.setattr(module.metadata, "version", versions.__getitem__)
    digest = hashlib.sha256(b"identity-only fixture, never loaded").hexdigest()
    for name in (*module.CUBINS, "host.pyd"):
        (tmp_path / name).write_bytes(b"identity-only fixture, never loaded")
    value = PhySolvePayload(str(tmp_path / "host.pyd"), str(tmp_path), digest,
                           tuple((name, digest) for name in module.CUBINS))
    assert value.validate() == ((tmp_path / "host.pyd").resolve(), tmp_path.resolve())
    versions.update({"physolve-tyx-frontend": "0.0.0a8", "tyx-compute": "0.2.0a1"})
    assert value.validate() == ((tmp_path / "host.pyd").resolve(), tmp_path.resolve())
    for frontend, tyx in (("0.0.0a7", "0.2.0a1"), ("0.0.0a8", "0.1.0a0"),
                          ("0.0.0a8", "0.2.0a2"), ("0.0.0a9", "0.2.0a1")):
        versions.update({"physolve-tyx-frontend": frontend, "tyx-compute": tyx})
        with pytest.raises(RuntimeError, match="requires an exact"):
            value.validate()
    versions.update({"physolve-tyx-frontend": "0.0.0a7", "tyx-compute": "0.1.0a0"})
    for name, rejected in (("physolve-tyx-frontend", "0.0.0a6"),
                           ("physolve-tyx-frontend", "0.0.0a8"), ("mujoco", "3.7.0")):
        previous = versions[name]
        versions[name] = rejected
        with pytest.raises(RuntimeError, match="requires"):
            value.validate()
        versions[name] = previous
    (tmp_path / module.CUBINS[0]).write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="frozen payload mismatch"):
        value.validate()


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_freefall_forward_reset_and_masked_worlds(payload, nworld, monkeypatch):
    import mujoco
    record = compile_mujoco_xml(FREE, require_asset=False)
    with PhySolveRuntime(record, payload=payload, nworld=nworld) as runtime:
        initial = runtime.snapshot()
        def forbidden(*args, **kwargs):
            raise AssertionError("CPU simulation/forward after preparation is forbidden")
        monkeypatch.setattr(mujoco, "mj_step", forbidden)
        monkeypatch.setattr(mujoco, "mj_forward", forbidden)
        for _ in range(10):
            runtime.step()
        after = runtime.snapshot()
        np.testing.assert_allclose(after["qpos"][:, 2], 1 - 9.81 * .002**2 * 55,
                                   atol=1e-6, rtol=0)
        np.testing.assert_allclose(after["qvel"][:, 2], -9.81 * .002 * 10,
                                   atol=1e-6, rtol=0)
        runtime.forward()
        forward = runtime.snapshot()
        for field in ("qpos", "qvel", "time"):
            np.testing.assert_array_equal(forward[field], after[field])
        mask = np.arange(nworld) % 2 == 0
        runtime.reset(mask)
        reset = runtime.snapshot()
        for field in ("qpos", "qvel", "time"):
            np.testing.assert_array_equal(reset[field][mask], initial[field][mask])
            np.testing.assert_array_equal(reset[field][~mask], after[field][~mask])
        runtime.reset()
        for _ in range(10):
            runtime.step()
        repeated = runtime.snapshot()
        for field in ("qpos", "qvel", "time"):
            np.testing.assert_array_equal(repeated[field], after[field])
        with pytest.raises(ValueError):
            repeated["qpos"].flags.writeable = True
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.step()


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_joint_force_control_atomicity_failure_recovery(payload, nworld, monkeypatch):
    record = compile_mujoco_xml(SLIDER, require_asset=False)
    with PhySolveRuntime(record, payload=payload, nworld=nworld) as runtime:
        forces = np.linspace(-2, 2, nworld, dtype=np.float32)[:, None]
        for _ in range(10):
            runtime.step(qfrc_applied=forces)
        after = runtime.snapshot()
        np.testing.assert_allclose(after["qpos"][:, 0], forces[:, 0] / 2 * .002**2 * 55,
                                   atol=1e-6, rtol=0)
        np.testing.assert_array_equal(after["qfrc_applied"], forces)
        with pytest.raises(ValueError, match="float32"):
            runtime.step(qfrc_applied=np.full((nworld, 1), np.inf))
        with pytest.raises(ValueError, match="mask"):
            runtime.reset(np.zeros(nworld, dtype=np.int32))
        for field in after:
            np.testing.assert_array_equal(runtime.snapshot()[field], after[field])
        original = runtime._validate
        def rejected(data):
            original(data)
            raise RuntimeError("injected post-execution publication failure")
        monkeypatch.setattr(runtime, "_validate", rejected)
        with pytest.raises(RuntimeError, match="publication failure"):
            runtime.step()
        for field in after:
            np.testing.assert_array_equal(runtime.snapshot()[field], after[field])
        monkeypatch.setattr(runtime, "_validate", original)
        runtime.reset()
        runtime.step(qfrc_applied=forces)
        np.testing.assert_allclose(runtime.snapshot()["qpos"][:, 0],
                                   forces[:, 0] / 2 * .002**2, atol=1e-6, rtol=0)


@pytest.mark.cuda
def test_contact_native_response_and_capacity_preflight(payload):
    record = compile_mujoco_xml(CONTACT, require_asset=False)
    with pytest.raises(ValueError, match="capacity"):
        PhySolveRuntime(record, payload=payload, njmax=1)
    with PhySolveRuntime(record, payload=payload, nworld=8) as runtime:
        initial = runtime.snapshot()
        assert np.all(initial["qfrc_constraint"][:, 2] > 9.81)
        runtime.step()
        assert np.all(runtime.snapshot()["qpos"][:, 2] > initial["qpos"][:, 2])
        runtime.reset()
        repeated = runtime.snapshot()
        for field in ("qpos", "qvel", "qacc", "qfrc_constraint"):
            np.testing.assert_array_equal(repeated[field], initial[field])


@pytest.mark.cuda
@pytest.mark.parametrize("name", ["panda", "fanuc"])
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_robot_dual_bank_matches_native_single_bank(payload, name, nworld):
    import mujoco
    from test_physolve_models import robot_path
    record = compile_mujoco(robot_path(name), require_asset=False)
    with PhySolveRuntime(record, payload=payload, nworld=nworld, nconmax=128) as runtime:
        initial = runtime.snapshot()
        # Independent persistent native bank is the reference for history/state
        # migration, using exactly the same frozen CUBINs and frontend model.
        with runtime._scope():
            cpu = runtime._cpu_model
            data = mujoco.MjData(cpu)
            mujoco.mj_forward(cpu, data)
            reference = runtime._frontend.put_data(cpu, data, nworld=nworld,
                                                   nconmax=128, njmax=256)
            handle = runtime._native._create_runtime(runtime._model, reference, "tyx", "tyx")
            try:
                runtime._native._prepare_runtime(handle)
                runtime._native._forward_runtime(handle)
                runtime._complete()
                for field in initial:
                    np.testing.assert_allclose(initial[field], getattr(reference, field).numpy(),
                                               rtol=0, atol=1e-6,
                                               err_msg=f"initial forward: {field}")
                forces = np.broadcast_to(np.linspace(-.05, .05, cpu.nv, dtype=np.float32),
                                         (nworld, cpu.nv)).copy()
                forces *= np.arange(1, nworld + 1, dtype=np.float32)[:, None] / nworld
                for step_index in range(6):
                    reference.qfrc_applied.assign(forces, stream=runtime._stream)
                    runtime._native._step_runtime(handle)
                    runtime.step(qfrc_applied=forces)
                    actual = runtime.snapshot()
                    for field in actual:
                        np.testing.assert_allclose(actual[field], getattr(reference, field).numpy(),
                                                   rtol=0, atol=1e-6,
                                                   err_msg=f"step {step_index + 1}: {field}")
                before = runtime.snapshot()
                assert np.any(before["qvel"] != initial["qvel"])
                np.testing.assert_allclose(before["time"], 6 * cpu.opt.timestep,
                                           atol=1e-6, rtol=0)
                runtime.forward()
                for field in ("qpos", "qvel", "time"):
                    np.testing.assert_array_equal(runtime.snapshot()[field], before[field])
                mask = np.arange(nworld) % 2 == 0
                runtime.reset(mask)
                after = runtime.snapshot()
                for field in ("qpos", "qvel", "time", "qfrc_applied"):
                    np.testing.assert_array_equal(after[field][mask], initial[field][mask])
                    np.testing.assert_array_equal(after[field][~mask], before[field][~mask])
                runtime.reset()
                for _ in range(6):
                    runtime.step(qfrc_applied=forces)
                for field in ("qpos", "qvel", "time"):
                    np.testing.assert_array_equal(runtime.snapshot()[field], before[field])
            finally:
                runtime._complete()
                runtime._native._close_runtime(handle, True)


@pytest.mark.cuda
@pytest.mark.parametrize("integrator", [None, "implicitfast"])
def test_close_failure_retains_owner_for_retry(payload, monkeypatch, integrator):
    xml = SLIDER if integrator is None else SLIDER.replace(
        "<option ", f'<option integrator="{integrator}" '
    )
    runtime = PhySolveRuntime(compile_mujoco_xml(xml, require_asset=False), payload=payload)
    close = runtime._native._close_runtime
    calls = []
    def fail_second(handle, complete):
        calls.append(handle)
        if len(calls) == 2:
            raise RuntimeError("injected close failure")
        return close(handle, complete)
    monkeypatch.setattr(runtime._native, "_close_runtime", fail_second)
    try:
        with pytest.raises(RuntimeError, match="close failure"):
            runtime.close()
        assert runtime._handles[0] is None
        assert runtime._handles[1] is not None
        assert runtime._stream is not None and runtime._banks and runtime._model is not None
        with pytest.raises(RuntimeError, match="closing"):
            runtime.step()
    finally:
        monkeypatch.setattr(runtime._native, "_close_runtime", close)
        runtime.close()
    assert runtime._closed and not runtime._banks


@pytest.mark.cuda
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_panda_finger_box_contacts_match_cpu_geometry(payload, device):
    import mujoco
    from tyx.compat import warp as wp
    from physolve_tyx._src.collision_primitive_core import box_box
    from test_physolve_models import robot_path

    @wp.kernel
    def contacts(p: wp.array[wp.vec3], r: wp.array[wp.mat33], s: wp.array[wp.vec3],
                 positions: wp.array[wp.vec3], distances: wp.array[float]):
        dist, pos, normal = box_box(p[0], r[0], s[0], p[1], r[1], s[1])
        for index in range(8):
            positions[index] = pos[index]
            distances[index] = dist[index]

    model = mujoco.MjModel.from_xml_path(str(robot_path('panda')))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pair = [36, 40]
    expected = np.array([c.pos.copy() for c in data.contact if list(c.geom) == pair])
    assert len(expected) > 0
    p = wp.array(data.geom_xpos[pair].astype(np.float32), dtype=wp.vec3, device=device)
    r = wp.array(data.geom_xmat[pair].reshape(2, 3, 3).astype(np.float32), dtype=wp.mat33, device=device)
    s = wp.array(model.geom_size[pair].astype(np.float32), dtype=wp.vec3, device=device)
    pos, dist = wp.zeros(8, dtype=wp.vec3, device=device), wp.zeros(8, device=device)
    wp.launch(contacts, 1, inputs=[p, r, s, pos, dist], device=device)
    actual = pos.numpy()[dist.numpy() < 0]
    assert len(actual) > 0
    # Contact ordering/duplicate points can differ. Every produced point must
    # still lie at a CPU geometric contact, at the existing absolute tolerance.
    nearest = np.min(np.max(np.abs(actual[:, None] - expected[None]), axis=2), axis=1)
    np.testing.assert_array_less(nearest, 1e-6)


@pytest.mark.cuda
def test_actuator_control_and_two_runtime_isolation(payload):
    xml = SLIDER.replace('<joint type=', '<joint name="axis" type=')
    xml = xml.replace('</mujoco>', '<actuator><motor joint="axis"/></actuator></mujoco>')
    record = compile_mujoco_xml(xml, require_asset=False)
    with PhySolveRuntime(record, payload=payload, nworld=8) as first, \
            PhySolveRuntime(record, payload=payload, nworld=8) as second:
        initial = second.snapshot()
        assert first._stream is not second._stream
        command = np.full((8, 1), 2., dtype=np.float32)
        first.step(ctrl=command)
        np.testing.assert_allclose(first.snapshot()["qpos"][:, 0], .002**2,
                                   atol=1e-6, rtol=0)
        np.testing.assert_array_equal(first.snapshot()["qfrc_actuator"], command)
        np.testing.assert_array_equal(first.snapshot()["qfrc_applied"], np.zeros((8, 1)))
        for field, value in second.snapshot().items():
            np.testing.assert_array_equal(value, initial[field])
        first.close()
        second.step(ctrl=-command)
        np.testing.assert_allclose(second.snapshot()["qpos"][:, 0], -.002**2,
                                   atol=1e-6, rtol=0)
        second.reset()
        np.testing.assert_array_equal(second.snapshot()["ctrl"], initial["ctrl"])


@pytest.mark.cuda
@pytest.mark.parametrize("name", ["panda", "fanuc"])
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_robot_forward_and_masked_reset_native_state(payload, name, nworld):
    from test_physolve_models import robot_path
    record = compile_mujoco(robot_path(name), require_asset=False)
    with PhySolveRuntime(record, payload=payload, nworld=nworld, nconmax=128) as runtime:
        initial = runtime.snapshot()
        force = np.full((nworld, record.mj_model.nv), .03, dtype=np.float32)
        for _ in range(3):
            runtime.step(qfrc_applied=force)
        advanced = runtime.snapshot()
        assert np.any(advanced["qvel"] != initial["qvel"])
        runtime.forward()
        for field in ("qpos", "qvel", "time"):
            np.testing.assert_array_equal(runtime.snapshot()[field], advanced[field])
        mask = np.arange(nworld) % 2 == 0
        runtime.reset(mask)
        reset = runtime.snapshot()
        for field in ("qpos", "qvel", "time", "qfrc_applied"):
            np.testing.assert_array_equal(reset[field][mask], initial[field][mask])
            np.testing.assert_array_equal(reset[field][~mask], advanced[field][~mask])
        runtime.reset()
        for field in ("qpos", "qvel", "time", "qfrc_applied"):
            np.testing.assert_array_equal(runtime.snapshot()[field], initial[field])


def test_follower_without_independent_control_rejected_before_native_work():
    with pytest.raises(ValueError, match="require independent scalar controls"):
        PhySolveRuntime(None, payload=None, follower_drives=(("follower",),))


def test_control_upload_failure_retains_owner_until_close_retry(payload, monkeypatch, tmp_path):
    from phymeo.backends._physolve_controls import PreparedControls
    from test_native_mimic import asset, compile_asset
    from dataclasses import replace
    template = compile_asset(replace(asset(tmp_path), prepare_follower_drives=True))
    initialize = PreparedControls.initialize
    native = payload.load()
    with monkeypatch.context() as patch:
        def fail(plan, *args, **kwargs):
            initialize(plan, *args, **kwargs)
            plan.sources.zero_()
            raise RuntimeError("injected control upload failure")
        def close(*args):
            raise RuntimeError("injected native close")
        patch.setattr(PreparedControls, "initialize", fail)
        patch.setattr(native, "_close_runtime", close)
        with pytest.raises(RuntimeError, match="control upload failure") as caught:
            template.bind(1).create_runtime(payload=payload)
        retained = caught.value.runtime
        assert retained._controls_binding.sources.ptr
        assert retained._controls_binding.affine.ptr
        assert retained._stream is not None
    retained.close()
    assert retained._closed and retained._controls_binding is None
