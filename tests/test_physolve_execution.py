"""Real CUDA execution contract; no Newton/Warp and no snapshot-based executor."""

import numpy as np
import pytest
from test_physolve_runtime import CONTACT, FREE, SLIDER
from test_physolve_runtime import payload as payload_fixture

from phymeo import PhySolveRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

payload = payload_fixture


@pytest.mark.cuda
@pytest.mark.parametrize('nworld', [1, 8, 64])
def test_device_selected_update_preserves_unselected_observations(payload, nworld, monkeypatch):
    import tyx as tx
    with PhySolveRuntime(compile_mujoco_xml(CONTACT, require_asset=False), payload=payload,
                        nworld=nworld, body_observations=True, contact_observations=True) as runtime:
        runtime.step()
        storage = runtime.storage_binding()
        names = ('qpos', 'qvel', 'qacc', 'qfrc_actuator', 'qfrc_constraint',
                 'body_position', 'body_com_twist', 'contact_sample_time')
        before = {name: storage.borrow(name, runtime.active_bank).numpy(stream=storage.stream).copy() for name in names}
        contacts = contact_records(runtime)
        mask = np.arange(nworld) % 2 == 0
        device_mask = tx.Array(mask, device=storage.device)
        ticket = runtime.begin_execution()
        bank = runtime.writable_bank
        q = before['qpos'].copy()
        q[mask, 2] += .2
        storage.borrow('qpos', bank, write=True).assign(q, stream=storage.stream)
        with monkeypatch.context() as patch:
            patch.setattr(tx, 'empty', lambda *a, **k: pytest.fail('hot allocation'))
            patch.setattr(tx.Array, 'numpy', lambda *a, **k: pytest.fail('hot snapshot'))
            runtime.enqueue_update(ticket, refresh_state=True, selected_worlds=device_mask)
            runtime.complete_execution(ticket)
            runtime.accept_execution(ticket)
        for name, old in before.items():
            actual = storage.borrow(name, runtime.active_bank).numpy(stream=storage.stream)
            np.testing.assert_array_equal(actual[~mask], old[~mask])
        np.testing.assert_array_equal(storage.borrow('qpos', runtime.active_bank).numpy(), q)
        after = contact_records(runtime)
        # Contact row order is not an identity contract; this model has one
        # pair per world. Compare retained records by world, without sorting production.
        old_rows = np.flatnonzero(~mask[contacts['contact_world'].reshape(-1)])
        new_rows = np.flatnonzero(~mask[after['contact_world'].reshape(-1)])
        old_rows = old_rows[np.argsort(contacts['contact_world'].reshape(-1)[old_rows])]
        new_rows = new_rows[np.argsort(after['contact_world'].reshape(-1)[new_rows])]
        for name in contacts:
            np.testing.assert_array_equal(after[name][new_rows], contacts[name][old_rows])
        ticket = runtime.begin_execution()
        with pytest.raises(ValueError, match='bool'):
            runtime.enqueue_update(ticket, selected_worlds=mask)
        runtime.discard_execution(ticket)


def contact_records(runtime):
    storage = runtime.storage_binding()
    def read(name):
        return storage.borrow(name, runtime.active_bank).numpy(stream=storage.stream).copy()
    count = int(read('contact_count')[0])
    return {name: read(name)[:count] for name in (
        'contact_world', 'contact_geom', 'contact_dim', 'contact_position',
        'contact_frame', 'contact_distance', 'contact_wrench', 'contact_wrench_local')}


@pytest.mark.cuda
@pytest.mark.parametrize('nworld', [1, 8, 64])
def test_contact_measurements_force_and_masked_publication(payload, nworld, monkeypatch):
    import tyx as tx
    with PhySolveRuntime(compile_mujoco_xml(CONTACT, require_asset=False), payload=payload,
                        nworld=nworld, contact_observations=True) as runtime:
        initial = contact_records(runtime)
        assert len(initial['contact_world']) == nworld
        np.testing.assert_array_equal(initial['contact_geom'], np.tile([0, 1], (nworld, 1)))
        assert np.all(initial['contact_wrench'][:, 2] > 0)  # force on the ball
        np.testing.assert_allclose(initial['contact_wrench'][:, :2], 0, atol=1e-6, rtol=0)
        np.testing.assert_allclose(initial['contact_wrench_local'][:, 0], initial['contact_wrench'][:, 2], atol=1e-6, rtol=0)
        def forbidden(*args, **kwargs):
            raise AssertionError('contact publication must not allocate/read full host arrays')
        with monkeypatch.context() as patch:
            patch.setattr(tx.Array, 'numpy', forbidden)
            patch.setattr(tx, 'empty', forbidden)
            ticket = runtime.begin_execution()
            runtime.enqueue_step(ticket, steps=4)
            runtime.complete_execution(ticket)
            runtime.accept_execution(ticket)
        before = contact_records(runtime)
        times = runtime.storage_binding().borrow('contact_sample_time', runtime.active_bank).numpy()
        np.testing.assert_allclose(times, .006, atol=1e-9, rtol=0)
        assert runtime.execution_costs()['contact_measurement_extractions'] == 1
        assert runtime.execution_costs()['validation_d2h_bytes'] == 4
        ticket = runtime.begin_execution()
        runtime.enqueue_update(ticket)
        runtime.complete_execution(ticket)
        runtime.accept_execution(ticket)
        for name, value in before.items():
            np.testing.assert_array_equal(contact_records(runtime)[name], value)
        mask = np.arange(nworld) % 2 == 0
        runtime.reset(mask)
        after = contact_records(runtime)
        times = runtime.storage_binding().borrow('contact_sample_time', runtime.active_bank).numpy()
        np.testing.assert_allclose(times, np.where(mask, 0., .006), atol=1e-9, rtol=0)
        for world in range(nworld):
            expected = initial if mask[world] else before
            selection = after['contact_world'][:, 0] == world
            reference = expected['contact_world'][:, 0] == world
            for name in after:
                np.testing.assert_array_equal(after[name][selection], expected[name][reference])
        counts = runtime.storage_binding().borrow('contact_count_by_world', runtime.active_bank).numpy()
        np.testing.assert_array_equal(counts, np.bincount(after['contact_world'][:, 0], minlength=nworld))
        runtime.forward()
        assert np.isfinite(contact_records(runtime)['contact_wrench']).all()
        print('B3G_CONTACT_COST', nworld, runtime.execution_costs())


@pytest.mark.cuda
def test_contact_validation_rejects_candidate_and_recovers(payload, monkeypatch):
    with PhySolveRuntime(compile_mujoco_xml(CONTACT, require_asset=False), payload=payload,
                        contact_observations=True) as runtime:
        original = contact_records(runtime)
        operation = runtime._contact_observations.update
        for fault in ('index', 'address', 'nan'):
            def injected(index, stream, **kwargs):
                data = runtime._banks[index]
                if fault == 'index':
                    data.contact.worldid.assign(np.full(data.contact.worldid.shape, 9, np.int32), stream=stream)
                elif fault == 'address':
                    data.contact.efc_address.assign(np.full(data.contact.efc_address.shape, 100, np.int32), stream=stream)
                else:
                    data.contact.pos.view(runtime._tx.f32).assign(
                        np.full((data.naconmax, 3), np.nan, np.float32), stream=stream)
                operation(index, stream, **kwargs)
            ticket = runtime.begin_execution()
            with monkeypatch.context() as patch:
                patch.setattr(runtime._contact_observations, 'update', injected)
                runtime.enqueue_forward(ticket)
                with pytest.raises(RuntimeError, match='contact'):
                    runtime.complete_execution(ticket)
            for name, value in original.items():
                np.testing.assert_array_equal(contact_records(runtime)[name], value)
            runtime.discard_execution(ticket)
        runtime.forward()
        assert np.isfinite(contact_records(runtime)['contact_wrench']).all()


@pytest.mark.cuda
def test_contact_empty_stale_and_mixed_observation_capacity(payload):
    with PhySolveRuntime(compile_mujoco_xml(FREE, require_asset=False), payload=payload,
                        contact_observations=True) as runtime:
        storage = runtime.storage_binding()
        assert not storage.fields['contact_wrench'].writable
        runtime.step()
        assert contact_records(runtime)['contact_world'].size == 0
    with pytest.raises(RuntimeError, match='stale'):
        storage.borrow('contact_wrench', 0)
    with PhySolveRuntime(compile_mujoco_xml(CONTACT, require_asset=False), payload=payload,
                        nworld=2, nconmax=1, contact_observations=True) as runtime:
        # Synthetic accepted observations fill the shared capacity in the
        # unselected world. A selected new record cannot silently truncate it.
        values, ids, count, by_world = runtime._contact_observations.raw[runtime.active_bank]
        old_ids = ids.numpy().copy()
        old_ids[:, 0] = 1
        ids.assign(old_ids, stream=runtime.storage_binding().stream)
        before = contact_records(runtime)
        with pytest.raises(RuntimeError, match='mixed contact observation capacity'):
            runtime.reset(np.array([True, False]))
        for name, value in before.items():
            np.testing.assert_array_equal(contact_records(runtime)[name], value)
        assert runtime.pending_execution is None
        runtime.reset()
        assert len(contact_records(runtime)['contact_world']) == 2


@pytest.mark.cuda
def test_multiple_contact_records_preserve_unselected_set(payload):
    xml = CONTACT.replace('<geom type="sphere" size=".1" mass="1"/>',
        '<geom type="sphere" size=".1" pos="-.2 0 0" mass=".5"/>'
        '<geom type="sphere" size=".1" pos=".2 0 0" mass=".5"/>')
    with PhySolveRuntime(compile_mujoco_xml(xml, require_asset=False), payload=payload,
                        nworld=8, contact_observations=True) as runtime:
        runtime.step()
        before = contact_records(runtime)
        runtime.reset(np.arange(8) % 2 == 0)
        after = contact_records(runtime)
        for world in (1, 3, 5, 7):
            a = np.flatnonzero(after['contact_world'][:, 0] == world)
            b = np.flatnonzero(before['contact_world'][:, 0] == world)
            assert len(a) == len(b) == 2
            a = a[np.argsort(after['contact_geom'][a, 1])]
            b = b[np.argsort(before['contact_geom'][b, 1])]
            for field in after:
                np.testing.assert_array_equal(after[field][a], before[field][b])


@pytest.mark.cuda
def test_contact_sixdim_rotated_frame(payload):
    xml = CONTACT.replace('type="plane"', 'type="plane" quat=".923879532511 0 .382683432365 0" condim="6"')
    xml = xml.replace('pos="0 0 .08"', 'pos=".056568542495 0 .056568542495"')
    xml = xml.replace('type="sphere"', 'type="sphere" condim="6"')
    with PhySolveRuntime(compile_mujoco_xml(xml, require_asset=False), payload=payload,
                        contact_observations=True) as runtime:
        storage = runtime.storage_binding()
        ticket = runtime.begin_execution()
        storage.borrow('qvel', runtime.writable_bank, write=True).assign(
            np.array([[0., 0., 0., 0., 0., 2.]], np.float32), stream=storage.stream)
        runtime.enqueue_update(ticket, refresh_state=True)
        runtime.complete_execution(ticket)
        runtime.accept_execution(ticket)
        record = contact_records(runtime)
        np.testing.assert_array_equal(record['contact_dim'], [[6]])
        frame = record['contact_frame'][0].reshape(3, 3)
        np.testing.assert_allclose(record['contact_wrench'][0].reshape(2, 3),
            record['contact_wrench_local'][0].reshape(2, 3) @ frame, atol=1e-6, rtol=1e-6)
        assert record['contact_wrench'][0, 5] < 0  # opposes imposed positive spin


@pytest.mark.cuda
def test_contact_prepare_close_failure_retains_measurement_owner(payload, monkeypatch):
    from phymeo.backends._physolve_contacts import PreparedContacts
    prepare = PreparedContacts.prepare
    native = payload.load()
    with monkeypatch.context() as patch:
        def fail_prepare(self, *args):
            prepare(self, *args)
            raise RuntimeError('injected contact prepare')
        def fail_close(*args):
            raise RuntimeError('injected contact close')
        patch.setattr(PreparedContacts, 'prepare', fail_prepare)
        patch.setattr(native, '_close_runtime', fail_close)
        with pytest.raises(RuntimeError, match='contact prepare') as captured:
            PhySolveRuntime(compile_mujoco_xml(CONTACT, require_asset=False), payload=payload,
                            contact_observations=True)
        retained = captured.value.runtime
        assert retained._contact_observations is not None
        assert retained._contact_observations.raw[0][0].ptr
    retained.close()
    assert retained._closed and retained._contact_observations is None


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_endpoint_bodies_keep_solver_time_and_masked_observations(payload, nworld, monkeypatch):
    import tyx as tx
    with PhySolveRuntime(compile_mujoco_xml(FREE, require_asset=False), payload=payload,
                        nworld=nworld, body_observations=True) as runtime:
        storage = runtime.storage_binding()
        assert storage.fields['body_position'].sample == 'integration endpoint'
        def read(name):
            return storage.borrow(name, runtime.active_bank).numpy(stream=storage.stream).copy()
        initial = {name: read(name) for name in ('body_position', 'body_rotation_wxyz', 'body_com_twist')}
        def forbidden(*args, **kwargs):
            raise AssertionError('endpoint execution must not allocate/read whole host state')
        with monkeypatch.context() as patch:
            patch.setattr(tx.Array, 'numpy', forbidden)
            patch.setattr(tx, 'empty', forbidden)
            ticket = runtime.begin_execution()
            runtime.enqueue_step(ticket, steps=4)
            runtime.complete_execution(ticket)
            runtime.accept_execution(ticket)
        q = read('qpos')
        body = read('body_position')
        np.testing.assert_allclose(body[:, 1], q[:, :3], atol=1e-6, rtol=0)
        np.testing.assert_allclose(read('body_com_twist')[:, 1, 2], -9.81*.002*4, atol=1e-6, rtol=0)
        assert np.all(read('xpos')[:, 1, 2] != body[:, 1, 2])
        assert runtime.execution_costs()['endpoint_kinematics_passes'] == 1
        before = {name: read(name) for name in initial}
        runtime.reset(np.arange(nworld) % 2 == 0)
        for name in initial:
            after = read(name)
            np.testing.assert_array_equal(after[::2], initial[name][::2])
            np.testing.assert_array_equal(after[1::2], before[name][1::2])
        assert runtime.execution_costs()['endpoint_kinematics_passes'] == 1
        runtime.forward()
        assert runtime.execution_costs()['endpoint_kinematics_passes'] == 1


@pytest.mark.cuda
def test_endpoint_body_com_velocity_and_command_only_preservation(payload):
    xml = '''<mujoco><option timestep=".002" gravity="0 0 0"/>
    <worldbody><body name="hinge"><joint axis="0 0 1"/>
    <inertial pos=".2 0 0" mass="2" diaginertia=".1 .1 .1"/>
    <geom type="sphere" size=".1" contype="0" conaffinity="0"/>
    </body></worldbody></mujoco>'''
    with PhySolveRuntime(compile_mujoco_xml(xml, require_asset=False), payload=payload,
                        nworld=8, body_observations=True) as runtime:
        storage = runtime.storage_binding()
        ticket = runtime.begin_execution()
        storage.borrow('qvel', runtime.writable_bank, write=True).assign(np.full((8, 1), 2., np.float32), stream=storage.stream)
        runtime.enqueue_step(ticket, steps=4)
        runtime.complete_execution(ticket)
        runtime.accept_execution(ticket)
        def read(name):
            return storage.borrow(name, runtime.active_bank).numpy(stream=storage.stream).copy()
        q = read('qpos')[:, 0]
        twist = read('body_com_twist')[:, 1]
        np.testing.assert_allclose(twist[:, :3], np.stack((-.4*np.sin(q), .4*np.cos(q), np.zeros(8)), axis=1), atol=1e-6, rtol=0)
        np.testing.assert_allclose(twist[:, 3:], np.tile([0., 0., 2.], (8, 1)), atol=1e-6, rtol=0)
        before = {name: read(name) for name in ('body_position', 'body_rotation_wxyz', 'body_com_twist')}
        ticket = runtime.begin_execution()
        runtime.enqueue_update(ticket)
        runtime.complete_execution(ticket)
        runtime.accept_execution(ticket)
        for name, value in before.items():
            np.testing.assert_array_equal(read(name), value)
        assert runtime.execution_costs()['endpoint_kinematics_passes'] == 1


@pytest.mark.cuda
def test_endpoint_failure_never_publishes_and_discard_recovers(payload, monkeypatch):
    with PhySolveRuntime(compile_mujoco_xml(SLIDER, require_asset=False), payload=payload,
                        body_observations=True) as runtime:
        storage = runtime.storage_binding()
        active = runtime.active_bank
        before = storage.borrow('body_position', active).numpy().copy()
        operation = runtime._body_observations.update
        for fault in ('nan', 'enqueue'):
            def injected(index, stream, *, integrated):
                operation(index, stream, integrated=integrated)
                if fault == 'enqueue':
                    raise RuntimeError('injected endpoint enqueue')
                target = runtime._body_observations.banks[index]['body_position']
                target.assign(np.full(target.shape, np.nan, np.float32), stream=stream)
            ticket = runtime.begin_execution()
            with monkeypatch.context() as patch:
                patch.setattr(runtime._body_observations, 'update', injected)
                with pytest.raises(RuntimeError, match='body_position|endpoint enqueue'):
                    runtime.enqueue_step(ticket)
                    runtime.complete_execution(ticket)
            assert runtime.active_bank == active
            np.testing.assert_array_equal(storage.borrow('body_position', active).numpy(), before)
            with pytest.raises(RuntimeError, match='completed'):
                runtime.accept_execution(ticket)
            runtime.discard_execution(ticket)
        runtime.step()
        assert runtime.active_bank != active
        assert np.isfinite(storage.borrow('body_position', runtime.active_bank).numpy()).all()


@pytest.mark.cuda
def test_endpoint_prepare_and_close_failure_retains_owner(payload, monkeypatch):
    from phymeo.backends._physolve_bodies import PreparedBodyObservations
    native = payload.load()
    operation = PreparedBodyObservations.prepare
    def failed_prepare(self, stream, mask):
        operation(self, stream, mask)
        raise RuntimeError('injected endpoint prepare')
    def failed_close(*args):
        raise RuntimeError('injected endpoint cleanup')
    with monkeypatch.context() as patch:
        patch.setattr(PreparedBodyObservations, 'prepare', failed_prepare)
        patch.setattr(native, '_close_runtime', failed_close)
        with pytest.raises(RuntimeError, match='endpoint prepare') as captured:
            PhySolveRuntime(compile_mujoco_xml(SLIDER, require_asset=False), payload=payload,
                            body_observations=True)
    owner = captured.value.runtime
    assert owner._body_observations is not None
    assert owner._handles
    owner.close()
    assert owner._body_observations is None


@pytest.mark.cuda
def test_endpoint_multibranch_fixed_child_matches_kinematic_reference(payload):
    import mujoco
    xml = '''<mujoco><compiler fusestatic="false"/><option timestep=".002" gravity="0 0 0"/>
    <worldbody><body name="a" pos=".4 -.2 .3"><joint axis="0 0 1"/>
    <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/>
    <body name="fixed" pos=".1 .2 0" euler=".3 .2 .1">
    <inertial pos=".05 .02 0" mass="1" diaginertia=".1 .1 .1"/>
    <geom type="sphere" size=".05" contype="0" conaffinity="0"/></body></body>
    <body name="b" pos="-.3 .1 .2"><joint type="slide" axis="1 0 0"/>
    <geom type="sphere" size=".1" mass="2" contype="0" conaffinity="0"/>
    </body></worldbody></mujoco>'''
    record = compile_mujoco_xml(xml, require_asset=False)
    cpu = record.fresh_native_model()
    data = mujoco.MjData(cpu)
    with PhySolveRuntime(record, payload=payload, nworld=8, body_observations=True) as runtime:
        storage = runtime.storage_binding()
        ticket = runtime.begin_execution()
        index = runtime.writable_bank
        q = np.stack((np.linspace(-.4, .4, 8), np.linspace(.1, .3, 8)), axis=1).astype(np.float32)
        storage.borrow('qpos', index, write=True).assign(q, stream=storage.stream)
        storage.borrow('qvel', index, write=True).assign(np.tile([.3, .5], (8, 1)).astype(np.float32), stream=storage.stream)
        runtime.enqueue_step(ticket, steps=4)
        runtime.complete_execution(ticket)
        runtime.accept_execution(ticket)
        def read(name):
            return storage.borrow(name, runtime.active_bank).numpy(stream=storage.stream)
        q, v = read('qpos'), read('qvel')
        positions, rotations, twists = read('body_position'), read('body_rotation_wxyz'), read('body_com_twist')
        for world in range(8):
            data.qpos[:], data.qvel[:] = q[world], v[world]
            mujoco.mj_kinematics(cpu, data)
            mujoco.mj_comPos(cpu, data)
            mujoco.mj_comVel(cpu, data)
            np.testing.assert_allclose(positions[world], data.xpos, atol=1e-6, rtol=0)
            np.testing.assert_allclose(rotations[world], data.xquat, atol=1e-6, rtol=0)
            expected = data.cvel[:, 3:] + np.cross(data.cvel[:, :3], data.xipos-data.subtree_com[cpu.body_rootid])
            np.testing.assert_allclose(twists[world, :, :3], expected, atol=1e-6, rtol=0)
            np.testing.assert_allclose(twists[world, :, 3:], data.cvel[:, :3], atol=1e-6, rtol=0)


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
        assert runtime.observation_execution_contract_version == 1
        assert runtime.observation_bank == old
        assert not storage.fields['execution_error'].writable
        assert storage.borrow('execution_error', 0).ptr == storage.borrow('execution_error', 1).ptr
        initial = runtime.snapshot()
        ticket = runtime.begin_execution()
        bank = runtime.writable_bank
        assert bank == 1 - old
        with pytest.raises(RuntimeError, match='observations require'):
            _ = runtime.observation_bank
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
        assert runtime.observation_bank == bank
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
        assert runtime.observation_bank == runtime.active_bank
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
            assert runtime.storage_binding().borrow('execution_error', 0).numpy()[0] != 0
            runtime.discard_execution(ticket)
            assert runtime.storage_binding().borrow('execution_error', 0).numpy(stream=runtime.storage_binding().stream)[0] == 0
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
