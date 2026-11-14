"""M1 forest contracts: authored without execution; parent owns all test runs.

One GPU, independent worlds, native joint addresses, and freely rotating payloads.
The CPU oracle is evaluated outside TYX step and never overwrites product state.
"""

import sys
from contextlib import ExitStack

import numpy as np
import pytest
import test_mujoco_tyx_constraints as mixed
import test_mujoco_tyx_normal_contacts as normal
from test_mujoco_tyx_floating import _same, _save

from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.resident_stepper import PhysicsStepFailure

MODES = normal.MODES
H = 0.001
# Reuse the original strict field and continuous-window gates without widening.
Q_ATOL, V_ATOL, TIME_ATOL = 1e-4, 1e-3, 2e-6
SMOOTH_CASES = (
    ("rp", False, 2, 1),
    ("prp", True, 3, 1),
    ("nine", False, 9, 1),
    ("two-free", True, 3, 2),
)


def source(name="rp", *, object_first=False, contact=0):
    """Fixed authored R/P trees and independent FREE bodies; no pose feedback."""
    if name not in {row[0] for row in SMOOTH_CASES}:
        raise ValueError(name)
    kinds, parents = {
        "rp": ("RP", (-1, 0)),
        "prp": ("PRP", (-1, 0, 0)),
        "nine": ("RRPRPRRPP", (-1, 0, 1, 2, 3, 4, 5, 6, 7)),
        "two-free": ("PRP", (-1, 0, 0)),
    }[name]
    if contact and name not in ("rp", "prp"):
        raise ValueError("contact fixtures are the two declared small trees")

    def link(index):
        kind = kinds[index]
        axis = "0 0 1" if kind == "R" else ("0 1 0" if index == 0 else "1 0 0")
        if index == 2 and name in ("prp", "two-free"):
            axis = "0 0 1"
        position = "0 0 .5" if index == 0 else (
            "-.12 .09 .05" if index == 2 and name != "nine" else ".2 0 0"
        )
        geom = (
            '<geom name="tip" type="sphere" size=".1" pos=".12 .04 .02" '
            'contype="1" conaffinity="2"/>'
            if index == 1 and contact
            else '<geom type="capsule" size=".025 .06"/>'
        )
        children = "".join(link(child) for child, parent in enumerate(parents) if parent == index)
        return (
            f'<body name="link{index}" pos="{position}">'
            f'<joint name="j{index}" type="{"hinge" if kind == "R" else "slide"}" '
            f'axis="{axis}" damping=".08" armature=".025" stiffness=".3" springref=".04"/>'
            '<inertial pos=".06 -.02 .025" mass=".7" '
            'fullinertia=".025 .03 .035 .001 -.0006 .0008"/>'
            + geom + children + "</body>"
        )

    def payload(index):
        position = ".458 .04 .52" if contact else (".55 -.2 .8" if index == 0 else "-.45 .3 .9")
        masks = 'contype="2" conaffinity="1"' if contact else ""
        return (
            f'<body name="object{index}" pos="{position}"><freejoint name="free{index}"/>'
            '<inertial pos=".015 -.012 .018" mass="1.3" '
            'fullinertia=".04 .055 .065 .002 -.001 .003"/>'
            f'<geom name="payload{index}" type="box" size=".04 .055 .06" {masks}/>'
            f'<body name="marker{index}" pos=".04 .02 -.01"><geom type="sphere" size=".008" '
            'contype="0" conaffinity="0" mass=".01"/></body></body>'
        )

    robot = link(0)
    objects = "".join(payload(i) for i in range(2 if name == "two-free" else 1))
    motors = "".join(
        f'<general name="drive{i}" joint="j{i}" gear="{-0.75 if i % 2 else 1.25}" '
        'dyntype="none" gaintype="fixed" gainprm="1.2" biastype="affine" '
        'biasprm=".02 -.3 -.04" ctrllimited="true" ctrlrange="-.5 .5" '
        'forcelimited="true" forcerange="-.7 .7"/>'
        for i in range(len(kinds))
    )
    return (
        '<mujoco model="forest"><compiler angle="radian" fusestatic="false"/>'
        '<option timestep=".001" gravity=".2 -.3 -9.81" integrator="Euler" '
        'solver="Newton" iterations="128" tolerance="1e-10" jacobian="dense" cone="pyramidal"/>'
        f'<default><geom contype="0" conaffinity="0" condim="{contact or 1}" '
        'friction=".4 .01 .001" solref=".02 1" solimp=".8 .9 .002 .5 2"/></default>'
        '<worldbody><body name="fixed-marker" pos="-2 1 .4"><geom size=".01"/></body>'
        + (objects + robot if object_first else robot + objects)
        + "</worldbody><actuator>" + motors + "</actuator></mujoco>"
    )


def initial(mj, model, *, contact=False, slot=0):
    q, v = model.qpos0.copy(), np.zeros(model.nv)
    for joint in range(model.njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint)
        qa, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        if model.jnt_type[joint] == mj.mjtJoint.mjJNT_FREE:
            ordinal = int(name.removeprefix("free"))
            if not contact:
                quat = np.array([.94, .12, -.18, .09 + .02 * ordinal])
                q[qa + 3 : qa + 7] = quat / np.linalg.norm(quat)
                q[qa : qa + 3] += slot * np.array([.01, -.02, .03])
            v[dof : dof + 6] = np.array([.018, -.012, .009, .06, -.045, .035]) * (
                1 + .1 * slot + .2 * ordinal
            )
        else:
            ordinal = int(name.removeprefix("j"))
            if not contact:
                q[qa] += (-1 if ordinal % 2 else 1) * (.04 + .003 * ordinal)
            v[dof] = (-1 if ordinal % 2 else 1) * (.015 + .002 * ordinal) * (1 + .1 * slot)
    return MuJoCoInitialState(q, v, time=.125)


def packet(model, tick=0, slot=0):
    # Only actual robot actuators receive commands. No object force or scripted wrench.
    sign = -1 if tick // 4 % 2 else 1
    return normal._stored(
        MuJoCoControl(
            np.array([sign * (0.12 + 0.01 * i + 0.005 * slot) for i in range(model.nu)]),
            qfrc_applied=np.zeros(model.nv),
            xfrc_applied=np.zeros((model.nbody, 6)),
        )
    )


def runtime(xml, mode, states):
    return MuJoCoRuntime(
        compile_mujoco_xml(xml, require_asset=False), mode=mode, implementation="tyx",
        world_keys=tuple(states), initial_states=states,
    )


def native_data(mj, model, state):
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.time = state.q, state.v, state.time
    return data


def assert_fields(row, expected):
    for field, (rtol, atol) in normal.TOL.items():
        value = getattr(row.dynamics, field)
        assert value.shape == expected[field].shape
        np.testing.assert_allclose(value, expected[field], rtol=rtol, atol=atol, err_msg=field)
        with pytest.raises(ValueError):
            value.setflags(write=True)
    for field in ("qacc", "integration_acceleration"):
        np.testing.assert_allclose(getattr(row, field), expected[field], **normal.ACC_TOL)
    np.testing.assert_allclose(row.qfrc_actuator, expected["qfrc_actuator"], rtol=4e-4, atol=4e-4)
    np.testing.assert_allclose(row.qfrc_constraint, expected["qfrc_constraint"], **normal.FORCE_TOL)


def assert_successor(mj, model, before, after, command, native, continuous):
    row = after.last_step
    np.testing.assert_array_equal(row.input_q, before.q)
    np.testing.assert_array_equal(row.input_v, before.v)
    np.testing.assert_array_equal(row.dynamics.input_q, before.q)
    np.testing.assert_array_equal(row.dynamics.input_v, before.v)
    assert row.dynamics.time == row.input_time == before.time
    assert row.output_time == after.time and row.tick == after.tick == before.tick + 1
    for field in ("ctrl", "qfrc_applied", "xfrc_applied"):
        np.testing.assert_array_equal(getattr(row.stored_control, field), getattr(command, field))
    for reference in (native, continuous):
        np.testing.assert_allclose(after.q, reference.qpos, rtol=0, atol=Q_ATOL)
        np.testing.assert_allclose(after.v, reference.qvel, rtol=0, atol=V_ATOL)
        assert after.time == pytest.approx(reference.time, abs=TIME_ATOL)
    for joint in np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_FREE):
        qa = int(model.jnt_qposadr[joint])
        quat = after.q[qa + 3 : qa + 7]
        assert abs(quat @ quat - 1) < 2e-6
    assert after.q.shape == (model.nq,) and after.v.shape == (model.nv,)


def free_dofs(mj, model):
    return np.array([
        dof for joint in np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_FREE)
        for dof in range(int(model.jnt_dofadr[joint]), int(model.jnt_dofadr[joint]) + 6)
    ], dtype=int)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name,object_first,scalars,free_count", SMOOTH_CASES)
def test_forest_native_dynamics_and_continuous_motion(
    mode, name, object_first, scalars, free_count, tmp_path,
):
    mj = normal._deps()
    xml = source(name, object_first=object_first)
    model = mj.MjModel.from_xml_string(xml)
    assert (model.nq, model.nv, model.nu) == (scalars + 7 * free_count, scalars + 6 * free_count, scalars)
    begin = initial(mj, model)
    with runtime(xml, mode, {19: begin}) as owner:
        start = owner.integration_snapshot()[19]
        continuous = native_data(mj, model, start)
        fcols = free_dofs(mj, model)
        rcols = np.setdiff1d(np.arange(model.nv), fcols)
        for tick in range(12):
            before = owner.integration_snapshot()[19]
            command = packet(model, tick)
            expected = mixed.native(mj, model, before.q, before.v, command, before.time)
            local = native_data(mj, model, before)
            for data in (local, continuous):
                normal._set(data, command)
                mj.mj_step(model, data)
            owner.set_controls({19: command})
            assert owner.step(H) == {19: True}
            after = owner.integration_snapshot()[19]
            _save(tmp_path / "smooth.jsonl", dict(
                source=xml if tick == 0 else None, before=before, actual=after,
                oracle=expected, native_q=local.qpos.copy(), native_v=local.qvel.copy(),
                continuous_q=continuous.qpos.copy(), continuous_v=continuous.qvel.copy(),
            ))
            assert_fields(after.last_step, expected)
            assert_successor(mj, model, before, after, command, local, continuous)
            assert not after.last_step.contacts
            # Separate trees share contact solves, never an invented inertial coupling.
            np.testing.assert_allclose(
                after.last_step.dynamics.mass_matrix[np.ix_(fcols, rcols)], 0, rtol=0, atol=1e-7,
            )
        for joint in np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_FREE):
            qa, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
            assert np.all(start.v[dof : dof + 6] != 0)
            assert np.linalg.norm(after.q[qa + 3 : qa + 7] - start.q[qa + 3 : qa + 7]) > 1e-5
            assert np.linalg.norm(after.q[qa : qa + 3] - start.q[qa : qa + 3]) > 1e-5


def named_addresses(mj, model):
    qcols, vcols = [], []
    joints = sorted((mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j), j) for j in range(model.njnt))
    for _, joint in joints:
        free = model.jnt_type[joint] == mj.mjtJoint.mjJNT_FREE
        qa, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        qcols.extend(range(qa, qa + (7 if free else 1)))
        vcols.extend(range(dof, dof + (6 if free else 1)))
    bodies = sorted((mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b), b) for b in range(model.nbody))
    return qcols, vcols, [b for _, b in bodies]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ("rp", "two-free"))
def test_forest_xml_order_preserves_named_physics(mode, name):
    mj = normal._deps()
    with ExitStack() as stack:
        owners, models = [], []
        for object_first in (False, True):
            xml = source(name, object_first=object_first)
            model = mj.MjModel.from_xml_string(xml)
            models.append(model)
            owners.append(stack.enter_context(runtime(xml, mode, {7: initial(mj, model)})))
        assert models[0].jnt_type[0] != models[1].jnt_type[0]
        for tick in range(2):
            for owner, model in zip(owners, models, strict=True):
                owner.set_controls({7: packet(model, tick)})
                owner.step(H)
            left, right = [owner.integration_snapshot()[7] for owner in owners]
            qa, va, ba = named_addresses(mj, models[0])
            qb, vb, bb = named_addresses(mj, models[1])
            np.testing.assert_allclose(left.q[qa], right.q[qb], rtol=0, atol=Q_ATOL)
            np.testing.assert_allclose(left.v[va], right.v[vb], rtol=0, atol=V_ATOL)
            a, b = left.last_step.dynamics, right.last_step.dynamics
            for field in ("body_position", "body_rotation", "body_com"):
                np.testing.assert_allclose(getattr(a, field)[ba], getattr(b, field)[bb], rtol=3e-5, atol=4e-5)
            np.testing.assert_allclose(a.jacobian_com[ba][:, :, va], b.jacobian_com[bb][:, :, vb], rtol=2e-4, atol=5e-5)
            np.testing.assert_allclose(a.mass_matrix[np.ix_(va, va)], b.mass_matrix[np.ix_(vb, vb)], rtol=3e-4, atol=5e-5)
            for field in ("bias", "passive", "applied", "smooth_acceleration"):
                rtol, atol = normal.TOL[field]
                np.testing.assert_allclose(getattr(a, field)[va], getattr(b, field)[vb], rtol=rtol, atol=atol)
            np.testing.assert_allclose(left.last_step.qfrc_actuator[va], right.last_step.qfrc_actuator[vb], rtol=4e-4, atol=4e-4)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ("rp", "prp"))
@pytest.mark.parametrize("dim", (1, 3))
def test_forest_real_contact_couples_robot_and_free_object(mode, name, dim, tmp_path):
    mj = normal._deps()
    xml = source(name, object_first=name == "prp", contact=dim)
    model = mj.MjModel.from_xml_string(xml)
    with runtime(xml, mode, {29: initial(mj, model, contact=True)}) as owner:
        continuous = native_data(mj, model, owner.integration_snapshot()[29])
        fcols = free_dofs(mj, model)
        rcols = np.setdiff1d(np.arange(model.nv), fcols)
        for tick in range(12):
            before = owner.integration_snapshot()[29]
            command = packet(model, tick)
            oracle = mixed.native if dim == 3 else normal._native
            expected = oracle(mj, model, before.q, before.v, command, before.time)
            local = native_data(mj, model, before)
            for data in (local, continuous):
                normal._set(data, command)
                mj.mj_step(model, data)
            owner.set_controls({29: command})
            assert owner.step(H) == {29: True}
            after = owner.integration_snapshot()[29]
            row = after.last_step
            _save(tmp_path / "contact.jsonl", dict(
                source=xml if tick == 0 else None, before=before, actual=after,
                oracle=expected, continuous_q=continuous.qpos.copy(), continuous_v=continuous.qvel.copy(),
            ))
            (mixed.assert_receipt if dim == 3 else normal._assert_receipt)(row, expected, model)
            assert_successor(mj, model, before, after, command, local, continuous)
            if tick == 0:
                assert len(row.contacts) == 1 and row.contacts[0].normal_force > .01
                assert np.linalg.norm(row.qfrc_constraint[rcols]) > .001
                assert np.linalg.norm(row.qfrc_constraint[fcols[:3]]) > .001
                assert np.linalg.norm(row.qfrc_constraint[fcols[3:]]) > .0001
                angular_response = row.qacc[fcols[3:]] - row.dynamics.smooth_acceleration[fcols[3:]]
                assert np.linalg.norm(angular_response) > 0.001
                contact = row.contacts[0]
                assert np.linalg.norm(contact.jacobian[..., rcols]) > .01
                assert np.linalg.norm(contact.jacobian[..., fcols]) > .01
                np.testing.assert_allclose(
                    contact.force_on_geometry1_world + contact.force_on_geometry2_world,
                    np.zeros(3), rtol=0, atol=1e-6,
                )


def batch_inputs(mj):
    models, specs, worlds = {}, [], []
    for name in ("rp", "prp"):
        xml = source(name, object_first=name == "prp", contact=3)
        models[name] = mj.MjModel.from_xml_string(xml)
        specs.append(MuJoCoTyxModelSpec(name, compile_mujoco_xml(xml, require_asset=False)))
    for key, name, slot in ((41, "rp", 0), (7, "prp", 1), (113, "rp", 2)):
        worlds.append(MuJoCoTyxWorldSpec(key, name, initial(mj, models[name], contact=True, slot=slot)))
    return models, tuple(specs), tuple(worlds)


def batch_packets(models, worlds, tick):
    return {world.world: packet(models[world.model_key], tick, slot) for slot, world in enumerate(worlds)}


@pytest.mark.cuda
def test_forest_heterogeneous_cuda_world_isolation_and_partial_reset(tmp_path):
    mj = normal._deps()
    models, specs, worlds = batch_inputs(mj)
    with ExitStack() as stack:
        batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda"))
        independent = {
            world.world: stack.enter_context(MuJoCoTyxHeterogeneousBatch(
                tuple(spec for spec in specs if spec.key == world.model_key), (world,), mode="cuda",
            )) for world in worlds
        }
        for tick in range(3):
            commands = batch_packets(models, worlds, tick)
            batch.set_controls(commands)
            assert batch.step(H) == dict.fromkeys(batch.world_keys, True)
            actual = batch.integration_snapshot()
            for key, single in independent.items():
                single.set_controls({key: commands[key]})
                single.step(H)
                _same({key: actual[key]}, single.integration_snapshot())
        before = batch.integration_snapshot()
        batch.reset((7,))
        independent[7].reset()
        after = batch.integration_snapshot()
        _same({key: before[key] for key in (41, 113)}, {key: after[key] for key in (41, 113)})
        assert after[7].tick == 0 and after[7].last_step is None
        commands = batch_packets(models, worlds, 3)
        batch.set_controls(commands)
        batch.step(H)
        for key, single in independent.items():
            single.set_controls({key: commands[key]})
            single.step(H)
            _same({key: batch.integration_snapshot()[key]}, single.integration_snapshot())
        _save(tmp_path / "isolation-reset.jsonl", dict(before=before, after=batch.integration_snapshot()))


@pytest.mark.cuda
def test_forest_last_bucket_download_failure_retains_all_worlds_and_retries(monkeypatch, tmp_path):
    mj = normal._deps()
    models, specs, worlds = batch_inputs(mj)
    with (
        MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda") as batch,
        MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda") as reference,
    ):
        for owner in (batch, reference):
            owner.set_controls(batch_packets(models, worlds, 0))
            owner.step(H)
            owner.set_controls(batch_packets(models, worlds, 1))
        before = batch.checkpoint().to_bytes()
        states, receipt = batch.integration_snapshot(), batch.last_step
        count = batch.counters["completed_kernels"]
        target = batch._buckets[-1]
        original = target._device.download

        def fail_after_download():
            original()
            raise RuntimeError("forest last actual device readback failed")

        with monkeypatch.context() as patch:
            patch.setattr(target._device, "download", fail_after_download)
            with pytest.raises(PhysicsStepFailure) as caught:
                batch.step(H)
        assert caught.value.accepted_worlds == dict.fromkeys(batch.world_keys, False)
        assert batch.checkpoint().to_bytes() == before
        assert batch.last_step is receipt
        assert batch.counters["completed_kernels"] > count
        _same(states, batch.integration_snapshot())
        assert batch.step(H) == reference.step(H)
        _same(batch.integration_snapshot(), reference.integration_snapshot())
        _save(tmp_path / "late-failure.jsonl", dict(before=states, retry=batch.integration_snapshot()))


@pytest.mark.parametrize("mode", MODES)
def test_forest_step_has_no_cpu_dynamics_fallback(mode, monkeypatch, tmp_path):
    mj = normal._deps()
    xml = source("prp", contact=3)
    model = mj.MjModel.from_xml_string(xml)
    with runtime(xml, mode, {0: initial(mj, model, contact=True)}) as owner:
        before = owner.integration_snapshot()[0]
        command = packet(model)
        expected = mixed.native(mj, model, before.q, before.v, command, before.time)
        owner.set_controls({0: command})
        launches = owner.counters["completed_kernels"]

        def forbidden(*args, **kwargs):
            raise AssertionError("forest step attempted a CPU dynamics fallback")

        with monkeypatch.context() as patch:
            for name in (
                "mj_step", "mj_forward", "mj_fullM", "mj_rne", "mj_jac",
                "mj_collision", "mj_fwdConstraint",
            ):
                patch.setattr(mj, name, forbidden)
            for name in ("solve", "inv", "cholesky", "lstsq"):
                patch.setattr(np.linalg, name, forbidden)
            patch.setitem(sys.modules, "warp", None)
            patch.setitem(sys.modules, "mujoco_warp", None)
            assert owner.step(H) == {0: True}
        actual = owner.integration_snapshot()[0].last_step
        assert owner.counters["completed_kernels"] > launches
        assert owner.capabilities["gpu"] is (mode == "cuda")
        mixed.assert_receipt(actual, expected, model)
        _save(tmp_path / "no-fallback.jsonl", dict(oracle=expected, actual=actual))
