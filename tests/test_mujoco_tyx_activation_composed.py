"""ROB2b composed actuation routes, tests authored before product changes.

The native oracle always receives the actual entry act, in addition to q/v.
Independent continuous native states are initialized once and never reseeded.
"""
import copy

import numpy as np
import pytest

from phymeo.backends.mujoco_checkpoint import MuJoCoTyxCheckpoint
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.resident_stepper import PhysicsStepFailure
from test_mujoco_tyx_activation import analytic, encode, save, set_native

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
H = .001
ACT = dict(rtol=3e-5, atol=3e-6)
FORCE = dict(rtol=3e-4, atol=2e-4)
DYNAMICS = {
    "body_position": (3e-5, 4e-5), "body_rotation": (3e-5, 4e-5),
    "body_com": (3e-5, 4e-5), "jacobian_com": (2e-4, 5e-5),
    "mass_matrix": (3e-4, 5e-5), "bias": (4e-4, 4e-4),
    "passive": (4e-4, 4e-4), "applied": (4e-4, 4e-4),
    "smooth_acceleration": (2e-3, 4e-3),
}
# Predeclared route, steps and features. No runtime selection by outcomes.
CASES = (
    ("free", 1, False, False, 64),
    ("free", 1, True, False, 64),
    ("support", 1, True, False, 96),
    ("contact", 3, False, False, 64),
    ("contact", 6, True, False, 64),
    ("limits", 1, True, False, 64),
    ("limits", 1, False, True, 64),
    ("mixed", 3, True, False, 64),
)


def source(case, condim=1, early=True, disabled=False, count=2, stateless=False):
    free = case in ("free", "contact", "mixed", "capacity")
    contact = case in ("support", "contact", "mixed")
    limited = case in ("limits", "mixed")
    disable = ' actuatorgroupdisable="1"' if disabled else ""
    plane = '<geom name="floor" type="plane" size="2 2 .1"/>' if contact else ""
    root = ""
    if free:
        root = f'''<body name="root" pos="0 0 {'.099' if contact else '.6'}">
        <freejoint name="rootjoint"/><inertial pos=".012 -.006 .008" mass="2"
         diaginertia=".09 .11 .13"/>
        <geom name="round" type="sphere" size=".1" contype="{int(contact)}" conaffinity="{int(contact)}"/>'''
    if case == "support":
        bodies = '''<body name="link0" pos="0 0 .099"><joint name="j0" type="slide" axis="0 0 1"/>
        <inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>
        <geom name="round" size=".1"/></body>'''
        names = ["j0"]
    else:
        names = [f"j{i}" for i in range(count)]
        bodies = ""
        for i in range(count):
            limit = ' limited="true" range="-.2 .2" margin=".01" frictionloss=".07"' if limited and i == 0 else ""
            kind, axis = ("hinge", "0 1 0") if i % 2 == 0 else ("slide", "1 0 0")
            bodies += f'''<body name="link{i}" pos=".08 .015 .045">
             <joint name="j{i}" type="{kind}" axis="{axis}"{limit}/>
             <inertial pos=".015 -.005 .01" mass=".3" diaginertia=".003 .004 .005"/>
             <geom size=".025" contype="0" conaffinity="0"/>'''
        bodies += "</body>" * count
    if free:
        bodies = root + bodies + "</body>"
    if case == "capacity":
        actors = "".join(f'<general joint="{name}" dyntype="filterexact" dynprm=".004" gear="{-1 if i % 2 else 1}"/>' for i, name in enumerate(names))
    else:
        dynamics = "none" if stateless else "filterexact"
        state_limit = "false" if stateless else "true"
        force_range = ".1 .25" if disabled else "-.7 .7"
        actors = f'''<motor joint="{names[0]}" gear="-.5"/>
         <general joint="{names[-1]}" gear="-2" dyntype="{dynamics}" dynprm=".004"
          gainprm="1.2" biastype="affine" biasprm=".01 -.3 -.04" actearly="{str(early).lower()}"
          ctrllimited="true" ctrlrange="-.4 .4" actlimited="{state_limit}" actrange="-.2 .3"
          forcelimited="true" forcerange="-.7 .7"/>
         <general joint="{names[0]}" gear="1.5" group="1" dyntype="{dynamics}" dynprm=".002"
          gainprm=".8" forcelimited="true" forcerange="{force_range}"/>
         <general joint="{names[0]}" gear="-.25" dyntype="{dynamics}" dynprm=".008" gainprm=".6"/>'''
    return f'''<mujoco model="rob2b-{case}"><compiler angle="radian"/>
    <option timestep=".001" gravity="0 0 -9.81" integrator="Euler" solver="Newton"
     iterations="128" tolerance="1e-10" jacobian="dense" cone="pyramidal"{disable}/>
    <default><joint armature=".01" damping=".04"/>
     <geom condim="{condim}" friction=".6 .02 .003" solref=".02 1" solimp=".8 .9 .002 .5 2"/></default>
    <worldbody>{plane}{bodies}</worldbody><actuator>{actors}</actuator></mujoco>'''


def fixture(xml, mode, case, keys=(7,)):
    import mujoco as mj
    assert mj.__version__ == "3.3.7"
    model = mj.MjModel.from_xml_string(xml)
    compiled = compile_mujoco_xml(xml, require_asset=False)
    initial = {}
    for slot, key in enumerate(keys):
        q = model.qpos0.copy()
        if case == "free":
            q[3:7] = [.9238795325112867, 0, .3826834323650898, 0]
        for j in range(model.njnt):
            if int(model.jnt_type[j]) in (2, 3):
                q[model.jnt_qposadr[j]] = .205 if model.jnt_limited[j] else (0 if case == "support" else .02)
        v = np.full(model.nv, -.015 - slot * .002)
        initial[key] = MuJoCoInitialState(q, v, act=np.full(model.na, .34 + slot * .01), time=.125)
    runtime = MuJoCoRuntime(compiled, world_keys=keys, mode=mode, implementation="tyx", initial_states=initial)
    return mj, model, compiled, initial, runtime


def packet(model, tick, total, slot=0):
    u = .8 if tick < total // 4 else (-.75 if tick < total // 2 else 0.)
    wrench = np.zeros((model.nbody, 6))
    wrench[-1] = [.02, -.015, .01, .005, -.003, .004]
    return MuJoCoControl(np.full(model.nu, u + (slot * .015 if u else 0)),
        qfrc_applied=np.full(model.nv, .012), xfrc_applied=wrench)


def native_fields(mj, model, data):
    # All quantities refer to this mj_forward's input q/v/act/control, not its successor.
    mj.mj_forward(model, data)
    mass = np.empty((model.nv, model.nv)); mj.mj_fullM(model, mass, data.qM)
    jac = np.zeros((model.nbody, 6, model.nv))
    applied = data.qfrc_actuator.copy() + data.qfrc_applied
    for b in range(model.nbody):
        mj.mj_jac(model, data, jac[b, :3], jac[b, 3:], data.xipos[b], b)
        applied += jac[b].T @ data.xfrc_applied[b]
    rhs = applied + data.qfrc_passive - data.qfrc_bias
    contacts = []
    for ci in range(data.ncon):
        c = data.contact[ci]; frame = c.frame.reshape(3, 3).copy()
        ids = tuple(int(g) for g in c.geom); bodies = tuple(int(model.geom_bodyid[g]) for g in ids)
        cj = np.zeros((6, model.nv))
        for sign, b in ((-1, bodies[0]), (1, bodies[1])):
            jp = np.zeros((3, model.nv)); jr = np.zeros_like(jp)
            mj.mj_jac(model, data, jp, jr, c.pos, b)
            cj[:3] += sign * frame @ jp; cj[3:] += sign * frame @ jr
        force = np.zeros(6); mj.mj_contactForce(model, data, ci, force)
        contacts.append(dict(geometry_ids=ids, body_ids=bodies, dimension=int(c.dim), position=c.pos.copy(),
            frame=frame, distance=float(c.dist), include_margin=float(c.includemargin),
            active=c.efc_address >= 0, jacobian=cj, wrench_contact_frame=force))
    return dict(body_position=data.xpos.copy(), body_rotation=data.xmat.reshape(model.nbody, 3, 3).copy(),
        body_com=data.xipos.copy(), jacobian_com=jac, mass_matrix=mass, bias=data.qfrc_bias.copy(),
        passive=data.qfrc_passive.copy(), applied=applied, smooth_acceleration=np.linalg.solve(mass, rhs),
        qacc=data.qacc.copy(), qfrc_constraint=data.qfrc_constraint.copy(), scalar_force=data.actuator_force.copy(),
        derivative=data.act_dot.copy(), qfrc_actuator=data.qfrc_actuator.copy(), contacts=contacts,
        efc_type=data.efc_type.copy(), efc_id=data.efc_id.copy(), efc_J=data.efc_J.reshape(data.nefc, model.nv).copy(),
        efc_R=data.efc_R.copy(), efc_aref=data.efc_aref.copy(), efc_force=data.efc_force.copy())


def compare_receipt(receipt, expected, independent, entry, state, constrained):
    a = receipt.actuation
    np.testing.assert_array_equal(a.input_act, entry.act)
    np.testing.assert_array_equal(a.output_act, state.act)
    for channel, (actual, target) in enumerate(zip((a.output_act, a.activation_derivative, a.clamped_control,
        a.force_input, a.raw_force, a.scalar_force, a.generalized_force), independent, strict=True)):
        np.testing.assert_allclose(actual, target, **(ACT if channel < 2 else FORCE))
        with pytest.raises(ValueError): actual.setflags(write=True)
    np.testing.assert_allclose(a.scalar_force, expected["scalar_force"], **FORCE)
    np.testing.assert_allclose(a.activation_derivative, expected["derivative"], **ACT)
    np.testing.assert_array_equal(a.generalized_force, receipt.qfrc_actuator)
    for name, (rtol, atol) in DYNAMICS.items():
        np.testing.assert_allclose(getattr(receipt.dynamics, name), expected[name], rtol=rtol, atol=atol)
    np.testing.assert_allclose(receipt.qacc, expected["qacc"], rtol=3e-3 if constrained else 2e-3, atol=1e-2 if constrained else 4e-3)
    np.testing.assert_allclose(receipt.qfrc_constraint, expected["qfrc_constraint"], rtol=3e-3, atol=3e-3)
    assert len(receipt.contacts) == len(expected["contacts"])
    for contact, native in zip(receipt.contacts, expected["contacts"], strict=True):
        for key in ("geometry_ids", "body_ids", "dimension", "active"):
            assert getattr(contact, key) == native[key]
        for key in ("position", "frame", "distance", "include_margin"):
            np.testing.assert_allclose(getattr(contact, key), native[key], rtol=3e-5, atol=4e-5)
        cj = native["jacobian"] if contact.jacobian.ndim == 2 else native["jacobian"][0]
        np.testing.assert_allclose(contact.jacobian, cj, rtol=2e-4, atol=5e-5)
        np.testing.assert_allclose(contact.wrench_contact_frame, native["wrench_contact_frame"], rtol=3e-3, atol=3e-3)
        np.testing.assert_allclose(contact.force_on_geometry1_world + contact.force_on_geometry2_world, 0, atol=1e-12)
        for side, sign in ((1, -1), (2, 1)):
            b = contact.body_ids[side - 1]
            force = getattr(contact, f"force_on_geometry{side}_world")
            couple = sign * contact.frame.T @ contact.wrench_contact_frame[3:]
            target = np.r_[force, couple + np.cross(contact.position - receipt.dynamics.body_com[b], force)]
            np.testing.assert_allclose(getattr(contact, f"wrench_on_body{side}_world"), target, rtol=3e-5, atol=4e-5)
    if receipt.constraints:
        assert len(receipt.constraints) == len(expected["efc_type"])
        for i, row in enumerate(receipt.constraints):
            assert row.constraint_type == expected["efc_type"][i]
            assert row.source_id == expected["efc_id"][i]
            np.testing.assert_allclose(row.jacobian, expected["efc_J"][i], rtol=2e-4, atol=5e-5)
            for key, source_key in (("regularization", "efc_R"), ("reference_acceleration", "efc_aref")):
                np.testing.assert_allclose(getattr(row, key), expected[source_key][i], rtol=5e-4, atol=4e-4)
            np.testing.assert_allclose(row.force, expected["efc_force"][i], rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case,condim,early,disabled,steps", CASES)
def test_composed_activation_continuous(mode, case, condim, early, disabled, steps, tmp_path, monkeypatch):
    xml = source(case, condim, early, disabled)
    mj, model, _, initial, runtime = fixture(xml, mode, case)
    history = dict(xml=xml, initial=initial, case=case, mode=mode, steps=[])
    constrained = case != "free"
    expected_launches = 4 if case == "free" else (6 if case == "support" else 8)
    seen_contact = seen_limit = seen_dry = False
    try:
        before = runtime.integration_snapshot()[7]
        reference = mj.MjData(model)
        reference.qpos[:], reference.qvel[:], reference.act[:], reference.time = before.q, before.v, before.act, before.time
        for tick in range(steps):
            command = packet(model, tick, steps)
            entry = runtime.integration_snapshot()[7]
            independent = analytic(model, entry, command)
            same = mj.MjData(model)
            same.qpos[:], same.qvel[:], same.act[:], same.time = entry.q, entry.v, entry.act, entry.time
            set_native(same, command)
            expected = native_fields(mj, model, same)
            mj.mj_step(model, same)
            set_native(reference, command); mj.mj_step(model, reference)
            runtime.set_controls({7: command})
            kernels = runtime.counters["completed_kernels"]
            history["pending"] = dict(entry=entry, control=command, native=expected, analytic=independent)
            def forbidden(*args, **kwargs):
                raise AssertionError("native dynamics during composed TYX step")
            with monkeypatch.context() as patch:
                patch.setattr(mj, "mj_step", forbidden); patch.setattr(mj, "mj_forward", forbidden)
                assert runtime.step(H) == {7: True}
                state = runtime.integration_snapshot()[7]
            receipt = state.last_step
            history["steps"].append(dict(history.pop("pending"), receipt=receipt, output=state,
                same_output=(same.qpos.copy(), same.qvel.copy(), same.act.copy()),
                continuous=(reference.qpos.copy(), reference.qvel.copy(), reference.act.copy())))
            assert runtime.counters["completed_kernels"] - kernels == expected_launches
            compare_receipt(receipt, expected, independent, entry, state, constrained)
            for native in (same, reference):
                np.testing.assert_allclose(state.q, native.qpos, rtol=0, atol=2e-4 if constrained else 1e-4)
                np.testing.assert_allclose(state.v, native.qvel, rtol=0, atol=3e-3 if constrained else 1e-3)
                np.testing.assert_allclose(state.act, native.act, **ACT)
            assert abs(state.time - reference.time) <= 2e-6
            assert state.tick == tick + 1
            if model.nq != model.nv:
                np.testing.assert_array_equal(receipt.qfrc_actuator[:6], np.zeros(6))
            seen_contact |= any(c.normal_force > 0 for c in receipt.contacts)
            seen_limit |= any(r.constraint_type == 3 and r.force > 0 for r in receipt.constraints)
            seen_dry |= any(r.constraint_type == 1 and abs(r.force) > 0 for r in receipt.constraints)
            if disabled:
                assert receipt.actuation.raw_force[2] == 0
                np.testing.assert_allclose(receipt.actuation.scalar_force[2], .1, rtol=0, atol=2e-8)
                assert receipt.actuation.integration_derivative[1] == 0
        if case in ("support", "contact", "mixed"): assert seen_contact
        if case in ("limits", "mixed"): assert seen_limit and seen_dry
        if case == "free": assert np.any(state.v[:6] != before.v[:6])
        # Zero requested control does not reset the activation bank: decay is dynamic.
        assert np.any(history["steps"][steps // 2]["receipt"].actuation.output_act != 0)
    except BaseException as error:
        history["failure"] = repr(error)
        raise
    finally:
        history["counters"] = dict(runtime.counters)
        runtime.close(); save(tmp_path / "composed-history.json.gz", history)


@pytest.mark.parametrize("mode", MODES)
def test_activation_free_root_29_joint_capacity(mode, tmp_path):
    xml = source("capacity", count=29)
    mj, model, _, initial, runtime = fixture(xml, mode, "capacity")
    history = dict(xml=xml, initial=initial, steps=[])
    assert (model.nq, model.nv, model.na) == (36, 35, 29)
    try:
        for tick in range(4):
            entry = runtime.integration_snapshot()[7]; command = packet(model, tick, 4)
            same = mj.MjData(model)
            same.qpos[:], same.qvel[:], same.act[:], same.time = entry.q, entry.v, entry.act, entry.time
            set_native(same, command); expected = native_fields(mj, model, same)
            independent = analytic(model, entry, command); mj.mj_step(model, same)
            runtime.set_controls({7: command}); runtime.step(H)
            state = runtime.integration_snapshot()[7]
            history["steps"].append(dict(entry=entry, control=command, native=expected, output=state))
            compare_receipt(state.last_step, expected, independent, entry, state, False)
            np.testing.assert_allclose(state.q, same.qpos, rtol=0, atol=1e-4)
            np.testing.assert_allclose(state.v, same.qvel, rtol=0, atol=1e-3)
    except BaseException as error:
        history["failure"] = repr(error); raise
    finally:
        runtime.close(); save(tmp_path / "capacity-history.json.gz", history)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", ("free", "mixed"))
def test_composed_checkpoint_clone_reset(mode, case, tmp_path):
    xml = source(case, 3 if case == "mixed" else 1)
    _, model, compiled, initial, runtime = fixture(xml, mode, case, (7, 91))
    with runtime:
        for tick in range(4):
            runtime.set_controls({k: packet(model, tick, 16, i) for i, k in enumerate(runtime.world_keys)})
            runtime.step(H)
        before = runtime.snapshot(); cp = MuJoCoTyxCheckpoint.from_bytes(runtime.checkpoint().to_bytes())
        assert cp.binding["schema"] == "mujoco-tyx-exact-checkpoint-v2"
        with runtime.clone() as clone:
            assert encode(clone.snapshot()) == encode(before)
            runtime.step(H); successor = runtime.snapshot(); clone.step(H)
            assert encode(clone.snapshot()) == encode(successor)
            runtime.restore(cp, expected_revision=runtime.checkpoint_revision)
            assert encode(runtime.snapshot()) == encode(before)
            runtime.step(H); assert encode(runtime.snapshot()) == encode(successor)
        retained = runtime.snapshot(); runtime.reset((91,)); reset = runtime.snapshot()
        assert encode(reset[7]) == encode(retained[7])
        np.testing.assert_array_equal(reset[91].act, np.asarray(initial[91].act, np.float32))
        with MuJoCoRuntime(compiled, world_keys=(91,), mode=mode, initial_states={91: initial[91]}) as fresh:
            command = packet(model, 8, 16)
            runtime.set_controls({7: command, 91: command}); fresh.set_controls({91: command})
            runtime.step(H); fresh.step(H)
            assert encode(runtime.snapshot()[91]) == encode(fresh.snapshot()[91])
        save(tmp_path / "composed-restore.json.gz", dict(xml=xml, initial=initial, checkpoint=cp.to_bytes().hex(),
            before=before, successor=successor, reset=reset, final=runtime.snapshot(), counters=dict(runtime.counters)))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", ("free", "mixed"))
@pytest.mark.parametrize("fault", ("input_act", "last_world_status", "restore_readback"))
def test_composed_late_failure_retains_all_state(mode, case, fault, monkeypatch, tmp_path):
    import phymeo.backends.mujoco_checkpoint as checkpoint_module
    _, model, _, initial, runtime = fixture(source(case, 3 if case == "mixed" else 1), mode, case, (7, 91))
    with runtime:
        runtime.set_controls({k: packet(model, 0, 16, i) for i, k in enumerate(runtime.world_keys)})
        runtime.step(H); cp = runtime.checkpoint(); before = runtime.snapshot()
        with runtime.clone() as clone:
            with monkeypatch.context() as patch:
                if fault == "restore_readback":
                    def fail(*args, **kwargs): raise RuntimeError("authored composed late restore")
                    patch.setattr(checkpoint_module, "_verify_bank", fail)
                    with pytest.raises(RuntimeError, match="authored composed"):
                        runtime.restore(cp, expected_revision=runtime.checkpoint_revision)
                else:
                    original = runtime._device.download
                    def damaged():
                        result = original()
                        if fault == "input_act": result["input_act"][-1, 0] += .25
                        else: result["status"][-1] = 5
                        return result
                    patch.setattr(runtime._device, "download", damaged)
                    with pytest.raises(PhysicsStepFailure): runtime.step(H)
                assert encode(runtime.snapshot()) == encode(before)
                assert runtime.checkpoint().to_bytes() == cp.to_bytes()
            runtime.step(H); clone.step(H)
            assert encode(runtime.snapshot()) == encode(clone.snapshot())
            save(tmp_path / "composed-failure.json.gz", dict(fault=fault, initial=initial, before=before,
                checkpoint=cp.to_bytes().hex(), retry=runtime.snapshot(), uninterrupted=clone.snapshot(), counters=dict(runtime.counters)))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("target", ("free", "ball"))
def test_scalar_activation_rejects_nonscalar_joint_target(mode, target, monkeypatch):
    from phymeo.backends.mujoco_tyx import MuJoCoTyxUnsupported, _floating_device, _tree_device
    xml = source("free")
    if target == "free": xml = xml.replace('joint="j0"', 'joint="rootjoint"')
    else: xml = xml.replace('name="j0" type="hinge"', 'name="j0" type="ball"')
    compiled = compile_mujoco_xml(xml, require_asset=False)
    def forbidden(*args, **kwargs): raise AssertionError("nonscalar actuator reached device allocation")
    monkeypatch.setattr(_floating_device.TyxFloatingDevice, "__init__", forbidden)
    monkeypatch.setattr(_tree_device.TyxTreeDevice, "__init__", forbidden)
    with pytest.raises(MuJoCoTyxUnsupported): MuJoCoRuntime(compiled, mode=mode)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case,dim,launches", (("free", 1, 3), ("support", 1, 5), ("mixed", 3, 7)))
def test_stateless_composed_routes_keep_original_banks_and_launches(mode, case, dim, launches):
    _, model, _, _, runtime = fixture(source(case, dim, stateless=True), mode, case)
    with runtime:
        runtime.set_controls({7: packet(model, 0, 16)})
        before = runtime.counters["completed_kernels"]; runtime.step(H)
        assert runtime.counters["completed_kernels"] - before == launches
        state = runtime.integration_snapshot()[7]
        assert state.act.size == 0 and not hasattr(state.last_step, "actuation")
        assert runtime.checkpoint().binding["schema"] == "mujoco-tyx-exact-checkpoint-v1"
