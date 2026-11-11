"""ROB2a tests-first declaration; no execution in the author lane.

Physics gates are the existing T1 FP32 gates. New act/scalar-force gates below
are fixed before execution, not fit to returned trajectories.
"""
from dataclasses import fields, is_dataclass
from collections.abc import Mapping
import gzip
import json
import math

import numpy as np
import pytest

from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.backends.mujoco_checkpoint import MuJoCoTyxCheckpoint
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.resident_stepper import PhysicsStepFailure

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
H = 0.001
ACT_RTOL, ACT_ATOL = 3e-5, 3e-6
FORCE_RTOL, FORCE_ATOL = 3e-4, 2e-4
Q_ATOL, V_ATOL, TIME_ATOL = 5e-5, 3e-4, 2e-6
DYNAMICS_TOL = {
    "mass_matrix": (2e-4, 3e-5), "bias": (3e-4, 2e-4),
    "passive": (3e-4, 2e-4), "applied": (3e-4, 2e-4),
    "smooth_acceleration": (5e-4, 8e-4),
}


def source(case="hinge", *, early=False, tau=.004, disabled=False):
    kind = "slide" if case == "slide" else "hinge"
    child = ""
    if case == "coupled":
        child = '''<body name="tip" pos=".2 .03 .04" quat=".9238795325 0 0 .3826834324">
          <joint name="slide" type="slide" axis="1 0 0" armature=".05" damping=".11"/>
          <inertial pos=".03 -.02 .01" mass=".8" diaginertia=".02 .03 .04"/>
          <geom size=".03" contype="0" conaffinity="0"/></body>'''
    extra = ' actuatorgroupdisable="1"' if disabled else ""
    early_text = "true" if early else "false"
    if case == "coupled":
        actors = f'''<motor name="direct" joint="joint" gear="-.5"/>
          <general name="filter-slide" joint="slide" gear="-2" dyntype="filterexact"
           dynprm="{tau}" gainprm="1.2" biastype="affine" biasprm=".01 -.3 -.04"
           actearly="{early_text}"/>
          <general name="filter-hinge" joint="joint" gear="1.5" dyntype="filterexact"
           dynprm=".002" gainprm=".8"/>
          <general name="second-hinge" joint="joint" gear="-.25" dyntype="filterexact"
           dynprm=".008" gainprm=".6"/>'''
    else:
        force_range = '.1 .25' if disabled else '-.7 .7'
        gear = -2 if case == "slide" else 1.5
        actors = f'''<general name="filter" joint="joint" gear="{gear}" group="1"
         dyntype="filterexact" dynprm="{tau}" gainprm="1.4"
         biastype="affine" biasprm=".03 -.2 -.05" actearly="{early_text}"
         ctrllimited="true" ctrlrange="-.4 .4"
         actlimited="true" actrange="-.2 .3"
         forcelimited="true" forcerange="{force_range}"/>'''
    return f'''<mujoco model="rob2a-{case}"><compiler angle="radian"/>
      <option timestep="{H}" gravity=".1 -.2 -1" integrator="Euler"{extra}/>
      <worldbody><body name="base" pos=".04 .02 .3">
       <joint name="joint" type="{kind}" axis="0 1 0" armature=".04" damping=".12"/>
       <inertial pos=".03 .01 -.02" mass="1.1" diaginertia=".04 .05 .06"/>
       <geom size=".04" contype="0" conaffinity="0"/>{child}
      </body></worldbody><actuator>{actors}</actuator></mujoco>'''


def encode(value):
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": list(value.shape), "bytes": value.tobytes().hex()}
    if is_dataclass(value):
        return {"record": type(value).__name__, "fields": {f.name: encode(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, Mapping):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(x) for x in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save(path, payload):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(encode(payload), stream, allow_nan=False, separators=(",", ":"))


def fixture(xml, mode, keys=(17,)):
    import mujoco as mj
    assert mj.__version__ == "3.3.7"
    model = mj.MjModel.from_xml_string(xml)
    compiled = compile_mujoco_xml(xml, require_asset=False)
    initial = {
        key: MuJoCoInitialState(model.qpos0 + .06, np.full(model.nv, -.035),
                               act=np.full(model.na, .34 + slot * .01), time=.125)
        for slot, key in enumerate(keys)
    }
    runtime = MuJoCoRuntime(compiled, world_keys=keys, mode=mode,
                           implementation="tyx", initial_states=initial)
    return mj, model, compiled, initial, runtime


def control(model, tick, slot=0):
    command = .8 if tick < 8 else (-.75 if tick < 20 else 0.)
    wrench = np.zeros((model.nbody, 6))
    wrench[-1] = [.02, -.015, .01, .005, -.003, .004]
    return MuJoCoControl(np.full(model.nu, command + slot * .015),
                         qfrc_applied=np.full(model.nv, .012), xfrc_applied=wrench)


def set_native(data, packet):
    for name in ("ctrl", "qfrc_applied", "xfrc_applied"):
        getattr(data, name)[:] = np.asarray(getattr(packet, name), np.float32)


def analytic(model, state, packet):
    # Independent native field interpretation, not a call into product lowering.
    output = state.act.copy()
    derivative = np.zeros(model.na)
    used = np.empty(model.nu)
    clamped = np.empty(model.nu)
    raw = np.empty(model.nu)
    force = np.empty(model.nu)
    generalized = np.zeros(model.nv)
    for actor in range(model.nu):
        u = float(np.float32(packet.ctrl[actor]))
        if model.actuator_ctrllimited[actor]:
            u = float(np.clip(u, *model.actuator_ctrlrange[actor]))
        clamped[actor] = u
        adr = int(model.actuator_actadr[actor])
        disabled = bool(model.opt.disableactuator & (1 << int(model.actuator_group[actor])))
        value = u
        if adr >= 0:
            tau = max(1e-15, float(model.actuator_dynprm[actor, 0]))
            derivative[adr] = (u - state.act[adr]) / tau
            candidate = state.act[adr] if disabled else state.act[adr] + (u-state.act[adr]) * -math.expm1(-H/tau)
            if model.actuator_actlimited[actor]:
                candidate = float(np.clip(candidate, *model.actuator_actrange[actor]))
            output[adr] = candidate
            value = candidate if model.actuator_actearly[actor] else state.act[adr]
        used[actor] = value
        joint = int(model.actuator_trnid[actor, 0])
        qa, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        gear = float(model.actuator_gear[actor, 0])
        b = model.actuator_biasprm[actor, :3]
        raw[actor] = 0 if disabled else float(model.actuator_gainprm[actor, 0])*value + b[0] + b[1]*gear*state.q[qa] + b[2]*gear*state.v[dof]
        force[actor] = raw[actor]
        if model.actuator_forcelimited[actor]:
            force[actor] = np.clip(force[actor], *model.actuator_forcerange[actor])
        generalized[dof] += gear * force[actor]
    return output, derivative, clamped, used, raw, force, generalized

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case,early,tau,disabled", (
    ("hinge", False, .004, False), ("hinge", True, .004, False),
    ("hinge", False, .00025, False), ("hinge", True, .00025, False),
    ("hinge", True, 100., False), ("slide", True, .004, False),
    ("slide", False, .004, True), ("coupled", True, .004, False),
))
def test_activation_force_motion_native_continuous(mode, case, early, tau, disabled, tmp_path):
    xml = source(case, early=early, tau=tau, disabled=disabled)
    mj, model, _, initial, runtime = fixture(xml, mode)
    history = {"source": xml, "initial": initial, "mode": mode, "steps": []}
    try:
        before = runtime.snapshot()[17]
        reference = mj.MjData(model)
        reference.qpos[:], reference.qvel[:], reference.act[:] = before.q, before.v, before.act
        reference.time = before.time
        for tick in range(32):
            packet = control(model, tick)
            entry = runtime.snapshot()[17]
            independent = analytic(model, entry, packet)
            same = mj.MjData(model)
            same.qpos[:], same.qvel[:], same.act[:] = entry.q, entry.v, entry.act
            same.time = entry.time
            set_native(same, packet)
            mj.mj_forward(model, same)
            matrix = np.zeros((model.nv, model.nv))
            mj.mj_fullM(model, matrix, same.qM)
            expected = dict(mass_matrix=matrix, bias=same.qfrc_bias.copy(),
                            passive=same.qfrc_passive.copy(),
                            applied=same.qfrc_smooth.copy()-same.qfrc_passive+same.qfrc_bias,
                            smooth_acceleration=same.qacc.copy(),
                            scalar_force=same.actuator_force.copy(), derivative=same.act_dot.copy())
            mj.mj_step(model, same)
            set_native(reference, packet)
            mj.mj_step(model, reference)
            runtime.set_controls({17: packet})
            kernels = runtime.counters["completed_kernels"]
            assert runtime.step(H) == {17: True}
            state = runtime.snapshot()[17]
            receipt = state.last_step
            details = {"entry": entry, "packet": packet, "receipt": receipt,
                       "state": state, "analytic": independent, "native": expected,
                       "same_output": (same.qpos.copy(), same.qvel.copy(), same.act.copy()),
                       "continuous": (reference.qpos.copy(), reference.qvel.copy(), reference.act.copy())}
            history["steps"].append(details)
            assert runtime.counters["completed_kernels"] - kernels == 4
            a = receipt.actuation
            np.testing.assert_array_equal(a.input_act, entry.act)
            np.testing.assert_array_equal(a.output_act, state.act)
            for channel, (actual, target) in enumerate(zip((a.output_act, a.activation_derivative, a.clamped_control,
                                       a.force_input, a.raw_force, a.scalar_force, a.generalized_force), independent, strict=True)):
                np.testing.assert_allclose(actual, target, rtol=ACT_RTOL if channel < 2 else FORCE_RTOL,
                                           atol=ACT_ATOL if channel < 2 else FORCE_ATOL)
                with pytest.raises(ValueError):
                    actual.setflags(write=True)
            np.testing.assert_allclose(a.scalar_force, expected["scalar_force"], rtol=FORCE_RTOL, atol=FORCE_ATOL)
            np.testing.assert_allclose(a.activation_derivative, expected["derivative"], rtol=ACT_RTOL, atol=ACT_ATOL)
            np.testing.assert_array_equal(a.generalized_force, receipt.qfrc_actuator)
            for name, (rtol, atol) in DYNAMICS_TOL.items():
                np.testing.assert_allclose(getattr(receipt.dynamics, name), expected[name], rtol=rtol, atol=atol)
            for native in (same, reference):
                np.testing.assert_allclose(state.q, native.qpos, rtol=0, atol=Q_ATOL)
                np.testing.assert_allclose(state.v, native.qvel, rtol=0, atol=V_ATOL)
                np.testing.assert_allclose(state.act, native.act, rtol=ACT_RTOL, atol=ACT_ATOL)
            assert abs(state.time-reference.time) <= TIME_ATOL
            if disabled:
                np.testing.assert_allclose(a.raw_force, 0, atol=0)
                np.testing.assert_allclose(a.scalar_force, .1, rtol=0, atol=2e-8)
                # Native raw derivative is nonzero even though integration uses zero.
                assert np.any(a.activation_derivative != 0)
                np.testing.assert_array_equal(a.integration_derivative, np.zeros(model.na))
            assert receipt.contacts == ()
    except BaseException as error:
        history["failure"] = repr(error)
        raise
    finally:
        history["counters"] = dict(runtime.counters)
        runtime.close()
        save(tmp_path / "activation-history.json.gz", history)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("keys", ((17,), (3, 91), (1, 7, 19, 33, 58, 91, 120, 257)))
def test_activation_checkpoint_clone_partial_reset(mode, keys, tmp_path):
    mj, model, compiled, initial, runtime = fixture(source("coupled"), mode, keys)
    with runtime:
        for tick in range(4):
            runtime.set_controls({key: control(model, tick, slot) for slot, key in enumerate(keys)})
            runtime.step(H)
        before = encode(runtime.snapshot())
        capsule = MuJoCoTyxCheckpoint.from_bytes(runtime.checkpoint().to_bytes())
        assert capsule.binding["schema"] == "mujoco-tyx-exact-checkpoint-v2"
        clone = runtime.clone()
        with clone:
            assert encode(clone.snapshot()) == before
            runtime.step(H)
            expected = encode(runtime.snapshot())
            clone.step(H)
            assert encode(clone.snapshot()) == expected
            runtime.restore(capsule, expected_revision=runtime.checkpoint_revision)
            assert encode(runtime.snapshot()) == before
            runtime.step(H)
            assert encode(runtime.snapshot()) == expected
        retained = runtime.snapshot()
        runtime.reset((keys[-1],))
        actual = runtime.snapshot()
        for key in keys[:-1]:
            assert encode(actual[key]) == encode(retained[key])
        np.testing.assert_array_equal(actual[keys[-1]].act, np.asarray(initial[keys[-1]].act, np.float32))
        with MuJoCoRuntime(compiled, world_keys=(keys[-1],), mode=mode,
                           initial_states={keys[-1]: initial[keys[-1]]}) as fresh:
            packet = control(model, 6)
            all_packets = {key: packet for key in keys}
            runtime.set_controls(all_packets)
            fresh.set_controls({keys[-1]: packet})
            runtime.step(H)
            fresh.step(H)
            assert encode(runtime.snapshot()[keys[-1]]) == encode(fresh.snapshot()[keys[-1]])
        old = encode(runtime.snapshot())
        with pytest.raises(ValueError):
            runtime.restore(capsule, expected_revision=runtime.checkpoint_revision-1)
        assert encode(runtime.snapshot()) == old
        save(tmp_path / "activation-restore.json.gz", dict(checkpoint=capsule.to_bytes().hex(),
              initial=initial, before=before, successor=expected, final=runtime.snapshot(), counters=dict(runtime.counters)))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fault", ("download", "input_act", "restore_readback"))
def test_activation_late_failure_retains_and_retries(mode, fault, monkeypatch, tmp_path):
    import phymeo.backends.mujoco_checkpoint as checkpoint_module
    _, model, _, _, runtime = fixture(source("coupled"), mode, (7, 91))
    with runtime:
        runtime.set_controls({key: control(model, 0) for key in runtime.world_keys})
        runtime.step(H)
        cp = runtime.checkpoint()
        before = encode(runtime.snapshot())
        clone = runtime.clone()
        with clone:
            with monkeypatch.context() as patch:
                if fault == "restore_readback":
                    def fail_readback(*args, **kwargs):
                        raise RuntimeError("authored late activation restore readback")
                    patch.setattr(checkpoint_module, "_verify_bank", fail_readback)
                    with pytest.raises(RuntimeError, match="authored late"):
                        runtime.restore(cp, expected_revision=runtime.checkpoint_revision)
                else:
                    original = runtime._device.download
                    def damaged_download():
                        result = original()
                        if fault == "download":
                            raise RuntimeError("authored late activation download")
                        result["input_act"][-1, 0] += .25
                        return result
                    patch.setattr(runtime._device, "download", damaged_download)
                    with pytest.raises(PhysicsStepFailure):
                        runtime.step(H)
                assert encode(runtime.snapshot()) == before
                assert runtime.checkpoint().to_bytes() == cp.to_bytes()
            runtime.step(H)
            clone.step(H)
            assert encode(runtime.snapshot()) == encode(clone.snapshot())
            save(tmp_path / "activation-fault.json.gz", dict(fault=fault, before=before,
                 retry=runtime.snapshot(), uninterrupted=clone.snapshot(), counters=dict(runtime.counters)))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("feature", ("filter", "integrator", "free", "ball", "limit", "capacity"))
def test_activation_unsupported_preallocation(mode, feature, monkeypatch):
    from phymeo.backends.mujoco_tyx import MuJoCoTyxUnsupported
    from phymeo.backends.mujoco_tyx import _tree_device
    xml = source()
    if feature in ("filter", "integrator"):
        xml = xml.replace('dyntype="filterexact"', f'dyntype="{feature}"')
    elif feature == "free":
        xml = xml.replace('<body name="base"', '<body name="floating"><freejoint/><inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/><body name="base"').replace('</worldbody>', '</body></worldbody>')
    elif feature == "ball":
        xml = xml.replace('type="hinge"', 'type="ball"')
    elif feature == "limit":
        xml = xml.replace('armature=".04"', 'limited="true" range="-1 1" armature=".04"')
    else:
        extra = ''.join(f'<general name="extra-{i}" joint="joint" dyntype="filterexact" dynprm=".004"/>' for i in range(32))
        xml = xml.replace('</actuator>', extra+'</actuator>')
    compiled = compile_mujoco_xml(xml, require_asset=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported activation reached device allocation")
    monkeypatch.setattr(_tree_device.TyxTreeDevice, "__init__", forbidden)
    with pytest.raises(MuJoCoTyxUnsupported):
        MuJoCoRuntime(compiled, mode=mode)


@pytest.mark.parametrize("mode", MODES)
def test_stateless_path_keeps_three_launches_and_v1_checkpoint(mode):
    xml = source().replace('dyntype="filterexact"', 'dyntype="none"').replace('actlimited="true"', 'actlimited="false"')
    _, model, _, _, runtime = fixture(xml, mode)
    with runtime:
        runtime.set_controls({17: control(model, 0)})
        before = runtime.counters["completed_kernels"]
        runtime.step(H)
        assert runtime.counters["completed_kernels"]-before == 3
        state = runtime.snapshot()[17]
        assert type(state.last_step).__name__ == "MuJoCoTyxStepReceipt"
        assert not hasattr(state.last_step, "actuation")
        assert runtime.checkpoint().binding["schema"] == "mujoco-tyx-exact-checkpoint-v1"

@pytest.mark.parametrize("mode", MODES)
def test_activation_step_uses_device_not_native_forward(mode, monkeypatch):
    mj, model, _, _, runtime = fixture(source("coupled", early=True), mode)
    with runtime:
        packet = control(model, 0)
        before = runtime.integration_snapshot()[17]
        runtime.set_controls({17: packet})
        def forbidden(*args, **kwargs):
            raise AssertionError("native dynamics called during TYX activation step")
        with monkeypatch.context() as patch:
            patch.setattr(mj, "mj_step", forbidden)
            patch.setattr(mj, "mj_forward", forbidden)
            runtime.step(H)
            after = runtime.integration_snapshot()[17]
            assert np.any(after.act != before.act)
            assert np.any(after.v != before.v)
            np.testing.assert_array_equal(after.act, after.last_step.actuation.output_act)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("bad", ("shape", "nonfinite", "bank-order", "source"))
def test_activation_state_identity_rejects_before_replacement(mode, bad, monkeypatch):
    import phymeo.backends.mujoco_checkpoint as checkpoint_module
    xml = source("coupled")
    _, _, compiled, initial, runtime = fixture(xml, mode)
    with runtime:
        if bad in ("shape", "nonfinite"):
            value = [0.] if bad == "shape" else [0., float("nan"), 0.]
            with pytest.raises(ValueError):
                MuJoCoRuntime(compiled, mode=mode, initial_states={0: MuJoCoInitialState(act=value)})
            return
        runtime.step(H)
        cp = runtime.checkpoint()
        destination = runtime
        other = None
        if bad == "bank-order":
            saved = cp._read(cp.binding)
            saved = dict(saved)
            # Corrupt only authoritative act bytes; unchanged mirrors/receipt
            # make the swap detectable. Unlabelled full-state changes are not
            # claimed to be inferable by an engine from values alone.
            altered = saved["act"].copy()
            altered[:, [0, 1]] = altered[:, [1, 0]]
            if np.array_equal(altered, saved["act"]):
                altered[0, 0] += .125
            saved["act"] = altered
            cp = checkpoint_module._make_checkpoint(cp.binding, cp.source_revision, saved)
        else:
            changed = compile_mujoco_xml(xml.replace('dynprm="0.004"', 'dynprm="0.005"'), require_asset=False)
            other = MuJoCoRuntime(changed, world_keys=(17,), mode=mode, initial_states=initial)
            destination = other
        try:
            before = encode(destination.snapshot())
            def forbidden(*args, **kwargs):
                raise AssertionError("invalid activation identity reached replacement allocation")
            with monkeypatch.context() as patch:
                patch.setattr(destination, "_build_device", forbidden)
                with pytest.raises(ValueError):
                    destination.restore(cp, expected_revision=destination.checkpoint_revision)
            assert encode(destination.snapshot()) == before
        finally:
            if other is not None:
                other.close()
