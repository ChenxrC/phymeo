"""Predeclared C2b native device action cases; parent owns all execution."""
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import fields, is_dataclass

import numpy as np
import pytest
import tyx as tx

from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.backends.mujoco_task_workspace import MuJoCoTyxTaskSpec
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.resident_stepper import PhysicsStepFailure

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
KEYS = (5, 41)
H = .002
INERTIAL = '<inertial pos=".02 -.01 .03" mass="1.2" fullinertia=".04 .05 .06 .001 -.001 .001"/>'
VISUAL = '<geom type="sphere" size=".04" contype="0" conaffinity="0"/>'
CHILD = f"""<body pos=".2 0 .1"><joint name="a" type="hinge" axis="0 1 0"
 damping=".03" armature=".02" limited="true" range="-.5 .5"/>{INERTIAL}{VISUAL}
 <body pos=".1 .1 0"><joint name="b" type="slide" axis="1 0 0"
 damping=".03" stiffness=".4" springref=".04"/>{INERTIAL}{VISUAL}</body></body>"""


def engine(case, mode, keys=KEYS):
    assert pytest.importorskip("mujoco").__version__ == "3.3.7"
    body = CHILD
    if case != "fixed":
        shape = '<geom type="sphere" size=".1"/>' if case == "mixed" else VISUAL
        body = f'<body pos="0 0 .099"><freejoint/>{INERTIAL}{shape}{CHILD}</body>'
    ground = '<geom type="plane" size="2 2 .1"/>' if case == "mixed" else ""
    actuator = "" if case == "mixed" else """<actuator><general joint="a"
      gaintype="fixed" gainprm="1.2" biastype="affine" biasprm="0 -.2 -.04"
      ctrllimited="true" ctrlrange="-.05 .05" forcelimited="true"
      forcerange="-.08 .08"/></actuator>"""
    source = compile_mujoco_xml(f"""<mujoco><compiler angle="radian" autolimits="true"/>
      <option timestep=".002" gravity="0 0 -9.81" integrator="Euler" cone="pyramidal"
      iterations="128" tolerance="1e-10"/><default><geom condim="3" friction=".6 .02 .003"
      solref=".02 1"/></default><worldbody>{ground}{body}</worldbody>{actuator}</mujoco>""",
      require_asset=False)
    model = source.fresh_native_model()
    initial = {}
    for key in keys:
        q, v = model.qpos0.copy(), np.full(model.nv, .015)
        if case != "fixed":
            orientation = np.array([1., .01, -.015, .02])
            q[3:7] = orientation / np.linalg.norm(orientation)
            v[:6] = [.025, -.03, -.01, .08, -.05, .03]
        q[-1], v[-1] = (.03, .02) if key == 5 else (.04, .08)
        if case == "mixed":
            q[-2] = .501
        initial[key] = MuJoCoInitialState(q, v)
    return MuJoCoRuntime(source, mode=mode, world_keys=keys, initial_states=initial,
                        contact_capacity=16, constraint_capacity=64)


def exact(value):
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if is_dataclass(value):
        return (type(value).__name__, tuple((f.name, exact(getattr(value, f.name))) for f in fields(value)))
    if isinstance(value, Mapping):
        return tuple((k, exact(v)) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(exact(v) for v in value)
    return value


def state(e):
    return exact(e.snapshot())


def channels(e):
    model = e._model
    ctrl = None if model.nu == 0 else np.array([[.12], [-.11]], np.float32)
    force = np.full((2, model.nv), .017, np.float32)
    force[1] *= -1
    wrench = np.zeros((2, model.nbody, 6), np.float32)
    wrench[:, -1] = [.03, -.02, .04, .005, -.004, .003]
    return (ctrl, force, wrench)


def device_values(stack, mode, values):
    result = []
    for value in values:
        if value is None:
            result.append(None)
        elif mode == "interpret":
            result.append(value.copy())
        else:
            item = tx.CudaArray(value)
            stack.callback(item.free)
            result.append(item)
    return tuple(result)


def submit(e, values, **kwargs):
    return e.set_controls_from_device(values[0], qfrc_applied=values[1],
        xfrc_applied=values[2], expected_revision=e.checkpoint_revision, **kwargs)


def host_submit(e, values):
    e.set_controls({key: MuJoCoControl([] if values[0] is None else values[0][i],
        qfrc_applied=None if values[1] is None else values[1][i],
        xfrc_applied=None if values[2] is None else values[2][i])
        for i, key in enumerate(e.world_keys)})


def policy(count: tx.i32, nu: tx.i32, nv: tx.i32, nbody: tx.i32,
           observation: tx.ndarray(tx.f32, 2), ctrl: tx.ndarray(tx.f32, 2),
           force: tx.ndarray(tx.f32, 2), wrench: tx.ndarray(tx.f32, 3)):
    for w in tx.range(count):
        for i in range(nu):
            ctrl[w, i] = .02 + .1 * observation[w, 0]
        for i in range(nv):
            force[w, i] = .01 * (i + 1) * observation[w, 1]
        for b in range(nbody):
            for i in range(6):
                wrench[w, b, i] = 0.0
                if b == nbody - 1:
                    wrench[w, b, i] = .002 * (i + 1) * observation[w, 0]


policy = tx.kernel(policy)


@pytest.mark.parametrize("mode", MODES)
def test_real_task_policy_actions_and_complete_host_equivalence(mode):
    with engine("fixed", mode) as e, engine("fixed", mode) as ref, ExitStack() as stack:
        spec = MuJoCoTyxTaskSpec((0,), (1,), scale=[1.2, -.7], bias=[.1, .2])
        with e.prepare_task_workspace(spec) as ws:
            frame = ws.refresh(expected_revision=e.checkpoint_revision)
            values = device_values(stack, mode, (np.zeros((2, 2), np.float32),
                np.zeros(2, np.float32), np.zeros(2, np.int32)))
            runtime = tx.Runtime(backend="cuda" if mode == "cuda" else "cpu")
            stack.callback(runtime.close)
            stream = None
            if mode == "cuda":
                stream = tx.CudaStream()
                stack.callback(stream.close)
            ws.copy_to(frame, *values, stream=stream)
            actions = device_values(stack, mode, tuple(np.zeros_like(x) for x in channels(e)))
            with tx.use_runtime(runtime):
                options = dict(mode=mode)
                if stream is not None:
                    options["stream"] = stream
                policy(2, e._model.nu, e._model.nv, e._model.nbody, values[0], *actions, **options)
                runtime.sync()
            expected = tuple(x.copy() if mode == "interpret" else x.to_numpy(stream=stream) for x in actions)
            if stream is not None:
                stream.synchronize()
            revision = e.checkpoint_revision
            receipt = submit(e, actions, producer_stream=stream)
            assert receipt.source_revision == revision and receipt.revision == revision + 1
            assert receipt.binding["world_keys"] == KEYS
            with pytest.raises(ValueError):
                ws.readback(frame)
            host_submit(ref, expected)
            uploaded = e.counters["uploaded_bytes"]
            for _ in range(3):
                e.step(H)
                ref.step(H)
                assert state(e) == state(ref)
            assert e.counters["uploaded_bytes"] == uploaded
            assert e.counters["control_audit_downloaded_bytes"] == (
                sum(x.nbytes for x in expected) if mode == "cuda" else 0)
            assert e.counters["device_control_submissions"] == 1


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", ("fixed", "free", "mixed"))
def test_native_channels_clipping_coupling_and_omitted_channels(case, mode):
    with engine(case, mode) as e, engine(case, mode) as ref, ExitStack() as stack:
        original = channels(e)
        values = device_values(stack, mode, original)
        receipt = submit(e, values)
        host_submit(ref, original)
        for _ in range(3):
            e.step(H)
            ref.step(H)
            assert state(e) == state(ref)
        for key in KEYS:
            np.testing.assert_array_equal(e.snapshot()[key].last_step.stored_control.qfrc_applied,
                                          receipt.controls[key].qfrc_applied)
        submit(e, (values[0], None, None))
        host_submit(ref, (original[0], None, None))
        e.step(H)
        ref.step(H)
        assert state(e) == state(ref)
        assert all(not np.any(s.held_control.qfrc_applied) and not np.any(s.held_control.xfrc_applied)
                   for s in e.snapshot().values())
        if case == "mixed":
            assert any(s.last_step.constraints for s in e.snapshot().values())


@pytest.mark.parametrize("mode", MODES)
def test_caller_lifetime_and_explicit_world_permutation(mode):
    with engine("free", mode) as e, engine("free", mode, KEYS[::-1]) as permuted, ExitStack() as stack:
        original = channels(e)
        values = device_values(stack, mode, original)
        swapped = device_values(stack, mode, tuple(x[::-1].copy() for x in original))
        submit(e, values)
        submit(permuted, swapped)
        for value in values:
            if mode == "interpret":
                value.fill(17)
            else:
                value.free()
        e.step(H)
        permuted.step(H)
        for key in KEYS:
            assert exact(e.snapshot()[key]) == exact(permuted.snapshot()[key])
        # Row labels are a caller declaration, never inferred from numeric values.
        assert e._control_packet.binding["world_keys"] == KEYS
        assert permuted._control_packet.binding["world_keys"] == KEYS[::-1]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("bad", ("nan", "inf", "subnormal", "negative_subnormal", "shape", "dtype", "noncontiguous"))
def test_bad_values_metadata_preserve_complete_previous_state(mode, bad):
    with engine("fixed", mode) as e, ExitStack() as stack:
        original = channels(e)
        values = device_values(stack, mode, original)
        submit(e, values)
        before, revision, old = state(e), e.checkpoint_revision, e._control_packet
        wrong = original[0].copy()
        if bad in ("nan", "inf", "subnormal", "negative_subnormal"):
            bits = {"nan": 0x7FC00001, "inf": 0x7F800000,
                    "subnormal": 1, "negative_subnormal": 0x80000001}[bad]
            wrong.view(np.uint32)[0, 0] = bits
        elif bad == "shape":
            wrong = np.zeros((2, 2), np.float32)
        elif bad == "dtype":
            wrong = wrong.astype(np.float64)
        else:
            if mode == "cuda":
                # Public TYX arrays are contiguous; an arbitrary strided HOST input is refused.
                wrong = np.zeros((2, 2), np.float32)[:, ::2]
            else:
                wrong = np.zeros((2, 2), np.float32)[:, ::2]
        invalid = wrong if bad == "noncontiguous" else device_values(stack, mode, (wrong,))[0]
        with pytest.raises((ValueError, TypeError)):
            submit(e, (invalid, values[1], values[2]))
        assert state(e) == before and e.checkpoint_revision == revision and e._control_packet is old
        e.step(H)  # The old accepted packet remains usable after a raw-bit rejection.


@pytest.mark.parametrize("mode", MODES)
def test_negative_zero_is_preserved_in_device_audit_and_physics_receipt(mode):
    with engine("fixed", mode) as e, ExitStack() as stack:
        original = channels(e)
        for value in original:
            value.view(np.uint32).flat[0] = 0x80000000
        values = device_values(stack, mode, original)
        receipt = submit(e, values)
        for value in (receipt.controls[KEYS[0]].ctrl, receipt.controls[KEYS[0]].qfrc_applied,
                      receipt.controls[KEYS[0]].xfrc_applied):
            assert value.astype(np.float32).view(np.uint32).flat[0] == 0x80000000
        e.step(H)
        stored = e.snapshot()[KEYS[0]].last_step.stored_control
        for value in (stored.ctrl, stored.qfrc_applied, stored.xfrc_applied):
            assert value.astype(np.float32).view(np.uint32).flat[0] == 0x80000000


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("stage", ("copy", "audit", "pack"))
def test_late_submission_failure_never_publishes(mode, stage, monkeypatch):
    from phymeo.backends.mujoco_device_controls import _DeviceControlPacket
    with engine("fixed", mode) as e, ExitStack() as stack:
        values = device_values(stack, mode, channels(e))
        submit(e, values)
        before, revision, old = state(e), e.checkpoint_revision, e._control_packet
        name = {"copy": "_copy_native", "audit": "_audit", "pack": "_pack"}[stage]
        method = getattr(_DeviceControlPacket, name)
        def fail(self, *args, **kwargs):
            method(self, *args, **kwargs)
            raise RuntimeError("authored " + stage + " failure")
        with monkeypatch.context() as patch:
            patch.setattr(_DeviceControlPacket, name, fail)
            with pytest.raises(RuntimeError, match="authored"):
                submit(e, values)
        assert state(e) == before and e.checkpoint_revision == revision and e._control_packet is old
        e.step(H)


@pytest.mark.parametrize("mode", MODES)
def test_stale_alias_wrong_owner_and_closed_inputs(mode, monkeypatch):
    with engine("fixed", mode) as e, engine("fixed", mode) as other, ExitStack() as stack:
        values = device_values(stack, mode, channels(e))
        submit(e, values)
        before, revision = state(e), e.checkpoint_revision
        with pytest.raises(ValueError, match="revision"):
            e.set_controls_from_device(values[0], expected_revision=revision - 1)
        with pytest.raises(ValueError, match="owner"):
            e._control_packet.inputs(other)
        with pytest.raises(ValueError, match="alias"):
            submit(e, (e._device.controls, values[1], values[2]))
        with monkeypatch.context() as patch:
            patch.setattr(e, "_control_epoch", e._control_epoch + 1)
            with pytest.raises(ValueError, match="epoch"):
                e._control_packet.inputs(e)
        if mode == "cuda":
            dead = tx.CudaStream()
            dead.close()
            with pytest.raises(Exception):
                submit(e, values, producer_stream=dead)
            with monkeypatch.context() as patch:
                patch.setattr(values[0], "device_ordinal", 1)
                with pytest.raises(ValueError, match="device"):
                    submit(e, values)
            with monkeypatch.context() as patch:
                live = tx.CudaStream()
                stack.callback(live.close)
                patch.setattr(live, "device_ordinal", 1)
                with pytest.raises(ValueError, match="stream"):
                    submit(e, values, producer_stream=live)
        assert state(e) == before and e.checkpoint_revision == revision


@pytest.mark.parametrize("mode", MODES)
def test_channel_overlap_refused(mode):
    with engine("fixed", mode) as e, ExitStack() as stack:
        values = device_values(stack, mode, channels(e))
        if mode == "interpret":
            overlapping = values[1][:, :1]
        else:
            overlapping = tx.CudaArray.from_external(values[1].device_ptr, (2, 1), np.float32,
                                                      owner=values[1])
            stack.callback(overlapping.free)
        with pytest.raises(ValueError, match="overlap|storage"):
            submit(e, (overlapping, values[1], values[2]))


@pytest.mark.parametrize("mode", MODES)
def test_physics_failure_reset_restore_and_host_supersession(mode, monkeypatch):
    with engine("free", mode) as e, ExitStack() as stack:
        values = device_values(stack, mode, channels(e))
        submit(e, values)
        held, revision = exact(e._controls), e.checkpoint_revision
        before = state(e)
        advance = e._device.advance
        def rejected(*args):
            advance(*args)
            raise RuntimeError("authored physics fault")
        with monkeypatch.context() as patch:
            patch.setattr(e._device, "advance", rejected)
            with pytest.raises(PhysicsStepFailure):
                e.step(H)
        assert state(e) == before and exact(e._controls) == held and e.checkpoint_revision == revision
        e.step(H)
        cp = e.checkpoint()
        e.step(H)
        expected = state(e)
        e.restore(cp, expected_revision=e.checkpoint_revision)
        assert e._control_packet is None
        e.step(H)
        assert state(e) == expected
        submit(e, values)
        keep = exact(e._controls[KEYS[1]])
        e.reset(keys=(KEYS[0],))
        assert e._control_packet is None and exact(e._controls[KEYS[1]]) == keep
        assert not np.any(e._controls[KEYS[0]].ctrl)
        submit(e, values)
        host_submit(e, channels(e))
        assert e._control_packet is None
        e.step(H)


@pytest.mark.parametrize("mode", MODES)
def test_post_commit_cleanup_failure_is_retained_and_close_attempts_all(mode, monkeypatch):
    e = engine("fixed", mode)
    try:
        with ExitStack() as stack:
            values = device_values(stack, mode, channels(e))
            submit(e, values)
            first = e._control_packet
            first_close = first.close
            attempts = []
            def fail_first():
                attempts.append("first")
                raise RuntimeError("authored cleanup")
            with monkeypatch.context() as patch:
                patch.setattr(first, "close", fail_first)
                receipt = submit(e, values)
                second = e._control_packet
                assert receipt.revision == e.checkpoint_revision
                assert first in e._control_retired
                assert e.control_cleanup_errors
                second_close = second.close
                def fail_second():
                    attempts.append("second")
                    raise RuntimeError("second cleanup")
                patch.setattr(second, "close", fail_second)
                attempts.clear()
                with pytest.raises(ExceptionGroup):
                    e.close()
                assert attempts == ["first", "second"]
                assert first in e._control_retired and second in e._control_retired
            first_close()
            second_close()
            e.close()
            assert not e._control_retired
    finally:
        e.close()
