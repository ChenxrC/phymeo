"""Predeclared C2 device task tests; no HOST policy observations in refresh."""

from contextlib import ExitStack
from dataclasses import replace
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tyx as tx

from phymeo.backends.mujoco_task_workspace import (
    MuJoCoTyxTaskFailure, MuJoCoTyxTaskSpec,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
KEYS = (3, 19)
H = .002
INERTIAL = '<inertial pos=".02 -.01 .03" mass="1.2" fullinertia=".04 .05 .06 .001 -.001 .001"/>'
SHAPE = '<geom type="sphere" size=".04" contype="0" conaffinity="0"/>'
CHILD = f"""<body name="link" pos=".2 0 .1"><joint name="a" type="hinge" axis="0 0 1"
  damping=".03" armature=".02"/>{INERTIAL}{SHAPE}
  <body name="tool" pos=".1 .1 0"><joint name="b" type="slide" axis="1 0 0"
  damping=".03"/>{INERTIAL}{SHAPE}</body></body>"""
FREE = f'<body name="root" pos=".1 0 .7"><freejoint/>{INERTIAL}{SHAPE}{CHILD}</body>'
MIXED = f"""<geom type="plane" size="2 2 .1"/><body name="root" pos="0 0 .099">
  <freejoint/>{INERTIAL}<geom type="sphere" size=".1"/>{CHILD}</body>"""


def compiled(case):
    assert pytest.importorskip("mujoco").__version__ == "3.3.7"
    body = {"fixed": CHILD, "free": FREE, "mixed": MIXED}[case]
    text = f"""<mujoco model="device-task"><compiler angle="radian" autolimits="true"/>
      <option timestep=".002" gravity="0 0 -9.81" integrator="Euler" cone="pyramidal"
        solver="Newton" iterations="128" tolerance="1e-10"/>
      <default><geom condim="3" friction=".6 .02 .003" solref=".02 1"/></default>
      <worldbody>{body}</worldbody><actuator><motor joint="a"/></actuator></mujoco>"""
    return compile_mujoco_xml(text, require_asset=False)


def engine_for(case, mode):
    source = compiled(case)
    model = source.fresh_native_model()
    declarations = {}
    for slot, key in enumerate(KEYS):
        q, v = model.qpos0.copy(), np.full(model.nv, .015)
        if case != "fixed":
            orientation = np.array([1., .01, -.015, .02])
            q[3:7] = orientation / np.linalg.norm(orientation)
            v[:6] = [.025, -.03, -.01, .08, -.05, .03]
        q[-1] = .03 + .01 * slot
        v[-1] = .02 + .06 * slot
        declarations[key] = MuJoCoInitialState(q, v)
    return MuJoCoRuntime(source, mode=mode, world_keys=KEYS, initial_states=declarations,
                        contact_capacity=16, constraint_capacity=64)


def task_spec(engine):
    nq, nv = engine.capabilities["nq"], engine.capabilities["nv"]
    return MuJoCoTyxTaskSpec(
        (0, nq - 1), (0, nv - 1), scale=[1.2, -.7, 2., .3],
        bias=[.2, .3, -.4, .1], target=[.1, -.1, .03, .02], weight=[.2, .4, .6, .8],
        lower=[-10.] * 4, upper=[10., 10., 10., .05],
    )


def reference(states, spec):
    feature = np.stack([np.r_[s.q[list(spec.q_indices)], s.v[list(spec.v_indices)]]
                       for s in states.values()])
    observation = feature * spec.scale + spec.bias
    reward = -np.sum(spec.weight * (feature - spec.target) ** 2, axis=1)
    terminated = np.any((feature < spec.lower) | (feature > spec.upper), axis=1).astype(np.int32)
    return observation, reward, terminated


def qv(engine):
    return tuple((s.q.tobytes(), s.v.tobytes(), s.time, s.tick)
                 for s in engine.snapshot().values())


def output_arrays(stack, mode, count, width, *, sentinel=0):
    arrays = (np.full((count, width), sentinel, np.float32),
              np.full(count, sentinel, np.float32), np.full(count, sentinel, np.int32))
    if mode == "interpret":
        return arrays
    values = tuple(tx.CudaArray(value) for value in arrays)
    for value in values:
        stack.callback(value.free)
    return values


def host(values, mode, stream=None):
    if mode == "interpret":
        return tuple(value.copy() for value in values)
    result = tuple(value.to_numpy(stream=stream) for value in values)
    if stream is not None:
        stream.synchronize()
    return result


def consume(count: tx.i32, width: tx.i32,
            observation: tx.ndarray(tx.f32, 2), reward: tx.ndarray(tx.f32, 1),
            terminated: tx.ndarray(tx.i32, 1), output: tx.ndarray(tx.f32, 1)):
    for w in tx.range(count):
        value = reward[w] + 100.0 * terminated[w]
        for feature in range(width):
            value = value + observation[w, feature] * (feature + 1)
        output[w] = value


consume = tx.kernel(consume)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", ("fixed", "free", "mixed"))
def test_device_outputs_and_real_consumer_without_host_observation(case, mode, monkeypatch):
    with engine_for(case, mode) as engine:
        engine.set_controls({key: MuJoCoControl([.02]) for key in KEYS})
        for _ in range(3):
            engine.step(H)
        spec = task_spec(engine)
        expected = reference(engine.snapshot(), spec)
        physical = qv(engine)
        with engine.prepare_task_workspace(spec) as workspace:
            def forbidden(*args, **kwargs):
                raise AssertionError("task refresh read physics/observations to HOST")
            with monkeypatch.context() as patch:
                patch.setattr(engine, "snapshot", forbidden)
                patch.setattr(engine._mj, "mj_forward", forbidden)
                patch.setattr(engine._device, "_get", forbidden)
                if mode == "cuda":
                    for array in (*engine._device.q, *engine._device.v, *engine._device.time,
                                  *workspace._observation, *workspace._reward, *workspace._terminated):
                        patch.setattr(array, "to_numpy", forbidden)
                frame = workspace.refresh(expected_revision=engine.checkpoint_revision)
            assert qv(engine) == physical
            assert workspace.counters["task_status_downloaded_bytes"] == (8 if mode == "cuda" else 0)
            assert workspace.counters["explicit_output_readback_bytes"] == 0
            assert frame.binding["spec_sha256"] == spec.sha256
            assert frame.binding["world_keys"] == KEYS
            with ExitStack() as stack:
                consumer_runtime = tx.Runtime(backend="cuda" if mode == "cuda" else "cpu")
                stack.callback(consumer_runtime.close)
                stream = None
                if mode == "cuda":
                    stream = stack.enter_context(_stream())
                targets = output_arrays(stack, mode, 2, 4)
                workspace.copy_to(frame, *targets, stream=stream)
                output = np.zeros(2, np.float32)
                if mode == "cuda":
                    output = tx.CudaArray(output)
                    stack.callback(output.free)
                with tx.use_runtime(consumer_runtime):
                    options = {"mode": mode}
                    if stream is not None:
                        options["stream"] = stream
                    consume(2, 4, *targets, output, **options)
                    consumer_runtime.sync()
                actual = host(targets, mode, stream)
                scalar = host((output,), mode, stream)[0]
            np.testing.assert_allclose(actual[0], expected[0], rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(actual[1], expected[1], rtol=2e-5, atol=2e-6)
            np.testing.assert_array_equal(actual[2], expected[2])
            assert set(actual[2]) == {0, 1}
            np.testing.assert_allclose(scalar, actual[1] + 100 * actual[2]
                                       + actual[0] @ np.arange(1, 5), rtol=3e-6, atol=3e-6)
            assert workspace.counters["explicit_output_readback_bytes"] == 0


class _stream:
    def __enter__(self):
        self.value = tx.CudaStream()
        return self.value

    def __exit__(self, *args):
        self.value.synchronize()
        self.value.close()


@pytest.mark.parametrize("mode", MODES)
def test_same_and_other_consumer_stream_copies_are_independent(mode):
    with engine_for("free", mode) as engine, engine.prepare_task_workspace(task_spec(engine)) as ws:
        frame = ws.refresh(expected_revision=engine.checkpoint_revision)
        original = ws.readback(frame)
        physical = qv(engine)
        with ExitStack() as stack:
            targets = output_arrays(stack, mode, 2, 4)
            streams = [None]
            if mode == "cuda":
                streams += [ws._stream, stack.enter_context(_stream())]
            for stream in streams:
                ws.copy_to(frame, *targets, stream=stream)
                for actual, expected in zip(host(targets, mode, stream), original, strict=True):
                    np.testing.assert_array_equal(actual, expected)
            # Mutating the caller's output storage cannot mutate workspace or physics.
            if mode == "interpret":
                targets[0][:] = 123
            else:
                targets[0].copy_from(np.full((2, 4), 123, np.float32))
            assert qv(engine) == physical
            np.testing.assert_array_equal(ws.readback(frame)[0], original[0])


@pytest.mark.parametrize("mode", MODES)
def test_all_copy_destinations_preflight_before_any_write(mode, monkeypatch):
    with engine_for("fixed", mode) as engine, engine.prepare_task_workspace(task_spec(engine)) as ws:
        frame = ws.refresh(expected_revision=engine.checkpoint_revision)
        with ExitStack() as stack:
            targets = output_arrays(stack, mode, 2, 4, sentinel=17)
            bad = np.zeros(2, np.float32)  # Wrong third output dtype.
            if mode == "cuda":
                bad = tx.CudaArray(bad)
                stack.callback(bad.free)
            with pytest.raises(ValueError, match="dtype/shape"):
                ws.copy_to(frame, targets[0], targets[1], bad)
            assert np.all(host(targets[:2], mode)[0] == 17)
            assert np.all(host(targets[:2], mode)[1] == 17)
            with pytest.raises(ValueError, match="alias"):
                ws.copy_to(frame, ws._observation[ws._active], targets[1], targets[2])
            if mode == "cuda":
                dead = tx.CudaArray(np.zeros(2, np.int32))
                dead.free()
                with pytest.raises(Exception):
                    ws.copy_to(frame, targets[0], targets[1], dead)
                assert np.all(host(targets[:2], mode)[0] == 17)
                stream = tx.CudaStream()
                stream.close()
                with pytest.raises(Exception):
                    ws.copy_to(frame, *targets, stream=stream)
                assert np.all(host(targets[:2], mode)[1] == 17)
                with monkeypatch.context() as patch:
                    patch.setattr(targets[2], "device_ordinal", 1)
                    with pytest.raises(ValueError, match="device"):
                        ws.copy_to(frame, *targets)
                assert np.all(host(targets[:2], mode)[0] == 17)


@pytest.mark.parametrize("mode", MODES)
def test_physical_bank_alias_and_overlapping_caller_outputs_refused(mode):
    with engine_for("fixed", mode) as engine:
        spec = MuJoCoTyxTaskSpec((0,) * 8)
        with engine.prepare_task_workspace(spec) as ws, ExitStack() as stack:
            frame = ws.refresh(expected_revision=engine.checkpoint_revision)
            physical = qv(engine)
            targets = output_arrays(stack, mode, 2, 8)
            with pytest.raises(ValueError, match="alias"):
                ws.copy_to(frame, engine._device.q[engine._device.active], targets[1], targets[2])
            if mode == "interpret":
                # Exact overlapping f32/int32 views of the same caller allocation.
                overlapping = targets[1].view(np.int32)
            else:
                overlapping = tx.CudaArray.from_external(
                    int(targets[1].device_ptr), (2,), np.int32, owner=targets[1])
                stack.callback(overlapping.free)
            with pytest.raises(ValueError, match="overlap"):
                ws.copy_to(frame, targets[0], targets[1], overlapping)
            assert qv(engine) == physical


@pytest.mark.parametrize("mode", MODES)
def test_revision_owner_reset_restore_and_close_lifetimes(mode):
    engine = engine_for("free", mode)
    ws = engine.prepare_task_workspace(task_spec(engine))
    try:
        frame = ws.refresh(expected_revision=engine.checkpoint_revision)
        cp = engine.checkpoint()
        with pytest.raises(ValueError, match="foreign|superseded"):
            ws.readback(replace(frame))
        with engine.prepare_task_workspace(ws.spec) as other:
            with pytest.raises(ValueError, match="foreign|superseded"):
                other.readback(frame)
        engine.set_controls({key: MuJoCoControl([.01]) for key in KEYS})
        with pytest.raises(ValueError, match="stale"):
            ws.readback(frame)
        with pytest.raises(ValueError, match="stale"):
            ws.refresh(expected_revision=frame.revision)
        fresh = ws.refresh(expected_revision=engine.checkpoint_revision)
        engine.step(H)
        with pytest.raises(ValueError, match="stale"):
            ws.readback(fresh)
        engine.reset(keys=(KEYS[0],))
        after_reset = ws.refresh(expected_revision=engine.checkpoint_revision)
        assert after_reset.ticks == (0, 1)
        engine.restore(cp, expected_revision=engine.checkpoint_revision)
        with pytest.raises(ValueError, match="stale"):
            ws.readback(after_reset)
        restored = ws.refresh(expected_revision=engine.checkpoint_revision)
        assert restored.ticks == frame.ticks and restored.times == frame.times
        expected = reference(engine.snapshot(), ws.spec)
        for actual, wanted in zip(ws.readback(restored), expected, strict=True):
            np.testing.assert_allclose(actual, wanted, rtol=2e-5, atol=2e-6)
        engine.close()
        with pytest.raises(RuntimeError, match="closed"):
            ws.readback(restored)
    finally:
        engine.close()
        ws.close()
    with pytest.raises(RuntimeError, match="closed"):
        ws.readback(restored)


@pytest.mark.parametrize("mode", MODES)
def test_failed_refresh_preserves_latest_frame_only_for_same_physics_revision(mode, monkeypatch):
    with engine_for("fixed", mode) as engine, engine.prepare_task_workspace(task_spec(engine)) as ws:
        frame = ws.refresh(expected_revision=engine.checkpoint_revision)
        original = ws.readback(frame)
        kernel = ws._kernel
        def rejected(*args, **kwargs):
            kernel(*args, **kwargs)
            if mode == "interpret":
                args[6][:] = 777
                args[-1][:] = 2
            else:
                kwargs["stream"].synchronize()
                bad_output, bad_status = np.full((2, 4), 777, np.float32), np.full(2, 2, np.int32)
                args[6].copy_from(bad_output, stream=kwargs["stream"])
                args[-1].copy_from(bad_status, stream=kwargs["stream"])
                kwargs["stream"].synchronize()
        with monkeypatch.context() as patch:
            patch.setattr(ws, "_kernel", rejected)
            with pytest.raises(MuJoCoTyxTaskFailure) as caught:
                ws.refresh(expected_revision=engine.checkpoint_revision)
            assert caught.value.physics_accepted and not caught.value.physics_mutated
            assert not caught.value.workspace_published
            for actual, wanted in zip(ws.readback(frame), original, strict=True):
                np.testing.assert_array_equal(actual, wanted)
            engine.step(H)
            physical = qv(engine)
            with pytest.raises(MuJoCoTyxTaskFailure):
                ws.refresh(expected_revision=engine.checkpoint_revision)
            with pytest.raises(ValueError, match="stale"):
                ws.readback(frame)
            assert qv(engine) == physical
        assert ws.counters["failed_refreshes"] == 2
        fresh = ws.refresh(expected_revision=engine.checkpoint_revision)
        assert fresh.epoch == frame.epoch + 1


@pytest.mark.parametrize("mode", MODES)
def test_actual_nonfinite_task_output_rejects_without_physics_mutation(mode):
    with engine_for("fixed", mode) as engine:
        limit = float(np.finfo(np.float32).max)
        spec = MuJoCoTyxTaskSpec((engine.capabilities["nq"] - 1,), scale=[limit], bias=[limit])
        with engine.prepare_task_workspace(spec) as ws:
            physical = qv(engine)
            with pytest.raises(MuJoCoTyxTaskFailure):
                ws.refresh(expected_revision=engine.checkpoint_revision)
            assert qv(engine) == physical and ws.counters["published_frames"] == 0


@pytest.mark.parametrize("mode", MODES)
def test_torch_export_explicitly_refuses_cpu_only_torch(mode, monkeypatch):
    with engine_for("fixed", mode) as engine, engine.prepare_task_workspace(task_spec(engine)) as ws:
        frame = ws.refresh(expected_revision=engine.checkpoint_revision)
        fake = SimpleNamespace(version=SimpleNamespace(cuda=None),
                               cuda=SimpleNamespace(is_available=lambda: False))
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "torch", fake)
            with pytest.raises(RuntimeError, match="CUDA"):
                ws.to_torch(frame)
        assert ws.counters["output_d2d_bytes"] == 0


@pytest.mark.parametrize("mode", MODES)
def test_native_bounds_are_inclusive_and_do_not_compare_scaled_observation(mode):
    with engine_for("fixed", mode) as engine:
        states = engine.snapshot()
        bound = float(states[KEYS[0]].q[-1])
        spec = MuJoCoTyxTaskSpec((engine.capabilities["nq"] - 1,), scale=[-10], bias=[20],
                                lower=[bound], upper=[bound])
        with engine.prepare_task_workspace(spec) as ws:
            frame = ws.refresh(expected_revision=engine.checkpoint_revision)
            observation, reward, terminated = ws.readback(frame)
            assert observation[0, 0] > 10
            np.testing.assert_array_equal(terminated, [0, 1])
            np.testing.assert_array_equal(reward, [0, 0])


def test_spec_is_immutable_and_bounded():
    scale = np.array([2., 3.])
    spec = MuJoCoTyxTaskSpec((0,), (1,), scale=scale)
    digest = spec.sha256
    scale[:] = 99
    np.testing.assert_array_equal(spec.scale, [2., 3.])
    with pytest.raises(ValueError):
        spec.scale[0] = 7
    assert spec.sha256 == digest
    assert MuJoCoTyxTaskSpec((0,), (1,), scale=[2, 3]).sha256 == digest
    for args, kwargs in [
        (((), ()), {}), (((True,), ()), {}), (((-1,), ()), {}),
        (((0,) * 257, ()), {}), (((0,), ()), {"weight": [-1e-100]}),
        (((0,), ()), {"lower": [2], "upper": [1]}),
        (((0,), ()), {"scale": [float("nan")]}),
    ]:
        with pytest.raises(ValueError):
            MuJoCoTyxTaskSpec(*args, **kwargs)


@pytest.mark.parametrize("mode", MODES)
def test_native_indices_reject_before_workspace_runtime_allocation(mode, monkeypatch):
    with engine_for("fixed", mode) as engine:
        physical = qv(engine)
        nq, nv = engine.capabilities["nq"], engine.capabilities["nv"]
        def forbidden(*args, **kwargs):
            raise AssertionError("invalid native index allocated a task runtime")
        with monkeypatch.context() as patch:
            patch.setattr(tx, "Runtime", forbidden)
            for spec in (MuJoCoTyxTaskSpec((nq,)), MuJoCoTyxTaskSpec((), (nv,))):
                with pytest.raises(ValueError, match="outside native"):
                    engine.prepare_task_workspace(spec)
        assert qv(engine) == physical
