"""M2 public native-state IO: heterogeneous CUDA/interpret routing and transactions."""

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import fields, is_dataclass

import numpy as np
import pytest
import test_mujoco_tyx_activation as activation
import test_mujoco_tyx_forest as forest
import tyx as tx

from phymeo import MuJoCoTyxHeterogeneousWorkspace
from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.resident_stepper import PhysicsStepFailure

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
KEYS = (41, 7, 305, 11, 203, 59, 91, 2)
KINDS = dict(zip(KEYS, ("forest", "activation", "passive", "two-free") * 2, strict=True))
H = 0.001


def exact(value):
    """Compare every field/byte; mapping insertion order is not semantic state."""
    if isinstance(value, np.ndarray):
        return value.dtype.str, value.shape, value.tobytes()
    if is_dataclass(value):
        return type(value).__name__, tuple(
            (f.name, exact(getattr(value, f.name))) for f in fields(value)
        )
    if isinstance(value, Mapping):
        return tuple(
            (k, exact(value[k])) for k in sorted(value, key=lambda k: (type(k).__name__, repr(k)))
        )
    if isinstance(value, (tuple, list)):
        return tuple(exact(x) for x in value)
    return value


def first_difference(left, right, path=()):
    if type(left) is not type(right):
        return path, type(left).__name__, type(right).__name__
    if isinstance(left, tuple):
        if len(left) != len(right):
            return path, len(left), len(right)
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            if a != b:
                return first_difference(a, b, (*path, i))
        return None
    return path, repr(left)[:160], repr(right)[:160]


def declarations(keys=KEYS):
    sources = {
        "forest": forest.source("prp", contact=3),
        "two-free": forest.source("two-free", object_first=False),
        "activation": activation.source("hinge", early=True),
        "passive": '<mujoco><option timestep=".001" gravity="0 0 0"/>'
        '<worldbody><body><joint type="slide" axis="1 0 0"/>'
        '<geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/>'
        "</body></worldbody></mujoco>",
    }
    names = tuple(dict.fromkeys(KINDS[k] for k in keys))
    specs = tuple(
        MuJoCoTyxModelSpec(name, compile_mujoco_xml(sources[name], require_asset=False))
        for name in names
    )
    models = {s.key: s.compilation.fresh_native_model() for s in specs}
    starts = {}
    for k in keys:
        model = models[KINDS[k]]
        starts[k] = MuJoCoInitialState(
            model.qpos0,
            np.full(model.nv, 0.01 + k * 0.00001),
            act=np.full(model.na, 0.02),
            time=0.125,
        )
    return specs, tuple(MuJoCoTyxWorldSpec(k, KINDS[k], starts[k]) for k in keys)


def buffers(stack, mode, values):
    result = []
    for value in values:
        if value is None:
            result.append(None)
        elif mode == "interpret":
            result.append(value.copy())
        else:
            device = tx.CudaArray(value)
            stack.callback(device.free)
            result.append(device)
    return tuple(result)


def commands(batch, tick=0):
    dims = {
        name: max(1, max(getattr(r, name) for r in batch.layout.values()))
        for name in ("nu", "nv", "nbody")
    }
    n = len(batch.world_keys)
    values = [
        np.full((n, dims["nu"]), np.nan, np.float32),
        np.full((n, dims["nv"]), np.nan, np.float32),
        np.full((n, dims["nbody"], 6), np.nan, np.float32),
    ]
    packets = {}
    for i, k in enumerate(batch.world_keys):
        row = batch.layout[k]
        values[0][i, : row.nu] = 0.03 + 0.001 * tick + 0.0001 * k
        values[1][i, : row.nv] = 0.0
        values[1][i, 0] = -0.0
        values[2][i, : row.nbody] = 0.0
        packets[k] = MuJoCoControl(
            values[0][i, : row.nu],
            qfrc_applied=values[1][i, : row.nv],
            xfrc_applied=values[2][i, : row.nbody],
        )
    if not any(r.nu for r in batch.layout.values()):
        values[0] = None
    return tuple(values), packets


def submit(batch, values, **kwargs):
    return batch.set_controls_from_device(
        values[0],
        qfrc_applied=values[1],
        xfrc_applied=values[2],
        world_keys=batch.world_keys,
        expected_revision=batch.checkpoint_revision,
        **kwargs,
    )


def compare_frame(batch, ws, frame):
    data = ws.readback(frame)
    states = batch.integration_snapshot()
    for i, k in enumerate(batch.world_keys):
        row, state = batch.layout[k], states[k]
        for name, width in (("q", row.nq), ("v", row.nv), ("act", row.na)):
            np.testing.assert_array_equal(data[name][i, :width], getattr(state, name))
            assert np.all(data[name][i, width:] == 0)
            np.testing.assert_array_equal(
                data[name + "_mask"][i], np.arange(data[name].shape[1]) < width
            )
        assert float(data["clock"][i, 0]) - float(data["clock"][i, 1]) == state.time
        np.testing.assert_array_equal(
            data["ctrl_mask"][i], np.arange(data["ctrl_mask"].shape[1]) < row.nu
        )
        assert not data["q"].flags.writeable
    return data


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("reverse", (False, True))
def test_w8_native_masks_device_action_and_w1_successors(mode, reverse, tmp_path, monkeypatch):
    keys = KEYS[::-1] if reverse else KEYS
    specs, worlds = declarations(keys)
    with ExitStack() as stack:
        batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode))
        ws = stack.enter_context(batch.prepare_device_workspace())
        assert isinstance(ws, MuJoCoTyxHeterogeneousWorkspace)
        single = {
            w.world: stack.enter_context(
                MuJoCoRuntime(
                    next(s.compilation for s in specs if s.key == w.model_key),
                    world_keys=(w.world,),
                    initial_states={w.world: w.initial_state},
                    mode=mode,
                    implementation="tyx",
                )
            )
            for w in worlds
        }
        frame = ws.refresh(expected_revision=batch.checkpoint_revision)
        compare_frame(batch, ws, frame)
        assert ws.binding["free_v"] == "world-linear,body-local-angular"
        assert ws.model_metadata["two-free"]["jnt_qposadr"][3] > 0
        assert not ws.model_metadata["forest"]["jnt_qposadr"].flags.writeable
        history = []
        for tick in range(3):
            values, packets = commands(batch, tick)
            with ExitStack() as inputs:
                supplied = buffers(inputs, mode, values)
                receipt = submit(batch, supplied)
                assert receipt.world_keys == keys
                for leaf in receipt.receipts:
                    for k, control in leaf.controls.items():
                        assert exact(control) == exact(packets[k])
                # Accepted packets own their input data.
                if mode == "interpret":
                    supplied[0][:] = 99
            for k, owner in single.items():
                owner.set_controls({k: packets[k]})
                owner.step(H)
            batch.step(H)
            states = batch.integration_snapshot()
            for k, owner in single.items():
                reference = owner.integration_snapshot()[k]
                for name in ("q", "v", "act", "ctrl", "time", "tick"):
                    assert exact(getattr(states[k], name)) == exact(getattr(reference, name))
            with monkeypatch.context() as guard:
                guard.setattr(
                    type(batch._buckets[0]),
                    "snapshot",
                    lambda *a, **kw: pytest.fail("CPU endpoint forward in refresh"),
                )
                guard.setattr(
                    type(batch._buckets[0]),
                    "integration_snapshot",
                    lambda *a, **kw: pytest.fail("HOST state in refresh"),
                )
                frame = ws.refresh(expected_revision=batch.checkpoint_revision)
            history.append((batch.last_step, compare_frame(batch, ws, frame)))
        activation.save(tmp_path / "w8.json.gz", history)


@pytest.mark.parametrize("mode", MODES)
def test_zero_width_sentinel_and_existing_t0(mode):
    specs, worlds = declarations((305,))
    with (
        MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode) as batch,
        batch.prepare_device_workspace() as ws,
    ):
        data = compare_frame(batch, ws, ws.refresh(expected_revision=0))
        assert data["act"].shape == (1, 1) and data["ctrl_mask"].shape == (1, 1)
        assert not data["act_mask"].any() and not data["ctrl_mask"].any()
        batch.set_controls_from_device(None, world_keys=(305,), expected_revision=0)
        batch.step(H)
        compare_frame(batch, ws, ws.refresh(expected_revision=batch.checkpoint_revision))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("phase", ("audit", "publication", "observation", "reset", "physics"))
def test_late_failure_preserves_complete_publication_and_retries(mode, phase, monkeypatch):
    from phymeo.backends.mujoco_device_controls import _DeviceControlPacket

    specs, worlds = declarations((41, 7, 305))
    with (
        MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode) as batch,
        batch.prepare_device_workspace() as ws,
    ):
        with ExitStack() as stack:
            values = buffers(stack, mode, commands(batch)[0])
            submit(batch, values)
        batch.step(H)
        frame = ws.refresh(expected_revision=batch.checkpoint_revision)
        before, old_frame = exact(batch.integration_snapshot()), exact(ws.readback(frame))
        revision, last = batch.checkpoint_revision, batch.last_step
        with monkeypatch.context() as patch:
            if phase == "audit":
                original = _DeviceControlPacket._audit

                def fail(packet):
                    result = original(packet)
                    if packet.engine is batch._buckets[-1]:
                        raise RuntimeError("authored late audit")
                    return result

                patch.setattr(_DeviceControlPacket, "_audit", fail)
            elif phase == "publication":

                def fail(*args):
                    raise RuntimeError("authored final publication")

                patch.setattr(batch, "_before_commit", fail)
            elif phase == "observation":
                original, calls = ws._scatter, []

                def fail(*args, **kwargs):
                    original(*args, **kwargs)
                    calls.append(1)
                    if len(calls) == 12:
                        raise RuntimeError("authored final scatter")

                patch.setattr(ws, "_scatter", fail)
            elif phase == "reset":

                def fail(operation, result):
                    if operation == "reset-states":
                        raise RuntimeError("authored reset before publication")

                patch.setattr(batch, "_before_commit", fail)
            else:
                original = batch._buckets[-1]._device.download

                def fail(*args, **kwargs):
                    original(*args, **kwargs)
                    raise RuntimeError("authored last physics readback")

                patch.setattr(batch._buckets[-1]._device, "download", fail)
            with pytest.raises((RuntimeError, PhysicsStepFailure)):
                if phase in ("audit", "publication"):
                    with ExitStack() as stack:
                        submit(batch, buffers(stack, mode, commands(batch, 1)[0]))
                elif phase == "observation":
                    ws.refresh(expected_revision=revision)
                elif phase == "reset":
                    batch.reset_states(
                        {41: MuJoCoInitialState(), 7: MuJoCoInitialState()},
                        expected_revision=revision,
                    )
                else:
                    batch.step(H)
        assert exact(batch.integration_snapshot()) == before
        assert batch.checkpoint_revision == revision and batch.last_step is last
        assert exact(ws.readback(frame)) == old_frame
        with ExitStack() as stack:
            submit(batch, buffers(stack, mode, commands(batch, 1)[0]))
        batch.step(H)
        compare_frame(batch, ws, ws.refresh(expected_revision=batch.checkpoint_revision))


@pytest.mark.parametrize("mode", MODES)
def test_partial_state_reset_and_checkpoint_replay(mode):
    specs, worlds = declarations()
    with (
        MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode) as batch,
        batch.prepare_device_workspace() as ws,
    ):
        with ExitStack() as stack:
            submit(batch, buffers(stack, mode, commands(batch)[0]))
        batch.step(H)
        old = ws.refresh(expected_revision=batch.checkpoint_revision)
        before = batch.integration_snapshot()
        selection = (41, 59)
        reset = {
            k: MuJoCoInitialState(
                next(w.initial_state.q for w in worlds if w.world == k),
                np.full(batch.layout[k].nv, 0.025),
                act=before[k].act,
            )
            for k in selection
        }
        batch.reset_states(reset, expected_revision=batch.checkpoint_revision)
        after = batch.integration_snapshot()
        for k in batch.world_keys:
            if k not in selection:
                assert exact(before[k]) == exact(after[k]), (
                    k,
                    first_difference(exact(before[k]), exact(after[k])),
                )
            else:
                np.testing.assert_array_equal(after[k].v, np.asarray(reset[k].v, np.float32))
                assert after[k].time == 0 and after[k].tick == 0 and after[k].last_step is None
        with pytest.raises(ValueError, match="stale"):
            ws.readback(old)
        compare_frame(batch, ws, ws.refresh(expected_revision=batch.checkpoint_revision))
        capsule = batch.checkpoint()
        with ExitStack() as stack:
            values = buffers(stack, mode, commands(batch, 1)[0])
            submit(batch, values)
            batch.step(H)
            successor = exact(batch.integration_snapshot())
            batch.restore(capsule, expected_revision=batch.checkpoint_revision)
            compare_frame(batch, ws, ws.refresh(expected_revision=batch.checkpoint_revision))
            submit(batch, values)
            batch.step(H)
        assert exact(batch.integration_snapshot()) == successor
        before_revision = batch.checkpoint_revision
        batch.reset_states({}, expected_revision=before_revision)
        assert batch.checkpoint_revision == before_revision


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "bad", ("nan", "inf", "subnormal", "shape", "dtype", "order", "stale", "alias")
)
def test_invalid_actions_rejected_before_any_publication(mode, bad):
    specs, worlds = declarations((7,))
    with (
        MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode) as batch,
        batch.prepare_device_workspace() as ws,
    ):
        values = list(commands(batch)[0])
        if bad in ("nan", "inf", "subnormal"):
            values[0][0, 0] = {
                "nan": np.nan,
                "inf": np.inf,
                "subnormal": np.nextafter(np.float32(0), np.float32(1)),
            }[bad]
        elif bad == "shape":
            values[0] = np.zeros((1, 2), np.float32)
        elif bad == "dtype":
            values[0] = values[0].astype(np.float64)
        before, revision = exact(batch.integration_snapshot()), batch.checkpoint_revision
        with ExitStack() as stack:
            supplied = list(buffers(stack, mode, values))
            if bad == "alias":
                ws.refresh(expected_revision=revision)
                supplied[0] = ws._banks[ws._active]["q"]
            with pytest.raises((ValueError, TypeError)):
                batch.set_controls_from_device(
                    supplied[0],
                    qfrc_applied=supplied[1],
                    xfrc_applied=supplied[2],
                    world_keys=(False,) if bad == "order" else batch.world_keys,
                    expected_revision=-1 if bad == "stale" else revision,
                )
        assert (
            exact(batch.integration_snapshot()) == before and batch.checkpoint_revision == revision
        )


@pytest.mark.cuda
def test_cuda_torch_owned_exports_and_consumer_action_stream(monkeypatch):
    import torch
    from tyx.torch import from_torch, stream_from_torch

    specs, worlds = declarations()
    with (
        MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda") as batch,
        batch.prepare_device_workspace() as ws,
    ):
        frame = ws.refresh(expected_revision=0)
        arrays = ws.to_torch(frame)
        before = {k: v.clone() for k, v in arrays.items()}
        assert all(v.is_cuda for v in arrays.values())
        producer = torch.cuda.Stream()
        with torch.cuda.stream(producer), ExitStack() as stack:
            # Public state/masks feed native actuator requests on CUDA.
            ctrl = torch.full(arrays["ctrl_mask"].shape, 0.04, device="cuda")
            ctrl[~arrays["ctrl_mask"].bool()] = float("nan")
            view = from_torch(ctrl, backend="cuda")
            stack.callback(view.free)
            stream = stream_from_torch(producer, backend="cuda")
            stack.callback(stream.close)
            batch.set_controls_from_device(
                view, world_keys=batch.world_keys, expected_revision=0, producer_stream=stream
            )
        batch.step(H)
        new = ws.refresh(expected_revision=batch.checkpoint_revision)
        exported = ws.to_torch(new)
        assert exported["q"].data_ptr() != arrays["q"].data_ptr()
        for name in arrays:
            assert torch.equal(arrays[name], before[name])
        arrays["q"].fill_(999)
        compare_frame(batch, ws, new)
        with pytest.raises(ValueError, match="stale|superseded"):
            ws.to_torch(frame)
        closed = batch.prepare_device_workspace()
        closed.close()
        with pytest.raises(RuntimeError, match="closed"):
            closed.refresh(expected_revision=batch.checkpoint_revision)


@pytest.mark.parametrize("mode", MODES)
def test_invalid_late_reset_keeps_all_worlds_and_default_templates(mode):
    specs, worlds = declarations((41, 7))
    with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode) as batch:
        before, revision = exact(batch.integration_snapshot()), batch.checkpoint_revision
        with pytest.raises(ValueError):
            batch.reset_states(
                {41: MuJoCoInitialState(), 7: MuJoCoInitialState([0, 0])},
                expected_revision=revision,
            )
        assert (
            exact(batch.integration_snapshot()) == before and batch.checkpoint_revision == revision
        )
        with pytest.raises(ValueError, match="zero"):
            batch.reset_states({7: MuJoCoInitialState(time=1)}, expected_revision=revision)
        batch.reset_states(
            {7: MuJoCoInitialState([0.1], [0.2], act=[0.03])}, expected_revision=revision
        )
        batch.reset((7,))
        assert exact(batch.integration_snapshot()) == before


@pytest.mark.cuda
def test_workspace_unknown_release_keeps_dependencies_and_never_double_frees(monkeypatch):
    specs, worlds = declarations((7,))
    batch = MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda")
    ws = batch.prepare_device_workspace()
    # Inject after actual release: its effect is deliberately unknowable to the owner.
    original_release = ws._resources._buffers[-1]
    calls = []

    def fail():
        original_release.callback()
        calls.append(1)
        raise RuntimeError("authored ambiguous native free")

    from dataclasses import replace

    ws._resources._buffers[-1] = replace(original_release, callback=fail)
    with pytest.raises(BaseExceptionGroup, match="UNKNOWN_RELEASE"):
        ws.close()
    assert ws._resources.state == "UNKNOWN_RELEASE"
    assert ws._resources.runtime_pending and ws._dependencies.state == "OPEN"
    with pytest.raises(BaseExceptionGroup, match="UNKNOWN_RELEASE"):
        ws.close()
    assert calls == [1]
    with pytest.raises(BaseExceptionGroup, match="heterogeneous cleanup"):
        batch.close()
    assert calls == [1] and ws in batch._workspaces
