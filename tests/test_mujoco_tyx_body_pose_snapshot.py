"""Only committed world/body transforms leave the native observation workspace."""
from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import pytest
import test_mujoco_tyx_body_observation as bodies
import test_mujoco_tyx_runtime as slider

from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl


def _sliders(mode):
    spec = MuJoCoTyxModelSpec("slide", slider.compile_source(slider.SLIDER))
    worlds = tuple(MuJoCoTyxWorldSpec(key, spec.key) for key in (41, 7))
    return spec.compilation.fresh_native_model(), MuJoCoTyxHeterogeneousBatch(
        (spec,), worlds, mode=mode
    )


@pytest.mark.parametrize("mode", slider.MODES)
def test_controlled_worlds_publish_only_body_poses(mode):
    mj = slider.deps(mode)
    model, engine = _sliders(mode)
    keys, steps = (41, 7), 12
    controls = {41: 4., 7: -4.}
    native = {key: mj.MjData(model) for key in keys}
    snapshots = []
    with engine as batch, batch.prepare_device_workspace(include_bodies=True) as ws:
        if mode == "cuda":
            import torch

            device = ws.binding["device"]
            consumer = torch.cuda.Stream(device=device)
        batch.set_controls({k: MuJoCoControl([value]) for k, value in controls.items()})
        for tick in (0, steps, 2 * steps):
            if tick:
                for _ in range(steps):
                    assert batch.step(.001) == {41: True, 7: True}
                    for key, state in native.items():
                        state.ctrl[:] = controls[key]
                        mj.mj_step(model, state)
            frame = ws.refresh(expected_revision=batch.checkpoint_revision)
            if tick:
                assert frame.revision == batch.last_step.revision_after
            counters = ws.counters
            with torch.cuda.stream(consumer) if mode == "cuda" else nullcontext():
                snap = ws.body_pose_snapshot(frame)
            assert snap.frame is frame and snap.frame.ticks == (tick, tick)
            assert snap.world_keys == keys and snap.body_offsets == (0, 2, 4)
            transforms = snap.transforms
            assert transforms.shape == (4, 3, 4)
            if mode == "cuda":
                assert transforms.device == torch.device("cuda", device)
                assert transforms.is_contiguous() and transforms.dtype == torch.float32
                assert ws.counters["output_d2d_bytes"] - counters["output_d2d_bytes"] == 4 * 48
                assert ws.counters["body_pose_layout_upload_bytes"] == 3 * 4
                actual = transforms.cpu().numpy()
            else:
                assert transforms.dtype == np.float32 and transforms.flags.c_contiguous
                assert ws.counters["output_d2d_bytes"] == 0
                actual = transforms
            assert ws.counters["explicit_readback_bytes"] == counters["explicit_readback_bytes"]
            assert ws.counters["status_downloaded_bytes"] == counters["status_downloaded_bytes"]
            assert ws.counters["body_pose_snapshots"] == counters["body_pose_snapshots"] + 1
            for row, key in enumerate(keys):
                state = native[key]
                mj.mj_forward(model, state)
                start, stop = snap.body_offsets[row:row + 2]
                np.testing.assert_allclose(actual[start:stop, :, 3], state.xpos, atol=4e-5)
                np.testing.assert_allclose(actual[start:stop, :, :3], state.xmat.reshape(-1, 3, 3),
                                           atol=4e-5)
            if tick:
                assert actual[1, 2, 3] > 0 > actual[3, 2, 3]
            snapshots.append((snap, actual.copy()))

        # Mutating a consumer's output cannot change even a later export of the same frame.
        pristine = snapshots[-1][1]
        snap.transforms[:] = 42
        repeated = ws.body_pose_snapshot(frame).transforms
        np.testing.assert_array_equal(repeated.cpu().numpy() if mode == "cuda" else repeated, pristine)
    for snap, expected in snapshots[:-1]:
        value = snap.transforms.cpu().numpy() if mode == "cuda" else snap.transforms
        np.testing.assert_array_equal(value, expected)


def test_packed_native_body_indices_and_rotations():
    mj = slider.deps("interpret")
    specs, worlds, models = bodies.declarations(mj, (59, 7))
    with (
        MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="interpret") as batch,
        batch.prepare_device_workspace(include_bodies=True) as ws,
    ):
        assert batch.step(.001) == {59: True, 7: True}
        frame = ws.refresh(expected_revision=batch.checkpoint_revision)
        snap = ws.body_pose_snapshot(frame)
        states = batch.integration_snapshot()
        counts = tuple(batch.layout[key].nbody for key in snap.world_keys)
        assert counts[0] != counts[1] and snap.body_offsets == (0, counts[0], sum(counts))
        assert snap.transforms.shape == (sum(counts), 3, 4)
        for row, world in enumerate(worlds):
            start, stop = snap.body_offsets[row:row + 2]
            ref = bodies.native_bodies(mj, models[world.model_key], states[world.world])
            np.testing.assert_allclose(snap.transforms[start:stop, :, 3], ref["body_position"],
                                       **bodies.POSE_TOL)
            np.testing.assert_allclose(snap.transforms[start:stop, :, :3], ref["body_rotation"],
                                       **bodies.POSE_TOL)


def test_reject_wrong_source_and_keep_previous_publication(monkeypatch):
    slider.deps("interpret")
    _, engine = _sliders("interpret")
    with (
        engine as batch,
        batch.prepare_device_workspace(include_bodies=True) as ws,
        batch.prepare_device_workspace() as plain,
    ):
        previous = ws.refresh(expected_revision=0)
        foreign = plain.refresh(expected_revision=0)
        with pytest.raises(ValueError, match="include_bodies"):
            plain.body_pose_snapshot(foreign)
        for invalid in (foreign, replace(previous)):
            with pytest.raises(ValueError, match="foreign|superseded"):
                ws.body_pose_snapshot(invalid)
        initial = ws.body_pose_snapshot(previous).transforms.copy()
        with monkeypatch.context() as fault:
            def fail_refresh(*args):
                raise RuntimeError("injected observation failure")

            fault.setattr(ws._body_observer, "refresh_bucket", fail_refresh)
            with pytest.raises(RuntimeError, match="observation failure"):
                ws.refresh(expected_revision=0)
        np.testing.assert_array_equal(ws.body_pose_snapshot(previous).transforms, initial)
        with pytest.raises(ValueError, match="successful observation refresh"):
            ws.accepted_frame_time(previous, 41)
        recovered = ws.refresh(expected_revision=0)
        assert ws.accepted_frame_time(recovered, 41) == 0.
        assert batch.step(.001) == {41: True, 7: True}
        with pytest.raises(ValueError, match="stale"):
            ws.body_pose_snapshot(recovered)
        latest = ws.refresh(expected_revision=batch.checkpoint_revision)
        with pytest.raises(ValueError, match="superseded"):
            ws.body_pose_snapshot(previous)
        retained = ws.body_pose_snapshot(latest)
    with pytest.raises(RuntimeError, match="closed"):
        ws.body_pose_snapshot(latest)
    np.testing.assert_array_equal(retained.transforms, initial)
