"""Public same-entry success-frame assembly, not learned manipulation success."""

from contextlib import ExitStack
from dataclasses import replace

import numpy as np
import pytest
from test_mujoco_tyx_activation import save
from test_mujoco_tyx_body_observation import POSE_TOL, TWIST_TOL, native_bodies
from test_mujoco_tyx_checkpoint import exact

from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.manipulation_observation import ManipulationSuccessReader
from phymeo.training.manipulation_tasks import H, evaluate_goal, manipulation_scenes

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
KEYS = (41, 7, 305, 11, 203, 59, 91, 2)


def declarations(scenes, keys):
    specs = tuple(
        MuJoCoTyxModelSpec(s.key, compile_mujoco_xml(s.mjcf, require_asset=False)) for s in scenes
    )
    worlds = tuple(MuJoCoTyxWorldSpec(k, s.key) for k, s in zip(keys, scenes, strict=True))
    return specs, worlds


def loads(scene, model, receipt):
    objects = {int(model.geom(name).id) for name in scene.object_geom_names}
    fingers = [int(model.geom(name).id) for name in scene.finger_geom_names]
    support = int(model.geom(scene.support_geom_name).id)
    force, normal = 0.0, [0.0, 0.0]
    for contact in receipt.contacts:
        if not contact.active:
            continue
        a, b = contact.geometry_ids
        if a in objects and b not in objects:
            other, wrench = b, contact.force_on_geometry1_world
        elif b in objects and a not in objects:
            other, wrench = a, contact.force_on_geometry2_world
        else:
            continue
        if other in fingers:
            normal[fingers.index(other)] += contact.normal_force
        if other == support:
            force += wrench[2]
    return np.asarray(normal), max(0.0, force)


@pytest.mark.parametrize("mode", MODES)
def test_eight_scenes_consume_same_input_once_and_restore_cursor(mode, tmp_path):
    import mujoco as mj

    scenes = manipulation_scenes()
    specs, worlds = declarations(scenes, KEYS)
    by_world = dict(zip(KEYS, scenes, strict=True))
    models = {
        key: spec.compilation.fresh_native_model() for key, spec in zip(KEYS, specs, strict=True)
    }
    episodes = dict.fromkeys(KEYS, 0)
    histories, seen_support, evidence, frames = {}, set(), [], []
    with ExitStack() as stack:
        batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode))
        ws = stack.enter_context(batch.prepare_device_workspace(include_contacts=True))
        reader = ManipulationSuccessReader(ws, by_world)
        initial = ws.refresh(expected_revision=batch.checkpoint_revision)
        assert not reader.read(initial, episodes=episodes)
        assert all(record[2] == -1 for record in reader.checkpoint().records)
        for tick in range(32):
            before = batch.integration_snapshot()
            controls = {key: MuJoCoControl(np.full(batch.layout[key].nu, 0.025)) for key in KEYS}
            batch.set_controls(controls)
            assert batch.step(H) == dict.fromkeys(KEYS, True)
            frame = ws.refresh(expected_revision=batch.checkpoint_revision)
            observed = reader.read(frame, episodes=episodes)
            assert set(observed) == set(KEYS)
            cursor = reader.checkpoint()
            assert not reader.read(frame, episodes=episodes)
            assert reader.checkpoint() == cursor
            frames.append(observed)
            oracle = {}
            for key, scene in by_world.items():
                value, receipt, model = observed[key], batch.last_step.receipts[key], models[key]
                qa = int(model.jnt_qposadr[model.joint("object-free").id])
                assert value.tick == tick and value.episode == 0
                assert value.time_s == receipt.input_time
                assert value.phase == "accepted-entry"
                np.testing.assert_allclose(
                    value.object_position, receipt.input_q[qa : qa + 3], **POSE_TOL
                )
                np.testing.assert_array_equal(value.object_wxyz, receipt.input_q[qa + 3 : qa + 7])
                fingers, support = loads(scene, model, receipt)
                np.testing.assert_allclose(
                    value.finger_normal_forces_n, fingers, rtol=3e-3, atol=3e-3
                )
                np.testing.assert_allclose(value.support_up_force_n, support, rtol=3e-3, atol=3e-3)
                if support > 0.01:
                    seen_support.add(key)
                if tick in (0, 16, 31):
                    expected = native_bodies(mj, model, before[key])
                    body = int(model.body("object").id)
                    oracle[key] = expected
                    for field, source, tol in (
                        ("object_position", "body_position", POSE_TOL),
                        ("linear_velocity", "body_linear_velocity", TWIST_TOL),
                        ("angular_velocity", "body_angular_velocity", TWIST_TOL),
                    ):
                        np.testing.assert_allclose(
                            getattr(value, field), expected[source][body], **tol
                        )
                result = evaluate_goal(
                    scene, scene.goals[scene.default_goal], value, histories.get(key)
                )
                histories[key] = result.history
                assert not result.success  # These untrained 32 steps are not the task goal.
            evidence.append(
                dict(
                    frame=frame,
                    observed=observed,
                    receipts=batch.last_step,
                    workspace=ws.readback(frame),
                    cursor=cursor,
                    native=oracle,
                )
            )
            save(tmp_path / "same-entry-success.json.gz", evidence)
            if tick == 7:
                physics_cp, cursor_cp = batch.checkpoint(), reader.checkpoint()
                (tmp_path / "success-physics.checkpoint").write_bytes(physics_cp.to_bytes())
        assert seen_support == set(KEYS)
        # A control-only revision makes a fresh public frame, not a new input sample.
        batch.set_controls(controls)
        frame = ws.refresh(expected_revision=batch.checkpoint_revision)
        assert not reader.read(frame, episodes=episodes)
        # Cursor is a distinct explicit owner, paired with the physical capsule.
        batch.restore(physics_cp, expected_revision=batch.checkpoint_revision)
        frame = ws.refresh(expected_revision=batch.checkpoint_revision)
        with pytest.raises(ValueError, match="paired cursor"):
            reader.read(frame, episodes=episodes)
        reader.restore(cursor_cp, frame=frame)
        replay = ManipulationSuccessReader(ws, by_world)
        replay.restore(cursor_cp, frame=frame)
        assert not reader.read(frame, episodes=episodes)
        assert batch.step(H) == dict.fromkeys(KEYS, True)
        frame = ws.refresh(expected_revision=batch.checkpoint_revision)
        repeated = reader.read(frame, episodes=episodes)
        assert exact(repeated) == exact(frames[8])
        assert exact(replay.read(frame, episodes=episodes)) == exact(repeated)
        # Reject malformed cursor without partially advancing reader state.
        stable = reader.checkpoint()
        with pytest.raises(ValueError, match="binding"):
            reader.restore(replace(stable, binding=()), frame=frame)
        bad = (stable.records[0][:2] + (-2,) + stable.records[0][3:], *stable.records[1:])
        with pytest.raises(ValueError, match="tick"):
            reader.restore(replace(stable, records=bad), frame=frame)
        assert reader.checkpoint() == stable
        # Reset one world: absence is omitted, other unchanged inputs are not consumed twice.
        batch.reset_states(
            {KEYS[0]: worlds[0].initial_state}, expected_revision=batch.checkpoint_revision
        )
        frame = ws.refresh(expected_revision=batch.checkpoint_revision)
        with pytest.raises(ValueError, match="episode"):
            reader.read(frame, episodes=episodes)
        assert batch.step(H) == dict.fromkeys(KEYS, True)
        frame = ws.refresh(expected_revision=batch.checkpoint_revision)
        with pytest.raises(ValueError, match="episode"):
            reader.read(frame, episodes=episodes)
        assert reader.checkpoint() == stable
        episodes[KEYS[0]] = 1
        late_wrong = dict(episodes)
        late_wrong[KEYS[-1]] = 1
        with pytest.raises(ValueError, match="episode"):
            reader.read(frame, episodes=late_wrong)
        assert reader.checkpoint() == stable
        reset_values = reader.read(frame, episodes=episodes)
        assert reset_values[KEYS[0]].episode == 1 and reset_values[KEYS[0]].tick == 0
        assert all(reset_values[k].tick == 9 for k in KEYS[1:])
        # Skipping an actual physical input is rejected and leaves all cursors unchanged.
        stable = reader.checkpoint()
        for _ in range(2):
            assert batch.step(H) == dict.fromkeys(KEYS, True)
        frame = ws.refresh(expected_revision=batch.checkpoint_revision)
        with pytest.raises(ValueError, match="skipped"):
            reader.read(frame, episodes=episodes)
        assert reader.checkpoint() == stable
        evidence.append(dict(replay=repeated, reset=reset_values, cursor=stable))
        save(tmp_path / "same-entry-success.json.gz", evidence)


@pytest.mark.parametrize("mode", MODES)
def test_success_reader_requires_public_entry_and_exact_scene_source(mode):
    scene = manipulation_scenes()[0]
    specs, worlds = declarations((scene,), (41,))
    with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode) as batch:
        with batch.prepare_device_workspace() as endpoint:
            with pytest.raises(ValueError, match="accepted-entry"):
                ManipulationSuccessReader(endpoint, {41: scene})
        with batch.prepare_device_workspace(include_contacts=True) as ws:
            with pytest.raises(ValueError, match="XML"):
                ManipulationSuccessReader(ws, {41: replace(scene, mjcf=scene.mjcf + "\n")})
            with pytest.raises(ValueError, match="Scene"):
                ManipulationSuccessReader(ws, {})
            reader = ManipulationSuccessReader(ws, {41: scene})
            frame = ws.refresh(expected_revision=batch.checkpoint_revision)
            for episodes in ({}, {41: True}, {41: -1}, {41: 0, 7: 0}):
                with pytest.raises(ValueError, match="episode"):
                    reader.read(frame, episodes=episodes)
            assert reader.checkpoint().records == ()


@pytest.mark.parametrize("mode", MODES)
def test_tick_zero_reset_requires_real_boundary_and_paired_restore(mode, tmp_path):
    scene = manipulation_scenes()[0]
    specs, worlds = declarations((scene,), (41,))
    with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode) as batch:
        with batch.prepare_device_workspace(include_contacts=True) as ws:
            reader = ManipulationSuccessReader(ws, {41: scene})
            never_read_cursor = reader.checkpoint()
            empty = ws.refresh(expected_revision=batch.checkpoint_revision)
            assert not reader.read(empty, episodes={41: 0})
            empty_cursor = reader.checkpoint()
            reader.restore(empty_cursor, frame=empty)
            assert batch.step(H) == {41: True}
            frame = ws.refresh(expected_revision=batch.checkpoint_revision)
            first = reader.read(frame, episodes={41: 0})[41]
            original_cursor, physics = reader.checkpoint(), batch.checkpoint()
            with pytest.raises(ValueError, match="episode"):
                reader.read(frame, episodes={41: 1})
            with pytest.raises(ValueError, match="paired physical"):
                reader.restore(empty_cursor, frame=frame)
            assert reader.checkpoint() == original_cursor
            # No intervening read of the empty reset: identical first-step bytes
            # still belong to a different committed physical episode.
            batch.reset()
            assert batch.step(H) == {41: True}
            frame = ws.refresh(expected_revision=batch.checkpoint_revision)
            with pytest.raises(ValueError, match="episode"):
                reader.read(frame, episodes={41: 0})
            with pytest.raises(ValueError, match="reset boundary"):
                reader.restore(original_cursor, frame=frame)
            second = reader.read(frame, episodes={41: 1})[41]
            assert replace(second, episode=0) == first
            assert not reader.read(frame, episodes={41: 1})
            # An observed empty reset is checkpointable and consumes no SuccessFrame.
            batch.reset_states(
                {41: worlds[0].initial_state}, expected_revision=batch.checkpoint_revision
            )
            frame = ws.refresh(expected_revision=batch.checkpoint_revision)
            with pytest.raises(ValueError, match="empty physical"):
                reader.restore(never_read_cursor, frame=frame)
            assert not reader.read(frame, episodes={41: 2})
            reset_cursor, reset_physics = reader.checkpoint(), batch.checkpoint()
            assert reset_cursor.records[0][2] == -1
            assert batch.step(H) == {41: True}
            frame = ws.refresh(expected_revision=batch.checkpoint_revision)
            assert reader.read(frame, episodes={41: 2})[41].tick == 0
            batch.restore(reset_physics, expected_revision=batch.checkpoint_revision)
            frame = ws.refresh(expected_revision=batch.checkpoint_revision)
            reader.restore(reset_cursor, frame=frame)
            assert not reader.read(frame, episodes={41: 2})
            assert batch.step(H) == {41: True}
            frame = ws.refresh(expected_revision=batch.checkpoint_revision)
            assert reader.read(frame, episodes={41: 2})[41].tick == 0
            # Restore is not a reset even if the historical bytes match exactly.
            batch.restore(physics, expected_revision=batch.checkpoint_revision)
            frame = ws.refresh(expected_revision=batch.checkpoint_revision)
            with pytest.raises(ValueError, match="paired cursor"):
                reader.read(frame, episodes={41: 3})
            reader.restore(original_cursor, frame=frame)
            assert not reader.read(frame, episodes={41: 0})
            save(
                tmp_path / "tick-zero-boundaries.json.gz",
                dict(
                    first=first,
                    second=second,
                    original=original_cursor,
                    reset=reset_cursor,
                    restored=reader.checkpoint(),
                    frame=frame,
                ),
            )
