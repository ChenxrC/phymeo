"""Compiled public scene consumption without private runtime access on hot paths."""

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_scene_compiler import H, _world

import phymeo as ph
from phymeo.builders.scene_compiler import SceneDeclaration, compile_scene
from phymeo.training.resident_stepper import PhysicsStepFailure


def cache(tmp_path):
    return Path(os.environ.get("PHYMEO_RESIDENT_CACHE", tmp_path))


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
@pytest.mark.parametrize("count", [1, 8, 64])
@pytest.mark.parametrize("contact", [False, True])
def test_compiled_scene_prepared_controls_fields_contact_reset(
    mode, count, contact, tmp_path, monkeypatch
):
    initial = ph.RobotCommand(
        np.array([0.25]), target_velocity=np.array([0.5]), feedforward=np.array([0.125])
    )
    worlds = tuple(
        replace(_world(k, contact=contact), initial_control=initial) for k in range(count)
    )
    with compile_scene(SceneDeclaration(worlds), mode, cache_root=cache(tmp_path)) as scene:
        keys = tuple(reversed(scene.world_keys))
        with (
            scene.prepare_control_write(
                keys,
                fields=("target_position", "target_velocity", "feedforward", "external_effort"),
            ) as control,
            scene.prepare_field_read(keys) as state,
            scene.prepare_body_read(keys, fields=("position", "linear_velocity")) as body,
            scene.prepare_reset() as reset,
        ):
            values = np.empty(control.layout["shape"], np.float32)
            control.read_into(values)
            np.testing.assert_array_equal(values, np.tile([0.25, 0.5, 0.125, 0], (count, 1)))
            values[:, 2] = 1
            values[:, 3] = np.arange(count) / 64
            qv = np.empty(state.layout["shape"], np.float32)
            poses = np.empty(body.layout["shape"], np.float32)
            state.read_into(qv)
            initial_state = qv.copy()
            with monkeypatch.context() as patch:
                patch.setattr(scene, "snapshot", lambda *a, **kw: pytest.fail("snapshot hot path"))
                patch.setattr(
                    scene, "set_controls", lambda *a, **kw: pytest.fail("packet hot path")
                )
                control.write(values)
                assert all(scene.step(H, require_all=True).values())
                state.read_into(qv)
                body.read_into(poses)
            assert np.all(qv[:, 1] > 0)
            np.testing.assert_allclose(poses[:, 2], 3 / 32 + qv[:, 0], atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(poses[:, 5], qv[:, 1], atol=1e-6, rtol=1e-5)
            if contact:
                assert len(scene.contact_receipts) == count
            previous = qv.copy()
            mask = np.zeros(count, bool)
            mask[0] = True
            before_reset = scene.stats()["batch"]
            reset.reset(mask)
            state.read_into(qv)
            np.testing.assert_array_equal(qv[-1], initial_state[-1])
            np.testing.assert_array_equal(qv[:-1], previous[:-1])
            actual = np.empty_like(values)
            control.read_into(actual)
            np.testing.assert_array_equal(actual[-1], [0.25, 0.5, 0.125, 0])
            np.testing.assert_array_equal(actual[:-1], values[:-1])
            if contact:
                assert 0 not in scene.contact_receipts
            # Legacy reset must also preserve commands written by prepared plans.
            scene.reset([0])
            control.read_into(actual)
            np.testing.assert_array_equal(actual[:-1], values[:-1])
            assert all(scene.step(H, require_all=True).values())
            counters = scene.stats()["batch"]
            assert counters["device_allocations"] == before_reset["device_allocations"]
            print(
                "COMPILED_RESIDENT_IO "
                + json.dumps(
                    dict(
                        mode=mode,
                        worlds=count,
                        contact=contact,
                        actual_state=qv.tolist() if count == 1 else None,
                        stats=dict(scene.stats()),
                        control=dict(control.stats()),
                    ),
                    default=str,
                )
            )


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_prepared_target_limits_atomic_and_stale(mode, tmp_path):
    with compile_scene(
        SceneDeclaration((_world(3), _world(9))), mode, cache_root=cache(tmp_path)
    ) as scene:
        control = scene.prepare_control_write(
            fields=("target_position", "target_velocity", "external_effort")
        )
        values = np.zeros((2, 3), np.float32)
        before = values.copy()
        control.read_into(before)
        for column, bad in ((0, 3), (1, 21), (2, np.nan)):
            values.fill(0)
            values[1, column] = bad
            with pytest.raises(ValueError):
                control.write(values)
            control.read_into(values)
            np.testing.assert_array_equal(values, before)
        with pytest.raises(ValueError, match="nonempty"):
            scene.prepare_control_write([], fields=("target_position",))
        scene.close()
        assert control.stats()["scene_limit_workspace_bytes"] == 0
        with pytest.raises(RuntimeError, match="closed"):
            control.write(values)
        control.close()


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_atomic_rejection_keeps_all_worlds_and_allows_recovery(mode, tmp_path):
    worlds = (_world(0), _world(1))
    with compile_scene(SceneDeclaration(worlds), mode, cache_root=cache(tmp_path)) as scene:
        with (
            scene.prepare_control_write(fields=("external_effort",)) as controls,
            scene.prepare_field_read() as state,
        ):
            original = np.empty(state.layout["shape"], np.float32)
            state.read_into(original)
            controls.write(np.array([[1.0], [3e38]], np.float32))
            with pytest.raises(PhysicsStepFailure) as caught:
                scene.step(1 / 8, require_all=True)
            assert caught.value.acceptance_known and not caught.value.physics_accepted
            assert not scene.needs_reset
            current = np.empty_like(original)
            state.read_into(current)
            np.testing.assert_array_equal(current, original)
            controls.write(np.zeros((2, 1), np.float32))
            assert scene.step(H, require_all=True) == {0: True, 1: True}


def test_host_refuses_prepared_io_and_all_world_contract():
    with compile_scene(SceneDeclaration((_world(0),))) as scene:
        with pytest.raises(NotImplementedError):
            scene.prepare_field_read()
        with pytest.raises(NotImplementedError):
            scene.prepare_reset()
        with pytest.raises(NotImplementedError):
            scene.step(H, require_all=True)
        assert scene.step(H) == {0: True}


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_closed_scene_plans_do_not_accumulate_owners(mode, tmp_path):
    with compile_scene(SceneDeclaration((_world(0),)), mode, cache_root=cache(tmp_path)) as scene:
        baseline = len(scene._owners)
        for _ in range(12):
            with scene.prepare_control_write(fields=("target_position",)):
                assert len(scene._owners) == baseline + 1
            assert len(scene._owners) == baseline
        live = scene.prepare_control_write(fields=("target_position",))
        scene.close()
        assert not scene._owners
        assert live.stats()["scene_limit_workspace_bytes"] == 0
        live.close()


def test_scene_close_keeps_parent_after_child_cleanup_failure(tmp_path, monkeypatch):
    with compile_scene(
        SceneDeclaration((_world(0, contact=True),)), "cuda", cache_root=cache(tmp_path)
    ) as scene:
        # Failure injection only: actual consumers use the public methods above.
        child, raw = scene._plan, scene._raw
        original = child.close

        def fail():
            raise RuntimeError("injected contact close")

        monkeypatch.setattr(child, "close", fail)
        with pytest.raises(RuntimeError, match="cleanup"):
            scene.close()
        assert not raw._closed and child in scene._owners
        monkeypatch.setattr(child, "close", original)
        scene.close()
        assert raw._closed


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_prepared_reset_recovers_partial_scene_only_as_full_batch(mode, tmp_path):
    with compile_scene(
        SceneDeclaration((_world(0), _world(1))), mode, cache_root=cache(tmp_path)
    ) as scene:
        with (
            scene.prepare_control_write(fields=("external_effort",)) as controls,
            scene.prepare_reset() as reset,
            scene.prepare_field_read() as state,
        ):
            controls.write(np.array([[1.0], [3e38]], np.float32))
            with pytest.raises(PhysicsStepFailure) as caught:
                scene.step(1 / 8)
            assert caught.value.accepted_worlds == {0: True, 1: False}
            assert scene.needs_reset
            values = np.empty(state.layout["shape"], np.float32)
            with pytest.raises(RuntimeError, match="reset all"):
                state.read_into(values)
            with pytest.raises(RuntimeError, match="reset all"):
                reset.reset(np.array([False, True]))
            reset.reset()
            assert not scene.needs_reset
            state.read_into(values)
            np.testing.assert_array_equal(values, 0)
            assert scene.step(H, require_all=True) == {0: True, 1: True}


@pytest.mark.cuda
@pytest.mark.parametrize("retryable", [True, False], ids=["before-release", "unknown-release"])
def test_scene_construction_cleanup_failure_keeps_complete_owner(tmp_path, monkeypatch, retryable):
    from phymeo.backends._resident_cleanup import (
        ResidentCleanupError,
        RetryableReleaseError,
        UnknownReleaseError,
    )
    from phymeo.backends.tyx_world_batch import TyxWorldBatch
    from phymeo.builders.scene_compiler import CompiledScene

    allocate, free = TyxWorldBatch._buffer, TyxWorldBatch._free_buffer
    children, enabled, releases = [], [True], []

    def fail_allocate(owner, *args, **kwargs):
        if children:
            raise RuntimeError("injected second allocation")
        value = allocate(owner, *args, **kwargs)
        children.append(owner)
        return value

    def fail_free(owner, *args):
        releases.append(owner)
        if enabled[0]:
            if retryable:
                raise RetryableReleaseError("injected before native release")
            raise RuntimeError("injected cleanup outcome unknown")
        return free(owner, *args)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(TyxWorldBatch, "_buffer", fail_allocate)
            patch.setattr(TyxWorldBatch, "_free_buffer", fail_free)
            with pytest.raises(ResidentCleanupError) as caught:
                compile_scene(SceneDeclaration((_world(0),)), "cuda", cache_root=cache(tmp_path))
        scene = caught.value.cleanup_owner
        assert isinstance(scene, CompiledScene)
        assert not children[0]._runtime.closed
        enabled[0] = False
        if retryable:
            before_retry = len(releases)
            scene.close()
            assert children[0]._runtime.closed
            assert len(releases) == before_retry + 1
            scene.close()
            assert len(releases) == before_retry + 1
        else:
            for _ in range(2):
                with pytest.raises(UnknownReleaseError, match="UNKNOWN_RELEASE"):
                    scene.close()
                assert len(releases) == 1
                assert not children[0]._runtime.closed
    finally:
        enabled[0] = False
