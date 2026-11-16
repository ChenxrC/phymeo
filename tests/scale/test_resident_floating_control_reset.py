"""Prepared floating controls/reset: owner binding, masked publication and failures."""

import json

import numpy as np
import pytest
from test_resident_floating_read import FIELDS, H, cache, specification

import phymeo as ph

CONTROLS = (
    "target_position",
    "target_velocity",
    "feedforward",
    "external_effort",
    "com_force",
    "com_torque",
)


def read(plan):
    result = np.empty(plan.layout["shape"], np.float32)
    plan.read_into(result)
    return result


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
@pytest.mark.parametrize("count", [1, 8, 64])
@pytest.mark.parametrize("child", [False, True])
def test_floating_controls_masked_reset(mode, count, child, tmp_path, monkeypatch):
    spec = specification(child)
    keys = tuple(7 + 3 * i for i in range(count))
    with ph.TyxWorldBatch([(k, spec) for k in keys], mode, cache_root=cache(tmp_path)) as batch:
        with (
            batch.prepare_control_write(fields=CONTROLS) as controls,
            batch.prepare_control_write(keys[::-1], fields=CONTROLS) as reverse,
            batch.prepare_body_read(fields=FIELDS) as body,
            batch.prepare_field_read(fields=("position", "velocity", "applied_effort")) as joints,
            batch.prepare_reset() as reset,
        ):
            initial = batch.snapshot()
            initial_body, initial_joint, initial_control = read(body), read(joints), read(controls)
            values = np.full(controls.layout["shape"], 1 / 32, np.float32)
            values += np.arange(count, dtype=np.float32)[:, None] / 1024
            reverse.write(values)
            np.testing.assert_array_equal(read(controls), values[::-1])
            for _ in range(2):
                assert all(batch.step(H, require_all=True).values())
            current = batch.snapshot()
            current_body, current_joint, current_control = read(body), read(joints), read(controls)
            for parity in (0, 1):
                mask = np.arange(count) % 2 == parity
                before, cost = batch.stats(), reset.stats()
                with monkeypatch.context() as patch:
                    patch.setattr(batch, "snapshot", lambda *a: pytest.fail("hot snapshot"))
                    patch.setattr(batch, "restore", lambda *a: pytest.fail("hot restore"))
                    patch.setattr(batch, "_read", lambda *a: pytest.fail("hot download"))
                    reset.reset(mask)
                after = batch.stats()
                for counter in (
                    "device_allocations",
                    "floating_fk_kernel_launches",
                    "floating_energy_kernel_launches",
                ):
                    assert after[counter] == before[counter]
                assert after["synchronizations"] - before["synchronizations"] == int(mask.any())
                assert reset.stats()["uploaded_bytes"] - cost["uploaded_bytes"] == (
                    4 * count if mode == "cuda" and mask.any() else 0
                )
                for plan, original, previous in (
                    (body, initial_body, current_body),
                    (joints, initial_joint, current_joint),
                    (controls, initial_control, current_control),
                ):
                    actual = read(plan)
                    np.testing.assert_array_equal(actual[mask], original[mask])
                    np.testing.assert_array_equal(actual[~mask], previous[~mask])
                actual = batch.snapshot()
                for i, key in enumerate(keys):
                    assert actual[key] == (initial[key] if mask[i] else current[key])
                current = actual
                current_body, current_joint, current_control = (
                    read(body),
                    read(joints),
                    read(controls),
                )
            assert batch.snapshot() == initial
            # Legacy packet replacement must not leave prepared plans pointing at old arrays.
            batch.set_controls({keys[0]: spec.inputs(np.full((int(child), 3), 1 / 16))})
            reverse.write(values)
            np.testing.assert_array_equal(read(controls), values[::-1])
            batch.step(H, require_all=True)
            reset.reset()
            assert batch.snapshot() == initial
            print(
                "FLOATING_CONTROL_RESET",
                mode,
                count,
                child,
                json.dumps({"control": controls.stats(), "reset": reset.stats()}),
            )


@pytest.mark.parametrize("failure", ["kernel", "sync", "control"])
def test_floating_reset_failure_preserves_owner_and_retry(failure, tmp_path, monkeypatch):
    with ph.TyxWorldBatch([(0, specification(True))], "cuda", cache_root=cache(tmp_path)) as batch:
        with (
            batch.prepare_control_write(fields=CONTROLS) as control,
            batch.prepare_reset() as reset,
        ):
            initial = batch.snapshot()
            control.write(np.full(control.layout["shape"], 1 / 32, np.float32))
            batch.step(H, require_all=True)
            before, commands = batch.snapshot(), read(control)
            epoch = batch.stats()["state_epoch"]
            launch, sync, copy = batch._launch, batch._sync, np.copyto
            fired = False

            def bad_launch(kernel, *args):
                if kernel in reset._floating.kernels:
                    raise RuntimeError("injected floating reset")
                return launch(kernel, *args)

            def bad_sync():
                nonlocal fired
                if not fired:
                    fired = True
                    raise RuntimeError("injected floating reset")
                return sync()

            def bad_copy(dst, src, **kwargs):
                nonlocal fired
                if dst is batch._floating_trees._held_effort and not fired:
                    fired = True
                    dst.fill(99)
                    raise RuntimeError("injected floating reset")
                return copy(dst, src, **kwargs)

            with monkeypatch.context() as patch:
                if failure == "kernel":
                    patch.setattr(batch, "_launch", bad_launch)
                elif failure == "sync":
                    patch.setattr(batch, "_sync", bad_sync)
                else:
                    patch.setattr(np, "copyto", bad_copy)
                with pytest.raises(RuntimeError, match="injected floating reset"):
                    reset.reset()
            assert batch.stats()["state_epoch"] == epoch
            assert batch.snapshot() == before
            np.testing.assert_array_equal(read(control), commands)
            reset.reset()
            assert batch.snapshot() == initial


def test_floating_control_precheck_and_multi_owner_rollback(tmp_path, monkeypatch):
    with ph.TyxWorldBatch(
        [(0, specification(True))], "interpret", cache_root=cache(tmp_path)
    ) as batch:
        with batch.prepare_control_write(fields=CONTROLS) as plan:
            before = read(plan)
            values = np.full(plan.layout["shape"], 1 / 32, np.float32)
            bad = values.copy()
            bad[0, -1] = np.nan
            with pytest.raises(ValueError, match="finite"):
                plan.write(bad)
            np.testing.assert_array_equal(read(plan), before)
            put, calls = np.put, 0

            def fail(array, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 4:
                    array.fill(99)
                    raise RuntimeError("injected control scatter")
                return put(array, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(np, "put", fail)
                with pytest.raises(RuntimeError, match="injected control scatter"):
                    plan.write(values)
            np.testing.assert_array_equal(read(plan), before)
            plan.write(values)
            np.testing.assert_array_equal(read(plan), values)


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_public_scene_floating_authored_control_reset(mode, tmp_path, monkeypatch):
    from phymeo.builders.scene_compiler import ArticulationWorld, SceneDeclaration, compile_scene
    from phymeo.dynamics.floating_robot import FloatingRobotState
    from phymeo.runtime._floating_robot_participant import FloatingRobotCommand
    from phymeo.solvers.floating_tree import FloatingRobotInputs

    path = tmp_path / "root.urdf"
    path.write_text("""<robot name="free"><link name="root"><inertial><mass value="2"/>
      <inertia ixx=".5" iyy=".5" izz=".5" ixy="0" ixz="0" iyz="0"/></inertial>
      </link></robot>""")
    asset = ph.load_floating_urdf(path, gravity=(0, 0, -8))
    state = FloatingRobotState(np.array([0.0, 0, 1, 0, 0, 0, 1]), np.zeros(6))
    initial_command = FloatingRobotCommand(
        None, FloatingRobotInputs(np.empty(0), np.array([[0.0, 0, 16]]), np.zeros((1, 3)))
    )
    worlds = tuple(
        ArticulationWorld(
            i,
            asset,
            state,
            kp=np.empty(0),
            kd=np.empty(0),
            effort_limit=np.empty(0),
            initial_control=initial_command,
        )
        for i in range(8)
    )
    with compile_scene(SceneDeclaration(worlds), mode, cache_root=cache(tmp_path)) as scene:
        with (
            scene.prepare_control_write(fields=("com_force",)) as control,
            scene.prepare_body_read(fields=("position", "com_velocity")) as body,
            scene.prepare_reset() as reset,
        ):
            np.testing.assert_array_equal(read(control), np.tile([0, 0, 16], (8, 1)))
            initial = read(body)
            values = np.tile(np.array([2, 0, 16], np.float32), (8, 1))
            with monkeypatch.context() as patch:
                patch.setattr(scene, "snapshot", lambda *a, **k: pytest.fail("hot snapshot"))
                patch.setattr(scene, "set_controls", lambda *a, **k: pytest.fail("hot packets"))
                control.write(values)
                assert all(scene.step(H, require_all=True).values())
                current = read(body)
                reset.reset(np.arange(8) % 2 == 0)
            # One newton acceleration: 2 N / 2 kg. Original FP32 acceptance unchanged.
            np.testing.assert_allclose(current[:, 3], H, atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(current[:, 5], 0, atol=1e-6, rtol=1e-5)
            actual = read(body)
            np.testing.assert_array_equal(actual[::2], initial[::2])
            np.testing.assert_array_equal(actual[1::2], current[1::2])
            actual_control = read(control)
            np.testing.assert_array_equal(actual_control[::2], np.tile([0, 0, 16], (4, 1)))
            np.testing.assert_array_equal(actual_control[1::2], values[1::2])


def test_mixed_control_allocation_failure_precedes_any_publication(tmp_path, monkeypatch):
    from dataclasses import replace

    class NoCopy(np.ndarray):
        def copy(self, *args, **kwargs):
            raise MemoryError("injected control preparation")

    spec = specification(True)
    with ph.TyxWorldBatch(
        [(0, ph.FreeSphere(1, 0.125)), (1, spec)], "interpret", cache_root=cache(tmp_path)
    ) as batch:
        storage = batch._floating_trees
        before = batch.snapshot()
        dirty = batch._dirty, storage._dirty
        command = {0: (1, 0, 0), 1: spec.inputs(np.full((1, 3), 0.25))}
        with monkeypatch.context() as patch:
            patch.setattr(storage, "_held_effort", storage._held_effort.view(NoCopy))
            with pytest.raises(MemoryError, match="injected control preparation"):
                batch.set_controls(command)
        assert batch.snapshot() == before
        assert (batch._dirty, storage._dirty) == dirty
        validate = storage.validate_controls

        def prepared(*args):
            stage = validate(*args)
            # Publishing must consume already-prepared owners without copying again.
            return replace(
                stage,
                drive=stage.drive.view(NoCopy),
                effort=stage.effort.view(NoCopy),
                loads=stage.loads.view(NoCopy),
            )

        with monkeypatch.context() as patch:
            patch.setattr(storage, "validate_controls", prepared)
            batch.set_controls(command)
        with batch.prepare_control_write([1], fields=("feedforward",)) as plan:
            np.testing.assert_array_equal(read(plan), [[0.25]])
            plan.write(np.array([[0.5]], np.float32))
            np.testing.assert_array_equal(read(plan), [[0.5]])


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_mixed_reset_preserves_nonzero_delay_history(mode, tmp_path):
    from dataclasses import replace

    from test_resident_field_read import world

    fixed, fixed_command = world("tree")
    floating = replace(specification(True), delay_steps=1)
    command = floating.inputs(np.array([[0.125, 0.25, 0.5]]))
    with ph.TyxWorldBatch(
        [(3, fixed), (7, floating)],
        mode,
        cache_root=cache(tmp_path),
        initial_controls={3: fixed_command(0), 7: command},
    ) as batch:
        initial = batch.snapshot()
        with batch.prepare_reset() as reset:
            for _ in range(3):
                assert all(batch.step(H, require_all=True).values())
            before = batch.snapshot()
            assert np.any(np.asarray(before[7].history))
            fixed_status = batch._trees.status_codes[3]
            reset.reset(np.array([False, True]))
            actual = batch.snapshot()
            assert actual[3] == before[3] and actual[7] == initial[7]
            assert batch._trees.status_codes[3] == fixed_status
            reset.reset(np.array([True, False]))
            assert batch.snapshot() == initial
