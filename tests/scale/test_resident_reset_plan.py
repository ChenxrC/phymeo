"""Prepared reset preserves initial/control history and unselected accepted observations."""

import json

import numpy as np
import pytest
from test_resident_field_read import world
from test_resident_publication import cache

from phymeo.backends.tyx_world_batch import FreeSphere, SpherePlaneWorld, TyxWorldBatch


def assert_geometry(actual, expected):
    assert actual.tick == expected.tick
    for name in ("position", "velocity", "link_position_m", "link_rotation", "com_position_m"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
@pytest.mark.parametrize("count", [1, 8, 64])
@pytest.mark.parametrize("kind", ["sphere", "plane", "slider", "rotor", "tree"])
def test_initial_and_masked_reset_preserve_all_accepted_data(
    kind, count, mode, tmp_path, monkeypatch
):
    if kind == "plane":
        spec = SpherePlaneWorld(1, 0.125, position=(0, 0, 0.125), gravity=(0, 0, -8))

        def command(key):
            return (0, 0, -key / 1024)
    else:
        spec, command = world(kind)
    keys = tuple(7 + 3 * i for i in range(count))
    with TyxWorldBatch([(key, spec) for key in keys], mode, cache_root=cache(tmp_path)) as batch:
        initial = batch.snapshot()
        initial_geometry = batch.tree_kinematics() if kind == "tree" else None
        initial_summary = batch.summary()
        batch.set_controls({key: command(i) for i, key in enumerate(keys)})
        for _ in range(2):
            batch.step(1 / 1024)
        current = batch.snapshot()
        geometry = batch.tree_kinematics() if kind == "tree" else None
        summary = batch.summary()
        with batch.prepare_reset() as plan:
            assert plan.layout["world_keys"] == keys
            assert batch.snapshot() == current
            assert batch.summary() == summary
            if geometry:
                for key, value in batch.tree_kinematics().items():
                    assert_geometry(value, geometry[key])
            for parity in (0, 1):
                mask = np.arange(count) % 2 == parity
                before = batch.stats()
                previous = plan.stats()
                compiled = tuple(batch._runtime.kernel_cache)
                with monkeypatch.context() as patch:
                    patch.setattr(batch, "snapshot", lambda *a: pytest.fail("hot snapshot"))
                    patch.setattr(batch, "restore", lambda *a: pytest.fail("hot restore parsing"))
                    patch.setattr(batch, "_read", lambda *a: pytest.fail("hot HOST download"))
                    if kind == "tree":
                        patch.setattr(batch._trees, "_geometry", lambda *a: pytest.fail("hot FK"))
                        patch.setattr(batch._trees, "_energy", lambda *a: pytest.fail("hot energy"))
                    epoch = plan.reset(mask)
                assert tuple(batch._runtime.kernel_cache) == compiled
                assert epoch == before["state_epoch"] + int(mask.any())
                after = batch.stats()
                for counter in (
                    "device_allocations",
                    "tree_fk_kernel_launches",
                    "tree_energy_kernel_launches",
                ):
                    assert after.get(counter, 0) == before.get(counter, 0)
                assert after["synchronizations"] - before["synchronizations"] == int(mask.any())
                assert plan.stats()["uploaded_bytes"] - previous["uploaded_bytes"] == (
                    4 * count if mode == "cuda" and mask.any() else 0
                )
                actual = batch.snapshot()
                actual_summary = batch.summary()
                actual_geometry = batch.tree_kinematics() if kind == "tree" else None
                for row, key in enumerate(keys):
                    expected = initial[key] if mask[row] else current[key]
                    assert actual[key] == expected
                    assert actual_summary[key] == (
                        initial_summary[key] if mask[row] else summary[key]
                    )
                    if actual_geometry:
                        assert_geometry(
                            actual_geometry[key],
                            initial_geometry[key] if mask[row] else geometry[key],
                        )
                current, summary, geometry = actual, actual_summary, actual_geometry
            assert batch.snapshot() == initial
            # Interleave old reset/control APIs and exercise the prepared other bank.
            batch.set_controls({key: command(0) for key in keys})
            batch.step(1 / 1024)
            batch.reset([keys[0]])
            plan.reset()
            assert batch.snapshot() == initial
            print(
                "RESIDENT_RESET "
                + json.dumps(dict(mode=mode, kind=kind, worlds=count, **plan.stats()))
            )


@pytest.mark.parametrize("failure", ["kernel", "sync", "control"])
def test_failed_reset_keeps_accepted_bank_and_retry_is_initial(failure, tmp_path, monkeypatch):
    spec, command = world("tree")
    with TyxWorldBatch(
        [(0, FreeSphere(1, 0.125)), (1, spec)], "cuda", cache_root=cache(tmp_path)
    ) as batch:
        initial = batch.snapshot()
        batch.set_controls({0: (1, 0, 0), 1: command(0)})
        batch.step(1 / 1024)
        before = batch.snapshot()
        with batch.prepare_reset() as plan:
            epoch = batch.stats()["state_epoch"]
            launch, sync, copy = batch._launch, batch._sync, np.copyto
            raised = False

            def bad_launch(kernel, *args):
                if kernel in plan._tree_kernels:
                    raise RuntimeError("injected reset kernel")
                return launch(kernel, *args)

            def bad_sync():
                nonlocal raised
                if not raised:
                    raised = True
                    raise RuntimeError("injected reset completion")
                return sync()

            def bad_copy(dst, src, **kwargs):
                nonlocal raised
                if dst is batch._trees._held and not raised:
                    raised = True
                    dst.reshape(-1)[0] = 99
                    raise RuntimeError("injected reset control")
                return copy(dst, src, **kwargs)

            with monkeypatch.context() as patch:
                if failure == "kernel":
                    patch.setattr(batch, "_launch", bad_launch)
                elif failure == "sync":
                    patch.setattr(batch, "_sync", bad_sync)
                else:
                    patch.setattr(np, "copyto", bad_copy)
                with pytest.raises(RuntimeError, match="injected reset"):
                    plan.reset()
            assert batch.stats()["state_epoch"] == epoch
            assert batch.snapshot() == before
            plan.reset()
            assert batch.snapshot() == initial


@pytest.mark.parametrize("bad", ["dtype", "shape", "stride", "alias"])
def test_mask_precheck_is_atomic(bad, tmp_path):
    with TyxWorldBatch(
        [(i, FreeSphere(1, 0.125)) for i in range(2)], "interpret", cache_root=cache(tmp_path)
    ) as batch:
        batch.step(1 / 1024)
        with batch.prepare_reset() as plan:
            before = batch.snapshot()
            values = np.ones(2, bool)
            if bad == "dtype":
                values = values.astype(np.int32)
            elif bad == "shape":
                values = values[:1]
            elif bad == "stride":
                values = np.ones(4, bool)[::2]
            else:
                values = plan._mask
            with pytest.raises((TypeError, ValueError)):
                plan.reset(values)
            assert batch.snapshot() == before
            assert plan.stats()["resets"] == 0


def test_empty_and_lifetime(tmp_path):
    with TyxWorldBatch([], "interpret", cache_root=cache(tmp_path)) as batch:
        plan = batch.prepare_reset()
        assert plan.reset() == batch.stats()["state_epoch"]
        assert plan.stats()["device_allocations"] == 0
    assert plan.stats()["closed"]
    with pytest.raises(RuntimeError, match="closed"):
        plan.reset()


def test_cuda_close_failure_retains_plan_and_retries(tmp_path, monkeypatch):
    with TyxWorldBatch([(0, FreeSphere(1, 0.125))], "cuda", cache_root=cache(tmp_path)) as batch:
        plan = batch.prepare_reset()
        failing = plan._selected_ticks
        free = type(failing).free

        def fail_once(array):
            if array is failing:
                raise RuntimeError("injected reset free")
            return free(array)

        with monkeypatch.context() as patch:
            patch.setattr(type(failing), "free", fail_once)
            with pytest.raises(RuntimeError, match="injected reset free"):
                plan.close()
            assert plan in batch._reset_plans
            assert plan.stats()["device_owned_bytes"] > 0
            with pytest.raises(RuntimeError, match="closing"):
                plan.reset()
        plan.close()
        assert plan.stats()["device_owned_bytes"] == 0
        assert batch.step(1 / 1024, require_all=True) == {0: True}


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_reset_sources_cannot_be_overwritten_by_other_plans(mode, tmp_path):
    with TyxWorldBatch([(0, FreeSphere(1, 0.125))], mode, cache_root=cache(tmp_path)) as batch:
        with (
            batch.prepare_reset() as reset,
            batch.prepare_control_write(fields=("com_force",)) as controls,
            batch.prepare_field_read(fields=("position",)) as fields,
        ):
            target = reset._controls[0][1].reshape(-1)[:3].reshape(1, 3)
            for reader in (controls, fields):
                with pytest.raises(ValueError, match="alias"):
                    reader.read_into(target)
            if mode == "interpret":
                with pytest.raises(ValueError, match="alias"):
                    fields.read_into(reset._initial[:, :3])


def assert_diagnostic(actual, expected):
    from dataclasses import fields

    if expected is None:
        assert actual is None
        return
    for field in fields(expected):
        np.testing.assert_array_equal(getattr(actual, field.name), getattr(expected, field.name))


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_two_plans_preserve_current_diagnostics_and_status(mode, tmp_path, monkeypatch):
    spec, command = world("tree")
    with TyxWorldBatch([(0, spec), (1, spec)], mode, cache_root=cache(tmp_path)) as batch:
        batch.set_controls({0: command(0), 1: command(0)})
        assert all(batch.step(1 / 1024).values())
        original = batch.tree_diagnostics()
        assert all(value is not None for value in original.values())
        with batch.prepare_reset() as first, batch.prepare_reset() as second:
            for key, value in batch.tree_diagnostics().items():
                assert_diagnostic(value, original[key])
            for i, plan in enumerate((first, second, first, second)):
                batch.step(1 / 1024)
                if i == 2:
                    batch.reset([0])
                before = batch.tree_diagnostics()
                codes = batch.stats()["tree_status_codes"]
                selected = i % 2
                mask = np.array([selected == 0, selected == 1], bool)
                plan.reset(mask)
                after = batch.tree_diagnostics()
                assert after[selected] is None
                assert_diagnostic(after[1 - selected], before[1 - selected])
                assert batch.stats()["tree_status_codes"][1 - selected] == codes[1 - selected]
                accepted = batch.snapshot()
                codes = batch.stats()["tree_status_codes"]
                with monkeypatch.context() as patch:
                    patch.setattr(
                        plan,
                        "_launch",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("reset failed")),
                    )
                    with pytest.raises(RuntimeError, match="reset failed"):
                        plan.reset(~mask)
                assert batch.snapshot() == accepted
                assert batch.stats()["tree_status_codes"] == codes
                for key, value in batch.tree_diagnostics().items():
                    assert_diagnostic(value, after[key])


def test_control_restore_and_rollback_failure_locks_batch_but_allows_close(tmp_path, monkeypatch):
    with TyxWorldBatch(
        [(0, FreeSphere(1, 0.125))], "interpret", cache_root=cache(tmp_path)
    ) as batch:
        batch.set_controls({0: (1, 0, 0)})
        plan = batch.prepare_reset()
        copy = np.copyto

        def fail_owner(dst, src, **kwargs):
            if dst is batch._held:
                raise RuntimeError("control rollback unavailable")
            return copy(dst, src, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(np, "copyto", fail_owner)
            with pytest.raises(RuntimeError, match="control rollback unavailable"):
                plan.reset()
        with pytest.raises(RuntimeError, match="rollback failed"):
            batch.step(1 / 1024)
        with pytest.raises(RuntimeError, match="rollback failed"):
            plan.reset()
        batch.close()
        assert plan.stats()["closed"]


def test_failed_prepare_retains_cleanup_owner(tmp_path, monkeypatch):
    from test_resident_cleanup import fail_free

    from phymeo.backends._resident_cleanup import ResidentCleanupError
    from phymeo.backends._resident_reset_plan import ResidentResetPlan

    with TyxWorldBatch([(0, FreeSphere(1, 0.125))], "cuda", cache_root=cache(tmp_path)) as batch:
        allocate = ResidentResetPlan._allocate
        allocated, switches = [], []

        def failing_allocate(plan, initial):
            if allocated:
                raise MemoryError("reset allocation failure")
            array = allocate(plan, initial)
            allocated.append(array)
            switches.append(fail_free(monkeypatch, array)[0])
            return array

        try:
            with monkeypatch.context() as patch:
                patch.setattr(ResidentResetPlan, "_allocate", failing_allocate)
                with pytest.raises(ResidentCleanupError) as caught:
                    batch.prepare_reset()
            owner = caught.value.cleanup_owner
            assert owner in batch._reset_plans and not owner.stats()["closed"]
            assert owner.stats()["device_owned_bytes"] == allocated[0].nbytes
            switches[0][0] = False
            owner.close()
            assert owner not in batch._reset_plans
            assert batch.step(1 / 1024, require_all=True) == {0: True}
        finally:
            for switch in switches:
                switch[0] = False


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_nonzero_initial_pose_velocity_and_tree_history_are_restored(mode, tmp_path):
    from test_resident_tree_step import _action, _program, _rr_model

    from phymeo.backends.tyx_world_batch import FixedRobotTree

    program, initial = _program(_rr_model(), q=[1 / 8, -1 / 16], v=[1 / 32, -1 / 64], delay=1)
    with program.stepper() as native:
        result = native.step(initial, _action(program, ff=[1 / 32, 1 / 64]), 1 / 1024)
        assert result.report.accepted
    tree = FixedRobotTree(program, result.state)
    sphere = FreeSphere(1, 0.125, position=(2, 3, 4), velocity=(0.5, -0.25, 1))
    with TyxWorldBatch([(0, sphere), (1, tree)], mode, cache_root=cache(tmp_path)) as batch:
        expected = batch.snapshot()
        batch.step(1 / 1024)
        with batch.prepare_reset() as plan:
            plan.reset()
            assert batch.snapshot() == expected
