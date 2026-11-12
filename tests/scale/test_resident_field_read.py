"""Caller-buffer reads use accepted physical banks, not detached snapshots."""

import json

import numpy as np
import pytest
from test_resident_publication import cache

from phymeo.backends.tyx_world_batch import (
    FixedRobotTree,
    FreeSphere,
    PrismaticSlider,
    RevoluteRotor,
    TyxWorldBatch,
)


def world(kind):
    if kind == "sphere":
        return FreeSphere(2, 0.125, gravity=(0, 0, -8)), lambda w: (w + 1, 0, 0)
    if kind == "slider":
        from test_resident_step import slider

        program, initial = slider()
        return PrismaticSlider(program, initial), lambda w: program.inputs(np.array([w / 64]))
    if kind == "rotor":
        from test_resident_revolute_step import _action, _program

        program, initial = _program()
        return RevoluteRotor(program, initial), lambda w: _action(program, ff=(w + 1) / 64)
    from test_resident_tree_step import _action, _program, _rr_model

    program, initial = _program(_rr_model(), delay=1)
    return FixedRobotTree(program, initial), lambda w: _action(program, ff=[(w + 1) / 64, 1 / 32])


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
@pytest.mark.parametrize("count", [1, 8, 64])
@pytest.mark.parametrize("kind", ["sphere", "slider", "rotor", "tree"])
def test_fields_match_accepted_banks_and_local_reset(kind, count, mode, tmp_path, monkeypatch):
    spec, command = world(kind)
    fields = ("velocity", "applied_effort", "position")
    with TyxWorldBatch(
        [(i, spec) for i in range(count)], mode, cache_root=cache(tmp_path)
    ) as batch:
        keys = tuple(reversed(range(0, count, 2)))
        batch.set_controls({i: command(i) for i in range(count)})
        with batch.prepare_field_read(keys, fields=fields) as plan:
            layout = plan.layout
            assert layout["world_keys"] == keys and layout["contract_version"] == 1
            unit = "rad" if kind in ("rotor", "tree") else "m"
            assert set(layout["fields"]["position"]["units"]) == {unit}
            assert layout["fields"]["applied_effort"]["source"] == (
                "external_com_force" if kind == "sphere" else "actuator_drive"
            )
            with pytest.raises(TypeError):
                layout["fields"]["position"]["offset"] = 0
            shape = layout["shape"]
            guarded = np.full(np.prod(shape) + 4, -123, np.float32)
            values = guarded[2:-2].reshape(shape)
            tick_guard = np.full(len(keys) + 4, -123, np.int32)
            ticks = tick_guard[2:-2]
            rejected = []
            for advance in range(3):
                accepted_mask = batch.step(1 / 1024)
                rejected.append([key for key, accepted in accepted_mask.items() if not accepted])
                assert all(accepted_mask.values()) or kind == "tree"
                assert any(accepted_mask.values())
                expected = batch.snapshot()
                before, plan_before = batch.stats(), plan.stats()
                compiled = tuple(batch._runtime.kernel_cache)
                with monkeypatch.context() as patch:
                    patch.setattr(batch, "snapshot", lambda *a, **k: pytest.fail("snapshot read"))
                    patch.setattr(
                        batch, "_read", lambda *a, **k: pytest.fail("intermediate HOST copy")
                    )
                    epoch = plan.read_into(values, ticks=ticks)
                assert tuple(batch._runtime.kernel_cache) == compiled
                assert epoch == before["state_epoch"]
                reference = np.array(
                    [
                        [v for field in fields for v in getattr(expected[key], field)]
                        for key in keys
                    ],
                    np.float32,
                )
                np.testing.assert_array_equal(values, reference)
                np.testing.assert_array_equal(ticks, [expected[key].tick for key in keys])
                np.testing.assert_array_equal(guarded[[0, 1, -2, -1]], -123)
                np.testing.assert_array_equal(tick_guard[[0, 1, -2, -1]], -123)
                after = batch.stats()
                assert after["device_allocations"] == before["device_allocations"]
                assert after["synchronizations"] - before["synchronizations"] == int(mode == "cuda")
                for counter in ("tree_fk_kernel_launches", "tree_dynamics_kernel_launches"):
                    assert after.get(counter, 0) == before.get(counter, 0)
                assert plan.stats()["transfers"] - plan_before["transfers"] == (
                    2 if mode == "cuda" else 0
                )
                assert plan.stats()["downloaded_bytes"] - plan_before["downloaded_bytes"] == (
                    values.nbytes + ticks.nbytes if mode == "cuda" else 0
                )
            accepted = values.copy()
            batch.reset([keys[0]])
            plan.read_into(values, ticks=ticks)
            assert ticks[0] == 0
            np.testing.assert_array_equal(values[1:], accepted[1:])
            before = plan.stats()
            plan.read_into(values)
            assert plan.stats()["transfers"] - before["transfers"] == int(mode == "cuda")
            assert plan.stats()["downloaded_bytes"] - before["downloaded_bytes"] == (
                values.nbytes if mode == "cuda" else 0
            )
            print(
                "RESIDENT_FIELD_READ "
                + json.dumps(
                    dict(
                        mode=mode,
                        worlds=count,
                        kind=kind,
                        selected=len(keys),
                        shape=shape,
                        bytes_per_values_read=values.nbytes if mode == "cuda" else 0,
                        hot_device_allocations=0,
                        explicit_syncs_per_read=int(mode == "cuda"),
                        unpublished_tree_worlds=rejected,
                    )
                )
            )


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
@pytest.mark.parametrize("invalid", ["dtype", "shape", "stride", "readonly", "ticks", "alias"])
def test_prevalidation_leaves_all_caller_outputs_untouched(mode, invalid, tmp_path):
    with TyxWorldBatch([(0, FreeSphere(1, 0.125))], mode, cache_root=cache(tmp_path)) as batch:
        with batch.prepare_field_read() as plan:
            values = np.full((1, 6), -99, np.float32)
            ticks = np.full(1, -99, np.int32)
            if invalid == "dtype":
                values = values.astype(np.float64)
            elif invalid == "shape":
                values = np.full((1, 5), -99, np.float32)
            elif invalid == "stride":
                values = np.full((1, 12), -99, np.float32)[:, ::2]
            elif invalid == "readonly":
                values.flags.writeable = False
            elif invalid == "ticks":
                ticks = ticks.astype(np.int64)
            else:
                ticks = values.view(np.int32).reshape(-1)[:1]
            old_values, old_ticks = values.copy(), ticks.copy()
            before = batch.stats()
            with pytest.raises((ValueError, TypeError)):
                plan.read_into(values, ticks=ticks)
            np.testing.assert_array_equal(values, old_values)
            np.testing.assert_array_equal(ticks, old_ticks)
            assert batch.stats() == before


def test_cpu_output_cannot_overwrite_authoritative_storage(tmp_path):
    with TyxWorldBatch(
        [(0, FreeSphere(1, 0.125))], "interpret", cache_root=cache(tmp_path)
    ) as batch:
        with batch.prepare_field_read(fields=("position",)) as plan:
            output = batch._states[0][:, :3]
            before = batch.snapshot()
            with pytest.raises(ValueError, match="alias"):
                plan.read_into(output)
            assert batch.snapshot() == before


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_host_control_alias_rejected_after_reset_replaces_owner(mode, tmp_path):
    with TyxWorldBatch([(0, FreeSphere(1, 0.125))], mode, cache_root=cache(tmp_path)) as batch:
        with batch.prepare_field_read(fields=("position",)) as plan:
            for reset in (False, True):
                if reset:
                    batch.reset()
                control = batch._held[:, :3]
                previous = control.copy()
                with pytest.raises(ValueError, match="alias"):
                    plan.read_into(control)
                np.testing.assert_array_equal(control, previous)


def test_empty_and_rejected_layouts_allocate_no_device_storage():
    with TyxWorldBatch([], "interpret") as batch:
        with batch.prepare_field_read() as plan:
            assert plan.layout["shape"] == (0, 0)
            plan.read_into(np.empty((0, 0), np.float32), ticks=np.empty(0, np.int32))
            assert plan.stats()["device_allocations"] == 0
        for fields in ((), ("unknown",), ("position", "position")):
            with pytest.raises(ValueError):
                batch.prepare_field_read(fields=fields)


def test_mixed_layout_rejected_and_closed_plan_is_stale(tmp_path):
    from test_resident_step import slider

    program, initial = slider()
    with TyxWorldBatch(
        [(0, FreeSphere(1, 0.125)), (1, PrismaticSlider(program, initial))],
        "interpret",
        cache_root=cache(tmp_path),
    ) as batch:
        before = batch.stats()
        with pytest.raises(ValueError, match="homogeneous"):
            batch.prepare_field_read()
        with pytest.raises(ValueError):
            batch.prepare_field_read([0, 0])
        assert batch.stats() == before
        plan = batch.prepare_field_read([0])
        batch.close()
        with pytest.raises(RuntimeError, match="closed"):
            plan.read_into(np.empty(plan.layout["shape"], np.float32))


def test_cuda_read_plan_cleanup_retries_without_releasing_batch(tmp_path, monkeypatch):
    from test_resident_cleanup import fail_free

    batch = TyxWorldBatch([(0, FreeSphere(1, 0.125))], "cuda", cache_root=cache(tmp_path))
    plan = batch.prepare_field_read()
    enabled, _ = fail_free(monkeypatch, plan._output)
    try:
        with pytest.raises(RuntimeError, match="free failure"):
            plan.close()
        assert plan in batch._readback_plans and not batch._runtime.closed
        with pytest.raises(RuntimeError, match="closed"):
            plan.read_into(np.empty(plan.layout["shape"], np.float32))
        enabled[0] = False
        plan.close()
        assert plan.stats()["device_owned_bytes"] == 0
        assert batch.step(1 / 1024, require_all=True) == {0: True}
    finally:
        enabled[0] = False
        batch.close()


def test_transfer_failure_reports_partial_output_without_changing_physics(tmp_path, monkeypatch):
    with TyxWorldBatch([(0, FreeSphere(1, 0.125))], "cuda", cache_root=cache(tmp_path)) as batch:
        batch.step(1 / 1024, require_all=True)
        with batch.prepare_field_read() as plan:
            values = np.full(plan.layout["shape"], -99, np.float32)
            ticks = np.full(1, -99, np.int32)
            original = type(plan._ticks).copy_to_host

            def fail_ticks(array, target, *args, **kwargs):
                if array is plan._ticks:
                    raise RuntimeError("injected tick transfer failure")
                return original(array, target, *args, **kwargs)

            before = batch.snapshot()
            with monkeypatch.context() as patch:
                patch.setattr(type(plan._ticks), "copy_to_host", fail_ticks)
                with pytest.raises(RuntimeError, match="tick transfer failure"):
                    plan.read_into(values, ticks=ticks)
            assert plan.stats()["reads"] == 0 and plan.stats()["failed_reads"] == 1
            assert plan.stats()["transfers"] == 1
            assert np.any(values != -99) and ticks.tolist() == [-99]
            assert batch.snapshot() == before
            plan.read_into(values, ticks=ticks)
            assert ticks.tolist() == [1] and plan.stats()["reads"] == 1


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_unselected_floating_control_owner_is_protected(mode, tmp_path):
    from test_resident_floating_step import _p03_host

    from phymeo.backends.tyx_world_batch import FloatingRobotTree

    model, state = _p03_host()
    floating = FloatingRobotTree(
        model, state, kp=np.zeros(1), kd=np.zeros(1), effort_limit=np.array([8.0])
    )
    with TyxWorldBatch(
        [(0, FreeSphere(1, 0.125)), (1, floating)], mode, cache_root=cache(tmp_path)
    ) as batch:
        with pytest.raises(NotImplementedError, match="fixed trees"):
            batch.prepare_field_read([1])
        with batch.prepare_field_read([0], fields=("position",)) as plan:
            # Fault injection: an owner replaces its immutable HOST control with
            # equivalent writable storage. The accepted read still cannot alias it.
            storage = batch._floating_trees
            storage._held_drive = storage._held_drive.copy()
            output = storage._held_drive.reshape(1, 3)
            original = output.copy()
            with pytest.raises(ValueError, match="alias"):
                plan.read_into(output)
            np.testing.assert_array_equal(output, original)
            plan.read_into(np.empty((1, 3), np.float32))
