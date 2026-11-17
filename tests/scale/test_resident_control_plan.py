"""Prepared HOST controls preserve native commands, histories and world isolation."""

import json

import numpy as np
import pytest
from test_resident_field_read import world
from test_resident_publication import cache

from phymeo.backends.tyx_world_batch import FreeSphere, TyxWorldBatch


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
@pytest.mark.parametrize("count", [1, 8, 64])
@pytest.mark.parametrize("kind", ["sphere", "slider", "rotor", "tree"])
def test_prepared_controls_match_native_packets_and_reset(kind, count, mode, tmp_path, monkeypatch):
    spec, command = world(kind)
    field = {"sphere": "com_force", "slider": "target_position"}.get(kind, "feedforward")
    worlds = [(i, spec) for i in range(count)]
    with (
        TyxWorldBatch(worlds, mode, cache_root=cache(tmp_path)) as batch,
        TyxWorldBatch(worlds, mode, cache_root=cache(tmp_path)) as reference,
    ):
        keys = tuple(reversed(range(0, count, 2)))
        with batch.prepare_control_write(keys, fields=(field,)) as plan:
            values = np.zeros(plan.layout["shape"], np.float32)
            for row, key in enumerate(keys):
                if kind == "sphere":
                    values[row] = command(key)
                elif kind == "slider":
                    values[row, 0] = key / 64
                elif kind == "rotor":
                    values[row, 0] = (key + 1) / 64
                else:
                    values[row] = ((key + 1) / 64, 1 / 32)
            reference.set_controls({key: command(key) for key in keys})
            before = batch.stats()
            with monkeypatch.context() as patch:
                patch.setattr(batch, "set_controls", lambda *a: pytest.fail("packet wrapping"))
                patch.setattr(batch, "snapshot", lambda *a: pytest.fail("full snapshot"))
                patch.setattr(batch, "_upload", lambda *a: pytest.fail("immediate upload"))
                patch.setattr(batch, "_sync", lambda *a: pytest.fail("immediate sync"))
                plan.write(values)
                read = np.full_like(values, -99)
                plan.read_into(read)
            np.testing.assert_array_equal(read, values)
            assert batch.stats()["state_epoch"] == before["state_epoch"]
            assert batch.snapshot() == reference.snapshot()
            for _ in range(3):
                # Same implementation and same input; no cross-solver numerical oracle.
                assert batch.step(1 / 1024) == reference.step(1 / 1024)
                assert batch.snapshot() == reference.snapshot()
            saved = batch.snapshot()
            batch.reset([keys[0]])
            reference.reset([keys[0]])
            restored = batch.snapshot()
            assert restored == reference.snapshot()
            for key in range(count):
                if key != keys[0]:
                    assert restored[key] == saved[key]
            # A reset/legacy update may replace the held owner. The plan stays current.
            batch.set_controls({key: command(key) for key in keys})
            plan.read_into(read)
            np.testing.assert_array_equal(read, values)
            plan.write(values)
            assert batch.stats()["device_allocations"] == before["device_allocations"]
            print(
                "RESIDENT_CONTROL "
                + json.dumps(
                    dict(mode=mode, worlds=count, kind=kind, selected=len(keys), **plan.stats())
                )
            )


@pytest.mark.parametrize("kind", ["slider", "rotor", "tree"])
def test_field_order_units_and_unwritten_columns(kind, tmp_path):
    import phymeo as ph

    spec, _ = world(kind)
    n = spec.program.model.dof_count
    force = np.full((n, 3), 1 / 8)
    torque = np.full((n, 3), 1 / 16)
    packet = spec.program.inputs(
        np.full(n, 1 / 4),
        np.full(n, 1 / 2),
        feedforward=np.full(n, 3 / 4),
        loads=ph.RobotInputs(np.full(n, 1.0), force, torque),
    )
    fields = ("external_effort", "target_velocity", "feedforward", "target_position")
    if kind != "slider":
        fields += ("com_torque", "com_force")
    with TyxWorldBatch([(2, spec), (8, spec)], "interpret", cache_root=cache(tmp_path)) as batch:
        batch.set_controls({2: packet, 8: packet})
        old = batch.snapshot()
        with batch.prepare_control_write([8], fields=fields) as plan:
            result = np.empty(plan.layout["shape"], np.float32)
            plan.read_into(result)
            expected = [1.125 if kind == "slider" else 1] * n
            expected += [0.5] * n + [0.75] * n + [0.25] * n
            if kind != "slider":
                expected += [1 / 16] * (3 * n) + [1 / 8] * (3 * n)
            np.testing.assert_array_equal(result, [expected])
            assert plan.layout["fields"]["external_effort"]["source"] == (
                "total_axial_load" if kind == "slider" else "held_command"
            )
            with pytest.raises(TypeError):
                plan.layout["fields"]["external_effort"]["offset"] = 100
            # Rewrite only velocity via a second plan; all other columns survive.
            with batch.prepare_control_write([8], fields=("target_velocity",)) as partial:
                partial.write(np.full((1, n), -0.25, np.float32))
            expected[n : 2 * n] = [-0.25] * n
            plan.read_into(result)
            np.testing.assert_array_equal(result, [expected])
            assert batch.snapshot([2])[2] == old[2]


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
@pytest.mark.parametrize("bad", ["dtype", "shape", "stride", "nan", "inf", "subnormal", "alias"])
def test_invalid_controls_are_atomic(mode, bad, tmp_path):
    with TyxWorldBatch(
        [(0, FreeSphere(1, 0.125)), (1, FreeSphere(1, 0.125))], mode, cache_root=cache(tmp_path)
    ) as batch:
        with batch.prepare_control_write([1, 0], fields=("com_force",)) as plan:
            initial = batch.snapshot()
            values = np.ones((2, 3), np.float32)
            if bad == "dtype":
                values = values.astype(np.float64)
            elif bad == "shape":
                values = values[:1]
            elif bad == "stride":
                values = np.ones((2, 6), np.float32)[:, ::2]
            elif bad == "nan":
                values[-1, -1] = np.nan
            elif bad == "inf":
                values[-1, -1] = np.inf
            elif bad == "subnormal":
                values[-1, -1] = np.nextafter(np.float32(0), np.float32(1))
            else:
                values = batch._held.reshape(-1)[:6].reshape(2, 3)
            with pytest.raises((ValueError, TypeError)):
                plan.write(values)
            assert batch.snapshot() == initial
            assert not batch._dirty
            assert plan.stats()["writes"] == 0


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_partial_write_failure_rollback_and_close(rollback_fails, tmp_path, monkeypatch):
    with TyxWorldBatch(
        [(0, FreeSphere(1, 0.125))], "interpret", cache_root=cache(tmp_path)
    ) as batch:
        plan = batch.prepare_control_write(fields=("com_force",))
        original = batch.snapshot()
        put = np.put
        calls = 0

        def failing_put(a, indices, values, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                put(a, indices[:1], values[:1], **kwargs)
                raise RuntimeError("partial scatter")
            if rollback_fails:
                raise RuntimeError("rollback failed")
            return put(a, indices, values, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(np, "put", failing_put)
            with pytest.raises(RuntimeError, match="partial scatter"):
                plan.write(np.ones((1, 3), np.float32))
        if rollback_fails:
            with pytest.raises(RuntimeError, match="rollback failed"):
                batch.step(1 / 1024)
            with pytest.raises(RuntimeError, match="rollback failed"):
                plan.read_into(np.empty((1, 3), np.float32))
        else:
            assert batch.snapshot() == original
            assert not batch._dirty
            assert plan.stats()["rollbacks"] == 1
            plan.write(np.ones((1, 3), np.float32))
        batch.close()
        assert plan.stats()["closed"] and plan.stats()["host_workspace_bytes"] == 0
        with pytest.raises(RuntimeError, match="closed"):
            plan.write(np.ones((1, 3), np.float32))


def test_selection_and_layout_rejection_before_registration(tmp_path):
    slider, _ = world("slider")
    with TyxWorldBatch(
        [(0, FreeSphere(1, 0.125)), (1, slider)], "interpret", cache_root=cache(tmp_path)
    ) as batch:
        for keys, fields in (
            ([], ("com_force",)),
            ([0, 0], ("com_force",)),
            ([5], ("com_force",)),
            ([0, 1], ("com_force",)),
            ([0], ()),
            ([0], ("com_force", "com_force")),
            ([1], ("com_torque",)),
        ):
            with pytest.raises((KeyError, ValueError)):
                batch.prepare_control_write(keys, fields=fields)
        assert batch.stats()["control_plan_live_count"] == 0


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
@pytest.mark.parametrize("kind", ["sphere", "tree"])
def test_equal_commands_keep_upload_clean_and_readonly_input_is_accepted(kind, mode, tmp_path):
    spec, _ = world(kind)
    field = "com_force" if kind == "sphere" else "feedforward"
    with TyxWorldBatch([(0, spec), (1, spec)], mode, cache_root=cache(tmp_path)) as batch:
        with batch.prepare_control_write([1], fields=(field,)) as plan:
            values = np.full(plan.layout["shape"], 1 / 32, np.float32)
            values.flags.writeable = False
            plan.write(values)
            batch.step(1 / 1024)
            before = batch.stats()
            plan.write(values)
            batch.step(1 / 1024)
            after = batch.stats()
            for key in before:
                if key.endswith("control_uploaded_bytes"):
                    assert after[key] == before[key]
            with pytest.raises(ValueError):
                plan.read_into(values)


def test_floating_controls_and_foreign_owner_alias(tmp_path):
    from test_resident_floating_step import _p03_host

    from phymeo.backends.tyx_world_batch import FloatingRobotTree

    model, state = _p03_host()
    floating = FloatingRobotTree(
        model, state, kp=np.zeros(1), kd=np.zeros(1), effort_limit=np.array([8.0])
    )
    with TyxWorldBatch(
        [(0, FreeSphere(1, 0.125)), (1, floating)], "interpret", cache_root=cache(tmp_path)
    ) as batch:
        with batch.prepare_control_write([1], fields=("feedforward",)) as plan:
            values = np.full(plan.layout["shape"], 0.25, np.float32)
            plan.write(values)
            result = np.empty_like(values)
            plan.read_into(result)
            np.testing.assert_array_equal(result, values)
        with pytest.raises(ValueError, match="homogeneous"):
            batch.prepare_control_write(fields=("com_force",))
        with batch.prepare_control_write([0], fields=("com_force",)) as plan:
            storage = batch._floating_trees
            storage._held_drive = storage._held_drive.copy()
            alias = storage._held_drive.reshape(1, 3)
            with pytest.raises(ValueError, match="alias"):
                plan.write(alias)
            with pytest.raises(ValueError, match="alias"):
                plan.read_into(alias)
