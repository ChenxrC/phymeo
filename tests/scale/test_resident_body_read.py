"""Endpoint body fields use accepted FK/qdot, never entry-time diagnostics."""

import json

import numpy as np
import pytest
from test_resident_publication import cache
from test_resident_tree_step import ATOL, RTOL, _program, _rr_model

import phymeo as ph
from phymeo.backends.tyx_world_batch import FixedRobotTree, FreeSphere, TyxWorldBatch

FIELDS = (
    "position",
    "com_position",
    "rotation",
    "linear_velocity",
    "com_velocity",
    "angular_velocity",
)


def model():
    return ph.RobotModel(
        np.array([-1, 0, -1]),
        ("revolute", "prismatic", "revolute"),
        np.array([[2.0, -1, 0.5], [1.0, 0, 0.2], [2.0, 0, 0.5]]),
        np.array([[0.0, 0, 0, 1], [2**-0.5, 0, 0, 2**-0.5], [0.0, 0, 0, 1]]),
        np.array([[0.0, 0, 1], [0.0, 0, 1], [1.0, 0, 0]]),
        np.ones(3),
        np.array([[0.5, 0.1, 0], [0.2, 0, 0.3], [0.3, 0.2, 0.1]]),
        np.full((3, 3), 0.2),
        np.zeros(3),
    )


def geometry(m, q):
    """Independent FP64 homogeneous transforms, not PHYMEO FK helpers."""
    transforms = []
    for j, parent in enumerate(m.parents):
        x, y, z, w = m.joint_orientation[j]
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        base = np.eye(4)
        base[:3, :3], base[:3, 3] = rotation, m.joint_position[j]
        joint = np.eye(4)
        axis = m.joint_axis[j]
        if m.joint_types[j] == "prismatic":
            joint[:3, 3] = axis * q[j]
        else:
            x, y, z = axis
            cross = np.array([[0.0, -z, y], [z, 0, -x], [-y, x, 0]])
            joint[:3, :3] += np.sin(q[j]) * cross + (1 - np.cos(q[j])) * (cross @ cross)
        transforms.append((np.eye(4) if parent < 0 else transforms[parent]) @ base @ joint)
    transforms = np.asarray(transforms)
    rotations, positions = transforms[:, :3, :3], transforms[:, :3, 3]
    com = positions + np.einsum("bij,bj->bi", rotations, m.com)
    return positions, com, rotations


def expected_tree(m, q, v):
    q, v = np.asarray(q, float), np.asarray(v, float)
    pos, com, rot = geometry(m, q)
    eps = 1e-5
    plus, minus = geometry(m, q + eps * v), geometry(m, q - eps * v)
    linear = (plus[0] - minus[0]) / (2 * eps)
    com_v = (plus[1] - minus[1]) / (2 * eps)
    omega_matrix = np.einsum("bij,bkj->bik", (plus[2] - minus[2]) / (2 * eps), rot)
    omega = omega_matrix[:, [2, 0, 1], [1, 2, 0]]
    return np.concatenate(
        [pos.ravel(), com.ravel(), rot.ravel(), linear.ravel(), com_v.ravel(), omega.ravel()]
    )


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
@pytest.mark.parametrize("count", [1, 8, 64])
@pytest.mark.parametrize("kind", ["sphere", "tree"])
def test_endpoint_body_fields_reset_and_cost(kind, count, mode, tmp_path, monkeypatch):
    m = model()
    program, initial = _program(m, q=[0.3, 0.2, -0.4], v=[0.4, -0.2, 0.3])
    spec = (
        FixedRobotTree(program, initial)
        if kind == "tree"
        else FreeSphere(
            2, 0.125, position=(2.0, -1, 0.5), velocity=(0.4, -0.2, 0.3), gravity=(0, 0, -8)
        )
    )
    with TyxWorldBatch(
        [(k, spec) for k in range(count)], mode, cache_root=cache(tmp_path)
    ) as batch:
        selected = tuple(reversed(range(0, count, 2)))
        with (
            batch.prepare_body_read(selected, fields=FIELDS) as plan,
            batch.prepare_reset() as reset,
        ):
            values = np.empty(plan.layout["shape"], np.float32)
            ticks = np.empty(len(selected), np.int32)
            plan.read_into(values, ticks=ticks)
            initial_values = values.copy()
            for _ in range(2):
                assert all(batch.step(1 / 1024, require_all=True).values())
                snapshots = batch.snapshot()
                expected = []
                for key in selected:
                    state = snapshots[key]
                    if kind == "tree":
                        expected.append(expected_tree(m, state.position, state.velocity))
                    else:
                        expected.append(
                            np.concatenate(
                                [
                                    state.position,
                                    state.position,
                                    np.eye(3).ravel(),
                                    state.velocity,
                                    state.velocity,
                                    np.zeros(3),
                                ]
                            )
                        )
                before, counters = batch.stats(), plan.stats()
                compiled = tuple(batch._runtime.kernel_cache)
                with monkeypatch.context() as patch:
                    patch.setattr(
                        batch, "snapshot", lambda *a, **k: pytest.fail("snapshot hot read")
                    )
                    patch.setattr(batch, "_read", lambda *a, **k: pytest.fail("HOST intermediate"))
                    plan.read_into(values, ticks=ticks)
                np.testing.assert_allclose(values, expected, atol=ATOL, rtol=RTOL)
                np.testing.assert_array_equal(ticks, [snapshots[k].tick for k in selected])
                after = batch.stats()
                assert after["device_allocations"] == before["device_allocations"]
                assert after["synchronizations"] - before["synchronizations"] == int(mode == "cuda")
                assert tuple(batch._runtime.kernel_cache) == compiled
                for counter in ("tree_fk_kernel_launches", "tree_dynamics_kernel_launches"):
                    assert after.get(counter, 0) == before.get(counter, 0)
                assert plan.stats()["transfers"] - counters["transfers"] == 2 * int(mode == "cuda")
            previous = values.copy()
            mask = np.zeros(count, bool)
            mask[selected[0]] = True
            reset.reset(mask)
            plan.read_into(values, ticks=ticks)
            np.testing.assert_array_equal(values[0], initial_values[0])
            np.testing.assert_array_equal(values[1:], previous[1:])
            assert ticks[0] == 0
            before = plan.stats()
            plan.read_into(values)
            assert plan.stats()["transfers"] - before["transfers"] == int(mode == "cuda")
            print(
                "RESIDENT_BODY_READ "
                + json.dumps(
                    dict(
                        mode=mode,
                        worlds=count,
                        kind=kind,
                        selected=len(selected),
                        shape=values.shape,
                        stats=plan.stats(),
                        hot_device_allocations=0,
                        hot_fk=0,
                        syncs_per_read=int(mode == "cuda"),
                    )
                )
            )


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
@pytest.mark.parametrize("invalid", ["dtype", "shape", "stride", "readonly", "ticks", "alias"])
def test_body_caller_prevalidation(mode, invalid, tmp_path):
    with TyxWorldBatch([(0, FreeSphere(1, 0.125))], mode, cache_root=cache(tmp_path)) as batch:
        with batch.prepare_body_read(fields=("position",)) as plan:
            values, ticks = np.full((1, 3), -99, np.float32), np.full(1, -99, np.int32)
            if invalid == "dtype":
                values = values.astype(np.float64)
            elif invalid == "shape":
                values = np.full((1, 4), -99, np.float32)
            elif invalid == "stride":
                values = np.full((1, 6), -99, np.float32)[:, ::2]
            elif invalid == "readonly":
                values.flags.writeable = False
            elif invalid == "ticks":
                ticks = ticks.astype(np.int64)
            else:
                ticks = values.view(np.int32).reshape(-1)[:1]
            old, old_ticks = values.copy(), ticks.copy()
            with pytest.raises((ValueError, TypeError)):
                plan.read_into(values, ticks=ticks)
            np.testing.assert_array_equal(values, old)
            np.testing.assert_array_equal(ticks, old_ticks)


def test_body_layout_rejection_empty_stale_and_alias(tmp_path):
    from test_resident_step import slider

    from phymeo.backends.tyx_world_batch import PrismaticSlider

    program, initial = slider()
    with TyxWorldBatch(
        [(0, FreeSphere(1, 0.125)), (1, PrismaticSlider(program, initial))],
        "interpret",
        cache_root=cache(tmp_path),
    ) as batch:
        for fields in ((), ("unknown",), ("position", "position")):
            with pytest.raises(ValueError):
                batch.prepare_body_read([0], fields=fields)
        with pytest.raises(NotImplementedError):
            batch.prepare_body_read([1])
        with pytest.raises(ValueError):
            batch.prepare_body_read([0, 0])
        with batch.prepare_body_read([], fields=FIELDS) as empty:
            empty.read_into(np.empty((0, 0), np.float32))
            assert empty.stats()["device_allocations"] == 0
        plan = batch.prepare_body_read([0], fields=("position",))
        with pytest.raises(ValueError, match="alias"):
            plan.read_into(batch._held[:1, :3])
        batch.close()
        with pytest.raises(RuntimeError, match="closed"):
            plan.read_into(np.empty((1, 3), np.float32))


@pytest.mark.parametrize("failure", ["launch", "sync", "transfer", "close"])
def test_body_plan_failure_preserves_physics_and_retries(failure, tmp_path, monkeypatch):
    from test_resident_cleanup import fail_free

    program, initial = _program(_rr_model(), q=[0.2, -0.1], v=[0.1, 0.2])
    with TyxWorldBatch(
        [(0, FixedRobotTree(program, initial))], "cuda", cache_root=cache(tmp_path)
    ) as batch:
        plan = batch.prepare_body_read(fields=FIELDS)
        values = np.empty(plan.layout["shape"], np.float32)
        before = batch.snapshot()
        with monkeypatch.context() as patch:

            def fail(*a, **k):
                raise RuntimeError("injected body read failure")

            if failure == "close":
                enabled, _ = fail_free(patch, plan._output)
                with pytest.raises(RuntimeError):
                    plan.close()
                assert plan in batch._readback_plans and not batch._runtime.closed
                enabled[0] = False
            else:
                if failure == "launch":
                    patch.setattr(batch, "_launch", fail)
                elif failure == "sync":
                    patch.setattr(batch, "_sync", fail)
                else:
                    patch.setattr(type(plan._output), "copy_to_host", fail)
                with pytest.raises(RuntimeError, match="injected"):
                    plan.read_into(values)
        if failure == "sync":
            # The recovery completion also failed: the existing runtime must lock
            # further use until close/rebuild, even though reads never write physics.
            with pytest.raises(RuntimeError, match="cannot synchronize"):
                batch.snapshot()
            with pytest.raises(RuntimeError, match="cannot synchronize"):
                plan.read_into(values)
            with pytest.raises(RuntimeError, match="cannot synchronize"):
                batch.step(1 / 1024)
        else:
            assert batch.snapshot() == before
        if failure not in ("close", "sync"):
            plan.read_into(values)
        plan.close()
        assert plan.stats()["device_owned_bytes"] == 0


@pytest.mark.parametrize("failure", ["allocation", "warmup"])
def test_body_prepare_and_cleanup_failure_keeps_owner(failure, tmp_path, monkeypatch):
    from test_resident_cleanup import fail_free

    from phymeo.backends._resident_body_read import ResidentBodyReadPlan
    from phymeo.backends._resident_cleanup import ResidentCleanupError

    with TyxWorldBatch([(0, FreeSphere(1, 0.125))], "cuda", cache_root=cache(tmp_path)) as batch:
        allocate = ResidentBodyReadPlan._allocate
        switches, arrays = [], []

        def fail_allocate(owner, *args):
            if arrays and failure == "allocation":
                raise RuntimeError("injected allocation")
            array = allocate(owner, *args)
            arrays.append(array)
            if len(arrays) == 1:
                enabled, _ = fail_free(monkeypatch, array)
                switches.append(enabled)
            return array

        def fail_warmup(*args):
            raise RuntimeError("injected warmup")

        try:
            with monkeypatch.context() as patch:
                patch.setattr(ResidentBodyReadPlan, "_allocate", fail_allocate)
                if failure == "warmup":
                    patch.setattr(ResidentBodyReadPlan, "_launch_read", fail_warmup)
                with pytest.raises(ResidentCleanupError) as caught:
                    batch.prepare_body_read()
            owner = caught.value.cleanup_owner
            assert owner in batch._readback_plans
            assert owner.stats()["device_owned_bytes"] == arrays[0].nbytes
            switches[0][0] = False
            owner.close()
            assert owner not in batch._readback_plans
            assert batch.step(1 / 1024, require_all=True) == {0: True}
        finally:
            for enabled in switches:
                enabled[0] = False


def test_body_read_cannot_alias_accepted_geometry_or_unselected_control(tmp_path):
    from test_resident_floating_step import _p03_host

    from phymeo.backends.tyx_world_batch import FloatingRobotTree

    m = ph.RobotModel(
        np.array([-1]),
        ("revolute",),
        np.array([[1.0, 2, 3]]),
        np.array([[0.0, 0, 0, 1]]),
        np.array([[0.0, 0, 1]]),
        np.ones(1),
        np.array([[0.5, 0, 0]]),
        np.ones((1, 3)),
        np.zeros(3),
    )
    program, initial = _program(m)
    fm, fs = _p03_host()
    floating = FloatingRobotTree(
        fm, fs, kp=np.zeros(1), kd=np.zeros(1), effort_limit=np.array([8.0])
    )
    with TyxWorldBatch(
        [(0, FixedRobotTree(program, initial)), (1, floating)],
        "interpret",
        cache_root=cache(tmp_path),
    ) as batch:
        with batch.prepare_body_read([0], fields=("position",)) as plan:
            foreign = batch._floating_trees
            foreign._held_drive = foreign._held_drive.copy()
            for array in (
                batch._trees.anchor,
                batch._trees.fk[batch._active][0, :, :3],
                foreign._held_drive.reshape(1, 3),
            ):
                previous = array.copy()
                with pytest.raises(ValueError, match="alias"):
                    plan.read_into(array)
                np.testing.assert_array_equal(array, previous)
        with pytest.raises(NotImplementedError):
            batch.prepare_body_read([1])
