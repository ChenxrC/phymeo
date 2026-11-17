"""Prepared floating root/child observations through public APIs, no HOST snapshot hot path."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.core.rigid import RigidBodyModel
from phymeo.core.robot import RobotModel
from phymeo.dynamics.floating_robot import FloatingRobotModel, FloatingRobotState

H, ATOL, RTOL = 1 / 1024, 1e-6, 1e-5
FIELDS = (
    "position",
    "com_position",
    "rotation",
    "linear_velocity",
    "com_velocity",
    "angular_velocity",
)
R = np.array([[0.0, -1, 0], [1.0, 0, 0], [0, 0, 1.0]])


def specification(child, *, eccentric=False):
    children = None
    if child:
        children = RobotModel(
            np.array([-1]),
            ("prismatic",),
            np.zeros((1, 3)),
            np.array([[0.0, 0, 0, 1]]),
            np.array([[1.0, 0, 0]]),
            np.array([1.0]),
            np.zeros((1, 3)),
            np.full((1, 3), 0.25),
            np.array([0.0, 0, -8]),
        )
    model = FloatingRobotModel(
        RigidBodyModel(np.array([0.5]), np.full((1, 3), 0.5), np.array([0.0, 0, -8])),
        np.array([0.25, -0.125, 0.0625]) if eccentric else np.zeros(3),
        children,
    )
    q = np.r_[[2.0, -1, 0.5, 0, 0, 2**-0.5, 2**-0.5], [0.125] if child else []]
    v = np.zeros(6 + int(child))
    if eccentric:
        v[:6] = [0.5, -0.25, 0.125, 0, 0, 0.5]
    state = FloatingRobotState(q, v)
    return ph.FloatingRobotTree(
        model,
        state,
        kp=np.zeros(int(child)),
        kd=np.zeros(int(child)),
        effort_limit=np.full(int(child), 8.0),
    )


def cache(tmp_path):
    return Path(os.environ.get("PHYMEO_RESIDENT_CACHE", tmp_path))


def split(plan, values):
    return {
        name: values[:, meta["offset"] : meta["offset"] + meta["width"]].reshape(
            len(values), *meta["shape"]
        )
        for name, meta in plan.layout["fields"].items()
    }


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
@pytest.mark.parametrize("count", [1, 8, 64])
@pytest.mark.parametrize("child", [False, True])
def test_floating_prepared_fields_body_recoil_and_reset(mode, count, child, tmp_path, monkeypatch):
    spec = specification(child)
    with ph.TyxWorldBatch(
        [(i, spec) for i in range(count)], mode, cache_root=cache(tmp_path)
    ) as batch:
        keys = tuple(reversed(range(count)))
        with (
            batch.prepare_body_read(keys, fields=FIELDS) as body,
            batch.prepare_field_read(
                keys, fields=("position", "velocity", "applied_effort")
            ) as joints,
        ):
            assert body.layout["family"] == "floating_tree"
            assert body.layout["body_order"] == "root_then_model_links"
            values = np.empty(body.layout["shape"], np.float32)
            joint_values = np.empty(joints.layout["shape"], np.float32)
            ticks = np.empty(count, np.int32)
            body.read_into(values, ticks=ticks)
            initial = values.copy()
            joints.read_into(joint_values)
            initial_joints = joint_values.copy()
            if child:
                packet = spec.inputs(np.array([[0.125, 0, 3.0]]))
                batch.set_controls({i: packet for i in range(count)})
            before = batch.stats()
            for _ in range(2):
                assert all(batch.step(H, require_all=True).values())
            # Do not let prepared observations silently use high-level snapshots/FK.
            with monkeypatch.context() as patch:
                patch.setattr(batch, "snapshot", lambda *a, **k: pytest.fail("snapshot hot path"))
                patch.setattr(
                    batch, "tree_kinematics", lambda *a, **k: pytest.fail("kinematics readback")
                )
                body.read_into(values, ticks=ticks)
                joints.read_into(joint_values)
            out = split(body, values)
            np.testing.assert_array_equal(ticks, 2)
            expected_v = np.array([0.0, -3 * H if child else 0.0, -16 * H])
            expected_p = np.array([2.0, -1 - (4.5 * H**2 if child else 0.0), 0.5 - 24 * H**2])
            np.testing.assert_allclose(
                out["position"][:, 0], np.tile(expected_p, (count, 1)), atol=ATOL, rtol=RTOL
            )
            np.testing.assert_allclose(
                out["com_velocity"][:, 0], np.tile(expected_v, (count, 1)), atol=ATOL, rtol=RTOL
            )
            np.testing.assert_allclose(
                out["rotation"].reshape(count, 1 + int(child), 3, 3),
                np.broadcast_to(R, (count, 1 + int(child), 3, 3)),
                atol=ATOL,
                rtol=RTOL,
            )
            if child:
                np.testing.assert_allclose(
                    joint_values,
                    np.tile([0.125 + 13.5 * H**2, 9 * H, 3.0], (count, 1)),
                    atol=ATOL,
                    rtol=RTOL,
                )
                np.testing.assert_allclose(
                    out["com_velocity"][:, 1, 1], 6 * H, atol=ATOL, rtol=RTOL
                )
                np.testing.assert_allclose(
                    out["position"][:, 1, 1], -0.875 + 9 * H**2, atol=ATOL, rtol=RTOL
                )
            else:
                assert joint_values.shape == (count, 0)
            after = batch.stats()
            assert after["device_allocations"] == before["device_allocations"]
            # Legacy public masked reset is qualified here; prepared reset remains separate work.
            previous, previous_joint = values.copy(), joint_values.copy()
            batch.reset(list(range(0, count, 2)))
            body.read_into(values, ticks=ticks)
            joints.read_into(joint_values)
            mask = np.array([k % 2 == 0 for k in keys])
            np.testing.assert_array_equal(values[mask], initial[mask])
            np.testing.assert_array_equal(values[~mask], previous[~mask])
            np.testing.assert_array_equal(joint_values[mask], initial_joints[mask])
            np.testing.assert_array_equal(joint_values[~mask], previous_joint[~mask])
            np.testing.assert_array_equal(ticks, np.where(mask, 0, 2))
            print("FLOATING_READ_COST", mode, count, child, json.dumps(body.stats()))


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_eccentric_root_origin_com_velocity_and_validation(mode, tmp_path):
    spec = specification(False, eccentric=True)
    with ph.TyxWorldBatch([(5, spec)], mode, cache_root=cache(tmp_path)) as batch:
        with batch.prepare_body_read(fields=FIELDS) as body:
            values = np.empty(body.layout["shape"], np.float32)
            body.read_into(values)
            out = split(body, values)
            offset = R @ np.array([0.25, -0.125, 0.0625])
            v, w = np.array([0.5, -0.25, 0.125]), np.array([0.0, 0, 0.5])
            np.testing.assert_allclose(
                out["com_position"][0, 0], [2, -1, 0.5] + offset, atol=ATOL, rtol=RTOL
            )
            np.testing.assert_allclose(
                out["linear_velocity"][0, 0], v - np.cross(w, offset), atol=ATOL, rtol=RTOL
            )
            np.testing.assert_allclose(out["com_velocity"][0, 0], v, atol=ATOL, rtol=RTOL)
            np.testing.assert_allclose(out["angular_velocity"][0, 0], w, atol=ATOL, rtol=RTOL)
            assert batch.step(H, require_all=True) == {5: True}
            accepted = batch.snapshot()[5]
            q, v = np.asarray(accepted.position), np.asarray(accepted.velocity)
            x, y, z, scalar = q[3:7]
            rotation = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * scalar), 2 * (x * z + y * scalar)],
                    [2 * (x * y + z * scalar), 1 - 2 * (x * x + z * z), 2 * (y * z - x * scalar)],
                    [2 * (x * z - y * scalar), 2 * (y * z + x * scalar), 1 - 2 * (x * x + y * y)],
                ]
            )
            offset = rotation @ np.array([0.25, -0.125, 0.0625])
            body.read_into(values)
            out = split(body, values)
            np.testing.assert_allclose(out["position"][0, 0], q[:3], atol=ATOL, rtol=RTOL)
            np.testing.assert_allclose(
                out["com_position"][0, 0], q[:3] + offset, atol=ATOL, rtol=RTOL
            )
            np.testing.assert_allclose(
                out["rotation"][0, 0].reshape(3, 3), rotation, atol=ATOL, rtol=RTOL
            )
            np.testing.assert_allclose(
                out["linear_velocity"][0, 0], v[:3] - np.cross(v[3:6], offset), atol=ATOL, rtol=RTOL
            )
            np.testing.assert_allclose(out["com_velocity"][0, 0], v[:3], atol=ATOL, rtol=RTOL)
            np.testing.assert_allclose(out["angular_velocity"][0, 0], v[3:6], atol=ATOL, rtol=RTOL)
            for bad in (
                np.full(values.shape, 99.0, np.float64),
                np.full((1, values.shape[1] - 1), 99.0, np.float32),
            ):
                with pytest.raises((TypeError, ValueError)):
                    body.read_into(bad)
                np.testing.assert_array_equal(bad, 99)
            with batch.prepare_body_read([], fields=FIELDS) as empty:
                empty.read_into(np.empty(empty.layout["shape"], np.float32))


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_compiled_floating_scene_public_read_boundary(mode, tmp_path, monkeypatch):
    from phymeo.builders.scene_compiler import ArticulationWorld, SceneDeclaration, compile_scene

    path = tmp_path / "root.urdf"
    path.write_text("""<robot name="free"><link name="root"><inertial><mass value="2"/>
      <inertia ixx=".5" iyy=".5" izz=".5" ixy="0" ixz="0" iyz="0"/></inertial>
      </link></robot>""")
    asset = ph.load_floating_urdf(path, gravity=(0, 0, -8))
    state = FloatingRobotState(np.array([2.0, -1, 0.5, 0, 0, 0, 1]), np.zeros(6))
    worlds = tuple(
        ArticulationWorld(i, asset, state, kp=np.empty(0), kd=np.empty(0), effort_limit=np.empty(0))
        for i in range(8)
    )
    scene = compile_scene(SceneDeclaration(worlds), mode, cache_root=cache(tmp_path))
    with scene:
        body = scene.prepare_body_read(fields=("position", "com_velocity"))
        values = np.empty(body.layout["shape"], np.float32)
        with monkeypatch.context() as patch:
            patch.setattr(scene, "snapshot", lambda *a, **k: pytest.fail("snapshot hot path"))
            assert all(scene.step(H, require_all=True).values())
            body.read_into(values)
        np.testing.assert_allclose(
            values, np.tile([2.0, -1, 0.5 - 8 * H**2, 0, 0, -8 * H], (8, 1)), atol=ATOL, rtol=RTOL
        )
        scene.reset([0])
        body.read_into(values)
        np.testing.assert_array_equal(values[0], [2, -1, 0.5, 0, 0, 0])
    with pytest.raises(RuntimeError, match="closed"):
        body.read_into(values)


@pytest.mark.parametrize("failure", ["launch", "transfer", "close"])
def test_floating_read_failure_keeps_physics_and_owner(failure, tmp_path, monkeypatch):
    from test_resident_cleanup import fail_free

    with ph.TyxWorldBatch([(0, specification(True))], "cuda", cache_root=cache(tmp_path)) as batch:
        plan = batch.prepare_body_read(fields=FIELDS)
        values = np.empty(plan.layout["shape"], np.float32)
        before = batch.snapshot()
        with monkeypatch.context() as patch:
            if failure == "close":
                enabled, _ = fail_free(patch, plan._output)
                with pytest.raises(RuntimeError):
                    plan.close()
                assert plan in batch._readback_plans
                assert plan._floating is batch._floating_trees
                enabled[0] = False
            else:

                def fail(*a, **k):
                    raise RuntimeError("floating read injected")

                if failure == "launch":
                    patch.setattr(batch, "_launch", fail)
                else:
                    patch.setattr(type(plan._output), "copy_to_host", fail)
                with pytest.raises(RuntimeError, match="floating read injected"):
                    plan.read_into(values)
        assert batch.snapshot() == before
        plan.close()
        assert plan._floating is None and plan._twists == ()
        assert plan.stats()["device_owned_bytes"] == 0
