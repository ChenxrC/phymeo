"""Native MuJoCo checks for the single- and twin-pad manipulation fixtures."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


mj = pytest.importorskip("mujoco")


@pytest.fixture
def env_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return import_module("bimanual_musa_env")


@pytest.mark.parametrize(("task", "original_sha"), [
    ("carry", "be7d29bced8a3f15398bb9c9e77b23c47606de8c9ebadfa91ea737e2b7b55912"),
    ("push_t", "4f792f48a5d1271e2081f2206a838aa66c257d463b95e717ae4b8181b711820e"),
])
def test_twin_pad_preserves_registered_plant_and_changes_tool_inertia(env_module, task, original_sha):
    old = env_module.make_scene(task, seed=701)
    twin = env_module.make_scene(task, seed=701, tool_profile="twin-pad")
    assert old.xml_sha256 == original_sha
    assert twin.xml_sha256 != old.xml_sha256
    assert old.goal_xy == twin.goal_xy and old.goal_yaw == twin.goal_yaw
    a, b = (mj.MjModel.from_xml_string(sc.xml) for sc in (old, twin))
    for name in ("qpos0", "actuator_ctrlrange", "jnt_range"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    np.testing.assert_allclose(a.body_mass, b.body_mass, rtol=1e-8)
    np.testing.assert_allclose(a.body_ipos, b.body_ipos, rtol=1e-7, atol=1e-10)
    assert not np.allclose(a.body_inertia, b.body_inertia)
    assert a.neq == b.neq == 0
    data = mj.MjData(b)
    mj.mj_forward(b, data)
    obj_geoms = {b.geom(f"payload_{i}").id for i in range(len(twin.boxes))}
    for side in ("left", "right"):
        ids = [b.geom(f"{side}_pad_{s}").id for s in ("a", "b")]
        np.testing.assert_allclose(data.geom_xpos[ids].mean(axis=0), data.site(f"{side}_tool").xpos)
        delta = data.geom_xpos[ids[1]]-data.geom_xpos[ids[0]]
        expected_span = (.064, 0, 0) if task == "carry" else (0, .064, 0)
        np.testing.assert_allclose(delta, expected_span, atol=1e-12)
        assert all(not (set(c.geom).intersection(ids) and set(c.geom).intersection(obj_geoms))
                   for c in data.contact)


def test_two_loaded_pad_pairs_and_moments_from_native_contacts(env_module):
    scene = env_module.make_scene("carry", seed=701, tool_profile="twin-pad")
    model = mj.MjModel.from_xml_string(scene.xml)
    data = mj.MjData(model)
    task = env_module.BimanualTask(scene, model)
    for side, count, sign in zip(("left", "right"), scene.arm_dofs, (1, -1), strict=True):
        data.joint(f"{side}_j0").qpos[0] = -sign*.04
        data.joint(f"{side}_j{count-1}").qpos[0] = sign*.04
    mj.mj_forward(model, data)
    expected = np.zeros((2, 6))
    channels = np.zeros(4)
    rows = []
    payload = model.geom("payload_0").id
    pad_ids = [model.geom(f"{side}_pad_{letter}").id
               for side in ("left", "right") for letter in ("a", "b")]
    for i, c in enumerate(data.contact):
        a, b = map(int, c.geom)
        frame = np.array(c.frame).reshape(3, 3)
        local = np.empty(6)
        mj.mj_contactForce(model, data, i, local)
        rows.append(SimpleNamespace(geometry_ids=(a, b), position=c.pos.copy(),
                                    wrench_contact_frame=local, frame=frame))
        if payload not in (a, b):
            continue
        pad = b if a == payload else a
        if pad not in pad_ids:
            continue
        k = pad_ids.index(pad)
        sign = 1 if b == payload else -1
        force = sign*(local[:3] @ frame)
        channels[k] += local[0]
        expected[k // 2, :3] += force
        expected[k // 2, 3:] += np.cross(c.pos, force) + sign*(local[3:] @ frame)
    assert np.all(channels > 1.), channels
    info = task.evaluate(data.qpos, data.qvel, rows)
    np.testing.assert_allclose(info["individual_pad_payload_normal_n"], channels)
    np.testing.assert_allclose(info["pad_normal_n"], channels.reshape(2, 2).sum(axis=1))
    np.testing.assert_allclose(info["wrist_force_world_n"], -expected[:, :3])
    np.testing.assert_allclose(info["pad_payload_wrench_about_world_origin"], expected)
    assert info["all_individual_pads_loaded_s"] == pytest.approx(scene.timestep)


def test_direct_torque_uses_original_limits_and_leaves_parked_arm_unpowered(env_module):
    with env_module.BimanualEnv("carry", seed=701, tool_profile="twin-pad", action_repeat=2) as env:
        env.reset(active_arms="left")
        action = np.tile([-1., 1.], 7)
        _, _, _, _, info = env.step_torque(action)
        n = env.scene.arm_dofs[0]
        limits = env.model.actuator_ctrlrange[:n]
        expected = np.where(action[:n] > 0, limits[:, 1], limits[:, 0])
        np.testing.assert_array_equal(info["applied_ctrl"][:n], expected)
        np.testing.assert_array_equal(info["applied_ctrl"][n:], np.zeros(env.model.nu-n))
        assert info["tick"] == 2
