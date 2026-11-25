"""Native MuJoCo checks for the single- and twin-pad manipulation fixtures."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

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


def test_side_lip_is_an_opt_in_single_free_compound_with_a_clear_insertion_path(env_module):
    standard = env_module.make_scene("carry", seed=701, tool_profile="twin-pad")
    scene = env_module.make_scene("carry", seed=701, tool_profile="twin-pad", load_profile="side-lip")
    assert scene.xml_sha256 != standard.xml_sha256
    assert scene.start_xyz == standard.start_xyz and scene.goal_xy == standard.goal_xy
    with pytest.raises(ValueError, match="carry-only"):
        env_module.make_scene("push_t", load_profile="side-lip")
    with pytest.raises(ValueError):
        env_module.make_scene("carry", load_profile="other")
    xml = ET.fromstring(scene.xml)
    geoms = xml.findall("worldbody/body[@name='payload']/geom")
    assert len(geoms) == 3
    volumes = [8*np.prod(np.fromstring(g.get("size"), sep=" ")) for g in geoms]
    masses = np.array([float(g.get("mass")) for g in geoms])
    np.testing.assert_allclose(masses/volumes, 6./sum(volumes))
    model = mj.MjModel.from_xml_string(scene.xml)
    base = mj.MjModel.from_xml_string(standard.xml)
    assert model.body("payload").mass[0] == pytest.approx(6.)
    assert model.body("payload").jntnum[0] == 1 and model.joint("payload_free").type[0] == mj.mjtJoint.mjJNT_FREE
    assert model.neq == 0 and model.nmocap == 0
    np.testing.assert_array_equal(base.actuator_ctrlrange, model.actuator_ctrlrange)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    pads = {model.geom(f"{s}_pad_{p}").id for s in ("left","right") for p in ("a","b")}
    sides = {model.geom(f"payload_{i}").id for i in (1,2)}
    workpiece = sides | {model.geom("payload_0").id}
    assert not any(pads.intersection(c.geom) and workpiece.intersection(c.geom) for c in data.contact)
    lip = model.geom("payload_1").id
    pad = model.geom("left_pad_a").id
    bottom = data.geom_xpos[lip,2]-model.geom_size[lip,2]
    assert bottom == pytest.approx(.082)
    assert bottom-data.geom_xpos[pad,2]-model.geom_size[pad,0] == pytest.approx(.006)
    assert data.geom_xpos[lip,2]+model.geom_size[lip,2] == pytest.approx(.100)

    # Pose only the two arms for an independent native underside-contact check.
    # The free body's native initial pose and both force channels stay untouched.
    for side, count, sign in zip(("left","right"), scene.arm_dofs, (1,-1), strict=True):
        data.joint(f"{side}_j0").qpos[0] = -sign*.025
        data.joint(f"{side}_j{count-1}").qpos[0] = sign*.025
        data.joint(f"{side}_j1").qpos[0] = -.026
    mj.mj_forward(model, data)
    under = set()
    for i, contact in enumerate(data.contact):
        a,b = map(int, contact.geom)
        if not ({a,b}.intersection(pads) and {a,b}.intersection(sides)):
            continue
        local = np.empty(6)
        mj.mj_contactForce(model, data, i, local)
        direction_z = contact.frame.reshape(3,3)[0,2] * (1. if b in sides else -1.)
        if contact.dist < 0 and direction_z > .97 and local[0] > 1.:
            under.add(b if b in sides else a)
    assert under == sides


@pytest.mark.parametrize("load_profile", ["standard", "side-lip"])
def test_bent_elbow_preserves_plant_load_and_initial_wrist_frames(env_module, load_profile):
    args = dict(task="carry", seed=701, tool_profile="twin-pad", load_profile=load_profile)
    original = env_module.make_scene(**args)
    bent = env_module.make_scene(**args, stance="bent-elbow")
    assert original.xml_sha256 != bent.xml_sha256
    assert (original.start_xyz, original.goal_xy, original.goal_yaw, original.boxes) == (
        bent.start_xyz, bent.goal_xy, bent.goal_yaw, bent.boxes)
    assert original.mass == bent.mass == 6.0
    models = [mj.MjModel.from_xml_string(s.xml) for s in (original, bent)]
    a, b = models
    assert a.nkey == 0 and b.nkey == 1
    for field in ("jnt_axis", "jnt_pos", "jnt_range", "dof_damping", "dof_armature",
                  "body_mass", "body_inertia", "body_ipos", "geom_type", "geom_size",
                  "geom_pos", "geom_quat", "geom_friction", "actuator_ctrlrange", "actuator_gear"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    fixed_bases = {b.body(f"{s}_link0").id for s in env_module.SIDES}
    moved = np.flatnonzero(np.max(abs(b.body_pos - a.body_pos), axis=1) > 1e-10)
    assert set(moved) == fixed_bases
    data = []
    for scene, model in zip((original, bent), models, strict=True):
        d = mj.MjData(model)
        d.qpos[:] = env_module.BimanualTask(scene, model).initial_state().q
        mj.mj_forward(model, d)
        data.append(d)
    da, db = data
    np.testing.assert_array_equal(da.qpos[:7], db.qpos[:7])
    np.testing.assert_array_equal(db.qpos, b.key_qpos[0])
    for side, count in zip(env_module.SIDES, bent.arm_dofs, strict=True):
        expected = np.zeros(count)
        expected[2], expected[-2] = .52, -.52
        actual = [db.joint(f"{side}_j{i}").qpos[0] for i in range(count)]
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        for suffix in ("tool", "wrist_camera"):
            first, second = (d.site(f"{side}_{suffix}") for d in data)
            np.testing.assert_allclose(first.xpos, second.xpos, atol=2e-8, rtol=0.)
            np.testing.assert_allclose(first.xmat, second.xmat, atol=1e-12, rtol=0.)
        for suffix in ("a", "b"):
            p = db.geom(f"{side}_pad_{suffix}").xpos
            lateral_gap = abs(p[1]-db.body("payload").xpos[1]) - .11 - env_module.PAD_RADIUS
            assert lateral_gap == pytest.approx(.008, abs=2e-8)
        elbow_levers = []
        for model, d in zip(models, data, strict=True):
            jac = np.empty((3, model.nv))
            mj.mj_jacSite(model, d, jac, None, model.site(f"{side}_tool").id)
            elbow_levers.append(abs(jac[2, model.joint(f"{side}_j2").dofadr[0]]))
        assert elbow_levers[0] - elbow_levers[1] > .14
    # Disabled motors stay off with the inactive arm parked in the new stance.
    parked = env_module.BimanualTask(bent, b, "left")
    parked_q = parked.initial_state().q
    assert parked_q[b.joint("right_j0").qposadr[0]] == pytest.approx(-1.15)
    assert parked_q[b.joint("right_j2").qposadr[0]] == pytest.approx(.52)
    assert not any(parked.enabled[bent.arm_dofs[0]:])


@pytest.mark.parametrize("seed", [701, 702])
def test_bent_elbow_can_reach_full_transport_while_keeping_wrist_level(env_module, seed):
    with pytest.raises(ValueError, match="carry-only"):
        env_module.make_scene("push_t", stance="bent-elbow")
    scene = env_module.make_scene("carry", seed, tool_profile="twin-pad", stance="bent-elbow")
    model = mj.MjModel.from_xml_string(scene.xml)
    for side, count in zip(env_module.SIDES, scene.arm_dofs, strict=True):
        data = mj.MjData(model)
        data.qpos[:] = model.key_qpos[0]
        mj.mj_forward(model, data)
        tool = model.site(f"{side}_tool").id
        orientation = data.site_xmat[tool].copy()
        target = data.site_xpos[tool] + (.12, 0., .045)
        joints = [model.joint(f"{side}_j{i}").id for i in (1, 2, count-2)]
        qadr, vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        jac = np.empty((3, model.nv))
        for _ in range(32):
            residual = target - data.site_xpos[tool]
            if np.linalg.norm(residual) < 1e-7:
                break
            mj.mj_jacSite(model, data, jac, None, tool)
            # Distal pitch cancels shoulder and elbow throughout the reach.
            coupled = jac[:, vadr[:2]] - jac[:, vadr[2], None]
            step = np.linalg.solve(coupled[[0, 2], :], residual[[0, 2]])
            step *= min(1., .08/max(np.max(abs(step)), 1e-12))
            data.qpos[qadr] += (step[0], step[1], -step.sum())
            assert np.all(data.qpos[qadr] >= model.jnt_range[joints, 0])
            assert np.all(data.qpos[qadr] <= model.jnt_range[joints, 1])
            mj.mj_forward(model, data)
        np.testing.assert_allclose(data.site_xpos[tool], target, atol=2e-7, rtol=0.)
        np.testing.assert_allclose(data.site_xmat[tool], orientation, atol=2e-12, rtol=0.)
