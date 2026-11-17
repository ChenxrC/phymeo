"""Native scene lowering with independent FP64 kinematic/dynamic references.

Frozen tolerances: 1e-10 for FP64 pose/mass; 2e-7 metres for native FP32 hull
vertices; 2e-7/3e-5 for existing resident
FP32 one-step mechanics. No assertion here qualifies PhySolve or the P2PE ABI.
"""

from dataclasses import replace

import numpy as np
import pytest
from test_scene_compiler import H, _command, _world

import phymeo as ph
from phymeo.builders.mujoco_scene import MuJoCoWorldProfile, compile_mujoco_world
from phymeo.builders.scene_compiler import SceneDeclaration, compile_scene
from phymeo.dynamics.floating_robot import floating_kinematics
from phymeo.dynamics.robot_dynamics import mass_matrix
from phymeo.dynamics.robot_kinematics import forward_kinematics
from phymeo.dynamics.rotation import quaternion_matrix

PROFILE = MuJoCoWorldProfile(H, (0.6, 0.01, 0.001), False)


@pytest.mark.parametrize("floating", [False, True])
@pytest.mark.parametrize("kind", ["revolute", "prismatic"])
def test_pose_com_joint_identity_and_initial_velocity(floating, kind):
    mj = pytest.importorskip("mujoco")
    world = _world(71, floating=floating, kind=kind)
    if floating:
        physical = replace(world.asset.model, root_com_m=np.array([0.13, -0.07, 0.03]))
        world = replace(
            world,
            asset=replace(world.asset, model=physical),
            initial_state=ph.FloatingRobotState(
                np.r_[[0.4, -0.2, 1], [0, 0, np.sqrt(0.5), np.sqrt(0.5)], 0.21],
                np.array([0.2, -0.3, 0.1, -0.7, 0.4, 0.2, 0.17]),
            ),
        )
        expected = floating_kinematics(physical, world.initial_state)
    else:
        world = replace(world, initial_state=ph.RobotState(np.array([0.21]), np.array([0.17])))
        expected = forward_kinematics(world.asset.model, world.initial_state.q)
    result = compile_mujoco_world(world, profile=PROFILE, instance="arm")
    model, data = result.fresh_native_data()
    mj.mj_forward(model, data)
    ids = dict(result.body_ids)
    selected = [
        ids[n] for n in ((world.asset.root_link,) if floating else ()) + world.asset.link_names
    ]
    np.testing.assert_allclose(data.xpos[selected], expected.position, atol=1e-10, rtol=0)
    np.testing.assert_allclose(data.xipos[selected], expected.com_position, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        data.xmat[selected].reshape(-1, 3, 3), expected.rotation, atol=1e-10, rtol=0
    )
    binding = result.joints[0]
    assert binding.name == world.asset.joint_names[0]
    assert data.qvel[binding.dof_address] == 0.17
    if floating:
        root = ids[world.asset.root_link]
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mj.mj_jacBodyCom(model, data, jp, jr, root)
        np.testing.assert_allclose(jp @ data.qvel, world.initial_state.v[:3], atol=1e-10, rtol=0)
        np.testing.assert_allclose(jr @ data.qvel, world.initial_state.v[3:6], atol=1e-10, rtol=0)
    else:
        native_mass = np.zeros((model.nv, model.nv))
        mj.mj_fullM(model, native_mass, data.qM)
        np.testing.assert_allclose(
            native_mass, mass_matrix(world.asset.model, world.initial_state.q), atol=1e-10, rtol=0
        )
    assert result.world == 71 and result.instance == "arm"
    assert result.compilation.collision_source is not None


def test_multiple_shapes_environment_and_explicit_contact_domain():
    world = _world(contact=True)
    first = world.asset.collisions[0]
    second = replace(
        first,
        name="other",
        shape=replace(first.shape, key=ph.CollisionKey(0, 42, 72)),
        local_vertices=((0.3, 0.0, 0.0),),
    )
    asset = replace(
        world.asset,
        collisions=(first, second),
        collision_model=ph.CollisionModel((first.shape, second.shape)),
    )
    result = compile_mujoco_world(
        replace(world, asset=asset), profile=PROFILE, instance="two-shapes"
    )
    model = result.compilation.mj_model
    assert len(result.shapes) == 3 and model.ngeom == 3 and model.npair == 2
    assert result.shapes[0].source_key != result.shapes[1].source_key
    assert (
        model.geom_bodyid[result.shapes[0].geom_id] == model.geom_bodyid[result.shapes[1].geom_id]
    )
    pairs = {frozenset((int(a), int(b))) for a, b in zip(model.pair_geom1, model.pair_geom2)}
    assert pairs == {
        frozenset((result.shapes[i].geom_id, result.shapes[2].geom_id)) for i in range(2)
    }
    disabled = replace(world, static_shapes=(replace(world.static_shapes[0], material=None),))
    assert (
        compile_mujoco_world(
            disabled, profile=PROFILE, instance="no-law"
        ).compilation.mj_model.npair
        == 0
    )


@pytest.mark.parametrize("cap,expected", [(0.0, 0.0), (2.0, 2.0), (100.0, 5.6)])
def test_explicit_native_drive_law_and_initial_control(cap, expected):
    mj = pytest.importorskip("mujoco")
    world = _world()
    world = replace(
        world,
        kp=np.array([10.0]),
        kd=np.array([2.0]),
        effort_limit=np.array([cap]),
        initial_state=ph.RobotState(np.array([0.1]), np.array([0.2])),
        initial_control=ph.RobotCommand(
            np.array([0.5]), target_velocity=np.array([0.7]), feedforward=np.array([0.6])
        ),
    )
    result = compile_mujoco_world(world, profile=PROFILE, instance="drive")
    model, data = result.fresh_native_data()
    mj.mj_forward(model, data)
    np.testing.assert_allclose(data.qfrc_actuator, [expected], atol=1e-10, rtol=0)
    assert result.drive_law == "mujoco-affine-pd-feedforward-inside-cap"


def test_compiled_snapshot_owns_initial_data_and_source():
    world = _world()
    result = compile_mujoco_world(world, profile=PROFILE, instance="owned")
    model, data = result.fresh_native_data()
    data.qpos[:] = 0.9
    model.body_mass[:] = 18
    next_model, next_data = result.fresh_native_data()
    np.testing.assert_array_equal(next_data.qpos, result.initial_qpos)
    assert next_model.body_mass[-1] == 2
    with pytest.raises(ValueError):
        result.initial_qpos.flags.writeable = True
    assert world.initial_state.q[0] == 0


def test_non_depth_first_tree_uses_explicit_q_and_dof_bindings():
    from test_scene_compiler import _asset

    from phymeo.builders.scene_compiler import ArticulationWorld

    mj = pytest.importorskip("mujoco")
    asset = _asset(n=3)
    asset = replace(asset, model=replace(asset.model, parents=np.array([-1, -1, 0])))
    world = ArticulationWorld(
        13,
        asset,
        ph.RobotState(np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])),
        kp=np.zeros(3),
        kd=np.zeros(3),
        effort_limit=np.ones(3),
    )
    result = compile_mujoco_world(world, profile=PROFILE, instance="branch")
    model, data = result.fresh_native_data()
    assert [row.qpos_address for row in result.joints] == [0, 2, 1]
    mj.mj_forward(model, data)
    expected = forward_kinematics(asset.model, world.initial_state.q)
    selected = [dict(result.body_ids)[n] for n in asset.link_names]
    np.testing.assert_allclose(data.xpos[selected], expected.position, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(
        data.qvel[[r.dof_address for r in result.joints]], world.initial_state.v
    )


def test_fixed_root_transform_is_not_applied_twice():
    mj = pytest.importorskip("mujoco")
    world = _world(kind="revolute")
    asset = replace(world.asset, root_xyz=(0.4, -0.5, 0.6), root_xyzw=(0.0, 0.0, 1.0, 0.0))
    result = compile_mujoco_world(
        replace(world, asset=asset), profile=PROFILE, instance="offset-root"
    )
    model, data = result.fresh_native_data()
    mj.mj_forward(model, data)
    expected = forward_kinematics(asset.model, world.initial_state.q)
    np.testing.assert_allclose(
        data.xpos[dict(result.body_ids)[asset.link_names[0]]],
        expected.position[0],
        atol=1e-10,
        rtol=0,
    )


def test_root_only_rigid_retains_principal_inertia_frame_and_com():
    mj = pytest.importorskip("mujoco")
    world = _world(floating=True)
    physical = replace(
        world.asset.model,
        children=None,
        joint_armature=np.zeros(0),
        joint_damping=np.zeros(0),
        root_com_m=np.array([0.1, -0.2, 0.3]),
        root_inertia_frame_xyzw=np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]),
    )
    asset = replace(
        world.asset,
        model=physical,
        link_names=(),
        joint_names=(),
        lower=(),
        upper=(),
        effort_limit=(),
        velocity_limit=(),
        collisions=(),
        collision_model=ph.CollisionModel(()),
    )
    world = replace(
        world,
        asset=asset,
        initial_state=ph.FloatingRobotState(
            np.array([0.4, -0.2, 1.0, 0.0, 0.0, 0.0, 1.0]), np.zeros(6)
        ),
        kp=np.zeros(0),
        kd=np.zeros(0),
        effort_limit=np.zeros(0),
        initial_control=None,
    )
    result = compile_mujoco_world(world, profile=PROFILE, instance="rigid")
    model, data = result.fresh_native_data()
    mj.mj_forward(model, data)
    assert model.nq == 7 and model.nv == 6 and model.nu == 0
    np.testing.assert_allclose(data.xipos[1], [0.5, -0.4, 1.3], atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        model.body_iquat[1], [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], atol=1e-10, rtol=0
    )


def test_real_panda_asset_lowering_matches_public_fk():
    from phymeo.builders.panda import load_panda
    from phymeo.builders.scene_compiler import ArticulationWorld

    mj = pytest.importorskip("mujoco")
    asset = load_panda()
    n = len(asset.joint_names)
    q = np.clip(np.zeros(n), asset.lower, asset.upper)
    world = ArticulationWorld(
        0,
        asset,
        ph.RobotState(q, np.zeros(n)),
        kp=np.zeros(n),
        kd=np.zeros(n),
        effort_limit=np.asarray(asset.effort_limit),
    )
    result = compile_mujoco_world(world, profile=PROFILE, instance="panda")
    model, data = result.fresh_native_data()
    mj.mj_forward(model, data)
    expected = forward_kinematics(asset.model, q)
    ids = [dict(result.body_ids)[name] for name in asset.link_names]
    np.testing.assert_allclose(data.xpos[ids], expected.position, atol=1e-10, rtol=0)
    native_mass = np.zeros((model.nv, model.nv))
    mj.mj_fullM(model, native_mass, data.qM)
    order = [row.dof_address for row in result.joints]
    np.testing.assert_allclose(
        native_mass[np.ix_(order, order)], mass_matrix(asset.model, q), atol=1e-10, rtol=0
    )
    for frame in result.frames:
        position, rotation = asset.authored_link_pose(frame.name, q)
        body_rotation = data.xmat[frame.body_id].reshape(3, 3)
        np.testing.assert_allclose(
            data.xpos[frame.body_id] + body_rotation @ frame.local_position,
            position,
            atol=1e-10,
            rtol=0,
        )
        np.testing.assert_allclose(
            body_rotation @ quaternion_matrix(np.array(frame.local_orientation_xyzw)[None])[0],
            rotation,
            atol=1e-10,
            rtol=0,
        )
    motions = asset.collision_motion(q)
    colliders = {row.shape.key: row for row in asset.collisions}
    for binding in result.shapes:
        row = colliders[binding.source_key]
        geom = binding.geom_id
        rotation = data.geom_xmat[geom].reshape(3, 3)
        if row.shape.kind == "sphere":
            points = data.geom_xpos[geom][None]
        elif row.shape.kind == "capsule":
            offset = rotation[:, 2] * model.geom_size[geom, 1]
            points = np.array([data.geom_xpos[geom] - offset, data.geom_xpos[geom] + offset])
        elif row.shape.kind == "hull":
            mesh = model.geom_dataid[geom]
            start, size = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            points = data.geom_xpos[geom] + model.mesh_vert[start : start + size] @ rotation.T
        else:
            pytest.fail(f"unverified Panda geometry {row.shape.kind}")
        source = motions[binding.source_key].start
        np.testing.assert_allclose(points.min(axis=0), source.min(axis=0), atol=2e-7, rtol=0)
        np.testing.assert_allclose(points.max(axis=0), source.max(axis=0), atol=2e-7, rtol=0)


def test_contact_dimension_does_not_silently_discard_friction():
    with pytest.raises(ValueError, match="ignore"):
        MuJoCoWorldProfile(H, (0.6, 0.01, 0.001), False, contact_dim=3)
    result = compile_mujoco_world(_world(contact=True), profile=PROFILE, instance="all-friction")
    np.testing.assert_array_equal(result.compilation.mj_model.pair_dim, [6])


def test_same_numeric_shape_keys_have_distinct_source_domains():
    from phymeo.builders.scene_compiler import FiniteStaticShape

    world = _world(0)
    original = world.asset.collisions[0].shape.key
    env = FiniteStaticShape.sphere(
        0, original.body, [0, 0, -1], 0.1, material=ph.RobotContactMaterial(64, 0, 0)
    )
    env = replace(env, shape=replace(env.shape, key=original))
    result = compile_mujoco_world(
        replace(world, static_shapes=(env,)), profile=PROFILE, instance="domains"
    )
    assert result.shapes[0].source_key == result.shapes[1].source_key
    assert len({r.source_identity for r in result.shapes}) == 2


def test_group_mask_excludes_only_disabled_environment_pair():
    world = _world(contact=True)
    env = world.static_shapes[0]
    result = compile_mujoco_world(
        replace(world, static_shapes=(replace(env, shape=replace(env.shape, mask=0)),)),
        profile=PROFILE,
        instance="masked",
    )
    assert result.compilation.mj_model.ngeom == 2
    assert result.compilation.mj_model.npair == 0


def test_floating_passive_parameters_and_rotated_child_frames_are_retained():
    mj = pytest.importorskip("mujoco")
    world = _world(floating=True, kind="revolute")
    children = replace(
        world.asset.model.children,
        joint_orientation=np.array([[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]]),
    )
    physical = replace(
        world.asset.model,
        children=children,
        joint_armature=np.array([0.13]),
        joint_damping=np.array([0.21]),
    )
    world = replace(world, asset=replace(world.asset, model=physical))
    result = compile_mujoco_world(world, profile=PROFILE, instance="passive")
    model, data = result.fresh_native_data()
    address = result.joints[0].dof_address
    assert model.dof_armature[address] == 0.13 and model.dof_damping[address] == 0.21
    mj.mj_forward(model, data)
    expected = floating_kinematics(physical, world.initial_state)
    ids = [i for _, i in result.body_ids]
    np.testing.assert_allclose(
        data.xmat[ids].reshape(-1, 3, 3), expected.rotation, atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("case", ["mimic", "self_contact", "wrong_type"])
def test_unadmitted_semantics_fail_before_native_compilation(case, monkeypatch):
    import phymeo.builders.mujoco_scene as module

    world = _world(affine=case == "mimic")
    if case == "self_contact":
        world = replace(world, self_collision=True)
    if case == "wrong_type":
        world = object()

    def forbidden(*args, **kwargs):
        raise AssertionError("should reject before native compile")

    monkeypatch.setattr(module, "compile_mujoco_xml", forbidden)
    with pytest.raises((NotImplementedError, TypeError)):
        compile_mujoco_world(world, profile=PROFILE, instance="bad")


@pytest.mark.tyx
def test_same_declaration_resident_needs_no_native_compiler(tyx_mode, tmp_path, monkeypatch):
    import phymeo.builders.mujoco_scene as module

    def forbidden(*args, **kwargs):
        raise AssertionError("resident must not depend on the native compiler")

    monkeypatch.setattr(module, "compile_mujoco_xml", forbidden)
    world = _world(71)
    with compile_scene(SceneDeclaration((world,)), tyx_mode, cache_root=tmp_path) as runtime:
        runtime.set_controls({71: _command(world, 1)})
        assert runtime.step(H) == {71: True}
        np.testing.assert_allclose(runtime.snapshot()[71].v, [H / 2], rtol=3e-5, atol=2e-7)
