"""M1 same-declaration multi-case contracts. AUTHORING_ONLY / UNEXECUTED.

Expected motions are independent scalar mass/inertia and contact-law formulas,
not another product solver. HOST and explicitly selected TYX run the same cases.
"""

from dataclasses import FrozenInstanceError, replace
from itertools import product

import numpy as np
import pytest

import phymeo as ph
from phymeo.builders._robot_asset import LinkCollision, RobotAsset
from phymeo.builders.floating_urdf import FloatingRobotAsset
from phymeo.builders.scene_compiler import (
    ArticulationWorld,
    FiniteStaticShape,
    SceneCompilationError,
    SceneDeclaration,
    compile_scene,
    scene_capabilities,
)
from phymeo.dynamics.floating_mimic import FloatingMimicRobotAsset
from phymeo.dynamics.joint_mimic import MimicRobotAsset
from phymeo.training.resident_stepper import PhysicsStepFailure

H = 1 / 128


def _asset(kind="prismatic", *, floating=False, n=1, collider=True):
    offsets = np.zeros((n, 3))
    if not floating:
        offsets[:, 2] = 3 / 32
    com = np.zeros((n, 3))
    if kind == "revolute":
        com[:, 0] = 0.5
    model = ph.RobotModel(
        np.full(n, -1, dtype=np.int64),
        (kind,) * n,
        offsets,
        np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
        np.tile([0.0, 0.0, 1.0], (n, 1)),
        np.full(n, 2.0),
        com,
        np.full((n, 3), 0.25),
        np.zeros(3),
    )
    key = ph.CollisionKey(0, 42, 71)
    shape = ph.CollisionShape(key, "sphere", (901,), radius=1 / 8)
    collisions = (
        LinkCollision(
            "probe", "link0", 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
            shape, ((0.0, 0.0, 0.0),), True,
        ),
    ) if collider else ()
    fields = (
        "analytic", "root", tuple(f"link{i}" for i in range(n)),
        tuple(f"joint{i}" for i in range(n)), (-2.0,) * n, (2.0,) * n,
        (100.0,) * n, (20.0,) * n, collisions,
        ph.CollisionModel(tuple(row.shape for row in collisions)),
    )
    if floating:
        full = ph.FloatingRobotModel(
            ph.RigidBodyModel(np.array([0.25]), np.full((1, 3), 0.5), np.zeros(3)),
            np.zeros(3), model,
        )
        return FloatingRobotAsset(fields[0], fields[1], full, *fields[2:])
    return RobotAsset(fields[0], fields[1], model, *fields[2:])


def _world(world=17, *, kind="prismatic", floating=False, affine=False, contact=False):
    asset = _asset(kind, floating=floating, n=2 if affine else 1)
    if affine:
        asset = (
            FloatingMimicRobotAsset(asset, (("joint0", "joint1", -1.0, 0.0),))
            if floating else MimicRobotAsset(asset, "joint0", "joint1", -1.0, 0.0)
        )
    state = (
        ph.FloatingRobotState(
            np.r_[[0.0, 0.0, 3 / 32], [0.0, 0.0, 0.0, 1.0], 0.0], np.zeros(7)
        ) if floating else ph.RobotState(np.zeros(1), np.zeros(1))
    )
    environment = (
        FiniteStaticShape.box(
            world, 900, (1.0, 1.0, 1 / 8), position=(0.0, 0.0, -1 / 8),
            material=ph.RobotContactMaterial(64, 0, 0),
        ),
    ) if contact else ()
    return ArticulationWorld(
        world, asset, state, kp=np.zeros(1), kd=np.zeros(1),
        effort_limit=np.full(1, 100.0), static_shapes=environment,
    )


def _command(world, effort):
    joint = ph.RobotCommand(np.zeros(1), feedforward=np.array([float(effort)]))
    return ph.FloatingRobotCommand(joint) if isinstance(
        world.initial_state, ph.FloatingRobotState
    ) else joint


def test_declaration_owns_original_asset_state_controls_and_geometry():
    original = _world(contact=True)
    declaration = SceneDeclaration((original,))
    identity = declaration.identity
    original_source = original.asset
    with pytest.raises(ValueError):
        original_source.model.mass.flags.writeable = True
    with pytest.raises(ValueError):
        original.kp.flags.writeable = True
    with pytest.raises(ValueError):
        original.static_shapes[0].points_m.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        original.world = 123
    assert declaration.identity == identity
    assert declaration.worlds[0].asset.collisions[0].shape.key == ph.CollisionKey(0, 42, 71)
    assert declaration.worlds[0].asset.joint_names == ("joint0",)


def test_capability_is_explicit_for_static_capsule_and_self_contact():
    world = _world()
    capsule = FiniteStaticShape.capsule(
        world.world, 900, ((0.0, 0.0, 2.0), (0.0, 0.0, 3.0)), 0.1,
        material=ph.RobotContactMaterial(64, 0, 0),
    )
    declaration = SceneDeclaration((replace(world, static_shapes=(capsule,)),))
    report = scene_capabilities(declaration, "host")
    assert not report.supported
    assert any(row.feature == "static_geometry" for row in report.issues)
    with pytest.raises(SceneCompilationError) as captured:
        compile_scene(declaration, "host")
    assert captured.value.report == report
    self_contact = SceneDeclaration((replace(world, self_collision=True),))
    assert any(row.feature == "self_collision" for row in scene_capabilities(self_contact, "cuda").issues)


def test_capability_rejects_33_joints_without_allocating_a_runtime():
    asset = _asset(n=33, collider=False)
    world = ArticulationWorld(
        409, asset, ph.RobotState(np.zeros(33), np.zeros(33)),
        kp=np.zeros(33), kd=np.zeros(33), effort_limit=np.full(33, 100.0),
    )
    report = scene_capabilities(SceneDeclaration((world,)), "cuda")
    assert not report.supported
    assert any(row.world == 409 and row.feature == "tyx_lowering" for row in report.issues)


def test_static_box_pose_keeps_complete_material_vertices():
    shape = FiniteStaticShape.box(
        71, 900, (1.0, 2.0, 3.0), position=(4.0, -2.0, 1.0),
        orientation_xyzw=(0.0, 0.0, 1.0, 0.0), material=None,
    )
    expected = np.array(list(product((-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0))))
    expected = expected @ np.diag([-1.0, -1.0, 1.0]) + [4, -2, 1]
    np.testing.assert_array_equal(shape.points_m, expected)
    assert shape.shape.key == ph.CollisionKey(71, 900, 0)


def _free_matrix(backend, tmp_path):
    kinds = ("prismatic", "revolute", "fixed_affine", "floating", "floating_affine")
    worlds = tuple(
        _world(
            key, kind="revolute" if kind == "revolute" else "prismatic",
            floating=kind.startswith("floating"), affine="affine" in kind,
        ) for key, kind in zip((7, 103, 409, 711, 907), kinds, strict=True)
    )
    declaration = SceneDeclaration(worlds)
    with compile_scene(declaration, backend, cache_root=tmp_path) as scene:
        assert scene.world_keys == (7, 103, 409, 711, 907)
        scene.set_controls({world.world: _command(world, 1) for world in worlds})
        assert scene.step(H) == dict.fromkeys(scene.world_keys, True)
        state = scene.snapshot()
        for world, kind in zip(worlds, kinds, strict=True):
            row = state[world.world]
            effective_mass = 0.75 if kind == "revolute" else 4 if "affine" in kind else 2
            if kind == "floating":
                # A free 4 kg root + 2 kg slider: joint Schur mass = 4/3 kg.
                effective_mass = 4 / 3
                np.testing.assert_allclose(row.v[2], -H / 4, rtol=3e-5, atol=2e-7)
            if kind == "floating_affine":
                np.testing.assert_allclose(row.v[:6], 0, rtol=0, atol=2e-7)
            np.testing.assert_allclose(row.v[-1], H / effective_mass, rtol=3e-5, atol=2e-7)
            np.testing.assert_allclose(row.q[-1], H * H / effective_mass, rtol=3e-5, atol=2e-7)
            assert row.tick == 1 and row.time_s == H
            if "affine" in kind:
                np.testing.assert_allclose(row.full_q[-2:], [row.q[-1], -row.q[-1]], atol=2e-7)
        scene.reset((103,))
        assert scene.snapshot()[103].tick == 0
        assert scene.snapshot()[7].tick == 1
        assert scene.snapshot()[409].tick == 1
        assert scene.step(H / 2) == dict.fromkeys(scene.world_keys, True)
        assert scene.snapshot()[7].time_s == 3 * H / 2
        assert scene.snapshot()[103].time_s == H / 2


def test_host_multi_case_same_declaration(tmp_path):
    _free_matrix("host", tmp_path)


@pytest.mark.tyx
def test_tyx_multi_case_same_declaration(tyx_mode, tmp_path):
    _free_matrix(tyx_mode, tmp_path)


def _contact_oracle(backend, tmp_path, floating):
    world = _world(floating=floating, contact=True)
    declaration = SceneDeclaration((world,))
    with compile_scene(declaration, backend, cache_root=tmp_path) as scene:
        scene.set_controls({world.world: _command(world, 0)})
        assert scene.step(H) == {world.world: True}
        state = scene.snapshot()[world.world]
        if floating:
            # Force acts on the 2 kg slider, root is unloaded: slider-relative a=1.
            np.testing.assert_allclose(state.v[:6], 0, atol=2e-7, rtol=0)
        np.testing.assert_allclose(state.v[-1], H, atol=2e-7, rtol=3e-5)
        receipt = scene.contact_receipts[world.world]
        assert receipt is not None
        if backend == "host":
            forces = receipt.loads.inputs.rigid.force
            np.testing.assert_allclose(forces.sum(axis=0), [0, 0, -2], atol=1e-10, rtol=0)
        else:
            np.testing.assert_allclose(receipt.law_wrench_fp64[-1, :3], [0, 0, 2], atol=1e-10, rtol=0)
            np.testing.assert_allclose(
                sum((row.force_n for row in receipt.static_reactions.values()), np.zeros(3)),
                [0, 0, -2], atol=1e-10, rtol=0,
            )
        assert len(scene.shape_keys[world.world]) == len(world.asset.collisions)


@pytest.mark.parametrize("floating", (False, True))
def test_host_actual_contact_and_reaction(tmp_path, floating):
    _contact_oracle("host", tmp_path, floating)


@pytest.mark.tyx
@pytest.mark.parametrize("floating", (False, True))
def test_tyx_actual_contact_and_reaction(tyx_mode, tmp_path, floating):
    _contact_oracle(tyx_mode, tmp_path, floating)


def test_two_compiles_reset_controls_and_close_are_isolated(tmp_path):
    world = _world(711)
    declaration = SceneDeclaration((world,))
    with compile_scene(declaration) as first, compile_scene(declaration) as second:
        first.set_controls({711: _command(world, 1)})
        first.step(H)
        assert first.snapshot()[711].tick == 1 and second.snapshot()[711].tick == 0
        first.reset()
        assert first.snapshot()[711].tick == 0
        with pytest.raises(ValueError, match="exact"):
            second.set_controls({})
        first.close()
        with pytest.raises(RuntimeError, match="closed"):
            first.snapshot()
        assert second.step(H) == {711: True}


def test_actual_host_partial_publication_requires_full_reset():
    first, second = _world(7), _world(103)
    second = replace(second, asset=replace(second.asset, upper=(0.001,)))
    declaration = SceneDeclaration((first, second))
    with compile_scene(declaration) as scene:
        scene.set_controls({7: _command(first, 1), 103: _command(second, 1)})
        with pytest.raises(PhysicsStepFailure) as captured:
            scene.step(1 / 8)
        assert captured.value.acceptance_known
        assert captured.value.accepted_worlds == {7: True, 103: False}
        assert scene.snapshot()[7].tick == 1 and scene.snapshot()[103].tick == 0
        assert scene.needs_reset
        with pytest.raises(RuntimeError, match="all"):
            scene.reset((103,))
        scene.reset()
        scene.set_controls({7: _command(first, 0), 103: _command(second, 0)})
        assert scene.step(H) == {7: True, 103: True}

def _passive_oracle(backend, tmp_path):
    world = _world(floating=True)
    model = replace(
        world.asset.model, joint_armature=np.array([0.5]), joint_damping=np.array([0.25])
    )
    velocity = np.zeros(7)
    velocity[-1] = 0.2
    world = replace(
        world, asset=replace(world.asset, model=model),
        initial_state=ph.FloatingRobotState(world.initial_state.q, velocity),
    )
    with compile_scene(SceneDeclaration((world,)), backend, cache_root=tmp_path) as scene:
        scene.set_controls({world.world: _command(world, 1)})
        assert scene.step(H) == {world.world: True}
        actual = scene.snapshot()[world.world]
        # M=[[6,2],[2,5/2]], D_joint=1/4; old joint v=1/5.
        # (M+hD) delta_v = h*[0,19/20]. Passive inertia is not body mass.
        denominator = 11 + 3 * H / 2
        root_v = -19 * H / (10 * denominator)
        joint_v = 1 / 5 + 57 * H / (10 * denominator)
        np.testing.assert_allclose(actual.v[2], root_v, atol=2e-7, rtol=3e-5)
        np.testing.assert_allclose(actual.v[-1], joint_v, atol=2e-7, rtol=3e-5)
        np.testing.assert_allclose(actual.q[2], 3 / 32 + H * root_v, atol=2e-7, rtol=3e-5)
        np.testing.assert_allclose(actual.q[-1], H * joint_v, atol=2e-7, rtol=3e-5)


def test_host_preserves_original_floating_armature_and_damping(tmp_path):
    _passive_oracle("host", tmp_path)


@pytest.mark.tyx
def test_tyx_preserves_original_floating_armature_and_damping(tyx_mode, tmp_path):
    _passive_oracle(tyx_mode, tmp_path)


def test_actual_host_all_rejected_is_retryable_without_reset():
    world = _world()
    world = replace(world, asset=replace(world.asset, upper=(0.001,)))
    with compile_scene(SceneDeclaration((world,))) as scene:
        scene.set_controls({world.world: _command(world, 1)})
        with pytest.raises(PhysicsStepFailure) as captured:
            scene.step(1 / 8)
        assert captured.value.acceptance_known
        assert captured.value.accepted_worlds == {world.world: False}
        assert not scene.needs_reset
        assert scene.snapshot()[world.world].tick == 0
        scene.set_controls({world.world: _command(world, 0)})
        assert scene.step(H) == {world.world: True}


def test_command_validation_is_complete_before_any_control_replacement():
    first, second = _world(7), _world(103)
    with compile_scene(SceneDeclaration((first, second))) as scene:
        scene.set_controls({7: _command(first, 1), 103: _command(second, -1)})
        invalid = ph.RobotCommand(np.zeros(1), effort_limit=np.array([0.5]))
        with pytest.raises(ValueError, match="effort cap"):
            scene.set_controls({7: _command(first, 0), 103: invalid})
        scene.step(H)
        state = scene.snapshot()
        np.testing.assert_allclose(state[7].v, [H / 2], atol=1e-12, rtol=0)
        np.testing.assert_allclose(state[103].v, [-H / 2], atol=1e-12, rtol=0)


def test_declaration_does_not_depend_on_callers_mutable_asset_or_cap():
    asset = _asset()
    cap = np.array([100.0])
    world = ArticulationWorld(
        17, asset, ph.RobotState(np.zeros(1), np.zeros(1)),
        kp=np.zeros(1), kd=np.zeros(1), effort_limit=cap,
    )
    asset.model.mass.flags.writeable = True
    asset.model.mass[0] = 9
    cap[0] = 0
    np.testing.assert_array_equal(world.asset.model.mass, [2])
    np.testing.assert_array_equal(world.effort_limit, [100])
