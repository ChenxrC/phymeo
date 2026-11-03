"""Physical OpenUSD stage units, anchored scalar joints and local collider ownership."""

from pathlib import Path

import numpy as np
import pytest

import phymeo as ph

ASSETS = Path(__file__).parent / "reference_newton" / "robot"


def load(path):
    from phymeo.builders.usd import load_usd

    return load_usd(path)


@pytest.mark.parametrize("filename", ["arm.usda", "arm_cm_y.usda"])
def test_usd_units_and_up_axis_give_the_same_physical_arm(filename):
    pytest.importorskip("pxr.Usd")
    asset = load(ASSETS / filename)
    assert isinstance(asset.model, ph.RobotModel)
    assert asset.root_link == "base"
    assert asset.link_names == ("boom", "tool") and asset.joint_names == ("yaw", "slide")
    np.testing.assert_allclose(asset.model.mass, [1, 0.5], atol=2e-7)
    np.testing.assert_allclose(asset.model.inertia_diagonal, [[0.1, 0.1, 0.1]] * 2, atol=2e-7)
    np.testing.assert_allclose(
        asset.model.com, [[0.25, 0, 0], [0.125, -0.0625, 0.03125]], atol=2e-7
    )
    np.testing.assert_allclose(asset.lower, [-np.pi / 2, -0.1], atol=2e-7)
    np.testing.assert_allclose(asset.upper, [2 * np.pi / 3, 0.25], atol=2e-7)
    (marker,) = asset.collisions
    assert (marker.link_index, marker.name, marker.shape.kind) == (1, "marker", "sphere")
    assert marker.shape.radius == pytest.approx(0.03125)
    for q, v, expected, speed in [
        ([0.0, 0.0], [0.0, 0.0], [0.8125, -0.375, 0.15625], [0, 0, 0]),
        ([np.pi / 2, 0.125], [0.5, -0.125], [0, 0.0625, 0.15625], [-0.15625, -0.125, 0]),
        (
            [-np.arctan2(4, 3), -0.0625],
            [-0.75, 0.0625],
            [0.6375, -0.9125, 0.15625],
            [-0.259375, -0.253125, 0],
        ),
    ]:
        q, v = np.array(q), np.array(v)
        point = np.asarray(marker.local_vertices)
        port = ph.RobotPointPort(asset.model, ph.RobotState(q, v), [1], point)
        np.testing.assert_allclose(port.positions()[0], expected, rtol=0, atol=2e-7)
        np.testing.assert_allclose(port.velocities()[0], speed, rtol=0, atol=2e-7)
        np.testing.assert_allclose(
            asset.collision_motion(q)[marker.shape.key].start[0], expected, rtol=0, atol=2e-7
        )


def copy_stage():
    from pxr import Sdf, Usd

    layer = Sdf.Layer.CreateAnonymous("arm.usda")
    assert layer.ImportFromString((ASSETS / "arm.usda").read_text(encoding="utf-8"))
    return Usd.Stage.Open(layer)


@pytest.mark.parametrize("reverse", [False, True])
def test_both_anchor_frames_preserve_authored_link_points(reverse):
    pytest.importorskip("pxr.Usd")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1)
    rig = UsdGeom.Xform.Define(stage, "/rig")
    UsdPhysics.ArticulationRootAPI.Apply(rig.GetPrim())
    base = UsdGeom.Xform.Define(stage, "/rig/foot")
    body = UsdGeom.Xform.Define(stage, "/rig/shaft")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    pbody = body.GetPrim()
    mass = UsdPhysics.MassAPI.Apply(pbody)
    mass.CreateMassAttr(2)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.25, 0, 0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1, 2, 2.5))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
    body.AddTranslateOp().Set(Gf.Vec3d(0.5, 0, -0.125))
    body.AddOrientOp().Set(Gf.Quatf(0.5, 0.5, -0.5, 0.5))
    carrier = UsdGeom.Xform.Define(stage, "/rig/shaft/carrier")
    carrier.AddTranslateOp().Set(Gf.Vec3d(0.25, 0, 0))
    ball = UsdGeom.Sphere.Define(stage, "/rig/shaft/carrier/probe")
    ball.AddTranslateOp().Set(Gf.Vec3d(0, 0.125, 0))
    ball.CreateRadiusAttr(0.02)
    UsdPhysics.CollisionAPI.Apply(ball.GetPrim())
    mount = UsdPhysics.FixedJoint.Define(stage, "/rig/mount")
    mount.CreateBody0Rel().SetTargets([base.GetPath()])
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/rig/spin")
    h = np.sqrt(0.5)
    a = (base.GetPath(), (0.5, 0, 0), Gf.Quatf(h, 0, 0, h))
    b = (body.GetPath(), (0.125, 0, 0), Gf.Quatf(h, 0, h, 0))
    first, second = (b, a) if reverse else (a, b)
    joint.CreateBody0Rel().SetTargets([first[0]])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*first[1]))
    joint.CreateLocalRot0Attr(first[2])
    joint.CreateBody1Rel().SetTargets([second[0]])
    joint.CreateLocalPos1Attr(Gf.Vec3f(*second[1]))
    joint.CreateLocalRot1Attr(second[2])
    joint.CreateAxisAttr("X")
    joint.CreateLowerLimitAttr(-120)
    joint.CreateUpperLimitAttr(120 if not reverse else 60)

    asset = load(stage)
    q = np.array([-np.pi / 2 if reverse else np.pi / 2])
    v = np.array([-0.5 if reverse else 0.5])
    np.testing.assert_allclose(asset.model.com, [[0.125, 0, 0]], atol=2e-7)
    (marker,) = asset.collisions
    authored = np.array([[0.25, 0.125, 0]])
    np.testing.assert_allclose(marker.local_vertices, authored, atol=2e-7)
    local = asset.authored_to_model_points("shaft", authored)
    np.testing.assert_allclose(local, [[0.125, 0.125, 0]], atol=2e-7)
    position, rotation = asset.authored_link_pose("shaft", q)
    np.testing.assert_allclose(position, [0.375, 0, 0], atol=2e-7)
    np.testing.assert_allclose(rotation @ [1, 0, 0], [1, 0, 0], atol=2e-7)
    port = ph.RobotPointPort(asset.model, ph.RobotState(q, v), [marker.link_index], local)
    np.testing.assert_allclose(port.positions()[0], [0.625, 0, 0.125], atol=2e-7)
    np.testing.assert_allclose(port.velocities()[0], [0.0625, 0, -0.0625], atol=2e-7)
    np.testing.assert_allclose(
        asset.collision_motion(q)[marker.shape.key].start[0], port.positions()[0], atol=2e-7
    )
    np.testing.assert_allclose(asset.upper, [np.deg2rad(60 if reverse else 120)], atol=2e-7)


def test_metadata_change_alone_reduces_the_physical_robot():
    pytest.importorskip("pxr.Usd")
    from pxr import UsdGeom

    stage = copy_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    asset = load(stage)
    assert asset.collisions[0].shape.radius == pytest.approx(0.03125 / 100)
    np.testing.assert_allclose(asset.model.inertia_diagonal, np.full((2, 3), 0.00001), rtol=2e-7)
    np.testing.assert_allclose(asset.upper, [2 * np.pi / 3, 0.0025], atol=2e-7)


def test_y_up_world_metadata_rotates_geometry_once():
    pytest.importorskip("pxr.Usd")
    from pxr import UsdGeom

    stage = copy_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    asset = load(stage)
    (marker,) = asset.collisions
    point = asset.collision_motion(np.array([np.pi / 2, 0.125]))[marker.shape.key].start[0]
    np.testing.assert_allclose(point, [0, -0.15625, 0.0625], atol=2e-7)


def test_optional_sdk_is_loaded_only_by_the_entry_point(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pxr", None)
    import phymeo.builders.usd as usd

    with pytest.raises(ImportError, match="usd-core"):
        usd.load_usd(ASSETS / "arm.usda")


@pytest.mark.parametrize(
    "change,match",
    [
        ("ellipsoid", "sphere|scale"),
        ("kinematic", "kinematic"),
        ("cross_inertia", "inertia"),
        ("loop", "tree|cycle|joint"),
        ("detached", "link|body|target"),
        ("drive", "Drive|drive"),
        ("dynamic_frame", "time|animated"),
        ("bad_length", "metersPerUnit|unit"),
        ("instance", "instance"),
        ("body_scale", "scale"),
    ],
)
def test_unrepresented_physics_fails_at_the_public_entry(change, match):
    pytest.importorskip("pxr.Usd")
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = copy_stage()
    tool = stage.GetPrimAtPath("/arm/tool")
    joint = UsdPhysics.PrismaticJoint.Get(stage, "/arm/slide")
    if change == "ellipsoid":
        UsdGeom.Xformable(stage.GetPrimAtPath("/arm/tool/marker")).AddScaleOp().Set(
            Gf.Vec3f(2, 1, 1)
        )
    elif change == "kinematic":
        UsdPhysics.RigidBodyAPI(tool).CreateKinematicEnabledAttr(True)
    elif change == "cross_inertia":
        mass = UsdPhysics.MassAPI(tool)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1, 2, 2.5))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(0.92387953, 0, 0, 0.38268343))
    elif change == "loop":
        extra = UsdPhysics.RevoluteJoint.Define(stage, "/arm/loop")
        extra.CreateBody0Rel().SetTargets(["/arm/tool"])
        extra.CreateBody1Rel().SetTargets(["/arm/base"])
        extra.CreateLowerLimitAttr(-10)
        extra.CreateUpperLimitAttr(10)
    elif change == "detached":
        joint.CreateBody1Rel().SetTargets(["/arm/unknown"])
    elif change == "drive":
        UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear").CreateStiffnessAttr(10)
    elif change == "dynamic_frame":
        tool.GetAttribute("xformOp:translate").Set(Gf.Vec3d(1, 2, 3), 1)
    elif change == "instance":
        UsdGeom.PointInstancer.Define(stage, "/arm/tool/copies")
    elif change == "body_scale":
        UsdGeom.Xformable(tool).AddScaleOp().Set(Gf.Vec3f(1, 2, 1))
    else:
        stage.SetMetadata("metersPerUnit", -1.0)
    with pytest.raises(ValueError, match=match):
        load(stage)


def test_scaled_cube_active_parent_filters_and_scene_gravity():
    pytest.importorskip("pxr.Usd")
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = copy_stage()
    cube = UsdGeom.Cube.Define(stage, "/arm/boom/block")
    cube.CreateSizeAttr(0.1)
    cube.AddTranslateOp().Set(Gf.Vec3d(0.25, 0, 0))
    cube.AddOrientOp().Set(Gf.Quatf(np.sqrt(0.5), 0, 0, np.sqrt(0.5)))
    cube.AddScaleOp().Set(Gf.Vec3f(2, 4, 6))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    scene = UsdPhysics.Scene.Define(stage, "/physics")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, -2, 0))
    scene.CreateGravityMagnitudeAttr(4)
    asset = load(stage)
    box, ball = asset.collisions
    lo, hi = np.min(box.local_vertices, axis=0), np.max(box.local_vertices, axis=0)
    np.testing.assert_allclose(lo, [0.05, -0.1, -0.3], atol=2e-7)
    np.testing.assert_allclose(hi, [0.45, 0.1, 0.3], atol=2e-7)
    np.testing.assert_allclose(asset.model.gravity, [0, -4, 0])
    assert asset.collision_model.excluded == ((box.shape.key, ball.shape.key),)
    UsdPhysics.Joint.Get(stage, "/arm/slide").CreateCollisionEnabledAttr(True)
    assert load(stage).collision_model.excluded == ()


@pytest.mark.tyx
def test_cm_y_up_device_kinematics(tmp_path, tyx_mode):
    pytest.importorskip("pxr.Usd")
    from phymeo.backends.tyx_robot_math import TyxRobotMath

    asset = load(ASSETS / "arm_cm_y.usda")
    (marker,) = asset.collisions
    local = asset.authored_to_model_points("tool", marker.local_vertices)
    link = marker.link_index
    q = np.array([-np.arctan2(4, 3), -0.0625])
    with TyxRobotMath(asset.model, tyx_mode, cache_root=tmp_path) as kernels:
        fk = kernels.geometry(q)
        jac, _ = kernels.jacobians(q, [link], local)
        stats = kernels.stats()
    assert stats["completed_launches"] > 0
    if tyx_mode == "cuda":
        assert stats["downloads"] > 0
    np.testing.assert_allclose(
        fk.position[link] + fk.rotation[link] @ local[0],
        [0.6375, -0.9125, 0.15625],
        rtol=0,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        jac[0] @ [-0.75, 0.0625], [-0.259375, -0.253125, 0], rtol=0, atol=2e-6
    )
