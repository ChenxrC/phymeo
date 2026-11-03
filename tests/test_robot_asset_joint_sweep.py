"""Joint intervals must follow FK through the interior, including off-centre signed turns."""

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.dynamics.rotation import quaternion_matrix

ROOT = Path(__file__).parent / "reference_newton" / "robot"
ARM = ROOT / "joint_interval_arm.xml"


def joints(asset, **values):
    return np.array([values.get(n, 0.0) for n in asset.joint_names])


def by_name(asset):
    return {c.name: c for c in asset.collisions}


def at(path, t):
    if path.turn is None:
        return path.start if path.end is None else path.start + t * (path.end - path.start)
    turn = path.turn
    axis = np.array(turn.axis)
    p0 = np.array(turn.pivot_start)
    p1 = np.array(turn.pivot_end)
    source = path.start - p0
    theta = t * turn.angle
    c, s = np.cos(theta), np.sin(theta)
    return (
        p0
        + t * (p1 - p0)
        + c * source
        + s * np.cross(axis, source)
        + (1 - c) * (source @ axis)[:, None] * axis
    )


def detect_pair(asset, one, two, paths):
    model = ph.CollisionModel((one, two))
    with ph.CollisionWorkspace(model) as ws:
        result = ph.detect(
            model, {s.key: paths[s.key] for s in (one, two)}, ph.CollisionPolicy(), ws
        )
    assert result.status == "ok"
    return result.pairs[0]


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("changed", [False, True])
def test_two_separated_poses_hit_the_finite_capsule_interior_with_original_material(
    sign, changed, tmp_path
):
    translation = np.array([0.5, -0.25, 0.125]) if changed else np.zeros(3)
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) if changed else np.eye(3)
    file = ARM
    if changed:
        file = tmp_path / "reframed.xml"
        file.write_text(
            ARM.read_text("utf8").replace(
                '<body name="base">',
                '<body name="base" pos="0.5 -0.25 0.125" quat="0.5 0.5 0.5 0.5">',
            ),
            encoding="utf8",
        )
    asset = ph.load_mjcf(file)
    c = by_name(asset)
    begin, end = joints(asset), joints(asset, spin=sign * np.pi)
    paths = asset.collision_motion(begin, end)
    arm = c["pointer"].shape
    ball = c["north" if sign > 0 else "south"].shape
    w = detect_pair(asset, arm, ball, paths)
    assert w.status == "hit" and w.sweep.start_gap_m > 0 and w.sweep.end_gap_m > 0
    t = math.acos(1 / 6) / math.pi
    assert (
        w.sweep.first_t[0] <= t <= w.sweep.first_t[1]
        and w.sweep.first_t[1] - w.sweep.first_t[0] < 1e-8
    )
    boundary = w.sweep.witness
    foot = math.sqrt(35) / 4 - 1
    weights = boundary.first_weights if w.first == arm.key else boundary.second_weights
    # MJCF's registered -Z/+Z endpoints are the author's to/from points.
    np.testing.assert_allclose(
        c["pointer"].local_vertices, [[1, 0, 0], [0.5, 0, 0]], atol=2e-15, rtol=0
    )
    values = np.array([dict(weights)[i] for i in arm.vertex_ids])
    np.testing.assert_allclose(values, [foot, 1 - foot], atol=3e-9, rtol=0)
    centerline = values @ at(paths[arm.key], t)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(boundary.first_point) - centerline), arm.radius, atol=3e-9, rtol=0
    )
    point = np.array([math.sqrt(35) / 96, sign * (0.75 - 1 / 96), 0])
    np.testing.assert_allclose(
        boundary.first_point, translation + rotation @ point, atol=3e-9, rtol=0
    )
    np.testing.assert_allclose(
        boundary.second_point, translation + rotation @ point, atol=3e-9, rtol=0
    )
    np.testing.assert_allclose(paths[arm.key].turn.axis, rotation @ [0, 0, 1], atol=2e-14)
    assert paths[arm.key].turn.angle == sign * np.pi and not paths[ball.key].moving
    other = c["south" if sign > 0 else "north"].shape
    assert detect_pair(asset, arm, other, paths).status == "separated"


def test_moving_pivot_two_slides_and_independent_rotation_match_interior_fk_and_body_pose():
    asset = ph.load_mjcf(ARM)
    c = by_name(asset)
    first = joints(
        asset,
        advance=-0.125,
        diagonal=0.25,
        spin=-0.25,
        extend=0.375,
        bend=0.2,
        independent_spin=0.3,
    )
    last = joints(
        asset,
        advance=0.625,
        diagonal=-0.25,
        spin=1.125,
        extend=0.375,
        bend=0.2,
        independent_spin=-0.875,
    )
    original_first = first.copy()
    original_last = last.copy()
    paths = asset.collision_motion(first, last)
    main = paths[c["pointer"].shape.key].turn
    other = paths[c["other"].shape.key].turn
    np.testing.assert_allclose(main.pivot_start, [-0.125, 0.15, 0.2], atol=1e-14)
    np.testing.assert_allclose(main.pivot_end, [0.625, -0.15, -0.2], atol=1e-14)
    np.testing.assert_allclose(main.axis, [0, 0, 1])
    np.testing.assert_allclose(other.axis, [0, 1, 0])
    np.testing.assert_allclose(other.pivot_start, [4, 0, 0])
    assert main.angle == 1.375
    assert not paths[c["north"].shape.key].moving and not paths[c["south"].shape.key].moving
    key = c["other"].shape.key
    np.testing.assert_array_equal(
        paths[key].body_orientation_xyzw, asset.collision_motion(first)[key].body_orientation_xyzw
    )
    for t in (0.0, 0.125, 0.3, 0.5, 0.875, 1.0):
        static = asset.collision_motion(first + t * (last - first))
        for collider in c.values():
            np.testing.assert_allclose(
                at(paths[collider.shape.key], t),
                static[collider.shape.key].start,
                atol=2e-14,
                rtol=0,
            )
        theta = t * other.angle
        cos, sin = np.cos(theta), np.sin(theta)
        ry = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
        start = quaternion_matrix(np.asarray(paths[key].body_orientation_xyzw)[None, :])[0]
        sample = quaternion_matrix(np.asarray(static[key].body_orientation_xyzw)[None, :])[0]
        np.testing.assert_allclose(ry @ start, sample, atol=2e-14, rtol=0)
    np.testing.assert_array_equal(first, original_first)
    np.testing.assert_array_equal(last, original_last)


def test_signed_full_turn_cannot_be_collapsed_and_identical_endpoints_preserve_static_objects():
    asset = ph.load_mjcf(ARM)
    c = by_name(asset)
    first = joints(asset, spin=0.375, extend=0.25, bend=-0.125, independent_spin=0.5)
    static = asset.collision_motion(q=first)
    for k, path in asset.collision_motion(first, first.copy()).items():
        assert not path.moving and path.turn is None and path.end is None
        np.testing.assert_array_equal(path.start, static[k].start)
    end = first.copy()
    end[asset.joint_names.index("spin")] -= 2 * np.pi
    path = asset.collision_motion(first, end)[c["pointer"].shape.key]
    assert path.moving and path.turn.angle == -2 * np.pi
    np.testing.assert_allclose(
        at(path, 0.5),
        asset.collision_motion((first + end) / 2)[c["pointer"].shape.key].start,
        atol=2e-14,
    )


@pytest.mark.parametrize("secondary,owner", [("extend", "finger"), ("bend", "wrist")])
def test_irrepresentable_descendant_motion_names_the_owner_and_both_joints(secondary, owner):
    asset = ph.load_mjcf(ARM)
    start = joints(asset)
    end = joints(asset, spin=0.5, **{secondary: 0.125})
    with pytest.raises(ValueError) as trapped:
        asset.collision_motion(start, end)
    message = str(trapped.value)
    assert owner in message and "spin" in message and secondary in message
    np.testing.assert_array_equal(start, joints(asset))
    assert asset.collision_motion(start, joints(asset, spin=0.5))


def test_unattached_visual_branch_does_not_veto_other_registered_sweeps(tmp_path):
    path = tmp_path / "no_wrist_shape.xml"
    text = ARM.read_text("utf8")
    line = next(line for line in text.splitlines() if 'name="wrist_tag"' in line)
    path.write_text(text.replace(line, ""), encoding="utf8")
    asset = ph.load_mjcf(path)
    paths = asset.collision_motion(joints(asset), joints(asset, spin=0.4, bend=0.5))
    assert paths[by_name(asset)["pointer"].shape.key].moving


def test_urdf_closed_material_surface_translates_rigidly_to_the_first_sphere(tmp_path):
    folder = tmp_path / "mesh_assets"
    shutil.copytree(ROOT / "mesh_assets", folder)
    file = folder / "mesh_arm.urdf"
    text = file.read_text("utf8")
    file.write_text(
        text.replace('name="tilt" type="revolute"', 'name="tilt" type="prismatic"'), encoding="utf8"
    )
    asset = ph.load_urdf(file)
    mesh = by_name(asset)["bracket"].shape
    p = ph.CollisionShape(ph.CollisionKey(0, 15, 0), "sphere", (29,), 1 / 64)
    paths = asset.collision_motion([0.0], [0.25])
    paths[p.key] = ph.CollisionMotion([[0.5, 0.375, -0.0625]])
    row = detect_pair(asset, mesh, p, paths)
    assert row.status == "hit" and row.mesh_round is not None
    lo, hi = row.sweep.first_t
    assert lo <= 7 / 16 <= hi and hi - lo < 1e-8
    w = row.sweep.witness
    assert set(i for i, weight in w.first_weights if weight) > set()
    assert any(
        set(i for i, weight in w.first_weights if weight) <= set(face) for face in mesh.triangles
    )
    np.testing.assert_allclose(w.first_point, [0.5, 0.375 - 1 / 64, -0.0625], atol=2e-9, rtol=0)


def test_invalid_endpoint_shape_and_precision_are_explicit():
    asset = ph.load_mjcf(ARM)
    q = joints(asset)
    for invalid in (q[:-1], np.full(len(q), np.nan), np.asarray(q, dtype=np.float32)):
        with pytest.raises((ValueError, TypeError)):
            asset.collision_motion(q, invalid)


def test_oblique_authored_hinge_frame_precedes_the_start_joint_angle(tmp_path):
    file = tmp_path / "diagonal_hinge.xml"
    text = ARM.read_text("utf8").replace(
        '<body name="rotor">',
        '<body name="rotor" pos="0.25 -0.5 0.125" euler="0 0 1.5707963267948966">',
    )
    text = text.replace(
        'name="spin" type="hinge" axis="0 0 1"', 'name="spin" type="hinge" axis="0.6 0.8 0"'
    )
    file.write_text(text, encoding="utf8")
    asset = ph.load_mjcf(file)
    q0 = joints(asset, spin=-0.4, diagonal=0.125)
    q1 = joints(asset, spin=0.65, diagonal=0.375)
    key = by_name(asset)["pointer"].shape.key
    p = asset.collision_motion(q0, q1)[key]
    np.testing.assert_allclose(p.turn.axis, [-0.8, 0.6, 0], atol=2e-14)
    np.testing.assert_allclose(p.turn.pivot_start, [0.25, -0.425, 0.225], atol=2e-14)
    np.testing.assert_allclose(p.turn.pivot_end, [0.25, -0.275, 0.425], atol=2e-14)
    for t in (0.125, 0.43, 0.91):
        np.testing.assert_allclose(
            at(p, t), asset.collision_motion(q0 + t * (q1 - q0))[key].start, rtol=0, atol=2e-14
        )


def test_real_accepted_hinge_drift_supplies_the_coordinates_to_the_asset_interval():
    asset = ph.load_urdf(ROOT / "solid_arm.urdf", gravity=(0.0, 0.0, 0.0))
    state = ph.SceneState({"robot": ph.RobotState(np.array([0.125]), np.array([1.5]))})
    inputs = ph.SceneInputs({"robot": ph.RobotInputs.zeros(1)})
    with ph.make_robot_runtime(asset.model, settings=ph.StepSettings(substeps=1)) as rt:
        result = rt.step(state, inputs, 0.25)
    assert result.report.accepted
    q = state.domains["robot"].q
    end = result.state.domains["robot"].q
    np.testing.assert_allclose(end, [0.5], rtol=0, atol=2e-14)
    key = by_name(asset)["barrel"].shape.key
    arc = asset.collision_motion(q, end)[key]
    assert arc.turn.angle == pytest.approx(0.375)
    for t in (0.25, 0.625, 1.0):
        np.testing.assert_allclose(
            at(arc, t), asset.collision_motion(q + t * (end - q))[key].start, atol=2e-14, rtol=0
        )


@pytest.mark.tyx
def test_actual_device_fk_and_collision_candidates_receive_the_asset_interval(tmp_path, tyx_mode):
    from phymeo.backends.tyx_robot_math import TyxRobotMath

    asset = ph.load_mjcf(ARM)
    cs = by_name(asset)
    a, b = cs["pointer"].shape, cs["north"].shape
    q0, q1 = joints(asset), joints(asset, spin=np.pi)
    arc = asset.collision_motion(q0, q1)
    with TyxRobotMath(asset.model, tyx_mode, cache_root=tmp_path / "fk") as fk:
        mid = fk.geometry((q0 + q1) / 2)
        report = fk.stats()
        assert report["completed_launches"] > 0
        if tyx_mode == "cuda":
            assert report["downloads"] > 0
    owner = cs["pointer"].link_index
    vertices = mid.position[owner] + np.array(cs["pointer"].local_vertices) @ mid.rotation[owner].T
    np.testing.assert_allclose(vertices, at(arc[a.key], 0.5), rtol=0, atol=2e-6)
    model = ph.CollisionModel([a, b])
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "broad") as ws:
        hit = ph.detect(
            model, {a.key: arc[a.key], b.key: arc[b.key]}, ph.CollisionPolicy(), ws
        ).pairs[0]
        stats = ws.stats()
        assert stats["completed_launches"] > 0
    expected = math.acos(1 / 6) / math.pi
    assert hit.sweep.first_t[0] <= expected <= hit.sweep.first_t[1]
    print(
        f"device={tyx_mode} fk={report['completed_launches']} broad={stats['completed_launches']} first={hit.sweep.first_t}"
    )


def test_usd_authored_child_anchor_offset_survives_the_whole_interval():
    pytest.importorskip("pxr.Usd")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1)
    rig = UsdGeom.Xform.Define(stage, "/rig")
    UsdPhysics.ArticulationRootAPI.Apply(rig.GetPrim())
    base = UsdGeom.Xform.Define(stage, "/rig/base")
    arm = UsdGeom.Xform.Define(stage, "/rig/arm")
    for part in (base, arm):
        UsdPhysics.RigidBodyAPI.Apply(part.GetPrim())
    arm.AddTranslateOp().Set(Gf.Vec3d(1, 0, 0))
    mass = UsdPhysics.MassAPI.Apply(arm.GetPrim())
    mass.CreateMassAttr(2)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
    ball = UsdGeom.Sphere.Define(stage, "/rig/arm/probe")
    ball.AddTranslateOp().Set(Gf.Vec3d(0.25, 0, 0))
    ball.CreateRadiusAttr(0.05)
    UsdPhysics.CollisionAPI.Apply(ball.GetPrim())
    mount = UsdPhysics.FixedJoint.Define(stage, "/rig/mount")
    mount.CreateBody1Rel().SetTargets([base.GetPath()])
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/rig/tilt")
    joint.CreateBody0Rel().SetTargets([base.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.5, 0, 0))
    joint.CreateBody1Rel().SetTargets([arm.GetPath()])
    joint.CreateLocalPos1Attr(Gf.Vec3f(-0.5, 0, 0))
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-150)
    joint.CreateUpperLimitAttr(150)
    asset = ph.load_usd(stage)
    (collider,) = asset.collisions
    q0 = np.array([-0.4])
    q1 = np.array([1.2])
    arc = asset.collision_motion(q0, q1)[collider.shape.key]
    np.testing.assert_allclose(arc.turn.pivot_start, [0.5, 0, 0], atol=3e-7)
    for t in (0.0, 0.25, 0.5, 0.875, 1.0):
        angle = q0[0] + t * (q1[0] - q0[0])
        np.testing.assert_allclose(
            at(arc, t), [[0.5 + 0.75 * np.cos(angle), 0.75 * np.sin(angle), 0]], rtol=0, atol=3e-7
        )


def test_rotating_closed_mesh_does_not_hide_unsupported_round_pair():
    asset = ph.load_urdf(ROOT / "mesh_assets" / "mesh_arm.urdf")
    mesh = by_name(asset)["bracket"].shape
    sphere = ph.CollisionShape(ph.CollisionKey(0, 13, 0), "sphere", (11,), 1 / 64)
    paths = asset.collision_motion([0.0], [0.25])
    assert paths[mesh.key].turn.angle == 0.25
    paths[sphere.key] = ph.CollisionMotion(paths[mesh.key].start[:1])
    row = detect_pair(asset, mesh, sphere, paths)
    assert (
        row.status == "unsupported"
        and row.sweep is None
        and row.reason == "mesh_rounding_or_rotation"
    )


def test_lost_moving_pivot_translation_is_rejected_before_publishing_false_arc(tmp_path):
    file = tmp_path / "distant.xml"
    file.write_text(
        ARM.read_text("utf8").replace(
            '<body name="base">', '<body name="base" pos="4503599627370496 0 0">'
        ),
        encoding="utf8",
    )
    asset = ph.load_mjcf(file)
    start = joints(asset)
    # A quarter metre cannot be encoded as the difference between these two
    # binary64 world pivot coordinates. Pure translational paths have no such subtraction.
    with pytest.raises(ValueError, match="pivot.*resolution|resolution.*pivot"):
        asset.collision_motion(start, joints(asset, advance=0.25, spin=0.5))
    path = asset.collision_motion(start, joints(asset, advance=0.25))[
        by_name(asset)["pointer"].shape.key
    ]
    np.testing.assert_array_equal(
        np.subtract(path.turn.pivot_end, path.turn.pivot_start), [0.25, 0, 0]
    )
    with pytest.raises(ValueError, match="spin.*representable"):
        asset.collision_motion(joints(asset, spin=-1e308), joints(asset, spin=1e308))
