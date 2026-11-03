"""Fixed-base URDF assets through the public robot, point and collision operators."""

from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.dynamics.robot_kinematics import forward_kinematics

ARM = Path(__file__).parent / "reference_newton" / "robot" / "arm.urdf"
INERTIAL = """
  <inertial><origin xyz="0 0 0" rpy="0 0 1.5707963267948966"/><mass value="2"/>
  <inertia ixx="1" ixy="0" ixz="0" iyy="2" iyz="0" izz="2.5"/></inertial>"""
SLIDE = """
  <joint name="rail" type="prismatic"><parent link="pedestal"/><child link="carriage"/>
  <limit lower="-1" upper="1" effort="9" velocity="4"/></joint>"""
CARRIAGE = f'<link name="carriage">{INERTIAL}</link>'


def load(source, **kwargs):
    from phymeo.builders.urdf import load_urdf

    return load_urdf(source, **kwargs)


def write_asset(tmp_path, body):
    path = tmp_path / "asset.urdf"
    path.write_text(f'<robot name="arbitrary">{body}</robot>', encoding="utf-8")
    return path


def native_arm():
    h = np.sqrt(0.5)
    return ph.RobotModel(
        np.array([-1, 0]),
        ("revolute", "prismatic"),
        np.array([[0.25, -0.5, 0.125], [0.5, 0, 0]]),
        np.array([[0.0, 0, 0, 1], [0, 0, h, h]]),
        np.array([[0.0, 0, 1], [1, 0, 0]]),
        np.array([1, 0.5]),
        np.array([[0.25, 0, 0], [0.125, -0.0625, 0.03125]]),
        np.full((2, 3), 0.1),
        np.array([0.0, 0, -9.81]),
    )


POSES = [
    ([0.0, 0.0], [0.0, 0.0], [0.8125, -0.375, 0.15625], [0, 0, 0]),
    ([np.pi / 2, 0.125], [0.5, -0.125], [0, 0.0625, 0.15625], [-0.15625, -0.125, 0]),
    (
        [-np.arctan2(4, 3), -0.0625],
        [-0.75, 0.0625],
        [0.6375, -0.9125, 0.15625],
        [-0.259375, -0.253125, 0],
    ),
]


@pytest.mark.parametrize("q,v,expected,velocity", POSES)
def test_import_matches_native_fk_and_velocity(q, v, expected, velocity):
    asset = load(ARM)
    assert isinstance(asset.model, ph.RobotModel)
    assert asset.name == "rp_arm" and asset.root_link == "base"
    assert asset.link_names == ("boom", "tool") and asset.joint_names == ("yaw", "slide")
    assert asset.model.joint_types == ("revolute", "prismatic")
    np.testing.assert_allclose(asset.lower, [-np.pi / 2, -0.1])
    np.testing.assert_allclose(asset.upper, [2 * np.pi / 3, 0.25])
    np.testing.assert_allclose(asset.effort_limit, [12, 8])
    np.testing.assert_allclose(asset.velocity_limit, [3, 1])
    (marker,) = asset.collisions
    assert (marker.name, marker.link_name, marker.link_index) == ("marker", "tool", 1)
    assert marker.shape.kind == "sphere" and marker.shape.radius == 0.03125
    np.testing.assert_array_equal(marker.local_vertices, [[0.125, -0.0625, 0.03125]])
    q, v = np.asarray(q), np.asarray(v)
    state = ph.RobotState(q, v)
    reference = native_arm()
    for field in ("parents", "joint_position", "joint_axis", "mass", "com", "inertia_diagonal"):
        np.testing.assert_allclose(
            getattr(asset.model, field), getattr(reference, field), atol=1e-15
        )
    for model in (asset.model, reference):
        port = ph.RobotPointPort(
            model, state, [marker.link_index], np.asarray(marker.local_vertices)
        )
        np.testing.assert_allclose(port.positions()[0], expected, atol=2e-12, rtol=0)
        np.testing.assert_allclose(port.velocities()[0], velocity, atol=2e-12, rtol=0)
    motion = asset.collision_motion(q)
    np.testing.assert_allclose(motion[marker.shape.key].start[0], expected, atol=2e-12, rtol=0)


def test_collision_box_and_fixed_root_reach_public_query(tmp_path):
    root = """<link name="pedestal">
      <collision name="fixed_ball"><origin xyz="0.25 0 0"/>
        <geometry><sphere radius="0.125"/></geometry></collision>
      <collision name="guard"><origin xyz="2 0 0" rpy="0 0 1.5707963267948966"/>
        <geometry><box size="2 0.5 0.25"/></geometry></collision></link>"""
    moving = CARRIAGE.replace(
        "</link>",
        """
      <visual><geometry><mesh filename="uncached.stl"/></geometry></visual>
      <collision><geometry><sphere radius="0.125"/></geometry></collision></link>""",
    )
    asset = load(write_asset(tmp_path, SLIDE + moving + root))
    fixed, box, ball = asset.collisions
    assert (fixed.link_index, fixed.link_name, fixed.shape.key.body) == (-1, "pedestal", 0)
    assert box.shape.kind == "hull" and box.link_index == -1
    assert ball.name and (ball.link_index, ball.link_name, ball.shape.key.body) == (
        0,
        "carriage",
        1,
    )
    np.testing.assert_allclose(asset.model.inertia_diagonal, [[2.0, 1.0, 2.5]], atol=1e-12)
    np.testing.assert_equal(asset.model.joint_axis, [[1, 0, 0]])
    assert isinstance(asset.collision_model, ph.CollisionModel)
    at_zero = asset.collision_motion(np.zeros(1))
    corners = at_zero[box.shape.key].start
    np.testing.assert_allclose(np.min(corners, axis=0), [1.75, -1, -0.125], atol=1e-12)
    np.testing.assert_allclose(np.max(corners, axis=0), [2.25, 1, 0.125], atol=1e-12)
    assert len(np.unique(corners, axis=0)) == 8
    with ph.CollisionWorkspace(asset.collision_model) as workspace:
        for q, wanted, gap in [
            (-0.125, "separated", 0.125),
            (0.0, "touching", 0.0),
            (0.0625, "overlap", -0.0625),
        ]:
            paths = asset.collision_motion(np.array([q]))
            np.testing.assert_array_equal(paths[fixed.shape.key].start, [[0.25, 0, 0]])
            result = ph.detect(
                asset.collision_model, paths, ph.CollisionPolicy(margin=0.2), workspace
            )
            pair = next(
                p for p in result.pairs if (p.first, p.second) == (fixed.shape.key, ball.shape.key)
            )
            assert pair.status == wanted
            assert pair.witness.gap_m == pytest.approx(gap)


def test_joint_axes_and_topological_order(tmp_path):
    root = '<link name="pedestal"/>'
    leaf = f'<link name="tip">{INERTIAL}</link>'
    second = """<joint name="hinge" type="revolute">
      <parent link="carriage"/><child link="tip"/><origin xyz="0.4 0 0"/>
      <axis xyz="0 0 1"/><limit lower="-0.4" upper="0.8" effort="3" velocity="2"/></joint>"""
    first = SLIDE.replace(
        "<limit",
        """
      <origin xyz="1 2 3" rpy="1.5707963267948966 0 1.5707963267948966"/>
      <axis xyz="0 0.6 0.8"/><limit""",
    )
    asset = load(write_asset(tmp_path, second + leaf + first + root + CARRIAGE), gravity=(0, -1, 0))
    assert asset.link_names == ("carriage", "tip") and asset.joint_names == ("rail", "hinge")
    np.testing.assert_array_equal(asset.model.parents, [-1, 0])
    np.testing.assert_array_equal(asset.model.gravity, [0, -1, 0])
    geometry = forward_kinematics(asset.model, np.array([0.25, 0.0]))
    np.testing.assert_allclose(geometry.axis[0], [0.8, 0, 0.6], atol=1e-12)
    np.testing.assert_allclose(geometry.position, [[1.2, 2, 3.15], [1.2, 2.4, 3.15]], atol=1e-12)


@pytest.mark.parametrize(
    "inertia",
    [
        INERTIAL.replace('ixy="0"', 'ixy="0.2"'),
        INERTIAL.replace("1.5707963267948966", "0.35"),
    ],
)
def test_unrepresentable_tensor_is_rejected_in_its_named_link(tmp_path, inertia):
    path = write_asset(
        tmp_path, '<link name="pedestal"/>' + CARRIAGE.replace(INERTIAL, inertia) + SLIDE
    )
    with pytest.raises(ValueError, match=r"(?i)(carriage.*inertia|inertia.*carriage)"):
        load(path)


@pytest.mark.parametrize(
    "old,new,match",
    [
        ('type="prismatic"', 'type="fixed"', "fixed"),
        ('type="prismatic"', 'type="floating"', "floating"),
        ("<limit ", '<mimic joint="other"/><limit ', "mimic"),
        ("<limit ", '<axis xyz="0 0 0"/><limit ', "axis"),
        ('upper="1"', 'upper="-2"', "limit"),
        ('velocity="4"', 'velocity="nan"', "velocity"),
        ('<child link="carriage"/>', '<child link="missing"/>', "missing"),
        ('<child link="carriage"/>', "", "child"),
        ('<link name="carriage">', "<link>", "name"),
        ('<joint name="rail"', "<joint", "name"),
        ('velocity="4"', 'velocity="4" acceleration="3"', "acceleration"),
        ('radius="0.125"', 'radius="0.125" scale="2"', "scale"),
        ('<mass value="2"/>', "", "mass"),
        ("<inertia ", "<dummy ", "inertia|dummy"),
        (INERTIAL, "", "inertial"),
        ('<geometry><sphere radius="0.125"/></geometry>', "", "geometry"),
        ('<sphere radius="0.125"/>', '<mesh filename="part.stl"/>', "mesh"),
        ('<sphere radius="0.125"/>', '<sphere radius="0"/>', "radius"),
        ('<sphere radius="0.125"/>', '<box size="1 -1 1"/>', "size"),
        ('<sphere radius="0.125"/>', '<sphere radius="1"/><box size="1 1 1"/>', "geometry"),
    ],
)
def test_rejects_unsupported_or_invalid_urdf(tmp_path, old, new, match):
    shape = '<collision name="body"><geometry><sphere radius="0.125"/></geometry></collision>'
    text = '<link name="pedestal"/>' + CARRIAGE.replace("</link>", shape + "</link>") + SLIDE
    with pytest.raises(ValueError, match=match):
        load(write_asset(tmp_path, text.replace(old, new)))


@pytest.mark.parametrize(
    "append,match",
    [
        ('<link name="carriage"/>', "duplicate"),
        ('<link name="island"/>', "root|connected"),
        (SLIDE.replace('name="rail"', 'name="other"'), "parent|child"),
    ],
)
def test_ambiguous_or_disconnected_tree_is_rejected(tmp_path, append, match):
    with pytest.raises(ValueError, match=match):
        load(write_asset(tmp_path, '<link name="pedestal"/>' + CARRIAGE + SLIDE + append))


@pytest.mark.tyx
def test_imported_fk_runs_tyx(tyx_mode, tmp_path):
    from phymeo.backends.tyx_robot_math import TyxRobotMath

    asset = load(ARM)
    (marker,) = asset.collisions
    with TyxRobotMath(asset.model, tyx_mode, cache_root=tmp_path) as kernels:
        for coords, speed, point, expected_v in POSES:
            q, v = np.array(coords), np.array(speed)
            geometry = kernels.geometry(q)
            row = marker.link_index
            local = np.asarray(marker.local_vertices)
            world_point = geometry.position[row] + geometry.rotation[row] @ local[0]
            jac, _ = kernels.jacobians(q, [row], local)
            np.testing.assert_allclose(world_point, point, rtol=0, atol=2e-6)
            np.testing.assert_allclose(jac[0] @ v, expected_v, rtol=0, atol=2e-6)
        assert kernels.stats()["completed_launches"] > 0
        if tyx_mode == "cuda":
            assert kernels.stats()["downloads"] > 0
