"""Local MJCF in the fixed-base robot asset domain, including authored units and frames."""

from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.builders.urdf import RobotAsset
from phymeo.dynamics.robot_kinematics import forward_kinematics


ARM = Path(__file__).parent / "reference_newton" / "robot" / "arm.mjcf.xml"
INERTIA = '<inertial pos="0.125 0 0" mass="2" diaginertia="1 2 2.5"/>'
MODEL = f"""
<mujoco model="local_rig">
  <compiler angle="degree" inertiafromgeom="false"/>
  <option gravity="0 -2 0"/>
  <worldbody><body name="mount" pos="1 2 3" euler="90 0 90">
    <geom name="base_mark" type="sphere" size="0.05" pos="0 0.25 0"/>
    <body name="car" pos="0.5 0 0" quat="0.7071067811865476 0 0 0.7071067811865476">
      <joint name="travel" type="slide" axis="0 0.6 0.8" range="-0.25 0.5"/>
      {INERTIA}
      <geom name="probe" type="sphere" size="0.02" pos="0.125 0 0"/>
      <geom name="display" type="box" size="4 4 4" contype="0" conaffinity="0"/>
      <body name="lever" pos="0 0 0.25">
        <joint name="spin" type="hinge" range="-45 60"/>
        {INERTIA}
        <geom name="tipbox" type="box" size="0.25 0.125 0.0625"/>
      </body>
    </body>
  </body></worldbody>
</mujoco>"""


def read(source, **kwargs):
    from phymeo.builders.mjcf import load_mjcf

    return load_mjcf(source, **kwargs)


def file(tmp_path, xml):
    path = tmp_path / "rig.xml"
    path.write_text(xml, encoding="utf-8")
    return path


def test_mjcf_match_reference_arm():
    asset = read(ARM)
    assert isinstance(asset, RobotAsset)
    assert asset.root_link == "base"
    assert asset.link_names == ("boom", "tool") and asset.joint_names == ("yaw", "slide")
    assert asset.model.joint_types == ("revolute", "prismatic")
    np.testing.assert_allclose(asset.lower, [-np.pi / 2, -0.1])
    np.testing.assert_allclose(asset.upper, [2 * np.pi / 3, 0.25])
    assert asset.effort_limit == (None, None) and asset.velocity_limit == (None, None)
    (marker,) = asset.collisions
    assert marker.name == "marker" and marker.link_index == 1 and marker.shape.radius == 0.03125
    for q, v, p, speed in [
        ([np.pi / 2, 0.125], [0.5, -0.125], [0, 0.0625, 0.15625], [-0.15625, -0.125, 0]),
        ([-np.arctan2(4, 3), -0.0625], [-0.75, 0.0625], [0.6375, -0.9125, 0.15625],
         [-0.259375, -0.253125, 0]),
    ]:
        q = np.array(q)
        port = ph.RobotPointPort(asset.model, ph.RobotState(q, np.array(v)), [1],
                                 np.asarray(marker.local_vertices))
        np.testing.assert_allclose(port.positions()[0], p, atol=2e-12, rtol=0)
        np.testing.assert_allclose(port.velocities()[0], speed, atol=2e-12, rtol=0)
        np.testing.assert_allclose(asset.collision_motion(q)[marker.shape.key].start[0], p,
                                   atol=2e-12, rtol=0)


@pytest.mark.parametrize("unit,sequence,base,axis,point", [
    ("degree", "xyz", [0.75, 2, 3], [0, -0.8, -0.6], [0.875, 1.8, 3.35]),
    ("radian", "xyz", [0.75, 2, 3], [0, -0.8, -0.6], [0.875, 1.8, 3.35]),
    ("degree", "XYZ", [1, 2, 3.25], [0.8, -0.6, 0], [1.2, 2.35, 3.125]),
])
def test_nested_euler_quaternion_frames_and_shape_owners(tmp_path, unit, sequence, base, axis, point):
    xml = MODEL.replace('angle="degree"', f'angle="{unit}" eulerseq="{sequence}"')
    if unit == "radian":
        xml = xml.replace("90 0 90", "1.5707963267948966 0 1.5707963267948966")
        xml = xml.replace("-45 60", "-0.7853981633974483 1.0471975511965976")
    asset = read(file(tmp_path, xml))
    assert asset.root_link == "mount" and asset.link_names == ("car", "lever")
    np.testing.assert_array_equal(asset.model.parents, [-1, 0])
    np.testing.assert_array_equal(asset.model.mass, [2, 2])
    np.testing.assert_array_equal(asset.model.gravity, [0, -2, 0])
    np.testing.assert_allclose(asset.model.inertia_diagonal, [[1, 2, 2.5]] * 2)
    np.testing.assert_allclose(asset.lower, [-0.25, -np.pi / 4])
    np.testing.assert_allclose(asset.upper, [0.5, np.pi / 3])
    root, ball, box = asset.collisions
    assert [(x.name, x.link_index) for x in (root, ball, box)] == [
        ("base_mark", -1), ("probe", 0), ("tipbox", 1)
    ]
    assert asset.collision_model.excluded == ((ball.shape.key, box.shape.key),)
    np.testing.assert_allclose(np.max(box.local_vertices, axis=0), [0.25, 0.125, 0.0625])
    np.testing.assert_allclose(np.min(box.local_vertices, axis=0), [-0.25, -0.125, -0.0625])
    q = np.array([0.25, np.pi / 6])
    fk = forward_kinematics(asset.model, q)
    np.testing.assert_allclose(fk.axis[0], axis, atol=2e-12)
    paths = asset.collision_motion(q)
    np.testing.assert_allclose(paths[root.shape.key].start[0], base, atol=2e-12)
    np.testing.assert_allclose(paths[ball.shape.key].start[0], point, atol=2e-12)


def test_inertial_frame_and_explicit_gravity_override(tmp_path):
    xml = MODEL.replace('diaginertia="1 2 2.5"',
                        'diaginertia="1 2 2.5" quat="0.7071067811865476 0 0 0.7071067811865476"')
    asset = read(file(tmp_path, xml), gravity=(0, 0, 0))
    np.testing.assert_allclose(asset.model.inertia_diagonal, [[2, 1, 2.5]] * 2, atol=1e-12)
    np.testing.assert_array_equal(asset.model.gravity, [0, 0, 0])


@pytest.mark.parametrize("old,new,match", [
    ('angle="degree"', 'angle="grad"', "angle"),
    ('angle="degree"', 'angle="degree" eulerseq="xqz"', "eulerseq"),
    ('inertiafromgeom="false"', 'inertiafromgeom="true"', "inertiafromgeom"),
    ('range="-0.25 0.5"', 'range="-0.25 0.5" limited="false"', "limited|bounded"),
    ('<joint name="travel"', '<joint pos="0.1 0 0" name="travel"', "pos|anchor"),
    ('<joint name="travel"', '<joint ref="0.25" name="travel"', "ref"),
    ('<joint name="spin"', '<joint type="slide" name="extra"/><joint name="spin"', "joint"),
    ('type="hinge"', 'type="ball"', "ball"),
    ('<geom name="probe"', '<geom friction="0.2 0.1 0.1" name="probe"', "friction"),
    ('<geom name="probe"', '<geom contype="2" name="probe"', "mask|contype"),
    ('type="box" size="0.25 0.125 0.0625"', 'type="cylinder" size="0.25 0.125"', "cylinder"),
    ('<worldbody>', '<default/><worldbody>', "default"),
    (INERTIA, '', "inertial"),
    ('diaginertia="1 2 2.5"', 'diaginertia="1 2 2.5" euler="0 0 30"', "inertia"),
    ('euler="90 0 90"', 'euler="90 0 90" quat="1 0 0 0"', "orient|quat|euler"),
    ('quat="0.7071067811865476 0 0 0.7071067811865476"', 'quat="0 0 0 0"', "quat"),
])
def test_mjcf_reports_unrepresented_semantics(tmp_path, old, new, match):
    with pytest.raises(ValueError, match=match):
        read(file(tmp_path, MODEL.replace(old, new)))
