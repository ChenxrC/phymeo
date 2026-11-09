"""Frame and admission checks independent of a humanoid-specific asset."""

import numpy as np
import pytest

from phymeo.builders.floating_mjcf import load_floating_mjcf
from phymeo.dynamics.floating_robot import FloatingRobotState, floating_kinematics
from phymeo.dynamics.rotation import quaternion_matrix

XML = """<mujoco model="principal-tree"><compiler angle="radian"/>
<worldbody><body name="root"><freejoint/>
<inertial mass="2" pos="0 0 .1" diaginertia=".2 .3 .4"/>
<body name="arm" pos="1 0 0">
<joint name="hinge" axis="0 0 1" range="-2 2"/>
<inertial mass="1" pos=".2 0 0" quat=".9238795325112867 0 0 .3826834323650898" diaginertia=".1 .2 .25"/>
<geom type="capsule" fromto="0 0 0 .4 0 0" size=".05"/>
</body></body></worldbody>
<actuator><motor joint="hinge" ctrlrange="-10 10"/></actuator></mujoco>"""


def asset(tmp_path, text=XML):
    path = tmp_path / "robot.xml"
    path.write_text(text)
    return load_floating_mjcf(path)


@pytest.mark.parametrize("angle", (0.0, 0.37, -0.7))
def test_authored_pose_and_capsule_survive_principal_frame(tmp_path, angle):
    robot = asset(tmp_path)
    state = FloatingRobotState(np.array([0, 0, 0, 0, 0, 0, 1, angle]), np.zeros(7))
    geometry = floating_kinematics(robot.model, state)
    c, s = np.cos(angle), np.sin(angle)
    expected_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    principal = quaternion_matrix(np.array([robot.principal_to_authored_xyzw[0]]))[0]
    np.testing.assert_allclose(geometry.rotation[1] @ principal.T, expected_rotation, atol=1e-14)
    np.testing.assert_allclose(geometry.position[1], [1, 0, 0], atol=1e-14)
    motion = robot.collision_motion(state.q)
    shape = robot.collisions[0].shape
    expected = np.array([[1, 0, 0], [1 + 0.4 * c, 0.4 * s, 0]])
    np.testing.assert_allclose(motion[shape.key].start, expected, atol=1e-14)
    np.testing.assert_allclose(robot.model.children.com[0] @ principal.T, [0.2, 0, 0], atol=1e-14)
    assert shape.kind == "capsule" and shape.radius == 0.05


@pytest.mark.parametrize(
    "old,new",
    (
        ('angle="radian"', 'angle="degree"'),
        ('ctrlrange="-10 10"', 'ctrlrange="-10 20"'),
        ('axis="0 0 1"', 'axis="0 0 0"'),
    ),
)
def test_unsupported_declarations_are_rejected(tmp_path, old, new):
    with pytest.raises(ValueError):
        asset(tmp_path, XML.replace(old, new))


@pytest.mark.parametrize("attribute", ("frictionloss",))
def test_implicit_actuator_terms_are_not_silently_dropped(tmp_path, attribute):
    modified = XML.replace('<joint name="hinge"', f'<joint {attribute}=".01" name="hinge"')
    with pytest.raises(ValueError, match=attribute):
        asset(tmp_path, modified)


def test_implicit_degree_units_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="radian"):
        asset(tmp_path, XML.replace('<compiler angle="radian"/>', ""))


def test_named_physical_geometries_resolve_through_the_shared_scene(tmp_path):
    import phymeo as ph

    text = XML.replace('<geom type="capsule"', '<geom name="sole" type="capsule"')
    text = text.replace(
        "</body></body>",
        '<geom name="toe" type="sphere" size=".03"/>'
        '<geom type="box" size=".02 .02 .02"/></body></body>',
    )
    robot = asset(tmp_path, text)
    assert [row.name_is_authored for row in robot.collisions] == [True, True, False]
    builder = ph.ContactSceneBuilder(gravity=(0, 0, -9.81))
    builder.add_floating_urdf(
        1,
        tmp_path / "robot.xml",
        loader=load_floating_mjcf,
        robot_state=FloatingRobotState(np.array([0, 0, 2, 0, 0, 0, 1, 0.0]), np.zeros(7)),
        kp=[0.0],
        kd=[0.0],
        effort_limit=[10.0],
    )
    with builder.build(backend="host") as scene:
        assert scene.shape_key(1, "arm", "sole") != scene.shape_key(1, "arm", "toe")
        with pytest.raises(ValueError, match="missing or ambiguous"):
            scene.shape_key(1, "arm", "arm-collision-2")
