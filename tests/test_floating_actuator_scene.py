"""Analytic free-root damping response through the real shared-scene boundary."""

import numpy as np

import phymeo as ph
from phymeo.builders.floating_mjcf import load_floating_mjcf
from phymeo.runtime._scene_tangent_ports import _robot_port

XML = """<mujoco model="damped-free-root"><compiler angle="radian"/>
<worldbody><body name="root"><freejoint/>
<inertial mass="2" pos="0 0 0" diaginertia="2 2 2"/>
<body name="child"><joint name="spin" axis="0 0 1" range="-2 2"
armature=".5" damping="3"/>
<inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>
<geom type="sphere" size=".05"/>
</body></body></worldbody>
<actuator><motor joint="spin" ctrlrange="-10 10"/></actuator></mujoco>"""


def build(tmp_path):
    path = tmp_path / "damped.xml"
    path.write_text(XML)
    builder = ph.ContactSceneBuilder(gravity=(0, 0, 0))
    state = ph.FloatingRobotState(
        np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=float),
        np.array([0, 0, 0, 0, 0, 0, 1], dtype=float),
    )
    builder.add_floating_urdf(
        1, path, loader=load_floating_mjcf, robot_state=state, kp=[0], kd=[0], effort_limit=[10]
    )
    return builder.build(backend="host")


def command():
    return ph.FloatingRobotCommand(ph.RobotCommand([0], feedforward=[0]))


def test_shared_scene_applies_passive_damping_once_with_root_reaction(tmp_path):
    # Physical z block [[3,1],[1,1.5]], step z block [[3,1],[1,1.8]].
    # (M+hD) delta_v = [0,-.3] => omega=3/44, joint_rate=35/44.
    with build(tmp_path) as scene:
        scene.step(0.1, actions={1: command()})
        row = scene.observe().robots[0, 1]
        np.testing.assert_allclose(row.v[5:], [3 / 44, 35 / 44], atol=2e-14)
        np.testing.assert_allclose(row.q[-1], 3.5 / 44, atol=2e-14)
        assert abs(3 * row.v[5] + row.v[6] - 1) < 1e-14
        assert np.all(row.drive_effort == 0)
        assert np.all(row.contact_effort == 0)


def test_tangent_response_uses_step_metric_without_changing_physical_mass(tmp_path):
    with build(tmp_path) as scene:
        participant = scene._participants[0, 1]
        prepared = participant.prepare(participant.initial, command(), 0.1)
        port = _robot_port(scene, (0, 1), prepared)
        assert port.mass[-1, -1] == 1.5
        assert port.response_metric[-1, -1] == 1.8
        velocity = port.free_velocity(np.zeros(7), 0.1)
        np.testing.assert_allclose(velocity[5:], [3 / 44, 35 / 44], atol=2e-14)
        assert scene.observe().step_number == 0
