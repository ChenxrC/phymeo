"""A01-A06: several real URDF owners share one accepted physical step.

No load is injected in place of contact. The first oracle uses two independently
loaded 2 kg sliders and one 1 kg sphere, with 1 mm skin overlap at 2000 N/m.
Project execution belongs to the parent serial runner.
"""

from itertools import combinations

import numpy as np

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotState
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_scene import RobotCommand

RADIUS = 1 / 16
LAW = RobotContactMaterial(2000.0, 0.0, 0.0)
H = 0.001


def _slider(path, *, mass=2.0, center=(0, 0, 0.25), axis=(1, 0, 0)):
    xyz = " ".join(str(float(value)) for value in center)
    direction = " ".join(str(float(value)) for value in axis)
    path.write_text(
        f'''<robot name="independent_slider">
<link name="base"/>
<link name="pusher">
  <inertial><mass value="{mass}"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
  </inertial>
  <collision name="round_tool"><geometry><sphere radius="{RADIUS}"/></geometry></collision>
</link>
<joint name="slide" type="prismatic">
  <parent link="base"/><child link="pusher"/>
  <origin xyz="{xyz}"/><axis xyz="{direction}"/>
  <limit lower="-0.1" upper="0.1" effort="128" velocity="10"/>
</joint>
</robot>
''',
        encoding="utf-8",
    )
    return path


def _symmetric_builder(tmp_path):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    for body, x in ((10, -0.124), (20, 0.124)):
        path = _slider(tmp_path / f"slider-{body}.urdf", center=(x, 0, 0.25))
        builder.add_urdf(
            body, path, robot_state=RobotState(np.zeros(1), np.zeros(1)), kp=[0], kd=[0]
        )
    builder.add_sphere(30, RADIUS, mass=1, position=(0, 0, 0.25))
    for first, second in combinations((10, 20, 30), 2):
        builder.pair(first, second, None if (first, second) == (10, 20) else LAW)
    return builder


def _zero_actions():
    return {body: RobotCommand([0], feedforward=[0]) for body in (10, 20)}


def test_a01_two_robots_share_one_sphere_and_one_actual_drift(tmp_path):
    with _symmetric_builder(tmp_path).build(backend="host") as scene:
        before = scene.observe()
        result = scene.step(H, actions=_zero_actions())
        left, right = (result.robots[0, body] for body in (10, 20))
        row = scene.rigid_rows[0, 30]
        assert before.step_number == 0 and before.time_s == 0
        assert result.step_number == 1 and result.time_s == H and result.load_time_s == 0
        assert set(result.robots) == {(0, 10), (0, 20)}
        np.testing.assert_allclose(left.contact_effort, [-2], atol=2e-12, rtol=0)
        np.testing.assert_allclose(right.contact_effort, [2], atol=2e-12, rtol=0)
        np.testing.assert_allclose(left.v, [-H], atol=2e-14, rtol=0)
        np.testing.assert_allclose(right.v, [H], atol=2e-14, rtol=0)
        np.testing.assert_allclose(left.q, [-H * H], atol=2e-16, rtol=0)
        np.testing.assert_allclose(right.q, [H * H], atol=2e-16, rtol=0)
        np.testing.assert_allclose(result.rigid.velocity[row], 0, atol=2e-14, rtol=0)
        np.testing.assert_allclose(result.rigid.position[row], [0, 0, 0.25], atol=2e-14)
        np.testing.assert_allclose(result.rigid.angular_velocity, 0, atol=2e-14, rtol=0)
        np.testing.assert_allclose(
            2 * (left.v[0] + right.v[0]) + result.rigid.velocity[row, 0],
            0,
            atol=2e-14,
            rtol=0,
        )
        assert len(result.loads.contacts) == 2
        np.testing.assert_allclose(
            [pair.force_n for pair in result.loads.contacts], [2, 2], atol=2e-12, rtol=0
        )
        reading, geometry = scene.observe_scene()
        assert reading.step_number == result.step_number and geometry.time_s == H
        assert len(geometry.poses) == 3
        assert set(geometry.poses) == {shape.key for shape in geometry.model.shapes}
        owners = {
            owner.body
            for record in scene.contact_events.active
            for owner in (record.first_owner, record.second_owner)
        }
        assert owners == {10, 20, 30}
