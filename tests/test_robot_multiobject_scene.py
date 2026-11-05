"""One physical input frame couples a URDF and both free bodies.

This catches omitted free/free contact, two sequential advances of the arm,
and integrating body 2 after body 1 has already moved. The oracle is hand
derived from two 1 mm compressions at 2000 N/m, not from runtime helpers.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotState
from phymeo.runtime.contact_scene import ContactSceneBuilder

PUSHER = Path(__file__).parent / "reference_newton" / "robot" / "multiobject_pusher.urdf"
LAW = RobotContactMaterial(2000.0, 0.0, 0.0)
H = 0.001


def chain_scene():
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    builder.add_urdf(
        10,
        PUSHER,
        robot_state=RobotState(np.array([0.0]), np.array([0.0])),
        kp=[0.0],
        kd=[0.0],
    )
    builder.add_sphere(20, 0.0625, mass=1.0, position=(0.124, 0.0, 0.25))
    builder.add_sphere(30, 0.0625, mass=1.0, position=(0.248, 0.0, 0.25))
    builder.add_box(40, (0.5, 0.5, 0.03125), mass=None, position=(0, 0, -0.03125))
    for first, second in combinations((10, 20, 30, 40), 2):
        builder.pair(first, second, LAW)
    return builder.build(backend="host")


def test_same_entry_three_body_chain_preserves_both_reactions_and_one_drift():
    with chain_scene() as scene:
        # Import only after the builder's missing normal entry has been exercised.
        from phymeo.runtime.robot_scene import RobotCommand

        before = scene.observe()
        result = scene.step(H, actions={10: RobotCommand(target_q=[0.0], feedforward=[0.0])})
        robot = result.robots[0, 10]
        a, b, table = (scene.rigid_rows[0, key] for key in (20, 30, 40))

        assert before.step_number == 0 and before.time_s == 0.0
        assert result.step_number == 1 and result.time_s == H and result.load_time_s == 0.0
        np.testing.assert_allclose(robot.drive_effort, [0.0], atol=1e-13, rtol=0)
        np.testing.assert_allclose(robot.contact_effort, [-2.0], atol=1e-12, rtol=0)
        np.testing.assert_allclose(robot.v, [-0.001], atol=1e-14, rtol=0)
        np.testing.assert_allclose(robot.q, [-0.000001], atol=1e-16, rtol=0)
        np.testing.assert_allclose(result.rigid.velocity[a], [0, 0, 0], atol=1e-14, rtol=0)
        np.testing.assert_allclose(result.rigid.velocity[b], [0.002, 0, 0], atol=1e-14, rtol=0)
        np.testing.assert_allclose(result.rigid.position[a], [0.124, 0, 0.25], atol=1e-14, rtol=0)
        np.testing.assert_allclose(
            result.rigid.position[b], [0.248002, 0, 0.25], atol=1e-14, rtol=0
        )
        np.testing.assert_array_equal(result.rigid.velocity[table], np.zeros(3))
        np.testing.assert_array_equal(result.rigid.angular_velocity, 0)
        np.testing.assert_allclose(
            2 * robot.v[0] + result.rigid.velocity[a, 0] + result.rigid.velocity[b, 0],
            0.0,
            atol=1e-14,
            rtol=0,
        )

        assert len(result.loads.contacts) == 2
        assert sorted(pair.force_n for pair in result.loads.contacts) == pytest.approx(
            [2.0, 2.0], abs=1e-12
        )
        np.testing.assert_allclose(result.loads.inputs.rigid.force[a], [0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(result.loads.inputs.rigid.force[b], [2, 0, 0], atol=1e-12)
        reading, geometry = scene.observe_scene()
        assert reading.step_number == result.step_number and geometry.time_s == result.time_s
        assert set(geometry.poses) == {shape.key for shape in geometry.model.shapes}
        assert len(geometry.poses) == 4
