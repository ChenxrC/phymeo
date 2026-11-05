"""Accepted-step feedback and actual motion for a bounded, asset-independent gripper."""

from itertools import combinations, product
from pathlib import Path

import numpy as np
import pytest

from phymeo.builders.mimic_urdf import load_mimic_urdf
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotState
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_payload_scene import PadFace

SOURCE = Path(__file__).parent / "reference_newton" / "robot" / "mimic_jaws.urdf"
H = 0.001


def _scene():
    builder = ContactSceneBuilder()
    pads = tuple(
        PadFace.named(side, side + "_rubber", 0, sign, tuple(product((-.005, .005), repeat=2)))
        for side, sign in (("left", 1), ("right", -1))
    )
    builder.add_urdf(
        10, SOURCE, loader=load_mimic_urdf,
        robot_state=RobotState(np.zeros(2), np.zeros(2)),
        kp=[2700, 6000], kd=[95, 35], effort_limit=[40, 8], pads=pads,
    )
    builder.add_box(20, (.02, .02, .02), mass=.1, position=(.25, -.125, .25))
    builder.add_sphere(30, .015, mass=.03, position=(.33, -.125, .245))
    builder.add_box(40, (.12, .12, .01), mass=None, position=(.25, -.125, .22))
    for a, b in combinations((10, 20, 30, 40), 2):
        law = RobotContactMaterial(1200, 1.2, .65, .04) if (a, b) == (10, 20) else (
            RobotContactMaterial(2200, 3, .45, .04) if b == 40 and a != 10 else
            RobotContactMaterial(6000, 16, .35, .04)
        )
        builder.pair(a, b, law)
    return builder.build()


def _controller(scene):
    from phymeo.control.gripper_acquisition import GripperAcquisition

    return GripperAcquisition(
        scene.assets[0, 10], robot=(0, 10), payload_key=scene.body_keys[0, 20][0],
        pad_groups=tuple((scene.shape_key(10, side, side + "_rubber"),) for side in ("left", "right")),
        joint_indices=(1,), open_q=(0,), closed_q=(.018,), speed_limit=(.02,),
        effort_limit=(40, 8), force_target_N=2.0, force_band_N=.25,
        contact_min_N=1.0, dwell_s=.020, loss_dwell_s=.010, stop_speed=(.002,), timeout_s=1.5,
    )


def test_close_proposal_tracks_accepted_steps_without_teleport_or_repeat_accumulation():
    with _scene() as scene:
        controller = _controller(scene)
        before = scene.observe()
        first = controller.propose(before, dt_s=H)
        repeat = controller.propose(before, dt_s=H)
        assert first.phase == "closing" and not first.contact_ready
        np.testing.assert_array_equal(first.bilateral_N, [0, 0])
        np.testing.assert_allclose(first.command.target_q, [0, .000020], atol=1e-18, rtol=0)
        np.testing.assert_allclose(first.command.target_velocity, [0, .02], atol=1e-18, rtol=0)
        np.testing.assert_array_equal(first.command.effort_limit, [40, 8])
        np.testing.assert_array_equal(repeat.command.target_q, first.command.target_q)
        np.testing.assert_array_equal(repeat.command.target_velocity, first.command.target_velocity)
        assert scene.observe().step_number == 0 and scene.last_frame is None
        np.testing.assert_array_equal(scene.observe().robots[0, 10].q, before.robots[0, 10].q)
        with pytest.raises(ValueError):
            scene.step(-H, actions={10: first.command})
        retry = controller.propose(scene.observe(), dt_s=H)
        np.testing.assert_array_equal(retry.command.target_q, first.command.target_q)
        after = scene.step(H, actions={10: retry.command})
        # The command is a target, not a state write. The real inertia advances by less.
        assert 0 < after.robots[0, 10].q[1] < first.command.target_q[1]
        assert after.robots[0, 10].v[1] > 0
        second = controller.propose(after, dt_s=H)
        np.testing.assert_allclose(second.command.target_q, [0, .000040], atol=1e-18, rtol=0)
        again = controller.propose(after, dt_s=H)
        np.testing.assert_array_equal(second.command.target_q, again.command.target_q)
        assert scene.observe().step_number == 1 and scene.observe().time_s == H
