"""Two force-limited planar hands share a measured material presentation before capture."""

import numpy as np
import pytest

from phymeo.dynamics.robot_kinematics import forward_kinematics, point_jacobians
from phymeo.experiments.handoff_grippers import grippers_for


def arguments(sign):
    return dict(
        donor=np.array([-0.08 * sign, 0.14]),
        donor_velocity=[0.0, 0.0],
        receiver=np.array([0.12 * sign, 0.06]),
        receiver_velocity=[0.0, 0.0],
        material=[-0.04 * sign, 0.08],
        waypoint=[-0.04 * sign, 0.08],
        waypoint_velocity=[0.0, 0.0],
        held_height=0.14,
        loaded=True,
        closing=False,
        dt=0.01,
        peak_speed=0.3,
        peak_acceleration=2.0,
    )


def test_paired_rendezvous_is_mirrored_and_keeps_the_donor_height():
    from phymeo.control.planar_presentation import PlanarPresentation

    outputs = [PlanarPresentation().update(**arguments(sign)) for sign in (1, -1)]
    a, va, b, vb, d = outputs[0]
    np.testing.assert_allclose(d["donor_goal_xz_m"], [0, 0.14], atol=1e-13)
    np.testing.assert_allclose(d["receiver_waypoint_xz_m"], [0.04, 0.08], atol=1e-13)
    assert a[0] > -0.08 and b[0] < 0.12 and va[0] > 0 > vb[0]
    for v, w in zip(outputs[0][:4], outputs[1][:4], strict=True):
        np.testing.assert_allclose(w, v * np.array([-1, 1]), atol=1e-14)
    assert max(np.linalg.norm(va), np.linalg.norm(vb)) <= 0.01 * 2 + 1e-14
    assert a[1] == 0.14


def test_unloaded_empty_scene_has_no_presentation_and_live_material_is_observed_each_tick():
    from phymeo.control.planar_presentation import PlanarPresentation

    policy = PlanarPresentation()
    inp = arguments(1)
    assert policy.update(**{**inp, "loaded": False}) is None
    assert policy.goal is None
    policy.update(**inp)
    shifted = policy.update(**{**inp, "donor": [-0.06, 0.15], "waypoint": [0.01, 0.10]})
    np.testing.assert_allclose(shifted[-1]["receiver_waypoint_xz_m"], [0.07, 0.10], atol=1e-12)
    paused = policy.update(**{**inp, "donor": [-0.055, 0.15], "closing": True})
    np.testing.assert_allclose(paused[-1]["donor_goal_xz_m"], [-0.055, 0.15])
    retry = policy.update(**{**inp, "donor": [-0.047, 0.15], "loaded": False})
    assert (
        retry[-1]["action"] == "hold"
        and retry[-1]["donor_goal_xz_m"] == paused[-1]["donor_goal_xz_m"]
    )


@pytest.mark.parametrize("role", ["A", "B"])
def test_real_palm_cannot_move_its_wrist_in_the_jaw_axis_direction(role):
    grip = grippers_for("articulated-palm")[role]
    expected = None
    for offset in ((0.0, 0.0), (0.1, -0.12), (-0.08, 0.15)):
        q = grip.initial.q.copy()
        q[:2] += offset
        position = forward_kinematics(grip.model, q)
        J, _ = point_jacobians(grip.model, q, [grip.tool_link], grip.tool_point[None, :])
        np.testing.assert_allclose(J[0, 1], 0, atol=2e-14, rtol=0)
        tip = (
            position.position[grip.tool_link] + position.rotation[grip.tool_link] @ grip.tool_point
        )
        assert abs(tip[1]) < 2e-14
        if expected is None:
            expected = position.position[grip.tool_link, 1]
        assert position.position[grip.tool_link, 1] == pytest.approx(expected, abs=2e-14)


def test_joint_receiver_rate_is_published_before_optional_structural_checks():
    from phymeo.experiments.handoff_reception import CooperativeReception

    receiver = np.array([0.0, 0.08])
    donor = np.array([-0.08, 0.12])
    c = CooperativeReception(
        np.array([0, 1]),
        0.01,
        0.48,
        0.0035,
        np.array([-0.08, 0.06]),
        receiver,
        np.array([0.02, 0.18]),
        paired_presentation=True,
    )
    xz = np.array([[-0.02, 0.06], [-0.02, 0.10], [0.02, 0.10], [0.02, 0.06]])
    frame = dict(
        t=0.10,
        actual_tool_xz_m={"A": donor, "B": receiver},
        actual_tool_velocity_xz_m_s={"A": [0.0, 0.0], "B": [0.0, 0.0]},
        measurements=dict(
            bottom_m=0.04,
            table_up_n=0.0,
            A_both_normal_n=0.5,
            A_up_n=0.48,
            B_normal_n=0.0,
            B_both_normal_n=0.0,
        ),
        cloth_position_m=[[0.07, 0.0, 0.10], [0.09, 0.0, 0.13]],
        cloth_velocity_m_s=np.zeros((2, 3)),
        patch_vertices_m={
            n: np.c_[xz[:, 0], np.full(4, y), xz[:, 1]]
            for n, y in (("B_left", -0.013), ("B_right", 0.013))
        },
    )
    c.command("lift_A", frame, donor, 0.0)
    frame["t"] += 0.01
    motion = c.command("lift_A", frame, donor, 0.01)
    assert motion.observation["presentation"]["action"] == "present"
    np.testing.assert_array_equal(
        motion.observation["tool_velocity_command_m_s"], motion.receiver_rate
    )
