"""The pregrasp servo must move from the measured tool, even if the material edge moves fast."""

import numpy as np
import pytest

from phymeo.experiments.handoff_reception import CooperativeReception


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_distant_moving_material_cannot_jump_the_open_receiver(direction):
    dt = 0.01
    weight = 0.48
    donor_home = np.array([0.2, 0.22])
    donor = donor_home + [0.0, 0.04]
    receiver = donor + [direction * 0.16, 0.0]
    transfer = donor_home + [direction * 0.025, 0.12]
    control = CooperativeReception(
        np.array([0, 1]), dt, weight, 0.0035, donor_home, receiver, transfer
    )
    offsets = np.array([[-0.022, -0.019], [-0.022, 0.019], [0.022, 0.019], [0.022, -0.019]])
    pad = np.column_stack((receiver[0] + offsets[:, 0], np.zeros(4), receiver[1] + offsets[:, 1]))
    x = [[donor[0], 0.0, donor[1] + delta] for delta in (-0.012, 0.012)]
    frame = dict(
        t=0.25,
        actual_tool_xz_m={"A": donor, "B": receiver},
        actual_tool_velocity_xz_m_s={"A": [0.0, 0.0], "B": [0.0, 0.0]},
        measurements=dict(
            bottom_m=0.04,
            table_up_n=0.0,
            A_normal_n=1.0,
            A_both_normal_n=0.5,
            A_up_n=weight,
            B_normal_n=0.0,
            B_link_normal_n=0.0,
            B_both_normal_n=0.0,
            B_up_n=0.0,
        ),
        patch_vertices_m={"B_left": pad - [0, 0.013, 0], "B_right": pad + [0, 0.013, 0]},
        cloth_position_m=x,
        cloth_velocity_m_s=[[direction * 2.0, 0.0, 1.0]] * 2,
    )
    control.command("lift_A", frame, donor, 0.0)
    frame["t"] += dt
    command = control.command("lift_A", frame, donor, dt)

    # The existing transfer reference is a rest-to-rest cubic of duration 0.55 seconds.
    travel = np.linalg.norm(transfer - donor_home)
    peak_speed = 1.5 * travel / 0.55
    peak_acceleration = 6 * travel / 0.55**2
    assert not command.closed and control.anchor is None
    assert np.linalg.norm(command.receiver - receiver) <= dt * peak_speed
    assert command.receiver_rate is not None
    assert np.linalg.norm(command.receiver_rate) <= dt * peak_acceleration + 1e-12
    assert direction * (command.receiver[0] - receiver[0]) < 0.0


def test_tracking_reference_preserves_incoming_velocity_and_brakes_an_overspeed_tool():
    from phymeo.control.grasp_transfer import bounded_approach

    position = np.array([1.0, -2.0, 3.0])
    velocity = np.array([0.0, 2.0, 0.0])
    endpoint, rate = bounded_approach(
        position, velocity, position, np.zeros(3), 0.25, peak_speed=1.0, peak_acceleration=2.0
    )
    np.testing.assert_allclose(rate, [0.0, 1.5, 0.0])
    np.testing.assert_allclose(endpoint, position + 0.125 * (velocity + rate))
    with pytest.raises(ValueError):
        bounded_approach(
            position, velocity, position, [0, 0], 0.25, peak_speed=1, peak_acceleration=2
        )


@pytest.mark.parametrize("cloth_speed,clear", [(0.0, True), (25.0, False)])
def test_pregrasp_checks_the_moving_face_against_the_actual_registered_shaft(cloth_speed, clear):
    from test_cloth_link_contact import make_scene

    import phymeo as ph
    from phymeo.experiments.handoff_reception import GripMotion

    program, state, _ = make_scene(height=0.5)
    state.domains["cloth"].velocity[:, 2] = cloth_speed

    class PrismaticTool:
        @staticmethod
        def position(robot):
            return np.array([0.0, 0.5 + robot.q[0]])

        @staticmethod
        def velocity(robot):
            return np.array([0.0, robot.v[0]])

        @staticmethod
        def command(robot, target, aperture, dt, *, tool_velocity_m_s):
            return ph.RobotInputs.zeros(1), np.array([target[1] - 0.5])

    start = np.array([0.0, 0.5])
    control = CooperativeReception(
        np.array([0, 1]), 0.01, 0.48, 0.0035, start, start, start + [0.1, 0.1]
    )
    motion = GripMotion(
        start,
        start + [0.0, 0.0001],
        False,
        receiver_rate=np.array([0.0, 0.02]),
        observation={"action": "approach"},
    )
    before = state.domains["arm"].q.copy()
    _, joints = control.drive_approach(
        "arm", PrismaticTool(), state, motion, 0.0, program.cloth_links
    )
    assert motion.observation["predicted_structural_clear"] is clear
    assert motion.observation["approach_fraction"] == (1.0 if clear else 0.0)
    if not clear:
        assert motion.observation["blocked_approach_candidates"][-1]["pair"] == ["beam", "0"]
        np.testing.assert_equal(joints, before)
    np.testing.assert_equal(state.domains["arm"].q, before)
