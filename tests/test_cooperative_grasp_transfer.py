"""Coordinated motion begins at measured poses and uses the existing actuator speed budget."""

import numpy as np
import pytest

from phymeo.control.grasp_transfer import PairedTransport, SoloLiftHold
from phymeo.experiments.handoff_reception import (
    CooperativeReception,
    _shared_support,
    _solo_support,
)


def test_hold_requires_consecutive_solo_support_and_latches_the_actual_pose():
    hold = SoloLiftHold()
    assert hold.observe([0.0, 0.0], True) is None
    assert hold.observe([0.0, 1.0], False) is None
    assert hold.observe([0.0, 2.0], True) is None
    actual = np.array([2.0, 3.0])
    np.testing.assert_array_equal(hold.observe(actual, True), [2.0, 3.0])
    actual[:] = 100.0
    np.testing.assert_array_equal(hold.observe([8.0, 9.0], False), [2.0, 3.0])


def test_paired_transport_respects_peak_speed_and_distinct_endpoints():
    starts = np.array([[0.0, 1.0, 0.0], [-3.0, 2.0, 1.0]])
    goals = np.array([[0.0, 4.0, 0.0], [-3.0, 2.0, 5.0]])
    motion = PairedTransport(starts, goals, peak_speed=2.0)
    assert motion.duration == pytest.approx(3.0)
    for t, expected in ((-1.0, starts), (0.0, starts), (3.0, goals), (5.0, goals)):
        position, velocity = motion.sample(t)
        np.testing.assert_allclose(position, expected)
        np.testing.assert_array_equal(velocity, np.zeros_like(starts))
    pos, vel = motion.sample(1.5)
    np.testing.assert_allclose(pos, (goals + starts) / 2)
    np.testing.assert_allclose(vel, [[0.0, 1.5, 0.0], [0.0, 0.0, 2.0]])
    starts[:] = 9.0
    np.testing.assert_allclose(motion.sample(0.0)[0], [[0.0, 1.0, 0.0], [-3.0, 2.0, 1.0]])


def test_zero_displacement_pair_stays_still_and_invalid_budget_is_rejected():
    poses = np.array([[0.0, 1.0], [-1.0, 4.0]])
    m = PairedTransport(poses, poses, peak_speed=0.2)
    assert m.duration == 0.0
    p, v = m.sample(0.0)
    np.testing.assert_array_equal(p, poses)
    np.testing.assert_array_equal(v, np.zeros_like(poses))
    with pytest.raises(ValueError):
        PairedTransport(poses, poses, peak_speed=0.0)


def test_donor_must_lift_alone_before_receiver_can_try_to_grasp():
    w = 0.48
    m = dict(
        bottom_m=0.03,
        table_up_n=0.0,
        A_both_normal_n=0.5,
        A_up_n=0.47,
        B_normal_n=0.0,
        B_both_normal_n=0.0,
        B_up_n=0.0,
    )
    assert _solo_support(m, w) and not _shared_support(m, w)
    for key, val in (
        ("bottom_m", 0.01),
        ("table_up_n", 0.2),
        ("A_both_normal_n", 0.05),
        ("B_normal_n", 0.2),
    ):
        assert not _solo_support({**m, key: val}, w)
    shared = {**m, "B_normal_n": 1.0, "B_both_normal_n": 0.45, "B_up_n": 0.2, "A_up_n": 0.27}
    assert _shared_support(shared, w)
    assert not _shared_support({**shared, "A_up_n": 0.0}, w)


def test_coordinator_waits_for_observed_material_then_moves_both_actual_tools():
    a = np.array([-0.08, 0.08])
    b = np.array([0.0, 0.08])
    c = CooperativeReception(
        np.array([0, 1]),
        0.01,
        0.48,
        0.0035,
        a - np.array([0.0, 0.04]),
        b,
        b + np.array([0.0, 0.04]),
    )

    def face(y):
        return [[-0.022, y, 0.061], [-0.022, y, 0.099], [0.022, y, 0.099], [0.022, y, 0.061]]

    m = dict(
        bottom_m=0.04,
        table_up_n=0.0,
        A_both_normal_n=0.5,
        A_up_n=0.48,
        A_normal_n=1.0,
        B_normal_n=0.0,
        B_both_normal_n=0.0,
        B_up_n=0.0,
    )
    frame = dict(
        t=0.1,
        actual_tool_xz_m={"A": a.tolist(), "B": b.tolist()},
        measurements=m,
        patch_vertices_m={"B_left": face(-0.013), "B_right": face(0.013)},
        cloth_position_m=[[-0.01, 0.0, 0.08], [0.01, 0.0, 0.08]],
        cloth_velocity_m_s=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    first = c.command("lift_A", frame, a, 0.0)
    assert not first.closed and c.anchor is None
    frame["t"] += 0.01
    catch = c.command("lift_A", frame, a, 0.01)
    assert catch.closed and catch.observation["action"] == "track" and c.holder is None
    m.update(B_both_normal_n=0.4, B_normal_n=1.0, B_up_n=0.22)
    frame["t"] += 0.01
    hold = c.command("lift_A", frame, a, 0.02)
    assert hold.closed and hold.observation["action"] == "hold"
    c.join(a, b)
    shared = c.command("share", frame, a, c.motion.duration / 2)
    assert shared.closed and shared.observation["action"] == "shared_transport"
    assert np.linalg.norm(shared.donor_rate) > 0.0
    assert [e["event"] for e in c.events] == [
        "solo_lift_held",
        "receiver_address_acquired",
        "receiver_bilateral_contact",
    ]
    dropped = {**m, "A_normal_n": 0.0, "B_normal_n": 0.0}
    assert [c.lost(dropped) for _ in range(4)] == [False, False, False, True]
