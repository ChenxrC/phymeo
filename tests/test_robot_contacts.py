"""Independent mechanics checks for joint stops and body collisions."""

import numpy as np
import pytest

import phymeo as ph


def slider(*, mass=1.0, axis=(1.0, 0.0, 0.0), base=(0.0, 0.0, 0.0), kind="prismatic"):
    return ph.RobotModel(
        np.array([-1]),
        (kind,),
        np.array([base]),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([axis]),
        np.array([mass]),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )


def test_joint_stops_are_forces_with_spring_energy_not_position_clamps():
    assert hasattr(ph, "JointLimits"), "public joint-stop model is missing"
    bounds = ph.JointLimits(np.array([-0.2]), np.array([0.3]), 1000.0, 8.0)
    s = ph.RobotState(np.array([0.32]), np.array([0.5]))
    row = bounds.evaluate(s)
    np.testing.assert_allclose(row["effort"], [-24.0], atol=1e-12)
    assert row["potential_j"] == pytest.approx(0.2)
    assert row["violation"][0] == pytest.approx(0.02)
    assert s.q[0] == 0.32
    opening = bounds.evaluate(ph.RobotState(s.q, -s.v))
    assert opening["effort"][0] == pytest.approx(-20.0)
    low = bounds.evaluate(ph.RobotState(np.array([-0.21]), np.array([-0.25])))
    assert low["effort"][0] == pytest.approx(12.0)
    assert not np.any(bounds.evaluate(ph.RobotState(np.zeros(1), np.ones(1)))["effort"])
    assert not bounds.lower.flags.writeable
    for args in (
        (np.array([0.0]), np.array([0.0]), 1.0, 0.0),
        (np.array([0.0]), np.array([np.inf]), 1.0, 0.0),
        (np.array([0.0]), np.array([1.0]), -1.0, 0.0),
    ):
        with pytest.raises((TypeError, ValueError)):
            ph.JointLimits(*args)
    with pytest.raises(ValueError):
        bounds.evaluate(ph.RobotState(np.zeros(2), np.zeros(2)))


def test_bounded_ik_does_not_hide_unreachable_targets_or_change_the_seed():
    assert hasattr(ph, "JointLimits"), "public joint-stop model is missing"
    model = slider()
    limits = ph.JointLimits(np.array([-0.2]), np.array([0.3]), 1000.0, 8.0)
    seed = np.array([0.32])  # permitted physical soft-stop compression
    result = ph.solve_point_ik(
        model, seed, 0, np.zeros(3), np.array([0.5, 0.0, 0.0]), limits=limits
    )
    assert not result.converged
    assert result.status == "stalled"
    assert result.configuration[0] == 0.3
    assert result.error_m == pytest.approx(0.2)
    assert seed[0] == 0.32
    good = ph.solve_point_ik(model, seed, 0, np.zeros(3), np.array([0.1, 0.0, 0.0]), limits=limits)
    assert good.converged and good.error_m < 1e-7
