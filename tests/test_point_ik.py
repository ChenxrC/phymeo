"""Public position IK returns control targets without modifying the simulated tree."""

import numpy as np
import pytest

import phymeo as ph
from phymeo.dynamics.robot_kinematics import forward_kinematics


def tree():
    # A rotated base, revolute/prismatic ancestors, and one independent finger.
    return ph.RobotModel(
        np.array([-1, 0, 0]),
        ("revolute", "prismatic", "prismatic"),
        np.array([[0.3, -0.4, 0.2], [0.25, 0.0, 0.0], [0.0, 0.07, 0.0]]),
        np.array([[0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.ones(3),
        np.zeros((3, 3)),
        np.ones((3, 3)),
        np.zeros(3),
    )


def world(model, q, point):
    fk = forward_kinematics(model, q)
    return fk.position[1] + fk.rotation[1] @ point


def test_offset_tool_on_rotated_branched_tree():
    model = tree()
    q = np.array([0.2, 0.03, 0.81])
    point = np.array([0.12, 0.01, 0.04])
    # Fixed base cyclically maps local (x,y,z) to world (z,x,y).
    target = model.joint_position[0] + [0.04, -0.01, 0.25 + 0.17 + 0.12]
    result = ph.solve_point_ik(model, q, 1, point, target)
    assert result.converged and result.status == "converged"
    np.testing.assert_allclose(world(model, result.configuration, point), target, atol=1e-7)
    assert result.error_m <= 1e-7 and result.iterations > 0
    assert result.configuration[2] == q[2]
    np.testing.assert_array_equal(q, [0.2, 0.03, 0.81])
    assert not result.configuration.flags.writeable


def test_selected_joint_only_and_bounded_unreachable_target():
    model = tree()
    seed = np.array([0.3, 0.01, 0.85])
    point = np.array([0.1, 0.0, 0.0])
    target = world(model, seed + [0.0, 0.08, 0.0], point)
    result = ph.solve_point_ik(model, seed, 1, point, target, active_joints=(1,))
    assert result.converged
    np.testing.assert_array_equal(result.configuration[[0, 2]], seed[[0, 2]])
    wrong_plane = ph.solve_point_ik(
        model, seed, 1, point, target + [0.2, 0.0, 0.0], active_joints=(1,)
    )
    assert not wrong_plane.converged and wrong_plane.status == "stalled"
    assert 0.199999 < wrong_plane.error_m < 0.200001
    one_step = ph.solve_point_ik(
        model, seed, 1, point, target, active_joints=(1,), max_iterations=1, max_step=0.01
    )
    assert one_step.status == "iteration_limit" and one_step.iterations == 1
    assert abs(one_step.configuration[1] - seed[1]) <= 0.0100000001


def test_seed_already_solves_and_invalid_inputs():
    model = tree()
    seed = np.zeros(3)
    point = np.zeros(3)
    target = world(model, seed, point)
    solve = ph.solve_point_ik
    result = solve(model, seed, 1, point, target)
    assert result.converged and result.iterations == 0
    for options in (
        dict(active_joints=(0, 0)),
        dict(active_joints=(True,)),
        dict(active_joints=()),
        dict(max_iterations=0),
        dict(damping=0.0),
        dict(max_step=np.inf),
    ):
        with pytest.raises((TypeError, ValueError)):
            solve(model, seed, 1, point, target, **options)
    with pytest.raises((TypeError, ValueError)):
        solve(model, seed, -1, point, target)
