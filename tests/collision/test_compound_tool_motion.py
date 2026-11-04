"""Swept compound joint motion of a finite link, including downstream translation."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from test_articulated_hull_cloth import case, run

import phymeo as ph
from phymeo.dynamics.robot_kinematics import forward_kinematics
from phymeo.geometry.articulated_hull_clearance import link_hull_motion, require_cloth_clearance


def compound(types, axes, origins, local_center, q0, q1):
    n = len(types)
    model = ph.RobotModel(
        np.arange(n) - 1,
        tuple(types),
        np.array(origins, float),
        np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
        np.array(axes, float),
        np.ones(n),
        np.zeros((n, 3)),
        np.ones((n, 3)),
        np.zeros(3),
    )
    q0, q1 = np.array(q0, float), np.array(q1, float)
    points = np.array(list(product((-1 / 64, 1 / 64), repeat=3))) + local_center
    spec = ph.LinkHullTool("tool", "arm", n - 1, points, ph.NormalLayer(0.125, 64))
    hull = ph.CollisionShape(ph.CollisionKey(0, 1, 0), "hull", tuple(range(8)))
    start, end = (forward_kinematics(model, q) for q in (q0, q1))
    return link_hull_motion(model, spec, hull, q0, q1, start, end)


def face_at(x, y, z):
    return ph.CollisionMotion(
        np.array([[x - 0.07, y - 0.06, z], [x + 0.07, y - 0.06, z], [x, y + 0.08, z]])
    )


@pytest.mark.parametrize("slide", [False, True])
def test_equal_or_safe_endpoints_cannot_mask_a_compound_path_crossing(slide):
    if slide:
        # Half-turn followed by a true simultaneous radial slide.
        support, slack = compound(
            ["revolute", "prismatic"],
            [[0, 0, 1], [1, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
            [0.25, 0, 0],
            [0, 0],
            [np.pi, 0.5],
        )
        crossing = (0, 0.5, 0)
        far = (0, 0.9, 0)
    else:
        # Rz(pi*t) @ (ex/2 + Ry(pi*t) @ ex/2), with nonparallel hinges.
        support, slack = compound(
            ["revolute", "revolute"],
            [[0, 0, 1], [0, 1, 0]],
            [[0, 0, 0], [0.5, 0, 0]],
            [0.5, 0, 0],
            [0, 0],
            [np.pi, np.pi],
        )
        crossing = (0, 0.5, -0.5)
        far = (0, 0.9, -0.5)
    face = ph.CollisionShape(ph.CollisionKey(0, 2, 0), "triangle", (0, 1, 2))
    mid, err = support.pose(Fraction(1, 2))
    assert float(err) < 1e-12
    np.testing.assert_allclose(mid.start.mean(0), crossing, atol=2e-15)
    for t in (Fraction(0), Fraction(1)):
        pose, _ = support.pose(t)
        assert np.max(abs(pose.start[:, 1])) < 0.02
    with pytest.raises(ph.SolverFailure, match="clearance|subdivision"):
        require_cloth_clearance(support, face, face_at(*crossing), 1 / 128, slack)
    require_cloth_clearance(support, face, face_at(*far), 1 / 128, slack)


def test_loaded_two_joint_wrist_exchanges_effort_with_all_free_cloth_nodes():
    program, state, command = case("double", speed=0.1)
    obs = program.observe_hull_tools(state)
    assert obs["contacts"] and np.linalg.norm(obs["efforts"]["arm"]) > 0.25
    initial = state.domains["arm"].q.copy()
    for _ in range(4):
        step = run(program, state, command)
        assert step.report.accepted, step.report.reason
        state = step.state
    assert np.all(abs(state.domains["arm"].q - initial) > 1e-6)
    assert np.linalg.norm(state.domains["sheet"].velocity) > 0.01
