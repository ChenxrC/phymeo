"""Material-relative receiver commands carry no forces or constraints on the sheet."""

import numpy as np
import pytest

from phymeo.control.material_grasp import ParallelJawWindow
from phymeo.experiments.handoff_grippers import grippers_for


def jaws(y, orientation=None, offset=None):
    corners = np.array([[-2, y, -1], [-2, y, 1], [2, y, 1], [2, y, -1]], dtype=float)
    if orientation is not None:
        corners = corners @ orientation.T + offset
    return corners


def test_clipped_segment_capture_and_fixed_material_weights_follow_the_motion():
    window = ParallelJawWindow(jaws(-0.5), jaws(0.5))
    x = np.array([[9, 9, 9], [0, -1, 0], [0, 1, 0]], dtype=float)
    original = x.copy()
    target = window.acquire(x, (1, 2))
    assert target.nodes == (1, 2)
    assert target.fraction == pytest.approx(0.5)
    x[1] += [1, 0, 0]
    x[2] += [0, 0, 2]
    np.testing.assert_allclose(target.evaluate(x), [0.5, 0, 1])
    np.testing.assert_allclose(target.evaluate(x - original), [0.5, 0, 1])
    assert window.acquire(original + [3, 0, 0], (1, 2)) is None
    assert window.acquire(original + [0, 2, 0], (1, 2)) is None


def test_plane_footprint_clipping_selects_a_real_interval_and_commutes_with_rotation():
    angle = 0.76
    rot = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    offset = np.array([3.0, -2.0, 1.0])
    pts = np.array([[-3, 0.2, 0.3], [3, 0.2, -0.3], [3, 2, 0.3]])
    w = ParallelJawWindow(jaws(-0.5), jaws(0.5))
    a = w.acquire(pts, (0, 1, 2))
    rw = ParallelJawWindow(jaws(-0.5, rot, offset), jaws(0.5, rot, offset))
    b = rw.acquire(pts @ rot.T + offset, (0, 1, 2))
    assert a.nodes == b.nodes == (0, 1)
    assert b.fraction == pytest.approx(a.fraction)
    np.testing.assert_allclose(b.evaluate(pts @ rot.T + offset), rot @ a.evaluate(pts) + offset)


def test_transport_lead_is_bounded_by_both_finite_faces_and_preserves_the_nominal_normal():
    w = ParallelJawWindow(jaws(-0.5), jaws(0.5))
    material = np.array([0.0, 0.06, 0.0])
    goal = np.array([3.0, 0.2, 4.0])
    np.testing.assert_allclose(w.limit_transport(material, goal, clearance=0.25), [0.45, 0.2, 0.60])
    nearby = np.array([0.1, 0.2, -0.05])
    np.testing.assert_allclose(w.limit_transport(material, nearby, clearance=0.25), nearby)
    with pytest.raises(ValueError):
        w.limit_transport(material, goal, clearance=1.1)


@pytest.mark.parametrize("layout", ["cartesian", "articulated"])
def test_velocity_tracking_changes_actual_effort_with_original_limits(layout):
    g = grippers_for(layout, roles=("B",))["B"]
    state = g.initial
    target = g.position(state)
    baseline, old_goal = g.command(state, target, 0.0, 0.01)
    moving, goal = g.command(state, target, 0.0, 0.01, tool_velocity_m_s=np.array([0.2, 0.1]))
    np.testing.assert_array_equal(goal, old_goal)
    assert np.linalg.norm(moving.effort[:2] - baseline.effort[:2]) > 0.01
    assert np.all(abs(moving.effort) <= g.effort_limits)
    np.testing.assert_array_equal(moving.effort[2:], baseline.effort[2:])
