"""A waiting receiver can stage beyond the whole moving sheet, then capture a finite interior site."""

import numpy as np
import pytest

from phymeo.control.material_grasp import ParallelJawWindow
from phymeo.experiments.handoff_reception import CooperativeReception


def fixture(x, center=0.0):
    pad = np.array(
        [[-0.022, 0, -0.019], [-0.022, 0, 0.019], [0.022, 0, 0.019], [0.022, 0, -0.019]]
    ) + [center, 0, 0.1]
    faces = {"B_left": pad - [0, 0.013, 0], "B_right": pad + [0, 0.013, 0]}
    frame = dict(
        t=0.2,
        actual_tool_xz_m={"A": [-0.1, 0.1], "B": [center, 0.1]},
        actual_tool_velocity_xz_m_s={"A": [0.0, 0.0], "B": [0.0, 0.0]},
        patch_vertices_m=faces,
        cloth_position_m=x,
        cloth_velocity_m_s=np.zeros_like(x),
        measurements=dict(
            bottom_m=0.04,
            table_up_n=0.0,
            A_normal_n=1.0,
            A_both_normal_n=0.5,
            A_up_n=0.48,
            B_normal_n=0.0,
            B_link_normal_n=0.0,
            B_both_normal_n=0.0,
            B_up_n=0.0,
        ),
    )
    control = CooperativeReception(
        np.array([0, 1]),
        0.01,
        0.48,
        0.0035,
        np.array([-0.1, 0.04]),
        np.array([center, 0.1]),
        np.array([0.0, 0.2]),
    )
    return frame, control


@pytest.mark.parametrize("side", [1.0, -1.0])
def test_open_receiver_stages_from_its_side_without_claiming_a_grasp(side):
    x = np.array([[0.0, 0, 0.08], [0.0, 0, 0.12], [side * 0.015, 0, 0.10]])
    frame, c = fixture(x, side * 0.09)
    frame["measurements"]["table_up_n"] = 0.4
    cmd = c.command("lift_A", frame, np.array([-0.1, 0.10]), 0.0)
    assert not cmd.closed and c.anchor is None and c.solo.position is None
    assert cmd.observation["action"] == "stage"
    assert side * (cmd.receiver[0] - side * 0.09) < 0
    direction = np.array(cmd.observation["standoff_direction"])
    target = np.array(cmd.observation["exterior_point_m"])
    window = ParallelJawWindow(
        frame["patch_vertices_m"]["B_left"], frame["patch_vertices_m"]["B_right"]
    )
    np.testing.assert_allclose(direction, [side, 0, 0])
    assert np.min((target - x) @ direction) >= window.outer_radius + c.clearance - 1e-12


@pytest.mark.parametrize(
    "offset,captured", [(0.014, True), (0.021, False), (-0.014, True), (-0.021, False)]
)
def test_acquisition_uses_the_clear_interior_of_both_finite_fronts(offset, captured):
    x = np.array([[offset, 0, 0.095], [offset, 0, 0.105]])
    frame, c = fixture(x)
    c.command("lift_A", frame, np.array([-0.1, 0.10]), 0.0)
    frame["t"] += 0.01
    cmd = c.command("lift_A", frame, np.array([-0.1, 0.10]), 0.01)
    assert cmd.closed is captured and (c.anchor is not None) is captured
    assert [e["event"] for e in c.events].count("receiver_address_acquired") == int(captured)


def test_collinear_receiver_brakes_and_keeps_the_edge_unclaimed():
    x = np.array([[0.0, 0.0, 0.085], [0.0, 0.0, 0.115]])
    frame, controller = fixture(x)
    frame["actual_tool_velocity_xz_m_s"]["B"] = [0.0, 0.02]
    cmd = controller.command("lift_A", frame, np.array([-0.1, 0.10]), 0.0)
    assert cmd.observation["reason"] == "no_exterior_direction"
    assert not cmd.closed and controller.anchor is None
    assert 0 <= cmd.receiver_rate[1] < 0.02


def test_side_margin_preserves_axial_reach_and_clips_a_segment_with_both_ends_outside():
    x = np.array([[-0.04, 0.0125, 0.1], [0.04, 0.0125, 0.1]])
    frame, _ = fixture(x)
    window = ParallelJawWindow(
        frame["patch_vertices_m"]["B_left"], frame["patch_vertices_m"]["B_right"]
    )
    target = window.acquire(x, [0, 1], clearance=0.0035)
    np.testing.assert_allclose(target.evaluate(x), [0.0, 0.0125, 0.1], atol=1e-15)
    assert window.acquire(x + [0.0, 0.001, 0.0], [0, 1], clearance=0.0035) is None
    with pytest.raises(ValueError, match="clearance"):
        window.acquire(x, [0, 1], clearance=window.radius)
