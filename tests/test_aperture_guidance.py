"""Acquire only inside finite fronts, but steer a planar hand toward an axially reachable address."""

import numpy as np
import pytest
from test_exterior_grasp import fixture

from phymeo.control.material_grasp import MaterialEdgeTarget, ParallelJawWindow


@pytest.mark.parametrize("side", [1.0, -1.0])
def test_receiving_hand_retargets_a_reachable_part_of_the_same_ungrasped_edge(side):
    x = np.array([[0.08, 0.0, 0.09], [0.05, side * 0.024, 0.11]])
    frame, controller = fixture(x)
    donor = np.asarray(frame["actual_tool_xz_m"]["A"])
    controller.command("lift_A", frame, donor, 0.0)
    assert side * controller.standoff.target.evaluate(x)[1] > 0.013
    frame["t"] += 0.01
    motion = controller.command("lift_A", frame, donor, 0.01)
    assert not motion.closed and controller.anchor is None
    assert abs(motion.observation["material_point_m"][1]) <= 0.013 + 1e-14
    assert motion.observation["action"] == "approach"
    frame["cloth_position_m"] = x + [0, side * 0.04, 0]
    frame["t"] += 0.01
    waiting = controller.command("lift_A", frame, donor, 0.02)
    assert not waiting.closed and controller.anchor is None
    assert waiting.observation["action"] == "await_aperture"
    assert "exterior_point_m" in waiting.observation


@pytest.mark.parametrize("rotated", [False, True])
def test_axial_window_guidance_clips_the_edge_independently_of_lateral_capture(rotated):
    x = np.array([[0.08, 0, 0.1], [0.05, 0.020, 0.1]])
    frame, _ = fixture(x)
    Q = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) if rotated else np.eye(3)
    shift = np.array([-0.2, 0.4, -0.3]) if rotated else np.zeros(3)
    points = x @ Q.T + shift
    faces = [
        np.array(frame["patch_vertices_m"]["B_" + side]) @ Q.T + shift for side in ("left", "right")
    ]
    window = ParallelJawWindow(*faces)
    assert window.acquire(points, [0, 1]) is None
    preferred = MaterialEdgeTarget((0, 1), 1.0)
    actual = window.approach_target(points, [0, 1], preferred=preferred, axial_clearance=0.003)
    assert actual.fraction == pytest.approx(0.5, abs=1e-12)
    reverse = window.approach_target(points, [1, 0], axial_clearance=0.003)
    np.testing.assert_allclose(reverse.evaluate(points), actual.evaluate(points), atol=1e-14)
    already = MaterialEdgeTarget((0, 1), 0.25)
    assert (
        window.approach_target(points, [0, 1], preferred=already, axial_clearance=0.003) is already
    )
    assert window.approach_target(points, [0, 1], axial_clearance=0.02) is None
    with pytest.raises(ValueError, match="clearance"):
        window.approach_target(points, [0, 1], axial_clearance=float("nan"))
