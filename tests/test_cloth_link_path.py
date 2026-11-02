"""The finite shaft guard covers motions between the two recorded poses."""

from math import pi

import numpy as np
import pytest
from test_cloth_link_contact import make_scene

import phymeo as ph


def check_interval(program, state, end):
    program.cloth_links.check_path(
        state, end, program._link_geometry(state), program._link_geometry(end)
    )


def test_shaft_and_face_can_translate_together_with_small_positive_clearance():
    program, state, _ = make_scene(height=1 / 16)
    last = ph.SceneState(
        {
            "cloth": ph.ParticleState(
                state.domains["cloth"].position + [0, 0, 0.25], np.zeros((3, 3))
            ),
            "arm": ph.RobotState(np.array([0.25]), np.zeros(1)),
        }
    )
    # Every material point translates equally. The exact axis-to-plane distance stays 1/16.
    check_interval(program, state, last)


@pytest.mark.parametrize("angle,clear", [(-pi, True), (2 * pi, False)])
def test_hinge_uses_its_joint_arc_when_both_endpoint_axes_are_above_cloth(angle, clear):
    program, state, _ = make_scene(revolute=True, height=0.20)
    last = ph.SceneState(
        {"cloth": state.domains["cloth"], "arm": ph.RobotState(np.array([angle]), np.zeros(1))}
    )
    # The local shaft radii are [0.25, 0.75]. For the negative half turn
    # every point stays at z>=0.20. The positive full turn passes through z=0.
    if clear:
        check_interval(program, state, last)
    else:
        with pytest.raises(ph.SolverFailure, match="axis"):
            check_interval(program, state, last)


def test_open_approach_uses_both_physical_radii_and_the_contact_layer():
    # At 1/4 the centerline is clear but the calibrated physical layer is active.
    close, nearby, _ = make_scene(height=0.25)
    assert close.cloth_links.approach_blocker("arm", nearby, nearby) == ("beam", "0")
    far, free, _ = make_scene(height=0.5)
    assert far.cloth_links.approach_blocker("arm", free, free) is None
    entering = ph.SceneState(
        {"cloth": free.domains["cloth"], "arm": ph.RobotState(np.array([-0.25]), np.zeros(1))}
    )
    assert far.cloth_links.approach_blocker("arm", free, entering) == ("beam", "0")
    with pytest.raises(ValueError, match="robot"):
        far.cloth_links.approach_blocker("missing", free, free)
