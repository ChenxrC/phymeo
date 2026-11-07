"""Tests first for opt-in fixed tooling on a real free-root robot.

Only the existing public loader is imported at collection time. The first RED
must be its missing collapse_fixed_joints behavior, after the old explicit
fixed-joint rejection succeeds. The fixture constants are independent inputs.
"""

from pathlib import Path

import numpy as np
import pytest

from phymeo.builders.floating_urdf import load_floating_urdf
from phymeo.dynamics.floating_robot import FloatingRobotState, floating_kinematics
from phymeo.solvers.floating_tree import FloatingRobotInputs, step_floating_robot


ASSETS = Path(__file__).parent / "fixtures" / "urdf"
COMPOSITE = ASSETS / "floating_fixed_composite.urdf"
PROBE = ASSETS / "floating_fixed_probe.urdf"
H = 1 / 1024
ATOL = 2e-12


def _root_state():
    return FloatingRobotState(
        np.array([0, 0, 0, 0, 0, 0, 1], dtype=float),
        np.zeros(6),
    )


def test_a01_public_opt_in_keeps_real_root_mass_and_point_load_response():
    with pytest.raises(ValueError, match="fixed"):
        load_floating_urdf(COMPOSITE, gravity=(0, 0, 0))

    asset = load_floating_urdf(
        COMPOSITE, gravity=(0, 0, 0), collapse_fixed_joints=True
    )
    assert asset.joint_names == () and asset.link_names == ()
    assert asset.model.children is None and asset.model.body_count == 1
    np.testing.assert_allclose(asset.model.root.inverse_mass, [1 / 3], atol=ATOL, rtol=0)
    np.testing.assert_allclose(asset.model.root_com_m, [1 / 12, 0, 0], atol=ATOL, rtol=0)
    np.testing.assert_allclose(
        asset.model.root.inertia_diagonal,
        [[3 / 10, 53 / 120, 13 / 24]],
        atol=ATOL,
        rtol=0,
    )

    initial = _root_state()
    # F=(0,3,0) at the authored tool origin x=1/4. Relative to
    # Cx=1/12 this has torque_z=1/2; a_y=1, alpha_z=12/13.
    applied = FloatingRobotInputs(
        np.zeros(0), np.array([[0.0, 3.0, 0.0]]), np.array([[0.0, 0.0, 0.5]])
    )
    result = step_floating_robot(asset.model, initial, applied, H)
    assert result.report.accepted
    np.testing.assert_allclose(
        result.state.v, [0, H, 0, 0, 0, H * 12 / 13], atol=ATOL, rtol=0
    )
    theta = H * H * 12 / 13
    np.testing.assert_allclose(
        result.state.q,
        [
            (1 - np.cos(theta)) / 12,
            H * H - np.sin(theta) / 12,
            0,
            0,
            0,
            np.sin(theta / 2),
            np.cos(theta / 2),
        ],
        atol=ATOL,
        rtol=0,
    )
    geometry = floating_kinematics(asset.model, result.state)
    np.testing.assert_allclose(
        geometry.com_position[0], [1 / 12, H * H, 0], atol=ATOL, rtol=0
    )
    np.testing.assert_array_equal(initial.q, _root_state().q)
    np.testing.assert_array_equal(initial.v, np.zeros(6))