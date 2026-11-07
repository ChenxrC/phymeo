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

# F02 constants come from the five original components, never from loaded model
# arrays. F03 is a passive elastic contact window, not a prescribed withdrawal.
WINDOW = {
    "duration_s": 1 / 4,
    "dt_s": (1 / 2048, 1 / 4096),
    "joint_q": (0.0, 1 / 8),
    "root_velocity_m_s": (0.0, 1 / 4, 0.0),
    "sphere_mass_kg": 1.0,
    "sphere_radius_m": 1 / 16,
    "initial_gap_m": 1 / 1024,
    "stiffness_n_per_m": 2048.0,
}
CYCLE = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])


def _rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _skew(p):
    x, y, z = p
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _probe_oracle(joints, position, rotation):
    """Five independent masses, direct serial transforms, world COM basis."""
    theta, extension = joints
    root = np.asarray(position)
    arm_r = rotation @ _rz(theta)
    tool_r = arm_r @ _rz(np.pi / 2)
    pivot = root + rotation @ np.array([1 / 4, 0, 0])
    carriage = pivot + arm_r @ np.array([1 / 2 + extension, 0, 0])
    tool = carriage + arm_r @ np.array([1 / 8, 0, 0])
    frames = {
        "base": (root, rotation, 0),
        "mount": (pivot, rotation, 0),
        "arm": (pivot, arm_r, 1),
        "carriage": (carriage, arm_r, 2),
        "tool": (tool, tool_r, 2),
        "witness": (tool + tool_r @ np.array([1 / 32, 0, 0]), tool_r, 2),
    }
    root_com = root + rotation @ np.array([1 / 20, 0, 0])
    hinge_axis = rotation[:, 2]
    slide_axis = arm_r[:, 0]

    def jacobian(point, body):
        linear, angular = np.zeros((3, 8)), np.zeros((3, 8))
        linear[:, :3] = np.eye(3)
        linear[:, 3:6] = -_skew(point - root_com)
        angular[:, 3:6] = np.eye(3)
        if body > 0:
            linear[:, 6] = np.cross(hinge_axis, point - pivot)
            angular[:, 6] = hinge_axis
        if body == 2:
            linear[:, 7] = slide_axis
        return linear, angular

    definitions = (
        ("base", 2.0, (0, 0, 0), (1 / 5, 3 / 10, 2 / 5)),
        ("mount", 1 / 2, (0, 0, 0), (1 / 50, 3 / 100, 1 / 25)),
        ("arm", 1.0, (1 / 4, 0, 0), (1 / 20, 3 / 40, 1 / 10)),
        ("carriage", 1 / 2, (0, 0, 0), (1 / 40, 1 / 40, 1 / 40)),
        ("tool", 1 / 4, (0, 0, 0), (1 / 100, 1 / 50, 3 / 100)),
    )
    parts, mass = [], np.zeros((8, 8))
    for name, weight, local_com, diagonal in definitions:
        origin, orient, body = frames[name]
        center = origin + orient @ np.array(local_com)
        inertia = orient @ np.diag(diagonal) @ orient.T
        linear, angular = jacobian(center, body)
        mass += weight * linear.T @ linear + angular.T @ inertia @ angular
        parts.append((body, weight, center, inertia, linear, angular))
    groups = []
    for body in range(3):
        entries = [p for p in parts if p[0] == body]
        weight = sum(p[1] for p in entries)
        center = sum(p[1] * p[2] for p in entries) / weight
        inertia = np.zeros((3, 3))
        for _, m, point, tensor, _, _ in entries:
            arm = point - center
            inertia += tensor + m * ((arm @ arm) * np.eye(3) - np.outer(arm, arm))
        groups.append((weight, center, inertia))
    return frames, tuple(parts), tuple(groups), mass, jacobian


@pytest.mark.parametrize(
    "joints,position,xyzw,rotation",
    [
        ((0.0, 1 / 8), (0, 0, 0), (0, 0, 0, 1), np.eye(3)),
        ((0.37, -1 / 16), (1 / 2, -1 / 4, 3 / 4), (0.5, 0.5, 0.5, 0.5), CYCLE),
        ((-0.51, 3 / 16), (-1 / 4, 1 / 2, 0), (0, 0, 0, 1), np.eye(3)),
    ],
)
def test_a02_probe_preserves_five_mass_geometry_full_inertia_and_tool_work(
    joints, position, xyzw, rotation
):
    from phymeo.dynamics.floating_robot import floating_point_jacobians
    from phymeo.solvers.floating_tree import floating_dynamics_terms

    asset = load_floating_urdf(PROBE, gravity=(0, 0, 0), collapse_fixed_joints=True)
    assert asset.link_names == ("arm", "carriage")
    assert asset.joint_names == ("hinge", "extension")
    assert asset.lower == (-1.5, -1 / 8) and asset.upper == (1.5, 1 / 4)
    assert asset.effort_limit == (16, 32) and asset.velocity_limit == (4, 2)
    assert asset.model.body_count == 3
    state = FloatingRobotState(
        np.array((*position, *xyzw, *joints), dtype=float),
        np.array([0.1, -0.2, 0.3, -0.1, 0.2, 0.1, 0.4, -0.3]),
    )
    frames, parts, groups, expected_mass, point_j = _probe_oracle(joints, position, rotation)
    geometry = floating_kinematics(asset.model, state)
    np.testing.assert_allclose(asset.model.root_com_m, [1 / 20, 0, 0], atol=ATOL, rtol=0)
    np.testing.assert_allclose(
        asset.model.root.inertia_diagonal, [[11 / 50, 71 / 200, 93 / 200]], atol=ATOL, rtol=0
    )
    observed_mass = np.r_[1 / asset.model.root.inverse_mass, asset.model.children.mass]
    observed_inertia = np.vstack(
        (asset.model.root.inertia_diagonal, asset.model.children.inertia_diagonal)
    )
    for i, (mass, center, tensor) in enumerate(groups):
        assert observed_mass[i] == pytest.approx(mass, abs=ATOL)
        np.testing.assert_allclose(geometry.com_position[i], center, atol=ATOL, rtol=0)
        np.testing.assert_allclose(
            geometry.rotation[i] @ np.diag(observed_inertia[i]) @ geometry.rotation[i].T,
            tensor, atol=ATOL, rtol=0,
        )
    for name, (point, orient, owner) in frames.items():
        assert asset.sites[name].body_index == owner
        p, r = asset.authored_link_pose(name, state.q)
        np.testing.assert_allclose(p, point, atol=ATOL, rtol=0)
        np.testing.assert_allclose(r, orient, atol=ATOL, rtol=0)
        mapped = asset.authored_to_model_points(name, [1 / 32, -1 / 64, 1 / 128])
        np.testing.assert_allclose(
            geometry.position[owner] + geometry.rotation[owner] @ mapped,
            point + orient @ np.array([1 / 32, -1 / 64, 1 / 128]),
            atol=ATOL, rtol=0,
        )
    terms = floating_dynamics_terms(asset.model, state, FloatingRobotInputs.zeros(2))
    np.testing.assert_allclose(terms.mass_matrix, expected_mass, atol=1e-11, rtol=0)
    expected_energy = sum(
        0.5 * m * np.dot(j @ state.v, j @ state.v)
        + 0.5 * (a @ state.v) @ tensor @ (a @ state.v)
        for _, m, _, tensor, j, a in parts
    )
    assert 0.5 * state.v @ terms.mass_matrix @ state.v == pytest.approx(
        expected_energy, abs=1e-11
    )
    local = np.array([[1 / 32, -1 / 64, 0.0]])
    point = frames["tool"][0] + frames["tool"][1] @ local[0]
    expected_j, expected_a = point_j(point, 2)
    linear, angular = floating_point_jacobians(
        asset.model, state, [2], asset.authored_to_model_points("tool", local)
    )
    np.testing.assert_allclose(linear[0], expected_j, atol=ATOL, rtol=0)
    np.testing.assert_allclose(angular[0], expected_a, atol=ATOL, rtol=0)
    force = rotation @ np.array([-1.0, 2.0, 0.5])
    assert float(force @ (linear[0] @ state.v)) == pytest.approx(
        float((expected_j.T @ force) @ state.v), abs=ATOL
    )
    # At zero velocity/gravity the independently assembled M and applied point
    # wrench determine acceleration without using any product-generated bias.
    rest = FloatingRobotState(state.q, np.zeros(8))
    loads = FloatingRobotInputs(
        np.array([1 / 4, -1 / 2]),
        np.vstack((np.zeros((2, 3)), force)),
        np.vstack((np.zeros((2, 3)), np.cross(point - groups[2][1], force))),
    )
    loaded = floating_dynamics_terms(asset.model, rest, loads)
    expected_applied = expected_j.T @ force + np.r_[np.zeros(6), [1 / 4, -1 / 2]]
    np.testing.assert_allclose(loaded.applied, expected_applied, atol=ATOL, rtol=0)
    np.testing.assert_allclose(
        loaded.acceleration, np.linalg.solve(expected_mass, expected_applied), atol=1e-11, rtol=0
    )
    np.testing.assert_allclose(loaded.bias, 0, atol=ATOL, rtol=0)
