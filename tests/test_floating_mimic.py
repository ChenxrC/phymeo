"""UNEXECUTED F2 tests-first: free-root affine mechanisms, not a friction fix.

Only existing modules are imported during collection. The first node validates
both old loader contracts before requesting the new floating mimic API.
All masses, authored frames and oracle inputs below are frozen test inputs.
"""

from importlib import import_module
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from phymeo.builders.floating_urdf import load_floating_urdf
from phymeo.builders.mimic_urdf import load_mimic_urdf
from phymeo.core.robot import RobotState
from phymeo.dynamics.floating_robot import (
    FloatingRobotState,
    floating_kinematics,
    floating_point_jacobians,
)
from phymeo.solvers.floating_tree import FloatingRobotInputs, floating_dynamics_terms


ASSETS = Path(__file__).parent / "fixtures" / "urdf"
JAWS = ASSETS / "floating_mimic_jaws.urdf"
ELBOW = ASSETS / "floating_mimic_elbow.urdf"
H = 1 / 1024
ATOL = 2e-12
ROOT_COM = np.array([1 / 32, -1 / 64, 0.0])
CYCLE = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])


def _load(path, *, gravity=(0.0, 0.0, 0.0), collapse_fixed_joints=False):
    module = "phymeo.builders.floating_mimic_urdf"
    try:
        api = import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module:
            raise
        pytest.fail("public load_floating_mimic_urdf is not implemented yet")
    return api.load_floating_mimic_urdf(
        path, gravity=gravity, collapse_fixed_joints=collapse_fixed_joints
    )


def _state(u=0.0, rate=0.0, *, quaternion=(0.0, 0.0, 0.0, 1.0)):
    return FloatingRobotState(
        np.array([0.0, 0, 0, *quaternion, u]),
        np.array([0.0, 0, 0, 0, 0, 0, rate]),
    )


def _assert_load_receipt(loads, count, full_speed):
    for name in (
        "acceleration_full",
        "actuator_effort_full",
        "contact_effort_full",
        "constraint_effort_full",
    ):
        value = getattr(loads, name)
        assert value.shape == (6 + count,), name
        assert np.isfinite(value).all(), name
        assert not value.flags.writeable, name
    np.testing.assert_allclose(loads.constraint_effort_full[:6], 0.0, atol=ATOL, rtol=0)
    assert loads.constraint_power_W == pytest.approx(
        loads.constraint_effort_full @ full_speed, abs=ATOL
    )
    assert loads.constraint_power_W == pytest.approx(sum(row.constraint_power_W for row in loads.row_forces), abs=ATOL)
    assert abs(loads.constraint_power_W) < ATOL
    assert all(abs(row.constraint_power_W) < ATOL for row in loads.row_forces)


def test_a01_public_floating_mimic_preserves_mass_and_root_reaction():
    # Both preconditions use existing APIs and the same real positive-mass URDF.
    with pytest.raises(ValueError, match="mimic"):
        load_floating_urdf(JAWS, gravity=(0, 0, 0))
    fixed = load_mimic_urdf(JAWS, gravity=(0, 0, 0))
    np.testing.assert_array_equal(fixed.matrix, [[1.0], [-1.0]])
    np.testing.assert_allclose(
        fixed.full_state(RobotState(np.zeros(1), np.zeros(1))).q, [0.0, 0.004]
    )

    asset = _load(JAWS)
    assert asset.tree.model.body_count == 3
    assert asset.tree.joint_names == ("driver", "follower")
    assert asset.joint_names == ("driver",)
    assert asset.independent_indices == (0,)
    np.testing.assert_array_equal(asset.matrix, [[1.0], [-1.0]])
    np.testing.assert_array_equal(asset.constraint_rows, [[1.0, 1.0]])
    np.testing.assert_allclose(asset.offset_vector, [0.0, 0.004], atol=ATOL)
    initial = _state()
    full = asset.full_state(initial)
    assert full.q.shape == (9,) and full.v.shape == (8,)
    np.testing.assert_array_equal(asset.reduce_state(full).q, initial.q)
    np.testing.assert_array_equal(asset.reduce_state(full).v, initial.v)
    result = asset.step(initial, effort=np.array([1.0]), dt_s=H)
    acceleration = np.array([1 / 14, 0, 0, 0, 0, 0, 5 / 14, -5 / 14])
    np.testing.assert_allclose(result.loads.acceleration_full, acceleration, atol=ATOL, rtol=0)
    np.testing.assert_allclose(result.full_state.v, H * acceleration, atol=ATOL, rtol=0)
    np.testing.assert_allclose(result.state.v, H * acceleration[:7], atol=ATOL, rtol=0)
    np.testing.assert_allclose(result.state.q[:3], [H * H / 14, 0, 0], atol=ATOL, rtol=0)
    np.testing.assert_allclose(result.full_state.q[7:], [H * H * 5 / 14, 0.004 - H * H * 5 / 14])
    np.testing.assert_allclose(
        result.loads.actuator_effort_full, [0, 0, 0, 0, 0, 0, 1, 0], atol=ATOL, rtol=0
    )
    np.testing.assert_array_equal(result.loads.contact_effort_full, np.zeros(8))
    np.testing.assert_allclose(
        result.loads.constraint_effort_full, [0, 0, 0, 0, 0, 0, -4 / 7, -4 / 7], atol=ATOL, rtol=0
    )
    assert result.loads.multiplier_force == pytest.approx(-4 / 7, abs=ATOL)
    assert [(r.driver, r.follower) for r in result.loads.row_forces] == [("driver", "follower")]
    assert result.loads.row_forces[0].driver_effort == pytest.approx(-4 / 7, abs=ATOL)
    _assert_load_receipt(result.loads, 2, full.v)
    np.testing.assert_array_equal(initial.q, _state().q)
    np.testing.assert_array_equal(initial.v, _state().v)


@pytest.mark.parametrize("quaternion,axis", [((0, 0, 0, 1), (1, 0, 0)), ((0.5,) * 4, (0, 1, 0))])
def test_a02_sixteen_real_steps_preserve_affine_state_and_linear_momentum(quaternion, axis):
    asset, state = _load(JAWS), _state(quaternion=quaternion)
    direction = np.array(axis, dtype=float)
    for step in range(1, 17):
        entry = asset.full_state(state)
        result = asset.step(state, effort=np.array([1.0]), dt_s=H)
        state, full = result.state, result.full_state
        speed = step * H
        displacement = step * (step + 1) * H * H / 2
        np.testing.assert_allclose(state.v[:3], direction * speed / 14, atol=ATOL, rtol=0)
        np.testing.assert_allclose(state.v[3:6], 0, atol=ATOL, rtol=0)
        np.testing.assert_allclose(state.q[:3], direction * displacement / 14, atol=ATOL, rtol=0)
        np.testing.assert_allclose(state.q[7:], [displacement * 5 / 14], atol=ATOL, rtol=0)
        np.testing.assert_allclose(full.q[8], -full.q[7] + 0.004, atol=ATOL, rtol=0)
        np.testing.assert_allclose(full.v[7], -full.v[6], atol=ATOL, rtol=0)
        # Original body velocities, not a projected generalized-momentum proxy.
        v_root = full.v[:3]
        momentum = 2 * v_root + (v_root + direction * full.v[6]) + 2 * (v_root + direction * full.v[7])
        np.testing.assert_allclose(momentum, 0, atol=ATOL, rtol=0)
        _assert_load_receipt(result.loads, 2, entry.v)


def _graph_asset(tmp_path, chain):
    document = ET.parse(JAWS)
    root = document.getroot()
    link = ET.SubElement(root, "link", name="gauge")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", value="0.5")
    ET.SubElement(inertial, "inertia", ixx="0.02", iyy="0.03", izz="0.04", ixy="0", ixz="0", iyz="0")
    joint = ET.SubElement(root, "joint", name="gauge_axis", type="prismatic")
    ET.SubElement(joint, "parent", link="base")
    ET.SubElement(joint, "child", link="gauge")
    ET.SubElement(joint, "origin", xyz="0.5 0 0")
    ET.SubElement(joint, "axis", xyz="1 0 0")
    ET.SubElement(joint, "limit", lower="-0.003", upper="0.007", effort="0.05", velocity="0.03")
    ET.SubElement(
        joint, "mimic", joint="follower" if chain else "driver", multiplier="-0.5" if chain else "0.5", offset="0.002"
    )
    path = tmp_path / ("chain.urdf" if chain else "fan.urdf")
    document.write(path, encoding="utf-8", xml_declaration=True)
    return _load(path)


def _kkt(mass, bias, applied, constraints):
    n, rows = len(bias), len(constraints)
    system = np.block([[mass, -constraints.T], [constraints, np.zeros((rows, rows))]])
    solution = np.linalg.solve(system, np.r_[applied - bias, np.zeros(rows)])
    return solution[:n], solution[n:]


@pytest.mark.parametrize("chain", [False, True], ids=["fan", "multi-level"])
def test_b01_original_graph_rows_match_unreduced_kkt_and_descendant_bounds(tmp_path, chain):
    asset = _graph_asset(tmp_path, chain)
    assert asset.tree.joint_names == ("driver", "follower", "gauge_axis")
    np.testing.assert_array_equal(asset.matrix, [[1.0], [-1.0], [0.5]])
    np.testing.assert_allclose(asset.offset_vector, [0.0, 0.004, 0.0 if chain else 0.002], atol=ATOL)
    np.testing.assert_allclose(asset.lower, [-0.006 if chain else -0.01], atol=ATOL)
    np.testing.assert_allclose(asset.upper, [0.014 if chain else 0.01], atol=ATOL)
    np.testing.assert_allclose(asset.velocity_limit, [0.06], atol=ATOL)
    np.testing.assert_allclose(asset.effort_limit, [1.5], atol=ATOL)
    state = _state(rate=0.02)
    full = asset.full_state(state)
    masses = np.array([1.0, 2.0, 0.5])
    # Exact X-only block from four original translating masses; all angular
    # coordinates decouple here because every offset, COM and force is on X.
    mass = np.zeros((4, 4))
    mass[0, 0], mass[0, 1:], mass[1:, 0] = 5.5, masses, masses
    mass[1:, 1:] = np.diag(masses)
    constraints = np.array([[0.0, 1, 1, 0], [0, 0, 0.5, 1] if chain else [0, -0.5, 0, 1]])
    force_x = np.array([1 / 8, 1 / 4, -3 / 8, 1 / 2])
    projected = np.r_[sum(force_x), force_x[1:]]
    motor = np.array([0.0, 3 / 8, 0, 0])
    acceleration, multipliers = _kkt(mass, np.zeros(4), projected + motor, constraints)
    forces = np.zeros((4, 3))
    forces[:, 0] = force_x
    external = FloatingRobotInputs(np.zeros(3), forces, np.zeros((4, 3)))
    result = asset.step(state, effort=np.array([3 / 8]), external=external, dt_s=H)
    expected_acc = np.r_[acceleration[0], np.zeros(5), acceleration[1:]]
    expected_reaction = np.r_[np.zeros(6), (constraints.T @ multipliers)[1:]]
    np.testing.assert_allclose(result.loads.acceleration_full, expected_acc, atol=2e-11, rtol=0)
    np.testing.assert_allclose(result.loads.constraint_effort_full, expected_reaction, atol=2e-11, rtol=0)
    np.testing.assert_allclose(result.loads.contact_effort_full, np.r_[projected[0], np.zeros(5), projected[1:]], atol=ATOL)
    np.testing.assert_allclose(result.loads.actuator_effort_full, np.r_[np.zeros(6), motor[1:]], atol=ATOL)
    assert result.loads.multiplier_force is None
    rows = result.loads.row_forces
    assert [(r.driver, r.follower) for r in rows] == [
        ("driver", "follower"), ("follower" if chain else "driver", "gauge_axis")
    ]
    for row, value in zip(rows, multipliers, strict=True):
        assert row.multiplier_force == pytest.approx(value, abs=2e-11)
    _assert_load_receipt(result.loads, 3, full.v)
    with pytest.raises(ValueError, match="position|interval|intersection"):
        asset.full_state(_state(u=float(asset.upper[0]) + 0.0001))
    with pytest.raises(ValueError, match="velocity|speed"):
        asset.full_state(_state(rate=0.061))


@pytest.mark.parametrize("field", ["position", "velocity", "dimension"])
def test_b02_reduce_rejects_inconsistent_full_state_without_changing_inputs(field):
    asset, state = _load(JAWS), _state(u=0.001, rate=0.02)
    full = asset.full_state(state)
    q, v = full.q.copy(), full.v.copy()
    if field == "position":
        q[-1] += 0.001
    elif field == "velocity":
        v[-1] += 0.001
    else:
        q, v = q[:-1], v[:-1]
    bad = FloatingRobotState(q, v)
    with pytest.raises((ValueError, TypeError), match="full|constraint|mimic|dimension|shape"):
        asset.reduce_state(bad)
    np.testing.assert_array_equal(asset.full_state(state).q, full.q)
    np.testing.assert_array_equal(asset.full_state(state).v, full.v)


@pytest.mark.parametrize("problem", ["cycle", "zero", "unknown", "mixed-type", "empty", "duplicate"])
def test_b03_invalid_graph_is_explicitly_rejected(tmp_path, problem):
    document = ET.parse(JAWS)
    follower = document.find(".//joint[@name='follower']")
    tag = follower.find("mimic")
    if problem == "cycle":
        ET.SubElement(document.find(".//joint[@name='driver']"), "mimic", joint="follower")
    elif problem == "zero":
        tag.set("multiplier", "0")
    elif problem == "unknown":
        tag.set("joint", "missing_axis")
    elif problem == "mixed-type":
        follower.set("type", "revolute")
    elif problem == "empty":
        follower.find("limit").set("lower", "0.25")
        follower.find("limit").set("upper", "0.5")
    else:
        ET.SubElement(follower, "mimic", joint="driver")
    path = tmp_path / f"invalid-{problem}.urdf"
    document.write(path, encoding="utf-8", xml_declaration=True)
    with pytest.raises(ValueError):
        _load(path)


@pytest.mark.parametrize("problem", ["motor-cap", "external-motor", "external-size", "candidate-limit"])
def test_b04_rejected_candidate_keeps_state_and_same_input_retry(problem):
    asset, state = _load(JAWS), _state()
    effort, external, step = np.array([1.0]), None, H
    if problem == "motor-cap":
        effort = np.array([1.501])
    elif problem == "external-motor":
        external = FloatingRobotInputs(np.array([0.0, 0.125]), np.zeros((3, 3)), np.zeros((3, 3)))
    elif problem == "external-size":
        external = FloatingRobotInputs.zeros(1)
    else:
        state = _state(u=0.015999, rate=0.079)
    before_q, before_v = state.q.copy(), state.v.copy()
    with pytest.raises(ValueError):
        asset.step(state, effort=effort, external=external, dt_s=step)
    np.testing.assert_array_equal(state.q, before_q)
    np.testing.assert_array_equal(state.v, before_v)
    retry_state = _state() if problem == "candidate-limit" else state
    expected = _load(JAWS).step(retry_state, effort=np.array([1.0]), dt_s=H)
    retry = asset.step(retry_state, effort=np.array([1.0]), dt_s=H)
    np.testing.assert_array_equal(retry.state.q, expected.state.q)
    np.testing.assert_array_equal(retry.state.v, expected.state.v)
    np.testing.assert_array_equal(retry.loads.constraint_effort_full, expected.loads.constraint_effort_full)


# F02: four original authored parts, not the loader's principal-frame aggregates.
ELBOW_PARTS = (
    (2.0, (1 / 32, -1 / 64, 0), (1 / 5, 3 / 10, 2 / 5), 0),
    (1.0, (1 / 8, 1 / 32, 0), (1 / 16, 5 / 64, 3 / 32), 1),
    (3 / 4, (1 / 16, -1 / 32, 1 / 64), (1 / 32, 5 / 128, 3 / 64), 2),
    (1 / 4, (1 / 32, 0, 1 / 64), (1 / 128, 5 / 512, 3 / 256), 2),
)
POSES = (
    ((0, 0, 1), 0.0, -1 / 4, -3 / 8),
    ((0, 1, 0), 3 / 8, 1 / 8, 1 / 4),
    ((1, 2, -1), -1 / 2, 3 / 8, 1 / 8),
)
GRAVITY = np.array([1 / 2, -1.0, -2.0])
FORCES = np.array([[1 / 4, -1 / 2, 3 / 4], [-1 / 2, 3 / 8, -1 / 4], [5 / 4, -3 / 4, 1 / 2]])
TORQUES = np.array([[1 / 32, -1 / 16, 3 / 64], [-1 / 32, 1 / 64, 1 / 16], [1 / 8, 1 / 16, -1 / 32]])
ELBOW_H = 1 / 4096
BIAS_ATOL = 2e-7
ACCEL_ATOL = 5e-6


def _skew(value):
    x, y, z = value
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _quat(axis, angle):
    axis = np.asarray(axis, dtype=float)
    return np.r_[axis / np.linalg.norm(axis) * np.sin(angle / 2), np.cos(angle / 2)]


def _quat_multiply(first, second):
    a, w = first[:3], first[3]
    b, z = second[:3], second[3]
    return np.r_[w * b + z * a + np.cross(a, b), w * z - a @ b]


def _rotation(quaternion):
    cross = _skew(quaternion[:3])
    return np.eye(3) + 2 * quaternion[3] * cross + 2 * cross @ cross


def _advance_orientation(quaternion, omega, duration):
    speed = np.linalg.norm(omega)
    if speed == 0:
        return quaternion.copy()
    return _quat_multiply(_quat(omega, duration * speed), quaternion)


def _rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _elbow_geometry(full_q):
    root, root_r = full_q[:3], _rotation(full_q[3:7])
    upper_r = root_r @ _rz(full_q[7])
    forearm_r = upper_r @ _ry(full_q[8])
    shoulder = root + root_r @ np.array([1 / 4, 0, 1 / 16])
    elbow = shoulder + upper_r @ np.array([3 / 8, 0, 0])
    tool = elbow + forearm_r @ np.array([3 / 16, 1 / 16, -1 / 32])
    frames = (
        (root, root_r), (shoulder, upper_r), (elbow, forearm_r), (tool, forearm_r @ _rz(np.pi / 2))
    )
    centers = np.array([origin + rotation @ np.array(part[1]) for (origin, rotation), part in zip(frames, ELBOW_PARTS, strict=True)])
    root_com = centers[0]
    axes = (root_r[:, 2], upper_r[:, 1])

    def jacobian(point, body):
        linear, angular = np.zeros((3, 8)), np.zeros((3, 8))
        linear[:, :3], angular[:, 3:6] = np.eye(3), np.eye(3)
        linear[:, 3:6] = -_skew(point - root_com)
        if body >= 1:
            linear[:, 6], angular[:, 6] = np.cross(axes[0], point - shoulder), axes[0]
        if body == 2:
            linear[:, 7], angular[:, 7] = np.cross(axes[1], point - elbow), axes[1]
        return linear, angular

    return frames, centers, axes, jacobian


def _constant_speed_configuration(full_q, full_v, time_s):
    # Hold the public world root-COM velocity and world omega constant; this
    # curve is solely the independent directional derivative for Jdot*v.
    q = full_q.copy()
    before_r = _rotation(full_q[3:7])
    q[3:7] = _advance_orientation(full_q[3:7], full_v[3:6], time_s)
    after_r = _rotation(q[3:7])
    q[:3] = full_q[:3] + time_s * full_v[:3] - (after_r - before_r) @ ROOT_COM
    q[7:] = full_q[7:] + time_s * full_v[6:]
    return q


def _elbow_oracle(full_q, full_v, gravity, delta):
    frames, centers, axes, jacobian = _elbow_geometry(full_q)
    samples = {
        t: _elbow_geometry(_constant_speed_configuration(full_q, full_v, t * delta))[1]
        for t in (-2, -1, 1, 2)
    }
    com_bias = (-samples[2] + 16 * samples[1] - 30 * centers + 16 * samples[-1] - samples[-2]) / (12 * delta * delta)
    omega0 = full_v[3:6]
    omega1 = omega0 + axes[0] * full_v[6]
    omega2 = omega1 + axes[1] * full_v[7]
    alpha1 = np.cross(omega0, axes[0]) * full_v[6]
    alpha2 = alpha1 + np.cross(omega1, axes[1]) * full_v[7]
    omegas = (omega0, omega1, omega2)
    alphas = (np.zeros(3), alpha1, alpha2)
    mass, bias, energy = np.zeros((8, 8)), np.zeros(8), 0.0
    for i, (weight, _, diagonal, body) in enumerate(ELBOW_PARTS):
        rotation = frames[i][1]
        inertia = rotation @ np.diag(diagonal) @ rotation.T
        linear, angular = jacobian(centers[i], body)
        mass += weight * linear.T @ linear + angular.T @ inertia @ angular
        omega = omegas[body]
        bias += linear.T @ (weight * (com_bias[i] - gravity))
        bias += angular.T @ (inertia @ alphas[body] + np.cross(omega, inertia @ omega))
        velocity = linear @ full_v
        energy += (weight * (velocity @ velocity) + omega @ inertia @ omega) / 2
    groups = []
    for body in range(3):
        members = [i for i, part in enumerate(ELBOW_PARTS) if part[3] == body]
        total = sum(ELBOW_PARTS[i][0] for i in members)
        center = sum(ELBOW_PARTS[i][0] * centers[i] for i in members) / total
        groups.append((center, *jacobian(center, body)))
    return mass, bias, energy, groups, frames, jacobian


def _elbow_state(case):
    axis, angle, position, speed = POSES[case]
    return FloatingRobotState(
        np.r_[[1 / 8, -1 / 4, 3 / 8], _quat(axis, angle), position],
        np.array([1 / 8, -1 / 16, 1 / 32, 1 / 4, -1 / 8, 3 / 16, speed]),
    )


def test_c00_fixed_tool_opt_in_and_original_authored_frame_are_explicit():
    with pytest.raises(ValueError, match="fixed"):
        _load(ELBOW)
    asset = _load(ELBOW, collapse_fixed_joints=True)
    assert asset.tree.model.body_count == 3
    assert asset.tree.joint_names == ("shaft", "gear")
    assert asset.joint_names == ("shaft",)
    np.testing.assert_allclose(asset.lower, [-3 / 4], atol=ATOL)
    np.testing.assert_allclose(asset.upper, [11 / 12], atol=ATOL)
    np.testing.assert_allclose(asset.velocity_limit, [1.0], atol=ATOL)
    np.testing.assert_allclose(asset.effort_limit, [4.0], atol=ATOL)
    assert asset.tree.sites["tool"].body_index == 2


@pytest.mark.parametrize("case", [0, 1, 2], ids=["identity", "tilted-root", "oblique-root"])
def test_c01_nonparallel_gears_match_raw_parts_mass_bias_and_full_kkt(case):
    state = _elbow_state(case)
    asset = _load(ELBOW, gravity=tuple(GRAVITY), collapse_fixed_joints=True)
    full = asset.full_state(state)
    expected_q = np.r_[state.q[:7], state.q[7], 1.5 * state.q[7] - 0.125]
    expected_v = np.r_[state.v[:6], state.v[6], 1.5 * state.v[6]]
    np.testing.assert_allclose(full.q, expected_q, atol=ATOL, rtol=0)
    np.testing.assert_allclose(full.v, expected_v, atol=ATOL, rtol=0)
    coarse = _elbow_oracle(expected_q, expected_v, GRAVITY, 1 / 512)
    mass, bias, energy, groups, frames, jacobian = _elbow_oracle(expected_q, expected_v, GRAVITY, 1 / 1024)
    # FD values and absolute SI tolerances are fixed before any execution.
    np.testing.assert_allclose(coarse[1], bias, atol=BIAS_ATOL, rtol=0)
    assert energy == pytest.approx(0.5 * expected_v @ mass @ expected_v, abs=2e-12)
    terms = floating_dynamics_terms(asset.tree.model, full, FloatingRobotInputs.zeros(2))
    np.testing.assert_allclose(terms.mass_matrix, mass, atol=2e-11, rtol=0)
    np.testing.assert_allclose(terms.bias, bias, atol=BIAS_ATOL, rtol=0)
    assert np.linalg.norm(mass[:6, 6:]) > 0.01
    geo = floating_kinematics(asset.tree.model, full)
    np.testing.assert_allclose(geo.com_position, [group[0] for group in groups], atol=2e-12, rtol=0)
    np.testing.assert_allclose(geo.com_velocity, [group[1] @ expected_v for group in groups], atol=2e-12, rtol=0)
    np.testing.assert_allclose(geo.angular_velocity, [group[2] @ expected_v for group in groups], atol=2e-12, rtol=0)
    anchor = np.array([1 / 16, 1 / 32, 1 / 32])
    local = asset.tree.authored_to_model_points("tool", anchor[None, :])
    material = frames[3][0] + frames[3][1] @ anchor
    np.testing.assert_allclose(geo.position[2] + geo.rotation[2] @ local[0], material, atol=ATOL, rtol=0)
    linear, angular = floating_point_jacobians(asset.tree.model, full, np.array([2]), local)
    expected_j, expected_w = jacobian(material, 2)
    np.testing.assert_allclose(linear[0], expected_j, atol=2e-12, rtol=0)
    np.testing.assert_allclose(angular[0], expected_w, atol=2e-12, rtol=0)
    external = FloatingRobotInputs(np.zeros(2), FORCES, TORQUES)
    applied = sum(group[1].T @ FORCES[i] + group[2].T @ TORQUES[i] for i, group in enumerate(groups))
    motor = np.r_[np.zeros(6), 13 / 32, 0.0]
    constraints = np.array([[0.0, 0, 0, 0, 0, 0, -1.5, 1.0]])
    acceleration, multipliers = _kkt(mass, bias, applied + motor, constraints)
    result = asset.step(state, effort=np.array([13 / 32]), external=external, dt_s=ELBOW_H)
    np.testing.assert_allclose(result.loads.acceleration_full, acceleration, atol=ACCEL_ATOL, rtol=0)
    np.testing.assert_allclose(result.loads.actuator_effort_full, motor, atol=ATOL, rtol=0)
    np.testing.assert_allclose(result.loads.contact_effort_full, applied, atol=2e-11, rtol=0)
    np.testing.assert_allclose(result.loads.constraint_effort_full, constraints.T[:, 0] * multipliers[0], atol=ACCEL_ATOL, rtol=0)
    row = result.loads.row_forces[0]
    assert (row.driver, row.follower) == ("shaft", "gear")
    assert row.multiplier_force == pytest.approx(multipliers[0], abs=ACCEL_ATOL)
    assert row.driver_effort == pytest.approx(-1.5 * multipliers[0], abs=ACCEL_ATOL)
    assert row.follower_effort == pytest.approx(multipliers[0], abs=ACCEL_ATOL)
    predicted_v = expected_v + ELBOW_H * acceleration
    predicted_q = _constant_speed_configuration(expected_q, predicted_v, ELBOW_H)
    np.testing.assert_allclose(result.full_state.v, predicted_v, atol=ACCEL_ATOL * ELBOW_H, rtol=0)
    np.testing.assert_allclose(result.full_state.q, predicted_q, atol=2e-11, rtol=0)
    # Independently require zero root constraint residual, not just zero labels.
    residual = terms.mass_matrix @ result.loads.acceleration_full + terms.bias - motor - applied
    np.testing.assert_allclose(residual[:6], 0.0, atol=2e-10, rtol=0)
    np.testing.assert_allclose(residual, result.loads.constraint_effort_full, atol=2e-10, rtol=0)
    _assert_load_receipt(result.loads, 2, expected_v)
    np.testing.assert_array_equal(state.q, _elbow_state(case).q)
    np.testing.assert_array_equal(state.v, _elbow_state(case).v)


def test_c02_rigid_world_change_preserves_gear_reactions_and_com_drift():
    original = _elbow_state(1)
    shifted = FloatingRobotState(
        np.r_[CYCLE @ original.q[:3] + [1 / 2, -1 / 4, 1 / 8], _quat_multiply(np.full(4, 0.5), original.q[3:7]), original.q[7:]],
        np.r_[CYCLE @ original.v[:3], CYCLE @ original.v[3:6], original.v[6:]],
    )
    first = _load(ELBOW, gravity=tuple(GRAVITY), collapse_fixed_joints=True).step(
        original, effort=np.array([13 / 32]), external=FloatingRobotInputs(np.zeros(2), FORCES, TORQUES), dt_s=ELBOW_H
    )
    second = _load(ELBOW, gravity=tuple(CYCLE @ GRAVITY), collapse_fixed_joints=True).step(
        shifted, effort=np.array([13 / 32]), external=FloatingRobotInputs(np.zeros(2), FORCES @ CYCLE.T, TORQUES @ CYCLE.T), dt_s=ELBOW_H
    )
    np.testing.assert_allclose(second.full_state.v[:3], CYCLE @ first.full_state.v[:3], atol=ATOL, rtol=0)
    np.testing.assert_allclose(second.full_state.v[3:6], CYCLE @ first.full_state.v[3:6], atol=ATOL, rtol=0)
    np.testing.assert_allclose(second.full_state.v[6:], first.full_state.v[6:], atol=ATOL, rtol=0)
    np.testing.assert_allclose(second.full_state.q[:3], CYCLE @ first.full_state.q[:3] + [1 / 2, -1 / 4, 1 / 8], atol=ATOL, rtol=0)
    np.testing.assert_allclose(_rotation(second.full_state.q[3:7]), CYCLE @ _rotation(first.full_state.q[3:7]), atol=ATOL, rtol=0)
    np.testing.assert_allclose(second.full_state.q[7:], first.full_state.q[7:], atol=ATOL, rtol=0)
    np.testing.assert_allclose(second.loads.constraint_effort_full, first.loads.constraint_effort_full, atol=2e-11, rtol=0)
