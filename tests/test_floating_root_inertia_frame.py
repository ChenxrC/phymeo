"""Independent full-inertia oracles; frozen before F3 implementation."""

import json
from dataclasses import replace
from math import cos, pi, sin

import numpy as np
import pytest

import phymeo as ph

MOMENTS = np.array([1 / 4, 1 / 2, 1 / 2])
CENTER = np.array([1 / 8, -1 / 16, 1 / 32])
BODY_I = np.array([[3 / 8, -1 / 8, 0], [-1 / 8, 3 / 8, 0], [0, 0, 1 / 2]])
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
CYCLE = np.array([0.5, 0.5, 0.5, 0.5])


def _skew(x):
    a, b, c = x
    return np.array([[0.0, -c, b], [c, 0.0, -a], [-b, a, 0.0]])


def _rot(q):
    vector, scalar = np.asarray(q[:3]), q[3]
    return np.eye(3) + 2 * scalar * _skew(vector) + 2 * _skew(vector) @ _skew(vector)


def _exp(v):
    angle = np.linalg.norm(v)
    if angle == 0:
        return np.eye(3)
    k = _skew(v / angle)
    return np.eye(3) + sin(angle) * k + (1 - cos(angle)) * (k @ k)


def _model(*, children=None, frame=None, gravity=None):
    root = ph.RigidBodyModel(
        np.array([0.5]),
        MOMENTS[None, :],
        np.zeros(3) if gravity is None else np.asarray(gravity),
    )
    assert root.body_count == 1
    if frame is None:
        frame = np.array([0.0, 0.0, sin(pi / 8), cos(pi / 8)])
    return ph.FloatingRobotModel(root, CENTER, children, root_inertia_frame_xyzw=frame)


def _close(actual, expected, *, atol=2e-12):
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=atol)


@pytest.mark.parametrize("pose", [IDENTITY, CYCLE])
def test_root_frame_preserves_authored_pose_and_exact_tensor_response(pose):
    rotation = _rot(pose)
    model = _model()
    state = ph.FloatingRobotState(np.r_[np.zeros(3), pose], np.zeros(6))
    force = np.zeros((1, 3))
    torque = (rotation @ [1.0, 0.0, 0.0])[None, :]
    terms = ph.floating_dynamics_terms(
        model, state, ph.FloatingRobotInputs(np.zeros(0), force, torque)
    )
    expected_mass = np.zeros((6, 6))
    expected_mass[:3, :3] = 2 * np.eye(3)
    expected_mass[3:, 3:] = rotation @ BODY_I @ rotation.T
    _close(terms.mass_matrix, expected_mass)
    _close(terms.acceleration[:3], 0)
    _close(terms.acceleration[3:], rotation @ [3.0, 1.0, 0.0])
    result = ph.step_free_root(model, state, ph.RigidBodyInputs(force, torque), 1 / 128)
    assert result.report.accepted
    _close(result.state.v[3:], rotation @ [3.0, 1.0, 0.0] / 128)
    actual_r = _rot(result.state.q[3:7])
    expected_r = _exp(rotation @ [3.0, 1.0, 0.0] / 128**2) @ rotation
    _close(actual_r, expected_r)
    _close(result.state.q[:3] + actual_r @ CENTER, rotation @ CENTER)


@pytest.mark.parametrize(
    "pose,shift", [(IDENTITY, np.zeros(3)), (CYCLE, np.array([0.5, -0.25, 0.75]))]
)
def test_root_frame_actual_forty_step_gyro_com_window(tmp_path, pose, shift):
    rotation = _rot(pose)
    force = rotation @ [1 / 4, -1 / 8, 1 / 16]
    torque = rotation @ [1 / 8, 1 / 4, -1 / 16]
    gravity = rotation @ [0.0, 0.0, -0.5]
    model = _model(gravity=gravity)
    velocity = rotation @ [1 / 8, -1 / 4, 1 / 16]
    omega = rotation @ [1 / 4, -1 / 8, 3 / 8]
    state = ph.FloatingRobotState(np.r_[shift, pose], np.r_[velocity, omega])
    center = shift + rotation @ CENTER
    h, trace = 1 / 1024, []
    loads = ph.RigidBodyInputs(force[None, :], torque[None, :])
    for number in range(1, 41):
        inertia = rotation @ BODY_I @ rotation.T
        alpha = np.linalg.solve(inertia, torque - np.cross(omega, inertia @ omega))
        velocity = velocity + h * (force / 2 + gravity)
        omega = omega + h * alpha
        center = center + h * velocity
        rotation = _exp(h * omega) @ rotation
        result = ph.step_free_root(model, state, loads, h)
        actual = result.state
        trace.append(
            dict(
                step=number,
                accepted=result.report.accepted,
                q=actual.q.tolist(),
                v=actual.v.tolist(),
                oracle_com=center.tolist(),
                oracle_rotation=rotation.tolist(),
                oracle_v=np.r_[velocity, omega].tolist(),
            )
        )
        (tmp_path / "root-frame-full.json").write_text(
            json.dumps(
                dict(
                    h_s=h,
                    force=force.tolist(),
                    torque=torque.tolist(),
                    gravity=gravity.tolist(),
                    frames=trace,
                ),
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        assert result.report.accepted
        _close(actual.v, np.r_[velocity, omega], atol=2e-10)
        actual_r = _rot(actual.q[3:7])
        _close(actual_r, rotation, atol=2e-10)
        _close(actual.q[:3], center - rotation @ CENTER, atol=2e-10)
        _close(actual.q[:3] + actual_r @ CENTER, center, atol=2e-10)
        actual_inertia = actual_r @ BODY_I @ actual_r.T
        actual_energy = (
            actual.v[:3] @ actual.v[:3] + 0.5 * actual.v[3:] @ actual_inertia @ actual.v[3:]
        )
        oracle_energy = velocity @ velocity + 0.5 * omega @ (rotation @ BODY_I @ rotation.T) @ omega
        _close(actual_energy, oracle_energy, atol=2e-10)
        state = actual


@pytest.mark.parametrize("pose", [IDENTITY, CYCLE])
def test_root_frame_full_tree_mass_bias_external_and_acceleration(pose):
    r0 = _rot(pose)
    children = ph.RobotModel(
        np.array([-1]),
        ("revolute",),
        np.array([[0.5, 0.0, 0.0]]),
        IDENTITY[None, :],
        np.array([[0.0, 0.0, 1.0]]),
        np.ones(1),
        np.array([[0.25, 0.0, 0.0]]),
        np.array([[1 / 16, 1 / 8, 3 / 16]]),
        r0 @ [0.0, 0.0, -0.5],
    )
    gravity = r0 @ [0.0, 0.0, -0.5]
    model = _model(children=children, gravity=gravity)
    qj, rate = 0.37, 0.5
    v0 = r0 @ [1 / 8, -1 / 4, 1 / 16]
    w0 = r0 @ [1 / 4, -1 / 8, 3 / 8]
    state = ph.FloatingRobotState(np.r_[np.zeros(3), pose, qj], np.r_[v0, w0, rate])
    r1 = r0 @ _exp(np.array([0.0, 0.0, qj]))
    offset, arm = r0 @ [0.5, 0.0, 0.0], r1 @ [0.25, 0.0, 0.0]
    center0, center1 = r0 @ CENTER, offset + arm
    axis = r0 @ [0.0, 0.0, 1.0]
    omega = np.array([w0, w0 + axis * rate])
    linear, angular = np.zeros((2, 3, 7)), np.zeros((2, 3, 7))
    linear[:, :, :3] = np.eye(3)
    angular[:, :, 3:6] = np.eye(3)
    linear[1, :, 3:6] = -_skew(center1 - center0)
    linear[1, :, 6] = np.cross(axis, arm)
    angular[1, :, 6] = axis
    inertia = np.array([r0 @ BODY_I @ r0.T, r1 @ np.diag([1 / 16, 1 / 8, 3 / 16]) @ r1.T])
    mass = sum(
        m * jv.T @ jv + jw.T @ ii @ jw
        for m, jv, jw, ii in zip((2.0, 1.0), linear, angular, inertia)
    )
    alpha1 = np.cross(w0, axis * rate)
    com_a1 = -np.cross(w0, np.cross(w0, center0))
    com_a1 += np.cross(w0, np.cross(w0, offset)) + np.cross(alpha1, arm)
    com_a1 += np.cross(omega[1], np.cross(omega[1], arm))
    bias_force = np.array([-2 * gravity, com_a1 - gravity])
    bias_torque = np.array(
        [
            np.cross(w0, inertia[0] @ w0),
            inertia[1] @ alpha1 + np.cross(omega[1], inertia[1] @ omega[1]),
        ]
    )
    bias = sum(
        jv.T @ ff + jw.T @ tt for jv, jw, ff, tt in zip(linear, angular, bias_force, bias_torque)
    )
    force = np.array([r0 @ [1 / 4, -1 / 8, 1 / 16], r0 @ [1.0, 2.0, -1.0]])
    torque = np.array([r0 @ [1 / 8, 1 / 4, -1 / 16], r0 @ [1 / 4, -1 / 2, 3 / 4]])
    applied = sum(jv.T @ ff + jw.T @ tt for jv, jw, ff, tt in zip(linear, angular, force, torque))
    applied[6] += 1 / 8
    loads = ph.FloatingRobotInputs(np.array([1 / 8]), force, torque)
    actual = ph.floating_dynamics_terms(model, state, loads)
    _close(actual.mass_matrix, mass)
    _close(actual.bias, bias)
    _close(actual.applied, applied)
    _close(actual.acceleration, np.linalg.solve(mass, applied - bias))
    h = 1 / 1024
    step = ph.step_floating_robot(model, state, loads, h)
    assert step.report.accepted
    _close(step.state.v, state.v + h * np.linalg.solve(mass, applied - bias))


@pytest.mark.parametrize(
    "bad",
    [np.zeros(3), np.zeros(4), np.array([0.0, 0.0, 0.0, 2.0]), np.array([0.0, np.nan, 0.0, 1.0])],
)
def test_root_frame_invalid_and_owned(bad):
    with pytest.raises((ValueError, TypeError)):
        _model(frame=bad)
    frame = np.array([0.0, 0.0, sin(pi / 8), cos(pi / 8)])
    model = _model(frame=frame)
    frame[:] = 0
    _close(
        _rot(model.root_inertia_frame_xyzw)
        @ np.diag(MOMENTS)
        @ _rot(model.root_inertia_frame_xyzw).T,
        BODY_I,
    )
    with pytest.raises((ValueError, TypeError)):
        model.root_inertia_frame_xyzw[0] = 1
    with pytest.raises(ValueError):
        model.root_inertia_frame_xyzw.setflags(write=True)


def test_root_frame_default_sign_and_rejection_preserve_state():
    root = ph.RigidBodyModel(np.array([0.5]), MOMENTS[None, :], np.zeros(3))
    default = ph.FloatingRobotModel(root, CENTER)
    _close(default.root_inertia_frame_xyzw, IDENTITY)
    model = _model()
    negative = replace(model, root_inertia_frame_xyzw=-model.root_inertia_frame_xyzw)
    state = ph.FloatingRobotState(np.r_[np.zeros(3), IDENTITY], np.zeros(6))
    ordinary = ph.RigidBodyInputs(np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0]]))
    a = ph.step_free_root(model, state, ordinary, 1 / 128)
    b = ph.step_free_root(negative, state, ordinary, 1 / 128)
    assert a.report.accepted and b.report.accepted
    _close(a.state.v, b.state.v)
    _close(_rot(a.state.q[3:7]), _rot(b.state.q[3:7]))
    huge = ph.RigidBodyInputs(np.zeros((1, 3)), np.array([[np.finfo(float).max, 0.0, 0.0]]))
    bad = ph.step_free_root(model, state, huge, 1.0)
    assert not bad.report.accepted
    np.testing.assert_array_equal(bad.state.q, state.q)
    np.testing.assert_array_equal(bad.state.v, state.v)
