"""Physical free-root coordinates, world twists and point Jacobians on scalar trees."""

from dataclasses import replace

import numpy as np
import pytest

from phymeo.core.rigid import RigidBodyModel
from phymeo.core.robot import RobotModel
from phymeo.dynamics.floating_robot import (
    FloatingRobotModel,
    FloatingRobotState,
    floating_kinematics,
    floating_point_jacobians,
)


def root():
    return RigidBodyModel(np.array([0.5]), np.array([[0.25, 0.375, 0.5]]), np.zeros(3))


def tree():
    return RobotModel(
        np.array([-1, 0, -1]),
        ("revolute", "prismatic", "prismatic"),
        np.array([[0.5, 0, 0], [0.25, 0, 0], [0, 0, 0.25]]),
        np.tile([0.0, 0, 0, 1], (3, 1)),
        np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]]),
        np.ones(3),
        np.array([[0.125, 0, 0], [0.0625, 0.0625, 0], [0, 0.03125, 0]]),
        np.ones((3, 3)),
        np.zeros(3),
    )


def pose():
    return FloatingRobotState(
        np.array([0.25, -0.5, 0.125, 0, 0, 0, 1, np.pi / 2, 0.125, 0.25]),
        np.array([0, 0.25, 0, 0, 0, 2, 0.5, -0.25, 0.125]),
    )


def quaternion_product(a, b):
    xyz = a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3])
    return np.r_[xyz, a[3] * b[3] - a[:3] @ b[:3]]


def test_root_only_origin_com_and_opposite_skin_points():
    model = FloatingRobotModel(root(), np.array([0.125, 0, 0]))
    state = FloatingRobotState(np.array([0.0, 0, 0, 0, 0, 0, 1]), np.array([0, 0.25, 0, 0, 0, 2]))
    data = floating_kinematics(model, state)
    np.testing.assert_array_equal(data.position, [[0, 0, 0]])
    np.testing.assert_array_equal(data.com_position, [[0.125, 0, 0]])
    np.testing.assert_array_equal(data.origin_velocity, [[0, 0, 0]])
    np.testing.assert_array_equal(data.com_velocity, [[0, 0.25, 0]])
    np.testing.assert_array_equal(data.angular_velocity, [[0, 0, 2]])
    points = np.array([[0.25, 0, 0], [-0.125, 0, 0], [0, 0, 0], [0.125, 0, 0]])
    linear, angular = floating_point_jacobians(model, state, [0, 0, 0, 0], points)
    np.testing.assert_array_equal(
        linear @ state.v, [[0, 0.5, 0], [0, -0.25, 0], [0, 0, 0], [0, 0.25, 0]]
    )
    np.testing.assert_array_equal(angular @ state.v, [[0, 0, 2]] * 4)
    np.testing.assert_array_equal(linear[-1, :, :3], np.eye(3))
    np.testing.assert_array_equal(linear[-1, :, 3:], np.zeros((3, 3)))


def test_whole_tree_world_twists_full_point_jacobian_and_virtual_work():
    model = FloatingRobotModel(root(), np.array([0.125, 0, 0]), tree())
    state = pose()
    data = floating_kinematics(model, state)
    np.testing.assert_allclose(
        data.position,
        [[0.25, -0.5, 0.125], [0.75, -0.5, 0.125], [0.75, -0.125, 0.125], [0.25, -0.25, 0.375]],
        atol=2e-15,
    )
    np.testing.assert_allclose(
        data.origin_velocity,
        [[0, 0, 0], [0, 1, 0], [-0.9375, 0.75, 0], [-0.5, 0.125, 0]],
        atol=2e-15,
    )
    np.testing.assert_allclose(
        data.angular_velocity, [[0, 0, 2], [0, 0, 2.5], [0, 0, 2.5], [0, 0, 2]], atol=2e-15
    )
    np.testing.assert_allclose(data.com_velocity[2], [-1.09375, 0.59375, 0], atol=2e-15)
    p = np.array([0, 0.125, 0.0625])
    np.testing.assert_allclose(
        data.position[2] + data.rotation[2] @ p, [0.625, -0.125, 0.1875], atol=2e-15
    )
    linear, angular = floating_point_jacobians(model, state, [2], p[None])
    expected = [
        [1, 0, 0, 0, 0.0625, -0.375, -0.375, 0, 0],
        [0, 1, 0, -0.0625, 0, 0.25, -0.125, 1, 0],
        [0, 0, 1, 0.375, -0.25, 0, 0, 0, 0],
    ]
    np.testing.assert_allclose(linear[0], expected, atol=2e-15)
    expected_angular = np.zeros((3, 9))
    expected_angular[:, 3:6] = np.eye(3)
    expected_angular[2, 6] = 1
    np.testing.assert_allclose(angular[0], expected_angular, atol=2e-15)
    force, couple = np.array([2, -3, 4]), np.array([0.5, -0.25, 0.75])
    generalized = linear[0].T @ force + angular[0].T @ couple
    np.testing.assert_allclose(
        generalized, [2, -3, 4, 2.1875, -1.125, -0.75, 0.375, -3, 0], atol=2e-15
    )
    assert generalized @ state.v == pytest.approx(-1.3125)
    assert force @ (linear[0] @ state.v) + couple @ data.angular_velocity[2] == pytest.approx(
        -1.3125
    )


def oblique_model():
    children = tree()
    axes, frames = children.joint_axis.copy(), children.joint_orientation.copy()
    axes[0], axes[1] = [0.6, 0.8, 0], [0, -0.8, 0.6]
    frames[1] = [0.5, 0.5, 0.5, 0.5]
    return FloatingRobotModel(
        root(),
        np.array([0.125, -0.0625, 0.03125]),
        replace(children, joint_axis=axes, joint_orientation=frames),
    )


def oblique_state():
    return FloatingRobotState(
        np.array([0.125, 0.375, -0.5, 0.5, 0.5, 0.5, 0.5, -0.37, 0.25, -0.125]),
        np.array([0.25, -0.375, -0.125, 1, -0.5, 0.75, -0.25, 0.0625, 0.125]),
    )


def test_oblique_frames_match_position_and_rotation_directional_derivatives():
    model, state = oblique_model(), oblique_state()
    points = np.array([[0.25, -0.125, 0], [0.25, 0, 0.0625], [-0.25, 0.5, 0.125], [0, 0.25, 0]])
    indices = np.arange(4)
    linear, angular = floating_point_jacobians(model, state, indices, points)
    data = floating_kinematics(model, state)
    center = state.q[:3] + model.root_com_m[[2, 0, 1]]
    eps = 2e-6
    for column in range(len(state.v)):
        point_pair, rotation_pair = [], []
        for sign in [-1, 1]:
            q = state.q.copy()
            step = sign * eps
            if column < 3:
                q[column] += step
            elif column < 6:
                axis = np.eye(3)[column - 3]
                rotvec = np.r_[axis * np.sin(step / 2), np.cos(step / 2)]
                q[3:7] = quaternion_product(rotvec, q[3:7])
                offset = model.root_com_m[[2, 0, 1]]
                rotated_offset = (
                    offset * np.cos(step)
                    + np.cross(axis, offset) * np.sin(step)
                    + axis * (axis @ offset) * (1 - np.cos(step))
                )
                q[:3] = center - rotated_offset
            else:
                q[column + 1] += step
            moved = floating_kinematics(model, FloatingRobotState(q, state.v))
            point_pair.append(moved.position + np.einsum("nij,nj->ni", moved.rotation, points))
            rotation_pair.append(moved.rotation)
        np.testing.assert_allclose(
            (point_pair[1] - point_pair[0]) / (2 * eps), linear[:, :, column], atol=8e-10
        )
        spin = (rotation_pair[1] - rotation_pair[0]) @ data.rotation.transpose(0, 2, 1) / (2 * eps)
        np.testing.assert_allclose(spin[:, [2, 0, 1], [1, 2, 0]], angular[:, :, column], atol=8e-10)
    np.testing.assert_allclose(
        linear @ state.v,
        data.origin_velocity
        + np.cross(data.angular_velocity, np.einsum("nij,nj->ni", data.rotation, points)),
        atol=2e-15,
    )
    np.testing.assert_allclose(angular @ state.v, data.angular_velocity, atol=2e-15)


def test_common_rotation_translation_and_reindexing_independent_branches():
    model, state = oblique_model(), oblique_state()
    points = np.array([[0, 0.25, 0], [-0.125, 0, 0], [0.375, 0.5, 0], [0.25, -0.25, 0]])
    original = floating_kinematics(model, state)
    linear, angular = floating_point_jacobians(model, state, np.arange(4), points)
    rotation = np.array([[-7, 24, 0], [24, 7, 0], [0, 0, -25]]) / 25
    shift = np.array([0.625, -0.125, 1])
    q, v = state.q.copy(), state.v.copy()
    q[:3] = rotation @ q[:3] + shift
    q[3:7] = quaternion_product(np.array([0.6, 0.8, 0, 0]), q[3:7])
    v[:3], v[3:6] = rotation @ v[:3], rotation @ v[3:6]
    transformed = FloatingRobotState(q, v)
    moved = floating_kinematics(model, transformed)
    for field in ("position", "com_position"):
        np.testing.assert_allclose(
            getattr(moved, field), getattr(original, field) @ rotation.T + shift, atol=2e-15
        )
    np.testing.assert_allclose(moved.rotation, rotation @ original.rotation, atol=2e-15)
    for field in ("origin_velocity", "com_velocity", "angular_velocity"):
        np.testing.assert_allclose(
            getattr(moved, field), getattr(original, field) @ rotation.T, atol=2e-15
        )
    tangent = np.eye(9)
    tangent[:3, :3] = tangent[3:6, 3:6] = rotation
    for new, old in zip(
        floating_point_jacobians(model, transformed, np.arange(4), points),
        (linear, angular),
        strict=True,
    ):
        np.testing.assert_allclose(new, rotation @ old @ tangent.T, atol=2e-15)

    order = [2, 0, 1]
    body_order, columns = [0, 3, 1, 2], [0, 1, 2, 3, 4, 5, 8, 6, 7]
    c = model.children
    reordered_children = replace(
        c,
        parents=np.array([-1, -1, 1]),
        joint_types=tuple(c.joint_types[i] for i in order),
        **{
            k: getattr(c, k)[order]
            for k in (
                "joint_position",
                "joint_orientation",
                "joint_axis",
                "com",
                "mass",
                "inertia_diagonal",
            )
        },
    )
    reordered = replace(model, children=reordered_children)
    s = FloatingRobotState(np.r_[state.q[:7], state.q[7:][order]], state.v[columns])
    obs = floating_kinematics(reordered, s)
    for field in (
        "position",
        "rotation",
        "com_position",
        "origin_velocity",
        "com_velocity",
        "angular_velocity",
    ):
        np.testing.assert_allclose(
            getattr(obs, field), getattr(original, field)[body_order], atol=2e-15
        )
    for new, old in zip(
        floating_point_jacobians(reordered, s, np.arange(4), points[body_order]),
        (linear, angular),
        strict=True,
    ):
        np.testing.assert_allclose(new, old[body_order][:, :, columns], atol=2e-15)


def test_state_model_and_point_queries_validate_the_native_coordinate_domain():
    model = FloatingRobotModel(root(), np.zeros(3), tree())
    q, v = pose().q.copy(), pose().v.copy()
    state = FloatingRobotState(q, v)
    q[:], v[:] = 0, 0
    np.testing.assert_array_equal(state.q, pose().q)
    np.testing.assert_array_equal(state.v, pose().v)
    for bad in (np.zeros(10), np.full(10, np.nan)):
        with pytest.raises(ValueError):
            FloatingRobotState(bad, state.v)
    with pytest.raises(ValueError):
        FloatingRobotState(state.q, state.v[:-1])
    with pytest.raises(TypeError):
        FloatingRobotState(state.q.astype(np.float32), state.v)
    with pytest.raises(ValueError):
        FloatingRobotModel(replace(root(), inverse_mass=np.zeros(1)), np.zeros(3))
    with pytest.raises(ValueError):
        FloatingRobotModel(root(), np.ones(2))
    with pytest.raises(ValueError):
        floating_kinematics(FloatingRobotModel(root(), np.zeros(3)), state)
    for ids, points, error in (
        ([4], np.zeros((1, 3)), ValueError),
        ([-1], np.zeros((1, 3)), ValueError),
        ([1.0], np.zeros((1, 3)), TypeError),
        ([1], np.zeros((2, 3)), ValueError),
    ):
        with pytest.raises(error):
            floating_point_jacobians(model, state, ids, points)
    for result in floating_point_jacobians(model, state, np.array([], dtype=int), np.empty((0, 3))):
        assert result.shape == (0, 3, 9)
