"""Independent tree geometry and complete, owned robot state boundaries."""

import importlib
from dataclasses import dataclass, replace

import numpy as np
import pytest

from phymeo.core.scene import BackendContext, DomainBinding, SceneInputs, SceneModel, SceneState
from phymeo.runtime.registry import StateRegistry


def robot_api():
    return importlib.import_module("phymeo.core.robot")


def kinematics_api():
    return importlib.import_module("phymeo.dynamics.robot_kinematics")


def robot_parameters(count=2):
    return dict(
        parents=np.arange(-1, count - 1, dtype=np.int64),
        joint_types=("revolute",) * count,
        joint_position=np.array([[0, 0, 0]] + [[0, -0.5, 0]] * (count - 1), dtype=float),
        joint_orientation=np.tile([0.0, 0.0, 0.0, 1.0], (count, 1)),
        joint_axis=np.tile([0.0, 0.0, 1.0], (count, 1)),
        mass=np.ones(count),
        com=np.tile([0.0, -0.25, 0.0], (count, 1)),
        inertia_diagonal=np.tile([1 / 48, 0.001, 1 / 48], (count, 1)),
        gravity=np.array([0.0, -9.81, 0.0]),
    )


def robot_model(count=2, **overrides):
    params = robot_parameters(count)
    params.update(overrides)
    return robot_api().RobotModel(**params)


def test_robot_data_is_owned_and_registered():
    api = robot_api()
    params = robot_parameters()
    model = api.RobotModel(**params)
    assert model.dof_count == 2
    for name, original in params.items():
        if isinstance(original, np.ndarray):
            assert not np.shares_memory(getattr(model, name), original)
            assert not getattr(model, name).flags.writeable
    q, v = np.array([0.2, -0.1]), np.array([0.0, 0.3])
    state = api.RobotState(q, v)
    inputs = api.RobotInputs.zeros(2)
    q[:] = 99
    np.testing.assert_array_equal(state.q, [0.2, -0.1])
    force = np.ones((2, 3))
    loaded = api.RobotInputs(np.ones(2), force, np.zeros((2, 3)))
    force[:] = 99
    np.testing.assert_array_equal(loaded.force, np.ones((2, 3)))
    family = importlib.import_module("phymeo.runtime.robot_family").robot_family()
    registry = StateRegistry()
    registry.register(family)
    scene = SceneModel((DomainBinding("arm", "robot", model),))
    owned, forces = registry.snapshot(
        scene, SceneState({"arm": state}), SceneInputs({"arm": inputs}), BackendContext()
    )
    assert not np.shares_memory(owned.domains["arm"].q, state.q)
    assert not np.shares_memory(forces.domains["arm"].effort, inputs.effort)
    with pytest.raises(ValueError):
        api.validate_robot_step(model, state, api.RobotInputs.zeros(3))
    with pytest.raises(ValueError):
        family.candidate(model, state, api.RobotState(np.zeros(3), np.zeros(3)), BackendContext())

    @dataclass
    class HiddenState(api.RobotState):
        warm: np.ndarray

    with pytest.raises(TypeError, match="unregistered"):
        family.copy_state(HiddenState(state.q, state.v, np.zeros(2)), BackendContext())


@pytest.mark.parametrize(
    "field,value",
    [
        ("parents", np.array([-1.0, 0.0])),
        ("parents", np.array([False, True])),
        ("parents", np.array([-1, 1])),
        ("parents", np.array([-1, -2])),
        ("parents", np.array([1, 0])),
        ("parents", np.array([-1])),
        ("joint_types", ("fixed", "revolute")),
        ("joint_types", ("revolute",)),
        ("joint_types", "revolute"),
        ("joint_position", np.zeros((2, 2))),
        ("joint_position", np.zeros((2, 3), np.float32)),
        ("joint_axis", np.zeros((2, 3))),
        ("joint_axis", np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]])),
        ("joint_orientation", np.tile([0.0, 0.0, 0.0, 2.0], (2, 1))),
        ("mass", np.array([1.0, 0.0])),
        ("mass", np.array([1.0, -1.0])),
        ("mass", np.array([1.0, np.inf])),
        ("mass", np.array([1.0, 1.0], np.float32)),
        ("com", np.full((2, 3), np.nan)),
        ("inertia_diagonal", np.array([[1.0, 1.0, 3.0], [1.0, 1.0, 1.0]])),
        ("inertia_diagonal", np.zeros((2, 3))),
        ("gravity", np.zeros(4)),
    ],
)
def test_reject_invalid_robot_model(field, value):
    with pytest.raises((TypeError, ValueError)):
        robot_model(**{field: value})


def test_empty_robot_is_explicitly_unsupported():
    params = robot_parameters()
    for name in params:
        if name != "gravity":
            params[name] = params[name][:0]
    with pytest.raises(ValueError):
        robot_api().RobotModel(**params)


@pytest.mark.parametrize("q,v", [([1], [0]), ([1.0], [0.0, 0.0]), ([np.nan], [0.0])])
def test_reject_invalid_state(q, v):
    with pytest.raises((TypeError, ValueError)):
        robot_api().RobotState(np.array(q), np.array(v))


def test_reject_float32_and_mismatched_input_fields():
    api = robot_api()
    with pytest.raises(TypeError):
        api.RobotState(np.zeros(2, np.float32), np.zeros(2, np.float32))
    with pytest.raises(ValueError):
        api.RobotInputs(np.zeros(2), np.zeros((3, 3)), np.zeros((2, 3)))
    with pytest.raises(TypeError):
        api.RobotInputs(np.zeros(2), np.zeros((2, 3), np.float32), np.zeros((2, 3)))
    with pytest.raises((TypeError, ValueError)):
        api.RobotInputs.zeros(True)


def test_planar_two_link_and_prismatic_geometry():
    model = robot_model(
        3,
        joint_types=("revolute", "revolute", "prismatic"),
        joint_axis=np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0]], dtype=float),
    )
    fk = kinematics_api().forward_kinematics(model, np.array([np.pi / 2, -np.pi / 2, 0.2]))
    np.testing.assert_allclose(fk.position, [[0, 0, 0], [0.5, 0, 0], [0.7, -0.5, 0]], atol=1e-15)
    np.testing.assert_allclose(
        fk.com_position, [[0.25, 0, 0], [0.5, -0.25, 0], [0.7, -0.75, 0]], atol=1e-15
    )
    np.testing.assert_allclose(fk.rotation[1:], np.tile(np.eye(3), (2, 1, 1)), atol=1e-15)


def test_nonidentity_zero_frame_and_com():
    model = robot_model(
        1,
        joint_position=np.array([[0.3, -0.4, 0.2]]),
        joint_orientation=np.array([[np.sqrt(0.5), 0, 0, np.sqrt(0.5)]]),
        com=np.array([[0.1, 0.2, 0.3]]),
    )
    fk = kinematics_api().forward_kinematics(model, np.zeros(1))
    np.testing.assert_allclose(fk.com_position, [[0.4, -0.7, 0.4]], atol=1e-15)
    np.testing.assert_allclose(fk.axis, [[0, -1, 0]], atol=1e-15)
    fk.position[:] = 99
    np.testing.assert_array_equal(model.joint_position, [[0.3, -0.4, 0.2]])


def branched_model():
    orientation = np.tile([0.0, 0.0, 0.0, 1.0], (4, 1))
    orientation[0] = [np.sin(0.15), 0, 0, np.cos(0.15)]
    orientation[1] = [0, 0, np.sin(0.2), np.cos(0.2)]
    return robot_model(
        4,
        parents=np.array([-1, 0, 0, 1]),
        joint_types=("revolute", "prismatic", "revolute", "revolute"),
        joint_axis=np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float),
        joint_orientation=orientation,
        joint_position=np.array([[0.1, 0.2, 0.3], [0, -0.5, 0], [0.4, 0.1, 0.2], [0.2, -0.3, 0]]),
    )


def test_branched_mixed_joint_jacobians_against_world_differences():
    kin = kinematics_api()
    model, q = branched_model(), np.array([0.2, 0.03, -0.4, 0.1])
    indices = np.array([3, 2, 3, 0])
    points = np.array([[0.1, 0.2, 0.3], [-0.1, 0.4, 0], [0, -0.2, 0.1], [0.3, 0, 0]])
    linear, angular = kin.point_jacobians(model, q, indices, points)
    assert linear.shape == angular.shape == (4, 3, 4)
    np.testing.assert_array_equal(linear[0, :, 2], 0)  # Sibling cannot move link 3.
    np.testing.assert_array_equal(linear[1, :, 1], 0)
    eps = 1e-6
    base = kin.forward_kinematics(model, q)
    for joint in range(4):
        delta = np.eye(4)[joint] * eps
        plus = kin.forward_kinematics(model, q + delta)
        minus = kin.forward_kinematics(model, q - delta)
        xp = plus.position[indices] + np.einsum("nij,nj->ni", plus.rotation[indices], points)
        xm = minus.position[indices] + np.einsum("nij,nj->ni", minus.rotation[indices], points)
        np.testing.assert_allclose(linear[:, :, joint], (xp - xm) / (2 * eps), atol=2e-8, rtol=0)
        derivative = (plus.rotation[indices] - minus.rotation[indices]) / (2 * eps)
        skew = derivative @ np.swapaxes(base.rotation[indices], 1, 2)
        omega = np.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], axis=1)
        np.testing.assert_allclose(angular[:, :, joint], omega, atol=2e-8, rtol=0)


@pytest.mark.parametrize("q", [np.zeros(1), np.zeros(2, np.float32), np.array([0.0, np.inf])])
def test_kinematics_rejects_invalid_q(q):
    with pytest.raises((ValueError, TypeError)):
        kinematics_api().forward_kinematics(robot_model(), q)


@pytest.mark.parametrize(
    "indices", [np.array([-1]), np.array([2]), np.array([0.0]), np.array([True])]
)
def test_jacobians_reject_invalid_indices(indices):
    with pytest.raises((ValueError, TypeError)):
        kinematics_api().point_jacobians(robot_model(), np.zeros(2), indices, np.zeros((1, 3)))


def test_empty_point_query_and_multiple_roots():
    kin = kinematics_api()
    model = robot_model(parents=np.array([-1, -1]))
    linear, angular = kin.point_jacobians(
        model, np.zeros(2), np.array([], dtype=int), np.empty((0, 3))
    )
    assert linear.shape == angular.shape == (0, 3, 2)
    linear, _ = kin.point_jacobians(model, np.zeros(2), np.array([1]), np.array([[0.2, 0, 0]]))
    np.testing.assert_array_equal(linear[:, :, 0], 0)
    with pytest.raises((ValueError, TypeError)):
        kin.point_jacobians(model, np.zeros(2), np.array([1]), np.zeros((1, 2)))


def test_bound_robot_model_mutation_is_rejected():
    model = robot_model()
    scene = SceneModel((DomainBinding("arm", "robot", model),))
    changed = replace(model, gravity=np.zeros(3))
    assert changed is not model
    object.__setattr__(model, "gravity", np.zeros(3))
    with pytest.raises(ValueError, match="changed"):
        scene.assert_unchanged()
