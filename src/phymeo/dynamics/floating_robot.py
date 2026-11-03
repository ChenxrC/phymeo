"""World kinematics of a physical free root and an optional scalar-joint tree (FP64)."""

from dataclasses import dataclass

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.rigid import RigidBodyModel
from ..core.robot import RobotModel, robot_array
from .robot_kinematics import _point_jacobians_at, forward_kinematics
from .rotation import quaternion_matrix


@dataclass(frozen=True, eq=False)
class FloatingRobotModel:
    """One massive root plus a tree expressed in its body frame.

    Body zero is the root. Body i+1 corresponds to scalar joint/link i of
    children. A child parent of -1 attaches to the root frame. root_com_m is
    measured from its body origin in body axes, in metres. Kinematics uses
    only geometry; root inertia axes are its body axes.
    """

    root: RigidBodyModel
    root_com_m: np.ndarray
    children: RobotModel | None = None

    def __post_init__(self):
        if not isinstance(self.root, RigidBodyModel):
            raise TypeError("root must be a RigidBodyModel")
        if self.root.body_count != 1 or self.root.inverse_mass[0] <= 0:
            raise ValueError("expected one dynamic root body")
        if self.root.inverse_mass.dtype != np.float64:
            raise TypeError("floating root must use float64")
        if self.children is not None and not isinstance(self.children, RobotModel):
            raise TypeError("children must be a RobotModel or None")
        center = robot_array(self.root_com_m, "root_com_m", ndim=1)
        if center.shape != (3,):
            raise ValueError("root_com_m must be a three-vector")
        center.flags.writeable = False
        object.__setattr__(self, "root_com_m", center)

    @property
    def body_count(self):
        return 1 + (0 if self.children is None else self.children.dof_count)


@dataclass(frozen=True, eq=False)
class FloatingRobotState:
    """q = [world origin xyz, body-to-world xyzw, child q].
    v = [world root COM velocity, world angular velocity, child rates].
    """

    q: np.ndarray
    v: np.ndarray

    def __post_init__(self):
        q, v = robot_array(self.q, "q", ndim=1), robot_array(self.v, "v", ndim=1)
        if len(q) < 7 or len(v) != len(q) - 1:
            raise ValueError("expected 7+n root/child coordinates and 6+n velocities")
        if abs(np.hypot.reduce(q[3:7]) - 1.0) > 1e-10:
            raise ValueError("root orientation must be a unit xyzw quaternion")
        q.flags.writeable = v.flags.writeable = False
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "v", v)


@dataclass(frozen=True, eq=False)
class FloatingKinematics:
    """Body-indexed world origin pose, COM position, and instantaneous velocities."""

    position: np.ndarray
    rotation: np.ndarray
    com_position: np.ndarray
    origin_velocity: np.ndarray
    com_velocity: np.ndarray
    angular_velocity: np.ndarray


def _root_and_children(model, state):
    if not isinstance(model, FloatingRobotModel) or not isinstance(state, FloatingRobotState):
        raise TypeError("expected a FloatingRobotModel and FloatingRobotState")
    if len(state.q) != 6 + model.body_count:
        raise ValueError("state scalar coordinates must match the model children")
    rotation = quaternion_matrix(state.q[None, 3:7])[0]
    local = None if model.children is None else forward_kinematics(model.children, state.q[7:])
    return rotation, local


def floating_kinematics(model, state):
    """Read world poses and twists at the supplied common configuration/velocity."""
    root_rotation, local = _root_and_children(model, state)
    count = model.body_count
    position, com, origin_v, com_v, omega = (np.empty((count, 3)) for _ in range(5))
    rotation = np.empty((count, 3, 3))
    position[0], rotation[0] = state.q[:3], root_rotation
    com_v[0], omega[0] = state.v[:3], state.v[3:6]
    try:
        with np.errstate(over="raise", invalid="raise"):
            offset = root_rotation @ model.root_com_m
            com[0] = position[0] + offset
            origin_v[0] = com_v[0] - np.cross(omega[0], offset)
            if local is not None:
                children = model.children
                position[1:] = position[0] + local.position @ root_rotation.T
                rotation[1:] = root_rotation @ local.rotation
                com[1:] = position[0] + local.com_position @ root_rotation.T
                axes = local.axis @ root_rotation.T
                for joint, parent in enumerate(children.parents):
                    body, parent_body = joint + 1, parent + 1
                    # Subtract within the root frame to retain small lever arms at large world offsets.
                    arm = local.position[joint]
                    if parent >= 0:
                        arm = arm - local.position[parent]
                    origin_v[body] = origin_v[parent_body] + np.cross(
                        omega[parent_body], root_rotation @ arm
                    )
                    omega[body] = omega[parent_body]
                    joint_twist = axes[joint] * state.v[6 + joint]
                    if children.joint_types[joint] == "revolute":
                        omega[body] += joint_twist
                    else:
                        origin_v[body] += joint_twist
                offsets = np.einsum("nij,nj->ni", rotation[1:], children.com)
                com_v[1:] = origin_v[1:] + np.cross(omega[1:], offsets)
            if not all(
                np.isfinite(field).all()
                for field in (position, rotation, com, origin_v, com_v, omega)
            ):
                raise FloatingPointError("non-finite floating geometry or twist")
    except (FloatingPointError, OverflowError) as exc:
        raise SolverFailure("floating poses or twists exceed FP64 range") from exc
    return FloatingKinematics(position, rotation, com, origin_v, com_v, omega)


def floating_point_jacobians(model, state, body_indices, local_points):
    """Return world linear and angular Jacobians with columns in public v order.

    Each point is expressed in its body's origin frame. A world force F at
    that point and couple T contribute J_linear.T @ F + J_angular.T @ T.
    """
    root_rotation, local = _root_and_children(model, state)
    indices = np.asarray(body_indices)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise TypeError("body_indices must be rank-one integers")
    if np.any(indices < 0) or np.any(indices >= model.body_count):
        raise ValueError("point body index outside model")
    points = robot_array(local_points, "local_points", ndim=2, columns=3)
    if len(points) != len(indices):
        raise ValueError("each point must have one body index")
    linear = np.zeros((len(points), 3, len(state.v)))
    angular = np.zeros_like(linear)
    linear[:, :, :3] = angular[:, :, 3:6] = np.eye(3)
    try:
        with np.errstate(over="raise", invalid="raise"):
            # At a root point, its body-local coordinate is already in the root frame.
            relative = points.copy()
            child_rows = indices != 0
            if np.any(child_rows):
                child_ids = indices[child_rows] - 1
                child_points = points[child_rows]
                relative[child_rows] = local.position[child_ids] + np.einsum(
                    "nij,nj->ni", local.rotation[child_ids], child_points
                )
                child_linear, child_angular = _point_jacobians_at(
                    model.children, local, child_ids, child_points
                )
                linear[child_rows, :, 6:] = root_rotation @ child_linear
                angular[child_rows, :, 6:] = root_rotation @ child_angular
            arms = (relative - model.root_com_m) @ root_rotation.T
            # One column per world basis spin, evaluated about the physical root COM.
            linear[:, :, 3:6] = np.cross(np.eye(3), arms[:, None, :]).transpose(0, 2, 1)
            if not np.isfinite(linear).all() or not np.isfinite(angular).all():
                raise FloatingPointError("non-finite floating Jacobian")
    except (FloatingPointError, OverflowError) as exc:
        raise SolverFailure("floating point Jacobian exceeds FP64 range") from exc
    return linear, angular
