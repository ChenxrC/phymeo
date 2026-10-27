"""World tree kinematics and ancestor Jacobians, independent of a simulator."""

from dataclasses import dataclass

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.robot import RobotModel, robot_array
from .rotation import quaternion_matrix


@dataclass(frozen=True)
class RobotKinematics:
    position: np.ndarray
    rotation: np.ndarray
    axis: np.ndarray
    com_position: np.ndarray


def _configuration(model, q):
    if not isinstance(model, RobotModel):
        raise TypeError("expected RobotModel")
    q = robot_array(q, "q", ndim=1)
    if q.shape != (model.dof_count,):
        raise ValueError("q must match the model DOF count")
    return q


def _axis_rotation(axis, angle):
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    cosine, sine = np.cos(angle), np.sin(angle)
    return cosine * np.eye(3) + (1 - cosine) * np.outer(axis, axis) + sine * skew


def forward_kinematics(model, q):
    q = _configuration(model, q)
    count = model.dof_count
    position, axis = np.empty((count, 3)), np.empty((count, 3))
    rotation = np.empty((count, 3, 3))
    zero = quaternion_matrix(model.joint_orientation)
    local_axis = model.joint_axis / np.hypot.reduce(model.joint_axis, axis=1, keepdims=True)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for i, parent in enumerate(model.parents):
                parent_rotation = np.eye(3) if parent == -1 else rotation[parent]
                parent_position = np.zeros(3) if parent == -1 else position[parent]
                frame = parent_rotation @ zero[i]
                axis[i] = frame @ local_axis[i]
                position[i] = parent_position + parent_rotation @ model.joint_position[i]
                if model.joint_types[i] == "revolute":
                    rotation[i] = frame @ _axis_rotation(local_axis[i], q[i])
                else:
                    rotation[i] = frame
                    position[i] += axis[i] * q[i]
            com = position + np.einsum("nij,nj->ni", rotation, model.com)
            if not all(np.isfinite(v).all() for v in (position, rotation, axis, com)):
                raise FloatingPointError("non-finite world tree geometry")
    except (FloatingPointError, OverflowError) as error:
        raise SolverFailure(f"robot geometry is not representable: {error}") from error
    return RobotKinematics(position, rotation, axis, com)


def point_jacobians(model, q, indices, local_points):
    geometry = forward_kinematics(model, q)
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise TypeError("point indices must be rank-one integers")
    if np.any(indices < 0) or np.any(indices >= model.dof_count):
        raise ValueError("point link index outside model")
    points = robot_array(local_points, "local_points", ndim=2, columns=3)
    if len(points) != len(indices):
        raise ValueError("each point must have a link index")
    linear = np.zeros((len(indices), 3, model.dof_count))
    angular = np.zeros_like(linear)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for row, link in enumerate(indices):
                world = geometry.position[link] + geometry.rotation[link] @ points[row]
                ancestor = int(link)
                while ancestor != -1:
                    axis = geometry.axis[ancestor]
                    if model.joint_types[ancestor] == "revolute":
                        linear[row, :, ancestor] = np.cross(
                            axis, world - geometry.position[ancestor]
                        )
                        angular[row, :, ancestor] = axis
                    else:
                        linear[row, :, ancestor] = axis
                    ancestor = int(model.parents[ancestor])
            if not np.isfinite(linear).all() or not np.isfinite(angular).all():
                raise FloatingPointError("non-finite point Jacobian")
    except (FloatingPointError, OverflowError) as error:
        raise SolverFailure(f"robot Jacobian is not representable: {error}") from error
    return linear, angular
