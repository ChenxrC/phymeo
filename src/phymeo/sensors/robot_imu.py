"""Ideal mounted site accelerometer and gyroscope sampled at one physical instant."""

from dataclasses import dataclass

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.robot import RobotModel, RobotState, robot_array
from ..core.validation import finite_real
from ..dynamics.floating_robot import (
    FloatingRobotModel,
    FloatingRobotState,
    floating_kinematics,
    floating_point_jacobians,
)
from ..dynamics.robot_kinematics import forward_kinematics
from ..dynamics.rotation import quaternion_matrix
from ..solvers.floating_tree import _bias_accelerations
from .robot_frame import RobotSite


@dataclass(frozen=True, eq=False)
class RobotIMUReading:
    """Sample timestamp [s], site-axis specific force [m/s²] and angular rate [rad/s]."""

    time_s: float
    specific_force_m_s2: np.ndarray
    angular_velocity_rad_s: np.ndarray


def _fixed_com_motion(model, state, acceleration, body):
    geometry = forward_kinematics(model, state.q)
    path = []
    link = body
    while link >= 0:
        path.append(link)
        link = model.parents[link]
    origin_acc, alpha, omega = np.zeros((3, 3))
    for joint in reversed(path):
        parent = model.parents[joint]
        arm = model.joint_position[joint].copy()
        if parent >= 0:
            arm = geometry.rotation[parent] @ arm
        axis = geometry.axis[joint]
        slide = model.joint_types[joint] == "prismatic"
        if slide:
            arm += axis * state.q[joint]
        origin_acc += np.cross(alpha, arm) + np.cross(omega, np.cross(omega, arm))
        rate = axis * state.v[joint]
        if slide:
            origin_acc += axis * acceleration[joint] + 2 * np.cross(omega, rate)
        else:
            alpha += axis * acceleration[joint] + np.cross(omega, rate)
            omega += rate
    rotation = geometry.rotation[body]
    com = model.com[body]
    offset = rotation @ com
    com_acc = origin_acc + np.cross(alpha, offset) + np.cross(omega, np.cross(omega, offset))
    return rotation, com, com_acc, alpha, omega


def _floating_com_motion(model, state, acceleration, body):
    geometry = floating_kinematics(model, state)
    if body == 0:
        return (
            geometry.rotation[0],
            model.root_com_m,
            acceleration[:3],
            acceleration[3:6],
            geometry.angular_velocity[0],
        )
    # The floating-tree dynamics uses the same COM/world velocity basis:
    # world body acceleration = J @ vdot + the zero-vdot convective acceleration.
    com = np.vstack((model.root_com_m, model.children.com))
    bias, angular_bias = _bias_accelerations(model, state, geometry, com)
    linear, angular = floating_point_jacobians(model, state, [body], com[body : body + 1])
    return (
        geometry.rotation[body],
        com[body],
        linear[0] @ acceleration + bias[body],
        angular[0] @ acceleration + angular_bias[body],
        geometry.angular_velocity[body],
    )


def measure_robot_imu(
    model, state, generalized_acceleration, site: RobotSite, *, time_s
) -> RobotIMUReading:
    """Sample q, v and vdot at time_s, using the site-to-body XYZW mount.

    For a fixed tree, site indices are scalar-link indices. For a floating
    tree, zero is the physical root and children start at one. Floating vdot
    starts with the root's world COM acceleration and world angular acceleration.
    A world site is stationary. Supply accelerations from the same instant as
    the state; a closure step's reported acceleration belongs with its input state.
    """
    if not isinstance(site, RobotSite):
        raise TypeError("site must be a RobotSite")
    timestamp = finite_real(time_s, "time_s")
    acceleration = robot_array(generalized_acceleration, "generalized_acceleration", ndim=1)
    if isinstance(model, RobotModel) and isinstance(state, RobotState):
        sample = RobotState(state.q, state.v)
        count = dofs = model.dof_count
        gravity = model.gravity
        motion = _fixed_com_motion
    elif isinstance(model, FloatingRobotModel) and isinstance(state, FloatingRobotState):
        sample = FloatingRobotState(state.q, state.v)
        count = model.body_count
        dofs = count + 5
        gravity = model.root.gravity
        if model.children is not None and not np.array_equal(gravity, model.children.gravity):
            raise ValueError("root and children must share the same world gravity")
        motion = _floating_com_motion
    else:
        raise TypeError("expected matching fixed or floating robot model and state")
    if len(sample.v) != dofs or acceleration.shape != (dofs,):
        raise ValueError("state and generalized acceleration must match model velocities")
    body = site.link_index
    if body is not None and body >= count:
        raise ValueError("IMU site link index outside model")
    mount = quaternion_matrix(np.array([site.local_orientation_xyzw]))[0]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if body is None:
                specific, gyro = mount.T @ (-gravity), np.zeros(3)
            else:
                rotation, com, com_acc, alpha, omega = motion(model, sample, acceleration, body)
                arm = rotation @ (np.asarray(site.local_position_m) - com)
                site_acc = com_acc + np.cross(alpha, arm) + np.cross(omega, np.cross(omega, arm))
                world_to_site = (rotation @ mount).T
                specific, gyro = world_to_site @ (site_acc - gravity), world_to_site @ omega
            if not np.isfinite(specific).all() or not np.isfinite(gyro).all():
                raise FloatingPointError("non-finite IMU value")
    except (FloatingPointError, OverflowError) as exc:
        raise SolverFailure("IMU reading exceeds FP64 range") from exc
    specific.flags.writeable = gyro.flags.writeable = False
    return RobotIMUReading(timestamp, specific, gyro)
