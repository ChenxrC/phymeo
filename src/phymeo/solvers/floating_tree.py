"""Dense FP64 forward dynamics for one physical free root and scalar-joint children."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.result import StepReport, StepResult
from ..core.rigid import RigidBodyInputs
from ..core.robot import robot_array
from ..core.validation import validate_dt
from ..dynamics.floating_robot import (
    FloatingRobotModel,
    FloatingRobotState,
    floating_kinematics,
    floating_point_jacobians,
)
from ..dynamics.robot_dynamics import _finite, _mass_from_terms, _numerical, _solve_mass_matrix
from ..dynamics.rotation import advance_world_orientation, quaternion_matrix
from .free_root import step_free_root


@dataclass(frozen=True, eq=False)
class FloatingRobotInputs:
    """Internal effort per scalar joint and external world force/torque about each body COM."""

    effort: np.ndarray
    force: np.ndarray
    torque: np.ndarray

    def __post_init__(self):
        effort = robot_array(self.effort, "effort", ndim=1)
        force = robot_array(self.force, "force", ndim=2, columns=3)
        torque = robot_array(self.torque, "torque", ndim=2, columns=3)
        if force.shape != (len(effort) + 1, 3) or torque.shape != force.shape:
            raise ValueError("expected one load per body and one effort per scalar joint")
        for name, value in (("effort", effort), ("force", force), ("torque", torque)):
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    @classmethod
    def zeros(cls, child_count):
        if (
            isinstance(child_count, (bool, np.bool_))
            or not isinstance(child_count, Integral)
            or child_count < 0
        ):
            raise ValueError("child_count must be a non-negative integer")
        return cls(
            np.zeros(child_count), np.zeros((child_count + 1, 3)), np.zeros((child_count + 1, 3))
        )


@dataclass(frozen=True, eq=False)
class FloatingDynamicsTerms:
    """M a + bias = applied in the public world COM/angular/child velocity basis."""

    mass_matrix: np.ndarray
    bias: np.ndarray
    applied: np.ndarray
    acceleration: np.ndarray


def _validate(model, state, inputs):
    if not isinstance(model, FloatingRobotModel) or not isinstance(state, FloatingRobotState):
        raise TypeError("expected FloatingRobotModel and FloatingRobotState")
    if not isinstance(inputs, FloatingRobotInputs):
        raise TypeError("expected FloatingRobotInputs")
    if len(state.v) != 5 + model.body_count or len(inputs.force) != model.body_count:
        raise ValueError("state and external/actuation input sizes must match the model")
    if model.children is not None and not np.array_equal(
        model.root.gravity, model.children.gravity
    ):
        raise ValueError("root and children must share the same world gravity")


def _bias_accelerations(model, state, obs, local_com):
    """World COM and angular accelerations with derivatives of all public velocities zero."""
    alpha, origin_acc, com_acc = (np.zeros((model.body_count, 3)) for _ in range(3))
    omega = obs.angular_velocity
    offsets = np.einsum("nij,nj->ni", obs.rotation, local_com)
    # Root COM has zero prescribed acceleration. Its eccentric origin generally does not.
    origin_acc[0] = -np.cross(omega[0], np.cross(omega[0], offsets[0]))
    children = model.children
    if children is None:
        return com_acc, alpha
    for joint, parent_index in enumerate(children.parents):
        child, parent = joint + 1, parent_index + 1
        axis = obs.rotation[child] @ children.joint_axis[joint]
        axis /= np.linalg.norm(children.joint_axis[joint])
        arm = obs.rotation[parent] @ children.joint_position[joint]
        slide = children.joint_types[joint] == "prismatic"
        if slide:
            arm += axis * state.q[7 + joint]
        spin = omega[parent]
        origin_acc[child] = (
            origin_acc[parent] + np.cross(alpha[parent], arm) + np.cross(spin, np.cross(spin, arm))
        )
        rate = axis * state.v[6 + joint]
        alpha[child] = alpha[parent]
        if slide:
            origin_acc[child] += 2 * np.cross(spin, rate)
        else:
            alpha[child] += np.cross(spin, rate)
        com_acc[child] = (
            origin_acc[child]
            + np.cross(alpha[child], offsets[child])
            + np.cross(omega[child], np.cross(omega[child], offsets[child]))
        )
    return com_acc, alpha


@_numerical
def floating_dynamics_terms(model, state, inputs):
    """Return full inertia, inertial/gravity bias, projected external/drive loads and acceleration."""
    _validate(model, state, inputs)
    masses = 1 / model.root.inverse_mass
    local_com = model.root_com_m[None]
    principal = model.root.inertia_diagonal
    if model.children is not None:
        masses = np.concatenate((masses, model.children.mass))
        local_com = np.vstack((local_com, model.children.com))
        principal = np.vstack((principal, model.children.inertia_diagonal))
    obs = floating_kinematics(model, state)
    linear, angular = floating_point_jacobians(model, state, np.arange(model.body_count), local_com)
    inertia = (obs.rotation * principal[:, None, :]) @ obs.rotation.transpose(0, 2, 1)
    mass = _mass_from_terms(masses, linear, angular, _finite(inertia))
    com_acc, alpha = _bias_accelerations(model, state, obs, local_com)
    forces = masses[:, None] * (com_acc - model.root.gravity)
    spin_momentum = np.einsum("nij,nj->ni", inertia, obs.angular_velocity)
    torques = np.einsum("nij,nj->ni", inertia, alpha) + np.cross(
        obs.angular_velocity, spin_momentum
    )

    def project(force, torque):
        return np.einsum("nki,nk->i", linear, force) + np.einsum("nki,nk->i", angular, torque)

    bias = _finite(project(forces, torques))
    applied = project(inputs.force, inputs.torque)
    applied[6:] += inputs.effort
    acceleration = _solve_mass_matrix(mass, _finite(applied - bias))
    return FloatingDynamicsTerms(mass, bias, applied, acceleration)


def step_floating_robot(model, state, inputs, dt) -> StepResult[FloatingRobotState]:
    """Take one velocity-first physical step for the full coupled tree, with root COM drift."""
    _validate(model, state, inputs)
    h = validate_dt(dt, np.float64)
    if model.children is None:
        return step_free_root(model, state, RigidBodyInputs(inputs.force, inputs.torque), h)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            terms = floating_dynamics_terms(model, state, inputs)
            v = _finite(state.v + h * terms.acceleration)
            old_quat = state.q[None, 3:7]
            new_quat = advance_world_orientation(old_quat, v[None, 3:6], h)
            orientation_change = quaternion_matrix(new_quat)[0] - quaternion_matrix(old_quat)[0]
            origin = state.q[:3] + (h * v[:3] - orientation_change @ model.root_com_m)
            q = _finite(np.concatenate((origin, new_quat[0], state.q[7:] + h * v[6:])))
            candidate = FloatingRobotState(q, v)
            floating_kinematics(model, candidate)
    except (FloatingPointError, OverflowError, SolverFailure) as exc:
        return StepResult(state, StepReport(accepted=False, reason=str(exc)))
    return StepResult(candidate, StepReport(accepted=True))
