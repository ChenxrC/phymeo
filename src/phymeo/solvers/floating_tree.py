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


def _inertial_geometry(model, state):
    masses = 1 / model.root.inverse_mass
    local_com = model.root_com_m[None]
    principal = model.root.inertia_diagonal
    if model.children is not None:
        masses = np.concatenate((masses, model.children.mass))
        local_com = np.vstack((local_com, model.children.com))
        principal = np.vstack((principal, model.children.inertia_diagonal))
    obs = floating_kinematics(model, state)
    linear, angular = floating_point_jacobians(model, state, np.arange(model.body_count), local_com)
    inertia_frames = obs.rotation
    if np.any(model.root_inertia_frame_xyzw[:3]):
        inertia_frames = obs.rotation.copy()
        inertia_frames[0] = (
            obs.rotation[0] @ quaternion_matrix(model.root_inertia_frame_xyzw[None, :])[0]
        )
    inertia = (inertia_frames * principal[:, None, :]) @ inertia_frames.transpose(0, 2, 1)
    return obs, local_com, masses, linear, angular, _finite(inertia)


def _with_joint_armature(model, mass):
    if np.any(model.joint_armature):
        indices = np.arange(model.body_count - 1) + 6
        mass[indices, indices] += model.joint_armature
        _finite(mass)
    return mass


@_numerical
def floating_mass_matrix(model, state):
    """Full inertia at this pose in root COM, root angular and joint-rate coordinates."""
    _, _, masses, linear, angular, inertia = _inertial_geometry(model, state)
    return _with_joint_armature(model, _mass_from_terms(masses, linear, angular, inertia))


def _with_implicit_joint_damping(model, mass, h):
    if np.any(model.joint_damping):
        mass = mass.copy()
        indices = np.arange(model.body_count - 1) + 6
        mass[indices, indices] += h * model.joint_damping
        _finite(mass)
    return mass


@_numerical
def floating_step_mass_matrix(model, state, dt):
    """Entry M+hD for the implicit-damping velocity update, not kinetic energy."""
    h = validate_dt(dt, np.float64)
    return _with_implicit_joint_damping(model, floating_mass_matrix(model, state), h)


@_numerical
def floating_kinetic_energy(model, state):
    """Effective kinetic energy, including reflected joint inertia.

    The additional energy is .5*sum(armature*qdot**2). This function does not
    reinterpret the body-only linear/angular momentum as resolved rotor momentum.
    """
    mass = floating_mass_matrix(model, state)
    return float(_finite(np.asarray(0.5 * (state.v @ mass @ state.v))))


@_numerical
def floating_dynamics_terms(model, state, inputs):
    """Return full inertia, inertial/gravity bias, projected external/drive loads and acceleration."""
    _validate(model, state, inputs)
    obs, local_com, masses, linear, angular, inertia = _inertial_geometry(model, state)
    mass = _with_joint_armature(model, _mass_from_terms(masses, linear, angular, inertia))
    com_acc, alpha = _bias_accelerations(model, state, obs, local_com)
    forces = masses[:, None] * (com_acc - model.root.gravity)
    spin_momentum = np.einsum("nij,nj->ni", inertia, obs.angular_velocity)
    torques = np.einsum("nij,nj->ni", inertia, alpha) + np.cross(
        obs.angular_velocity, spin_momentum
    )

    def project(force, torque):
        return np.einsum("nki,nk->i", linear, force) + np.einsum("nki,nk->i", angular, torque)

    bias = _finite(project(forces, torques))
    if np.any(model.joint_damping):
        bias[6:] += model.joint_damping * state.v[6:]
        _finite(bias)
    applied = project(inputs.force, inputs.torque)
    applied[6:] += inputs.effort
    acceleration = _solve_mass_matrix(mass, _finite(applied - bias))
    return FloatingDynamicsTerms(mass, bias, applied, acceleration)


def _drift_floating_state(model, state, velocity, dt):
    """Drift the root COM, world rotation and child scalars once using one final velocity."""
    old_quat = state.q[None, 3:7]
    new_quat = advance_world_orientation(old_quat, velocity[None, 3:6], dt)
    orientation_change = quaternion_matrix(new_quat)[0] - quaternion_matrix(old_quat)[0]
    origin = state.q[:3] + (dt * velocity[:3] - orientation_change @ model.root_com_m)
    q = _finite(np.concatenate((origin, new_quat[0], state.q[7:] + dt * velocity[6:])))
    candidate = FloatingRobotState(q, velocity)
    floating_kinematics(model, candidate)
    return candidate


def step_floating_robot(model, state, inputs, dt) -> StepResult[FloatingRobotState]:
    """Take one velocity-first step, treating passive linear damping implicitly.

    Geometry/bias stay at entry, while (M+hD)*delta_v=h*(applied-bias) evaluates
    viscous resistance at the updated velocity. Zero D retains the old path.
    """
    _validate(model, state, inputs)
    h = validate_dt(dt, np.float64)
    if model.children is None:
        return step_free_root(model, state, RigidBodyInputs(inputs.force, inputs.torque), h)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            terms = floating_dynamics_terms(model, state, inputs)
            if np.any(model.joint_damping):
                effective = _with_implicit_joint_damping(model, terms.mass_matrix, h)
                delta = _solve_mass_matrix(
                    _finite(effective), _finite(h * (terms.applied - terms.bias))
                )
                v = _finite(state.v + delta)
            else:
                v = _finite(state.v + h * terms.acceleration)
            candidate = _drift_floating_state(model, state, v, h)
    except (FloatingPointError, OverflowError, SolverFailure) as exc:
        return StepResult(state, StepReport(accepted=False, reason=str(exc)))
    return StepResult(candidate, StepReport(accepted=True))
