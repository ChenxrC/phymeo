"""Dense FP64 reference mechanics from world inertial loads and virtual work.

No optimized CRBA/ABA, damping, regularization or pseudo-inverse is implied.
External force and torque act at each COM in the world frame.
"""

from functools import wraps

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.robot import RobotInputs, robot_array, validate_robot_step
from .robot_kinematics import _point_jacobians_at, forward_kinematics, point_jacobians


def _finite(value):
    if not np.isfinite(value).all():
        raise FloatingPointError("non-finite robot result")
    return value


def _numerical(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = function(*args, **kwargs)
                # BLAS/einsum need not emit NumPy floating-point warnings.
                if isinstance(result, np.ndarray):
                    _finite(result)
                return result
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
            raise SolverFailure(f"robot dynamics failed: {error}") from error

    return checked


def _world_inertia(model, geometry):
    return _finite(
        (geometry.rotation * model.inertia_diagonal[:, None, :])
        @ geometry.rotation.transpose(0, 2, 1)
    )


def _com_jacobians(model, q):
    return point_jacobians(model, q, np.arange(model.dof_count), model.com)


@_numerical
def mass_matrix(model, q):
    """Generalized inertia, checked for numerical positive definiteness."""
    geometry = forward_kinematics(model, q)
    linear, angular = _com_jacobians(model, q)
    return _mass_from_terms(model, linear, angular, _world_inertia(model, geometry))


def _mass_from_terms(model, linear, angular, inertia):
    mass = np.zeros((model.dof_count, model.dof_count))
    for i in range(model.dof_count):
        mass += model.mass[i] * (linear[i].T @ linear[i])
        mass += angular[i].T @ inertia[i] @ angular[i]
    _finite(mass)
    if not np.allclose(mass, mass.T, rtol=1e-12, atol=0):
        raise SolverFailure("robot mass matrix lost symmetry")
    np.linalg.cholesky(mass)
    return mass


@_numerical
def solve_mass(model, q, load):
    """Solve M(q)*delta=load with an explicit normalized equation check."""
    load = robot_array(load, "generalized load", ndim=1)
    mass = mass_matrix(model, q)
    if load.shape != (model.dof_count,):
        raise ValueError("generalized load must match the robot DOF count")
    delta = _finite(np.linalg.solve(mass, load))
    residual = _finite(mass @ delta - load)
    scale = max(
        1.0,
        np.linalg.norm(load, ord=np.inf),
        np.linalg.norm(mass, ord=np.inf) * np.linalg.norm(delta, ord=np.inf),
    )
    _finite(scale)
    if np.linalg.norm(residual, ord=np.inf) / scale > 1e-10:
        raise SolverFailure("robot mass solve failed the normalized residual gate")
    return delta


@_numerical
def inverse_dynamics(model, state, acceleration):
    """Required generalized load for q/v/a, including gravity and inertial bias."""
    validate_robot_step(model, state, RobotInputs.zeros(model.dof_count))
    acceleration = robot_array(acceleration, "acceleration", ndim=1)
    if acceleration.shape != (model.dof_count,):
        raise ValueError("acceleration must match the robot DOF count")
    geometry = forward_kinematics(model, state.q)
    inertia = _world_inertia(model, geometry)
    linear, angular = _com_jacobians(model, state.q)
    return _inertial_load(model, state, acceleration, geometry, inertia, linear, angular)


def _inertial_load(model, state, acceleration, geometry, inertia, linear, angular):
    omega, alpha, origin_acc = (np.zeros((model.dof_count, 3)) for _ in range(3))
    forces, torques = np.zeros_like(omega), np.zeros_like(omega)
    zero = np.zeros(3)
    for i, parent in enumerate(model.parents):
        wp, ap = (zero, zero) if parent == -1 else (omega[parent], alpha[parent])
        parent_position = zero if parent == -1 else geometry.position[parent]
        parent_acc = zero if parent == -1 else origin_acc[parent]
        arm = geometry.position[i] - parent_position
        axis = geometry.axis[i]
        origin_acc[i] = parent_acc + np.cross(ap, arm) + np.cross(wp, np.cross(wp, arm))
        if model.joint_types[i] == "revolute":
            omega[i] = wp + axis * state.v[i]
            alpha[i] = ap + axis * acceleration[i] + np.cross(wp, axis * state.v[i])
        else:
            omega[i], alpha[i] = wp, ap
            origin_acc[i] += 2 * np.cross(wp, axis * state.v[i]) + axis * acceleration[i]
        com_arm = geometry.rotation[i] @ model.com[i]
        com_acc = (
            origin_acc[i]
            + np.cross(alpha[i], com_arm)
            + np.cross(omega[i], np.cross(omega[i], com_arm))
        )
        forces[i] = model.mass[i] * (com_acc - model.gravity)
        torques[i] = inertia[i] @ alpha[i] + np.cross(omega[i], inertia[i] @ omega[i])
    _finite(forces)
    _finite(torques)
    return np.einsum("nki,nk->i", linear, forces) + np.einsum("nki,nk->i", angular, torques)


def robot_com_kinematics(model, q):
    """Evaluate all COM Jacobians from the same world tree configuration."""
    geometry = forward_kinematics(model, q)
    linear, angular = _point_jacobians_at(model, geometry, np.arange(model.dof_count), model.com)
    return geometry, linear, angular


@_numerical
def robot_dynamics_terms(model, state):
    """Return FK, COM Jacobians, mass and bias for one physical state."""
    validate_robot_step(model, state, RobotInputs.zeros(model.dof_count))
    geometry, linear, angular = robot_com_kinematics(model, state.q)
    inertia = _world_inertia(model, geometry)
    mass = _mass_from_terms(model, linear, angular, inertia)
    bias = _inertial_load(
        model, state, np.zeros(model.dof_count), geometry, inertia, linear, angular
    )
    return geometry, linear, angular, mass, _finite(bias)


def bias_forces(model, state):
    return inverse_dynamics(model, state, np.zeros(model.dof_count))


@_numerical
def generalized_forces(model, state, inputs):
    validate_robot_step(model, state, inputs)
    linear, angular = _com_jacobians(model, state.q)
    return (
        inputs.effort
        + np.einsum("nki,nk->i", linear, inputs.force)
        + np.einsum("nki,nk->i", angular, inputs.torque)
    )


@_numerical
def forward_dynamics(model, state, inputs):
    applied = generalized_forces(model, state, inputs)
    load = _finite(applied - bias_forces(model, state))
    return solve_mass(model, state.q, load)
