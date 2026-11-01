"""Damped least-squares position IK for a material point on a robot link."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..constraints.robot_contact import JointLimits
from ..core.capabilities import SolverFailure
from ..core.robot import RobotModel, robot_array
from ..core.validation import finite_real
from ..dynamics.robot_kinematics import forward_kinematics, point_jacobians


@dataclass(frozen=True, eq=False)
class PointIKResult:
    """A joint target and its actual FK residual, never a published robot state."""

    configuration: np.ndarray
    error_m: float
    iterations: int
    status: str

    def __post_init__(self):
        configuration = robot_array(self.configuration, "configuration", ndim=1)
        configuration.flags.writeable = False
        object.__setattr__(self, "configuration", configuration)

    @property
    def converged(self):
        return self.status == "converged"


def solve_point_ik(
    model,
    seed,
    link_index,
    local_point,
    target_position,
    *,
    active_joints=None,
    limits=None,
    damping=0.001,
    max_step=0.25,
    tolerance_m=1e-7,
    max_iterations=40,
):
    """Move a link-local point toward a world position on a fixed-base RobotModel.

    Updates only the declared scalar joints. Damping and the step bound use
    native joint coordinates (radians or metres). The result can be passed to
    a bounded actuator controller. A stalled solve returns its best local
    iterate and measured error, without declaring the target unreachable.
    Optional JointLimits constrain updated target coordinates only, including
    the seed target. The caller's state and inactive coordinates are unchanged.
    """
    if not isinstance(model, RobotModel):
        raise TypeError("expected RobotModel")
    if (
        not isinstance(link_index, Integral)
        or isinstance(link_index, bool)
        or not 0 <= link_index < model.dof_count
    ):
        raise ValueError("invalid link index")
    q = robot_array(seed, "seed", ndim=1)
    point = robot_array(local_point, "local_point", ndim=1)
    target = robot_array(target_position, "target_position", ndim=1)
    if q.shape != (model.dof_count,) or point.shape != (3,) or target.shape != (3,):
        raise ValueError("invalid seed or point shape")
    if active_joints is None:
        ids = np.arange(model.dof_count)
    else:
        chosen = tuple(active_joints)
        if (
            not chosen
            or any(
                isinstance(i, (bool, np.bool_))
                or not isinstance(i, Integral)
                or not 0 <= i < model.dof_count
                for i in chosen
            )
            or len(set(chosen)) != len(chosen)
        ):
            raise ValueError("active joints must be unique valid joint indices")
        ids = np.array(chosen, dtype=np.intp)
    if limits is not None:
        if not isinstance(limits, JointLimits) or limits.lower.shape != q.shape:
            raise ValueError("limits must match the robot")
        # Only a target is projected. Inactive physical coordinates remain unchanged.
        q[ids] = np.clip(q[ids], limits.lower[ids], limits.upper[ids])
    damping = finite_real(damping, "damping")
    max_step = finite_real(max_step, "max_step")
    tolerance_m = finite_real(tolerance_m, "tolerance_m")
    if min(damping, max_step, tolerance_m) <= 0:
        raise ValueError("damping, max_step and tolerance_m must be positive")
    if (
        not isinstance(max_iterations, Integral)
        or isinstance(max_iterations, bool)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")

    def residual(configuration):
        fk = forward_kinematics(model, configuration)
        error = target - (fk.position[link_index] + fk.rotation[link_index] @ point)
        return error, float(np.hypot.reduce(error))

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            error, norm = residual(q)
            for iteration in range(max_iterations + 1):
                if norm <= tolerance_m:
                    return PointIKResult(q, norm, iteration, "converged")
                if iteration == max_iterations:
                    return PointIKResult(q, norm, iteration, "iteration_limit")
                jac, _ = point_jacobians(model, q, [link_index], point[None, :])
                u, singular, vt = np.linalg.svd(jac[0][:, ids], full_matrices=False)
                scales = np.hypot(singular, damping)
                delta = vt.T @ ((singular / scales) * ((u.T @ error) / scales))
                length = np.hypot.reduce(delta)
                if length > max_step:
                    delta *= max_step / length
                for power in range(12):
                    candidate = q.copy()
                    candidate[ids] += delta / (2**power)
                    if limits is not None:
                        candidate[ids] = np.clip(
                            candidate[ids], limits.lower[ids], limits.upper[ids]
                        )
                    new_error, new_norm = residual(candidate)
                    if new_norm < norm:
                        q, error, norm = candidate, new_error, new_norm
                        break
                else:
                    return PointIKResult(q, norm, iteration, "stalled")
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise SolverFailure(f"point IK arithmetic failed: {exc}") from exc
