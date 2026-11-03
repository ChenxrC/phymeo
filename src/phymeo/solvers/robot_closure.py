"""Ideal bilateral point closures on a fixed-base scalar tree, solved in the physical mass metric."""

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.result import StepReport
from ..core.robot import RobotState, validate_robot_step
from ..core.validation import finite_real, validate_dt
from ..dynamics.robot_dynamics import _finite, _solve_mass_matrix, mass_matrix, robot_dynamics_terms
from ..dynamics.robot_kinematics import _point_jacobians_at, forward_kinematics
from ..sensors import RobotSite


@dataclass(frozen=True, eq=False)
class RobotPointClosure:
    """Enforce axes_world @ (first_point - second_point) == 0.

    Each fixed orthonormal row is a direction in world space. A site with
    link_index=None is stationary in the world. Only site positions are used.
    """

    first: RobotSite
    second: RobotSite
    axes_world: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        if not isinstance(self.first, RobotSite) or not isinstance(self.second, RobotSite):
            raise TypeError("closure endpoints must be RobotSite values")
        if self.first.link_index is None and self.second.link_index is None:
            raise ValueError("at least one closure point must belong to a moving link")
        axes = np.asarray(self.axes_world)
        if (
            axes.dtype.kind not in "fiu"
            or axes.ndim != 2
            or axes.shape[1] != 3
            or not 1 <= len(axes) <= 3
        ):
            raise ValueError("axes_world must contain one to three real world direction rows")
        axes = axes.astype(np.float64)
        if not np.isfinite(axes).all() or not np.allclose(
            axes @ axes.T, np.eye(len(axes)), rtol=0, atol=1e-10
        ):
            raise ValueError("closure axes_world rows must be orthonormal")
        axes.flags.writeable = False
        object.__setattr__(self, "axes_world", axes)


@dataclass(frozen=True, eq=False)
class RobotClosureReaction:
    """Physical force and power at the beginning of the accepted physical step."""

    multipliers: np.ndarray
    force_first_world: np.ndarray
    force_second_world: np.ndarray
    power_first_w: np.ndarray
    power_second_w: np.ndarray
    generalized_force: np.ndarray
    acceleration: np.ndarray


@dataclass(frozen=True, eq=False)
class RobotClosureStep:
    state: RobotState
    report: StepReport
    reaction: RobotClosureReaction | None = None
    position_correction: np.ndarray | None = None
    velocity_projection_impulse: np.ndarray | None = None
    velocity_residual_m_s: float | None = None


def _rows(model, geom, closures, velocity=None):
    sites = [site for closure in closures for site in (closure.first, closure.second)]
    moving = np.array([s.link_index is not None for s in sites])
    indices = np.array([s.link_index for s in sites if s.link_index is not None], dtype=int)
    points = np.array([s.local_position_m for s in sites])
    jac = np.zeros((len(sites), 3, model.dof_count))
    local = points[moving]
    jac[moving], _ = _point_jacobians_at(model, geom, indices, local)
    points[moving] = geom.position[indices] + np.einsum("nij,nj->ni", geom.rotation[indices], local)

    def difference(values):
        return _finite(
            np.concatenate(
                [c.axes_world @ (values[2 * i] - values[2 * i + 1]) for i, c in enumerate(closures)]
            )
        )

    gap, rows = difference(points), difference(jac)
    if velocity is None:
        return gap, rows, None, None
    lv, av = _point_jacobians_at(
        model, geom, np.arange(model.dof_count), np.zeros((model.dof_count, 3))
    )
    origins_v, spins, point_v = lv @ velocity, av @ velocity, jac @ velocity
    bias = np.zeros_like(points)
    for i, site in enumerate(sites):
        joint = site.link_index
        while joint is not None and joint >= 0:
            parent = model.parents[joint]
            axis = geom.axis[joint]
            axis_dot = np.zeros(3) if parent < 0 else np.cross(spins[parent], axis)
            if model.joint_types[joint] == "revolute":
                derivative = np.cross(axis_dot, points[i] - geom.position[joint]) + np.cross(
                    axis, point_v[i] - origins_v[joint]
                )
            else:
                derivative = axis_dot
            bias[i] += derivative * velocity[joint]
            joint = int(parent)
    return gap, rows, difference(bias), _finite(point_v)


def _response(mass, rows, rhs):
    # W is the point-space inverse inertia. Normalize the rows before testing
    # rank so a difference in lever-arm length cannot by itself mimic a toggle.
    lifting = _solve_mass_matrix(mass, rows.T)
    w = _finite(rows @ lifting)
    diag = np.diag(w)
    if len(w) > len(mass) or np.any(diag <= 0):
        raise SolverFailure("closure Jacobian is rank deficient")
    scales = np.sqrt(diag)
    scaled = w / scales[:, None] / scales[None, :]
    scaled = 0.5 * scaled + 0.5 * scaled.T
    eigen = np.linalg.eigvalsh(scaled)
    if eigen[0] <= 1e-12 * eigen[-1]:
        raise SolverFailure("closure Jacobian is rank deficient")
    multiplier = _finite(_solve_mass_matrix(scaled, rhs / scales) / scales)
    return multiplier, _finite(lifting @ multiplier)


def step_robot_closures(
    model,
    state,
    inputs,
    closures,
    dt,
    *,
    position_tolerance_m=1e-10,
    velocity_tolerance_m_s=1e-10,
    max_iterations=12,
):
    """Advance one physical step from a closed, tangent state.

    Enforce the acceleration constraint with the full tree mass and its
    inertial bias. Correct the predicted coordinates with mass-weighted Newton
    iterations, then project the end velocity onto the final tangent space.
    StepReport.iterations counts position corrections. The velocity impulse
    and position correction are kept separate from the initial force sample.
    """
    validate_robot_step(model, state, inputs)
    h = validate_dt(dt, np.float64)
    pos_tol = finite_real(position_tolerance_m, "position_tolerance_m")
    vel_tol = finite_real(velocity_tolerance_m_s, "velocity_tolerance_m_s")
    if pos_tol <= 0 or vel_tol <= 0:
        raise ValueError("closure tolerances must be positive")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, Integral)
        or max_iterations < 0
    ):
        raise ValueError("max_iterations must be a non-negative integer")
    closures = tuple(closures)
    if not closures or any(not isinstance(c, RobotPointClosure) for c in closures):
        raise ValueError("one or more RobotPointClosure objects are required")
    if any(
        s.link_index is not None and s.link_index >= model.dof_count
        for c in closures
        for s in (c.first, c.second)
    ):
        raise ValueError("closure site index outside robot")
    count, error = 0, 0.0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            geom, linear, angular, mass, bias = robot_dynamics_terms(model, state)
            gap, rows, bias_acc, point_v = _rows(model, geom, closures, state.v)
            error = float(np.max(np.abs(gap)))
            if error > pos_tol:
                raise SolverFailure("initial closure position residual exceeds tolerance")
            if np.max(np.abs(rows @ state.v)) > vel_tol:
                raise SolverFailure("initial closure velocity residual exceeds tolerance")
            applied = (
                inputs.effort
                + np.einsum("nki,nk->i", linear, inputs.force)
                + np.einsum("nki,nk->i", angular, inputs.torque)
            )
            free_acc = _solve_mass_matrix(mass, _finite(applied - bias))
            lam, delta = _response(mass, rows, -(rows @ free_acc + bias_acc))
            acceleration = _finite(free_acc + delta)
            forces = []
            start = 0
            for c in closures:
                end = start + len(c.axes_world)
                forces.append(c.axes_world.T @ lam[start:end])
                start = end
            forces = _finite(np.asarray(forces))
            reaction = RobotClosureReaction(
                lam,
                forces,
                -forces,
                _finite(np.einsum("ij,ij->i", forces, point_v[::2])),
                _finite(-np.einsum("ij,ij->i", forces, point_v[1::2])),
                _finite(rows.T @ lam),
                acceleration,
            )
            predicted_v = _finite(state.v + h * acceleration)
            predicted_q = _finite(state.q + h * predicted_v)
            q = predicted_q.copy()
            while True:
                gap, rows, _, _ = _rows(model, forward_kinematics(model, q), closures)
                error = float(np.max(np.abs(gap)))
                end_mass = mass_matrix(model, q)
                if error <= pos_tol:
                    break
                if count >= max_iterations:
                    raise SolverFailure("closure position projection did not converge")
                _, correction = _response(end_mass, rows, -gap)
                q = _finite(q + correction)
                count += 1
            impulse_rows, correction = _response(end_mass, rows, -(rows @ predicted_v))
            v = _finite(predicted_v + correction)
            slip = float(np.max(np.abs(_finite(rows @ v))))
            if slip > vel_tol:
                raise SolverFailure("closure velocity projection did not converge")
            return RobotClosureStep(
                RobotState(q, v),
                StepReport(True, iterations=count, max_constraint_error=error),
                reaction,
                _finite(q - predicted_q),
                _finite(rows.T @ impulse_rows),
                slip,
            )
    except (SolverFailure, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        return RobotClosureStep(
            RobotState(state.q, state.v),
            StepReport(False, iterations=count, max_constraint_error=error, reason=str(exc)),
        )
