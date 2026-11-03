"""Bounded, weighted position and orientation IK for mounted sites on a fixed-base tree."""

from dataclasses import KW_ONLY, dataclass
from itertools import combinations
from numbers import Integral

import numpy as np

from ..constraints.robot_contact import JointLimits
from ..core.capabilities import SolverFailure
from ..core.robot import RobotModel, robot_array
from ..core.validation import finite_real
from ..dynamics.robot_kinematics import _point_jacobians_at, forward_kinematics
from ..dynamics.rotation import quaternion_matrix
from ..sensors import RobotSite
from ..sensors.robot_frame import _local_vector


@dataclass(frozen=True, eq=False)
class SiteIKTarget:
    """Optional world position plus either a full orientation or a single world direction.

    The direction is attached in site axes. Its roll remains free. Weights are
    residual multipliers; zero disables a channel while retaining its readout.
    """

    site: RobotSite
    _: KW_ONLY
    position_world_m: tuple | None = None
    orientation_world_xyzw: tuple | None = None
    direction_world: tuple | None = None
    local_direction: tuple = (1.0, 0.0, 0.0)
    position_weight: float = 1.0
    rotation_weight: float = 1.0

    def __post_init__(self):
        if not isinstance(self.site, RobotSite) or self.site.link_index is None:
            raise ValueError("an IK site must be attached to a robot link")
        if self.orientation_world_xyzw is not None and self.direction_world is not None:
            raise ValueError("choose either full orientation or a single direction")
        if (
            self.position_world_m is None
            and self.orientation_world_xyzw is None
            and self.direction_world is None
        ):
            raise ValueError("an IK target must have a position, orientation or direction")
        for name, size in (
            ("position_world_m", 3),
            ("orientation_world_xyzw", 4),
            ("direction_world", 3),
            ("local_direction", 3),
        ):
            value = getattr(self, name)
            if value is not None:
                value = _local_vector(value, size, name)
                if name != "position_world_m":
                    length = np.hypot.reduce(value)
                    if abs(length - 1.0) > 1e-10:
                        raise ValueError(f"{name} must be unit length")
                    value = tuple(np.asarray(value) / length)
                object.__setattr__(self, name, value)
        for name in ("position_weight", "rotation_weight"):
            value = finite_real(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class SiteIKResidual:
    position_m: float | None
    rotation_rad: float | None
    direction_rad: float | None


@dataclass(frozen=True, eq=False)
class MultiSiteIKResult:
    configuration: np.ndarray
    residuals: tuple[SiteIKResidual, ...]
    weighted_error: float
    iterations: int
    status: str
    active_bounds: tuple[int, ...]
    conflicting_pairs: tuple[tuple[int, int], ...]

    @property
    def converged(self):
        return self.status == "converged"


def _skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _rotvec(rotation):
    """Shortest world rotation; eigenspace is stable at a half-turn."""
    sine_axis = 0.5 * np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    sine = float(np.hypot.reduce(sine_axis))
    cosine = np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)
    angle = float(np.arctan2(sine, cosine))
    if cosine < 0 and sine < 1e-7:
        _, vectors = np.linalg.eigh((rotation + rotation.T) / 2)
        axis = vectors[:, -1]
        sign = axis @ sine_axis if sine > 1e-13 else axis[np.argmax(np.abs(axis))]
        return angle * (axis if sign >= 0 else -axis)
    return angle / sine * sine_axis if sine else sine_axis


def _direction_angle(a, b):
    return float(np.arctan2(np.hypot.reduce(np.cross(a, b)), np.clip(np.dot(a, b), -1.0, 1.0)))


def _conflicts(targets, mounts, pos_tol, ang_tol):
    """Necessary rigid-link invariants. Other local stalls do not certify infeasibility."""
    conflicts = []
    for i, j in combinations(range(len(targets)), 2):
        a, b = targets[i], targets[j]
        if a.site.link_index != b.site.link_index:
            continue
        bad = False
        if (
            a.position_world_m is not None
            and b.position_world_m is not None
            and a.position_weight
            and b.position_weight
        ):
            local = np.subtract(a.site.local_position_m, b.site.local_position_m)
            requested = np.subtract(a.position_world_m, b.position_world_m)
            bad = abs(np.hypot.reduce(local) - np.hypot.reduce(requested)) > 2 * pos_tol
        if a.rotation_weight and b.rotation_weight:
            bodies = [
                None
                if t.orientation_world_xyzw is None
                else quaternion_matrix(np.array([t.orientation_world_xyzw]))[0] @ mount.T
                for t, mount in ((a, mounts[i]), (b, mounts[j]))
            ]
            if all(r is not None for r in bodies):
                bad |= np.hypot.reduce(_rotvec(bodies[0] @ bodies[1].T)) > 2 * ang_tol
            elif a.direction_world is not None and b.direction_world is not None:
                before = _direction_angle(
                    mounts[i] @ a.local_direction, mounts[j] @ b.local_direction
                )
                after = _direction_angle(a.direction_world, b.direction_world)
                bad |= abs(before - after) > 2 * ang_tol
            else:
                for rotation, target, mount in (
                    (bodies[0], b, mounts[j]),
                    (bodies[1], a, mounts[i]),
                ):
                    if rotation is not None and target.direction_world is not None:
                        bad |= (
                            _direction_angle(
                                rotation @ mount @ target.local_direction, target.direction_world
                            )
                            > 2 * ang_tol
                        )
        if bad:
            conflicts.append((i, j))
    return tuple(conflicts)


def _evaluate(model, q, targets, ids, lower, upper, mounts, goal_rotations, pos_tol, ang_tol):
    geometry = forward_kinematics(model, q)
    links = np.array([t.site.link_index for t in targets])
    points = np.array([t.site.local_position_m for t in targets])
    linear, angular = _point_jacobians_at(model, geometry, links, points)
    residuals = []
    errors = []
    rows = []
    satisfied = True
    lo, hi = lower - q[ids], upper - q[ids]
    for k, (target, link) in enumerate(zip(targets, links, strict=True)):
        p = r = d = None
        rot = geometry.rotation[link] @ mounts[k]
        spin = angular[k][:, ids]
        if target.position_world_m is not None:
            e = target.position_world_m - (
                geometry.position[link] + geometry.rotation[link] @ points[k]
            )
            p = float(np.hypot.reduce(e))
            if target.position_weight:
                satisfied &= p <= pos_tol
                errors.append(target.position_weight * e)
                rows.append(target.position_weight * linear[k][:, ids])
        if target.orientation_world_xyzw is not None:
            vec = _rotvec(goal_rotations[k] @ rot.T)
            r = float(np.hypot.reduce(vec))
            if target.rotation_weight:
                satisfied &= r <= ang_tol
                if abs(r - np.pi) < 1e-13:
                    # Two shortest arcs coincide at pi. Choose the sign with
                    # more admissible joint motion at this configuration.
                    g = spin.T @ vec

                    def support(direction):
                        return np.maximum(direction, 0) @ np.minimum(hi, 1.0) + np.minimum(
                            direction, 0
                        ) @ np.maximum(lo, -1.0)

                    if support(-g) > support(g):
                        vec = -vec
                cross = _skew(vec)
                beta = (
                    1 / 12 + r * r / 720 if r < 1e-4 else (1 - 0.5 * r / np.tan(0.5 * r)) / (r * r)
                )
                rows.append(
                    target.rotation_weight
                    * ((np.eye(3) + 0.5 * cross + beta * cross @ cross) @ spin)
                )
                errors.append(target.rotation_weight * vec)
        if target.direction_world is not None:
            current = rot @ target.local_direction
            wanted = np.asarray(target.direction_world)
            d = _direction_angle(current, wanted)
            if target.rotation_weight:
                satisfied &= d <= ang_tol
                jac = -_skew(current) @ spin
                tangent = wanted - (current @ wanted) * current
                length = np.hypot.reduce(tangent)
                if d > np.pi - 1e-12 and length < 1e-12:
                    room = np.maximum(np.minimum(hi, 1.0), np.minimum(-lo, 1.0))
                    column = int(np.argmax(room * np.linalg.norm(jac, axis=0)))
                    tangent = jac[:, column] * (1 if hi[column] >= -lo[column] else -1)
                    length = np.hypot.reduce(tangent)
                    if length == 0:
                        basis = np.eye(3)[np.argmin(np.abs(current))]
                        tangent = np.cross(current, basis)
                        length = np.hypot.reduce(tangent)
                if length:
                    tangent = tangent * (d / length)
                errors.append(target.rotation_weight * tangent)
                rows.append(target.rotation_weight * jac)
        residuals.append(SiteIKResidual(p, r, d))
    e, jac = np.concatenate(errors), np.vstack(rows)
    if not np.isfinite(e).all() or not np.isfinite(jac).all():
        raise SolverFailure("non-finite site IK linearization")
    return e, jac, tuple(residuals), bool(satisfied)


def _bounded_delta(jac, error, lower, upper, damping):
    """Box-constrained damped least-squares step with active bound release."""
    x = np.zeros(jac.shape[1])
    active = np.zeros(len(x), dtype=int)
    for _ in range(4 * len(x) ** 2 + 4):
        free = active == 0
        trial = x.copy()
        if np.any(free):
            rhs = error - jac[:, ~free] @ x[~free]
            u, s, vt = np.linalg.svd(jac[:, free], full_matrices=False)
            scale = np.hypot(s, damping)
            trial[free] = vt.T @ ((s / scale) * ((u.T @ rhs) / scale))
        direction = trial - x
        below, above = trial < lower, trial > upper
        if np.any(below | above):
            fractions = np.full(len(x), np.inf)
            fractions[below] = (lower[below] - x[below]) / direction[below]
            fractions[above] = (upper[above] - x[above]) / direction[above]
            hit = int(np.argmin(fractions))
            x = np.clip(x + fractions[hit] * direction, lower, upper)
            active[hit] = -1 if below[hit] else 1
            x[hit] = lower[hit] if below[hit] else upper[hit]
            continue
        x = trial
        grad = jac.T @ (error - jac @ x) - damping * (damping * x)
        violations = -active * grad
        release = int(np.argmax(violations))
        if violations[release] <= 32 * np.finfo(float).eps * max(1.0, np.max(np.abs(grad))):
            break
        active[release] = 0
    return x


def solve_multi_site_ik(
    model,
    seed,
    targets,
    *,
    active_joints=None,
    limits=None,
    damping=0.001,
    max_step=0.25,
    tolerance_m=1e-7,
    tolerance_rad=1e-7,
    max_iterations=80,
) -> MultiSiteIKResult:
    """Find a local hard-bounded target and measure all requested channels by FK.

    Each positive-weight channel must meet its physical tolerance to converge.
    Inactive coordinates retain their seed values. A reported rigid-link conflict
    is a necessary incompatibility; an ordinary stall only describes this solve.
    """
    if not isinstance(model, RobotModel):
        raise TypeError("expected a fixed-base RobotModel")
    targets = tuple(targets)
    if not targets or any(
        not isinstance(t, SiteIKTarget) or t.site.link_index >= model.dof_count for t in targets
    ):
        raise ValueError("expected one or more targets on valid robot links")
    if not any(
        (t.position_world_m is not None and t.position_weight)
        or (
            (t.orientation_world_xyzw is not None or t.direction_world is not None)
            and t.rotation_weight
        )
        for t in targets
    ):
        raise ValueError("at least one target channel needs positive weight")
    q = robot_array(seed, "seed", ndim=1)
    if q.shape != (model.dof_count,):
        raise ValueError("seed must match robot coordinates")
    selected = tuple(range(len(q))) if active_joints is None else tuple(active_joints)
    if (
        not selected
        or any(
            isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or not 0 <= i < len(q)
            for i in selected
        )
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("active joints must be unique valid indices")
    ids = np.asarray(selected, dtype=np.intp)
    lower, upper = np.full(len(ids), -np.inf), np.full(len(ids), np.inf)
    if limits is not None:
        if not isinstance(limits, JointLimits) or limits.lower.shape != q.shape:
            raise ValueError("limits must match the robot")
        lower, upper = limits.lower[ids], limits.upper[ids]
        q[ids] = np.clip(q[ids], lower, upper)
    damping, max_step, tolerance_m, tolerance_rad = (
        finite_real(x, name)
        for x, name in (
            (damping, "damping"),
            (max_step, "max_step"),
            (tolerance_m, "tolerance_m"),
            (tolerance_rad, "tolerance_rad"),
        )
    )
    if min(damping, max_step, tolerance_m, tolerance_rad) <= 0:
        raise ValueError("damping, max_step and tolerances must be positive")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, Integral)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    mounts = quaternion_matrix(np.array([t.site.local_orientation_xyzw for t in targets]))
    goal_rotations = quaternion_matrix(
        np.array([t.orientation_world_xyzw or (0.0, 0, 0, 1) for t in targets])
    )

    def evaluate(configuration):
        return _evaluate(
            model,
            configuration,
            targets,
            ids,
            lower,
            upper,
            mounts,
            goal_rotations,
            tolerance_m,
            tolerance_rad,
        )

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            conflicts = _conflicts(targets, mounts, tolerance_m, tolerance_rad)
            e, j, residuals, converged = evaluate(q)
            for iteration in range(max_iterations + 1):
                merit = float(np.hypot.reduce(e))
                at_lower = q[ids] == lower
                at_upper = q[ids] == upper
                if converged or iteration == max_iterations:
                    status = "converged" if converged else "iteration_limit"
                    break
                step = _bounded_delta(j, e, lower - q[ids], upper - q[ids], damping)
                grad = j.T @ e
                outward = (at_lower & (grad < 0)) | (at_upper & (grad > 0))
                projected = grad.copy()
                projected[outward] = 0
                moved = False
                for direction in (step, projected):
                    length = np.hypot.reduce(direction)
                    if not length:
                        continue
                    if length > max_step:
                        direction = direction * (max_step / length)
                    for k in range(20):
                        candidate = q.copy()
                        candidate[ids] = np.clip(q[ids] + direction * 2.0 ** (-k), lower, upper)
                        if np.array_equal(candidate, q):
                            break
                        trial = evaluate(candidate)
                        if np.hypot.reduce(trial[0]) < merit:
                            q = candidate
                            e, j, residuals, converged = trial
                            moved = True
                            break
                    if moved:
                        break
                if not moved:
                    small = 64 * np.finfo(float).eps * max(1.0, np.linalg.norm(j) * merit)
                    status = (
                        "bound_limited"
                        if np.any(outward) and np.linalg.norm(projected) <= small
                        else "stalled"
                    )
                    break
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise SolverFailure(f"site IK arithmetic failed: {exc}") from exc
    q.flags.writeable = False
    return MultiSiteIKResult(
        q,
        residuals,
        merit,
        iteration,
        status,
        tuple(sorted(int(i) for i in ids[at_lower | at_upper])),
        conflicts,
    )
