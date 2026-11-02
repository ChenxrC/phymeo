"""Frictionless backing impulses for the finite-pad particle/articulated step."""

from dataclasses import dataclass

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.robot import RobotState
from ..core.scene import SceneState
from ..core.state import ParticleState
from ..geometry.pad_columns import check_column_segment
from ._compliant_motion import check_robot_contact


@dataclass(frozen=True)
class BackingProjection:
    velocity: np.ndarray
    impulses: np.ndarray
    residual_m_s: float
    work_j: float
    sweeps: int


def project_backing_velocity(rows, weighted_transpose, free_velocity, target):
    """Project the free drift velocity with all unilateral normal owners in one solve."""
    b, wbt, u, t = (
        np.asarray(a, dtype=np.float64) for a in (rows, weighted_transpose, free_velocity, target)
    )
    if b.ndim != 2 or wbt.shape != b.T.shape or u.shape != (b.shape[1],) or t.shape != (len(b),):
        raise ValueError("incompatible backing response shapes")
    if any(not np.isfinite(x).all() for x in (b, wbt, u, t)) or np.any(t > 0):
        raise ValueError("invalid backing response or closing limit")
    if len(b) > 128:
        raise SolverFailure("finite-pad backing owner capacity")
    impulses = np.zeros(len(b))
    free = b @ u
    speed = free.copy()
    if np.all(speed >= t):
        return BackingProjection(u.copy(), impulses, 0.0, 0.0, 0)
    response = b @ wbt
    diag = response.diagonal()
    if (
        not np.isfinite(response).all()
        or np.any(diag <= 0)
        or not np.allclose(response, response.T, rtol=1e-12, atol=1e-12)
    ):
        raise SolverFailure("inadmissible finite-pad backing mobility")
    for sweep in range(1, 129):
        for i in range(len(b)):
            new = max(0.0, impulses[i] - (speed[i] - t[i]) / diag[i])
            speed += (new - impulses[i]) * response[:, i]
            impulses[i] = new
        velocity = u + wbt @ impulses
        speed = b @ velocity
        residual = float(np.max(abs(np.minimum(diag * impulses, speed - t))))
        if residual <= 1e-8:
            work = float(0.5 * impulses @ (free + speed))
            if (
                not np.isfinite(velocity).all()
                or not np.isfinite(work)
                or work
                > 1e-8 * float(impulses.sum())
                + 32 * np.finfo(float).eps * max(1.0, abs(float(impulses @ free)))
            ):
                raise SolverFailure("finite-pad backing kinetic work")
            return BackingProjection(velocity, impulses, residual, work, sweep)
    raise SolverFailure("finite-pad backing velocity iterations exhausted")


def _project(program, state, accel, contacts, h):
    cloth = state.domains[program.cloth_domain]
    size = cloth.position.size
    velocities = [(cloth.velocity + h * accel[program.cloth_domain]).ravel()]
    robot_slices = {}
    for name in program.robots:
        speed = state.domains[name].v + h * accel[name]
        robot_slices[name] = slice(size, size + len(speed))
        size += len(speed)
        velocities.append(speed)
    free = np.concatenate(velocities)
    count = sum(
        len(contacts[p.geometry.key].get("_backing", {}).get("indices", ()))
        for p in program.backed_pads
    )
    if count > 128:
        raise SolverFailure("finite-pad backing owner capacity")
    b = np.zeros((count, size))
    target = np.empty(count)
    groups = []
    masses = {}
    offset = 0
    for pad in program.backed_pads:
        key = pad.geometry.key
        row = contacts[key]
        data = row.pop("_backing", None)
        row["penalty_normal_force_n"] = row["normal_force_n"].copy()
        row["backing_impulse_ns"] = np.zeros(len(row["gap_m"]))
        if data is None:
            continue
        ids, blocks, robot_row, mass = (
            data["indices"],
            data["particle"],
            data["robot"],
            data["mass"],
        )
        sl = slice(offset, offset + len(ids))
        offset = sl.stop
        for j in range(3):
            coords = 3 * program.ids[ids, j, None] + np.arange(3)
            b[np.arange(sl.start, sl.stop)[:, None], coords] += blocks[:, j]
        if pad.robot_domain is not None:
            b[sl, robot_slices[pad.robot_domain]] = robot_row
            masses[pad.robot_domain] = mass
        target[sl] = -np.maximum(row["middle_distance_m"][ids] - pad.backing_clearance_m, 0.0) / h
        groups.append((row, ids, sl))
    wbt = np.zeros((size, count))
    if count and np.any(b @ free < target):
        wbt[: cloth.position.size] = (
            np.repeat(program.cloth.inverse_mass, 3)[:, None] * b[:, : cloth.position.size].T
        )
        for name, mass in masses.items():
            sl = robot_slices[name]
            try:
                wbt[sl] = np.linalg.solve(mass, b[:, sl].T)
            except np.linalg.LinAlgError as error:
                raise SolverFailure("singular backing joint inertia") from error
    projection = project_backing_velocity(b, wbt, free, target)
    for row, ids, sl in groups:
        impulse = projection.impulses[sl]
        row["backing_impulse_ns"][ids] = impulse
        row["normal_force_n"][ids] += impulse / h
        row["force_n"][ids] += row["site_normals"][ids] * ((impulse / h)[:, None])
    v = projection.velocity

    def drift(dt):
        velocity = v[: cloth.position.size].reshape(cloth.position.shape)
        domains = {program.cloth_domain: ParticleState(cloth.position + dt * velocity, velocity)}
        for name, sl in robot_slices.items():
            original = state.domains[name]
            domains[name] = RobotState(original.q + dt * v[sl], v[sl])
        return SceneState(domains)

    return drift, dict(
        owners=count,
        sweeps=projection.sweeps,
        residual_m_s=projection.residual_m_s,
        work_j=projection.work_j,
        total_impulse_ns=float(projection.impulses.sum()),
    )


def backed_step(program, state, inputs, h):
    """Semi-implicit force kick, joint unilateral velocity solve, and checked common drift."""
    accel, initial, internal, body = program.accelerations(state, inputs, _backing=True)
    drift, certificate = _project(program, state, accel, initial, h)
    middle, end = drift(h / 2), drift(h)
    _, _, mid, mid_geo = program._loads(middle, dynamics=False)
    _, _, last, end_geo = program._loads(end, dynamics=False)
    for pad in program.finite_pads:
        key = pad.geometry.key
        for a, b in ((initial[key], mid[key]), (mid[key], last[key])):
            check_column_segment(
                a["column_ends_local_m"],
                a["middle_distance_m"],
                b["column_ends_local_m"],
                b["middle_distance_m"],
            )
    penetration = max(
        float(np.max(np.maximum(-r["gap_m"][r["in_patch"]], 0.0), initial=0.0))
        for rows in (initial, mid, last)
        for r in rows.values()
    )
    if body is not None:
        midbody, _ = program._body_sample(middle, mid_geo)
        endbody, _ = program._body_sample(end, end_geo)
        program._check_link_motion(state, middle, end, mid_geo, end_geo)
        bodies = (body, midbody, endbody)
        check_robot_contact(bodies, program.max_penetration_m, program.max_joint_violation)
        penetration = max(
            penetration,
            *(max(x["penetration_m"], x.get("cloth_link_penetration_m", 0.0)) for x in bodies),
        )
    if program.max_penetration_m is not None and penetration > program.max_penetration_m:
        raise SolverFailure(f"sampled hard penetration limit exceeded: {penetration:.6g} m")
    return end, dict(
        patches=initial,
        robot_contact=body,
        material_force_n=internal,
        sampled_penetration_m=penetration,
        force_sample="substep-entry",
        backing=certificate,
    )
