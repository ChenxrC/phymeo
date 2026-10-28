"""Exact affine-plane enclosures; no candidate generation or state repair."""

from fractions import Fraction

import numpy as np

from ..core.capabilities import SolverFailure
from ..dynamics.robot_kinematics import forward_kinematics, point_jacobians
from ..validation._material_path_math import cross, direction_bounds, norm_bounds, q
from ._robot_replay_ledger import audit_affine_geometry, dot, exact_float

ZERO = Fraction(0)


def geometry(workspace, state):
    w = workspace
    robot = state.domains[w.domain]
    refs = audit_affine_geometry(
        w.model,
        robot,
        np.r_[w.indices, w.indices],
        np.concatenate((w.local, w.anchors)),
        w.h,
        w.settings,
    )[len(w.rows) :]
    fk = forward_kinematics(w.model, robot.q)
    anchors = fk.position[w.indices] + np.sum(
        fk.rotation[w.indices] * w.anchors[:, None, :], axis=2
    )
    normals = np.sum(fk.rotation[w.indices] * w.local_normals[:, None, :], axis=2)
    if len(normals):
        normals /= np.hypot.reduce(normals, axis=1)[:, None]
    jac = point_jacobians(w.model, robot.q, w.indices, w.local)[0]
    bounds = []
    for row, anchor, ref, n in zip(w.rows, anchors, refs, normals, strict=True):
        point = state.domains[row[0]].position[row[1]]
        nl, nh = norm_bounds(list(map(q, n)))
        if nl <= 0 or max(abs(nl - 1), abs(nh - 1)) > q(min(w.settings.tolerance, 1e-10)):
            raise SolverFailure("normal direction exceeds precision budget")
        values = [
            sum((q(n[k]) * (q(point[k]) - a[k]) for k in range(3)), ZERO)
            for a in (list(map(q, anchor)), ref)
        ]
        values = [v / length - q(row[2].radius_m) for v in values for length in (nl, nh)]
        raw = [dot(fk.rotation[row[2].link_index, k], row[2].local_normal) for k in range(3)]
        _, directions = direction_bounds(raw)
        error = sum(
            max(abs(q(n[k]) - lo), abs(q(n[k]) - hi)) for k, (lo, hi) in enumerate(directions)
        )
        if error > q(min(w.settings.tolerance, 1e-10)):
            raise SolverFailure("rotated normal exceeds direction precision budget")
        interval = [ZERO, ZERO]
        for k, (lo, hi) in enumerate(directions):
            delta = q(point[k]) - ref[k]
            products = (lo * delta, hi * delta)
            interval[0] += min(products)
            interval[1] += max(products)
        values += [v - q(row[2].radius_m) for v in interval]
        bounds.append((min(values), max(values)))
    return (
        anchors,
        jac,
        normals,
        np.array([exact_float((a + b) / 2) for a, b in bounds]),
        tuple(bounds),
    )


def relative(workspace, state):
    robot = state.domains[workspace.domain]
    values = []
    for row, jac, normal in zip(workspace.rows, workspace.j0, workspace.normals, strict=True):
        velocity = state.domains[row[0]].velocity[row[1]]
        values.append(
            sum((q(normal[k]) * (q(velocity[k]) - dot(jac[k], robot.v)) for k in range(3)), ZERO)
        )
    return tuple(values)


def project_local(workspace, state):
    """Project the actual node point to its infinite plane at this configuration."""
    fk = forward_kinematics(workspace.model, state.domains[workspace.domain].q)
    result = []
    for name, i, c in workspace.rows:
        point = fk.rotation[c.link_index].T @ (
            state.domains[name].position[i] - fk.position[c.link_index]
        )
        result.append(point - c.local_normal * float(c.local_normal @ (point - c.local_point)))
    return np.array(result).reshape(-1, 3)


def paired_moment(workspace, state, local, world):
    references = audit_affine_geometry(
        workspace.model,
        state.domains[workspace.domain],
        workspace.indices,
        local,
        workspace.h,
        workspace.settings,
    )
    moment = ZERO
    for row, point, p in zip(workspace.rows, references, world, strict=True):
        delta = [q(state.domains[row[0]].position[row[1], k]) - point[k] for k in range(3)]
        moment += sum(map(abs, cross(delta, list(map(q, p)))))
    magnitude = sum((abs(q(p)) for p in world.flat), ZERO)
    if magnitude:
        residual = moment / (magnitude * workspace.exact_gap_scale)
    elif moment:
        raise SolverFailure("zero contact impulse has nonzero paired moment")
    else:
        residual = ZERO
    return moment, residual


def normal_projection_bounds(normal, vector):
    """Project actual stored vectors onto an exactly unit normal enclosure."""
    low, high = norm_bounds(list(map(q, normal)))
    value = sum((q(a) * q(b) for a, b in zip(normal, vector, strict=True)), ZERO)
    result = (value / low, value / high)
    return min(result), max(result)


def relative_bounds(workspace, state):
    robot = state.domains[workspace.domain]
    result = []
    for row, jac, n in zip(workspace.rows, workspace.j0, workspace.normals, strict=True):
        velocity = state.domains[row[0]].velocity[row[1]]
        vector = [q(velocity[k]) - dot(jac[k], robot.v) for k in range(3)]
        result.append(normal_projection_bounds(n, vector))
    return tuple(result)
