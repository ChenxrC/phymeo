"""Finite-contact geometry and independent stored-value physical audits."""

from fractions import Fraction as F

import numpy as np

from ..core.capabilities import SolverFailure
from ..dynamics.robot_kinematics import forward_kinematics, point_jacobians
from ..geometry.robot_patch_sweep import patch_geometry
from ..validation._material_path_math import cross, norm_bounds
from ._robot_replay_ledger import FrozenRobotLedger, audit_affine_geometry, rational, upper_float

ZERO = F(0)


def q(value):
    return value if isinstance(value, F) else rational(value)


def dot(a, b):
    return sum((q(x) * q(y) for x, y in zip(a, b, strict=True)), ZERO)


class RotorLedger(FrozenRobotLedger):
    """Native frozen equations plus independent constant-axis kinetic energy."""

    def __init__(self, model, state, inputs, h, settings, angular):
        super().__init__(model, state, inputs, h, settings, angular)
        self.inertia = None
        if model.joint_types == ("revolute",):
            self.inertia = q(model.inertia_diagonal[0, np.flatnonzero(model.joint_axis[0])[0]])
            if abs(q(self.mass[0, 0]) - self.inertia) / self.inertia > q(settings.tolerance):
                raise SolverFailure("native rotor mass differs from analytic constant inertia")

    def kinetic(self, velocity):
        return (
            self.inertia * q(velocity[0]) ** 2 / 2
            if self.inertia is not None
            else super().kinetic(velocity)
        )


def geometry(w, state):
    robot = state.domains[w.domain]
    try:
        samples = tuple(
            patch_geometry(w.model, robot.q, c, state.domains[name].position[i])
            for name, i, c in w.rows
        )
    except (ValueError, TypeError) as error:
        raise SolverFailure("invalid finite geometry: " + str(error)) from error
    if any(q(s.world_error_m) > w.bound for s in samples):
        raise SolverFailure("finite patch ideal/stored FK mapping exceeds length budget")
    local = np.array([s.local_point for s in samples])
    audit_affine_geometry(w.model, robot, w.indices, local, w.h, w.settings)
    jac = point_jacobians(w.model, robot.q, w.indices, local)[0]
    normals = np.array([np.zeros(3) if s.normal is None else s.normal for s in samples])
    bounds = tuple(
        (a - q(c.radius_m), b - q(c.radius_m))
        for s, (_, _, c) in zip(samples, w.rows, strict=True)
        for a, b in (s.exact_distance_bounds,)
    )
    w.samples = samples
    return (
        np.array([s.world_point for s in samples]),
        jac,
        normals,
        np.array([float((a + b) / 2) for a, b in bounds]),
        bounds,
    )


def normal_projection_bounds(normal, vector):
    low, high = norm_bounds(list(map(q, normal)))
    if low == 0:
        raise SolverFailure("undefined zero contact normal")
    value = dot(normal, vector)
    values = (value / low, value / high)
    return min(values), max(values)


def relative_bounds(w, state):
    result = []
    robot = state.domains[w.domain]
    for (name, i, _), jac, n in zip(w.rows, w.j1, w.terminal_normals, strict=True):
        v = state.domains[name].velocity[i]
        vector = [q(v[k]) - dot(jac[k], robot.v) for k in range(3)]
        result.append(normal_projection_bounds(n, vector))
    return tuple(result)


def paired_moment(w, state, local, world):
    fk = forward_kinematics(w.model, state.domains[w.domain].q)
    total = ZERO
    for (name, i, c), b, p in zip(w.rows, local, world, strict=True):
        y = fk.position[c.link_index] + fk.rotation[c.link_index] @ b
        delta = [q(a) - q(b) for a, b in zip(state.domains[name].position[i], y, strict=True)]
        total += sum(map(abs, cross(delta, list(map(q, p)))))
    magnitude = sum((abs(q(p)) for p in world.flat), ZERO)
    return total, ZERO if not magnitude else total / (magnitude * w.exact_gap_scale)


def axis_metrics(w):
    w.axis_report = {}
    if w.program.robot_kind != "rotor":
        return {}
    start = w.initial
    final = w.trial.result.state
    fk = forward_kinematics(w.model, start.domains[w.domain].q)
    axis = list(map(q, fk.axis[0]))
    origin = list(map(q, w.model.joint_position[0]))
    r0 = w.ledger.inertia * q(start.domains[w.domain].v[0])
    r1 = w.ledger.inertia * q(final.domains[w.domain].v[0])
    external = w.ledger.free_integral[0]
    defects = [ZERO] * 3
    held = w._loads()[2]
    terminal = w._node_terminal()
    for name in w.node_domains:
        model = w.models[name]
        for i, inverse in enumerate(model.inverse_mass):
            mass = 1 / q(inverse)
            x0 = list(map(q, start.domains[name].position[i]))
            x1 = list(map(q, final.domains[name].position[i]))
            v0 = list(map(q, start.domains[name].velocity[i]))
            v1 = list(map(q, final.domains[name].velocity[i]))
            initial = cross([x - origin[k] for k, x in enumerate(x0)], [mass * v for v in v0])
            last = cross([x - origin[k] for k, x in enumerate(x1)], [mass * v for v in v1])
            free = [
                q(w.h) * (mass * q(model.gravity[k]) + q(w.inputs.domains[name].force[i, k]))
                for k in range(3)
            ]
            ext = cross([x - origin[k] for k, x in enumerate(x0)], free)
            # Material defect is computed from particle equations independently
            # and per domain; it does not absorb a missing robot reaction.
            impulse = [free[k] + held[name][i][k] for k in range(3)]
            material = [
                last[k]
                - initial[k]
                - cross([x - origin[j] for j, x in enumerate(x0)], impulse)[k]
                - cross([x - origin[j] for j, x in enumerate(x1)], terminal[name][i])[k]
                for k in range(3)
            ]
            r0 += dot(axis, initial)
            r1 += dot(axis, last)
            external += dot(axis, ext)
            defects = [a + b for a, b in zip(defects, material, strict=True)]
    defect = abs(r1 - r0 - external - dot(axis, defects))
    limit = q(w.program.budgets.max_material_angular_defect_nms)
    w.axis_report = dict(
        before_nms=float(r0),
        after_nms=float(r1),
        original_external_nms=float(external),
        material_defect_nms=float(dot(axis, defects)),
        residual_nms=upper_float(defect),
        limit_nms=float(limit),
    )
    return {"rotor_axis_balance": upper_float(q(w.settings.tolerance) * defect / limit)}
