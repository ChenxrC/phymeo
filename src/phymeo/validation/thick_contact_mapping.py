"""Independent thick-point Jacobian, world balance and exact phase law audits."""

from fractions import Fraction as F

import numpy as np

from ..core.capabilities import SolverFailure
from ..geometry._prism_exact import Quadratic as Q
from ..geometry._prism_exact import cross, dot, scale, sub, vector
from ..solvers._dynamic_normal_geometry import normal_projection_bounds
from ..solvers._robot_replay_ledger import upper_float
from ._material_path_math import q

ZERO = F(0)


def phase_capacity(n0, held, n1, terminal, mu, initial_closed, is_terminal):
    """Project each actual phase on its own normal BEFORE exact interval addition."""
    p = normal_projection_bounds(n0, held)
    e = normal_projection_bounds(n1, terminal)
    if p[0] < 0 or e[0] < 0:
        raise SolverFailure("negative phase normal impulse")
    if is_terminal:
        bounds = e if initial_closed else (p[0] + e[0], p[1] + e[1])
    else:
        bounds = p if initial_closed else (ZERO, ZERO)
    return tuple(q(mu) * v for v in bounds)


def row_scales(diagonal, normal):
    k = float(normal @ diagonal @ normal / (normal @ normal))
    kappa = float(np.trace(diagonal))
    if not np.isfinite(k + kappa) or min(k, kappa) <= 0:
        raise SolverFailure("invalid positive phase response")
    return k, kappa


class ThickContactMappingLedger:
    """Reconstruct J from original node coordinates, never from port self-reports."""

    def __init__(self, positions, triangles, weights, offsets):
        self.positions = np.array(positions, float, copy=True)
        self.triangles = np.array(triangles, int, copy=True)
        self.weights = np.array(weights, float, copy=True)
        self.offsets = np.array(offsets, float, copy=True)
        expected = []
        points = []
        for ids, bary, s in zip(self.triangles, self.weights, self.offsets, strict=True):
            x = tuple(vector(row) for row in self.positions[ids])
            a, b = sub(x[1], x[0]), sub(x[2], x[0])
            c = cross(a, b)
            c2 = dot(c, c)
            if c2 <= 0 or c2.b:
                raise SolverFailure("degenerate independent thick mapping")
            length = Q(0, 1, c2.a)
            n = scale(c, 1 / length)
            block = np.empty((3, 3, 3), object)
            for i in range(3):
                for k in range(3):
                    basis = tuple(Q(int(j == k)) for j in range(3))
                    dc = (
                        cross(basis, sub(a, b))
                        if i == 0
                        else cross(basis, b)
                        if i == 1
                        else cross(a, basis)
                    )
                    dn = scale(sub(dc, scale(n, dot(n, dc))), q(s) / length)
                    for j in range(3):
                        block[i, j, k] = Q(q(bary[i]) * int(j == k)) + dn[j]
            expected.append(block)
            points.append(
                tuple(
                    sum((Q(q(w)) * p[k] for w, p in zip(bary, x, strict=True)), Q(0)) + q(s) * n[k]
                    for k in range(3)
                )
            )
        self.expected = tuple(expected)
        self.points = tuple(points)

    def aggregate_exact(self, blocks, world, count, *, triangles=None):
        ids = self.triangles if triangles is None else triangles
        result = np.full((count, 3), ZERO, object)
        for indices, block, s in zip(ids, blocks, world, strict=True):
            for i, node in enumerate(indices):
                for k in range(3):
                    result[node, k] += sum((q(block[i, j, k]) * q(s[j]) for j in range(3)), ZERO)
        return result

    def aggregate(self, blocks, world, count, *, triangles=None):
        return np.array(self.aggregate_exact(blocks, world, count, triangles=triangles), float)

    def response(self, blocks, inverse_mass, robot_jacobians, inverse_robot_mass):
        count = len(blocks)
        result = np.zeros((count, count, 3, 3))
        for i in range(count):
            for j in range(count):
                block = robot_jacobians[i] @ inverse_robot_mass @ robot_jacobians[j].T
                for a, na in enumerate(self.triangles[i]):
                    for b, nb in enumerate(self.triangles[j]):
                        if na == nb:
                            block = block + inverse_mass[na] * blocks[i, a] @ blocks[j, b].T
                result[i, j] = block
        return result

    @staticmethod
    def require_fp32(value):
        with np.errstate(all="ignore"):
            promoted = np.asarray(value).astype(np.float32).astype(np.float64)
        if not np.isfinite(promoted).all() or not np.array_equal(value, promoted):
            raise SolverFailure("actual native state is not an FP32 promotion")

    def require(self, blocks, world, node_requests, velocity, *, tolerance, speed=1.0, length=1.0):
        map_error = ZERO
        expected_lower = np.full((len(self.positions), 3), ZERO, object)
        expected_upper = np.full((len(self.positions), 3), ZERO, object)
        moment = [Q(0)] * 3
        power = Q(0)
        exact = self.aggregate_exact(blocks, world, len(self.positions))
        mapping = ZERO
        for stored, expected, impulse, point, ids in zip(
            blocks, self.expected, world, self.points, self.triangles, strict=True
        ):
            for index in np.ndindex(3, 3, 3):
                map_error = max(map_error, abs(Q(q(stored[index])) - expected[index]).bounds()[1])
            for i, node in enumerate(ids):
                for k in range(3):
                    weighted = sum((expected[i, j, k] * q(impulse[j]) for j in range(3)), Q(0))
                    lo, hi = weighted.bounds()
                    expected_lower[node, k] += lo
                    expected_upper[node, k] += hi
            target = cross(point, vector(impulse))
            for k in range(3):
                moment[k] -= target[k]
            material_velocity = tuple(
                sum(
                    (
                        expected[i, j, k] * q(velocity[ids[i], k])
                        for i in range(3)
                        for k in range(3)
                    ),
                    Q(0),
                )
                for j in range(3)
            )
            power -= dot(material_velocity, vector(impulse))
        for i, (position, load, v) in enumerate(
            zip(self.positions, node_requests, velocity, strict=True)
        ):
            for k in range(3):
                actual = q(load[k])
                mapping = max(
                    mapping,
                    abs(actual - exact[i, k]),
                    abs(actual - expected_lower[i, k]),
                    abs(actual - expected_upper[i, k]),
                )
            m = cross(vector(position), vector(load))
            for k in range(3):
                moment[k] += m[k]
            power += dot(vector(load), vector(v))

        def upper(value):
            return upper_float(Q.coerce(value).bounds()[1])

        force = max(
            (
                abs(
                    sum((q(load[k]) for load in node_requests), ZERO)
                    - sum((q(s[k]) for s in world), ZERO)
                )
                for k in range(3)
            ),
            default=ZERO,
        )
        report = dict(
            mapping=upper(max(map_error, mapping) / q(speed)),
            force=upper(force / q(speed)),
            moment=upper(sum((abs(v) for v in moment), Q(0)) / (q(speed) * q(length))),
            power=upper(abs(power) / (q(speed) * q(speed))),
        )
        if max(report.values()) > tolerance:
            raise SolverFailure("independent thick mapping/force/moment/power audit failed")
        return report


def phase_audit(normal, impulse, velocity, capacity, kappa, speed):
    """Original world invariants, preserving exact Fraction relative products."""
    from ..solvers._contact_replay_audit import sqrt_bounds

    def inner(a, b):
        return sum((q(x) * q(y) for x, y in zip(a, b, strict=True)), ZERO)

    n2 = inner(normal, normal)
    if n2 <= 0:
        raise SolverFailure("invalid phase normal")
    ns, nu = inner(normal, impulse), inner(normal, velocity)
    t2 = inner(impulse, impulse) - ns * ns / n2
    u2 = inner(velocity, velocity) - nu * nu / n2
    if min(t2, u2) < 0:
        raise SolverFailure("negative exact tangential squared norm")
    work = inner(impulse, velocity) - ns * nu / n2
    _, th = sqrt_bounds(t2)
    ul, uh = sqrt_bounds(u2)
    cl, ch = capacity
    if cl < 0 or ch < cl or normal_projection_bounds(normal, impulse)[0] < 0:
        raise SolverFailure("negative phase normal/capacity")
    k, v = q(kappa), q(speed)
    return (
        dict(
            cone=upper_float(k * max(th - cl, ZERO) / v),
            dissipation=upper_float(
                k * max(abs(work + c * u) for c in (cl, ch) for u in (ul, uh)) / (v * v)
            ),
            positive_work=upper_float(k * max(work, ZERO) / (v * v)),
        ),
        (cl, ch),
        work,
    )


def paired_moment_residual(prism_points, robot_points, world, tolerance, budget):
    total = ZERO
    for a, b, s in zip(prism_points, robot_points, world, strict=True):
        moment = cross(tuple(q(x) - q(y) for x, y in zip(a, b, strict=True)), tuple(map(q, s)))
        total += sum(map(abs, moment))
    return upper_float(q(tolerance) * total / q(budget))
