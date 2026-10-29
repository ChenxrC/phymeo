"""Finite convex closest witnesses and conservative curved robot-FK sweeps.

Floating geometry proposes directions. Rational primal/dual bounds and rational
Taylor FK intervals provide every accepted certificate.
"""

from dataclasses import dataclass
from fractions import Fraction as F
from numbers import Integral

import numpy as np

from ..constraints.robot_finite_patch_collider import RobotFinitePatchCollider, _cross, _dot, _sub
from ..core.capabilities import SolverFailure
from ..core.robot import robot_array
from ..core.scene import fingerprint
from ..dynamics.robot_kinematics import forward_kinematics
from ..solvers._contact_replay_audit import sqrt_bounds
from ..validation._material_path_math import interval_float
from ._patch_interval import (
    PatchCurve,
    add,
    divide,
    idot,
    interval,
    mul,
    scale,
    sub,
    validate_domain,
)

ZERO = F(0)


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _owned(value, dtype=float):
    out = np.array(value, dtype=dtype, copy=True)
    out.flags.writeable = False
    return out


def _closest(vertices, point):
    """Exact triangle-fan primal candidate, including all physical boundaries."""
    vertices = tuple(tuple(F(float(x)) for x in row) for row in vertices)
    point = tuple(F(float(x)) for x in point)
    candidates = []
    for j in range(1, len(vertices) - 1):
        ids = (0, j, j + 1)
        a, b, c = (vertices[i] for i in ids)
        u, v, w = _sub(b, a), _sub(c, a), _sub(point, a)
        uu, vv, uv, uw, vw = _dot(u, u), _dot(v, v), _dot(u, v), _dot(u, w), _dot(v, w)
        det = uu * vv - uv * uv
        if det:
            s, t = (uw * vv - vw * uv) / det, (vw * uu - uw * uv) / det
            if s >= 0 and t >= 0 and s + t <= 1:
                weights = [ZERO] * len(vertices)
                for i, x in zip(ids, (1 - s - t, s, t), strict=True):
                    weights[i] = x
                candidates.append(tuple(weights))
    for i, a in enumerate(vertices):
        j = (i + 1) % len(vertices)
        e = _sub(vertices[j], a)
        ee = _dot(e, e)
        if not ee:
            continue
        t = max(ZERO, min(F(1), _dot(_sub(point, a), e) / ee))
        weights = [ZERO] * len(vertices)
        weights[i], weights[j] = 1 - t, t
        candidates.append(tuple(weights))
    if not candidates:
        raise SolverFailure("degenerate closest-point geometry")

    def witness(weights):
        return tuple(
            sum((w * v[k] for w, v in zip(weights, vertices, strict=True)), ZERO) for k in range(3)
        )

    weights = min(candidates, key=lambda w: _dot(_sub(point, witness(w)), _sub(point, witness(w))))
    return weights, witness(weights)


def _distance_bounds(vertices, x, weights, d):
    """Primal feasible upper bound and one fixed dual separating direction."""
    y = tuple((ZERO, ZERO) for _ in range(3))
    for w, v in zip(weights, vertices, strict=True):
        y = tuple(add(a, scale(b, w)) for a, b in zip(y, v, strict=True))
    delta = tuple(sub(interval(a), b) for a, b in zip(x, y, strict=True))
    upper = sqrt_bounds(sum((max(abs(a), abs(b)) ** 2 for a, b in delta), ZERO))[1]
    d2 = _dot(d, d)
    lower = ZERO
    if d2:
        support = min(
            idot(
                tuple((v, v) for v in d),
                tuple(sub(interval(a), b) for a, b in zip(x, v, strict=True)),
            )[0]
            for v in vertices
        )
        if support > 0:
            lower = support / sqrt_bounds(d2)[1]
    return lower, upper


def _diagnostic_feature(vertices, exact_witness, rounded_local, node, ideal, actual):
    """Classify only a label using the existing FK/inverse error enclosure.

    Ideal FK is orthogonal, so world edge dot products and the exact local
    Gram inverse enclose z*, the local planar projection of the stored node.
    zhat is the exact planar projection of the actual rounded inverse input;
    zs uses the actual stored FK vertices in that same exact Gram expression.
    The two existing precision stages give E = ||zhat-zs|| + sup||zs-z*||.
    This sum avoids assuming cancellation of the stored/inverse and ideal FK
    errors. Convex projection is nonexpansive, so E also bounds the difference
    of the exact _closest witness and ideal closest point. Directed rational
    roots and squared edge comparisons add no epsilon or witness snapping.
    """
    a, b = _sub(vertices[1], vertices[0]), _sub(vertices[2], vertices[0])
    aa, ab, bb = _dot(a, a), _dot(a, b), _dot(b, b)
    determinant = aa * bb - ab * ab
    if determinant <= 0:
        raise SolverFailure("finite diagnostic projection Gram is unresolved")
    origin = vertices[0]
    raw = _sub(tuple(F(float(x)) for x in rounded_local), origin)
    ru, rv = _dot(a, raw), _dot(b, raw)
    uhat, vhat = (bb * ru - ab * rv) / determinant, (aa * rv - ab * ru) / determinant
    zhat = tuple(origin[k] + a[k] * uhat + b[k] * vhat for k in range(3))
    delta = tuple(sub(interval(x), y) for x, y in zip(node, ideal[0], strict=True))
    world_a = tuple(sub(y, x) for x, y in zip(ideal[0], ideal[1], strict=True))
    world_b = tuple(sub(y, x) for x, y in zip(ideal[0], ideal[2], strict=True))
    ru, rv = idot(world_a, delta), idot(world_b, delta)
    u = scale(sub(scale(ru, bb), scale(rv, ab)), 1 / determinant)
    v = scale(sub(scale(rv, aa), scale(ru, ab)), 1 / determinant)
    projected = tuple(
        add(interval(origin[k]), add(scale(u, a[k]), scale(v, b[k]))) for k in range(3)
    )
    stored = tuple(tuple(coordinate[0] for coordinate in row) for row in actual)
    stored_delta = _sub(tuple(F(float(x)) for x in node), stored[0])
    stored_u = _dot(_sub(stored[1], stored[0]), stored_delta)
    stored_v = _dot(_sub(stored[2], stored[0]), stored_delta)
    u = (bb * stored_u - ab * stored_v) / determinant
    v = (aa * stored_v - ab * stored_u) / determinant
    zs = tuple(origin[k] + a[k] * u + b[k] * v for k in range(3))
    inverse_error_squared = sum((x - y) ** 2 for x, y in zip(zhat, zs, strict=True))
    fk_error_squared = sum(
        max(abs(z - lo), abs(z - hi)) ** 2 for z, (lo, hi) in zip(zs, projected, strict=True)
    )
    error = sqrt_bounds(inverse_error_squared)[1] + sqrt_bounds(fk_error_squared)[1]
    error_squared = error * error

    near_vertices = set()
    near_edges = []
    for j, point in enumerate(vertices):
        delta = _sub(exact_witness, point)
        if _dot(delta, delta) <= error_squared:
            near_vertices.add(j)
        edge = _sub(vertices[(j + 1) % len(vertices)], point)
        extent = _dot(edge, edge)
        t = max(ZERO, min(F(1), _dot(delta, edge) / extent))
        separation = tuple(x - t * e for x, e in zip(delta, edge, strict=True))
        if _dot(separation, separation) <= error_squared:
            near_edges.append(j)
    if not near_edges:
        return (1, 0)
    common = set.intersection(*({j, (j + 1) % len(vertices)} for j in near_edges))
    if not common or len(near_vertices) > 1 or (near_vertices and not near_vertices <= common):
        raise SolverFailure("finite diagnostic topology unresolved: no unique common boundary")
    if len(common) == 1:
        return (3, common.pop())
    if near_vertices:
        return (3, near_vertices.pop())
    if len(near_edges) == 1:
        return (2, near_edges[0])
    raise SolverFailure("finite diagnostic topology unresolved: no common feature")


@dataclass(frozen=True)
class PatchGeometry:
    local_point: np.ndarray
    world_point: np.ndarray
    weights: np.ndarray
    normal: np.ndarray | None
    feature: tuple
    distance_bounds: tuple
    exact_distance_bounds: tuple
    exact_weights: tuple
    signed_support: tuple
    projection_inside: bool
    world_error_m: float


def patch_geometry(model, q, collider, node):
    """Own a closest witness, independently enclose ideal and stored FK distance."""
    validate_domain(model)
    q = robot_array(q, "q", ndim=1)
    node = robot_array(node, "node", ndim=1)
    if (
        q.shape != (model.dof_count,)
        or node.shape != (3,)
        or collider.link_index >= model.dof_count
    ):
        raise ValueError("invalid patch query dimensions/index")
    curve = PatchCurve(model, q, q, collider)
    fk = forward_kinematics(model, q)
    i = collider.link_index
    rotation, origin = fk.rotation[i], fk.position[i]
    local = np.linalg.solve(rotation, node - origin)
    weights, local_point = _closest(collider.local_vertices, local)
    local_point = np.array(local_point, dtype=float)
    world = origin + rotation @ local_point
    d = tuple(F(float(a)) - F(float(b)) for a, b in zip(node, world, strict=True))
    vertices = origin + (rotation @ collider.local_vertices.T).T
    ideal = curve.at(ZERO)
    actual = tuple(tuple(interval(x) for x in row) for row in vertices)
    lo, hi = _distance_bounds(ideal, node, weights, d)
    alo, ahi = _distance_bounds(actual, node, weights, d)
    bounds = min(lo, alo), max(hi, ahi)
    y = tuple((ZERO, ZERO) for _ in range(3))
    for w, v in zip(weights, ideal, strict=True):
        y = tuple(add(a, scale(b, w)) for a, b in zip(y, v, strict=True))
    error = sum(
        (
            max(abs(F(float(a)) - b[0]), abs(F(float(a)) - b[1]))
            for a, b in zip(world, y, strict=True)
        ),
        ZERO,
    )
    norm = np.hypot.reduce(node - world)
    normal = None if norm == 0 else _owned((node - world) / norm)
    # Diagnostic topology uses only physical patch boundaries, never fan
    # seams. It cannot modify any physical witness, weight, normal or bound.
    v = tuple(tuple(F(float(x)) for x in row) for row in collider.local_vertices)
    exact_local = tuple(
        sum((w * a[k] for w, a in zip(weights, v, strict=True)), ZERO) for k in range(3)
    )
    feature = _diagnostic_feature(v, exact_local, local, node, ideal, actual)
    n = _cross(_sub(v[1], v[0]), _sub(v[2], v[0]))
    side = _dot(n, _sub(tuple(F(float(a)) for a in local), v[0]))
    # The root-branch side is certified from the ideal plane, never guessed
    # from a rounded local inverse or from old active history.
    e = tuple(sub(b, a) for a, b in zip(ideal[0], ideal[1], strict=True))
    f = tuple(sub(b, a) for a, b in zip(ideal[0], ideal[2], strict=True))
    plane = tuple(
        sub(mul(e[(k + 1) % 3], f[(k + 2) % 3]), mul(e[(k + 2) % 3], f[(k + 1) % 3]))
        for k in range(3)
    )
    normlo = sum((ZERO if lo <= 0 <= hi else min(lo * lo, hi * hi) for lo, hi in plane), ZERO)
    normhi = sum((max(lo * lo, hi * hi) for lo, hi in plane), ZERO)
    length = (sqrt_bounds(normlo)[0], sqrt_bounds(normhi)[1])
    projected_side = idot(
        plane, tuple(sub(interval(x), a) for x, a in zip(node, ideal[0], strict=True))
    )
    signed = divide(projected_side, length)
    projected = tuple(F(float(a)) - side * b / _dot(n, n) for a, b in zip(local, n, strict=True))
    inside = all(
        _dot(_cross(_sub(v[(j + 1) % len(v)], a), _sub(projected, a)), n) >= 0
        for j, a in enumerate(v)
    )
    return PatchGeometry(
        _owned(local_point),
        _owned(world),
        _owned([float(w) for w in weights]),
        normal,
        feature,
        interval_float(bounds),
        bounds,
        weights,
        signed,
        inside,
        float(error),
    )


@dataclass(frozen=True)
class RobotPatchSweepSegment:
    start: F
    end: F
    distance_lower_m: float
    certified: bool
    depth: int
    rational_start: tuple | None = None
    rational_end: tuple | None = None

    def __post_init__(self):
        if self.rational_start is None:
            value = F(self.start)
            object.__setattr__(self, "rational_start", (value.numerator, value.denominator))
        if self.rational_end is None:
            value = F(self.end)
            object.__setattr__(self, "rational_end", (value.numerator, value.denominator))
        object.__setattr__(self, "start", float(self.start))
        object.__setattr__(self, "end", float(self.end))


@dataclass(frozen=True)
class RobotPatchSweepResult:
    status: str
    segments: tuple
    checked_intervals: int
    max_depth_reached: int
    input_fingerprint: str
    witness_time: float | None = None
    upper_distance_m: float | None = None
    method: str = "rational-curved-fk-separation-v1"


def robot_patch_sweep(
    model, q0, q1, collider, node_start, node_end, *, clearance_m, max_intervals=4096, max_depth=32
):
    """Prove complete synchronized linear-node/curved-FK clearance or fail closed."""
    if not isinstance(collider, RobotFinitePatchCollider) or collider.link_index >= model.dof_count:
        raise ValueError("valid finite patch required")
    max_intervals = _positive_integer(max_intervals, "max_intervals")
    max_depth = _positive_integer(max_depth, "max_depth")
    if type(clearance_m) is not float or not np.isfinite(clearance_m) or clearance_m <= 0:
        raise ValueError("clearance_m must be finite positive builtin float")
    q0, q1 = (robot_array(q, name, ndim=1) for q, name in ((q0, "q0"), (q1, "q1")))
    x0, x1 = (
        robot_array(q, name, ndim=1)
        for q, name in ((node_start, "node_start"), (node_end, "node_end"))
    )
    if (
        q0.shape != q1.shape
        or q0.shape != (model.dof_count,)
        or x0.shape != (3,)
        or x1.shape != (3,)
    ):
        raise ValueError("invalid sweep dimensions")
    signature = fingerprint(
        (model, q0, q1, collider, x0, x1, clearance_m, max_intervals, max_depth)
    )
    curve = PatchCurve(model, q0, q1, collider)
    x0, x1 = tuple(map(lambda x: F(float(x)), x0)), tuple(map(lambda x: F(float(x)), x1))

    def node(t):
        return tuple(a + t * (b - a) for a, b in zip(x0, x1, strict=True))

    stack = [(ZERO, F(1), 0)]
    segments = []
    checked = 0
    deepest = 0
    witness = upper = None
    clearance = F(clearance_m)
    while stack:
        a, b, depth = stack.pop()
        if checked >= max_intervals:
            segments.append(RobotPatchSweepSegment(a, b, 0.0, False, depth))
            continue
        checked += 1
        deepest = max(deepest, depth)
        mid = (a + b) / 2
        vertices = curve.at(mid)
        rounded = np.array([[(lo + hi) / 2 for lo, hi in v] for v in vertices], dtype=float)
        weights, y = _closest(rounded, node(mid))
        d = _sub(node(mid), y)
        _, up = _distance_bounds(vertices, node(mid), weights, d)
        if up < clearance:
            witness, upper = float(mid), interval_float((ZERO, up))[1]
            segments.append(RobotPatchSweepSegment(a, b, 0.0, False, depth))
            for s, e, z in stack:
                segments.append(RobotPatchSweepSegment(s, e, 0.0, False, z))
            stack = []
            break
        curvature = curve.projected_curvature(d)
        endpoints = []
        for t in (a, b):
            endpoints.append(
                tuple(
                    idot(
                        tuple((v, v) for v in d),
                        tuple(sub((x, x), v) for x, v in zip(node(t), vertex, strict=True)),
                    )[0]
                    for vertex in curve.at(t)
                )
            )
        projection = min(
            min(p, q) - c * (b - a) ** 2 / 8 for p, q, c in zip(*endpoints, curvature, strict=True)
        )
        norm_upper = sqrt_bounds(_dot(d, d))[1]
        lower = ZERO if not norm_upper or projection <= 0 else projection / norm_upper
        lower = max(lower, curve.plane_lower(x0, x1, a, b))
        safe = lower >= clearance
        if safe or depth >= max_depth:
            segments.append(
                RobotPatchSweepSegment(a, b, interval_float((lower, lower))[0], safe, depth)
            )
        else:
            stack.extend(((mid, b, depth + 1), (a, mid, depth + 1)))
    segments = tuple(sorted(segments, key=lambda v: F(*v.rational_start)))
    status = (
        "violation"
        if witness is not None
        else "certified"
        if all(s.certified for s in segments)
        else "unresolved"
    )
    return RobotPatchSweepResult(status, segments, checked, deepest, signature, witness, upper)
