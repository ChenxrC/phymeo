"""Material director chart, signed hinge events and objective virtual-work rows on two folded faces."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionModel
from ..core.capabilities import SolverFailure
from .deforming_material_seam import _minimum
from .surface_features import closest_point_triangle


def _unit(x):
    length = np.linalg.norm(x)
    if not np.isfinite(length) or length <= 0:
        raise SolverFailure("degenerate folded material direction")
    return x / length, length


def _min_length(a, b):
    delta = b - a
    t = 0.0 if delta @ delta == 0 else np.clip(-(a @ delta) / (delta @ delta), 0.0, 1.0)
    return np.linalg.norm(a + t * delta)


def _bilinear(a, da, b, db, func):
    return np.array([func(a, b), func(da, b) + func(a, db), func(da, db)])


def _pack(x):
    return tuple(map(float, x))


@dataclass(frozen=True)
class SpatialFoldFeature:
    face: CollisionKey
    feature: str
    distance_m: float
    point_m: tuple
    weights: tuple


@dataclass(frozen=True)
class FoldStation:
    face: CollisionKey
    tangent: tuple
    normal: tuple
    director: tuple
    material_point_m: tuple
    material_weights: tuple
    normal_gap_m: float
    director_gap_m: float
    fold_angle_rad: float
    spatial: tuple


@dataclass(frozen=True)
class FoldInterval:
    side: int
    t0: float
    t1: float
    gap_bounds: tuple
    rows: np.ndarray
    time_rows: np.ndarray


@dataclass(frozen=True)
class FoldTrace:
    intervals: tuple
    jacobian: np.ndarray
    event_bracket: tuple | None
    quadrature_difference: float


def _frames(x):
    center = (x[1] + x[2]) / 2
    u, length = _unit(x[2] - x[1])
    frame = []
    for sign, outer in ((-1.0, 3), (1.0, 4)):
        arm = x[outer] - center
        t, m = _unit(sign * (arm - u * (arm @ u)))
        frame.append((t, np.cross(t, u), arm, m))
    tangent, _ = _unit(frame[0][0] + frame[1][0])
    return center, u, length, frame, tangent, np.cross(tangent, u)


def _geometry(x, side):
    center, u, length, frames, chart, director = _frames(x)
    t, n, arm, m = frames[side]
    r = x[0] - center
    outer = (r @ chart) / (arm @ chart)
    axial = (r @ u - outer * (arm @ u)) / length
    w = np.array([(1 - outer) / 2 - axial, (1 - outer) / 2 + axial, outer])
    tri = [1, 2, 3 + side]
    q = w @ x[tri]
    return t, n, arm, m, u, length, w, q, frames, chart, director


def _jacobian(x, side):
    t, n, arm, m, u, length, w, q, *_ = _geometry(x, side)
    r = x[0] - q
    # Reverse derivative of (t dot r, n dot r), holding the material weights fixed.
    # n=t cross u, t is the projected material wing, u is the oriented shared edge.
    force = np.asarray((t, n))
    gt = np.asarray((r, np.cross(u, r)))
    gu = np.asarray((np.zeros(3), np.cross(r, t)))
    gp = (-1.0 if side == 0 else 1.0) * (gt - (gt @ t)[:, None] * t) / m
    gv = gp - (gp @ u)[:, None] * u
    gu = gu - (arm @ u) * gp - (gp @ u)[:, None] * arm
    ge = (gu - (gu @ u)[:, None] * u) / length
    J = np.zeros((2, 5, 3))
    J[:, 0] = force
    J[:, 1] = -w[0] * force - 0.5 * gv - ge
    J[:, 2] = -w[1] * force - 0.5 * gv + ge
    J[:, 3 + side] = -w[2] * force + gv
    return J


_GL4 = np.polynomial.legendre.leggauss(4)
_GL8 = np.polynomial.legendre.leggauss(8)


def _integrate(x, dx, side, a, b, depth=0):
    def rule(table):
        moments = np.zeros((2, 2, 5, 3))
        for z, w in zip(*table, strict=True):
            s = (a + b + (b - a) * z) / 2
            j = _jacobian(x + s * dx, side)
            moments[0] += w * j
            moments[1] += w * s * j
        return moments * ((b - a) / 2)

    coarse, fine = rule(_GL4), rule(_GL8)
    error = float(np.max(abs(fine - coarse)))
    if error <= 2e-11 * (b - a):
        return fine, error
    if depth == 7:
        raise SolverFailure("fold quadrature refinement budget exhausted")
    left, e0 = _integrate(x, dx, side, a, (a + b) / 2, depth + 1)
    right, e1 = _integrate(x, dx, side, (a + b) / 2, b, depth + 1)
    return left + right, e0 + e1


class FoldedMaterialChart:
    """A transverse slice through the middle of an oriented two-face material hinge."""

    def __init__(self, model, material_rows, point_row, max_angle):
        if not isinstance(model, CollisionModel) or len(model.shapes) != 2 or model.excluded:
            raise ValueError("fold chart requires two explicit material faces")
        a, b = sorted(model.shapes, key=lambda s: s.key)
        body = (a.key.world, a.key.body)
        if (b.key.world, b.key.body) != body or any(
            s.kind != "triangle" or s.radius for s in (a, b)
        ):
            raise ValueError("fold chart requires unrounded triangles of one material body")
        shared = set(a.vertex_ids) & set(b.vertex_ids)
        if len(shared) != 2:
            raise ValueError("disconnected, duplicated or ambiguous material hinge")

        def edge(s):
            return next(
                (i, j)
                for i, j in zip(s.vertex_ids, s.vertex_ids[1:] + s.vertex_ids[:1], strict=True)
                if {i, j} == shared
            )

        e = edge(a)
        if edge(b) != e[::-1]:
            raise ValueError("inconsistent folded material winding")
        outer = tuple(next(i for i in s.vertex_ids if i not in shared) for s in (a, b))
        self.ids = (*e, *outer)
        if set(material_rows) != set(self.ids):
            raise ValueError("each material hinge node needs its own particle row")
        rows = (point_row, *(material_rows[i] for i in self.ids))
        if any(isinstance(i, bool) or not isinstance(i, Integral) for i in rows) or set(
            rows
        ) != set(range(5)):
            raise ValueError("expected five distinct particle rows")
        self.rows = tuple(map(int, rows))
        self.keys = (a.key, b.key)
        self.body = body
        self.max_angle = max_angle

    def station(self, x, side, *, spatial=True):
        t, n, _, _, u, _, w, q, frames, _, director = _geometry(x, side)
        distance = float((x[0] - q) @ n)
        signed = float(
            np.arctan2(u @ np.cross(frames[0][0], frames[1][0]), frames[0][0] @ frames[1][0])
        )
        near = []
        if spatial:
            for k in (0, 1):
                found = closest_point_triangle(
                    np.array(x[0], dtype=float), np.array(x[[1, 2, 3 + k]], dtype=float)
                )
                ids = (*self.ids[:2], self.ids[2 + k])
                active = sum(v > 1e-12 for v in found.second_weights)
                near.append(
                    SpatialFoldFeature(
                        self.keys[k],
                        ("vertex", "edge", "face")[active - 1],
                        found.distance,
                        _pack(found.second),
                        tuple(zip(ids, _pack(found.second_weights), strict=True)),
                    )
                )
        return FoldStation(
            self.keys[side],
            _pack(t),
            _pack(n),
            _pack(director),
            _pack(q),
            tuple(zip((*self.ids[:2], self.ids[2 + side]), _pack(w), strict=True)),
            distance,
            float((x[0] - q) @ director),
            signed,
            tuple(near),
        )

    def start_side(self, x):
        c, _, _, _, t, _ = _frames(x)
        signed = float((x[0] - c) @ t)
        if abs(signed) <= 2e-11 * max(np.linalg.norm(x[3] - c), np.linalg.norm(x[4] - c)):
            raise SolverFailure("material station on an endpoint tie at the hinge")
        return int(signed > 0)

    def trace(self, x, last, width, eps):
        dx = last - x
        c0 = (x[1] + x[2]) / 2
        c1 = (last[1] + last[2]) / 2
        u, length = _unit(x[2] - x[1])
        ul, end_length = _unit(last[2] - last[1])
        scale = max(
            length,
            end_length,
            np.max(np.linalg.norm(x - c0, axis=1)),
            np.max(np.linalg.norm(last - c1, axis=1)),
        )
        relative = max(2e-11, 64 * eps)
        tol = relative * scale
        if min(length, end_length) < 1e-7 * scale or np.linalg.norm(ul - u) > relative:
            raise SolverFailure("folded material edge degenerates or its axis turns")
        if (
            max(np.max(abs((q[[0, 3, 4]] - center) @ u)) for q, center in ((x, c0), (last, c1)))
            > tol
        ):
            raise SolverFailure("folded material path leaves the transverse middle section")
        r = x[0] - c0
        dr = last[0] - c1 - r
        arms = []
        speeds = []
        curvatures = []
        for row, sign in ((3, -1.0), (4, 1.0)):
            arm = sign * (x[row] - c0)
            arm -= u * (arm @ u)
            end = sign * (last[row] - c1)
            end -= u * (end @ u)
            velocity = end - arm
            lower = _min_length(arm, end)
            if lower <= 1e-7 * scale:
                raise SolverFailure("folded material wing is degenerate during its affine path")
            if max(np.linalg.norm(r), np.linalg.norm(r + dr)) > 0.85 * lower * np.cos(
                self.max_angle / 2
            ):
                raise SolverFailure("folded material path is near a vertex or an outer boundary")
            arms.append((arm, velocity))
            speed = np.linalg.norm(velocity) / lower
            speeds.append(speed)
            curvatures.append(3 * speed**2)
        (v0, d0), (v1, d1) = arms
        dot = _bilinear(v0, d0, v1, d1, lambda a, b: a @ b)
        cross = _bilinear(v0, d0, v1, d1, lambda a, b: u @ np.cross(a, b))
        floor = relative * scale**2
        if (
            min(_minimum(np.tan(self.max_angle) * dot + sign * cross, 0, 1) for sign in (-1, 1))
            <= floor
        ):
            raise SolverFailure("fold exceeds the small signed-angle limit")
        if _minimum(np.sign(cross[0]) * cross, 0, 1) <= floor:
            raise SolverFailure("fold changes sign or flattens during the material path")

        def coordinate(s):
            p = r + s * dr
            value = 0.0
            rate = 0.0
            for v, d in arms:
                t, m = _unit(v + s * d)
                td = (d - t * (t @ d)) / m
                value += p @ t
                rate += dr @ t + p @ td
            return value, rate

        radius = max(np.linalg.norm(r), np.linalg.norm(r + dr))
        L1 = 2 * np.linalg.norm(dr) + radius * sum(speeds)
        L2 = 2 * np.linalg.norm(dr) * sum(speeds) + radius * sum(curvatures)
        z0, z1 = coordinate(0)[0], coordinate(1)[0]
        if min(abs(z0), abs(z1)) <= 4 * tol:
            raise SolverFailure("loaded material endpoint lies on the hinge")
        source = int(z0 > 0)
        dest = int(z1 > 0)
        bracket = None

        # Certify a sign barrier (same side) or a single transverse crossing (opposite sides).
        def barrier(a, b, depth):
            z, dz = coordinate((a + b) / 2)
            margin = (
                np.sign(z0) * z - L1 * (b - a) / 2
                if source == dest
                else np.sign(z1 - z0) * dz - L2 * (b - a) / 2
            )
            if margin > 4 * tol:
                return margin
            if depth == 8:
                raise SolverFailure("ambiguous material reversal or grazing at the hinge")
            m = (a + b) / 2
            return min(barrier(a, m, depth + 1), barrier(m, b, depth + 1))

        slope = barrier(0, 1, 0)
        if source != dest:
            lo, hi = 0.0, 1.0
            for _ in range(45):
                mid = (lo + hi) / 2
                if (coordinate(mid)[0] > 0) == (z0 > 0):
                    lo = mid
                else:
                    hi = mid
            root = (lo + hi) / 2
            bracket = (lo - 2 * tol / slope, hi + 2 * tol / slope)
            if bracket[0] <= 0 or bracket[1] >= 1:
                raise SolverFailure("material crossing too close to an endpoint")
            pieces = ((0.0, root, source), (root, 1.0, dest))
        else:
            pieces = ((0.0, 1.0, source),)
        out = []
        error = 0.0
        for a, b, side in pieces:
            arm, velocity = arms[side]
            numerator = _bilinear(r, dr, arm, velocity, lambda q, v: q @ np.cross(v, u))
            aa, ab = arm + a * velocity, arm + b * velocity
            low = _minimum(numerator, a, b) / max(np.linalg.norm(aa), np.linalg.norm(ab)) - tol
            high = -_minimum(-numerator, a, b) / _min_length(aa, ab) + tol
            if low <= 0 or high >= width:
                raise SolverFailure(
                    "cannot certify positive compression across this folded interval"
                )
            integrated, diff = _integrate(x, dx, side, a, b)
            out.append(
                FoldInterval(side, a, b, (float(low), float(high)), integrated[0], integrated[1])
            )
            error += diff
        return FoldTrace(tuple(out), sum(leg.rows for leg in out), bracket, error)
