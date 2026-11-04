"""Transverse material-edge events and integrated barycentric addresses on two affine moving faces."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionModel
from ..core.capabilities import SolverFailure


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _at(p, t):
    return p[..., 0] + t * (p[..., 1] + t * p[..., 2])


def _minimum(p, a, b):
    low = min(_at(p, a), _at(p, b))
    if p[2] > 0:
        t = -p[1] / (2 * p[2])
        if a < t < b:
            low = min(low, _at(p, t))
    return low


def _integrate(numerator, denominator, a, b, tolerance, depth=0):
    """Simpson panel with a fourth-derivative bound for N/D in the local unit interval."""
    dt = b - a
    n0, n1, n2 = (
        _at(numerator, a),
        dt * (numerator[:, 1] + 2 * a * numerator[:, 2]),
        dt**2 * numerator[:, 2],
    )
    d0, d1, d2 = (
        _at(denominator, a),
        dt * (denominator[1] + 2 * a * denominator[2]),
        dt**2 * denominator[2],
    )
    lower = _minimum((np.array([d0, d1, d2])), 0, 1)
    if lower <= 0:
        raise SolverFailure("deforming material quadrature has a singular face")
    one = max(abs(d1), abs(d1 + 2 * d2))
    two = abs(2 * d2)
    q0 = (abs(n0) + abs(n1) + abs(n2)) / lower
    q1 = (abs(n1) + 2 * abs(n2) + one * q0) / lower
    q2 = (2 * abs(n2) + 2 * one * q1 + two * q0) / lower
    q3 = (3 * one * q2 + 3 * two * q1) / lower
    q4 = (4 * one * q3 + 6 * two * q2) / lower
    error = dt * float(np.sum(q4)) / 2880
    if error <= dt * tolerance:
        q = (
            n0 / d0
            + 4 * (n0 + 0.5 * n1 + 0.25 * n2) / (d0 + 0.5 * d1 + 0.25 * d2)
            + (n0 + n1 + n2) / (d0 + d1 + d2)
        )
        return (dt / 6) * q, error
    if depth >= 12:
        raise SolverFailure("deforming barycentric quadrature budget exceeded")
    mid = (a + b) / 2
    left, e0 = _integrate(numerator, denominator, a, mid, tolerance, depth + 1)
    right, e1 = _integrate(numerator, denominator, mid, b, tolerance, depth + 1)
    return left + right, e0 + e1


@dataclass(frozen=True)
class MovingFaceInterval:
    face: CollisionKey
    node_ids: tuple
    local_rows: tuple
    t0: float
    t1: float
    entry: tuple
    exit: tuple
    integral: tuple


@dataclass(frozen=True)
class MovingTrace:
    spans: tuple
    weights: np.ndarray
    bracket: tuple | None
    error: float


class DeformingSeamChart:
    """Local row zero is the vertex; all remaining rows are actual independent material nodes."""

    def __init__(self, model, material_rows, point_row, normal):
        if not isinstance(model, CollisionModel) or len(model.shapes) != 2 or model.excluded:
            raise ValueError("deforming seam needs exactly two registered material faces")
        faces = tuple(sorted(model.shapes, key=lambda q: q.key))
        body = (faces[0].key.world, faces[0].key.body)
        if any(
            f.kind != "triangle" or f.radius or (f.key.world, f.key.body) != body for f in faces
        ):
            raise ValueError("deforming seam requires unrounded triangles of one material body")
        ids = tuple(sorted({n for f in faces for n in f.vertex_ids}))
        if set(material_rows) != set(ids) or any(
            isinstance(i, bool) or not isinstance(i, Integral) or i < 0
            for i in (point_row, *material_rows.values())
        ):
            raise ValueError("explicit nonnegative particle rows must cover every material ID")
        rows = (int(point_row), *(int(material_rows[i]) for i in ids))
        if len(set(rows)) != len(rows):
            raise ValueError(
                "the point and distinct material nodes require separate free particle rows"
            )
        self.rows = rows
        self.ids = ids
        self.body = body
        self.keys = tuple(f.key for f in faces)
        self.face_ids = tuple(f.vertex_ids for f in faces)
        self.indices = tuple(tuple(ids.index(i) + 1 for i in f.vertex_ids) for f in faces)
        shared = set(faces[0].vertex_ids) & set(faces[1].vertex_ids)
        if len(shared) == 3:
            raise ValueError("duplicate material face")
        if len(shared) == 2:

            def shared_direction(ids):
                return next(
                    (ids[i], ids[(i + 1) % 3])
                    for i in range(3)
                    if {ids[i], ids[(i + 1) % 3]} == shared
                )

            if shared_direction(self.face_ids[0]) != shared_direction(self.face_ids[1])[::-1]:
                raise ValueError("inconsistent winding around the shared material edge")
        self.shared = shared
        n = np.asarray(normal, dtype=float)
        if n.shape != (3,) or not np.isfinite(n).all() or abs(np.linalg.norm(n) - 1) > 2e-7:
            raise ValueError("a fixed unit world normal is required")
        self.normal = n / np.linalg.norm(n)
        v = np.eye(3)[np.argmin(abs(n))]
        u = v - self.normal * (v @ self.normal)
        u /= np.linalg.norm(u)
        self.basis = np.stack((u, np.cross(self.normal, u)), axis=1)

    def _polynomials(self, x0, x1):
        y0 = (x0 - x0[0]) @ self.basis
        y1 = (x1 - x1[0]) @ self.basis
        dy = y1 - y0
        scale = max(np.max(np.linalg.norm(y0, axis=1)), np.max(np.linalg.norm(y1, axis=1)))
        floor = 128 * np.finfo(float).eps * scale**2
        num = []
        for rows in self.indices:
            p, dp = y0[list(rows)], dy[list(rows)]
            coeff = np.array(
                [
                    [
                        _cross(p[j], p[k]),
                        _cross(dp[j], p[k]) + _cross(p[j], dp[k]),
                        _cross(dp[j], dp[k]),
                    ]
                    for j, k in ((1, 2), (2, 0), (0, 1))
                ]
            )
            if _minimum(coeff.sum(axis=0), 0, 1) <= floor:
                raise SolverFailure("deforming face flips or degenerates during the affine path")
            num.append(coeff)
        return num, floor

    def _which(self, polys, t, floor):
        values = [_at(p, t) for p in polys]
        owners = [i for i, w in enumerate(values) if np.min(w) >= -floor]
        if len(owners) != 1 or np.min(values[owners[0]]) <= floor:
            raise SolverFailure("deforming endpoint has no unique strict material face interior")
        return owners[0]

    def validate_plane(self, x0, x1, eps):
        scale = max(
            np.max(np.linalg.norm(x0[1:] - x0[1], axis=1)),
            np.max(np.linalg.norm(x1[1:] - x1[1], axis=1)),
        )
        for x in (x0, x1):
            if np.max(abs((x[1:] - x[1]) @ self.normal)) > 256 * eps * scale:
                raise SolverFailure(
                    "actual independent support nodes tilt out of the accepted coplanar domain"
                )
        for rows in self.indices:
            face = list(rows)
            e0 = ((x0[face] - x0[face[0]]) @ self.basis)[1:]
            e1 = ((x1[face] - x1[face[0]]) @ self.basis)[1:]
            try:
                f = np.linalg.solve(e0, e1).T
            except np.linalg.LinAlgError as err:
                raise SolverFailure("singular material reference face") from err
            if np.trace(f) <= 0 or abs(f[1, 0] - f[0, 1]) > 256 * eps * max(1.0, np.linalg.norm(f)):
                raise SolverFailure(
                    "material polar rotation is outside the spin-free deforming seam domain"
                )

    def trace(self, x0, x1):
        polys, floor = self._polynomials(x0, x1)
        start, end = self._which(polys, 0, floor), self._which(polys, 1, floor)
        bracket = None
        if start == end:
            if any(_minimum(p, 0, 1) <= floor for p in polys[start]):
                raise SolverFailure(
                    "deforming path touches an edge or leaves and returns to the same face"
                )
            if len(self.shared) != 2 and not any(
                -_minimum(-p, 0, 1) < -floor for p in polys[1 - start]
            ):
                raise SolverFailure("disconnected material overlap is unresolved")
            spans = ((start, 0.0, 1.0),)
        else:
            if len(self.shared) != 2:
                raise SolverFailure("deforming faces have disconnected material edges")
            opp = next(i for i, k in enumerate(self.face_ids[start]) if k not in self.shared)
            edge = polys[start][opp]
            if _at(edge, 1) >= -floor:
                raise SolverFailure("ambiguous or overlapping oriented material faces")
            lo, hi = 0.0, 1.0
            for _ in range(48):
                mid = (lo + hi) / 2
                if _at(edge, mid) > 0:
                    lo = mid
                else:
                    hi = mid
            tau = (lo + hi) / 2
            slope = abs(edge[1] + 2 * tau * edge[2])
            if slope <= floor:
                raise SolverFailure("material edge crossing is not transverse")
            margin = 4 * floor / slope
            if tau - margin <= 0 or tau + margin >= 1 or abs(edge[2]) * margin > slope / 4:
                raise SolverFailure("deforming material event is too close to a tie")
            bracket = (lo - margin, hi + margin)
            spans = ((start, 0.0, tau), (end, tau, 1.0))
            for index, a, b in spans:
                for i, node in enumerate(self.face_ids[index]):
                    if node in self.shared and _minimum(polys[index][i], a, b) <= floor:
                        raise SolverFailure(
                            "deforming path reaches a material vertex or another edge"
                        )
        out = []
        total = np.zeros(len(self.ids))
        bound = 0 if bracket is None else 2 * (bracket[1] - bracket[0])
        for j, a, b in spans:
            num = polys[j]
            den = np.sum(num, axis=0)
            wa, wb = _at(num, a) / _at(den, a), _at(num, b) / _at(den, b)
            integral, error = _integrate(num, den, a, b, 2e-12)
            if abs(np.sum(integral) - (b - a)) > 2e-12 or np.min(integral) < -2e-12:
                raise SolverFailure("invalid deforming material weight integral")
            idx = self.indices[j]
            total[np.asarray(idx) - 1] += integral
            bound += error + 64 * np.finfo(float).eps * (b - a)
            out.append(
                MovingFaceInterval(
                    self.keys[j], self.face_ids[j], idx, a, b, tuple(wa), tuple(wb), tuple(integral)
                )
            )
        return MovingTrace(tuple(out), total, bracket, bound)
