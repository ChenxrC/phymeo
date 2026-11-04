"""Discover material probe/hinge sites from finite registered geometry and clip their loaded intervals."""

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction as F
from numbers import Integral

import numpy as np

from ..constraints.collision_shapes import CollisionModel
from ..core.capabilities import SolverFailure
from ..core.state import ParticleState
from ..core.validation import finite_real
from ..solvers.collision_loads import CollisionLoadSystem
from ._sweep_polynomial import Roots, cross, difference, dot, mul, sub, value
from .folded_material_seam import FoldedMaterialChart, FoldStation
from .surface_features import closest_point_triangle


@dataclass(frozen=True, order=True)
class FoldSiteKey:
    point: tuple
    body: tuple
    edge: tuple


@dataclass(frozen=True)
class FoldActiveLeg:
    side: int
    t0: float
    t1: float
    start: FoldStation | None
    end: FoldStation | None


@dataclass(frozen=True)
class FoldActiveBand:
    """A positive-duration positive-compression episode, with zero-force boundary roots."""

    t0: float
    t1: float
    entry_bracket: tuple | None
    exit_bracket: tuple | None
    legs: tuple

    @property
    def start(self):
        return self.legs[0].start

    @property
    def end(self):
        return self.legs[-1].end


@dataclass(frozen=True)
class FoldSitePath:
    key: FoldSiteKey
    rows: tuple
    chart: FoldedMaterialChart
    intervals: tuple[FoldActiveBand, ...]


@dataclass(frozen=True)
class FoldSiteSelection:
    eligible: int
    broad_rejected: int
    examined: int
    sites: tuple[FoldSitePath, ...]


def _controls(p, dest):
    return tuple((F(float(x)), F(float(y)) - F(float(x))) for x, y in zip(p, dest, strict=True))


def _threshold_polynomial(x, xf, side, width):
    ps = [_controls(row, end) for row, end in zip(x, xf, strict=True)]
    a, b, c = ps[1], ps[2], ps[3 + side]
    normal = cross(difference(b, a), difference(c, a))
    height = dot(difference(ps[0], a), normal)
    return sub(mul(height, height), mul((F(float(width)) ** 2,), dot(normal, normal)))


def _root_cells(p, left, right):
    roots = Roots(p)
    result = []

    def isolate(a, b, count, depth):
        if not count:
            return
        if depth > 72:
            raise SolverFailure("fold layer threshold isolation budget exceeded")
        if count == 1 and value(p, b) == 0:
            result.append((b, b))
            return
        if count == 1 and b - a <= F(1, 2**44):
            result.append((a, b))
            return
        m = (a + b) / 2
        first = roots.count(a, m)
        isolate(a, m, first, depth + 1)
        isolate(m, b, count - first, depth + 1)

    if value(p, left) == 0:
        result.append((left, left))
    isolate(left, right, roots.count(left, right), 0)
    return result


def _loaded_windows(x, xf, side, a, b, bounds, width):
    if bounds[1] < width:
        return [(a, b, None, None)]
    if bounds[0] > width:
        return []
    p = _threshold_polynomial(x, xf, side, width)
    if not p:
        return []
    lo, hi = F(a), F(b)
    cells = _root_cells(p, lo, hi)
    if len(cells) > 3:
        raise SolverFailure("too many folded layer transitions in one material interval")
    interior = [c for c in cells if c != (lo, lo) and c != (hi, hi)]
    if any(c[0] <= lo or c[1] >= hi for c in interior):
        raise SolverFailure("layer threshold root is ambiguous at a material interval boundary")
    points = [(lo, lo), *interior, (hi, hi)]
    signs = [value(p, (left[1] + right[0]) / 2) < 0 for left, right in zip(points, points[1:])]
    if any(i == j for i, j in zip(signs, signs[1:])):
        raise SolverFailure("grazing folded layer threshold is unresolved")
    windows = []
    for inside, left, right in zip(signs, points, points[1:]):
        if inside:
            windows.append(
                (
                    float(sum(left) / 2),
                    float(sum(right) / 2),
                    tuple(map(float, left)) if left in cells else None,
                    tuple(map(float, right)) if right in cells else None,
                )
            )
    return windows


def _far(x, xf, radius):
    point_lo = np.minimum(x[0], xf[0])
    point_hi = np.maximum(x[0], xf[0])
    mesh_lo = np.minimum(x[1:], xf[1:]).min(axis=0)
    mesh_hi = np.maximum(x[1:], xf[1:]).max(axis=0)
    margin = 64 * np.finfo(float).eps * max(float(np.max(abs(x))), float(np.max(abs(xf))), 1.0)
    return (
        np.linalg.norm(np.maximum(0.0, np.maximum(point_lo - mesh_hi, mesh_lo - point_hi)))
        > radius + margin
    )


def _finite_clear(x, xf, triangles, radius):
    mid = (x + xf) / 2
    travel = (xf - x) - (xf[0] - x[0])
    margin = 64 * np.finfo(float).eps * max(float(np.max(abs(x))), float(np.max(abs(xf))), 1.0)
    for tri in triangles:
        rows = list(tri)
        distance = closest_point_triangle(mid[0], mid[rows]).distance
        if distance <= radius + margin + 0.5 * np.linalg.norm(travel[rows], axis=1).max():
            return False
    return True


class FoldedSiteSelector:
    """Topology-driven named material vertices and finite two-wing support charts.

    Particle shape registrations nominate probes. For a triangle-only model all material
    vertices are potential probes. A site needs two oppositely wound faces incident on the
    same actual edge. Same-body one-ring neighbours and filtered shape pairs are excluded.
    Unpaired faces near a probe or overlapping material chart owners are explicit failures.
    """

    def __init__(self, model, bindings, *, width_m, max_fold_angle_rad=np.pi / 12):
        if not isinstance(model, CollisionModel) or any(
            s.kind not in ("triangle", "particle") or s.radius for s in model.shapes
        ):
            raise ValueError(
                "fold selection requires unrounded particle and triangle registrations"
            )
        registered = CollisionLoadSystem(model, bindings, {})
        self.width = finite_real(width_m, "width_m")
        angle = finite_real(max_fold_angle_rad, "max_fold_angle_rad")
        if self.width <= 0 or not 0 < angle <= np.pi / 6:
            raise ValueError("invalid folded material search calibration")
        self.radius = self.width / np.cos(angle / 2)
        self._excluded = set(model.excluded)
        self._sources = defaultdict(list)
        self._rows = {}
        ring = defaultdict(set)
        incidence = defaultdict(list)
        triangles = []
        has_points = any(s.kind == "particle" for s in model.shapes)
        for shape in model.shapes:
            body = (shape.key.world, shape.key.body)
            for node, row in zip(
                shape.vertex_ids, registered.bindings[shape.key].rows, strict=True
            ):
                addr = (*body, node)
                self._rows[addr] = row
                if shape.kind == "particle" or not has_points:
                    self._sources[addr].append(shape)
            if shape.kind != "triangle":
                continue
            triangles.append(shape)
            for a, b in zip(
                shape.vertex_ids, shape.vertex_ids[1:] + shape.vertex_ids[:1], strict=True
            ):
                ring[(*body, a)].add((*body, b))
                ring[(*body, b)].add((*body, a))
                incidence[(body, tuple(sorted((a, b))))].append(shape)
        self._neighbours = ring
        self._hinges = []
        covered = set()
        for (body, edge), pair in sorted(incidence.items()):
            if len(pair) > 2:
                raise ValueError("nonmanifold material edge")
            if len(pair) == 1:
                continue
            nodes = sorted(set(pair[0].vertex_ids) | set(pair[1].vertex_ids))
            chart = FoldedMaterialChart(
                CollisionModel(pair), {n: k + 1 for k, n in enumerate(nodes)}, 0, angle
            )
            support = tuple(self._rows[(*body, n)] for n in chart.ids)
            self._hinges.append((body, edge, pair, chart, support))
            covered.update(s.key for s in pair)
        self._unpaired = tuple(s for s in triangles if s.key not in covered)
        self.row_count = max(self._rows.values(), default=-1) + 1

    def _allowed(self, point, faces):
        if not faces or faces[0].key.world != point[0]:
            return False
        target = set((s.key.world, s.key.body, i) for s in faces for i in s.vertex_ids)
        if point in target or not self._neighbours[point].isdisjoint(target):
            return False
        return any(
            all(
                (
                    s.group & t.mask
                    and t.group & s.mask
                    and tuple(sorted((s.key, t.key))) not in self._excluded
                )
                for t in faces
            )
            for s in self._sources[point]
        )

    def select(self, start, end=None, *, capacity=4096, detailed=True):
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, Integral)
            or not 0 <= capacity <= 262144
        ):
            raise ValueError("candidate capacity must be an integer in [0,262144]")

        def coords(p):
            return np.asarray(p.position if isinstance(p, ParticleState) else p)

        a = coords(start)
        b = a if end is None else coords(end)
        if (
            a.ndim != 2
            or a.shape[1] != 3
            or a.shape != b.shape
            or len(a) < self.row_count
            or a.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or b.dtype != a.dtype
            or not np.isfinite(a).all()
            or not np.isfinite(b).all()
        ):
            raise ValueError("expected matching finite physical particle rows")
        eps = float(np.finfo(a.dtype).eps)
        a = a.astype(float)
        b = b.astype(float)
        eligible = broad = examined = 0
        sites = []
        for point in sorted(self._sources):
            row = self._rows[point]
            for face in self._unpaired:
                if not self._allowed(point, (face,)):
                    continue
                idx = [
                    row,
                    *(self._rows[(face.key.world, face.key.body, n)] for n in face.vertex_ids),
                ]
                if not _far(a[idx], b[idx], self.radius) and not _finite_clear(
                    a[idx], b[idx], ((1, 2, 3),), self.radius
                ):
                    raise SolverFailure("unsupported nearby unpaired material face")
            for body, edge, faces, chart, support in self._hinges:
                if not self._allowed(point, faces):
                    continue
                eligible += 1
                rows = (row, *support)
                idx = list(rows)
                x, xf = a[idx], b[idx]
                if _far(x, xf, self.radius):
                    broad += 1
                    continue
                examined += 1
                if examined > capacity:
                    raise SolverFailure("folded candidate capacity exceeded")
                if _finite_clear(x, xf, ((1, 2, 3), (1, 2, 4)), self.radius):
                    continue
                pieces, _ = chart.intervals(x, xf, np.inf, eps)
                bands = []
                for side, lo, hi, gaps in pieces:
                    for left, right, entry, leave in _loaded_windows(
                        x, xf, side, lo, hi, gaps, self.width
                    ):
                        dx = xf - x
                        leg = FoldActiveLeg(
                            side,
                            left,
                            right,
                            chart.station(x + left * dx, side) if detailed else None,
                            chart.station(x + right * dx, side) if detailed else None,
                        )
                        if (
                            bands
                            and bands[-1].t1 == left
                            and bands[-1].exit_bracket is None
                            and entry is None
                        ):
                            prior = bands.pop()
                            bands.append(
                                FoldActiveBand(
                                    prior.t0, right, prior.entry_bracket, leave, (*prior.legs, leg)
                                )
                            )
                        else:
                            bands.append(FoldActiveBand(left, right, entry, leave, (leg,)))
                if not bands:
                    continue
                for other in sites:
                    if other.key.point == point and any(
                        max(i.t0, j.t0) < min(i.t1, j.t1) for i in bands for j in other.intervals
                    ):
                        raise SolverFailure(
                            "one material probe has simultaneous ambiguous folded charts"
                        )
                sites.append(
                    FoldSitePath(FoldSiteKey(point, body, edge), rows, chart, tuple(bands))
                )
        return FoldSiteSelection(eligible, broad, examined, tuple(sites))
