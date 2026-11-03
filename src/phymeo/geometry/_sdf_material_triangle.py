"""Cell ownership and rigorous scalar bounds on the actual finite material triangle."""

from dataclasses import dataclass, field
from fractions import Fraction as F
from itertools import permutations, product
from math import floor, prod

from ._prism_clip import edges, hull
from ._sheet_pair_bounds import _cross, _dot, _sub

_CHART = ((F(0), F(0)), (F(1), F(0)), (F(0), F(1)))


def average(points):
    return tuple(sum(c) / len(points) for c in zip(*points, strict=True))


def affine(points, weights):
    return tuple(
        sum(w * x for w, x in zip(weights, c, strict=True)) for c in zip(*points, strict=True)
    )


def _clip(poly, at, axis, limit, direction):
    heights = {a: direction * (at(a)[axis] - limit) for a in poly}
    kept = [a for a in poly if heights[a] >= 0]
    for a, b in edges(poly):
        ha, hb = heights[a], heights[b]
        if ha * hb < 0:
            t = ha / (ha - hb)
            kept.append(tuple(x + t * (y - x) for x, y in zip(a, b, strict=True)))
    return hull(kept)


class _Cell:
    """Stored tensor-product polynomial in fractional coordinates of one cell."""

    def __init__(self, grid, cell, radius):
        self.cell = cell
        x, y, z = cell
        values = grid.values_zyx_m[z : z + 2, y : y + 2, x : x + 2].ravel()
        self.valid = grid.band_m is None or all(abs(v) < grid.band_m for v in values)
        a, b, c, d, e, f, g, h = map(F, values)
        self.coeff = (
            a - radius,
            b - a,
            c - a,
            e - a,
            d - b - c + a,
            f - b - e + a,
            g - c - e + a,
            h - d - f - g + b + c + e - a,
        )

    def value_gradient(self, q):
        x, y, z = _sub(q, self.cell)
        a, b, c, d, e, f, g, h = self.coeff
        return (
            a + b * x + c * y + d * z + e * x * y + f * x * z + g * y * z + h * x * y * z,
            (
                b + e * y + f * z + h * y * z,
                c + e * x + g * z + h * x * z,
                d + f * x + g * y + h * x * y,
            ),
        )


@dataclass
class FieldPlan:
    addresses: dict = field(default_factory=dict)
    reasons: set = field(default_factory=set)
    lower: F | None = None
    cells_examined: int = 0
    cells_covered: int = 0


class _SampleCapacity(Exception):
    pass


class _Builder:
    def __init__(self, grid, q, radius, samples):
        self.grid, self.vertices, self.radius, self.limit = grid, q, radius, samples
        self.cells = {}
        self.plan = FieldPlan()

    def cell(self, index):
        if index not in self.cells:
            self.cells[index] = _Cell(self.grid, index, self.radius)
        return self.cells[index]

    def at(self, uv):
        return affine(self.vertices, (1 - sum(uv), *uv))

    def address(self, weights, q):
        if weights in self.plan.addresses:
            return self.plan.addresses[weights][2]
        if len(self.plan.addresses) >= self.limit:
            raise _SampleCapacity
        if any(v < 0 or v > n - 1 for v, n in zip(q, self.grid.shape_xyz, strict=True)):
            status, owner, gap = "outside_grid", None, None
        else:
            owner = tuple(min(floor(v), n - 2) for v, n in zip(q, self.grid.shape_xyz, strict=True))
            cell = self.cell(owner)
            status = "valid" if cell.valid else "outside_band"
            gap = cell.value_gradient(q)[0] if cell.valid else None
        if status != "valid":
            self.plan.reasons.add(status)
        self.plan.addresses[weights] = (status, owner, gap)
        return gap

    def probe(self, uv):
        return self.address((1 - sum(uv), *uv), self.at(uv))

    def patch(self, tri, cell, depth):
        # The restriction to a material subtriangle is a polynomial of total
        # degree <=3. Its ten cubic Bernstein coefficients bound all field
        # values. Edge coefficients come from endpoint directional derivatives.
        xyz = [self.at(a) for a in tri]
        vg = [cell.value_gradient(q) for q in xyz]
        coefficients = [v for v, _ in vg]
        edge = [
            vg[i][0] + _dot(vg[i][1], _sub(xyz[j], xyz[i])) / 3
            for i, j in permutations(range(3), 2)
        ]
        center = average(tri)
        vc = cell.value_gradient(self.at(center))[0]
        coefficients.append((27 * vc - sum(coefficients) - 3 * sum(edge)) / 6)
        lower = min(*coefficients, *edge)
        mids = [average((tri[i], tri[(i + 1) % 3])) for i in range(3)]
        witnessed = [self.probe(a) for a in (*tri, *mids, center)]
        if depth and lower <= 0 and not any(g is not None and g <= 0 for g in witnessed):
            a, b, c = tri
            ab, bc, ca = mids
            return min(
                self.patch(t, cell, depth - 1)
                for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))
            )
        return lower

    def triangle(self, max_cells, depth):
        bounds = [(min(v), max(v)) for v in zip(*self.vertices, strict=True)]
        ranges = []
        for (lo, hi), n in zip(bounds, self.grid.shape_xyz, strict=True):
            if hi < 0 or lo > n - 1:
                return
            ranges.append(range(max(0, min(n - 2, floor(lo))), min(n - 2, floor(hi)) + 1))
        cost = prod(map(len, ranges))
        if cost > max_cells:
            self.plan.reasons.add("cell_capacity")
            return
        self.plan.cells_examined = cost
        for index in product(*ranges):
            poly = _CHART
            for k, j in enumerate(index):
                poly = _clip(poly, self.at, k, j, 1)
                poly = _clip(poly, self.at, k, j + 1, -1)
                if not poly:
                    break
            if not poly:
                continue
            # Upper interior grid planes are owned by their positive-side cell.
            if any(
                j + 1 < n - 1 and all(self.at(a)[k] == j + 1 for a in poly)
                for k, (j, n) in enumerate(zip(index, self.grid.shape_xyz, strict=True))
            ):
                continue
            self.plan.cells_covered += 1
            cell = self.cell(index)
            if not cell.valid:
                self.plan.reasons.add("outside_band")
                self.probe(average(poly))
                continue
            triangles = [(poly[0], poly[j], poly[j + 1]) for j in range(1, len(poly) - 1)]
            if not triangles:
                # A clipped segment or point uses the same polynomial restricted
                # to a degenerate chart triangle, retaining its exact endpoint set.
                triangles = [(poly[0], poly[-1], poly[-1])]
            for tri in triangles:
                low = self.patch(tri, cell, depth)
                self.plan.lower = low if self.plan.lower is None else min(self.plan.lower, low)


def plan_material(grid, local, radius, max_cells, max_samples, max_depth):
    q = [
        tuple((v - o) / h for v, o, h in zip(p, grid._origin, grid._spacing, strict=True))
        for p in local
    ]
    b = _Builder(grid, q, F(radius), max_samples)
    try:
        if len(q) == 1:
            b.plan.lower = b.address((F(1),), q[0])
        elif not any(_cross(_sub(q[1], q[0]), _sub(q[2], q[0]))):
            b.plan.reasons.add("degenerate_triangle")
        else:
            for uv in _CHART:
                b.probe(uv)
            b.triangle(max_cells, max_depth)
    except _SampleCapacity:
        # Unfinished domain coverage cannot provide a lower bound. Previously
        # verified addresses can still prove material contact independently.
        b.plan.reasons.add("sample_capacity")
    return b.plan
