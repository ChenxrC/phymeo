"""Point-cloud convex solids, their true edges and stable surface material charts."""

from itertools import combinations

import numpy as np

from ..constraints.collision_shapes import MAX_CONVEX_VERTICES
from ..constraints.conforming_shell import _cyclic
from ..core.capabilities import SolverFailure
from ._prism_clip import hull, inside, orient
from ._round_sweep import _translation, _vector
from ._tetrahedral_surface import _clip
from ._triangle_pair_sweep import _cross3


def direction_key(vector, *, signed=False):
    scale = max(map(abs, vector))
    if not scale:
        return None
    first = next(v for v in vector if v)
    if not signed and first < 0:
        scale = -scale
    return tuple(v / scale for v in vector)


def plane_axes(normal):
    drop = max(range(3), key=lambda i: abs(normal[i]))
    return (drop + 1) % 3, (drop + 2) % 3


def sparse_weights(ids, weights):
    return tuple(sorted((i, w) for i, w in zip(ids, weights, strict=True) if w))


class MaterialChart:
    """Vertex, segment or convex polygon in an injective two-dimensional projection."""

    def __init__(self, coordinates):
        ring = hull(coordinates)
        self.ids = {p: coordinates[p] for p in ring}
        # Anchor the face fan to a material ID so a storage permutation or a
        # change of projection chart does not change the material interpolation.
        i = min(range(len(ring)), key=lambda i: coordinates[ring[i]])
        self.ring = ring[i:] + ring[:i]

    def weights(self, point):
        if point in self.ids:
            return ((self.ids[point], 1),)
        if not inside(point, self.ring):
            return None
        a = self.ring[0]
        if len(self.ring) == 2:
            b = self.ring[1]
            k = 0 if b[0] != a[0] else 1
            v = (point[k] - a[k]) / (b[k] - a[k])
            return sparse_weights((self.ids[a], self.ids[b]), (1 - v, v))
        for b, c in zip(self.ring[1:-1], self.ring[2:], strict=True):
            den = orient(a, b, c)
            u, v = orient(a, point, c) / den, orient(a, b, point) / den
            w = (1 - u - v, u, v)
            if min(w) >= 0:
                return sparse_weights((self.ids[a], self.ids[b], self.ids[c]), w)
        raise SolverFailure("convex_material_chart_unresolved")


class _Face:
    def __init__(self, normal, plane, points):
        self.normal = np.array(normal, dtype=object)
        self.plane = plane
        self.cols = plane_axes(normal)
        i, j = self.cols
        support = {(p[i], p[j]): node for node, p in points.items() if p @ self.normal == plane}
        self.chart = MaterialChart(support)
        ids = tuple(support[p] for p in self.chart.ring)
        if normal[3 - i - j] < 0:
            ids = tuple(reversed(ids))
        self.ids = _cyclic(ids)


class HullShape:
    def __init__(self, shape, path):
        if len(shape.vertex_ids) > MAX_CONVEX_VERTICES:
            raise SolverFailure("convex_vertex_capacity")
        self.points = dict(
            sorted((node, _vector(p)) for node, p in zip(shape.vertex_ids, path.start, strict=True))
        )
        if len({tuple(p) for p in self.points.values()}) != len(self.points):
            raise ValueError("convex_duplicate_coordinates")
        self.step = _translation(path)
        self.faces = self._boundary()
        self.vertices = tuple(sorted({i for f in self.faces for i in f.ids}))
        incidence = {}
        for face in self.faces:
            for i, j in zip(face.ids, face.ids[1:] + face.ids[:1], strict=True):
                incidence.setdefault(tuple(sorted((i, j))), []).append(i)
        if any(len(ends) != 2 or ends[0] == ends[1] for ends in incidence.values()):
            raise SolverFailure("convex_surface_incidence")
        self.edges = tuple(sorted(incidence))
        self.edge_rays = tuple(
            sorted({direction_key(self.points[j] - self.points[i]) for i, j in self.edges})
        )

    def _boundary(self):
        pts = self.points
        ids = tuple(pts)
        a, b = ids[:2]
        u = pts[b] - pts[a]
        c = next((k for k in ids[2:] if any(_cross3(u, pts[k] - pts[a]))), None)
        if c is None:
            raise ValueError("convex_not_full_dimensional")
        n = _cross3(u, pts[c] - pts[a])
        d = next((k for k in ids if n @ (pts[k] - pts[a])), None)
        if d is None:
            raise ValueError("convex_not_full_dimensional")
        initial = (a, b, c, d)
        interior = sum((pts[k] for k in initial)) / 4
        triangles = {}

        def add_triangle(i, j, k):
            normal = _cross3(pts[j] - pts[i], pts[k] - pts[i])
            sign = normal @ (interior - pts[i])
            if not sign:
                raise SolverFailure("convex_horizon_degenerate")
            if sign > 0:
                j, k, normal = k, j, -normal
            triangles[_cyclic((i, j, k))] = normal

        for face in combinations(initial, 3):
            add_triangle(*face)
        for k in ids:
            if k in initial:
                continue
            visible = [f for f, n in triangles.items() if n @ (pts[k] - pts[f[0]]) > 0]
            horizon = {}
            for f in visible:
                del triangles[f]
                for i, j in zip(f, f[1:] + f[:1], strict=True):
                    key = tuple(sorted((i, j)))
                    if key in horizon:
                        del horizon[key]
                    else:
                        horizon[key] = (i, j)
            for i, j in horizon.values():
                add_triangle(i, j, k)

        planes = set()
        for f, normal in triangles.items():
            direction = direction_key(normal, signed=True)
            h = pts[f[0]] @ direction
            planes.add((direction, h))
        faces = []
        for direction, h in sorted(planes):
            if any(p @ direction > h for p in pts.values()):
                raise SolverFailure("convex_surface_not_supporting")
            faces.append(_Face(direction, h, pts))
        return tuple(sorted(faces, key=lambda f: f.ids))

    def point(self, weights, time):
        return sum(w * self.points[i] for i, w in weights) + time * self.step


def clip_triangle_to_hull(heights):
    """Closed triangle section in its own barycentric coordinates and interior occupancy."""
    polygon = list(np.eye(3, dtype=int).astype(object))
    for column in heights.T:
        polygon = _clip(polygon, column)
        if not polygon:
            return [], False
    # The centroid lies strictly inside exactly when every plane has positive
    # slack somewhere on the section. This also covers line and point sections.
    interior = all(any(w @ column > 0 for w in polygon) for column in heights.T)
    return polygon, interior
