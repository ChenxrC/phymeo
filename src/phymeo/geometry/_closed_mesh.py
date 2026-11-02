"""Translating convex solids against a connected, closed triangle boundary."""

from dataclasses import replace
from fractions import Fraction as F
from itertools import combinations
from math import ceil, floor

import numpy as np

from ..constraints.collision_shapes import CollisionShape
from ..core.capabilities import SolverFailure
from ._convex_contact import HullPairQuery, _closest
from ._convex_hull import _Face, clip_triangle_to_hull, direction_key
from ._round_sweep import _GRID, _translation, _unit, _vector, _xyz
from ._tetrahedral_surface import _TriangleMotion
from ._triangle_pair_sweep import _cross3, _TrianglePair
from .collision_witness import (
    CollisionSweep,
    CollisionWitness,
    MaterialContactPoint,
    MeshContactPose,
    MeshContactTrace,
)


def _connected(adjacency):
    pending = [next(iter(adjacency))]
    visited = set()
    while pending:
        i = pending.pop()
        if i not in visited:
            visited.add(i)
            pending.extend(adjacency[i] - visited)
    return len(visited) == len(adjacency)


class _Facet:
    """One filled triangle with the vertex/edge/plane interface used by convex SAT."""

    def __init__(self, shape, points, ids, step):
        self.ids = ids
        self.vertices = tuple(sorted(ids))
        self.points = {i: points[i] for i in ids}
        self.step = step
        self.xyz = np.array([points[i] for i in ids], dtype=object)
        a, b, c = self.xyz
        normal = direction_key(_cross3(b - a, c - a), signed=True)
        if normal is None:
            raise ValueError("mesh_degenerate_triangle")
        self.face = _Face(normal, a @ normal, self.points)
        self.faces = (self.face,)
        self.edges = tuple(combinations(self.vertices, 2))
        self.edge_rays = tuple({direction_key(points[j] - points[i]) for i, j in self.edges})
        self.shape = CollisionShape(shape.key, "triangle", ids)
        self.stationary = _TriangleMotion(self.xyz, self.xyz, False)
        self.lo, self.hi = self.xyz.min(0), self.xyz.max(0)

    def point(self, weights, t):
        return sum(w * self.points[i] for i, w in weights) + t * self.step

    def weights(self, local):
        i, j = self.face.cols
        return self.face.chart.weights((local[i], local[j]))


class ClosedMesh:
    def __init__(self, shape, path):
        points = {i: _vector(p) for i, p in zip(shape.vertex_ids, path.start, strict=True)}
        if len(set(map(tuple, points.values()))) != len(points):
            raise ValueError("mesh_coincident_nodes")
        edges = {}
        links = {i: {} for i in points}
        adjacency = {i: set() for i in points}
        seen = set()
        for f in shape.triangles:
            key = tuple(sorted(f))
            if key in seen:
                raise ValueError("mesh_duplicate_face")
            seen.add(key)
            for i, j, k in (f, f[1:] + f[:1], f[2:] + f[:2]):
                edges.setdefault(tuple(sorted((i, j))), []).append((i, j))
                links[i].setdefault(j, set()).add(k)
                links[i].setdefault(k, set()).add(j)
                adjacency[i].update((j, k))
        if any(len(pair) != 2 or pair[0] != pair[1][::-1] for pair in edges.values()):
            raise ValueError("mesh_open_or_misoriented_edge")
        if not _connected(adjacency) or any(
            not _connected(link) or any(len(n) != 2 for n in link.values())
            for link in links.values()
        ):
            raise ValueError("mesh_component_or_vertex_link")
        self.points, self.step = points, _translation(path)
        self.facets = tuple(_Facet(shape, points, f, self.step) for f in shape.triangles)
        origin = next(iter(points.values()))
        volume6 = sum(
            (a - origin) @ _cross3(b - origin, c - origin)
            for a, b, c in (f.xyz for f in self.facets)
        )
        if volume6 <= 0:
            raise ValueError("mesh_nonpositive_or_inward")
        for a, b in combinations(self.facets, 2):
            if any(a.hi < b.lo) or any(b.hi < a.lo):
                continue
            shared = set(a.ids) & set(b.ids)
            if len(shared) == 2:
                if any(_cross3(a.face.normal, b.face.normal)):
                    continue
                if a.face.normal @ b.face.normal > 0:
                    continue
                raise ValueError("mesh_folded_shared_edge")
            # For a single shared node, a strict one-sided fan cannot meet again.
            if shared:
                heights = [
                    a.face.normal @ b.points[i] - a.face.plane for i in b.ids if i not in shared
                ]
                if min(heights) > 0 or max(heights) < 0:
                    continue
            leaf = _TrianglePair(a.shape, b.shape, a.stationary, b.stationary)
            if leaf.sample(F(0))[0].intersection_points:
                raise ValueError("mesh_self_intersection")

    def inside(self, point):
        """Parity for a point off the boundary. Retry rays hitting an edge or a vertex."""
        for k in range(8 * len(self.facets) + 3):
            ray = np.array((1, k, k * k), dtype=object)
            crossings = 0
            for facet in self.facets:
                face = facet.face
                den = ray @ face.normal
                height = face.plane - point @ face.normal
                if den == 0:
                    if height == 0:
                        break
                    continue
                t = height / den
                if t < 0:
                    continue
                weights = facet.weights(point + t * ray)
                if weights is None:
                    continue
                if len(weights) < 3 or t == 0:
                    break
                crossings += 1
            else:
                return bool(crossings % 2)
        raise SolverFailure("mesh_ray_capacity")


class MeshHullQuery:
    features = ()

    def __init__(self, mesh, hull, mesh_first):
        self.mesh, self.hull, self.mesh_first = mesh, hull, mesh_first
        self.parts = tuple(HullPairQuery(f, hull) for f in mesh.facets)

    def _nearest(self, t):
        # A vertex pair supplies a valid upper bound for pruning distant facets.
        hp = np.array([self.hull.points[i] for i in self.hull.vertices]) + t * self.hull.step
        upper = min(
            (d @ d) for p in self.mesh.points.values() for d in hp - (p + t * self.mesh.step)
        )
        low, high = hp.min(0), hp.max(0)
        candidates = []
        for f in self.mesh.facets:
            delta = np.maximum(0, np.maximum(f.lo + t * f.step - high, low - f.hi - t * f.step))
            if delta @ delta <= upper:
                a, b = (f, self.hull) if self.mesh_first else (self.hull, f)
                candidates.append(_closest(a, b, t))
        return min(candidates, key=lambda w: (w.gap_m, w.first_weights, w.second_weights))

    def sample(self, t):
        mesh, hull = self.mesh, self.hull
        relative = t * (mesh.step - hull.step)
        planes = {
            i: tuple(face.plane - (point + relative) @ face.normal for face in hull.faces)
            for i, point in mesh.points.items()
        }
        mesh_in_hull = all(min(row) >= 0 for row in planes.values())
        interior = False
        contacts = {}
        facets = []
        for tri in mesh.facets:
            heights = np.array([planes[i] for i in tri.ids], dtype=object)
            polygon, strict = clip_triangle_to_hull(heights)
            if not polygon:
                continue
            interior = interior or strict
            touch = False
            for w in polygon:
                local = w @ tri.xyz + relative
                wm = tuple(sorted((i, v) for i, v in zip(tri.ids, w, strict=True) if v))
                for column, face in zip(heights.T, hull.faces, strict=True):
                    if w @ column:
                        continue
                    i, j = face.cols
                    wh = face.chart.weights((local[i], local[j]))
                    if wh is None:
                        raise SolverFailure("mesh_hull_surface_address")
                    first, second = (wm, wh) if self.mesh_first else (wh, wm)
                    key = (first, second)
                    contacts[key] = MaterialContactPoint(
                        _xyz(local + t * hull.step),
                        tuple((i, float(v)) for i, v in first),
                        tuple((i, float(v)) for i, v in second),
                    )
                    touch = True
            if touch:
                facets.append(tri)
        if mesh_in_hull and interior:
            relation = "hull_contains_mesh"
        elif interior:
            relation = "overlap"
        else:
            center = sum(hull.points[i] for i in hull.vertices) / len(hull.vertices) - relative
            hull_in_mesh = mesh.inside(center)
            if mesh_in_hull and hull_in_mesh:
                relation = "coincident"
            elif hull_in_mesh:
                relation = "mesh_contains_hull"
            elif mesh_in_hull:
                relation = "hull_contains_mesh"
            else:
                relation = "touching" if contacts else "separated"
        if relation == "hull_contains_mesh":
            relation = "second_contains_first" if self.mesh_first else "first_contains_second"
        elif relation == "mesh_contains_hull":
            relation = "first_contains_second" if self.mesh_first else "second_contains_first"
        hits = tuple(contacts[k] for k in sorted(contacts))
        if hits:
            first = hits[0]
            rays = {tuple(f.face.normal) for f in facets}
            normal = None
            if relation == "touching" and len(rays) == 1:
                direction = _unit(np.array(next(iter(rays)), dtype=object))[0]
                normal = _xyz(direction if self.mesh_first else -direction)
            witness = CollisionWitness(
                0.0,
                normal,
                first.point,
                first.point,
                first.first_weights,
                first.second_weights,
                "host-closed-mesh-boundary",
                intersection_points=hits,
            )
            distance = 0.0
        else:
            boundary = self._nearest(t)
            distance = boundary.gap_m
            witness = replace(
                boundary,
                gap_m=distance if relation == "separated" else -distance,
                source="host-closed-mesh-boundary",
            )
        return MeshContactPose(relation, distance, witness, tuple(f.ids for f in facets), hits)

    def result(self, moving, proposals):
        start = self.sample(F(0))
        end = self.sample(F(1)) if moving else start
        first = F(0) if start.relation != "separated" else None
        if moving and first is None:
            seeds = proposals if proposals is not None else [None] * len(self.parts)
            windows = (p._window(s) for p, s in zip(self.parts, seeds, strict=True))
            first = min((w[0] for w in windows if w is not None), default=None)
        entry = (start if first == 0 else self.sample(first)) if first is not None else None
        ratio = (first.numerator, first.denominator) if first is not None else None
        trace = MeshContactTrace(start, end, ratio, entry)
        if not moving:
            status = end.relation if end.relation in ("separated", "touching") else "overlap"
            return status, end.relation, end.witness, None, trace
        bracket = (
            None if first is None else (floor(first * _GRID) / _GRID, ceil(first * _GRID) / _GRID)
        )
        sweep = CollisionSweep(
            start.witness.gap_m,
            end.witness.gap_m,
            bracket,
            float(first) if entry else None,
            entry.witness if entry else None,
        )
        return ("hit" if entry else "separated"), "closed_mesh_sweep", end.witness, sweep, trace
