"""Filled convex rigid bodies versus a finite, independently deforming material triangle."""

from fractions import Fraction as F
from itertools import pairwise

import numpy as np

from ..constraints.collision_shapes import CollisionShape
from ..core.capabilities import SolverFailure
from ._convex_hull import clip_triangle_to_hull, direction_key, sparse_weights
from ._prism_clip import hull as planar_hull
from ._round_sweep import _GRID, _unit, _xyz
from ._tetrahedral_solid import _CellPose
from ._tetrahedral_surface import _TriangleMotion
from ._triangle_pair_sweep import _cross3, _FacePath, _TrianglePair
from .collision_witness import (
    CollisionSweep,
    CollisionWitness,
    HullSurfacePose,
    HullSurfaceTrace,
    MaterialContactPoint,
)


class HullSurfaceQuery:
    def __init__(self, hull_shape, hull, cloth, mc, hull_first):
        self.hull, self.cloth, self.hull_first = hull, cloth, hull_first
        self.face = _FacePath(cloth, mc)
        self.parts, self.cells = [], []
        features = {}
        origin = min(hull.vertices)
        if self.face.valid:
            for face in hull.faces:
                for j, k in pairwise(face.ids[1:]):
                    nodes = (face.ids[0], j, k)
                    points = np.array([hull.points[i] for i in nodes], dtype=object)
                    facet = CollisionShape(hull_shape.key, "triangle", nodes)
                    motion = _TriangleMotion(points, points + hull.step, bool(any(hull.step)))
                    # A swept AABB can prune feature construction. All original
                    # faces remain available for exact nearest-boundary samples.
                    cloth0, cloth1 = self.face.start, self.face.start + self.face.step
                    near = not (
                        any(
                            np.maximum(points, points + hull.step).max(0)
                            < np.minimum(cloth0, cloth1).min(0)
                        )
                        or any(
                            np.minimum(points, points + hull.step).min(0)
                            > np.maximum(cloth0, cloth1).max(0)
                        )
                    )
                    leaf = _TrianglePair(facet, cloth, motion, mc, sweep=near)
                    features.update(dict.fromkeys(leaf.features))
                    self.parts.append(leaf)
                    if origin not in nodes:
                        tet = _CellPose(np.stack((hull.points[origin], *points)))
                        if tet.det:
                            self.cells.append(((origin, *nodes), tet))
        self.features = tuple(features)
        self.samples = {}

    def _material(self, p):
        for nodes, tet in self.cells:
            weights = tet.weights(p)
            if min(weights) >= 0:
                return sparse_weights(nodes, weights)
        raise SolverFailure("hull_interior_material_unresolved")

    def _boundary(self, t):
        nearest = None
        for i, leaf in enumerate(self.parts):
            w, d2 = leaf.sample(t)
            key = (d2, i)
            if nearest is None or key < nearest[0]:
                nearest = (key, w)
        w = nearest[1]
        if not self.hull_first:
            normal = None if w.normal is None else tuple(-v for v in w.normal)
            w = CollisionWitness(
                w.gap_m,
                normal,
                w.second_point,
                w.first_point,
                w.second_weights,
                w.first_weights,
                "host-hull-deforming-surface",
            )
        return w

    def _supports(self, local):
        # Enumerate both orientations of every face and edge-cross SAT direction
        # at this pose. Faces incident to a cut vertex need not separate the shapes.
        rigid = self.hull
        edges = [local[i] - local[i - 1] for i in range(3)]
        axes = {direction_key(f.normal) for f in rigid.faces}
        axes.add(direction_key(_cross3(edges[0], edges[1])))
        axes.update(direction_key(_cross3(e, q)) for e in rigid.edge_rays for q in edges)
        rays = []
        vertices = [rigid.points[i] for i in rigid.vertices]
        for key in sorted(a for a in axes if a is not None):
            v = np.array(key, dtype=object)
            for d in (v, -v):
                if max(p @ d for p in vertices) == min(local @ d):
                    n = _unit(d)[0]
                    rays.append(_xyz(n if self.hull_first else -n))
        return tuple(rays)

    def pose(self, t):
        if t in self.samples:
            return self.samples[t]
        if not self.face.valid:
            raise SolverFailure("triangle_degenerate_on_path")
        hull = self.hull
        tri = self.face.start + t * self.face.step
        local = tri - t * hull.step
        heights = np.array(
            [[f.plane - x @ f.normal for f in hull.faces] for x in local], dtype=object
        )
        polygon, interior = clip_triangle_to_hull(heights)
        relation = "separated"
        section = []
        touched = set()
        area = 0.0
        if polygon:
            # The last two simplex coordinates form an injective planar chart.
            ring = planar_hull([(w[1], w[2]) for w in polygon])
            coords = []
            for u, v in ring:
                w = np.array((1 - u - v, u, v), dtype=object)
                p = w @ local
                xyz = w @ tri
                coords.append(xyz)
                wc = sparse_weights(self.cloth.vertex_ids, w)
                wh = None
                for face, col in zip(hull.faces, heights.T, strict=True):
                    if w @ col == 0:
                        i, j = face.cols
                        mapped = face.chart.weights((p[i], p[j]))
                        if mapped is None:
                            raise SolverFailure("hull_face_material_unresolved")
                        touched.add(face.ids)
                        if wh is None:
                            wh = mapped
                if wh is None:
                    wh = self._material(p)
                a, b = (wh, wc) if self.hull_first else (wc, wh)
                section.append(
                    MaterialContactPoint(
                        _xyz(xyz),
                        tuple((k, float(x)) for k, x in a),
                        tuple((k, float(x)) for k, x in b),
                    )
                )
            av = sum(
                (
                    _cross3(coords[i] - coords[0], coords[i + 1] - coords[0])
                    for i in range(1, len(coords) - 1)
                ),
                np.array((F(0),) * 3, dtype=object),
            )
            area = _unit(av)[1] / 2
            relation = ("inside" if min(heights.flat) >= 0 else "cut") if interior else "touching"
        rays = self._supports(local) if relation == "touching" else ()
        if section:
            s = section[0]
            normal = rays[0] if len(rays) == 1 else None
            witness = CollisionWitness(
                0.0,
                normal,
                s.point,
                s.point,
                s.first_weights,
                s.second_weights,
                "host-filled-hull-surface",
                intersection_points=tuple(section),
            )
        else:
            witness = self._boundary(t)
        pose = HullSurfacePose(
            relation, area, tuple(section), tuple(sorted(touched)), rays, witness
        )
        self.samples[t] = pose
        return pose

    def _plane_entry(self, bracket):
        lo, hi = map(F, bracket)
        if lo == hi:
            return lo if self.pose(lo).section else None
        for plane in self.hull.faces:
            heights = self.face.start @ plane.normal - plane.plane
            if min(heights) <= 0:
                continue
            rates = (self.face.step - self.hull.step) @ plane.normal
            roots = [-v / r for v, r in zip(heights, rates, strict=True) if r < 0]
            if roots:
                t = min(roots)
                if lo <= t <= hi and self.pose(t).section:
                    return t
        return None

    def result(self, moving, seeds):
        start = self.pose(F(0))
        end = self.pose(F(1)) if moving else start
        first = (0.0, 0.0) if start.section else None
        exact = F(0) if start.section else None
        if moving and not start.section:
            brackets = [
                cell
                for f, seed in zip(self.features, seeds, strict=True)
                if (cell := f.entry(seed)) is not None
            ]
            if brackets:
                first = tuple(min(b[k] for b in brackets) / _GRID for k in (0, 1))
                exact = self._plane_entry(first)
        time = None if first is None else (exact if exact is not None else F(sum(first) / 2))
        entry = None if time is None else self.pose(time)
        trace = HullSurfaceTrace(
            start, end, entry, None if exact is None else (exact.numerator, exact.denominator)
        )
        if not moving:
            status = end.relation if end.relation in ("separated", "touching") else "overlap"
            return status, end.relation, end.witness, None, trace
        swept = CollisionSweep(
            start.witness.gap_m,
            end.witness.gap_m,
            first,
            None if time is None else float(time),
            None if entry is None else entry.witness,
        )
        return ("hit" if first else "separated"), "hull_surface_sweep", end.witness, swept, trace
