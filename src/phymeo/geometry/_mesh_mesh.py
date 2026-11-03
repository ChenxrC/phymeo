"""Physical surface witnesses for two independently registered closed rigid meshes."""

from dataclasses import dataclass, replace
from fractions import Fraction as F
from math import ceil, floor

import numpy as np

from ..core.capabilities import SolverFailure
from ._collision_bvh import CollisionTrees
from ._convex_contact import HullPairQuery, _Axis
from ._convex_hull import direction_key, sparse_weights
from ._prism_clip import hull, intersect
from ._round_sweep import _GRID, _unit, _xyz
from ._tetrahedral_surface import _TriangleMotion
from ._triangle_pair_sweep import _cross3, _TrianglePair
from .collision_witness import CollisionSweep, CollisionWitness


@dataclass(frozen=True)
class MeshMeshSupport:
    first_faces: tuple[tuple[int, int, int], ...]
    second_faces: tuple[tuple[int, int, int], ...]
    witness: CollisionWitness


@dataclass(frozen=True)
class MeshMeshPose:
    relation: str
    boundary_gap_m: float
    vertex_intrusion_m: float | None
    deepest: tuple[MeshMeshSupport, ...]
    boundary: tuple[MeshMeshSupport, ...]
    witness: CollisionWitness
    boundary_facet_pairs_tested: int


@dataclass(frozen=True)
class MeshMeshTrace:
    start: MeshMeshPose
    end: MeshMeshPose
    first_ratio: tuple[int, int] | None
    entry: MeshMeshPose | None


def _owners(mesh, weights):
    nodes = {i for i, w in weights if w}
    return tuple(f.ids for f in mesh.facets if nodes.issubset(f.ids))


def _support(a, b, wa, wb, t, *, intrusion=False, normal=None):
    pa = sum(w*a.points[i] for i, w in wa) + t*a.step
    pb = sum(w*b.points[i] for i, w in wb) + t*b.step
    direction, length = _unit(pb - pa)
    if direction is not None:
        normal = _xyz(-direction if intrusion else direction)
    w = CollisionWitness(
        -length if intrusion else length, normal, _xyz(pa), _xyz(pb),
        tuple((i, float(v)) for i, v in wa), tuple((i, float(v)) for i, v in wb),
        "host-mesh-mesh",
    )
    return MeshMeshSupport(_owners(a, wa), _owners(b, wb), w)


def _order(s):
    w = s.witness
    # This geometric ordering is invariant to exchanging the two owners.
    return tuple(sorted(((w.first_point, w.first_weights), (w.second_point, w.second_weights))))


def _representative(supports):
    first = min(supports, key=_order).witness
    directions = {s.witness.normal for s in supports}
    return first if len(directions) == 1 else replace(first, normal=None)


def _area2(xy):
    return abs(sum(p[0]*q[1] - p[1]*q[0] for p, q in zip(xy, xy[1:] + xy[:1], strict=True)))


def _slice(xyz, heights):
    points = {tuple(p) for p, h in zip(xyz, heights, strict=True) if not h}
    for i, j in ((0, 1), (1, 2), (2, 0)):
        u, v = heights[i], heights[j]
        if u*v < 0:
            points.add(tuple(xyz[i] + (u/(u-v))*(xyz[j] - xyz[i])))
    return points


def _intersection(a, b, t):
    """Exact intersection generators, projected coplanar area and strict transverse crossing."""
    x, y = a.xyz + t*a.step, b.xyz + t*b.step
    n, m = a.face.normal, b.face.normal
    ah = (x - y[0]) @ m
    bh = (y - x[0]) @ n
    if min(ah) > 0 or max(ah) < 0 or min(bh) > 0 or max(bh) < 0:
        return (), 0, False
    axis = _cross3(n, m)
    if not any(axis):
        i, j = a.face.cols
        k = 3-i-j
        xy = intersect(hull((v[i], v[j]) for v in x), hull((v[i], v[j]) for v in y))
        points = []
        for u, v in xy:
            p = x[0].copy()
            p[i], p[j] = u, v
            p[k] = (x[0] @ n - n[i]*u - n[j]*v)/n[k]
            points.append(p)
        return points, _area2(xy), False
    left, right = _slice(x, ah), _slice(y, bh)
    k = max(range(3), key=lambda j: abs(axis[j]))
    lo = max(min(p[k] for p in group) for group in (left, right))
    hi = min(max(p[k] for p in group) for group in (left, right))
    points = [np.array(p, dtype=object) for p in sorted(left | right) if lo <= p[k] <= hi]
    transverse = False
    if len(points) == 2:
        midpoint = (points[0] + points[1])/2
        wa, wb = a.weights(midpoint-t*a.step), b.weights(midpoint-t*b.step)
        transverse = len(wa) == len(wb) == 3
    return points, 0, transverse


def _point_nearest(faces, p):
    best, hits = None, set()
    for f in faces:
        d = np.maximum(0, np.maximum(f.lo - p, p - f.hi))
        if best is not None and d @ d > best:
            continue
        n = f.face.normal
        projected = p - n*((p @ n - f.face.plane)/(n @ n))
        wa = f.weights(projected)
        options = []
        if wa is not None:
            options.append((projected, wa))
        else:
            for i, j in f.edges:
                u = f.points[j] - f.points[i]
                along = min(F(1), max(F(0), ((p-f.points[i]) @ u)/(u @ u)))
                options.append((f.points[i] + along*u, sparse_weights((i, j), (1-along, along))))
        for q, w in options:
            delta = q - p
            d2 = delta @ delta
            if best is None or d2 < best:
                best, hits = d2, set()
            if d2 == best:
                hits.add(w)
    return best, hits


class MeshMeshQuery:
    features = ()

    def __init__(self, first, second):
        self.a, self.b = first, second
        self.trees, self.layouts = CollisionTrees(), {}

    @staticmethod
    def _boxes(mesh, t0, t1, margin):
        d0, d1 = mesh.step*t0, mesh.step*t1
        lo, hi = np.minimum(d0, d1)-margin, np.maximum(d0, d1)+margin
        bounds = np.array([(*tuple(f.lo+lo), *tuple(f.hi+hi)) for f in mesh.facets], dtype=float)
        maximum = np.finfo(np.float32).max
        raw = np.clip(bounds, -maximum, maximum).astype(np.float32)
        with np.errstate(over="ignore", under="ignore"):
            raw[:, :3] = np.nextafter(raw[:, :3], np.float32(-np.inf))
            raw[:, 3:] = np.nextafter(raw[:, 3:], np.float32(np.inf))
        return np.clip(raw, -maximum, maximum)

    def facet_pairs(self, t0, t1, margin=0):
        leaves = self._boxes(self.b, t0, t1, 0)
        queries = self._boxes(self.a, t0, t1, margin)
        rows = self.trees.facets(self.layouts, ("mesh", "second"), leaves, queries, None)
        return ((i, j) for i, row in enumerate(rows) for j in row)

    def _intrusions(self, t):
        largest, pairs, inside = None, set(), [False, False]
        for side, (source, target) in enumerate(((self.a, self.b), (self.b, self.a))):
            target_points = np.array(list(target.points.values()))
            lo, hi = target_points.min(0), target_points.max(0)
            shift = t*(source.step-target.step)
            for i, original in source.points.items():
                p = original+shift
                if any(p <= lo) or any(p >= hi):
                    continue
                d2, supports = _point_nearest(target.facets, p)
                if not d2 or not target.inside(p):
                    continue
                inside[side] = True
                if largest is None or d2 > largest:
                    largest, pairs = d2, set()
                if d2 == largest:
                    node = ((i, F(1)),)
                    pairs.update((w, node) if side else (node, w) for w in supports)
        deepest = sorted((_support(self.a, self.b, u, v, t, intrusion=True) for u, v in pairs),
                         key=_order)
        return tuple(deepest), inside

    def _nearest(self, t):
        a, b = self.a, self.b
        motion = t*(b.step-a.step)
        # Any vertex-pair L1 distance bounds the Euclidean boundary distance above.
        pad = min(sum(abs(q+motion-p)) for p in a.points.values() for q in b.points.values())
        candidates = []
        for i, j in self.facet_pairs(t, t, pad):
            f, g = a.facets[i], b.facets[j]
            d = np.maximum(0, np.maximum(f.lo - g.hi - motion, g.lo + motion - f.hi))
            candidates.append((d @ d, i, j))
        best, nearest = None, []
        for lower, i, j in sorted(candidates):
            if best is not None and lower > best:
                break
            f, g = a.facets[i], b.facets[j]
            x, y = f.xyz+t*f.step, g.xyz+t*g.step
            # Use the same internal owner order if a caller exchanges its shape keys.
            reverse = (tuple(sorted(map(tuple, y))), g.ids) < (tuple(sorted(map(tuple, x))), f.ids)
            u, v, left, right = (g, f, y, x) if reverse else (f, g, x, y)
            pair = _TrianglePair(u.shape, v.shape, _TriangleMotion(left,left,False),
                                 _TriangleMotion(right,right,False), sweep=False)
            w, d2 = pair.sample(F(0))
            if best is None or d2 < best:
                best, nearest = d2, []
            if d2 == best:
                wa, wb = w.first_weights, w.second_weights
                if reverse:
                    w = replace(w, normal=_xyz(-np.array(w.normal)), first_point=w.second_point,
                                second_point=w.first_point, first_weights=wb, second_weights=wa)
                nearest.append(MeshMeshSupport(_owners(a,w.first_weights), _owners(b,w.second_weights), w))
        if not nearest:
            raise SolverFailure("mesh_mesh_nearest_unresolved")
        return _representative(nearest)

    @staticmethod
    def _full_area(f):
        i, j = f.face.cols
        return _area2(tuple((p[i], p[j]) for p in f.xyz))

    def _separating(self, t, directions):
        pts = [np.array(list(m.points.values())) + t*m.step for m in (self.a, self.b)]
        for direction in directions:
            n = np.array(direction, dtype=object)
            p, q = (v @ n for v in pts)
            if max(p) <= min(q) or max(q) <= min(p):
                return True
        return False

    def sample(self, t, *, first_from_clear=False):
        a, b = self.a, self.b
        surfaces = {}
        areas_a, areas_b = [F(0)]*len(a.facets), [F(0)]*len(b.facets)
        opposed = set()
        crossing = False
        count = 0
        for i, j in self.facet_pairs(t, t):
            count += 1
            f, g = a.facets[i], b.facets[j]
            points, area, transverse = _intersection(f, g, t)
            crossing |= transverse
            if area:
                areas_a[i] += area
                areas_b[j] += area
                if f.face.normal @ g.face.normal > 0:
                    crossing = True  # Same outward side on an open common face patch.
                else:
                    opposed.add(tuple(f.face.normal))
            for p in points:
                wa, wb = f.weights(p - t*f.step), g.weights(p - t*g.step)
                if wa is None or wb is None:
                    raise SolverFailure("mesh_mesh_surface_address_unresolved")
                surfaces[wa, wb] = None
        deepest, inside = self._intrusions(t)
        direction = _unit(np.array(next(iter(opposed)), dtype=object))[0] if len(opposed) == 1 else None
        normal = None if direction is None else _xyz(direction)
        boundary = tuple(sorted((_support(a,b,u,v,t,normal=normal) for u,v in surfaces), key=_order))
        distance = 0.0
        if boundary:
            if all(area == self._full_area(f) for m, areas in ((a,areas_a),(b,areas_b))
                   for f, area in zip(m.facets, areas, strict=True)):
                relation = "coincident"
            elif deepest or crossing:
                relation = "boundary_overlap"
            elif first_from_clear or self._separating(t, [*opposed, (1,0,0), (0,1,0), (0,0,1)]):
                relation = "touching"
            else:
                relation = "boundary_intersection_unclassified"
            witness = _representative(deepest or boundary)
        else:
            nearest = self._nearest(t)
            distance = nearest.gap_m
            if any(inside):
                relation = "second_contains_first" if inside[0] else "first_contains_second"
                witness = _representative(deepest)
            else:
                relation, witness = "separated", nearest
        intrusion = -deepest[0].witness.gap_m if deepest else None
        return MeshMeshPose(relation, distance, intrusion, deepest, boundary, witness, count)

    @staticmethod
    def _facet_window(a, b):
        # Full-dimensional triangle Minkowski differences use the two face axes
        # and all edge-cross-edge axes already supplied by HullPairQuery.
        window = HullPairQuery(a, b)._window(None)
        if window is None or any(_cross3(a.face.normal, b.face.normal)):
            return window
        # Parallel triangles can meet coplanarly. Their in-plane separating
        # directions are absent from the usual 3D face/cross-product axis set.
        lo, hi = window
        directions = {direction_key(_cross3(f.face.normal, edge))
                      for f in (a, b) for edge in f.edge_rays}
        for d in sorted(directions):
            axis = _Axis(d, a, b).interval()
            if axis is None:
                return None
            lo, hi = max(lo, axis[0]), min(hi, axis[1])
            if lo > hi:
                return None
        return lo, hi

    def result(self, moving):
        start = self.sample(F(0))
        if not moving:
            trace = MeshMeshTrace(start, start, None, None)
            status = {"separated": "separated", "touching": "touching",
                      "boundary_intersection_unclassified": "unresolved"}.get(start.relation, "overlap")
            return status, start.relation, start.witness, None, trace

        first = F(0) if start.relation != "separated" else None
        if first is None:
            for i, j in self.facet_pairs(F(0), F(1)):
                window = self._facet_window(self.a.facets[i], self.b.facets[j])
                if window is not None and (first is None or window[0] < first):
                    first = window[0]
        end = self.sample(F(1), first_from_clear=first == 1)
        if first is None:
            if end.relation != "separated":
                raise SolverFailure("mesh_mesh_time_interval_inconsistent")
            sweep = CollisionSweep(start.witness.gap_m, end.witness.gap_m, None, None, None)
            return "separated", "mesh_mesh_swept_separated", end.witness, sweep, MeshMeshTrace(start,end,None,None)

        if first == 0:
            entry = start
        elif first == 1:
            entry = end
        else:
            entry = self.sample(first, first_from_clear=True)
        if first and (entry.relation != "touching" or not entry.boundary):
            raise SolverFailure("mesh_mesh_first_boundary_unresolved")
        bracket = (floor(first*_GRID)/_GRID, ceil(first*_GRID)/_GRID)
        sample_time = float(first)
        stored_time = F(sample_time)
        if stored_time == first:
            sampled_witness = entry.witness
        elif stored_time == 0:
            sampled_witness = start.witness
        elif stored_time == 1:
            sampled_witness = end.witness
        else:
            # The exact entry is kept separately. The ordinary sweep witness
            # must describe the actual binary time published in witness_time.
            sampled_witness = self.sample(stored_time).witness
        trace = MeshMeshTrace(start,end,(first.numerator,first.denominator),entry)
        sweep = CollisionSweep(start.witness.gap_m,end.witness.gap_m,bracket,sample_time,sampled_witness)
        return "hit", "mesh_mesh_swept_boundary", end.witness, sweep, trace
