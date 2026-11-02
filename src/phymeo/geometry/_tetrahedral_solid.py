"""Filled tetrahedral intersections and exterior events for distinct material bodies."""

from collections import defaultdict
from dataclasses import replace
from fractions import Fraction as F
from itertools import combinations, product

import numpy as np

from ..constraints.collision_shapes import CollisionShape
from ._prism_clip import hull
from ._round_sweep import _GRID
from ._tetrahedral_surface import _clip, _full, _ray, _TriangleMotion
from ._triangle_pair_sweep import _cross3, _FacePath, _TrianglePair
from .collision_witness import (
    CollisionSweep,
    CollisionWitness,
    MaterialContactPoint,
    TetrahedralBodyPair,
    TetrahedralBoundaryContact,
    TetrahedralBoundaryTrace,
    TetrahedralOccupancy,
    TetrahedralSolidResult,
)

# Outward winding for a positive linear tetrahedron, indexed by opposite node.
_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
_ZERO_VOLUME = (0, 1)


def _ratio(x):
    return x.numerator, x.denominator


def _bracket(bounds):
    bounds = [b for b in bounds if b is not None]
    return (min(b[0] for b in bounds), min(b[1] for b in bounds)) if bounds else None


def _relation(volume, va, vb, touched):
    if not volume:
        return "touching" if touched else "separated"
    if volume == va == vb:
        return "coincident"
    if volume == vb:
        return "first_contains_second"
    if volume == va:
        return "second_contains_first"
    return "overlap"


def _exact_points(x):
    return np.array([[F(v) for v in row] for row in x], dtype=object)


class _CellPose:
    def __init__(self, points):
        self.p = points
        u, v, w = points[1:] - points[0]
        self.duals = np.array([_cross3(v, w), _cross3(w, u), _cross3(u, v)])
        self.det = u @ self.duals[0]
        self.volume = self.det / 6

    def weights(self, point):
        b = (self.duals @ (point - self.p[0])) / self.det
        return (1 - sum(b), *b)


def _clipped_faces(source, weights_in_target):
    """All boundary polygons of one cell clipped by the four half-spaces of the other."""
    for face in _FACES:
        idx = list(face)
        vertices = source.p[idx]
        # The last two barycentric coordinates are an injective planar chart.
        polygon = list(np.eye(3, dtype=int).astype(object))
        for plane in weights_in_target[idx].T:
            polygon = _clip(polygon, plane)
            if not polygon:
                break
        if polygon:
            yield tuple(
                tuple(np.array((1 - u - v, u, v), dtype=object) @ vertices)
                for u, v in hull([(p[1], p[2]) for p in polygon])
            )


def _intersection(a, b, owners, ids):
    aw = np.array([b.weights(p) for p in a.p], dtype=object)
    bw = np.array([a.weights(p) for p in b.p], dtype=object)
    if min(aw.flat) >= 0:
        volume, point = a.volume, a.p.mean(axis=0)
    elif min(bw.flat) >= 0:
        volume, point = b.volume, b.p.mean(axis=0)
    else:
        faces = {}
        vertices = set()
        for src, weights in ((a, aw), (b, bw)):
            for polygon in _clipped_faces(src, weights):
                vertices.update(polygon)
                if len(polygon) >= 3:
                    # Coincident supporting planes can produce the same polygon twice.
                    faces[tuple(sorted(polygon))] = polygon
        if not vertices:
            return TetrahedralOccupancy("separated", _ZERO_VOLUME, None, None)
        point = np.array(sorted(vertices), dtype=object).mean(axis=0)
        volume = F(0)
        for polygon in faces.values():
            offsets = np.array(polygon, dtype=object) - point
            for i in range(1, len(offsets) - 1):
                volume += abs(offsets[0] @ _cross3(offsets[i], offsets[i + 1])) / 6
    materials = [
        tuple(sorted(zip(nodes, map(float, cell.weights(point)), strict=True)))
        for cell, nodes in zip((a, b), ids, strict=True)
    ]
    witness = MaterialContactPoint(tuple(map(float, point)), *materials)
    return TetrahedralOccupancy(
        _relation(volume, a.volume, b.volume, True), _ratio(volume), owners, witness
    )


class _CellPath:
    def __init__(self, shape, path, exposed):
        self.key, self.ids = shape.key, shape.vertex_ids
        self.start = _exact_points(path.start)
        end = self.start if path.end is None else _exact_points(path.end)
        self.step = end - self.start
        self.faces = []
        exterior = {frozenset(f) for f in exposed}
        for ix in _FACES:
            ids = tuple(self.ids[i] for i in ix)
            indices = list(ix)
            # Faces preserve exact relative motion even with shared translations.
            motion = _TriangleMotion(self.start[indices], end[indices], path.moving)
            self.faces.append(
                (CollisionShape(self.key, "triangle", ids), motion, frozenset(ids) in exterior)
            )

    def pose(self, t):
        return _CellPose(self.start + t * self.step)


def volume_mesh_issue(shapes, paths):
    """Check the disjoint material interiors needed to add cell intersection volumes.

    Opposite incidences of a shared face and positive cell determinants separate
    the two interiors throughout the step. For other neighboring cells, admit
    their common material edge or vertex, and check for remote coincidences.
    """
    boxes = {}
    cells = {}
    for shape in shapes:
        path = paths[shape.key]
        end = path.start if path.end is None else path.end
        boxes[shape.key] = (
            np.minimum(path.start.min(axis=0), end.min(axis=0)),
            np.maximum(path.start.max(axis=0), end.max(axis=0)),
        )
    for a, b in combinations(shapes, 2):
        if len(set(a.vertex_ids) & set(b.vertex_ids)) == 3:
            continue
        lo_a, hi_a = boxes[a.key]
        lo_b, hi_b = boxes[b.key]
        if any(hi_a < lo_b) or any(hi_b < lo_a):
            continue
        for shape in (a, b):
            if shape.key not in cells:
                cells[shape.key] = _CellPath(shape, paths[shape.key], ())
        ca, cb = cells[a.key], cells[b.key]
        start = _intersection(ca.pose(F(0)), cb.pose(F(0)), (a.key, b.key), (ca.ids, cb.ids))
        if start.volume_ratio[0]:
            return "invalid_geometry", "overlapping_cells_in_material_body"
        for (sa, xa, _), (sb, xb, _) in product(ca.faces, cb.faces):
            leaf = _TrianglePair(sa, sb, xa, xb)
            if start.witness is not None and leaf.sample(F(0))[0].intersection_points:
                return "unresolved", "nonshared_cell_boundary_contact"
            if any(f.entry(None) is not None for f in leaf.features):
                return "unresolved", "nonshared_cell_boundary_contact"
    return None


def _boundary_witness(w, ids_a, ids_b):
    points = tuple(
        replace(
            p,
            first_weights=_full(ids_a, p.first_weights),
            second_weights=_full(ids_b, p.second_weights),
        )
        for p in w.intersection_points
    )
    return replace(
        w,
        first_weights=_full(ids_a, w.first_weights),
        second_weights=_full(ids_b, w.second_weights),
        intersection_points=points,
    )


class TetrahedralCellQuery:
    """Both the closed material cells and the subset of their faces exposed on the two bodies."""

    def __init__(self, a, b, ma, mb, exposed):
        self.cells = (_CellPath(a, ma, exposed[a.key]), _CellPath(b, mb, exposed[b.key]))
        self.owners = a.key, b.key
        self.ids = a.vertex_ids, b.vertex_ids
        self.parts = []
        self.exposed = []
        all_features, surface_features = {}, set()
        for (sa, xa, ea), (sb, xb, eb) in product(self.cells[0].faces, self.cells[1].faces):
            leaf = _TrianglePair(sa, sb, xa, xb)
            self.parts.append(leaf)
            all_features.update(dict.fromkeys(leaf.features))
            if ea and eb:
                self.exposed.append(leaf)
                surface_features.update(leaf.features)
        self.features = tuple(all_features)
        self.surface_features = surface_features

    def _occupancy(self, t):
        a, b = (p.pose(t) for p in self.cells)
        return _intersection(a, b, self.owners, self.ids)

    def _nearest(self, t, exterior):
        leaves = self.exposed if exterior else self.parts
        if not leaves:
            return None, None
        best = None
        for leaf in leaves:
            w, d2 = leaf.sample(t)
            if best is None or d2 < best[0]:
                best = d2, leaf, w
        d2, leaf, w = best
        return TetrahedralBoundaryContact(
            leaf.a.ids, leaf.b.ids, _boundary_witness(w, *self.ids)
        ), d2

    def _filled_witness(self, t, occupancy, nearest=None):
        if occupancy.witness is not None:
            p = occupancy.witness
            return CollisionWitness(
                0.0,
                None,
                p.point,
                p.point,
                p.first_weights,
                p.second_weights,
                source="host-tetrahedral-occupancy",
            )
        if nearest is None or len(self.exposed) != len(self.parts):
            nearest, _ = self._nearest(t, False)
        return nearest.witness

    def _exact_surface_event(self, bracket, at_sample):
        lo, hi = map(F, bracket)
        if lo == hi:
            return _ratio(lo), at_sample
        # With a constant supporting face direction, all opposing vertices have
        # affine heights. Their first zero from a strict outside half-space
        # proves there was no earlier solid contact, without using a fixture time.
        for c, other in (self.cells, self.cells[::-1]):
            for shape, motion, is_outer in c.faces:
                if not is_outer:
                    continue
                face = _FacePath(shape, motion)
                n = _ray(face)
                if n is None:
                    continue
                heights = (other.start - face.start[0]) @ n
                if min(heights) <= 0:
                    continue
                rates = (other.step - face.step[0]) @ n
                roots = [
                    -height / rate for height, rate in zip(heights, rates, strict=True) if rate < 0
                ]
                if roots and lo <= (t := min(roots)) <= hi:
                    contact, d2 = self._nearest(t, True)
                    if d2 == 0:
                        return _ratio(t), contact
        return None, None

    def result(self, moving, seeds):
        zero, one = F(0), F(1)
        start = self._occupancy(zero)
        end = self._occupancy(one) if moving else start
        surf0, d0 = self._nearest(zero, True)
        surf1, _ = self._nearest(one, True) if moving else (surf0, d0)
        endpoint = self._filled_witness(one, end, surf1)
        entries = {}
        if moving and (start.relation == "separated" or (surf0 is not None and d0 != 0)):
            for feature, seed in zip(self.features, seeds, strict=True):
                ticks = feature.entry(seed)
                if ticks is not None:
                    entries[feature] = ticks[0] / _GRID, ticks[1] / _GRID
        boundary = None
        if surf0 is not None:
            first = (
                (0.0, 0.0) if d0 == 0 else _bracket(entries.get(f) for f in self.surface_features)
            )
            t = contact = exact_t = exact_w = None
            if first is not None:
                t = sum(first) / 2
                contact = surf0 if t == 0 else self._nearest(F(t), True)[0]
                exact_t, exact_w = self._exact_surface_event(first, contact)
            boundary = TetrahedralBoundaryTrace(surf0, surf1, first, t, contact, exact_t, exact_w)
        solid = TetrahedralSolidResult(start, end, boundary)
        if not moving:
            status = {"separated": "separated", "touching": "touching"}.get(end.relation, "overlap")
            return status, end.relation, endpoint, None, solid
        initial = self._filled_witness(zero, start, surf0)
        first = (0.0, 0.0) if start.relation != "separated" else _bracket(entries.values())
        t = witness = None
        if first is not None:
            t = sum(first) / 2
            witness = initial if t == 0 else self._filled_witness(F(t), self._occupancy(F(t)))
        sweep = CollisionSweep(initial.gap_m, endpoint.gap_m, first, t, witness)
        return ("hit" if first else "separated"), "closed_cell_sweep", endpoint, sweep, solid


def _body_sample(cells, volumes, phase):
    samples = [getattr(r.solid, phase) for r in cells if r.solid is not None]
    volume = sum((F(*s.volume_ratio) for s in samples), F(0))
    witness = max(
        (s for s in samples if s.witness is not None),
        key=lambda s: F(*s.volume_ratio),
        default=None,
    )
    relation = _relation(volume, *volumes, witness is not None)
    return TetrahedralOccupancy(
        relation,
        _ratio(volume),
        witness.owners if witness else None,
        witness.witness if witness else None,
    )


def solid_body_pairs(model, paths, pairs):
    """Sum disjoint cell interiors only for complete pairs of conforming volume meshes."""
    shapes = {s.key: s for s in model.shapes if s.kind == "tetrahedron"}
    groups = defaultdict(list)
    for row in pairs:
        a, b = row.first, row.second
        if a in shapes and b in shapes and a.world == b.world and a.body != b.body:
            groups[a.world, a.body, b.body].append(row)
    volumes = {}

    def body_volume(world, body):
        key = world, body
        if key not in volumes:
            v0, v1 = F(0), F(0)
            for k in shapes:
                if (k.world, k.body) == key:
                    path = paths[k]
                    start = _CellPose(_exact_points(path.start)).volume
                    v0 += start
                    v1 += _CellPose(_exact_points(path.end)).volume if path.moving else start
            volumes[key] = (v0, v1)
        return volumes[key]

    results = []
    for (world, a, b), rows in sorted(groups.items()):
        statuses = {r.status for r in rows}
        failure = next(
            (
                s
                for s in ("invalid_geometry", "unresolved", "unsupported", "filtered")
                if s in statuses
            ),
            None,
        )
        if failure:
            results.append(
                TetrahedralBodyPair(world, a, b, "partial" if failure == "filtered" else failure)
            )
            continue
        v0, v1 = zip(body_volume(world, a), body_volume(world, b), strict=True)
        start = _body_sample(rows, v0, "start")
        end = _body_sample(rows, v1, "end")
        first = (
            (0.0, 0.0)
            if start.relation != "separated"
            else _bracket(r.sweep.first_t for r in rows if r.sweep)
        )
        surface = _bracket(r.solid.boundary.first_t for r in rows if r.solid and r.solid.boundary)
        results.append(TetrahedralBodyPair(world, a, b, "ok", start, end, first, surface))
    return tuple(results)
