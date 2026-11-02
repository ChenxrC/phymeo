"""Exterior incidence and material paths for affine, positively oriented volume cells."""

from collections import defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction as F
from math import isqrt
from types import MappingProxyType

import numpy as np

from ..constraints.collision_shapes import CollisionShape
from ..constraints.conforming_shell import _cyclic, _parity
from ..core.capabilities import SolverFailure
from ._prism_clip import hull
from ._round_sweep import sweep_result
from ._sphere_triangle_sweep import _SphereFace
from ._sweep_polynomial import Roots, affine_vectors, cross, difference, dot
from ._triangle_pair_sweep import _cross3, _FacePath, _Pose, _TrianglePair
from .collision_witness import (
    CollisionSweep,
    CollisionWitness,
    MaterialContactPoint,
    TetrahedralFacetResult,
)


def exterior_faces(shapes):
    """Cancel opposite incidence only inside one world and material body."""
    facets = defaultdict(list)
    result = {}
    cells = set()
    for s in shapes:
        if s.kind != "tetrahedron":
            continue
        body = (s.key.world, s.key.body)
        cell = (*body, tuple(sorted(s.vertex_ids)))
        if cell in cells:
            raise ValueError("duplicate tetrahedron in the same material body")
        cells.add(cell)
        result[s.key] = []
        for i in range(4):
            face = s.vertex_ids[:i] + s.vertex_ids[i + 1 :]
            if i % 2:
                face = (face[0], face[2], face[1])
            facets[(*body, tuple(sorted(face)))].append((s.key, _cyclic(face)))
    for entries in facets.values():
        if len(entries) == 1:
            key, face = entries[0]
            result[key].append(face)
        elif len(entries) != 2 or _parity(entries[0][1]) == _parity(entries[1][1]):
            raise ValueError("tetrahedral face must have one owner or two opposite owners")
    return MappingProxyType({k: tuple(sorted(v)) for k, v in result.items()})


def positive_cell(path):
    start = np.array([[F(x) for x in row] for row in path.start])
    end = start if path.end is None else np.array([[F(x) for x in row] for row in path.end])
    a, b, c, d = affine_vectors(start, end - start)
    volume6 = dot(difference(b, a), cross(difference(c, a), difference(d, a)))
    return bool(volume6 and volume6[0] > 0 and not Roots(volume6).count(F(0), F(1)))


@dataclass(frozen=True)
class _TriangleMotion:
    # Lifted coordinates can be exact rationals between two stored FP64 numbers.
    start: np.ndarray
    end: np.ndarray
    moving: bool


def _ray(face):
    """Return an initial raw normal if its direction is constant on the valid path."""
    if not face.valid:
        raise SolverFailure("triangle_degenerate_on_path")
    n = np.array([p[0] if p else 0 for p in face.normal], dtype=object)
    for degree in (1, 2):
        coef = [p[degree] if len(p) > degree else 0 for p in face.normal]
        if any(_cross3(n, coef)):
            return None
    return n


def _rational_normal(raw):
    if raw is None:
        return None
    norm2 = raw @ raw
    length = F(isqrt(norm2.numerator), isqrt(norm2.denominator))
    return raw / length if length * length == norm2 else None


def _full(ids, weights):
    by_id = dict(weights)
    return tuple((node, by_id.get(node, 0.0)) for node in sorted(ids))


def _mapped(w, cell_ids, tet_first, material):
    def sides(a, b):
        addr, height = material(b)
        full = _full(cell_ids, a) if a else ()
        return (full, addr, None, height) if tet_first else (addr, full, height, None)

    def point(p):
        a, b, ha, hb = sides(p.first_weights, p.second_weights)
        return MaterialContactPoint(p.point, a, b, ha, hb)

    a, b, ha, hb = sides(w.first_weights, w.second_weights)
    return replace(
        w,
        first_weights=a,
        second_weights=b,
        first_height_m=ha,
        second_height_m=hb,
        normal=w.normal if tet_first or w.normal is None else tuple(-v for v in w.normal),
        first_point=w.first_point if tet_first else w.second_point,
        second_point=w.second_point if tet_first else w.first_point,
        intersection_points=tuple(map(point, w.intersection_points)),
        shared_material_points=tuple(map(point, w.shared_material_points)),
    )


def _clip(weights, plane):
    """Clip an ordered polygon in the tet-face barycentric plane by a closed half-space."""
    out = []
    for i, b in enumerate(weights):
        a = weights[i - 1]
        da, db = a @ plane, b @ plane
        if da * db < 0:
            out.append(a + da / (da - db) * (b - a))
        if db >= 0:
            out.append(b)
    return out


class _ClothLayer:
    """Full triangular material prism with the affine motion of its middle nodes."""

    def __init__(self, shape, mid, normal):
        self.shape, self.mid, self.normal = shape, mid, normal
        self.rho = F(shape.radius)
        x = np.vstack((mid.start - self.rho * normal, mid.start + self.rho * normal))
        motions = np.vstack((mid.step, mid.step))
        pieces = [(0, 2, 1), (3, 4, 5)]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            pieces.extend(((i, j, j + 3), (i, j + 3, i + 3)))
        self.skin = []
        for ids in pieces:
            idx = list(ids)
            self.skin.append(
                (
                    CollisionShape(shape.key, "triangle", ids),
                    _TriangleMotion(x[idx], (x + motions)[idx], bool(np.any(motions[idx]))),
                )
            )

    def skin_material(self, weights):
        vals = dict(weights)
        return (
            tuple(
                sorted(
                    (node, vals.get(i, 0.0) + vals.get(i + 3, 0.0))
                    for i, node in enumerate(self.shape.vertex_ids)
                )
            ),
            float(self.rho) * sum((1 if i >= 3 else -1) * w for i, w in weights),
        )

    def section(self, tet, time):
        """Intersect the entire finite layer and one face, including strict interior points."""
        target = _Pose(tet, time)
        mid = _Pose(self.mid, time)
        coords = []
        for p in target.p:
            delta = p - mid.p[0]
            u = (_cross3(delta, mid.v) @ mid.n) / mid.den
            v = (_cross3(mid.u, delta) @ mid.n) / mid.den
            coords.append((1 - u - v, u, v, delta @ self.normal))
        coords = np.array(coords, dtype=object)
        planes = [*coords[:, :3].T, self.rho - coords[:, 3], self.rho + coords[:, 3]]
        poly = list(np.eye(3, dtype=int).astype(object))
        for plane in planes:
            poly = _clip(poly, plane)
            if not poly:
                return None
        # Projecting face barycentrics to their last two coordinates is injective.
        points = []
        for u, v in hull([(p[1], p[2]) for p in poly]):
            wa = np.array((1 - u - v, u, v), dtype=object)
            wb = wa @ coords
            points.append(
                MaterialContactPoint(
                    tuple(map(float, wa @ target.p)),
                    tuple(sorted(zip(tet.ids, map(float, wa), strict=True))),
                    tuple(sorted(zip(self.mid.ids, map(float, wb[:3]), strict=True))),
                    None,
                    float(wb[3]),
                )
            )
        p = points[0]
        return CollisionWitness(
            0.0,
            None,
            p.point,
            p.point,
            p.first_weights,
            p.second_weights,
            "host-affine-cloth-layer-section",
            intersection_points=tuple(points),
            second_height_m=p.second_height_m,
        )


class ExteriorFacetQuery:
    def __init__(self, cell, other, cell_path, other_path, face_ids, tet_first):
        self.cell, self.other, self.face_ids, self.tet_first = cell, other, face_ids, tet_first
        face = CollisionShape(cell.key, "triangle", face_ids)
        ix = [cell.vertex_ids.index(node) for node in face_ids]
        end = cell_path.start if cell_path.end is None else cell_path.end
        mf = _TriangleMotion(cell_path.start[ix], end[ix], cell_path.moving)
        self.layer = self.mid = self.raw = None
        if other.kind == "sphere":
            self.leaves = (_SphereFace(face, other, mf, other_path),)
        else:
            self.mid = _FacePath(other, other_path)
            self.raw = _ray(self.mid)
            if other.radius:
                normal = _rational_normal(self.raw)
                if normal is None:
                    raise SolverFailure("unsupported_cloth_layer_director")
                self.layer = _ClothLayer(other, self.mid, normal)
                self.leaves = tuple(_TrianglePair(face, s, mf, m) for s, m in self.layer.skin)
            else:
                self.leaves = (_TrianglePair(face, other, mf, other_path),)
        self.features = tuple(f for leaf in self.leaves for f in leaf.features)

    def sample(self, t):
        cell = self.cell.vertex_ids
        if self.layer is not None:
            section = self.layer.section(self.leaves[0].a, t)
            if section is not None:
                # Section points already carry their middle-surface address and height.
                def address(p):
                    wa = _full(cell, p.first_weights)
                    if self.tet_first:
                        return replace(p, first_weights=wa)
                    return MaterialContactPoint(
                        p.point, p.second_weights, wa, p.second_height_m, None
                    )

                points = tuple(map(address, section.intersection_points))
                p = points[0]
                return replace(
                    section,
                    first_weights=p.first_weights,
                    second_weights=p.second_weights,
                    first_height_m=p.first_height_m,
                    second_height_m=p.second_height_m,
                    intersection_points=points,
                ), F(0)
            material = self.layer.skin_material
        else:
            height = None if self.mid is None else 0.0

            def material(weights):
                return weights, height

        w, signed = min((leaf.sample(t) for leaf in self.leaves), key=lambda pair: pair[1])
        return _mapped(w, cell, self.tet_first, material), signed

    def exact_event(self, bracket):
        if bracket is None:
            return None, None
        lo, hi = map(F, bracket)
        if lo == hi:
            witness, signed = self.sample(lo)
            return ((lo.numerator, lo.denominator), witness) if signed <= 0 else (None, None)
        if self.raw is None:
            return None, None
        # A whole tet face can meet a cloth cap between representable times.
        # All three heights must reach that cap simultaneously from its outside.
        tet = self.leaves[0].a
        direction = self.raw if self.layer is None else self.layer.normal
        heights = (tet.start - self.mid.start[0]) @ direction
        rates = (tet.step - self.mid.step[0]) @ direction
        rho = F(self.other.radius)
        for sign in (-1, 1):
            before = sign * heights - rho
            if min(before) <= 0 or any(r == 0 for r in rates):
                continue
            times = {-c / (sign * r) for c, r in zip(before, rates, strict=True)}
            if len(times) == 1:
                t = times.pop()
                if lo <= t <= hi:
                    witness, signed = self.sample(t)
                    if signed == 0:
                        return (t.numerator, t.denominator), witness
        return None, None


class TetrahedralSurfaceQuery:
    """Reduce owned boundary facets to one pair result while retaining all later events."""

    def __init__(self, cell, other, cell_path, other_path, exposed, tet_first, valid):
        self.facets = ()
        self.features = ()
        self.failure = None
        if not valid:
            self.failure = ("unresolved", "tetrahedron_nonpositive_on_path")
        elif cell.radius or (other_path.turn is not None and other_path.turn.angle != 0):
            self.failure = ("unsupported", "tetrahedral_surface_offset_or_turn")
        elif other.key.body == cell.key.body:
            self.failure = ("unsupported", "same_body_sheet_volume")
        else:
            try:
                self.facets = tuple(
                    ExteriorFacetQuery(cell, other, cell_path, other_path, ids, tet_first)
                    for ids in exposed
                )
                self.features = tuple(f for part in self.facets for f in part.features)
            except SolverFailure as error:
                status = (
                    "unsupported"
                    if str(error) == "unsupported_cloth_layer_director"
                    else "unresolved"
                )
                self.failure = (status, str(error))

    def result(self, moving, seeds):
        if self.failure is not None:
            return (*self.failure, None, None, ())
        if not self.facets:
            return "surface_clear", "no_exterior_facets", None, None, ()
        facets = []
        static_samples = []
        offset = 0
        for part in self.facets:
            if moving:
                stop = offset + len(part.features)
                witness, sweep = sweep_result(part, seeds[offset:stop])
                offset = stop
            else:
                witness, signed = part.sample(F(0))
                static_samples.append((signed, witness))
                sweep = None
            ratio, event = part.exact_event(sweep.first_t if sweep is not None else None)
            facets.append(TetrahedralFacetResult(part.face_ids, witness, sweep, ratio, event))
        if not moving:
            signed, end = min(static_samples, key=lambda item: item[0])
            if signed > 0:
                return "surface_clear", "exterior_boundary_clear", end, None, tuple(facets)
            status = "overlap" if signed < 0 else "touching"
            return status, "exterior_contact", end, None, tuple(facets)
        end = min((p.witness for p in facets), key=lambda w: w.gap_m)
        times = [f.sweep.first_t for f in facets if f.sweep.first_t is not None]
        start_gap = min(f.sweep.start_gap_m for f in facets)
        if times:
            lo, hi = min(b[0] for b in times), min(b[1] for b in times)
            t = (lo + hi) / 2
            w, _ = min((f.sample(F(t)) for f in self.facets), key=lambda pair: pair[1])
            sweep = CollisionSweep(start_gap, end.gap_m, (lo, hi), t, w)
            return "hit", "exterior_first_contact", end, sweep, tuple(facets)
        sweep = CollisionSweep(start_gap, end.gap_m, None, None, None)
        return "surface_clear", "exterior_boundary_clear", end, sweep, tuple(facets)
