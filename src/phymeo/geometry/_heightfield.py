"""Rigid sphere queries against a finite, two-sided triangle height graph."""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..constraints.collision_shapes import CollisionMotion, CollisionShape
from ..core.capabilities import SolverFailure
from ._round_sweep import _GRID, _translation, _unit, _vector, _xyz
from ._sphere_triangle_sweep import _SphereFace
from ._sweep_polynomial import add, first_entry, mul
from ._triangle_pair_sweep import _cross3
from .collision_witness import CollisionSweep, CollisionWitness


@dataclass(frozen=True)
class HeightfieldFace:
    """Material address at one pose. Triangle 0 is SW–SE–NE and 1 is SW–NE–NW."""

    cell: tuple[int, int]
    triangle: int
    vertex_ids: tuple[int, int, int]
    barycentric: tuple[float, float, float]


@dataclass(frozen=True)
class HeightfieldEvent:
    """The contact time is enclosed; both material points are sampled at witness_time."""

    interval: tuple[float, float]
    witness_time: float
    witness: CollisionWitness
    face: HeightfieldFace


@dataclass(frozen=True)
class HeightfieldTrace(HeightfieldFace):
    """Endpoint material address plus the first and last contact on a queried path."""

    first_contact: HeightfieldEvent | None = None
    last_contact: HeightfieldEvent | None = None


def _reversed_polynomial(poly):
    """Compose an exact feature condition with 1-t for the last feasible time."""
    reverse = ()
    for coefficient in reversed(poly):
        reverse = add((coefficient,), mul(reverse, (1, -1)))
    return reverse


class _TerrainSphere:
    def __init__(self, terrain, sphere, field_path, ball_path, field_first):
        self.terrain, self.sphere = terrain, sphere
        self.ball_path, self.field_first = ball_path, field_first
        self.start = _vector(ball_path.start[0])
        self.step = _translation(ball_path)
        self.positions = field_path.start
        self.columns = len(terrain.heightfield.heights_yx[0])
        vertices = self.positions.reshape(-1, self.columns, 3)
        corners = (vertices[:-1, :-1], vertices[:-1, 1:], vertices[1:, 1:], vertices[1:, :-1])
        self.lower = np.nextafter(np.minimum.reduce(corners), -np.inf)
        self.upper = np.nextafter(np.maximum.reduce(corners), np.inf)

    def faces(self, cells):
        for r, c in cells:
            row, col = int(r), int(c)
            sw = row * self.columns + col
            se, nw, ne = sw + 1, sw + self.columns, sw + self.columns + 1
            for index, slots in enumerate(((sw, se, ne), (sw, ne, nw))):
                ids = tuple(self.terrain.vertex_ids[k] for k in slots)
                face = CollisionShape(self.terrain.key, "triangle", ids)
                triangle = self.positions[list(slots)]
                query = _SphereFace(face, self.sphere, CollisionMotion(triangle), self.ball_path)
                yield row, col, index, ids, triangle, query

    def sample(self, time):
        # A vertex bounds the distance from above. A cell AABB bounds it from
        # below. Widen the screening for rounding the exact sampled center.
        exact_center = self.start + time * self.step
        center = np.asarray(exact_center, dtype=float)
        bound = np.hypot.reduce(self.positions - center, axis=-1).min()
        bound += 2 * np.hypot.reduce(np.spacing(abs(center))) + 8 * abs(np.spacing(bound))
        outside = np.maximum(np.maximum(self.lower - center, center - self.upper), 0.0)
        cells = np.argwhere(np.hypot.reduce(outside, axis=-1) <= bound)
        best = None
        for r, c, index, ids, triangle, query in self.faces(cells):
            witness, signed = query.sample(time)
            # All comparisons use exact squared distances on the original path.
            if best is None or signed < best[0]:
                weights = dict(witness.first_weights)
                face = HeightfieldFace((r, c), index, ids, tuple(weights[k] for k in ids))
                best = signed, witness, face, triangle
        if best is None:
            raise SolverFailure("heightfield_no_candidate_cell")
        signed, local, face, triangle = best
        normal, ball_point = local.normal, local.second_point
        if normal is None:
            p, q, s = map(_vector, triangle)
            direction, _ = _unit(_cross3(q - p, s - p))
            if direction is None:
                raise SolverFailure("heightfield_degenerate_world_face")
            normal = _xyz(direction)
            ball_point = _xyz(center - self.sphere.radius * direction)
        if self.field_first:
            points = local.first_point, ball_point
            weights = local.first_weights, local.second_weights
        else:
            points = ball_point, local.first_point
            weights = local.second_weights, local.first_weights
            normal = tuple(-v for v in normal)
        return CollisionWitness(local.gap_m, normal, *points, *weights, "host-heightfield"), face, signed

    def event_bounds(self):
        end = self.start + self.step
        radius = Fraction(self.sphere.radius)
        # The swept AABB is only a rejection test. Entry and exit use the exact
        # polynomial conditions for every Voronoi feature of each candidate face.
        low = np.nextafter(np.asarray(np.minimum(self.start, end) - radius, dtype=float), -np.inf)
        high = np.nextafter(np.asarray(np.maximum(self.start, end) + radius, dtype=float), np.inf)
        cells = np.argwhere(np.all((low <= self.upper) & (self.lower <= high), axis=-1))
        first = last = None
        for *_, query in self.faces(cells):
            for feature in query.features:
                enter = feature.entry(None)
                if enter is None:
                    continue
                reverse = first_entry(tuple(_reversed_polynomial(p) for p in feature.conditions), None)
                if reverse is None:
                    raise SolverFailure("heightfield_time_reversal_unresolved")
                leave = (_GRID - reverse[1], _GRID - reverse[0])
                first = enter if first is None else min(first, enter)
                last = leave if last is None else max(last, leave)
        return first, last


def heightfield_sphere_contact(a, b, ma, mb):
    """Return status, reason, endpoint witness/address and an optional continuous query."""
    field_first = a.kind == "heightfield"
    terrain, sphere, field_path, ball_path = (a, b, ma, mb) if field_first else (b, a, mb, ma)
    if field_path.moving or (ball_path.turn is not None and ball_path.turn.angle != 0):
        return "unsupported", "heightfield_swept_pair", None, None, None
    try:
        query = _TerrainSphere(terrain, sphere, field_path, ball_path, field_first)
        end, face, signed = query.sample(Fraction(int(ball_path.moving)))
        face_fields = (face.cell, face.triangle, face.vertex_ids, face.barycentric)
        if not ball_path.moving:
            status = "overlap" if signed < 0 else "touching" if signed == 0 else "separated"
            return status, "narrow_phase", end, HeightfieldTrace(*face_fields), None

        start, start_face, _ = query.sample(Fraction(0))
        first, last = query.event_bounds()
        if first is None:
            sweep = CollisionSweep(start.gap_m, end.gap_m, None, None, None)
            return "separated", "swept_separated", end, HeightfieldTrace(*face_fields), sweep

        def event(bounds):
            lo, hi = bounds
            time = Fraction(lo + hi, 2 * _GRID)
            if time == 0:
                witness, address = start, start_face
            elif time == 1:
                witness, address = end, face
            else:
                witness, address, _ = query.sample(time)
            return HeightfieldEvent((lo / _GRID, hi / _GRID), float(time), witness, address)

        entry, leave = event(first), event(last)
        sweep = CollisionSweep(start.gap_m, end.gap_m, entry.interval, entry.witness_time, entry.witness)
        return "hit", "first_contact", end, HeightfieldTrace(*face_fields, entry, leave), sweep
    except (SolverFailure, OverflowError, FloatingPointError) as error:
        return "unresolved", str(error), None, None, None
