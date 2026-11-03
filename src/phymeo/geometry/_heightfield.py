"""Discrete sphere distance to a finite, two-sided rigid height graph."""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..constraints.collision_shapes import CollisionMotion, CollisionShape
from ..core.capabilities import SolverFailure
from ._round_sweep import _unit, _vector, _xyz
from ._sphere_triangle_sweep import _SphereFace
from ._triangle_pair_sweep import _cross3
from .collision_witness import CollisionWitness


@dataclass(frozen=True)
class HeightfieldTrace:
    """Material address on the terrain; independent of which shape owns side zero.

    cell is (row, column). The SW–NE diagonal joins (r,c) and (r+1,c+1).
    Triangle 0 runs SW, SE, NE and triangle 1 runs SW, NE, NW.
    Barycentrics follow vertex_ids in that order.
    """

    cell: tuple[int, int]
    triangle: int
    vertex_ids: tuple[int, int, int]
    barycentric: tuple[float, float, float]


def _candidate_cells(grid, center):
    # A grid vertex gives an upper bound. A real world-space cell AABB gives a
    # lower bound even for tilted terrain, positive gaps and boundary features.
    vertices = grid.world_vertices().reshape(len(grid.heights_yx), -1, 3)
    corners = (vertices[:-1, :-1], vertices[:-1, 1:], vertices[1:, 1:], vertices[1:, :-1])
    upper = np.hypot.reduce(vertices - center, axis=-1).min()
    lo = np.nextafter(np.minimum.reduce(corners), -np.inf)
    hi = np.nextafter(np.maximum.reduce(corners), np.inf)
    outside = np.maximum(np.maximum(lo - center, center - hi), 0.0)
    lower = np.hypot.reduce(outside, axis=-1)
    # Widen the floating screening comparison by the rounding of the norm.
    upper = upper + 8 * abs(np.spacing(upper))
    return np.argwhere(lower <= upper)


def heightfield_sphere_contact(a, b, ma, mb):
    """Return status, reason, two physical points and the winning terrain face."""
    if ma.moving or mb.moving:
        return "unsupported", "heightfield_swept_pair", None, None

    field_first = a.kind == "heightfield"
    terrain, sphere, field_path, ball_path = (a, b, ma, mb) if field_first else (b, a, mb, ma)
    grid = terrain.heightfield
    columns = len(grid.heights_yx[0])
    center = ball_path.start[0]
    positions = field_path.start
    best = None
    try:
        for r, c in _candidate_cells(grid, center):
            row, col = int(r), int(c)
            sw = row * columns + col
            se, nw, ne = sw + 1, sw + columns, sw + columns + 1
            for index, slots in enumerate(((sw, se, ne), (sw, ne, nw))):
                ids = tuple(terrain.vertex_ids[k] for k in slots)
                face = CollisionShape(terrain.key, "triangle", ids)
                triangle = positions[list(slots)]
                query = _SphereFace(face, sphere, CollisionMotion(triangle), ball_path)
                witness, signed = query.sample(Fraction(0))
                # Exact squared distances. Row-major ordering resolves exact ties.
                if best is None or signed < best[0]:
                    weights = dict(witness.first_weights)
                    chart = HeightfieldTrace((row, col), index, ids, tuple(weights[k] for k in ids))
                    best = signed, witness, chart, triangle
        if best is None:
            raise SolverFailure("heightfield_no_candidate_cell")
        signed, local, chart, triangle = best
        normal, ball_point = local.normal, local.second_point
        if normal is None:
            # Center on the sheet. Pick the positive oriented normal of the same
            # material face that won the deterministic tie, including on a ridge.
            p, q, s = map(_vector, triangle)
            direction, _ = _unit(_cross3(q - p, s - p))
            if direction is None:
                raise SolverFailure("heightfield_degenerate_world_face")
            normal = _xyz(direction)
            ball_point = _xyz(center - sphere.radius * direction)
        if field_first:
            points = local.first_point, ball_point
            weights = local.first_weights, local.second_weights
        else:
            points = ball_point, local.first_point
            weights = local.second_weights, local.first_weights
            normal = tuple(-v for v in normal)
        pair = CollisionWitness(local.gap_m, normal, *points, *weights, "host-heightfield")
        status = "overlap" if signed < 0 else "touching" if signed == 0 else "separated"
        return status, "narrow_phase", pair, chart
    except (SolverFailure, OverflowError, FloatingPointError) as error:
        return "unresolved", str(error), None, None
