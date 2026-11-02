"""Independent references for the registered U-solid and tetrahedral cases.

These references operate on the fixture's boxes, half-spaces and motion.
They do not use the triangulation, closest-feature code or material inverse.
"""

from fractions import Fraction as F
from itertools import combinations
from math import hypot, prod

import numpy as np


def u_solid_path(mesh, hull):
    ring = mesh["ring_xy"]
    xl, il, ir, xr = sorted({F(p[0]) for p in ring})
    y0, y1, y2 = sorted({F(p[1]) for p in ring})
    z0, z1 = map(F, mesh["z"])
    # The bridge and two arms have disjoint interiors.
    parts = (
        ((xl, y0, z0), (xr, y1, z1)),
        ((xl, y1, z0), (il, y2, z1)),
        ((ir, y1, z0), (xr, y2, z1)),
    )
    v = [[F(a) for a in row] for row in hull["vertices"]]
    start, end = (list(map(F, hull[k])) for k in ("translation_start", "translation_end"))
    assert start[1:] == end[1:] == [0, 0]
    assert all(y1 < y < y2 and z0 < z < z1 for _, y, z in v)
    dx = end[0] - start[0]
    assert dx != 0
    xs = [row[0] + start[0] for row in v]
    assert il < min(xs) <= max(xs) < ir
    wall = ir if dx > 0 else il
    first = min((wall - x) / dx for x in xs)
    last = max((wall - x) / dx for x in xs)
    times = [("start", F(0)), ("end", F(1))]
    if first <= 1:
        assert first < last < 1
        times[1:1] = [("touch", first), ("cut", (first + last) / 2)]
    samples = []
    for phase, t in times:
        points = [[p[0] + start[0] + t * dx, p[1], p[2]] for p in v]
        lo, hi = [min(x) for x in zip(*points)], [max(x) for x in zip(*points)]
        margins = [
            min(
                *(edge - bound for edge, bound in zip(lo, low)),
                *(bound - edge for bound, edge in zip(high, hi)),
            )
            for low, high in parts
        ]
        box_distance = min(
            hypot(
                *(
                    max(low_box - high_hull, low_hull - high_box, 0)
                    for low_box, high_box, low_hull, high_hull in zip(low, high, lo, hi)
                )
            )
            for low, high in parts
        )
        if max(margins) > 0:
            kind, lower = "contained", float(max(margins))
        elif box_distance > 0:
            kind, lower = "separated", box_distance
        else:
            assert lo[0] <= wall <= hi[0]
            kind = "touch" if wall in (lo[0], hi[0]) else "cut"
            lower = 0.0
        samples.append({"phase": phase, "t": float(t), "kind": kind, "boundary_lower_m": lower})
    return {
        "first_t": float(first) if first <= 1 else None,
        "wall_x": float(wall),
        "volume_m3": float(sum(prod(b - a for a, b in zip(low, high)) for low, high in parts)),
        "samples": samples,
    }


def supporting_faces(vertices, cells):
    """Independent boundary oracle only for this convex two-cell bipyramid."""
    result = {}
    for owner, tet in enumerate(cells):
        for face in combinations(map(int, tet), 3):
            tri = vertices[list(face)]
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            heights = (vertices - tri[0]) @ normal
            if np.min(heights) >= 0 or np.max(heights) <= 0:
                key = tuple(sorted(face))
                assert key not in result
                result[key] = owner
    return result


def sphere_bipyramid(sphere):
    start, end = (np.array(sphere[key]) for key in ("center_start", "center_end"))
    radius = sphere["radius"]
    x, y, z = start
    assert np.array_equal(start[1:], end[1:]) and y > 0 and y + abs(z) < 1
    if np.array_equal(start, end):
        # Interior half-spaces x>=0, y>=0 and x+y+/-z<=1.
        clearance = min(x, y, (1 - x - y - abs(z)) / np.sqrt(3)) - radius
        assert clearance > 0 and z == 0
        return {"kind": "ball_contained", "boundary_gap_lower_m": float(clearance)}
    assert x < -radius and end[0] > x
    if end[0] < -radius:
        return {"kind": "separated", "boundary_gap_lower_m": float(-end[0] - radius)}
    time = (-radius - x) / (end[0] - x)
    assert z != 0 and 0 < time <= 1
    return {
        "kind": "external_touch",
        "first_t": float(time),
        "point_m": [0.0, float(y), float(z)],
        "owner": int(z < 0),
        "node_weights": [1 - y - abs(z), 0.0, y, max(z, 0.0), max(-z, 0.0)],
    }


def signed_volume(tet):
    a, b, c, d = tet
    return float(np.cross(b - a, c - a) @ (d - a) / 6)


def plane_weights(tet, points):
    """Four signed half-space heights relative to their opposite vertices."""
    columns = []
    for i, opposite in enumerate(tet):
        a, b, c = np.delete(tet, i, axis=0)
        normal = np.cross(b - a, c - a)
        columns.append((points - a) @ normal / ((opposite - a) @ normal))
    return np.column_stack(columns)


def static_tets(a, b):
    volumes = [signed_volume(t) for t in (a, b)]
    if min(volumes) <= 0:
        return {"kind": "invalid_geometry", "volumes_m3": volumes}
    for name, outer, inner in (("A", a, b), ("B", b, a)):
        weights = plane_weights(outer, inner)
        if np.all(weights > 0):
            # Convexity extends the four strict vertex tests to the entire inner solid.
            return {
                "kind": "contained_overlap",
                "outer": name,
                "overlap_volume_m3": signed_volume(inner),
                "outer_weights": weights.tolist(),
            }
    gap = np.maximum(a.min(axis=0) - b.max(axis=0), b.min(axis=0) - a.max(axis=0))
    assert max(gap) > 0
    return {"kind": "separated", "separating_gap_lower_m": float(max(gap))}


def deforming_touch(a, b):
    # Each cell has just one varying vertex. Its volume is affine on the step.
    for path in (a, b):
        assert np.count_nonzero(np.any(path[1] != path[0], axis=1)) == 1
        assert min(signed_volume(tet) for tet in path) > 0
    face = a[0, [0, 2, 3]]
    np.testing.assert_array_equal(a[1, [0, 2, 3]], face)
    normal = np.cross(face[1] - face[0], face[2] - face[0])
    if (a[0, 1] - face[0]) @ normal < 0:
        normal = -normal
    assert (a[1, 1] - face[0]) @ normal > 0
    heights = (b - face[0]) @ normal
    assert max(heights[0]) < 0 < heights[1, 0] and max(heights[1, 1:]) < 0
    time = -heights[0, 0] / (heights[1, 0] - heights[0, 0])
    point = (1 - time) * b[0, 0] + time * b[1, 0]
    a_hit = (1 - time) * a[0] + time * a[1]
    weights = plane_weights(a_hit, point[None])[0]
    assert abs(weights[1]) < 1e-12 and min(weights[[0, 2, 3]]) > 0
    # One strict witness in both interiors at the final pose proves positive overlap volume.
    witness = (15 * b[1, 0] + b[1].mean(axis=0)) / 16
    interior = [plane_weights(t, witness[None])[0] for t in (a[1], b[1])]
    assert all(min(w) > 0 for w in interior)
    return {
        "kind": "external_touch",
        "first_t": float(time),
        "point_m": point.tolist(),
        "a_weights": weights.tolist(),
        "end_interior_point_m": witness.tolist(),
    }
