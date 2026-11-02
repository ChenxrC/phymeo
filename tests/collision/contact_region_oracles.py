"""Independent references for the registered contact-region geometries.

The box reference uses closed interval intersections. The other fixtures use
plane cuts and barycentric coordinates. None calls a Phymeo query.
"""

from fractions import Fraction as F
from itertools import product
from math import hypot

import numpy as np


def box_center(shape, time):
    if "center" in shape:
        return tuple(map(F, shape["center"]))
    return tuple(
        F(a) + time * (F(b) - F(a))
        for a, b in zip(shape["center_start"], shape["center_end"], strict=True)
    )


def box_slide(a, b, time):
    ca, cb = box_center(a, time), box_center(b, time)
    ha, hb = tuple(map(F, a["half"])), tuple(map(F, b["half"]))
    lower = [max(c - h, d - k) for c, h, d, k in zip(ca, ha, cb, hb, strict=True)]
    upper = [min(c + h, d + k) for c, h, d, k in zip(ca, ha, cb, hb, strict=True)]
    width = [hi - lo for lo, hi in zip(lower, upper, strict=True)]
    assert cb[2] - hb[2] >= ca[2] + ha[2]  # the registered boxes touch or lie above
    xy = set(product(*zip(lower[:2], upper[:2], strict=True))) if min(width[:2]) >= 0 else set()
    area = max(width[0], 0) * max(width[1], 0)
    gap = hypot(*(float(min(w, 0)) for w in width))
    dimension = -1 if gap else sum(w > 0 for w in width)
    return xy, area, gap, dimension


def last_face_time(a, b):
    cx = F(a["center"][0]) + F(a["half"][0]) + F(b["half"][0])
    x0, x1 = F(b["center_start"][0]), F(b["center_end"][0])
    assert x0 < cx < x1
    return (cx - x0) / (x1 - x0)


def skew_edge_time(a, b):
    # Upper A support is a single x-edge, lower B support a single y-edge.
    # Symmetric transverse half-axes put their intersection on the z-axis.
    axes_a, axes_b = a["half_axes"], b["half_axes"]
    assert axes_a[0] == [1, 0, 0] and axes_b[0] == [0, 1, 0]
    for axes in (axes_a, axes_b):
        assert axes[1][2] == axes[2][2] > 0
        assert all(u == -v for u, v in zip(axes[1][:2], axes[2][:2], strict=True))
    az = F(a["center"][2]) + sum(abs(F(row[2])) for row in axes_a)
    bz = sum(abs(F(row[2])) for row in axes_b)
    z0, z1 = F(b["center_start"][2]), F(b["center_end"][2])
    assert z0 - bz > az > z1 - bz
    return (z0 - bz - az) / (z0 - z1), az


def barycentric(triangle, point):
    # Resolve two in-plane directions, then verify the third world coordinate.
    u, v = triangle[1:] - triangle[0]
    keep = np.arange(3) != np.argmax(np.abs(np.cross(u, v)))
    s, t = np.linalg.solve(np.column_stack((u[keep], v[keep])), (point - triangle[0])[keep])
    weights = np.array([1 - s - t, s, t])
    np.testing.assert_allclose(weights @ triangle, point, atol=1e-12, rtol=0)
    return weights


def capsule_face(tri, axis, radius):
    # A positive separating coordinate proves the whole capsule misses.
    box_sep = np.maximum(tri.min(axis=0) - axis.max(axis=0), axis.min(axis=0) - tri.max(axis=0))
    if max(box_sep) > radius:
        return {"kind": "separated", "gap_lower_m": float(max(box_sep) - radius)}
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    normal = normal / np.linalg.norm(normal)
    heights = (axis - tri[0]) @ normal
    if heights[0] * heights[1] < 0:
        s = heights[0] / (heights[0] - heights[1])
        kind = "axis_pierces"
    else:
        s = float(np.argmin(np.abs(heights)))
        assert abs(heights[int(s)]) == radius
        kind = "cap_tangent"
    point_on_axis = axis[0] + s * (axis[1] - axis[0])
    point = point_on_axis - ((point_on_axis - tri[0]) @ normal) * normal
    weights = barycentric(tri, point)
    assert min(weights) > 0
    return {
        "kind": kind,
        "axis_parameter": float(s),
        "point_m": point.tolist(),
        "face_weights": weights.tolist(),
    }


def shared_faces(vertices, faces):
    common = sorted(set(map(int, faces[0])).intersection(map(int, faces[1])))
    x, y = vertices[faces[0]], vertices[faces[1]]
    normal = np.cross(x[1] - x[0], x[2] - x[0])
    second_normal = np.cross(y[1] - y[0], y[2] - y[0])
    assert np.linalg.norm(normal) > 0 and np.linalg.norm(second_normal) > 0
    if len(common) == 2:
        # Noncoplanar face planes intersect on the shared edge's line.
        assert np.linalg.norm(np.cross(normal, second_normal)) > 0
        return {"kind": "shared_edge_only", "common_ids": common}
    assert len(common) == 1
    origin = vertices[common[0]]
    other_a, other_b = [vertices[[i for i in face if i not in common]] for face in faces]
    heights = (other_b - origin) @ normal
    if np.all(heights == 0):
        # Strictly opposite open half-planes outside their common vertex.
        direction = other_a.mean(axis=0) - other_b.mean(axis=0)
        assert min((other_a - origin) @ direction) > 0 > max((other_b - origin) @ direction)
        return {"kind": "shared_vertex_only", "common_ids": common}
    assert heights[0] * heights[1] < 0
    s = heights[0] / (heights[0] - heights[1])
    endpoint = other_b[0] + s * (other_b[1] - other_b[0])
    point = (origin + endpoint) / 2
    weights = np.array([barycentric(tri, point) for tri in (x, y)])
    assert np.all(weights > 0) and np.linalg.norm(point - origin) > 0
    return {
        "kind": "distinct_material_intersection",
        "common_ids": common,
        "point_m": point.tolist(),
        "weights": weights.tolist(),
    }
