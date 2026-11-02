"""Independent first-contact references for the three declared M0 motion families.

These formulas use catalog inputs and elementary geometry only. Each checks
the assumptions under which it can prove a hit or clearance on the whole path.
"""

from dataclasses import dataclass
from math import asin, atan2, cos, hypot, pi, radians, sin, sqrt

import numpy as np


@dataclass(frozen=True)
class Reference:
    time: float | None
    point: tuple[float, ...] = ()
    weights: tuple[float, ...] = ()
    separation_lower_m: float | None = None


def rotating_capsule(a, b):
    axis = np.asarray(a["local_axis"])
    rotation = a["rotation"]
    assert rotation["world_axis"] == [0, 0, 1] and rotation["law"] == "linear_angle"
    assert np.all(axis[:, 1:] == 0) and axis[0, 0] < 0 < axis[1, 0]
    x, y, z = b["center"]
    reach = hypot(x, y)
    assert 0 < reach < min(-axis[0, 0], axis[1, 0])
    bearing = atan2(y, x)
    start, end = (radians(rotation[key]) for key in ("angle_start_deg", "angle_end_deg"))
    assert bearing - pi / 2 < start < bearing < end < bearing + pi / 2
    radius = a["radius"] + b["radius"]
    if abs(z) > radius:
        return Reference(None, separation_lower_m=abs(z) - radius)
    assert 0 < radius < reach
    # The axial projection stays in the open segment for the entire arc.
    entry = bearing - asin(sqrt(radius**2 - z**2) / reach)
    assert start < entry < end
    direction = np.array([cos(entry), sin(entry), 0.0])
    center = np.asarray(b["center"])
    on_axis = (center @ direction) * direction
    point = on_axis + a["radius"] * (center - on_axis) / radius
    return Reference((entry - start) / (end - start), tuple(point))


def sphere_face(a, b):
    tri = np.asarray(a["vertices"], dtype=float)
    start, end = np.asarray(b["center_start"]), np.asarray(b["center_end"])
    radius = b["radius"]
    # A component-wise swept box separation is enough for the lateral miss.
    box_gap = np.maximum(
        np.min(tri, axis=0) - np.maximum(start, end), np.minimum(start, end) - np.max(tri, axis=0)
    )
    if np.max(box_gap) > radius:
        return Reference(None, separation_lower_m=float(np.max(box_gap) - radius))
    e1, e2 = tri[1:] - tri[0]
    normal = np.cross(e1, e2)
    normal /= np.linalg.norm(normal)
    if (start - tri[0]) @ normal < 0:
        normal = -normal
    h0, h1 = (float((c - tri[0]) @ normal) for c in (start, end))
    assert h0 > radius and h1 < -radius
    time = (h0 - radius) / (h0 - h1)
    point = start + time * (end - start) - radius * normal
    # Signed subtriangle areas give original material weights, independent
    # of the production closest-point implementation.
    double_area = np.cross(e1, e2)
    area2 = double_area @ double_area
    weights = [
        np.cross(tri[(i + 1) % 3] - point, tri[(i + 2) % 3] - point) @ double_area / area2
        for i in range(3)
    ]
    assert min(weights) > 0
    return Reference(time, tuple(point), tuple(weights))


def crossing_edges(a, b):
    shapes = [np.asarray([s["vertices_start"], s["vertices_end"]], dtype=float) for s in (a, b)]
    for path in shapes:
        assert np.array_equal(path[0, :2, :2], path[1, :2, :2])
        assert np.array_equal(path[:, 0, 2], path[:, 1, 2])
    lower, upper = sorted(shapes, key=lambda path: path[0, 0, 2])
    assert np.all(lower[:, 2, 2] < lower[:, 0, 2])
    assert np.all(upper[:, 2, 2] > upper[:, 0, 2])
    dh0, dh1 = upper[:, 0, 2] - lower[:, 0, 2]
    assert dh0 > 0
    if dh1 > 0:
        return Reference(None, separation_lower_m=float(min(dh0, dh1)))
    time = dh0 / (dh0 - dh1)
    first, second = (path[0, :2, :2] for path in shapes)
    u, v = first[1] - first[0], second[1] - second[0]
    params = np.linalg.solve(np.column_stack((u, -v)), second[0] - first[0])
    assert np.all((0 < params) & (params < 1))
    xy = first[0] + params[0] * u
    z = (1 - time) * lower[0, 0, 2] + time * lower[1, 0, 2]
    return Reference(float(time), (*xy, z), tuple(params))
