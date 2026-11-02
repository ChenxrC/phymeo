"""Independent geometry references for the six final M0 fixture families."""

from math import sqrt

import numpy as np
from solid_case_oracles import plane_weights


def ball_gap(shape, center, radius):
    if shape["type"] == "plane":
        return float(np.dot(shape["normal"], center) - shape["offset"] - radius)
    return float(np.linalg.norm(center - shape["center"]) - shape["radius"] - radius)


def ball_event(shape, ball):
    start, end = (np.array(ball[k], dtype=float) for k in ("center_start", "center_end"))
    r = ball["radius"]
    gaps = [ball_gap(shape, center, r) for center in (start, end)]
    time = None
    if gaps[0] <= 0:
        time = 0.0
    elif shape["type"] == "plane":
        if gaps[1] <= 0:
            time = gaps[0] / (gaps[0] - gaps[1])
    else:
        delta, offset = end - start, start - shape["center"]
        a, b = delta @ delta, offset @ delta
        c = offset @ offset - (r + shape["radius"]) ** 2
        disc = b * b - a * c
        if a > 0 and disc >= 0:
            roots = [t for t in ((-b - sqrt(disc)) / a, (-b + sqrt(disc)) / a) if 0 <= t <= 1]
            time = min(roots, default=None)
    point = None
    if time is not None:
        center = start + time * (end - start)
        if shape["type"] == "plane":
            n = np.array(shape["normal"])
            assert abs(np.linalg.norm(n) - 1) < 1e-12
        else:
            n = center - shape["center"]
            n /= np.linalg.norm(n)
        point = (center - r * n).tolist()
    return dict(first_t=time, endpoint_gap_m=gaps, point_m=point)


def aligned_capsules(a, b):
    x, y = (np.array(shape["axis"], dtype=float) for shape in (a, b))
    # Each segment independently spans at most one coordinate interval.
    assert all(np.count_nonzero(row[1] - row[0]) <= 1 for row in (x, y))
    delta = np.maximum(0, y.min(axis=0) - x.max(axis=0)) - np.maximum(
        0, x.min(axis=0) - y.max(axis=0)
    )
    distance = np.linalg.norm(delta)
    return dict(
        gap_m=float(distance - a["radius"] - b["radius"]),
        normal=(delta / distance).tolist() if distance else None,
    )


def rounded_point(polygon, radius, point):
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    assert low[2] == high[2]
    closest = np.clip(point, low, high)
    delta = point - closest
    distance = np.linalg.norm(delta)
    outside = np.count_nonzero(delta[:2])
    if outside:
        feature = "corner" if outside == 2 else "rim"
    else:
        feature = "front" if delta[2] > 0 else "back" if delta[2] < 0 else "middle"
    return dict(
        feature=feature,
        gap_m=float(distance - radius),
        middle_m=closest.tolist(),
        normal=(delta / distance).tolist() if distance else None,
    )


def plate_path(polygon, radius, body):
    if body["type"] == "cloth_triangle":
        start, end = (np.asarray(body[k]) for k in ("vertices_start", "vertices_end"))
        assert np.all(start[:, 2] == start[0, 2]) and np.all(end[:, 2] == end[0, 2])
        np.testing.assert_array_equal(start[:, :2], end[:, :2])
        # All four middle-polygon corners project strictly inside the moving triangle.
        area_axes = (start[1:, :2] - start[0, :2]).T
        bary = np.linalg.solve(area_axes, (polygon[:, :2] - start[0, :2]).T).T
        assert np.all(bary > 0) and np.all(bary.sum(axis=1) < 1)
        z0, z1 = start[0, 2], end[0, 2]
        span = np.ptp(polygon, axis=0)
        area = float(span[0] * span[1])
        lateral = 0.0
    else:
        start, end = (np.array(body[k]) for k in ("start", "end"))
        np.testing.assert_array_equal(start[:2], end[:2])
        center = np.clip(start[:2], polygon[:, :2].min(axis=0), polygon[:, :2].max(axis=0))
        lateral = np.linalg.norm(start[:2] - center)
        z0, z1, area = start[2], end[2], 0.0
    mid_z = polygon[0, 2]
    if lateral > radius:
        return dict(first_t=None, gap_lower_m=float(lateral - radius))
    top = mid_z + sqrt(radius**2 - lateral**2)
    assert z0 > top >= z1
    return dict(first_t=float((z0 - top) / (z0 - z1)), contact_area_m2=area)


def first_vertex_on_face(polygon, target_vertices, start, end):
    """A single moving tip reaches a supporting face before any other vertex."""
    cloud = np.vstack((start, end))
    lower = np.maximum(
        target_vertices.min(axis=0) - cloud.max(axis=0),
        cloud.min(axis=0) - target_vertices.max(axis=0),
    )
    if max(lower) > 0:
        return dict(first_t=None, gap_lower_m=float(max(lower)))
    normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
    normal /= np.linalg.norm(normal)
    initial, final = ((p - polygon[0]) @ normal for p in (start, end))
    assert min(initial) > 0
    roots = [(h / (h - k), i) for i, (h, k) in enumerate(zip(initial, final)) if k <= 0]
    if not roots:
        return dict(first_t=None, gap_lower_m=float(min(min(initial), min(final))))
    time, index = min(roots)
    at = start + time * (end - start)
    assert min(np.delete((at - polygon[0]) @ normal, index)) > 0
    point = at[index]
    sides = [
        np.cross(q - p, point - p) @ normal for p, q in zip(polygon, np.roll(polygon, -1, axis=0))
    ]
    assert min(sides) > 0
    return dict(first_t=float(time), vertex=index, point_m=point.tolist(), normal=normal.tolist())


def cloth_tet_patch(tet0, tet1, start, end, rho):
    """Parallel material layers against a possibly moving exterior x-face."""
    assert rho >= 0
    combined_tet, combined_sheet = np.vstack((tet0, tet1)), np.vstack((start, end))
    sideways = np.maximum(
        combined_tet[:, 1:].min(axis=0) - combined_sheet[:, 1:].max(axis=0),
        combined_sheet[:, 1:].min(axis=0) - combined_tet[:, 1:].max(axis=0),
    )
    if max(sideways) > 0:
        return dict(first_t=None, gap_lower_m=float(max(sideways)))
    for tet, sheet in ((tet0, start), (tet1, end)):
        assert np.all(tet[[0, 2, 3], 0] == tet[0, 0]) and tet[1, 0] > tet[0, 0]
        assert np.all(sheet[:, 0] == sheet[0, 0])
    np.testing.assert_array_equal(start[:, 1:], end[:, 1:])
    g0 = tet0[0, 0] - start[0, 0] - rho
    g1 = tet1[0, 0] - end[0, 0] - rho
    assert g0 > 0 > g1
    time = g0 / (g0 - g1)
    tet = tet0 + time * (tet1 - tet0)
    points = start + time * (end - start)
    points[:, 0] += rho
    weights = plane_weights(tet, points)
    assert np.max(np.abs(weights[:, 1])) < 1e-12 and np.min(weights[:, [0, 2, 3]]) > 0
    area = np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])) / 2
    return dict(
        first_t=float(time),
        area_m2=float(area),
        points_m=points.tolist(),
        tet_weights=weights.tolist(),
    )


def corner_clearance_reference(tet0, tet1, start, end, face, rho):
    """Prove a positive-clearance outcome for one corner and one fixture face."""
    (missing,) = set(range(4)) - set(face)
    spans = []
    for tet in (tet0, tet1):
        axes = tet[1:] - tet[0]
        lengths = np.diag(axes)
        np.testing.assert_array_equal(axes, np.diag(lengths))
        assert np.all(lengths > 0)
        spans.append(lengths)
    coordinate = [
        (p - tet[0]) / lengths
        for p, tet, lengths in zip((start, end), (tet0, tet1), spans, strict=True)
    ]
    lo, hi = np.min(coordinate, axis=0), np.max(coordinate, axis=0)
    max_inverse = 1 / np.min(spans, axis=0)
    # Each coordinate is affine divided by positive affine; its extrema
    # are at the endpoints. The sum bound also covers the moving oblique face.
    if missing == 0:
        low, high, grad_upper = 1 - hi.sum(), 1 - lo.sum(), np.linalg.norm(max_inverse)
    else:
        axis = missing - 1
        low, high, grad_upper = lo[axis], hi[axis], max_inverse[axis]
    plane_lower = max(low, -high, 0.0) / grad_upper
    if plane_lower > rho:
        return dict(status="certified", plane_distance_lower_m=float(plane_lower))

    if missing == 1:
        h0, h1 = start[0] - tet0[0, 0], end[0] - tet1[0, 0]
        assert h0 < 0 < h1
        u = -h0 / (h1 - h0)
        tet, point = (1 - u) * tet0 + u * tet1, (1 - u) * start + u * end
        kind = "axis_center_crossing"
    else:
        # Check the actual director segment at the final pose, in addition to
        # the spherical corner proxy used by the component sweep.
        left, right = end - np.array([rho, 0, 0]), end + np.array([rho, 0, 0])
        h0, h1 = plane_weights(tet1, np.stack((left, right)))[:, missing]
        assert h0 * h1 < 0
        fraction = h0 / (h0 - h1)
        tet, point, u = tet1, (1 - fraction) * left + fraction * right, 1.0
        assert np.linalg.norm(point - end) < rho
        kind = "terminal_layer_intersection"
    weights = plane_weights(tet, point[None])[0]
    assert abs(weights[missing]) < 1e-12 and min(np.delete(weights, missing)) > 0
    return dict(
        status="violation",
        proof=kind,
        t=float(u),
        point_m=point.tolist(),
        face_weights=weights.tolist(),
    )
