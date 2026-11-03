"""Independent face bounds from analytic coordinate extrema, and physical edge/face crossings."""

import math

import numpy as np


def harmonics(path):
    """x(t) = origin + drift*t + cosine*cos(a*t) + sine*sin(a*t)."""
    start = np.asarray(path.start)
    if path.turn is None:
        drift = np.zeros_like(start) if path.end is None else np.asarray(path.end) - start
        return start, drift, np.zeros_like(start), np.zeros_like(start), 0.0
    turn = path.turn
    pivot = np.array(turn.pivot_start)
    axis = np.asarray(turn.axis) / np.linalg.norm(turn.axis)
    local = start - pivot
    along = np.outer(local @ axis, axis)
    return (
        pivot + along,
        np.broadcast_to(np.array(turn.pivot_end) - pivot, start.shape),
        local - along,
        np.cross(axis, local),
        turn.angle,
    )


def position(terms, t):
    origin, drift, c, s, omega = terms
    return origin + drift * t + c * math.cos(omega * t) + s * math.sin(omega * t)


def exact_face_boxes(terms, indices, t0, t1):
    """Include both endpoints and every stationary point of each coordinate."""
    origin, drift, c, s, omega = terms
    a, b = position(terms, t0), position(terms, t1)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    for v, k in np.ndindex(origin.shape):
        amplitude = math.hypot(c[v, k], s[v, k])
        if not omega or not amplitude:
            continue
        threshold = -drift[v, k] / (omega * amplitude)
        if abs(threshold) > 1:
            continue
        low, high = sorted((omega * t0, omega * t1))
        phi = math.atan2(c[v, k], s[v, k])
        for base in (math.acos(threshold) - phi, -math.acos(threshold) - phi):
            for j in range(math.ceil((low-base)/math.tau), math.floor((high-base)/math.tau)+1):
                theta = base + j * math.tau
                component = (
                    origin[v, k] + drift[v, k] * theta / omega
                    + c[v, k] * math.cos(theta) + s[v, k] * math.sin(theta)
                )
                lo[v, k] = min(lo[v, k], component)
                hi[v, k] = max(hi[v, k], component)
    return lo[indices].min(axis=1), hi[indices].max(axis=1)


def face_indices(shape):
    row = {node: i for i, node in enumerate(shape.vertex_ids)}
    return np.array([[row[node] for node in face] for face in shape.triangles])


def all_box_pairs(first, second, margin=0.0):
    """Cartesian product, independent of a tree or the trajectory-bound implementation."""
    lo1, hi1 = first
    lo2, hi2 = second
    overlap = np.all(hi1[:, None] + margin >= lo2[None, :], axis=2)
    overlap &= np.all(hi2[None, :] + margin >= lo1[:, None], axis=2)
    return zip(*np.nonzero(overlap), strict=True)


def _edge_face(source, target):
    axis0, axis1 = target[1] - target[0], target[2] - target[0]
    n = np.cross(axis0, axis1)
    h = (source - target[0]) @ n
    normal_order = int(np.argmax(np.abs(n)))
    project = [i for i in range(3) if i != normal_order]
    basis = np.column_stack((axis0[project], axis1[project]))
    for i, j in ((0, 1), (1, 2), (2, 0)):
        if h[i] * h[j] >= 0:
            continue
        fraction = h[i] / (h[i] - h[j])
        point = source[i] + fraction * (source[j] - source[i])
        u, v = np.linalg.solve(basis, (point - target[0])[project])
        weights = np.array([1 - u - v, u, v])
        if np.min(weights) > 1e-9:  # Strict interior crossings have unambiguous ownership.
            edge = np.zeros(3)
            edge[i], edge[j] = 1 - fraction, fraction
            return edge, weights, point
    return None


def transverse_crossings(shapes, terms, time):
    """All face pairs with a strict edge-to-face crossing at the stated pose."""
    first, second = shapes
    a, b = (position(p, time)[face_indices(s)] for s, p in zip(shapes, terms, strict=True))
    crossings = {}
    for i, j in all_box_pairs((a.min(1), a.max(1)), (b.min(1), b.max(1))):
        hit = _edge_face(a[i], b[j])
        if hit is None:
            other = _edge_face(b[j], a[i])
            if other:
                wb, wa, p = other
                hit = wa, wb, p
        if hit:
            crossings[first.triangles[i], second.triangles[j]] = hit
    return crossings
