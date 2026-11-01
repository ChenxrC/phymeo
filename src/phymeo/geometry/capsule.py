"""Closest points of closed segments, including zero-length sphere centerlines."""

import numpy as np


def closest_segment_points(a, b, c, d):
    points = [np.asarray(p, dtype=np.float64) for p in (a, b, c, d)]
    if any(p.shape != (3,) or not np.isfinite(p).all() for p in points):
        raise ValueError("segment endpoints must be finite 3-vectors")
    a, b, c, d = points
    u, v, w = b - a, d - c, a - c
    aa, bb, cc = u @ u, u @ v, v @ v
    dd, ee = u @ w, v @ w
    candidates = []
    for p in (a, b):
        t = np.clip((p - c) @ v / cc, 0.0, 1.0) if cc else 0.0
        candidates.append((p, c + t * v))
    for p in (c, d):
        s = np.clip((p - a) @ u / aa, 0.0, 1.0) if aa else 0.0
        candidates.append((a + s * u, p))
    cross = np.cross(u, v)
    det = cross @ cross
    if det > 0:
        s, t = (bb * ee - cc * dd) / det, (aa * ee - bb * dd) / det
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append((a + s * u, c + t * v))
    p, q = min(candidates, key=lambda pair: np.sum((pair[0] - pair[1]) ** 2))
    return p.copy(), q.copy()
