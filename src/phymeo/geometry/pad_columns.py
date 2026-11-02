"""Finite material thickness segments against a rigid-local convex rounded pad."""

from dataclasses import dataclass

import numpy as np

from ..core.capabilities import SolverFailure


@dataclass(frozen=True)
class PadColumnSites:
    fraction: np.ndarray
    points_m: np.ndarray
    normal: np.ndarray
    gap_m: np.ndarray
    endpoints_m: np.ndarray


def _dot(a, b):
    return np.sum(a * b, axis=-1)


def rounded_pad_columns(front, front_normal, radius, middle, half):
    """Closest material coordinate on each full column against an authored pad.

    The validated convex polygon is the physical front. Its middle polygon
    lies one radius behind it. Material fractions run from -1 to +1.
    """
    start, end = middle - half, middle + half
    u = end - start
    uu = _dot(u, u)
    mid = front - radius * front_normal
    edges = np.roll(mid, -1, axis=0) - mid
    winding = np.cross(edges[0], edges[1])
    winding = winding / np.linalg.norm(winding)
    size = len(middle)
    best = np.full(size, np.inf)
    chosen = np.full(size, 0.5)
    direction = np.zeros_like(middle)

    def offer(t, other, valid=True):
        t = np.broadcast_to(t, (size,))
        delta = start + t[:, None] * u - other
        d2 = _dot(delta, delta)
        better = valid & ((d2 < best) | ((d2 == best) & (np.abs(t - 0.5) < np.abs(chosen - 0.5))))
        best[better] = d2[better]
        chosen[better] = t[better]
        direction[better] = delta[better]

    # Project the whole segment onto the polygon plane. Clip its parameter
    # against all polygon half-planes before minimizing its plane distance.
    lo, hi = np.zeros(size), np.ones(size)
    for vertex, edge in zip(mid, edges, strict=True):
        inward = np.cross(winding, edge)
        a = (start - vertex) @ inward
        b = u @ inward
        cut = np.divide(-a, b, out=np.zeros(size), where=b != 0)
        lo = np.where(b > 0, np.maximum(lo, cut), lo)
        hi = np.where(b < 0, np.minimum(hi, cut), hi)
        hi = np.where((b == 0) & (a < 0), -1.0, hi)
    h = (start - mid[0]) @ front_normal
    dh = u @ front_normal
    root = np.divide(-h, dh, out=(lo + hi) / 2, where=dh != 0)
    inside = lo <= hi
    if np.any(inside & (((dh != 0) & (lo <= root) & (root <= hi)) | ((dh == 0) & (h == 0)))):
        raise SolverFailure("thickness column intersects pad middle surface")
    t = np.clip(root, lo, hi)
    x = start + t[:, None] * u
    proj = x - (h + t * dh)[:, None] * front_normal
    offer(t, proj, inside)

    # Outside the polygon chart the closest pair lies on a true boundary edge.
    for vertex, v in zip(mid, edges, strict=True):
        vv = v @ v
        for t, x in ((0.0, start), (1.0, end)):
            s = np.clip((x - vertex) @ v / vv, 0.0, 1.0)
            offer(t, vertex + s[:, None] * v)
        for corner in (vertex, vertex + v):
            t = np.clip(
                np.divide(_dot(corner - start, u), uu, out=np.full(size, 0.5), where=uu != 0),
                0.0,
                1.0,
            )
            offer(t, corner)
        cross = np.cross(u, v)
        den = _dot(cross, cross)
        w = vertex - start
        t = np.divide(_dot(np.cross(w, v), cross), den, out=np.zeros(size), where=den != 0)
        s = np.divide(_dot(np.cross(w, u), cross), den, out=np.zeros(size), where=den != 0)
        valid = (den != 0) & (t >= 0) & (t <= 1) & (s >= 0) & (s <= 1)
        offer(np.clip(t, 0.0, 1.0), vertex + np.clip(s, 0.0, 1.0)[:, None] * v, valid)

    distance = np.sqrt(best)
    if np.any(distance <= 0) or not np.isfinite(distance).all():
        raise SolverFailure("unresolved normal at pad middle surface")
    return PadColumnSites(
        2 * chosen - 1,
        start + chosen[:, None] * u,
        direction / distance[:, None],
        distance - radius,
        np.stack((start, end), axis=1),
    )


def check_column_segment(a, d0, b, d1):
    """Reject a possible mid-surface crossing on a pad-local affine column segment."""
    motion = np.linalg.norm(b - a, axis=2).max(axis=1)
    # Distance to the fixed polygon is 1-Lipschitz in the segment's Hausdorff metric.
    if np.any(np.asarray(d0) + d1 <= motion):
        raise SolverFailure("continuous thickness column unresolved at pad middle surface")
