"""Static pressure integral for a round capsule against the exterior of one or two aligned boxes.

Inputs share a local metre frame. Pressure is gradient * max(radius - axis distance, 0).
The midpoint and numerical bounds describe this elastic reference law at the supplied pose.
"""

from itertools import product

import numpy as np

from phymeo.control.manipulation_push import _coordinates


def exterior_tiles(boxes):
    """Return each exposed rectangle once, including the end faces of a nonconvex union."""
    boxes = np.asarray(boxes, dtype=float)
    if boxes.shape not in ((1, 2, 3), (2, 2, 3)) or not np.isfinite(boxes).all():
        raise ValueError("expected one or two finite axis-aligned boxes")
    if np.any(boxes[:, 1] - boxes[:, 0] <= 1e-9):
        raise ValueError("box widths must exceed coordinate resolution")
    axes = [_coordinates(boxes[:, :, axis].ravel())[0] for axis in range(3)]
    solid = {}
    for cell in product(*(range(len(grid) - 1) for grid in axes)):
        lo = np.array([axes[i][j] for i, j in enumerate(cell)])
        hi = np.array([axes[i][j + 1] for i, j in enumerate(cell)])
        center = (lo + hi) / 2
        inside = ((boxes[:, 0] <= center) & (center <= boxes[:, 1])).all(1)
        if inside.any():
            solid[cell] = (lo, hi)
    tiles = []
    for cell, (lo, hi) in solid.items():
        for axis, side in product(range(3), (-1, 1)):
            neighbor = list(cell)
            neighbor[axis] += side
            if tuple(neighbor) in solid:
                continue
            half, center = (hi - lo) / 2, (lo + hi) / 2
            center[axis] += side * half[axis]
            half[axis] = 0.0
            normal = np.eye(3)[axis] * side
            tiles.append((center, half, normal))
    return tiles


def axis_box_clearance(ends, boxes, radius):
    """Minimum segment-to-solid distance minus radius, positive for separated solids."""
    first, last = np.asarray(ends, dtype=float)
    delta = last - first
    best = np.inf
    for lo, hi in boxes:
        cuts = [0.0, 1.0]
        for i in np.flatnonzero(delta):
            cuts.extend(
                t
                for t in ((lo[i] - first[i]) / delta[i], (hi[i] - first[i]) / delta[i])
                if 0 < t < 1
            )
        cuts = sorted(set(cuts))
        for left, right in zip(cuts[:-1], cuts[1:], strict=True):
            mid = first + (left + right) / 2 * delta
            offset = mid - np.clip(mid, lo, hi)
            outside = offset != 0
            velocity = delta[outside]
            t = (left + right) / 2
            if velocity @ velocity:
                t = np.clip(t - offset[outside] @ velocity / (velocity @ velocity), left, right)
            p = first + t * delta
            best = min(best, np.linalg.norm(p - np.clip(p, lo, hi)))
    return float(best - radius)


def _residual(points, ends):
    delta = ends[1] - ends[0]
    rel = points - ends[0]
    length2 = delta @ delta
    if length2 == 0:
        return rel
    fraction = np.clip((rel @ delta) / length2, 0, 1)
    return rel - fraction[..., None] * delta


def integrate(ends, boxes, radius, gradient, com, rotation, *, depth=8):
    """World force and moment about the given COM, on the rigid box union.

    For each leaf, convexity bounds the maximum capsule-axis distance by the maximum
    at its four corners. A supporting plane at the center bounds the minimum.
    Area, compressive load, three force and three moment components occupy rows 0..7.
    """
    ends, com, rotation = (np.asarray(x, dtype=float) for x in (ends, com, rotation))
    if (
        ends.shape != (2, 3)
        or com.shape != (3,)
        or rotation.shape != (3, 3)
        or not all(np.isfinite(a).all() for a in (ends, com, rotation))
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=3e-6, rtol=0)
        or np.linalg.det(rotation) <= 0
    ):
        raise ValueError("invalid capsule, center of mass or rotation")
    if not (np.isfinite(radius) and radius > 0 and np.isfinite(gradient) and gradient > 0):
        raise ValueError("positive finite radius and pressure gradient required")
    if type(depth) is not int or not 0 <= depth <= 10:
        raise ValueError("depth must be between zero and ten")
    bounds = np.zeros((8, 2))
    leaves = 0
    for center, half, normal in exterior_tiles(boxes):
        area = 4 * half[half > 0].prod()
        offsets = np.array(list(product(*[(-h, h) if h else (0.0,) for h in half])))
        centers = center[None]
        face_force = rotation @ (-normal)
        for level in range(depth + 1):
            rel = _residual(centers, ends)
            dist = np.linalg.norm(rel, axis=1)
            unit = np.divide(rel, dist[:, None], out=np.zeros_like(rel), where=dist[:, None] > 0)
            # Pad comparisons for floating-point geometry in this metre-scale reference.
            pad = 64 * np.finfo(float).eps * max(1.0, np.max(np.abs(ends)), np.max(np.abs(centers)))
            lower = np.maximum(0, dist - np.abs(unit) @ half - pad)
            live = lower < radius
            if not live.any():
                break
            centers, lower = centers[live], lower[live]
            if level < depth:
                half = half / 2
                offsets = offsets / 2
                centers = (centers[:, None] + offsets).reshape(-1, 3)
                area /= 4
                continue
            corners = centers[:, None] + offsets
            upper = np.linalg.norm(_residual(corners, ends), axis=2).max(1) + pad
            flo = area * gradient * np.maximum(radius - upper, 0)
            fhi = area * gradient * (radius - lower)
            bounds[0] += area * np.array([(upper < radius).sum(), len(upper)])
            bounds[1] += [flo.sum(), fhi.sum()]
            for i, factor in enumerate(face_force):
                bounds[2 + i] += np.sort(factor * np.array([flo.sum(), fhi.sum()]))
            lever = np.cross((corners - com) @ rotation.T, face_force)
            low, high = lever.min(1), lever.max(1)
            bounds[5:, 0] += np.minimum(flo[:, None] * low, fhi[:, None] * low).sum(0)
            bounds[5:, 1] += np.maximum(flo[:, None] * high, fhi[:, None] * high).sum(0)
            leaves += len(upper)
    return dict(
        bounds=bounds.tolist(), leaves=leaves, clearance_m=axis_box_clearance(ends, boxes, radius)
    )
