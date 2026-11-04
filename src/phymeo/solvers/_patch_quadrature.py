"""Material quadrature on ordered convex contact patches."""

from math import fsum

import numpy as np

from ..geometry.collision_witness import ConvexContactPoint


def patch_quadrature(patch):
    count = len(patch.points)
    if patch.dimension < 2:
        for alpha in np.eye(count):
            yield alpha, 1 / count, 0.0
        return
    points = np.array([p.first_point for p in patch.points])
    # Symmetric degree-two quadrature on a fan about one interior point.
    relative = points - points[0]
    arms = relative - relative.mean(axis=0)
    areas = 0.5 * np.abs(np.cross(arms, np.roll(arms, -1, axis=0)) @ np.asarray(patch.normal))
    total = areas.sum()
    if total <= 0 or not np.isclose(total, patch.area_m2, rtol=1e-8, atol=0):
        raise ValueError("contact patch area cannot be resolved")
    for i, area in enumerate(areas):
        if area == 0:
            continue
        for middle, left, right in (
            (2 / 3, 1 / 6, 1 / 6),
            (1 / 6, 2 / 3, 1 / 6),
            (1 / 6, 1 / 6, 2 / 3),
        ):
            alpha = np.full(count, middle / count)
            alpha[i] += left
            alpha[(i + 1) % count] += right
            yield alpha, area / (3 * total), area / 3


def material_sites(patch):
    """Use the same quadrature coordinates on both physical material charts."""
    first = np.array([p.first_point for p in patch.points])
    second = np.array([p.second_point for p in patch.points])

    def weights(side, alpha):
        totals = {}
        for point, factor in zip(patch.points, alpha, strict=True):
            if factor:
                for node, w in getattr(point, side):
                    totals[node] = totals.get(node, 0.0) + factor * w
        return tuple((node, float(w)) for node, w in sorted(totals.items()) if w)

    def point(coordinates, alpha):
        return tuple(fsum(alpha * column) for column in coordinates.T)

    sites, fractions = [], []
    for alpha, fraction, _ in patch_quadrature(patch):
        sites.append(
            ConvexContactPoint(
                point(first, alpha),
                point(second, alpha),
                weights("first_weights", alpha),
                weights("second_weights", alpha),
            )
        )
        fractions.append(float(fraction))
    return tuple(sites), tuple(fractions)
