"""Ten convex-family inputs with independent parallel support-plane gaps."""

from itertools import combinations_with_replacement, product

import numpy as np

from phymeo.constraints.collision_shapes import CollisionKey, CollisionMotion, CollisionShape

KINDS = ("sphere", "capsule", "box", "hull")
PAIRS = tuple(combinations_with_replacement(KINDS, 2))


def convex_case(first_kind, second_kind, gap, *, world=0):
    """Return registrations and stationary paths; expected signed gap is input gap."""
    shapes, paths = [], {}
    for body, kind in enumerate((first_kind, second_kind), 1):
        if kind == "sphere":
            points, radius, extent = [(0, 0, 0)], 0.05, 0.05
        elif kind == "capsule":
            points, radius, extent = [(-0.5, 0, 0), (0.5, 0, 0)], 0.05, 0.05
        elif kind == "box":
            points = list(product((-0.6, 0.6), (-0.1, 0.1), (-0.1, 0.1)))
            radius, extent = 0.0, 0.1
        elif kind == "hull":
            ring = ((-0.5, -0.1), (0.5, -0.1), (0.6, 0), (0.3, 0.1), (-0.5, 0.1))
            points = [(x, y, z) for x, y in ring for z in (-0.1, 0.1)]
            radius, extent = 0.0, 0.1
        else:
            raise ValueError("unknown convex family")
        xyz = np.asarray(points, dtype=float)
        if body == 1:
            first_extent = extent
        else:
            xyz[:, 2] += first_extent + extent + gap
        key = CollisionKey(world, body, 0)
        shape = CollisionShape(
            key,
            "hull" if kind == "box" else kind,
            tuple(range(body * 100, body * 100 + len(xyz))),
            radius=radius,
        )
        shapes.append(shape)
        paths[key] = CollisionMotion(xyz, xyz)
    return tuple(shapes), paths
