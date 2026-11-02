"""Prescribed two-body triangle queries with independently bound material IDs."""

import numpy as np
from case_catalog import geometry_for

import phymeo as ph

VARIANTS = {
    "DD01": ("base", "clear", "reused_ids", "swapped", "tip_holdout"),
    "DD02": ("base", "clear", "swap_reverse", "holdout"),
}


def scene(a0, a1, b0, b1, *, ids=((11, 7, 3), (31, 41, 53)), world=0):
    keys = (ph.CollisionKey(world, 1, 0), ph.CollisionKey(world, 2, 0))
    model = ph.CollisionModel(
        [ph.CollisionShape(key, "triangle", nodes) for key, nodes in zip(keys, ids, strict=True)]
    )
    paths = {keys[0]: ph.CollisionMotion(a0, a1), keys[1]: ph.CollisionMotion(b0, b1)}
    return model, paths


def authored(case, name, world=0):
    _, v, g = geometry_for(case, name)
    ids = v.get("node_ids", ((11, 7, 3), (31, 41, 53)))
    ids = [
        tuple(nodes[i] for i in v.get("order_" + side, (0, 1, 2)))
        for side, nodes in zip(("A", "B"), ids, strict=True)
    ]
    if v.get("swap"):
        ids.reverse()
    verts = []
    for side in ("A", "B"):
        geom = g[side]
        verts.extend(
            (
                geom["vertices"] if "vertices" in geom else geom["vertices_start"],
                geom.get("vertices_end"),
            )
        )
    return scene(*verts, ids=ids, world=world)


def graze(world=0, lower=0.0):
    # A's edge is the x axis. At y=0 the low edge of B has z=(3t-1)^2/16-lower.
    a = [[-1, 0, 0], [1, 0, 0], [0, 0, -0.25]]

    def b(k):
        return [
            [0, -1 + k / 4, k / 4 - lower],
            [0, 1 + k / 4, -k / 4 - lower],
            [0, k / 4, 2 - lower],
        ]

    return scene(a, None, b(-1), b(2), world=world)


def transit():
    a = [[-1, 0, 0], [1, 0, 0], [0, 0, -0.25]]
    b = np.array([[0, -1, 0.5], [0, 1, 0.5], [0, 0, 1.5]])
    return scene(a, None, b, b - [0, 0, 2])


def piercing(world=0):
    a = [[-2, -2, 0], [2, -2, 0], [0, 2, 0]]
    b = np.array([[0, -0.5, -1], [0, -0.5, 1], [0, 0.5, 1.0]])
    return scene(a, None, b, b + [0, 0, 3], world=world)
