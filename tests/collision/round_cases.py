"""Registered M0 controls and a moving-axis challenge shared by all M2 backends."""

import numpy as np
from case_catalog import geometry_for

import phymeo as ph

RR01 = ("base", "clear", "filtered", "sphere_pair", "axis_holdout")
RR02 = ("base", "tangent", "overlap", "endcaps", "parallel_holdout", "coincident")


def catalog_scene(case, name, reverse=False):
    _, variant, g = geometry_for(case, name)
    shapes = []
    paths = {}
    keys = (ph.CollisionKey(0, 2 if reverse else 1, 0), ph.CollisionKey(0, 1 if reverse else 2, 0))
    for i, side in enumerate(("A", "B")):
        item = g[side]
        key = keys[i]
        kind = item.get("type", "capsule")
        if kind == "plane":
            normal = np.asarray(item["normal"], dtype=float)
            shapes.append(ph.CollisionShape(key, kind, (19,), normal=normal))
            paths[key] = ph.CollisionMotion([normal * item["offset"]])
        elif kind == "sphere":
            shapes.append(ph.CollisionShape(key, kind, (7,), item["radius"]))
            start = item["center"] if "center" in item else item["center_start"]
            end = [item["center_end"]] if "center_end" in item else None
            paths[key] = ph.CollisionMotion([start], end)
        else:
            shapes.append(ph.CollisionShape(key, kind, (17, 6), item["radius"]))
            paths[key] = ph.CollisionMotion(item["axis"])
    model = ph.CollisionModel(shapes, [keys] if variant.get("exclude_pair") else [])
    return model, paths, variant, g


def crossing_capsules():
    a, b = ph.CollisionKey(0, 4, 0), ph.CollisionKey(0, 8, 0)
    model = ph.CollisionModel([ph.CollisionShape(k, "capsule", (0, 1), 0.125) for k in (a, b)])
    return model, {
        a: ph.CollisionMotion([[-1, 0, 0], [1, 0, 0]]),
        b: ph.CollisionMotion([[0, -1, 1], [0, 1, 1]], [[0, -1, -1], [0, 1, -1]]),
    }
