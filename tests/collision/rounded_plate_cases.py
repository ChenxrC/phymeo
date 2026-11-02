"""All authored finite rounded-plate queries, independent of the narrow phase."""

from math import sqrt

import numpy as np
from case_catalog import geometry_for

import phymeo as ph

NAMES = ("base", "large_face", "rim_bypass", "through_plate", "corner_holdout")


def authored(name, reverse=False, world=0):
    _, variant, g = geometry_for("RD03", name)
    plate_key = ph.CollisionKey(world, 9 if reverse else 1, 0)
    plate = g["A"]
    shapes = [ph.CollisionShape(plate_key, "plate", (10, 20, 30, 40), plate["radius"])]
    paths = {plate_key: ph.CollisionMotion(plate["polygon"])}
    cloth = g["B"]
    if cloth["type"] == "cloth_triangle":
        entries = [("triangle", cloth["vertices_start"], cloth["vertices_end"])]
    elif "start" in cloth:
        entries = [("particle", [cloth["start"]], [cloth["end"]])]
    else:
        points = list(cloth["points"])
        if "corner_point_expression" in cloth:
            points.append([0.5 + 0.15 / sqrt(2), 0.5 + 0.15 / sqrt(2), 0])
        entries = [("particle", [p], None) for p in points]
    for i, (kind, start, end) in enumerate(entries):
        k = ph.CollisionKey(world, 1 if reverse else 9, i)
        ids = tuple(range(70 + 3 * i, 70 + 3 * i + len(start)))
        shapes.append(ph.CollisionShape(k, kind, ids))
        paths[k] = ph.CollisionMotion(
            np.asarray(start, float), None if end is None else np.asarray(end, float)
        )
    return ph.CollisionModel(shapes), paths, variant["expected"]


def edge_case(world=0, parallel=False):
    plate = ph.CollisionKey(world, 1, 0)
    cloth = ph.CollisionKey(world, 2, 0)
    model = ph.CollisionModel(
        [
            ph.CollisionShape(plate, "plate", (11, 22, 33, 44), 0.125),
            ph.CollisionShape(cloth, "triangle", (70, 71, 72)),
        ]
    )
    polygon = [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]]
    if parallel:
        start, end = [[0.625, -1, 0], [0.625, 1, 0], [1.5, 0, 0.5]], None
    else:
        start = [[1, 0, -1], [1, 0, 1], [2, 0.25, 0]]
        end = [[0.5, 0, -1], [0.5, 0, 1], [1.75, 0.375, 0]]
    return model, {plate: ph.CollisionMotion(polygon), cloth: ph.CollisionMotion(start, end)}
