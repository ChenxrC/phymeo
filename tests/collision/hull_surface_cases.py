"""Declared affine cloth triangles and arbitrary translating convex rigid bodies."""

from itertools import product

import numpy as np
from case_catalog import geometry_for

import phymeo as ph

NAMES = ("base", "clear", "outside_box", "rigid_transform", "tip_holdout")


def scene(cloud, start, end, *, ids=(31, 32, 33), reverse=False, world=0, travel=(0, 0, 0)):
    hard = ph.CollisionKey(world, 9 if reverse else 1, 0)
    soft = ph.CollisionKey(world, 1 if reverse else 9, 0)
    model = ph.CollisionModel(
        [
            ph.CollisionShape(hard, "hull", tuple(range(101, 101 + len(cloud)))),
            ph.CollisionShape(soft, "triangle", ids),
        ]
    )
    paths = {
        hard: ph.CollisionMotion(cloud, turn=ph.RigidTurn((0, 0, 0), travel, (0, 0, 1), 0)),
        soft: ph.CollisionMotion(start, end),
    }
    return model, paths


def authored(name, reverse=False, world=0):
    _, variant, g = geometry_for("RD05", name)
    box = g["A"]
    coords = np.array(list(product((-1, 1), repeat=3)), float) * box["half"] + box["center"]
    cloth = g["B"]
    frame = np.array(variant.get("rotation", np.eye(3)))
    tr = np.array(variant.get("translation", [0, 0, 0]))
    ids = tuple(31 + i for i in variant.get("order_B", [0, 1, 2]))
    model, paths = scene(
        coords @ frame.T + tr,
        np.asarray(cloth["vertices_start"]) @ frame.T + tr,
        np.asarray(cloth["vertices_end"]) @ frame.T + tr,
        ids=ids,
        reverse=reverse,
        world=world,
    )
    return model, paths, variant["expected"]


def wide_face(world=0):
    hull = list(product((-0.5, 0.5), repeat=3))
    start = np.array([[-2, -1, 1], [2, -1, 1], [0, 2, 1]], float)
    return scene(hull, start, start - [0, 0, 1], world=world)


def changing_face(world=0):
    # At the origin the triangle's plane height is (2-3*t-t*t)/(2+t).
    cloud = [[0, 0, 0], [-0.125, -0.125, -1], [0.125, -0.125, -1], [0, 0.125, -1]]
    start = [[-1, -1, 1], [1, -1, 1], [0, 1, 1]]
    end = [[-1, -1, 0], [1, -1, 0], [0, 2, -2]]
    return scene(cloud, start, end, world=world)
