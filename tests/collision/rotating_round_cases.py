"""Independent circular-motion controls for the public RR04 query."""

from math import cos, pi, radians, sin

import numpy as np
from case_catalog import geometry_for
from case_oracles import rotating_capsule

import phymeo as ph


def scene(
    segment,
    ball,
    angle,
    *,
    axis=(0, 0, 1),
    pivot=(0, 0, 0),
    travel=(0, 0, 0),
    ball_travel=None,
    reverse=False,
    world=0,
    radii=(0.05, 0.05),
):
    cap = ph.CollisionKey(world, 5 if reverse else 1, 0)
    orb = ph.CollisionKey(world, 1 if reverse else 5, 0)
    pivot = np.asarray(pivot, float)
    ball = np.asarray(ball, float)
    model = ph.CollisionModel(
        [
            ph.CollisionShape(cap, "capsule", (37, 11), radii[0]),
            ph.CollisionShape(orb, "sphere", (19,), radii[1]),
        ]
    )
    turn = ph.RigidTurn(pivot, pivot + travel, axis, angle)
    paths = {
        cap: ph.CollisionMotion(segment, turn=turn),
        orb: ph.CollisionMotion([ball], [ball + (travel if ball_travel is None else ball_travel)]),
    }
    return model, paths


def authored(name, reverse=False, world=0):
    _, _, g = geometry_for("RR04", name)
    rot = g["A"]["rotation"]
    phi = radians(rot["angle_start_deg"])
    frame = np.array([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]])
    model, paths = scene(
        np.asarray(g["A"]["local_axis"]) @ frame.T,
        g["B"]["center"],
        radians(rot["angle_end_deg"] - rot["angle_start_deg"]),
        reverse=reverse,
        world=world,
        radii=(g["A"]["radius"], g["B"]["radius"]),
    )
    return model, paths, rotating_capsule(g["A"], g["B"])


def isolated(reverse=False, world=0, miss=False):
    # Only the positive cap can reach (0, 1.125, 0), exactly once per revolution.
    y = np.nextafter(1.125, 2.0) if miss else 1.125
    return scene(
        [[0.25, 0, 0], [1, 0, 0]],
        [0, y, 0],
        2 * pi,
        reverse=reverse,
        world=world,
        radii=(0.0625, 0.0625),
    )
