"""A true turning shaft and an independently moving, finite cloth face."""

from math import cos, pi, sin

import numpy as np

import phymeo as ph


def turning_face(lo=0, hi=1, *, clear=False, swap=False, world=0, change_frame=False):
    # Middle positions are tabulated at t=0 and t=1. At t=1/2 the face is
    # y=r, its triangle contains the shaft projection and earlier y is larger.
    r = 0.125
    xz = np.array([[0.625, -1], [1.375, -1], [1, 1.25]], float)
    d = np.array([0.125, 0.25, 0.375])

    def tri(t):
        y = np.full(3, 1.25) if clear else r + (1 - 2 * t) * d
        return np.column_stack((xz[:, 0], y, xz[:, 1]))

    def axis(t):
        c, s = cos(pi * t), sin(pi * t)
        return np.array([[s, -c, -0.5], [s, -c, 0.5]])

    q = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float) if change_frame else np.eye(3)
    b = np.array([4, -2, 1]) if change_frame else np.zeros(3)
    kcap, kface = (
        ph.CollisionKey(world, 3 if swap else 1, 7),
        ph.CollisionKey(world, 1 if swap else 3, 5),
    )
    model = ph.CollisionModel(
        [
            ph.CollisionShape(kcap, "capsule", (15, 6), r),
            ph.CollisionShape(kface, "triangle", (27, 36, 49)),
        ]
    )
    # All nodes and the pivot translate together in addition to the relative motion.
    drift = np.array([0.25, -0.125, 0.5])

    def world_pose(x, t):
        return (x + t * drift) @ q.T + b

    paths = {
        kcap: ph.CollisionMotion(
            world_pose(axis(lo), lo),
            turn=ph.RigidTurn(
                world_pose(np.zeros(3), lo),
                world_pose(np.zeros(3), hi),
                q @ [0, 0, 1],
                pi * (hi - lo),
            ),
        ),
        kface: ph.CollisionMotion(world_pose(tri(lo), lo), world_pose(tri(hi), hi)),
    }
    return model, paths, kcap, kface


def switching_segment(lo, hi, world=0):
    radius = 0.2
    cap, cloth = ph.CollisionKey(world, 1, 0), ph.CollisionKey(world, 2, 0)
    m = ph.CollisionModel(
        [
            ph.CollisionShape(cap, "capsule", (8, 9), radius),
            ph.CollisionShape(cloth, "triangle", (10, 11, 12)),
        ]
    )

    def triangle(t):
        xz = np.array([[-2, -2], [2, -2], [0, 2]], float)
        return np.column_stack((xz[:, 0], 0.15 + (1 - 2 * t) * 0.1 * xz[:, 1], xz[:, 1]))

    angle = 0.25 * lo
    points = [[0.25 * cos(angle), 0.25 * sin(angle), z] for z in (-0.5, 0.5)]
    paths = {
        cap: ph.CollisionMotion(
            points, turn=ph.RigidTurn((0, 0, 0), (0, 0, 0), (0, 0, 1), 0.25 * (hi - lo))
        ),
        cloth: ph.CollisionMotion(triangle(lo), triangle(hi)),
    }
    return m, paths
