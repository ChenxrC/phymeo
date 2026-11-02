"""Stable material inputs for the finite capsule/cloth controls."""

import numpy as np
from case_catalog import geometry_for

import phymeo as ph

NAMES = ("base", "clear", "cap_touch", "holdout", "reverse_holdout")
FACE_IDS = (7, 13, 31)
AXIS_IDS = (11, 29)


def scene(face, end_face, axis, end_axis, radius, world=0, swap=False, thickness=0.0):
    f, c = ph.CollisionKey(world, 2 if swap else 0, 4), ph.CollisionKey(world, 1, 8)
    shapes = (
        ph.CollisionShape(c, "capsule", AXIS_IDS, radius),
        ph.CollisionShape(f, "triangle", FACE_IDS, thickness),
    )
    return (
        ph.CollisionModel(shapes),
        {f: ph.CollisionMotion(face, end_face), c: ph.CollisionMotion(axis, end_axis)},
        f,
        c,
    )


def authored(name, world=0, swap=False):
    _, _, g = geometry_for("RD02", name)
    return scene(g["A"]["vertices"], None, g["B"]["axis"], None, g["B"]["radius"], world, swap)


def barrel_edge(world=0):
    face = [[0.5, -1, 0], [0.5, 1, 0], [0.5, 0, -0.25]]
    axis = np.array([[-1, 0, 0.75], [1, 0, 0.75]])
    return scene(face, None, axis, axis - [0, 0, 1.5], 0.25, world)


def deforming_edge(world=0):
    f = [(0, -0.375, -0.0625), (0.5, 1.25, -0.1875), (0.25, 0, -0.75)]
    g = [(0.25, -0.625, 0.0625), (0.75, 1.75, 0.1875), (0.375, 0, -0.875)]
    axis = np.array([[-1, 0, 0.75], [1, 0, 0.75]])
    return scene(f, g, axis, axis - [0, 0, 1], 0.25, world)


def tangent_span(world=0):
    f = [[-1, -1, 0], [1, -1, 0], [0, 1, 0]]
    axis = np.array([[-2, 0, 0.75], [2, 0, 0.75]])
    return scene(f, None, axis, axis - [0, 0, 1.0], 0.25, world)


def shaft_graze(world=0, *, miss=False):
    face = [[0.375, 0, 0], [0, 1, -0.25], [0.75, 1, -0.25]]
    y = -np.nextafter(0.25, 1.0) if miss else -0.25
    axis = np.array([[-1, y, 0.5], [1, y, 0.5]])
    return scene(face, None, axis, axis - [0, 0, 1.5], 0.25, world)


def parallel_turn(world=0, *, enters=False):
    lo, hi = (1.0, 0.0) if enters else (1.0, 1.0)
    f = [[-0.75, lo, 0], [0.75, lo, 0.5], [0, lo + 1, -0.25]]
    g = [[-0.75, hi, 0], [0.75, hi, -0.5], [0, hi + 1, -0.25]]
    return scene(f, g, [[-1, 0, 0], [1, 0, 0]], None, 0.25, world)


def edge_graze(world=0, *, miss=False):
    # In the transverse yz plane the edge is z - v*y/4 = 1/4 + v²/16,
    # v = 3*t-1. Only v=0 puts this line at distance 1/4 from the x axis.
    x = 0.375
    face = [[x, -0.25, 0.375], [x, 0.75, 0.125], [x, 0.25, 1.25]]
    end = [[x, -1, 0], [x, 0, 0.5], [x, -0.5, 1.25]]
    r = np.nextafter(0.25, 0.0) if miss else 0.25
    return scene(face, end, [[-1, 0, 0], [1, 0, 0]], None, r, world)
