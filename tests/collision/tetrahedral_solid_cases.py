"""Independent registrations for the two-volume detector."""

import numpy as np
from case_catalog import geometry_for

import phymeo as ph

NAMES = ("base", "clear", "collapsed", "swapped", "external_touch", "deforming_holdout")


def mesh(bodies, *, world=0, exclude=()):
    shapes, paths = [], {}
    for owner, (start, end, cells) in enumerate(bodies):
        x = np.asarray(start, dtype=float)
        y = x if end is None else np.asarray(end, dtype=float)
        for i, ids in enumerate(cells):
            key = ph.CollisionKey(world, owner, i)
            shapes.append(ph.CollisionShape(key, "tetrahedron", tuple(ids)))
            paths[key] = ph.CollisionMotion(x[list(ids)], y[list(ids)])
    return ph.CollisionModel(tuple(shapes), exclude), paths


def authored(name, world=0):
    _, _, g = geometry_for("DD05", name)
    return mesh(
        [(s["vertices"], s.get("vertices_end"), s["tets"]) for s in (g["A"], g["B"])], world=world
    )


def across_seam(world=0):
    outer = [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [0, 0, -2]]
    inner = [[0.25, 0.25, -0.125], [0.5, 0.25, -0.125], [0.25, 0.5, -0.125], [0.25, 0.25, 0.125]]
    return mesh(
        [(outer, None, [(0, 1, 2, 3), (0, 2, 1, 4)]), (inner, None, [(0, 1, 2, 3)])], world=world
    )


def crossed_cells(world=0):
    # The two alternating tetrahedra of a cube intersect in |x|+|y|+|z|<=1.
    a = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    return mesh([(a, None, [(0, 2, 1, 3)]), (-a, None, [(0, 1, 2, 3)])], world=world)


def edge_birth(world=0):
    # Only the upper edge of A and lower edge of B can meet at the first time.
    a = [[-1, 0, 0], [1, 0, 0], [0, -1, -1], [0, 1, -1]]
    b = np.array([[0, -1, 0.5], [0, 1, 0.5], [-1, 0, 1.5], [1, 0, 1.5]])
    return mesh([(a, None, [(0, 1, 2, 3)]), (b, b - [0, 0, 1], [(0, 1, 2, 3)])], world=world)


def crossing_internal_face(world=0):
    a = [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [0, 0, -2]]
    b = np.array(
        [[0.25, 0.25, -0.5], [0.375, 0.25, -0.5], [0.25, 0.375, -0.5], [0.25, 0.25, -0.375]]
    )
    return mesh(
        [(a, None, [(0, 1, 2, 3), (0, 2, 1, 4)]), (b, b + [0, 0, 1], [(0, 1, 2, 3)])], world=world
    )


def invalid_material_mesh(world=0, *, crossing=False):
    unit = np.vstack((np.zeros(3), np.eye(3)))
    if crossing:
        unit /= 4
        start = np.vstack((unit + [-1, 0, 0], unit + [1, 0, 0]))
        end = np.vstack((unit + [1, 0, 0], unit + [-1, 0, 0]))
        other = unit + [0, 0.125, 0]
    else:
        start = np.vstack((unit, unit))
        end = None
        other = (unit + 1) / 8
    return mesh(
        [(start, end, [(0, 1, 2, 3), (4, 5, 6, 7)]), (other, None, [(0, 1, 2, 3)])], world=world
    )
