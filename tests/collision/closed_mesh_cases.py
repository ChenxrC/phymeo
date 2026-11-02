"""The registered U mesh and convex probes expressed through normal shape registration."""

from itertools import product

import numpy as np
from case_catalog import geometry_for
from solid_case_oracles import u_solid_path
from test_solid_boundaries import triangulate_u

import phymeo as ph

NAMES = ("base", "flip_diagonal", "clear_notch", "permuted_hull", "left_arm_holdout")


def shapes(vertices, faces, points, *, step=(0, 0, 0), reverse=False, world=0, offset=(0, 0, 0)):
    mesh_key = ph.CollisionKey(world, 7 if reverse else 2, 0)
    hull_key = ph.CollisionKey(world, 2 if reverse else 7, 0)
    ids = tuple(100 + 3 * i for i in range(len(vertices)))
    triangles = tuple(tuple(ids[i] for i in face) for face in faces)
    mesh = ph.CollisionShape(mesh_key, "mesh", ids, triangles=triangles)
    hull = ph.CollisionShape(hull_key, "hull", tuple(3 + 5 * i for i in range(len(points))))
    paths = {
        mesh_key: ph.CollisionMotion(vertices),
        hull_key: ph.CollisionMotion(
            np.asarray(points) + offset, turn=ph.RigidTurn((0, 0, 0), step, (0, 0, 1), 0)
        ),
    }
    return ph.CollisionModel([hull, mesh]), paths


def authored(name, reverse=False, world=0):
    _, variant, g = geometry_for("RR05", name)
    vertices, faces = triangulate_u(g["A"], variant.get("flip_diagonal", False))
    body = g["B"]
    start, end = map(np.asarray, (body["translation_start"], body["translation_end"]))
    model, paths = shapes(
        vertices,
        faces,
        body["vertices"],
        step=end - start,
        reverse=reverse,
        world=world,
        offset=start,
    )
    return model, paths, u_solid_path(g["A"], body)


def containing(world=0):
    _, v, g = geometry_for("RR05", "base")
    vertices, faces = triangulate_u(g["A"], False)
    return shapes(vertices, faces, list(product((-3, 3), repeat=3)), world=world)
