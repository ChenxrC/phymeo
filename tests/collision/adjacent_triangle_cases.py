"""Same-body registrations preserve the authored global material node IDs."""

import numpy as np
from case_catalog import geometry_for

import phymeo as ph

NAMES = ("base", "flat", "hinge", "renumbered", "holdout")


def mesh(start, end, faces, *, ids=None, world=0):
    start = np.asarray(start, dtype=float)
    end = None if end is None else np.asarray(end, dtype=float)
    ids = tuple(range(len(start))) if ids is None else tuple(ids)
    shapes = []
    paths = {}
    for face_id, indices in enumerate(faces):
        key = ph.CollisionKey(world, 1, face_id)
        nodes = list(indices)
        shapes.append(ph.CollisionShape(key, "triangle", tuple(ids[i] for i in nodes)))
        paths[key] = ph.CollisionMotion(start[nodes], None if end is None else end[nodes])
    return ph.CollisionModel(tuple(reversed(shapes))), paths


def authored(name, world=0):
    _, variant, g = geometry_for("DD03", name)
    data = g["A"]
    vertices = np.asarray(data["current_vertices"], dtype=float)
    order = variant.get("node_order", list(range(len(vertices))))
    inverse = np.argsort(order)
    model, paths = mesh(
        vertices[order], None, inverse[np.asarray(data["faces"])], ids=order, world=world
    )
    return model, paths, tuple(ph.CollisionKey(world, 1, i) for i in g["pair_faces"])


def hinge_path(remote, world=0):
    start = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0.5 if remote else -0.5, -0.5]])
    end = start.copy()
    end[3, 2] = 1.0
    return mesh(start, end, [(0, 1, 2), (1, 0, 3)], world=world)


def vertex_path(world=0):
    # On B, z=y+s*x with s=1/2-t. Along y=0 the common ray is
    # born at t=1/2. A's long tip lies outside B; its edge hits B's opposite edge.
    start = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [1, -1, -0.5], [1, 1, 1.5]])
    end = start.copy()
    end[3:, 2] -= 1
    return mesh(start, end, [(0, 1, 2), (0, 3, 4)], world=world)


def row_for(out, keys):
    return next(r for r in out.pairs if (r.first, r.second) == keys)
