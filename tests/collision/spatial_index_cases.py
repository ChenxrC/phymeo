"""Meshed geometry and frames for exhaustive/accelerated collision comparisons."""

import numpy as np
from case_catalog import geometry_for
from closed_mesh_cases import shapes
from test_solid_boundaries import triangulate_u

import phymeo as ph


def sheet(side):
    # The sphere interacts with the deforming cloth. Cloth self contact is
    # excluded by an explicit symmetric group mask in this workload.
    shapes = []
    triangles = []
    for y in range(side):
        for x in range(side):
            a = y * (side + 1) + x
            for ids in ((a, a + 1, a + side + 2), (a, a + side + 2, a + side + 1)):
                k = ph.CollisionKey(0, 2, len(triangles))
                triangles.append(k)
                shapes.append(ph.CollisionShape(k, "triangle", ids, group=1, mask=2))
    ball = ph.CollisionKey(0, 3, 0)
    shapes.append(ph.CollisionShape(ball, "sphere", (1,), 0.125, 2, 1))
    model = ph.CollisionModel(shapes)
    base = np.array([(x, y, 0) for y in range(side + 1) for x in range(side + 1)], float)
    frames = []
    for n in range(3):
        vertices = base.copy()
        vertices[:, 2] = ((base[:, 0] + 2 * base[:, 1] + n) % 5) / 64
        end = vertices.copy()
        end[:, 2] += ((base[:, 0] - base[:, 1]) % 3) / 128
        # Reverse the spatial order of the strips on the last frame while
        # keeping every material triangle and its complete affine trace valid.
        if n == 2:
            vertices[:, 0] *= -1
            end[:, 0] *= -1
        paths = {
            s.key: ph.CollisionMotion(vertices[list(s.vertex_ids)], end[list(s.vertex_ids)])
            for s in shapes[:-1]
        }
        x = side / 2 + 0.125
        start = [-x if n == 2 else x, side / 2 + 0.25, 0.5]
        paths[ball] = ph.CollisionMotion([start], [[start[0], start[1], -0.5]])
        frames.append(paths)
    return model, frames


def mesh_scene(refine):
    _, _, geom = geometry_for("RR05", "base")
    vertices, faces = triangulate_u(geom["A"], False)
    if refine:
        points = vertices.tolist()
        edges = {}
        split = []

        def mid(i, j):
            key = tuple(sorted((i, j)))
            if key not in edges:
                edges[key] = len(points)
                points.append(((vertices[i] + vertices[j]) / 2).tolist())
            return edges[key]

        for i, j, k in faces:
            ij, jk, ki = mid(i, j), mid(j, k), mid(k, i)
            split.extend(((i, ij, ki), (ij, j, jk), (ki, jk, k), (ij, jk, ki)))
        vertices, faces = np.array(points), np.array(split)
    model, initial = shapes(vertices, faces, geom["B"]["vertices"], step=(1.1, 0, 0))
    mesh, hull = model.shapes
    frames = [initial]
    for offset, step in ((1.25, 0.125), (0, 0.6)):
        paths = dict(initial)
        paths[hull.key] = ph.CollisionMotion(
            initial[hull.key].start + [offset, 0, 0],
            turn=ph.RigidTurn((0, 0, 0), (step, 0, 0), (0, 0, 1), 0),
        )
        frames.append(paths)
    return model, frames
