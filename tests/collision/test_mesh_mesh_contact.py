"""Independent physical surface oracles for two closed rigid triangle meshes."""

from itertools import product

import numpy as np
import pytest

import phymeo as ph


def box(low, high, flip=False):
    x, y, z = zip(low, high, strict=True)
    vertices = np.array([(x[0], y[0], z[0]), (x[1], y[0], z[0]), (x[1], y[1], z[0]),
                         (x[0], y[1], z[0]), (x[0], y[0], z[1]), (x[1], y[0], z[1]),
                         (x[1], y[1], z[1]), (x[0], y[1], z[1])], dtype=float)
    faces = []
    for a, b, c, d in ((0,3,2,1), (4,5,6,7), (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7)):
        faces.extend(((a,b,d), (b,c,d)) if flip else ((a,b,c), (a,c,d)))
    return vertices, faces


def scene(one, two, *, reverse=False, travel=None, rotation=None, origin=(0,0,0)):
    shapes, paths = [], {}
    matrix = np.eye(3) if rotation is None else np.asarray(rotation)
    for side, (vertices, faces) in enumerate((one, two)):
        key = ph.CollisionKey(0, (1-side if reverse else side) + 1, 0)
        ids = tuple(11 + 5*i for i in range(len(vertices)))  # The owners can reuse local IDs.
        triangles = tuple(tuple(ids[i] for i in tri) for tri in faces)
        shapes.append(ph.CollisionShape(key, "mesh", ids, triangles=triangles))
        coords = np.asarray(vertices) @ matrix.T + origin
        paths[key] = ph.CollisionMotion(coords, None if travel is None else coords + travel[side])
    return ph.CollisionModel(shapes), paths


def detect(model, paths, margin=0):
    with ph.CollisionWorkspace(model) as work:
        (pair,) = ph.detect(model, paths, ph.CollisionPolicy(margin=margin), work).pairs
    return pair


def check_support(support, model, paths, t=0.0):
    witness = support.witness
    for shape, faces, weights, point in zip(
        model.shapes, (support.first_faces, support.second_faces),
        (witness.first_weights, witness.second_weights),
        (witness.first_point, witness.second_point), strict=True
    ):
        path = paths[shape.key]
        points = path.start if path.end is None else path.start + t * (path.end - path.start)
        lookup = dict(zip(shape.vertex_ids, points, strict=True))
        assert faces and all(face in shape.triangles for face in faces)
        assert all({i for i, w in weights if w} <= set(face) for face in faces)
        assert sum(w for _, w in weights) == pytest.approx(1)
        np.testing.assert_allclose(sum(w*lookup[i] for i, w in weights), point, atol=1e-12, rtol=0)
    if witness.normal is not None:
        d = np.subtract(witness.second_point, witness.first_point)
        assert np.dot(d, witness.normal) == pytest.approx(witness.gap_m, abs=1e-12)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("rotated", [False, True])
def test_incursion_returns_all_four_true_wall_supports(reverse, rotated):
    one = box((-1,-1,-1),(1,1,1))
    two = box((0.875,-0.25,-0.25),(1.5,0.25,0.25), flip=True)
    rotation = np.array([[0,0,1],[1,0,0],[0,1,0]]) if rotated else np.eye(3)
    origin = np.array([2,-1,0.5])
    model, paths = scene(one, two, reverse=reverse, rotation=rotation, origin=origin)
    row = detect(model, paths)
    assert row.status == "overlap" and row.sweep is None and row.family == "RR"
    pose = row.mesh_mesh.end
    assert pose.relation == "boundary_overlap" and pose.boundary_gap_m == 0
    assert pose.vertex_intrusion_m == 0.125 and len(pose.deepest) == 4 and pose.boundary
    found = set()
    for support in (*pose.deepest, *pose.boundary):
        check_support(support, model, paths)
    for support in pose.deepest:
        w = support.witness
        a, b = (w.second_point, w.first_point) if reverse else (w.first_point, w.second_point)
        local_a, local_b = rotation.T @ (a-origin), rotation.T @ (b-origin)
        assert local_a[0] == 1 and local_b[0] == 0.875 and w.gap_m == -0.125
        np.testing.assert_allclose(w.normal, rotation @ ([-1,0,0] if reverse else [1,0,0]), atol=1e-12)
        found.add(tuple(local_a[1:]))
    assert found == set(product((-0.25, 0.25), repeat=2))
    for support in pose.boundary:
        np.testing.assert_allclose(support.witness.first_point, support.witness.second_point, atol=1e-12)


def test_crossed_solids_without_interior_vertices_keep_boundary_evidence():
    one = box((-1,-0.125,-0.25),(1,0.125,0.25))
    two = box((-0.125,-1,-0.125),(0.125,1,0.125))
    model, paths = scene(one, two)
    row = detect(model, paths)
    pose = row.mesh_mesh.end
    assert row.status == "overlap" and pose.relation == "boundary_overlap"
    assert pose.boundary and pose.boundary_gap_m == 0
    assert pose.vertex_intrusion_m is None and not pose.deepest
    for support in pose.boundary:
        check_support(support, model, paths)


@pytest.mark.parametrize("reverse", [False, True])
def test_containment_has_positive_boundary_clearance(reverse):
    model, paths = scene(box((-1,-1,-1),(1,1,1)), box((-0.25,)*3,(0.25,)*3), reverse=reverse)
    row = detect(model, paths)
    p = row.mesh_mesh.end
    assert row.status == "overlap" and not p.boundary and p.boundary_gap_m == 0.75
    assert p.relation == ("second_contains_first" if reverse else "first_contains_second")
    assert p.vertex_intrusion_m == 0.75 and row.witness.normal is None
    for support in p.deepest:
        check_support(support, model, paths)


def test_coincident_different_diagonals_versus_coplanar_external_touch():
    cube = box((0,0,0),(1,1,1))
    model, paths = scene(cube, box((0,0,0),(1,1,1), flip=True))
    same = detect(model, paths)
    assert same.status == "overlap" and same.mesh_mesh.end.relation == "coincident"
    assert same.mesh_mesh.end.vertex_intrusion_m is None and same.mesh_mesh.end.boundary
    model, paths = scene(cube, box((1,0.25,0.25),(2,0.75,0.75), flip=True))
    touching = detect(model, paths)
    assert touching.status == "touching" and touching.mesh_mesh.end.relation == "touching"
    assert touching.mesh_mesh.end.vertex_intrusion_m is None and touching.mesh_mesh.end.boundary
    np.testing.assert_allclose(touching.witness.normal, [1,0,0], atol=1e-12)


def test_invalid_and_rotation_do_not_become_false_separation():
    cube = box((0,0,0),(1,1,1))
    invalid = cube[0], cube[1][:-1]
    model, paths = scene(cube, invalid, origin=(1,2,3))
    assert detect(model, paths).status == "invalid_geometry"
    model, paths = scene(cube, cube)
    key = max(paths)
    paths[key] = ph.CollisionMotion(paths[key].start, turn=ph.RigidTurn((0,0,0),(0,0,0),(0,0,1),1))
    assert detect(model, paths).status == "unsupported"
