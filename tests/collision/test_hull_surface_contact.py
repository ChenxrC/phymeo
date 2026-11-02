"""Rigid solids versus independently moving material vertices and full faces."""

from itertools import product
from math import sqrt

import numpy as np
import pytest
from hull_surface_cases import NAMES, authored, changing_face, scene, wide_face

import phymeo as ph


def row(model, paths):
    with ph.CollisionWorkspace(model) as work:
        return ph.detect(model, paths, ph.CollisionPolicy(), work).pairs[0]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("reverse", [False, True])
def test_registered_deforming_triangle_box(name, reverse):
    model, paths, expected = authored(name, reverse)
    p = row(model, paths)
    assert p.family == "RD"
    if expected["first_t"] is None:
        assert p.status == "separated" and (not p.candidate or p.sweep.first_t is None)
        return
    assert p.status == "hit" and p.sweep.first_t[0] <= expected["first_t"] <= p.sweep.first_t[1]
    trace = p.hull_surface
    assert trace.start.relation == "separated" and trace.entry.relation == "touching"
    assert trace.exact_first_ratio and trace.end.relation == "cut"
    (point,) = trace.entry.section
    np.testing.assert_allclose(point.point, expected["point_m"], atol=1e-12, rtol=0)
    cloth_weights = point.first_weights if reverse else point.second_weights
    assert dict(cloth_weights) == {31: 1}
    assert trace.end.area_m2 > 0 and trace.entry.boundary_faces


def test_face_first_with_every_cloth_vertex_outside_and_finite_area():
    p = row(*wide_face())
    assert p.status == "hit" and p.hull_surface.exact_first_ratio == (1, 2)
    e = p.hull_surface.entry
    assert e.relation == "touching" and e.area_m2 == pytest.approx(1)
    assert len(e.section) == 4 and len(e.boundary_faces) == 5
    top = {102, 104, 106, 108}
    assert sorted(len(set(face) & top) for face in e.boundary_faces) == [2, 2, 2, 2, 4]
    assert e.support_normals == ((0.0, 0.0, 1.0),) and e.witness.normal == e.support_normals[0]


def test_fully_contained_sheet_then_exits_and_tetrahedral_rigid_shape():
    cloud = [[-1, -1, -1], [2, -1, -1], [-1, 2, -1], [-1, -1, 2]]
    start = np.array([[-0.5, -0.5, -0.5], [0, -0.5, -0.5], [-0.5, 0, -0.5]])
    p = row(*scene(cloud, start, start + [0, 0, 3], reverse=True))
    assert p.sweep.first_t == (0, 0) and p.hull_surface.start.relation == "inside"
    assert p.hull_surface.end.relation == "separated"
    assert not p.hull_surface.start.boundary_faces and p.hull_surface.start.area_m2 > 0
    for s in p.hull_surface.start.section:
        total = sum(w * np.asarray(cloud[i - 101]) for i, w in s.second_weights)
        np.testing.assert_allclose(total, s.point, atol=1e-12)


def test_midstep_degeneracy_and_unsupported_offset_are_explicit():
    model, paths = wide_face()
    soft = model.shapes[1]
    s = paths[soft.key].start.copy()
    s[:, 2] = 0.5
    paths[soft.key] = ph.CollisionMotion(s, s[[0, 2, 1]])
    assert row(model, paths).status == "unresolved"
    expanded = ph.CollisionModel(
        [model.shapes[0], ph.CollisionShape(soft.key, "triangle", soft.vertex_ids, 0.125)]
    )
    assert row(expanded, paths).status == "unsupported"


def test_two_separated_endpoints_cross_the_box_on_its_edges():
    cloud = list(product((-0.5, 0.5), repeat=3))
    start = np.array([[1, -1, 0], [1, 1, 0], [1.25, 0, 0.25]])
    p = row(*scene(cloud, start, start - [2, 0, 0]))
    assert p.status == "hit" and p.sweep.first_t == (0.25, 0.25)
    assert p.hull_surface.end.relation == "separated"
    assert len(p.hull_surface.entry.section) == 2
    assert {round(s.point[1], 10) for s in p.hull_surface.entry.section} == {-0.5, 0.5}


def test_first_event_is_an_irrational_deforming_face_contact():
    p = row(*changing_face())
    root = (sqrt(17) - 3) / 2
    assert p.status == "hit" and p.sweep.first_t[0] <= root <= p.sweep.first_t[1]
    assert p.sweep.first_t[1] - p.sweep.first_t[0] < 1e-10
    assert p.hull_surface.exact_first_ratio is None
    assert abs(p.sweep.witness.gap_m) < 1e-10


def test_split_motion_with_translating_hull_preserves_global_event_and_material():
    model, paths, _ = authored("base")
    hard, soft = model.shapes
    start, end = paths[soft.key].start, paths[soft.key].end
    unit = np.array([0.125, -0.25, 0.5])
    first_paths = dict(paths)
    first_paths[soft.key] = ph.CollisionMotion(start, start * 0.75 + end * 0.25 + 0.25 * unit)
    first_paths[hard.key] = ph.CollisionMotion(
        paths[hard.key].start, turn=ph.RigidTurn([0, 0, 0], 0.25 * unit, [1, 0, 0], 0)
    )
    assert row(model, first_paths).status == "separated"
    second_paths = dict(paths)
    second_paths[hard.key] = ph.CollisionMotion(
        paths[hard.key].start + 0.25 * unit, turn=ph.RigidTurn([0, 0, 0], 0.75 * unit, [1, 0, 0], 0)
    )
    second_paths[soft.key] = ph.CollisionMotion(first_paths[soft.key].end, end + unit)
    p = row(model, second_paths)
    assert p.status == "hit"
    lo, hi = p.sweep.first_t
    assert 0.25 + 0.75 * lo <= 0.5 <= 0.25 + 0.75 * hi
    np.testing.assert_allclose(
        p.hull_surface.entry.section[0].point, np.array([0, 0, 0.1]) + unit * 0.5, atol=1e-12
    )
