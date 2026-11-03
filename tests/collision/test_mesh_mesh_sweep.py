"""Closed-surface translation times use complete facet-pair intervals."""

from fractions import Fraction

import numpy as np
import pytest
from test_mesh_mesh_contact import box, check_support, detect, scene


def check_time(row, root):
    assert row.status == "hit" and row.mesh_mesh.first_ratio == (root.numerator, root.denominator)
    lo, hi = row.sweep.first_t
    assert Fraction(lo) <= root <= Fraction(hi) and 0 <= hi - lo <= 2**-40
    assert lo <= row.sweep.witness_time <= hi


@pytest.mark.parametrize("reverse", [False, True])
def test_two_translating_meshes_bind_contact_to_both_material_faces(reverse):
    one = box((0, 0, 0), (1, 1, 1))
    two = box((-0.75, 0.25, 0.25), (-0.5, 0.75, 0.75), flip=True)
    common = np.array([0, 0.25, -0.125])
    model, paths = scene(one, two, reverse=reverse, travel=(common, common + [1.25, 0, 0]))
    row = detect(model, paths)
    check_time(row, Fraction(2, 5))
    assert row.mesh_mesh.start.relation == "separated"
    assert row.mesh_mesh.end.relation == (
        "second_contains_first" if reverse else "first_contains_second"
    )
    at = row.mesh_mesh.entry
    assert at.relation == "touching" and at.boundary_gap_m == 0 and at.boundary
    assert at.vertex_intrusion_m is None
    for support in at.boundary:
        check_support(support, model, paths, 0.4)
        assert support.witness.first_point[0] == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(at.witness.normal, [1 if reverse else -1, 0, 0], atol=1e-12)


def test_through_two_clear_endpoints_and_initial_containment_are_distinct():
    one = box((0, 0, 0), (1, 1, 1))
    crossing = box((-0.75, 0.25, 0.25), (-0.5, 0.75, 0.75))
    model, paths = scene(one, crossing, travel=([0, 0, 0], [2.5, 0, 0]))
    row = detect(model, paths)
    check_time(row, Fraction(1, 5))
    assert row.mesh_mesh.start.relation == row.mesh_mesh.end.relation == "separated"
    model, paths = scene(one, box((0.25,) * 3, (0.5,) * 3), travel=([0, 0, 0], [2, 0, 0]))
    row = detect(model, paths)
    check_time(row, Fraction(0))
    assert row.mesh_mesh.entry.relation == "first_contains_second"
    assert not row.mesh_mesh.entry.boundary


def test_isolated_corner_contact_and_one_ulp_miss():
    one = box((0, 0, 0), (1, 1, 1))
    for delta, hit in [(0.0, True), (np.spacing(1.5), False)]:
        two = box((0.5, 1.5 + delta, 0.25), (0.75, 1.75 + delta, 0.75))
        model, paths = scene(one, two, travel=([0, 0, 0], [1, -1, 0]))
        row = detect(model, paths)
        assert row.mesh_mesh.start.relation == row.mesh_mesh.end.relation == "separated"
        if hit:
            check_time(row, Fraction(1, 2))
            assert row.mesh_mesh.entry.relation == "touching"
            for s in row.mesh_mesh.entry.boundary:
                np.testing.assert_allclose(s.witness.first_point[:2], [1, 1], atol=1e-12)
        else:
            assert row.status == "separated" and row.sweep.first_t is None
            assert row.mesh_mesh.entry is None


def test_coplanar_motion_checks_in_plane_axes():
    one = box((0, 0, 0), (1, 1, 1))
    for y, hit in [(0.25, True), (1.25, False)]:
        two = box((1.5, y, 1), (1.75, y + 0.25, 1.25))
        model, paths = scene(one, two, travel=([0, 0, 0], [-1, 0, 0]))
        row = detect(model, paths, margin=1)
        if hit:
            check_time(row, Fraction(1, 2))
            assert row.mesh_mesh.entry.boundary
        else:
            assert row.status == "separated" and row.sweep.first_t is None
