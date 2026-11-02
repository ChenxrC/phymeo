"""Closed cell volume, independent material witnesses, and physical exterior events."""

import json
from dataclasses import asdict
from fractions import Fraction as F

import numpy as np
import pytest
from case_catalog import geometry_for
from solid_case_oracles import deforming_touch, plane_weights, static_tets
from tetrahedral_solid_cases import (
    NAMES,
    across_seam,
    authored,
    crossed_cells,
    crossing_internal_face,
    edge_birth,
    invalid_material_mesh,
    mesh,
)

import phymeo as ph


def run(model, paths):
    with ph.CollisionWorkspace(model) as work:
        return ph.detect(model, paths, ph.CollisionPolicy(), work)


def check_material(point, model, paths, owners, time, *, strict):
    for k, weights in zip(owners, (point.first_weights, point.second_weights), strict=True):
        shape = next(s for s in model.shapes if s.key == k)
        path = paths[k]
        x = path.start if path.end is None else (1 - time) * path.start + time * path.end
        by_node = dict(weights)
        assert set(by_node) == set(shape.vertex_ids)
        w = np.array([by_node[n] for n in shape.vertex_ids])
        assert w.sum() == pytest.approx(1.0) and (min(w) > 0 if strict else min(w) >= 0)
        np.testing.assert_allclose(w @ x, point.point, atol=2e-12, rtol=0)


@pytest.mark.parametrize("name", NAMES)
def test_catalog_closed_occupancy_and_exterior_ccd(name, tmp_path):
    model, paths = authored(name)
    out = run(model, paths)
    (pair,) = out.pairs
    if name in {"base", "swapped"}:
        assert pair.status == "overlap", pair
    elif name == "collapsed":
        assert pair.status == "invalid_geometry", pair
    elif name == "clear":
        assert pair.status == "separated"
    else:
        assert pair.status == "hit", pair
    (body,) = out.solid_pairs
    _, _, g = geometry_for("DD05", name)
    x, y = [np.array(g[s]["vertices"])[g[s]["tets"][0]] for s in ("A", "B")]
    if name in {"clear", "base", "swapped", "collapsed"}:
        ref = static_tets(x, y)
        if ref["kind"] == "invalid_geometry":
            assert body.status == "invalid_geometry" and pair.solid is None
            assert body.start is None and body.first_t is None
        elif ref["kind"] == "contained_overlap":
            relation = "first_contains_second" if ref["outer"] == "A" else "second_contains_first"
            assert (
                body.status == "ok" and body.start.relation == relation == pair.solid.start.relation
            )
            assert (
                body.start == body.end
                and body.first_t == (0.0, 0.0)
                and body.boundary_first_t is None
            )
            assert body.start.volume_m3 == pytest.approx(ref["overlap_volume_m3"])
            b = pair.solid.boundary
            assert b.start.witness.gap_m == pytest.approx(0.2) and b.first_t is None
            assert not b.start.witness.intersection_points
            check_material(body.start.witness, model, paths, body.start.owners, 0.0, strict=True)
        else:
            assert body.start.relation == "separated" and body.first_t is None
            assert body.start.volume_m3 == 0 and body.start.witness is None
    else:
        x1, y1 = [np.array(g[s]["vertices_end"])[g[s]["tets"][0]] for s in ("A", "B")]
        ref = deforming_touch(np.array([x, x1]), np.array([y, y1]))
        lo, hi = body.first_t
        assert lo <= ref["first_t"] + 1e-15 and hi >= ref["first_t"] - 1e-15 and hi - lo <= 2**-40
        assert body.boundary_first_t == pair.solid.boundary.first_t == pair.sweep.first_t
        assert (
            body.start.relation == "separated"
            and body.end.relation == "overlap"
            and body.end.volume_m3 > 0
        )
        check_material(body.end.witness, model, paths, body.end.owners, 1.0, strict=True)
        b = pair.solid.boundary
        assert float(F(*b.event_time_ratio)) == pytest.approx(ref["first_t"])
        w = b.event_witness.witness
        assert set(b.event_witness.first_face) == {0, 2, 3} and w.gap_m == 0
        np.testing.assert_allclose(w.first_point, ref["point_m"], atol=1e-12)
        np.testing.assert_allclose(
            [dict(w.first_weights)[i] for i in (0, 1, 2, 3)], ref["a_weights"], atol=1e-12
        )
        assert dict(w.second_weights)[0] == 1.0
    (tmp_path / "observation.json").write_text(
        json.dumps(dict(pair=asdict(pair), body=asdict(body)), allow_nan=False)
    )


def test_whole_body_containment_spans_canceled_face(tmp_path):
    model, paths = across_seam()
    out = run(model, paths)
    cells = [r for r in out.pairs if r.first.body != r.second.body]
    assert all(r.status == "overlap" for r in cells), cells
    assert all(r.solid.start.relation == "overlap" for r in cells)
    (body,) = out.solid_pairs
    assert body.start.relation == "first_contains_second" and body.boundary_first_t is None
    volume = F(1, 4) ** 3 / 6
    assert F(*body.start.volume_ratio) == volume
    assert sum(F(*c.solid.start.volume_ratio) for c in cells) == volume
    assert all(r.solid.boundary.start.witness.gap_m > 0 for r in cells)
    check_material(body.start.witness, model, paths, body.start.owners, 0.0, strict=True)
    (tmp_path / "observation.json").write_text(json.dumps(asdict(body), allow_nan=False))


def test_no_inside_vertex_still_has_positive_octahedral_volume():
    model, paths = crossed_cells()
    out = run(model, paths)
    (pair,) = out.pairs
    a, b = (p.start for p in paths.values())
    for outer, other in ((a, b), (b, a)):
        assert all(min(row) < 0 for row in plane_weights(outer, other))
    s = pair.solid.start
    assert s.relation == "overlap" and F(*s.volume_ratio) == F(4, 3)
    np.testing.assert_array_equal(s.witness.point, [0, 0, 0])
    check_material(s.witness, model, paths, s.owners, 0.0, strict=True)


@pytest.mark.parametrize(
    ("scales", "count"), [((1, 1, -1), 3), ((1, -1, -1), 2), ((-1, -1, -1), 1)]
)
def test_closed_face_edge_and_vertex_touch_have_zero_volume(scales, count):
    a = np.vstack((np.zeros(3), np.eye(3)))
    b = a * scales
    indices = (0, 2, 1, 3) if np.prod(scales) < 0 else (0, 1, 2, 3)
    model, paths = mesh([(a, None, [(0, 1, 2, 3)]), (b, None, [indices])])
    out = run(model, paths)
    (pair,) = out.pairs
    assert pair.status == "touching" and out.solid_pairs[0].start.volume_ratio == (0, 1)
    check_material(
        pair.solid.start.witness, model, paths, pair.solid.start.owners, 0.0, strict=False
    )
    assert sum(w > 0 for _, w in pair.solid.start.witness.first_weights) == count
    assert pair.solid.boundary.first_t == (0.0, 0.0)


def test_coincident_cells_count_each_coplanar_face_once():
    a = np.vstack((np.zeros(3), np.eye(3)))
    model, paths = mesh([(a, None, [(0, 1, 2, 3)]), (a, None, [(1, 2, 0, 3)])])
    (body,) = run(model, paths).solid_pairs
    assert body.start.relation == "coincident" and F(*body.start.volume_ratio) == F(1, 6)
    assert body.boundary_first_t == (0.0, 0.0)


def test_only_edge_interiors_create_the_first_exterior_contact():
    model, paths = edge_birth()
    out = run(model, paths)
    (p,) = out.pairs
    assert p.sweep.first_t == (0.5, 0.5) and p.solid.boundary.first_t == (0.5, 0.5)
    w = p.solid.boundary.event_witness.witness
    assert w.first_point == w.second_point == (0.0, 0.0, 0.0)
    assert dict(w.first_weights) == dict(w.second_weights) == {0: 0.5, 1: 0.5, 2: 0.0, 3: 0.0}
    assert out.solid_pairs[0].end.relation == "overlap"


def test_movement_across_internal_face_preserves_solid_containment():
    model, paths = crossing_internal_face()
    out = run(model, paths)
    body = out.solid_pairs[0]
    assert body.start.relation == body.end.relation == "first_contains_second"
    assert body.first_t == (0.0, 0.0) and body.boundary_first_t is None
    (upper,) = [r for r in out.pairs if r.first.shape == 0 and r.first.body != r.second.body]
    assert upper.solid.start.relation == "separated" and upper.sweep.first_t == (0.375, 0.375)
    assert upper.solid.boundary.first_t is None


def test_filled_hit_at_entry_and_later_boundary_exit_are_distinct():
    a = np.vstack((np.zeros(3), np.eye(3)))
    b = (a + 1) / 8
    model, paths = mesh([(a, None, [(0, 1, 2, 3)]), (b, b + [1, 0, 0], [(0, 1, 2, 3)])])
    out = run(model, paths)
    body = out.solid_pairs[0]
    assert body.first_t == (0.0, 0.0) and body.boundary_first_t == (0.5, 0.5)
    assert body.start.relation == "first_contains_second" and body.end.relation == "separated"


def test_clear_endpoints_do_not_hide_a_swept_solid_hit():
    a = np.vstack((np.zeros(3), np.eye(3)))
    b = a / 4 + [-2, 0.25, 0.25]
    model, paths = mesh([(a, None, [(0, 1, 2, 3)]), (b, b + [4, 0, 0], [(0, 1, 2, 3)])])
    (body,) = run(model, paths).solid_pairs
    assert body.start.relation == body.end.relation == "separated"
    assert body.first_t == body.boundary_first_t == (7 / 16, 7 / 16)


def test_partial_filters_never_assert_complete_body_containment():
    model, paths = across_seam()
    k0, k1, k2 = [s.key for s in model.shapes]
    filtered = ph.CollisionModel(model.shapes, ((k0, k2),))
    out = run(filtered, paths)
    (body,) = out.solid_pairs
    assert body.status == "partial" and body.start is None and body.first_t is None
    assert next(r for r in out.pairs if (r.first, r.second) == (k1, k2)).solid.start.volume_m3 > 0


def test_aabb_clear_cannot_label_a_collapsed_body_as_valid_separation():
    a = np.vstack((np.zeros(3), np.eye(3)))
    b = a.copy() + [4, 0, 0]
    b[:, 2] = 0
    model, paths = mesh([(a, None, [(0, 1, 2, 3)]), (b, None, [(0, 1, 2, 3)])])
    out = run(model, paths)
    (p,) = out.pairs
    assert not p.candidate and out.required_candidates == 0 and p.status == "invalid_geometry"
    assert out.solid_pairs[0].status == "invalid_geometry"


def test_two_valid_endpoints_with_a_midstep_collapse_are_invalid():
    a = np.vstack((np.zeros(3), np.eye(3)))
    model, paths = mesh([(a, None, [(0, 1, 2, 3)]), (a, a * [-1, -1, 1], [(0, 1, 2, 3)])])
    assert run(model, paths).solid_pairs[0].status == "invalid_geometry"


def test_overlapping_boxes_can_hold_disjoint_tetrahedra():
    a = np.vstack((np.zeros(3), np.eye(3)))
    b = a / 8 + [0.5, 0.5, 0.5]
    model, paths = mesh([(a, None, [(0, 1, 2, 3)]), (b, None, [(0, 1, 2, 3)])])
    (p,) = run(model, paths).pairs
    assert p.candidate and p.status == "separated" and p.solid.boundary.start.witness.gap_m > 0
    assert p.solid.start.witness is None


def test_body_box_cannot_fill_the_space_between_disconnected_components():
    unit = np.vstack((np.zeros(3), np.eye(3)))
    a = np.vstack((unit + [-3, 0, 0], unit + [3, 0, 0]))
    model, paths = mesh([(a, None, [(0, 1, 2, 3), (4, 5, 6, 7)]), (unit, None, [(0, 1, 2, 3)])])
    (body,) = run(model, paths).solid_pairs
    assert body.start.relation == "separated" and body.first_t is None


def test_first_time_and_owners_survive_permutation_and_a_moving_frame():
    model, paths = authored("external_touch")
    r = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    delta = np.array([2, -1, 3])
    drift = np.array([0.25, 0.5, -0.125])
    order = [1, 2, 0, 3]
    shapes = []
    transformed = {}
    for s in model.shapes:
        k = ph.CollisionKey(4, 5 - s.key.body, s.key.shape + 7)
        shapes.append(
            ph.CollisionShape(k, "tetrahedron", tuple(s.vertex_ids[i] + 10 for i in order))
        )
        p = paths[s.key]
        transformed[k] = ph.CollisionMotion(
            p.start[order] @ r.T + delta, p.end[order] @ r.T + delta + drift
        )
    other = ph.CollisionModel(tuple(reversed(shapes)))
    (row,) = run(other, transformed).pairs
    t = float(F(*row.solid.boundary.event_time_ratio))
    assert row.sweep.first_t[0] <= 0.75 <= row.sweep.first_t[1]
    witness = row.solid.boundary.event_witness.witness
    np.testing.assert_allclose(
        witness.first_point, r @ np.array([0, 0.4, 0.4]) + delta + t * drift, atol=2e-12
    )
    assert dict(witness.first_weights)[10] == 1.0
    check_material(
        row.solid.end.witness, other, transformed, row.solid.end.owners, 1.0, strict=True
    )


def test_duplicate_physical_cells_with_distinct_ids_cannot_double_the_intersection_volume():
    model, paths = invalid_material_mesh()
    out = run(model, paths)
    assert out.solid_pairs[0].status == "invalid_geometry" and out.solid_pairs[0].start is None


def test_remote_cells_crossing_midstep_make_body_exterior_unresolved():
    model, paths = invalid_material_mesh(crossing=True)
    assert run(model, paths).solid_pairs[0].status == "unresolved"


@pytest.mark.parametrize("shared", ["edge", "vertex"])
def test_legal_same_body_edge_or_vertex_does_not_disable_solid_detection(shared):
    unit = np.vstack((np.zeros(3), np.eye(3)))
    if shared == "edge":
        outer = np.vstack((unit, [[0, -1, 0], [0, 0, -1]]))
        cells = [(0, 1, 2, 3), (0, 1, 4, 5)]
    else:
        outer = np.vstack((unit, [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]))
        cells = [(0, 1, 2, 3), (0, 5, 4, 6)]
    end = outer.copy()
    end[-1, 2] -= 0.25
    small = (unit + 1) / 8
    model, paths = mesh([(outer, end, cells), (small, None, [(0, 1, 2, 3)])])
    body = run(model, paths).solid_pairs[0]
    assert (
        body.status == "ok" and body.start.relation == body.end.relation == "first_contains_second"
    )
