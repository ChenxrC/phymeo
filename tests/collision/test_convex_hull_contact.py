"""Full-dimensional convex face, edge, and point regions through the public detector."""

import json
from dataclasses import asdict, replace
from fractions import Fraction as F

import numpy as np
import pytest
from case_catalog import geometry_for
from contact_region_oracles import box_slide, last_face_time, skew_edge_time
from convex_hull_cases import (
    NAMES,
    authored,
    corners,
    isolated,
    over_axis_budget,
    passing,
    sample_paths,
    scene,
    sliding_frames,
    tetrahedra,
)

import phymeo as ph


def query(model, paths, margin=0.0):
    with ph.CollisionWorkspace(model) as work:
        return ph.detect(model, paths, ph.CollisionPolicy(margin=margin), work).pairs[0]


def material(point, model, paths, t):
    for side, shape in zip(("first", "second"), model.shapes, strict=True):
        p = paths[shape.key]
        if p.turn is not None:
            d = np.array(p.turn.pivot_end) - np.array(p.turn.pivot_start)
        else:
            d = p.end[0] - p.start[0] if p.end is not None else np.zeros(3)
        weights = dict(getattr(point, side + "_weights"))
        assert set(weights) <= set(shape.vertex_ids) and min(weights.values()) > 0
        assert sum(weights.values()) == pytest.approx(1.0)
        xyz = p.start + float(t) * d
        np.testing.assert_allclose(
            sum(w * xyz[shape.vertex_ids.index(i)] for i, w in weights.items()),
            getattr(point, side + "_point"),
            atol=2e-12,
            rtol=0,
        )


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("reverse", [False, True])
def test_original_hull_contacts_and_intervals(name, reverse, tmp_path):
    model, paths = authored(name, reverse=reverse)
    _, variant, g = geometry_for("RR03", name)
    a, b = g["A"], g["B"]
    row = query(model, paths, variant.get("prediction_margin_m", 0.0))
    assert row.status == ("separated" if name == "clear_gap" else "hit"), row
    trace = row.convex
    assert trace is not None and row.family == "RR" and row.candidate
    if name == "edge_edge":
        t, z = skew_edge_time(a, b)
        assert F(*trace.first_time_ratio) == t and F(*trace.last_time_ratio) == 1
        assert row.sweep.first_t[0] <= float(t) <= row.sweep.first_t[1]
        assert trace.first.relation == "touching" and trace.end.relation == "overlap"
        (patch,) = trace.first.patches
        assert patch.dimension == 0 and len(patch.points) == 1
        assert len(patch.first_support) == len(patch.second_support) == 2
        pt = patch.points[0]
        np.testing.assert_allclose(pt.first_point, [0, 0, float(z)], atol=1e-12)
        assert pt.first_point == pt.second_point
        assert [w for _, w in pt.first_weights] == pytest.approx([0.5, 0.5])
        assert [w for _, w in pt.second_weights] == pytest.approx([0.5, 0.5])
        material(pt, model, paths, t)
    elif name == "clear_gap":
        assert trace.first_time_ratio is trace.last_time_ratio is trace.first is trace.last is None
        assert not trace.start.patches and trace.start.relation == "separated"
        assert trace.start.witness.gap_m == pytest.approx(0.02)
        assert trace.end.witness.gap_m == pytest.approx(box_slide(a, b, F(1))[2])
    else:
        exit_time = last_face_time(a, b)
        assert F(*trace.first_time_ratio) == 0 and F(*trace.last_time_ratio) == exit_time
        assert row.sweep.first_t == (0.0, 0.0) and trace.end.relation == "separated"
        assert not trace.end.patches
        for t, pose in (
            (F(0), trace.start),
            (F(1, 5), query(model, sample_paths(model, paths, F(1, 5))).convex.start),
            (exit_time, trace.last),
        ):
            expected, area, _, dim = box_slide(a, b, t)
            assert pose.relation == "touching" and pose.patches
            for patch in pose.patches:
                assert patch.dimension == dim and patch.area_m2 == pytest.approx(float(area))
                assert {tuple(point.first_point[:2]) for point in patch.points} == {
                    tuple(map(float, xy)) for xy in expected
                }
                for point in patch.points:
                    assert point.first_point == point.second_point and point.first_point[2] == 0.5
                    material(point, model, paths, t)
            if t == 0:
                assert pose.patches[0].normal == (0.0, 0.0, -1.0 if reverse else 1.0)
    (tmp_path / "observation.json").write_text(json.dumps(asdict(row), allow_nan=False))


def test_two_separated_endpoints_keep_both_contact_times(tmp_path):
    model, paths = passing()
    row = query(model, paths)
    tr = row.convex
    assert row.status == "hit" and row.sweep.first_t == (0.25, 0.25)
    assert tr.start.relation == tr.end.relation == "separated"
    assert tr.first_time_ratio == (1, 4) and tr.last_time_ratio == (3, 4)
    for pose, normal in ((tr.first, (-1.0, 0.0, 0.0)), (tr.last, (1.0, 0.0, 0.0))):
        assert pose.patches[0].normal == normal and pose.patches[0].area_m2 == 1.0
    (tmp_path / "observation.json").write_text(json.dumps(asdict(row), allow_nan=False))


def test_isolated_corner_between_grid_times_has_three_tied_normals(tmp_path):
    model, paths = isolated()
    row = query(model, paths)
    assert row.convex.first_time_ratio == row.convex.last_time_ratio == (1, 3)
    assert row.sweep.first_t[0] <= 1 / 3 <= row.sweep.first_t[1]
    assert row.sweep.first_t[1] - row.sweep.first_t[0] <= 2**-40
    assert row.sweep.witness.gap_m > 0 and not row.sweep.witness.intersection_points
    entry = row.convex.first
    assert entry.witness.normal is None
    assert {patch.normal for patch in entry.patches} == {
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    }
    assert all(
        len(patch.points) == 1 and patch.points[0].first_point == (0.5, 0.5, 0.5)
        for patch in entry.patches
    )
    (tmp_path / "observation.json").write_text(json.dumps(asdict(row), allow_nan=False))


def test_tetrahedron_vertex_meets_an_unprovided_face_interior(tmp_path):
    model, paths = tetrahedra()
    row = query(model, paths)
    assert row.convex.first_time_ratio == (1, 2) and row.convex.axis_count > 3
    (patch,) = row.convex.first.patches
    assert set(patch.first_support) == {21, 40, 50} and patch.second_support == (2,)
    (point,) = patch.points
    assert point.first_point == point.second_point == (0.0, 0.25, 0.25)
    assert dict(point.first_weights) == {21: 0.5, 40: 0.25, 50: 0.25}
    material(point, model, paths, F(1, 2))
    (tmp_path / "observation.json").write_text(json.dumps(asdict(row), allow_nan=False))


def test_disconnected_projections_report_euclidean_clearance_from_the_corners():
    a = corners({"center": [0, 0, 0], "half": [0.5, 0.5, 0.5]})
    model, paths = scene([(a, None, range(8)), (a + [1.25, 1.5, 1.75], None, range(8))])
    row = query(model, paths, 2.0)
    assert row.status == "separated" and not row.convex.start.patches
    assert row.witness.gap_m == pytest.approx(np.sqrt(14) / 4)
    assert row.witness.first_point == (0.5, 0.5, 0.5) and row.witness.second_point == (
        0.75,
        1.0,
        1.25,
    )


def test_contained_cube_uses_full_escape_depth_and_all_symmetric_supports():
    a = corners({"center": [0, 0, 0], "half": [1, 1, 1]})
    model, paths = scene([(a, None, range(8)), (a / 4, None, range(8))])
    row = query(model, paths)
    assert row.status == "overlap" and row.witness.gap_m == -1.25 and row.witness.normal is None
    assert len(row.convex.start.patches) == 6 and not row.witness.intersection_points
    for patch in row.convex.start.patches:
        assert patch.area_m2 == 0.25 and len(patch.points) == 4
        for point in patch.points:
            material(point, model, paths, F(0))
            np.testing.assert_allclose(
                np.array(point.second_point) - point.first_point, -1.25 * np.array(patch.normal)
            )


def test_coplanar_and_interior_inputs_do_not_become_diagonals_or_support_owners():
    a = corners({"center": [0, 0, 0], "half": [0.5, 0.5, 0.5]})
    ref = query(*scene([(a, None, range(10, 18)), (a + [0, 0, 1], None, range(30, 38))]))
    extras = np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0.5]])
    x = np.vstack((extras, a))
    ids = (1, 2, 3, *range(10, 18))
    order = [7, 1, 8, 3, 5, 10, 4, 2, 0, 9, 6]
    model, paths = scene(
        [(x[order], None, [ids[i] for i in order]), (a + [0, 0, 1], None, range(30, 38))]
    )
    result = query(model, paths)
    assert result.convex == ref.convex and result.witness == ref.witness


def test_contact_interval_preserved_by_step_split_and_moving_frame():
    base, paths = authored("base")
    a, b = base.shapes
    x, y = paths[a.key].start, paths[b.key].start
    dx = np.array([1.25, 0, 0])
    left = query(*scene([(x, None, a.vertex_ids), (y, dx / 2, b.vertex_ids)]))
    right = query(*scene([(x, None, a.vertex_ids), (y + dx / 2, dx / 2, b.vertex_ids)]))
    assert F(*left.convex.last_time_ratio) == 1
    assert F(1, 2) + F(*right.convex.last_time_ratio) / 2 == F(4, 5)
    r = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    shift = np.array([2, -4, 1])
    drift = np.array([0.5, -0.25, 0.125])
    other, motion = scene(
        [(x @ r.T + shift, drift, a.vertex_ids), (y @ r.T + shift, dx @ r.T + drift, b.vertex_ids)],
        world=2,
        reverse=True,
    )
    transformed = query(other, motion)
    assert transformed.convex.last_time_ratio == (4, 5)
    expected = query(base, paths).convex.last
    for patch in expected.patches:
        normal = tuple(-(r @ patch.normal))
        (found,) = [v for v in transformed.convex.last.patches if v.normal == normal]
        assert (
            found.second_support == patch.first_support
            and found.first_support == patch.second_support
        )
        for point in patch.points:
            target = r @ point.first_point + shift + 0.8 * drift
            assert any(
                np.allclose(z.second_point, target, atol=1e-13, rtol=0) for z in found.points
            )


@pytest.mark.parametrize("bad", ["coplanar", "duplicate", "capacity"])
def test_unsupported_hull_input_cannot_publish_false_separation(bad):
    a = corners({"center": [0, 0, 0], "half": [0.5, 0.5, 0.5]})
    if bad == "coplanar":
        b = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    elif bad == "duplicate":
        b = np.vstack((a, a[:1]))
    else:
        b = np.array([[i, i * i, i**3] for i in range(65)], dtype=float)
    model, paths = scene([(a, None, range(8)), (b + [10, 0, 0], None, range(len(b)))])
    row = query(model, paths)
    assert not row.candidate and row.status == (
        "unresolved" if bad == "capacity" else "invalid_geometry"
    )
    assert row.convex is None and row.witness is None


def test_rounding_and_rotating_hulls_keep_explicit_scope():
    model, paths = authored("base")
    a, b = model.shapes
    rounded = ph.CollisionModel([replace(a, radius=0.125), b])
    assert query(rounded, paths).status == "unsupported"
    paths[b.key] = ph.CollisionMotion(
        paths[b.key].start, turn=ph.RigidTurn([0, 0, 0], [0, 0, 0], [0, 1, 0], 0.5)
    )
    assert query(model, paths).status == "unsupported"


def test_same_workspace_keeps_the_sliding_face_identity_until_exit(tmp_path):
    rows, _ = sliding_frames("host", tmp_path / "cache")
    (tmp_path / "history.json").write_text(json.dumps(list(map(asdict, rows)), allow_nan=False))


def test_axis_limit_returns_unresolved_instead_of_a_partial_contact(tmp_path):
    model, paths = over_axis_budget()
    row = query(model, paths)
    assert row.candidate and row.status == "unresolved" and row.reason == "convex_axis_capacity"
    assert row.convex is None and row.sweep is None and row.witness is None
    (tmp_path / "observation.json").write_text(json.dumps(asdict(row), allow_nan=False))
