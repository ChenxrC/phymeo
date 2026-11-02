"""Radius contact on a finite segment/triangle, independent of endpoint sampling."""

import json
from dataclasses import asdict

import numpy as np
import pytest
from capsule_face_cases import (
    AXIS_IDS,
    FACE_IDS,
    NAMES,
    authored,
    barrel_edge,
    deforming_edge,
    edge_graze,
    parallel_turn,
    scene,
    shaft_graze,
    tangent_span,
)
from case_catalog import geometry_for
from contact_region_oracles import capsule_face

import phymeo as ph


def query(parts, *, margin=0.0):
    model, paths, face, axis = parts
    with ph.CollisionWorkspace(model) as work:
        (row,) = ph.detect(model, paths, ph.CollisionPolicy(margin=margin), work).pairs
    return row


def save(tmp_path, row):
    (tmp_path / "observation.json").write_text(json.dumps(asdict(row), allow_nan=False))


def parameters(w):
    return tuple(p.axis_parameter for p in w.capsule_axis_span)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("swap", [False, True])
def test_catalog_entire_capsule_against_finite_face(name, swap, tmp_path):
    parts = authored(name, swap=swap)
    _, _, g = geometry_for("RD02", name)
    ref = capsule_face(np.array(g["A"]["vertices"]), np.array(g["B"]["axis"]), g["B"]["radius"])
    row = query(parts, margin=8.0)
    assert row.candidate and row.family == "RD"
    expected = {"separated": "separated", "axis_pierces": "overlap", "cap_tangent": "touching"}
    assert row.status == expected[ref["kind"]], row
    w = row.witness
    assert row.sweep is None
    if ref["kind"] == "separated":
        assert w.gap_m >= ref["gap_lower_m"] - 1e-12 and w.normal is not None
    else:
        axis = np.array(g["B"]["axis"])
        s = ref["axis_parameter"]
        assert parameters(w) == pytest.approx((s, s))
        for end in w.capsule_axis_span:
            np.testing.assert_allclose(
                end.axis_point, axis[0] + s * (axis[1] - axis[0]), atol=1e-12
            )
            np.testing.assert_allclose(end.face_point, ref["point_m"], atol=1e-12)
            assert dict(end.face_weights) == pytest.approx(
                dict(zip(FACE_IDS, ref["face_weights"], strict=True))
            )
        cap_w = dict(w.first_weights if swap else w.second_weights)
        assert cap_w == pytest.approx({AXIS_IDS[0]: 1 - s, AXIS_IDS[1]: s})
        if ref["kind"] == "axis_pierces":
            assert w.gap_m == pytest.approx(-g["B"]["radius"]) and w.normal is None
            assert (w.first_point if swap else w.second_point) is None
            np.testing.assert_allclose(
                w.second_point if swap else w.first_point, ref["point_m"], atol=1e-12
            )
        else:
            assert w.gap_m == 0 and w.first_point == w.second_point
            np.testing.assert_allclose(w.normal, [0, 0, -1 if swap else 1])
    save(tmp_path, row)


def test_first_contact_is_between_shaft_and_edge_interiors(tmp_path):
    parts = barrel_edge()
    row = query(parts)
    assert row.status == "hit", row
    lo, hi = row.sweep.first_t
    assert lo <= 1 / 3 <= hi and hi - lo <= 2**-40
    assert row.sweep.start_gap_m > 0 and row.sweep.end_gap_m > 0
    assert parameters(row.sweep.witness) == pytest.approx((0.75, 0.75))
    assert dict(row.sweep.witness.first_weights) == pytest.approx({7: 0.5, 13: 0.5, 31: 0.0})
    axis = parts[1][parts[3]]
    assert all(abs(x[0] - 0.5) > 0.25 for x in axis.start)
    save(tmp_path, row)


def test_deforming_edge_and_translating_barrel_have_distinct_time_and_parameter(tmp_path):
    row = query(deforming_edge())
    assert row.status == "hit" and row.sweep.first_t == (0.5, 0.5)
    w = row.sweep.witness
    assert parameters(w) == (0.625, 0.625)
    assert dict(w.first_weights) == {7: 0.75, 13: 0.25, 31: 0}
    assert w.first_point == w.second_point == (0.25, 0.0, 0.0) and w.normal == (0.0, 0.0, 1.0)
    save(tmp_path, row)


def test_parallel_face_reports_entire_contact_line_not_one_endpoint(tmp_path):
    row = query(tangent_span())
    assert row.status == "hit" and row.sweep.first_t == (0.5, 0.5)
    w = row.sweep.witness
    assert parameters(w) == (0.375, 0.625)
    assert [p.face_point for p in w.capsule_axis_span] == [(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)]
    assert w.first_point == w.second_point == (0.0, 0.0, 0.0)
    assert dict(w.second_weights) == {11: 0.5, 29: 0.5}
    save(tmp_path, row)


@pytest.mark.parametrize("miss", [False, True])
def test_unsampled_shaft_vertex_graze_and_one_float_miss(miss, tmp_path):
    parts = shaft_graze(miss=miss)
    row = query(parts, margin=0.125)
    if miss:
        assert row.status == "separated" and row.sweep.first_t is None
    else:
        assert row.status == "hit" and row.sweep.first_t[0] <= 1 / 3 <= row.sweep.first_t[1]
        assert parameters(row.sweep.witness) == pytest.approx((0.6875, 0.6875))
        assert dict(row.sweep.witness.first_weights) == {7: 1.0, 13: 0.0, 31: 0.0}
    save(tmp_path, row)


@pytest.mark.parametrize("enters", [False, True])
def test_parallel_supporting_lines_do_not_create_a_false_event(enters, tmp_path):
    row = query(parallel_turn(enters=enters), margin=2.0)
    assert row.candidate
    if enters:
        assert row.status == "hit" and row.sweep.first_t == (0.75, 0.75)
        assert parameters(row.sweep.witness) == (0.125, 0.125)
    else:
        assert row.status == "separated" and row.sweep.first_t is None
    save(tmp_path, row)


def test_persistently_parallel_edges_enter_as_a_finite_interval():
    # Triangle is below y=0. Its upper edge is parallel to the finite axis.
    tri = [[-0.75, 0, 0], [0.75, 0, 0], [0, -1, 0]]
    axis = np.array([[-1, 0.75, 0], [1, 0.75, 0]])
    row = query(scene(tri, None, axis, axis - [0, 1, 0], 0.25))
    assert row.sweep.first_t == (0.5, 0.5)
    w = row.sweep.witness
    assert parameters(w) == (0.125, 0.875)
    assert w.first_point == w.second_point == (0.0, 0.0, 0.0)


def test_persistent_tangency_first_begins_at_a_feature_boundary():
    tri = [[-1, -1, 0], [1, -1, 0], [0, 1, 0]]
    axis = np.array([[-3.5, 0, 0.25], [-1.5, 0, 0.25]])
    row = query(scene(tri, None, axis, axis + [2.0, 0, 0], 0.25))
    assert row.sweep.first_t == (0.5, 0.5)
    assert parameters(row.sweep.witness) == (1.0, 1.0)
    assert parameters(row.witness) == (0.5, 1.0)


def test_zero_length_axis_has_spherical_geometry_and_both_material_nodes():
    tri = [[-1, -1, 0], [1, -1, 0], [0, 1, 0]]
    axis = np.array([[0, 0, 0.75], [0, 0, 0.75]])
    parts = scene(tri, None, axis, axis - [0, 0, 1.0], 0.25)
    row = query(parts)
    model, paths, f, c = parts
    sphere = ph.CollisionModel((ph.CollisionShape(c, "sphere", (11,), 0.25), model.shapes[0]))
    other = {f: paths[f], c: ph.CollisionMotion(paths[c].start[:1], paths[c].end[:1])}
    with ph.CollisionWorkspace(sphere) as work:
        (s,) = ph.detect(sphere, other, ph.CollisionPolicy(), work).pairs
    assert row.sweep.first_t == s.sweep.first_t == (0.5, 0.5)
    w = row.sweep.witness
    assert parameters(w) == (0.0, 1.0) and dict(w.second_weights) == {11: 0.5, 29: 0.5}
    assert w.first_point == s.sweep.witness.first_point


def test_coplanar_axis_publishes_full_span_without_inventing_a_normal():
    row = query(
        scene([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], None, [[-2, 0, 0], [2, 0, 0]], None, 0.25)
    )
    assert row.status == "overlap" and row.witness.gap_m == -0.25
    assert row.witness.normal is None and parameters(row.witness) == (0.375, 0.625)
    assert row.witness.first_point == (0.0, 0.0, 0.0) and row.witness.second_point is None


def test_face_and_capsule_swap_with_axis_reversal_and_a_common_moving_frame():
    model, paths, f, c = deforming_edge()
    r = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    shift = np.array([4, -2, 1])
    drift = np.array([0.5, -0.25, 0.125])
    x, y = paths[f], paths[c]
    f2, c2 = ph.CollisionKey(4, 3, 9), ph.CollisionKey(4, 1, 8)
    fs = (2, 0, 1)
    cs = (1, 0)
    shapes = (
        ph.CollisionShape(c2, "capsule", tuple(AXIS_IDS[i] for i in cs), 0.25),
        ph.CollisionShape(f2, "triangle", tuple(FACE_IDS[i] for i in fs)),
    )

    def trans(arr, order):
        return arr[list(order)] @ r.T + shift

    mo = {
        f2: ph.CollisionMotion(trans(x.start, fs), trans(x.end, fs) + drift),
        c2: ph.CollisionMotion(trans(y.start, cs), trans(y.end, cs) + drift),
    }
    row = query((ph.CollisionModel(shapes), mo, f2, c2))
    w = row.sweep.witness
    assert row.sweep.first_t == (0.5, 0.5) and parameters(w) == (0.375, 0.375)
    assert dict(w.second_weights) == {7: 0.75, 13: 0.25, 31: 0.0}
    np.testing.assert_array_equal(w.first_point, r @ np.array([0.25, 0.0, 0.0]) + shift + drift / 2)
    np.testing.assert_array_equal(w.normal, r @ np.array([0, 0, -1]))


def test_split_step_preserves_global_time_and_axis_address():
    model, paths, f, c = deforming_edge()
    rows = []
    for left, right in ((0.0, 0.25), (0.25, 1.0)):
        segment = {
            k: ph.CollisionMotion(
                v.start + left * (v.end - v.start), v.start + right * (v.end - v.start)
            )
            for k, v in paths.items()
        }
        rows.append(query((model, segment, f, c), margin=1.0))
    assert rows[0].sweep.first_t is None
    lo, hi = rows[1].sweep.first_t
    assert lo <= 1 / 3 <= hi and 0.25 + 0.75 * lo <= 0.5 <= 0.25 + 0.75 * hi
    assert parameters(rows[1].sweep.witness) == pytest.approx((0.625, 0.625))


def test_starting_in_overlap_publishes_entry_and_clear_endpoint_separately():
    parts = authored("base")
    model, paths, f, c = parts
    x = paths[c].start
    paths[c] = ph.CollisionMotion(x, x + [0, 0, 2.0])
    row = query(parts)
    assert row.status == "hit" and row.sweep.first_t == (0.0, 0.0)
    assert row.sweep.witness.normal is None and row.witness.gap_m > 0


def test_thick_faces_rotating_axes_and_interior_collapse_keep_distinct_statuses():
    tri = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]])
    axis = np.array([[-1, 0, 0.25], [1, 0, 0.25]])
    row = query(scene(tri, None, axis, None, 0.5, thickness=0.125))
    assert row.status == "unsupported"
    model, paths, f, c = scene(tri, None, axis, None, 0.5)
    paths[c] = ph.CollisionMotion(axis, turn=ph.RigidTurn([0, 0, 0], [0, 0, 0], [0, 0, 1], 1.0))
    assert query((model, paths, f, c)).status == "unsupported"
    bad = query(scene(tri, tri * [-1, 1, 1], axis, None, 0.5))
    assert bad.status == "unresolved" and bad.reason == "triangle_degenerate_on_path"


def test_projection_to_an_infinite_face_or_line_cannot_create_contact():
    row = query(
        scene(
            [[-1, 0, 0], [1, 0, 0], [0, 1, 0]],
            None,
            [[1.25, -1, 0.125], [1.5, -1, 0.125]],
            None,
            0.125,
        ),
        margin=4.0,
    )
    assert row.candidate and row.status == "separated"
    assert row.witness.gap_m > 0.9


def test_reversed_capsule_preserves_the_closest_interval_world_endpoints():
    x = [[-1, -1, 0], [1, -1, 0], [0, 1, 0]]
    q = [[-1.5, 0, 0.5], [2.5, 0, 0.5]]
    a = query(scene(x, None, q, None, 0.25), margin=1.0).witness
    b = query(scene(x, None, q[::-1], None, 0.25), margin=1.0).witness
    assert parameters(a) == (0.25, 0.5) and parameters(b) == (0.5, 0.75)
    assert [p.axis_point for p in a.capsule_axis_span] == [
        p.axis_point for p in reversed(b.capsule_axis_span)
    ]
    assert a.first_point == b.first_point and a.second_point == b.second_point


@pytest.mark.parametrize("miss", [False, True])
def test_unsampled_shaft_edge_graze_is_a_repeated_radius_root(miss, tmp_path):
    row = query(edge_graze(miss=miss), margin=0.125)
    if miss:
        assert row.status == "separated" and row.sweep.first_t is None
    else:
        assert row.sweep.first_t[0] <= 1 / 3 <= row.sweep.first_t[1]
        assert row.sweep.first_t[1] - row.sweep.first_t[0] <= 2**-40
        w = row.sweep.witness
        assert parameters(w) == pytest.approx((0.6875, 0.6875))
        assert [v for _, v in w.first_weights] == pytest.approx([0.5, 0.5, 0.0], abs=1e-12)
        assert row.sweep.start_gap_m > 0 and row.sweep.end_gap_m > 0
    save(tmp_path, row)
