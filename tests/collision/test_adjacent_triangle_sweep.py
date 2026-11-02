"""A shared material address exempts only its own geometric coincidence."""

import json
from dataclasses import asdict
from fractions import Fraction as F

import numpy as np
import pytest
from adjacent_triangle_cases import NAMES, authored, hinge_path, mesh, row_for, vertex_path

import phymeo as ph


def detect(model, paths):
    with ph.CollisionWorkspace(model) as work:
        result = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert work.last_result is result
        return result


def sparse(weights):
    return {node: w for node, w in weights if w != 0}


def shared_only(w, n):
    assert w.gap_m == 0 and w.normal is None and not w.intersection_points
    assert w.first_point is None and w.second_point is None
    assert not w.first_weights and not w.second_weights
    assert len(w.shared_material_points) == n
    for pt in w.shared_material_points:
        assert sparse(pt.first_weights) == sparse(pt.second_weights)


def check_remote(w):
    assert w.normal is None and w.gap_m == 0 and w.intersection_points
    assert sparse(w.first_weights) != sparse(w.second_weights)
    for p in w.intersection_points:
        assert sparse(p.first_weights) != sparse(p.second_weights)
    for p in w.shared_material_points:
        assert sparse(p.first_weights) == sparse(p.second_weights)


def save(tmp_path, name, model, paths, result):
    (tmp_path / "observation.json").write_text(
        json.dumps(
            dict(
                name=name,
                model=asdict(model),
                paths=[
                    dict(
                        key=asdict(s.key),
                        start=paths[s.key].start.tolist(),
                        end=None if paths[s.key].end is None else paths[s.key].end.tolist(),
                    )
                    for s in model.shapes
                ],
                result=asdict(result),
            ),
            allow_nan=False,
        )
    )


@pytest.mark.parametrize("name", NAMES)
def test_catalog_same_body_material_identity(name, tmp_path):
    model, paths, keys = authored(name)
    out = detect(model, paths)
    row = row_for(out, keys)
    assert row.candidate and row.family == "DD" and row.first.body == row.second.body
    if name in ("flat", "hinge"):
        assert row.status == "adjacent" and row.reason == "shared_material_only"
        shared_only(row.witness, 1 if name == "flat" else 2)
    else:
        assert row.status == "touching" and row.reason == "distinct_material_intersection"
        w = row.witness
        check_remote(w)
        assert len(w.shared_material_points) == len(w.intersection_points) == 1
        assert w.shared_material_points[0].point == (0, 0, 0)
        if name == "holdout":
            xyz = (0.625, 0.25, 0)
            wa = (0.5625, 0.3125, 0.125)
            wb = (0.5, 0.375, 0.125)
        else:
            xyz = (0.5, 0.25, 0)
            wa = (0.625, 0.25, 0.125)
            wb = (0.5, 0.25, 0.25)
        assert w.first_point == w.second_point == xyz
        assert dict(w.first_weights) == dict(zip((0, 1, 2), wa, strict=True))
        assert dict(w.second_weights) == dict(zip((0, 3, 4), wb, strict=True))
        assert w.intersection_points[0].point == tuple(2 * v for v in xyz)
    save(tmp_path, name, model, paths, out)


def test_shared_edge_fold_can_cross_its_plane_legally(tmp_path):
    model, paths = hinge_path(False)
    out = detect(model, paths)
    (row,) = out.pairs
    assert row.candidate and row.status == "adjacent" and row.reason == "shared_material_only"
    assert row.sweep.first_t is None and row.sweep.witness is None
    assert row.sweep.start_gap_m == row.sweep.end_gap_m == 0
    shared_only(row.witness, 2)
    save(tmp_path, "moving_hinge_clear", model, paths, out)


def test_shared_edge_isolated_interior_overlap_at_one_third(tmp_path):
    model, paths = hinge_path(True)
    out = detect(model, paths)
    (row,) = out.pairs
    assert row.status == "hit" and row.reason == "first_contact"
    lo, hi = map(F, row.sweep.first_t)
    assert lo <= F(1, 3) <= hi and hi - lo <= F(1, 2**40)
    assert row.sweep.start_gap_m == row.sweep.end_gap_m == 0
    shared_only(row.sweep.witness, 2)
    near = row.sweep.feature_witness
    assert near is not None and sparse(near.first_weights) != sparse(near.second_weights)
    assert near.first_point[1] > 0.2 and near.second_point[1] > 0.2
    assert np.linalg.norm(np.subtract(near.first_point, near.second_point)) == pytest.approx(
        near.gap_m, abs=1e-15
    )
    assert 0 < near.gap_m < 2e-12 and not near.intersection_points
    save(tmp_path, "moving_hinge_hit", model, paths, out)


def test_incident_edge_to_opposite_edge_is_not_removed(tmp_path):
    model, paths = vertex_path()
    out = detect(model, paths)
    (row,) = out.pairs
    assert row.status == "hit" and row.sweep.first_t == (0.5, 0.5)
    w = row.sweep.witness
    check_remote(w)
    assert {p.point for p in w.shared_material_points} == {(0, 0, 0)}
    assert {p.point for p in w.intersection_points} == {(1, 0, 0)}
    assert w.first_point == w.second_point == (0.5, 0, 0)
    assert dict(w.first_weights) == {0: 0.875, 1: 0.125, 2: 0}
    assert dict(w.second_weights) == {0: 0.5, 3: 0.25, 4: 0.25}
    assert row.sweep.feature_witness is None
    left = {}
    right = {}
    for k, p in paths.items():
        mid = p.start + (p.end - p.start) / 4
        left[k] = ph.CollisionMotion(p.start, mid)
        right[k] = ph.CollisionMotion(mid, p.end)
    (left_row,) = detect(model, left).pairs
    assert left_row.status == "adjacent" and left_row.sweep.first_t is None
    (right_row,) = detect(model, right).pairs
    a, b = map(F, right_row.sweep.first_t)
    assert a <= F(1, 3) <= b
    save(tmp_path, "vertex_birth", model, paths, out)


def test_existing_remote_contact_is_at_zero_even_when_it_unloads():
    start = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [1, 0.5, 0.5], [1, 0.5, -0.5]])
    end = start.copy()
    end[3:, 2] += 0.75
    model, paths = mesh(start, end, [(0, 1, 2), (0, 3, 4)])
    (row,) = detect(model, paths).pairs
    assert row.sweep.first_t == (0, 0) and row.status == "hit"
    check_remote(row.sweep.witness)
    shared_only(row.witness, 1)


def test_split_ids_and_different_bodies_do_not_exempt_a_world_point():
    model, paths, _ = authored("flat")
    a = model.shapes[0]
    b = model.shapes[2]
    for target, nodes in ((b.key, (8, 3, 4)), (ph.CollisionKey(0, 2, 2), b.vertex_ids)):
        other = ph.CollisionShape(target, "triangle", nodes)
        pair = ph.CollisionModel([a, other])
        (row,) = detect(pair, {a.key: paths[a.key], target: paths[b.key]}).pairs
        assert row.candidate and row.status == "touching"
        assert len(row.witness.intersection_points) == 1 and not row.witness.shared_material_points
        assert row.witness.first_point == row.witness.second_point == (0.0, 0.0, 0.0)


def test_tiny_but_distinct_material_area_is_not_exempt():
    for y, expected in [(2**-46, "touching"), (-(2**-46), "adjacent")]:
        model, paths = mesh(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.25, y, 0]], None, [(0, 1, 2), (1, 0, 3)]
        )
        (row,) = detect(model, paths).pairs
        assert row.status == expected and row.candidate
        if expected == "touching":
            check_remote(row.witness)
            assert row.witness.first_point[1] > 0


def test_invalid_path_and_explicit_filter_have_separate_outcomes():
    model, paths = mesh(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]],
        [(0, 1, 2), (1, 0, 3)],
    )
    (row,) = detect(model, paths).pairs
    assert row.status == "unresolved" and row.reason == "triangle_degenerate_on_path"
    excluded = ph.CollisionModel(model.shapes, [(s0.key, s1.key) for s0, s1 in [model.shapes]])
    (row,) = detect(excluded, paths).pairs
    assert row.status == "filtered" and row.reason == "excluded" and not row.candidate


def test_full_shared_simplex_has_no_remote_event():
    start = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    end = start.copy()
    end[2, 2] = 1
    model, paths = mesh(start, end, [(0, 1, 2), (2, 1, 0)])
    (row,) = detect(model, paths).pairs
    assert row.status == "adjacent" and row.sweep.first_t is None
    shared_only(row.witness, 3)
