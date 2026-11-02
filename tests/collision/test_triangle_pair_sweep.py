"""DD01/DD02 through detect. No event pose or feature is supplied to the engine."""

import json
from dataclasses import asdict
from fractions import Fraction as F

import numpy as np
import pytest
from triangle_pair_cases import VARIANTS, authored, graze, piercing, scene, transit

import phymeo as ph


def query(model, paths, margin=0.0):
    with ph.CollisionWorkspace(model) as work:
        out = ph.detect(model, paths, ph.CollisionPolicy(margin=margin), work)
        assert work.last_result is out
        return out.pairs[0]


def hit(row, t):
    assert row.family == "DD" and row.candidate and row.status == "hit"
    lo, hi = map(F, row.sweep.first_t)
    assert lo <= t <= hi and hi - lo <= F(1, 2**40)
    assert float(lo) <= row.sweep.witness_time <= float(hi)
    return row.sweep.witness


def save(tmp_path, case, variant, model, paths, row):
    data = {
        "case": case,
        "variant": variant,
        "model": asdict(model),
        "paths": [
            {
                "key": asdict(s.key),
                "start": paths[s.key].start.tolist(),
                "end": None if paths[s.key].end is None else paths[s.key].end.tolist(),
            }
            for s in model.shapes
        ],
        "pair": asdict(row),
    }
    (tmp_path / "observation.json").write_text(json.dumps(data, allow_nan=False))


@pytest.mark.parametrize(("case", "name"), [(c, n) for c, names in VARIANTS.items() for n in names])
def test_catalog_times_and_both_material_owners(case, name, tmp_path):
    model, paths = authored(case, name)
    if name == "clear":
        assert query(model, paths).reason == "aabb"
        row = query(model, paths, 3.0)
        assert row.candidate and row.status == "separated" and row.sweep.first_t is None
        if case == "DD01":
            assert row.witness.gap_m >= 2
        else:
            assert row.witness.gap_m == pytest.approx(0.05)
    else:
        row = query(model, paths)
        expected_t = F(1, 2)
        if name == "tip_holdout":
            expected_t = F(2, 3)
        if name == "holdout":
            expected_t = F(0.3) / (F(0.3) + F(0.2))
        w = hit(row, expected_t)
        point = (0.125, 0.375, 0) if name == "tip_holdout" else (0, 0, 0)
        np.testing.assert_allclose(w.first_point, point, atol=2e-11)
        np.testing.assert_allclose(w.second_point, point, atol=2e-11)
        if case == "DD01":
            face, tip = (
                (w.second_weights, w.first_weights)
                if name == "swapped"
                else (w.first_weights, w.second_weights)
            )
            weights = (F(23, 96), F(29, 96), F(11, 24)) if name == "tip_holdout" else (F(1, 3),) * 3
            assert dict(face) == pytest.approx(
                dict(zip((11, 7, 3), map(float, weights), strict=True)), abs=1e-10
            )
            tip_id = 11 if name in ("reused_ids", "swapped") else 31
            assert dict(tip)[tip_id] == pytest.approx(1.0, abs=1e-10)
            assert row.first.body != row.second.body
        else:
            first, second = (
                ((31, 41, 53), (11, 7, 3)) if name == "swap_reverse" else ((11, 7, 3), (31, 41, 53))
            )
            for ids, ws in ((first, w.first_weights), (second, w.second_weights)):
                assert dict(ws) == pytest.approx(
                    dict(zip(ids, (0.5, 0.5, 0), strict=True)), abs=1e-10
                )
        if expected_t == F(1, 2):
            assert w.normal is None and len(w.intersection_points) == 1
    save(tmp_path, case, name, model, paths, row)


def test_both_ends_clear_and_global_substep_interval(tmp_path):
    model, paths = transit()
    row = query(model, paths)
    w = hit(row, F(1, 4))
    assert row.sweep.start_gap_m == 0.5 and row.witness.gap_m == 0.25
    assert len(w.intersection_points) == 1
    left = {}
    right = {}
    for k, p in paths.items():
        end = p.start if p.end is None else p.end
        mid = p.start + (end - p.start) / 8
        left[k] = ph.CollisionMotion(p.start, mid)
        right[k] = ph.CollisionMotion(mid, end)
    assert query(model, left, margin=2).status == "separated"
    hit(query(model, right), F(1, 7))
    save(tmp_path, "extra", "transit", model, paths, row)


def test_isolated_edge_edge_graze_between_time_samples(tmp_path):
    model, paths = graze()
    row = query(model, paths)
    w = hit(row, F(1, 3))
    assert row.sweep.start_gap_m > 0 and row.witness.gap_m > 0
    np.testing.assert_allclose(w.first_point, [0, 0, 0], atol=2e-11)
    assert dict(w.first_weights) == pytest.approx({11: 0.5, 7: 0.5, 3: 0}, abs=1e-10)
    assert dict(w.second_weights) == pytest.approx({31: 0.5, 41: 0.5, 53: 0}, abs=1e-10)
    save(tmp_path, "extra", "graze", model, paths, row)
    model, paths = graze(lower=-(2**-30))
    no = query(model, paths, margin=0.1)
    assert no.candidate and no.status == "separated"


def test_irrational_first_of_two_edge_roots():
    model, paths = graze(lower=1 / 32)
    row = query(model, paths)
    assert row.status == "hit"
    lo, hi = map(F, row.sweep.first_t)
    assert lo <= hi < F(1, 3) and hi - lo <= F(1, 2**40)
    assert 2 * (3 * lo - 1) ** 2 >= 1 >= 2 * (3 * hi - 1) ** 2


def test_initial_transverse_intersection_with_no_vf_or_ee_onset(tmp_path):
    model, paths = piercing()
    static = {k: ph.CollisionMotion(p.start) for k, p in paths.items()}
    initial = query(model, static)
    assert initial.status == "touching" and initial.witness.gap_m == 0
    w = hit(query(model, paths), F(0))
    pts = {tuple(p.point) for p in w.intersection_points}
    assert pts == {(0.0, -0.5, 0.0), (0.0, 0.0, 0.0)} and w.normal is None
    for p in w.intersection_points:
        for k, address in ((initial.first, p.first_weights), (initial.second, p.second_weights)):
            shape = next(s for s in model.shapes if s.key == k)
            weights = [dict(address)[i] for i in shape.vertex_ids]
            np.testing.assert_allclose(np.array(weights) @ paths[k].start, p.point, atol=1e-12)
    save(tmp_path, "extra", "initial_pierce", model, paths, query(model, paths))


def test_coplanar_hexagon_and_containment_are_material_intersections():
    a = np.array([[-2, -1, 0], [2, -1, 0], [0, 2, 0]])
    b = np.array([[-2, 1, 0], [2, 1, 0], [0, -2, 0]])
    model, paths = scene(a, None, b, None)
    w = query(model, paths).witness
    expected = [
        (-4 / 3, 0, 0),
        (-2 / 3, -1, 0),
        (-2 / 3, 1, 0),
        (2 / 3, -1, 0),
        (2 / 3, 1, 0),
        (4 / 3, 0, 0),
    ]
    assert w.normal is None
    np.testing.assert_allclose(
        sorted(tuple(p.point) for p in w.intersection_points), expected, atol=1e-12
    )
    model, paths = scene(a, None, a / 4, None)
    assert len(query(model, paths).witness.intersection_points) == 3


def test_coplanar_sliding_first_shared_line():
    a = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    b = np.array([[-1, 0.25, 0], [-1, 0.75, 0], [-2, 0.5, 0]])
    model, paths = scene(a, None, b, b + [2, 0, 0])
    w = hit(query(model, paths), F(1, 2))
    assert {p.point for p in w.intersection_points} == {(0.0, 0.25, 0.0), (0.0, 0.75, 0.0)}


def test_edges_become_parallel_without_contact():
    a = [[-1, 0, 0], [1, 0, 0], [0, -1, 0]]
    b = np.array([[-1, 1, 0], [1, 1, 0.5], [0, 2, 0]])
    end = b.copy()
    end[1, 2] = -0.5
    model, paths = scene(a, None, b, end)
    row = query(model, paths, margin=2.0)
    assert row.candidate and row.status == "separated" and row.sweep.first_t is None
    assert row.sweep.start_gap_m >= 1 and row.witness.gap_m >= 1


def test_degenerate_thick_and_partial_material_overlap_are_distinguished():
    a = [[-1, 0, 0], [1, 0, 0], [0, 1, 0]]
    b = [[-1, 0, 0.25], [1, 0, 0.25], [0, 1, 0.25]]
    model, paths = scene(a, [[-1, 0, 0], [1, 0, 0], [0, -1, 0]], b, [[x, y, -z] for x, y, z in b])
    row = query(model, paths)
    assert (
        row.candidate and row.status == "unresolved" and row.reason == "triangle_degenerate_on_path"
    )
    first, second = model.shapes
    thick = ph.CollisionModel(
        [ph.CollisionShape(first.key, "triangle", first.vertex_ids, 0.01), second]
    )
    assert query(thick, paths).status == "unsupported"
    other = ph.CollisionKey(0, 1, 1)
    same = ph.CollisionModel([first, ph.CollisionShape(other, "triangle", (11, 19, 20))])
    unchanged = ph.CollisionMotion(a)
    same_paths = {first.key: unchanged, other: ph.CollisionMotion([a[0], [0, 1, 0], [0, 0, 1]])}
    row = query(same, same_paths)
    # Their first node has the same material ID. The other endpoint of the
    # coincident edge has ID 3 on the first face and ID 19 on the second.
    assert (
        row.candidate
        and row.status == "touching"
        and row.reason == "distinct_material_intersection"
    )
    (shared,) = row.witness.shared_material_points
    (remote,) = row.witness.intersection_points
    assert shared.point == (-1.0, 0.0, 0.0) and remote.point == (0.0, 1.0, 0.0)
    assert dict(remote.first_weights)[3] == dict(remote.second_weights)[19] == 1.0
    assert row.witness.first_point == row.witness.second_point == (-0.5, 0.5, 0.0)
