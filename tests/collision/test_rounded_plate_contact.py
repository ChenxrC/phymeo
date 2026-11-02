"""Finite face, back, rim and corner queries through the public RD dispatch."""

import numpy as np
import pytest
from rounded_plate_cases import NAMES, authored, edge_case

import phymeo as ph


def rows(model, paths):
    with ph.CollisionWorkspace(model) as work:
        return [
            p
            for p in ph.detect(model, paths, ph.CollisionPolicy(margin=0.5), work).pairs
            if p.family == "RD"
        ]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("reverse", [False, True])
def test_registered_plate_cases(name, reverse):
    model, paths, expected = authored(name, reverse)
    pairs = rows(model, paths)
    if name in ("base", "corner_holdout"):
        wanted = expected.get("gaps_m", [expected.get("gap_m")])
        features = expected.get("features", [expected.get("feature")])
        assert [p.plate.end.feature for p in pairs] == features
        assert [p.witness.gap_m for p in pairs] == pytest.approx(wanted, abs=1e-12)
        assert all(p.candidate and p.sweep is None for p in pairs)
        if name == "base":
            dirs = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [2**-0.5, 2**-0.5, 0]])
            np.testing.assert_allclose(
                [p.witness.normal for p in pairs], -dirs if reverse else dirs, atol=1e-12
            )
        else:
            assert pairs[0].status == "overlap"
        return
    (row,) = pairs
    assert row.candidate
    if expected["first_t"] is None:
        assert row.status == "separated" and row.sweep.first_t is None
        return
    assert row.status == "hit"
    lo, hi = row.sweep.first_t
    assert lo <= expected["first_t"] <= hi and hi - lo < 1e-8
    assert row.plate.entry.feature == "front"
    if name == "large_face":
        patch = row.plate.entry.patch
        assert patch.dimension == 2 and patch.area_m2 == pytest.approx(1)
        assert len(patch.points) == 4
        for pt in patch.points:
            np.testing.assert_allclose(pt.first_point, pt.second_point, atol=1e-9)
            assert abs(sum(w for _, w in pt.first_weights) - 1) < 1e-12
            assert abs(sum(w for _, w in pt.second_weights) - 1) < 1e-12
    else:
        key = next(s.key for s in model.shapes if s.kind == "particle")
        middle = dict(paths)
        middle[key] = ph.CollisionMotion((paths[key].start + paths[key].end) / 2)
        (mid,) = rows(model, middle)
        assert mid.status == "overlap" and mid.witness.gap_m == pytest.approx(expected["mid_gap_m"])
        assert mid.witness.normal is None


def test_back_face_subdivision_and_nonconvex_shape():
    model, paths, _ = authored("large_face")
    cloth = next(s.key for s in model.shapes if s.kind == "triangle")
    start, end = paths[cloth].start.copy(), paths[cloth].end.copy()
    start[:, 2] *= -1
    end[:, 2] *= -1
    p0 = dict(paths)
    p0[cloth] = ph.CollisionMotion(start, (start + end) / 2)
    assert rows(model, p0)[0].status == "separated"
    p1 = dict(paths)
    p1[cloth] = ph.CollisionMotion((start + end) / 2, end)
    row = rows(model, p1)[0]
    assert row.plate.entry.feature == "back" and row.plate.entry.patch.area_m2 == pytest.approx(1)
    lo, hi = row.sweep.first_t
    assert (1 + lo) / 2 <= 2 / 3 <= (1 + hi) / 2
    bad = dict(paths)
    pk = next(s.key for s in model.shapes if s.kind == "plate")
    bad[pk] = ph.CollisionMotion(paths[pk].start[[0, 2, 1, 3]])
    assert rows(model, bad)[0].status == "invalid_geometry"


@pytest.mark.parametrize("parallel", [False, True])
def test_edge_interior_without_a_cloth_vertex_hit(parallel):
    model, paths = edge_case(parallel=parallel)
    (row,) = rows(model, paths)
    if parallel:
        assert row.status == "touching" and row.plate.end.feature == "rim"
        p = row.plate.end.patch
        assert p.dimension == 1 and p.area_m2 == 0
        assert sorted(x.first_point[1] for x in p.points) == [-0.5, 0.5]
    else:
        assert row.status == "hit" and row.sweep.first_t == (0.75, 0.75)
        p = row.plate.entry.patch
        assert row.plate.entry.feature == "rim" and p.dimension == 0
        assert p.points[0].second_weights == ((70, 0.5), (71, 0.5))
        np.testing.assert_allclose(p.points[0].first_point, [0.625, 0, 0])


def test_degenerate_triangle_between_valid_endpoints_is_unresolved():
    model, paths, _ = authored("large_face")
    cloth = next(s.key for s in model.shapes if s.kind == "triangle")
    start = paths[cloth].start
    paths[cloth] = ph.CollisionMotion(start, start[[0, 2, 1]])
    assert rows(model, paths)[0].status == "unresolved"
