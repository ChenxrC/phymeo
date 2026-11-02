"""Independent first-time and material-address tests at the public collision entry."""

import json
from dataclasses import asdict
from fractions import Fraction

import numpy as np
import pytest
from sphere_face_cases import IDS, NAMES, authored, deforming, scene

import phymeo as ph


def query(model, paths, margin=0.0):
    with ph.CollisionWorkspace(model) as work:
        out = ph.detect(model, paths, ph.CollisionPolicy(margin=margin), work)
        assert work.last_result is out
        return out.pairs[0]


def hit(row, first_t):
    assert row.family == "RD" and row.candidate and row.status == "hit"
    lo, hi = row.sweep.first_t
    assert Fraction(lo) <= Fraction(first_t) <= Fraction(hi)
    assert (hi - lo) <= 2**-40
    assert lo <= row.sweep.witness_time <= hi
    return row.sweep.witness


def save(tmp_path, name, model, paths, row):
    data = {
        "case": name,
        "model": asdict(model),
        "motions": [
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


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("reverse", [False, True])
def test_registered_whole_face_events(name, reverse, tmp_path):
    model, paths, face, ball = authored(name, reverse)
    if name == "clear":
        coarse = query(model, paths)
        assert not coarse.candidate and coarse.reason == "aabb"
        row = query(model, paths, margin=3)
        assert row.candidate and row.status == "separated" and row.sweep.first_t is None
        assert row.sweep.start_gap_m > 1.75 and row.witness.gap_m > 1.75
    else:
        row = query(model, paths)
        w = hit(row, Fraction(1, 4))
        weights = w.second_weights if reverse else w.first_weights
        expected = (0.25, 0.25, 0.5) if "holdout" in name else (1 / 3, 1 / 3, 1 / 3)
        assert dict(weights) == pytest.approx(dict(zip(IDS, expected, strict=True)), abs=1e-12)
        p = w.second_point if reverse else w.first_point
        np.testing.assert_allclose(p, [0, 0.5 if "holdout" in name else 0.0, 0], atol=1e-12)
        np.testing.assert_allclose(w.normal, [0, 0, -1 if reverse else 1], atol=1e-12)
        assert row.sweep.start_gap_m == row.witness.gap_m == 0.25
        # xy distance alone excludes every vertex over the original full path.
        assert (
            min(np.linalg.norm(v[:2] - paths[ball].start[0, :2]) for v in paths[face].start) > 0.25
        )
    save(tmp_path, name, model, paths, row)


def test_independent_affine_vertices_hit_at_the_interior(tmp_path):
    model, paths, face, _ = deforming()
    row = query(model, paths)
    w = hit(row, Fraction(1, 2))
    assert dict(w.first_weights) == pytest.approx(dict.fromkeys(IDS, 1 / 3), abs=1e-12)
    np.testing.assert_allclose(w.first_point, [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(w.normal, [0, 0, 1], atol=1e-12)
    assert len(np.unique(paths[face].end - paths[face].start, axis=0)) == 3
    save(tmp_path, "independent_affine", model, paths, row)


def test_changing_normal_material_and_substep_consistency():
    tri = np.array([[-2, -1, 0], [2, -1, 0], [0, 2, 0]], float)
    end = tri.copy()
    end[:, 2] = 1.5 * tri[:, 0]
    model, paths, face, ball = scene(tri, end, [0, 0, 13 / 16], [0, 0, -3 / 16])
    row = query(model, paths)
    w = hit(row, Fraction(1, 2))
    np.testing.assert_allclose(w.first_point, [3 / 20, 0, 9 / 80], atol=1e-12)
    np.testing.assert_allclose(w.normal, [-0.6, 0, 0.8], atol=1e-12)
    assert dict(w.first_weights) == pytest.approx(
        dict(zip(IDS, [71 / 240, 89 / 240, 1 / 3], strict=True))
    )
    first = {}
    last = {}
    for k in (face, ball):
        p = paths[k]
        mid = p.start + (p.end - p.start) / 4
        first[k] = ph.CollisionMotion(p.start, mid)
        last[k] = ph.CollisionMotion(mid, p.end)
    assert query(model, first, margin=1).status == "separated"
    hit(query(model, last), Fraction(1, 3))


def test_edge_graze_and_one_ulp_miss():
    tri = [[-1, 0, 0], [1, 0, 0], [0, 2, 0]]
    for y, valid in ((-0.25, True), (np.nextafter(-0.25, -np.inf), False)):
        model, paths, _, _ = scene(tri, None, [0, y, 1], [0, y, -1])
        row = query(model, paths, margin=0.125)
        if valid:
            w = hit(row, Fraction(1, 2))
            assert dict(w.first_weights) == dict(zip(IDS, [0.5, 0.5, 0], strict=True))
        else:
            assert row.candidate and row.status == "separated" and row.sweep.first_t is None


def test_vertex_tangent_isolated_between_device_sample_times():
    tri = [[-1, 0, 0], [1, 0, 0], [0, 2, 0]]
    model, paths, _, _ = scene(
        tri, None, [-19 / 16, -1 / 4, 0.5], [-19 / 16, -1 / 4, -1.0], radius=5 / 16
    )
    w = hit(query(model, paths), Fraction(1, 3))
    assert dict(w.first_weights) == dict(zip(IDS, [1, 0, 0], strict=True))
    np.testing.assert_allclose(w.first_point, [-1, 0, 0], atol=1e-12)


def test_identically_tangent_face_enters_on_projected_edge():
    model, paths, _, _ = scene(
        [[0, 0, 0], [2, 0, 0], [0, 2, 0]], None, [-1, 0.5, 0.25], [1, 0.5, 0.25]
    )
    w = hit(query(model, paths), Fraction(1, 2))
    assert dict(w.first_weights) == dict(zip(IDS, [0.75, 0, 0.25], strict=True))


def test_path_collapse_is_unresolved_despite_nondegenerate_endpoints():
    model, paths, _, _ = scene(
        [[-1, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[-1, 0, 0], [1, 0, 0], [0, -1, 0]],
        [0, 0, 0.4],
        [0, 0, -0.4],
    )
    row = query(model, paths)
    assert (
        row.candidate and row.status == "unresolved" and row.reason == "triangle_degenerate_on_path"
    )
    assert row.sweep is None and row.witness is None


def test_back_side_initial_overlap_and_common_frame():
    model, paths, _, _ = scene(
        [[-2, -1, 0], [2, -1, 0], [0, 2, 0]], None, [0, 0, -0.125], [0, 0, -1]
    )
    w = hit(query(model, paths), Fraction(0))
    assert w.gap_m == -0.125
    np.testing.assert_allclose(w.normal, [0, 0, -1])
    model, paths, _, _ = authored("holdout")
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    vec = np.array([4, -3, 2])
    # Move the whole geometry one metre during the step in the new z direction.
    converted = {}
    for k, p in paths.items():
        start = p.start @ rotation.T + vec
        end = (p.start if p.end is None else p.end) @ rotation.T + vec + [0, 0, 1]
        converted[k] = ph.CollisionMotion(start, end)
    w = hit(query(model, converted), Fraction(1, 4))
    np.testing.assert_allclose(w.first_point, [4, -3, 2.75], atol=1e-12)
    np.testing.assert_allclose(w.normal, [1, 0, 0])


def test_irrational_first_touch_on_a_tilting_face(tmp_path):
    tri = np.array([[-2, -1, 0], [2, -1, 0], [0, 2, 0]], float)
    end = tri.copy()
    end[:, 2] = 2 * tri[:, 0]
    # The centered ball is 3/8 above the origin. Its distance to z=2 t x
    # reaches R=1/4 when 16 t²=5; the projected point is strictly inside.
    model, paths, _, _ = scene(tri, end, [0, 0, 3 / 8], [0, 0, 3 / 8])
    row = query(model, paths)
    assert row.status == "hit" and row.sweep.start_gap_m == 1 / 8
    lo, hi = map(Fraction, row.sweep.first_t)
    assert 16 * lo**2 <= 5 <= 16 * hi**2 and hi - lo <= Fraction(1, 2**40)
    assert all(w > 0 for _, w in row.sweep.witness.first_weights)
    save(tmp_path, "irrational_tilt", model, paths, row)


def test_curved_ball_and_finite_face_thickness_keep_unsupported_candidates():
    model, paths, face, ball = authored("base")
    turn = ph.RigidTurn((0, 0, 0), (0, 0, 0), (0, 1, 0), np.pi)
    curved = {**paths, ball: ph.CollisionMotion(paths[ball].start, turn=turn)}
    altered = ph.CollisionModel(
        [
            ph.CollisionShape(face, "triangle", IDS, 0.125),
            ph.CollisionShape(ball, "sphere", (19,), 0.25),
        ]
    )
    for m, p in ((model, curved), (altered, paths)):
        row = query(m, p)
        assert row.candidate and row.status == "unsupported" and row.sweep is None
