"""Test M2 at the public query, independently of motion solvers."""

import json
from dataclasses import asdict
from fractions import Fraction

import numpy as np
import pytest
from round_cases import RR01, RR02, catalog_scene, crossing_capsules

import phymeo as ph


def detect_one(model, paths, policy=ph.CollisionPolicy()):
    with ph.CollisionWorkspace(model) as work:
        result = ph.detect(model, paths, policy, work)
        assert work.last_result is result
        return result


def at(row, expected):
    interval = row.sweep.first_t
    assert interval is not None and interval[0] <= interval[1]
    assert Fraction(interval[0]) <= expected <= Fraction(interval[1])
    assert interval[1] - interval[0] <= 1e-9
    assert interval[0] <= row.sweep.witness_time <= interval[1]


@pytest.mark.parametrize("name", RR01)
@pytest.mark.parametrize("reverse", [False, True])
def test_registered_ball_events(name, reverse, tmp_path):
    model, paths, v, g = catalog_scene("RR01", name, reverse)
    out = detect_one(model, paths)
    (row,) = out.pairs
    if name == "filtered":
        assert row.status == "filtered" and row.reason == "excluded" and row.sweep is None
    else:
        assert row.candidate and row.family == "RR"
        ends = v["expected"]
        if "start_gap" in ends:
            assert row.sweep.start_gap_m == pytest.approx(ends["start_gap"], abs=1e-12)
        if "end_gap" in ends:
            assert row.witness.gap_m == pytest.approx(ends["end_gap"], abs=1e-12)
        if name == "clear":
            assert row.status == "separated" and row.sweep.first_t is None
        else:
            # Use the exact binary input for the base, whose idealized decimal time is 1/2.
            if name == "base":
                c0, c1 = g["B"]["center_start"][2], g["B"]["center_end"][2]
                root = (Fraction(c0) - Fraction(g["B"]["radius"])) / (Fraction(c0) - Fraction(c1))
            elif name == "sphere_pair":
                root = (Fraction(0.5) - 2 * Fraction(0.1)) / (Fraction(0.5) - Fraction(0.1))
            else:
                root = Fraction(2, 3)
            assert row.status == "hit"
            at(row, root)
            assert row.sweep.witness.gap_m == pytest.approx(0, abs=1e-9)
            if g["A"]["type"] == "plane":
                assert out.bounds[1 if reverse else 0].lower is None
                witness = row.sweep.witness
                normal = np.asarray(g["A"]["normal"])
                np.testing.assert_allclose(
                    witness.normal, -normal if reverse else normal, atol=1e-12
                )
                point = witness.second_point if reverse else witness.first_point
                offset = witness.second_offset if reverse else witness.first_offset
                np.testing.assert_allclose(
                    np.array(point) - normal * g["A"]["offset"], offset, atol=1e-12
                )
                if "point_m" in ends:
                    np.testing.assert_allclose(point, ends["point_m"], atol=1e-12)
    (tmp_path / "observation.json").write_text(
        json.dumps(
            {
                "case": "RR01",
                "variant": name,
                "reverse": reverse,
                "geometry": g,
                "truth": v["expected"],
                "result": asdict(out),
            },
            allow_nan=False,
        )
    )


@pytest.mark.parametrize("name", RR02)
def test_registered_capsule_material_features(name, tmp_path):
    model, paths, v, g = catalog_scene("RR02", name)
    # Margin captures the separated controls; it is never added to the reported physical radii.
    (row,) = detect_one(model, paths, ph.CollisionPolicy(margin=0.25)).pairs
    gap = v["expected"]["gap_m"]
    assert row.status == ("overlap" if gap < 0 else "touching" if gap == 0 else "separated")
    w = row.witness
    assert w.gap_m == pytest.approx(gap, abs=1e-12)
    if name in ("base", "tangent", "overlap"):
        np.testing.assert_allclose(w.axis_parameter_span, [(0.5, 0.5), (0.5, 0.5)], atol=1e-12)
    elif name == "endcaps":
        np.testing.assert_allclose(w.axis_parameter_span, [(1, 0), (1, 0)], atol=1e-12)
    elif name == "parallel_holdout":
        np.testing.assert_allclose(w.axis_parameter_span, [(0.25, 0), (0.75, 1)], atol=1e-12)
    elif name == "coincident":
        assert w.normal is None
        np.testing.assert_allclose(w.axis_parameter_span, [(0, 0), (1, 1)], atol=1e-12)
    (tmp_path / "observation.json").write_text(
        json.dumps(
            {
                "case": "RR02",
                "variant": name,
                "geometry": g,
                "truth": v["expected"],
                "pair": asdict(row),
            },
            allow_nan=False,
        )
    )


def test_translating_axes_cross_between_two_clear_endpoints_and_split():
    model, paths = crossing_capsules()
    (row,) = detect_one(model, paths).pairs
    assert row.status == "hit" and row.sweep.start_gap_m == row.witness.gap_m == 0.75
    at(row, Fraction(3, 8))
    np.testing.assert_allclose(row.sweep.witness.axis_parameter_span, [(0.5, 0.5), (0.5, 0.5)])
    k = max(paths)
    p = paths[k]
    mid = p.start + (p.end - p.start) / 4
    left = {**paths, k: ph.CollisionMotion(p.start, mid)}
    right = {**paths, k: ph.CollisionMotion(mid, p.end)}
    assert detect_one(model, left).pairs[0].status == "separated"
    at(detect_one(model, right).pairs[0], Fraction(1, 6))


def test_capsule_endcap_graze_and_one_ulp_miss():
    a, b = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
    model = ph.CollisionModel(
        [
            ph.CollisionShape(a, "capsule", (4, 7), 0.125),
            ph.CollisionShape(b, "sphere", (1,), 0.125),
        ]
    )
    for x, hit in [(1.25, True), (np.nextafter(1.25, np.inf), False)]:
        paths = {
            a: ph.CollisionMotion([[0, 0, 0], [1, 0, 0]]),
            b: ph.CollisionMotion([[x, -1, 0]], [[x, 1, 0]]),
        }
        (row,) = detect_one(model, paths).pairs
        assert row.candidate
        if hit:
            at(row, Fraction(1, 2))
            assert dict(row.sweep.witness.first_weights) == {4: 0, 7: 1}
        else:
            assert row.status == "separated" and row.sweep.first_t is None


def test_moving_halfspace_and_initial_contact():
    plane, ball = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
    model = ph.CollisionModel(
        [
            ph.CollisionShape(plane, "plane", (3,), normal=(3, 4, 0)),
            ph.CollisionShape(ball, "sphere", (9,), 0.5),
        ]
    )
    # n·(center-anchor) starts at 5 m in unnormalized units, radius |n| = 2.5.
    paths = {
        plane: ph.CollisionMotion([[0, 0, 0]], [[0, 1.25, 0]]),
        ball: ph.CollisionMotion([[3, -1, 1000]]),
    }
    (row,) = detect_one(model, paths).pairs
    at(row, Fraction(1, 2))
    assert row.witness.gap_m == pytest.approx(-0.5)
    paths[ball] = ph.CollisionMotion([[0, 0, 1000]], [[6, 8, 1000]])
    (row,) = detect_one(model, paths).pairs
    at(row, Fraction(0))
    assert row.sweep.start_gap_m < 0 and row.witness.gap_m > 0


def test_plane_registration_rejects_invalid_normal_and_curved_sweep_stays_explicit():
    k, j = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
    for normal in ((0, 0, 0), (np.nan, 1, 0)):
        with pytest.raises(ValueError):
            ph.CollisionShape(k, "plane", (7,), normal=normal)
    with pytest.raises(ValueError):
        ph.CollisionShape(k, "sphere", (7,), 0.5, normal=(0, 0, 1))
    model = ph.CollisionModel(
        [
            ph.CollisionShape(k, "plane", (7,), normal=(0, 0, 1)),
            ph.CollisionShape(j, "sphere", (0,), 0.2),
        ]
    )
    paths = {
        k: ph.CollisionMotion([[0, 0, 0]], turn=ph.RigidTurn((0, 0, 0), (0, 0, 0), (1, 0, 0), 1)),
        j: ph.CollisionMotion([[0, 0, 1]]),
    }
    (row,) = detect_one(model, paths).pairs
    assert row.candidate and row.status == "unsupported" and row.sweep is None


def test_opposite_parallel_parameter_span_follows_both_axes():
    a, b = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
    model = ph.CollisionModel([ph.CollisionShape(k, "capsule", (11, 2), 0.125) for k in (a, b)])
    paths = {
        a: ph.CollisionMotion([[-1, 0, 0], [1, 0, 0]]),
        b: ph.CollisionMotion([[0.5, 0.125, 0], [-0.5, 0.125, 0]]),
    }
    (row,) = detect_one(model, paths).pairs
    assert row.status == "overlap"
    np.testing.assert_allclose(row.witness.axis_parameter_span, [(0.25, 1), (0.75, 0)])
    np.testing.assert_allclose(row.witness.normal, [0, 1, 0])


def test_zero_angle_pivot_translation_uses_actual_displacement():
    model, paths = crossing_capsules()
    k = max(paths)
    moving = paths[k]
    paths[k] = ph.CollisionMotion(
        moving.start, turn=ph.RigidTurn([4, 1, 3], [4, 1, 1], [0, 1, 0], 0.0)
    )
    at(detect_one(model, paths).pairs[0], Fraction(3, 8))


def test_plane_tangential_anchor_has_no_finite_aabb():
    plane, ball = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
    model = ph.CollisionModel(
        [
            ph.CollisionShape(plane, "plane", (12,), normal=(0, 0, 1)),
            ph.CollisionShape(ball, "sphere", (1,), 0.25),
        ]
    )
    paths = {
        plane: ph.CollisionMotion([[1e100, 0, 0]]),
        ball: ph.CollisionMotion([[0, 0, 1]], [[0, 0, -1]]),
    }
    result = detect_one(model, paths)
    (row,) = result.pairs
    at(row, Fraction(3, 8))
    assert result.bounds[0].lower is None and result.bounds[0].upper is None
    witness = row.sweep.witness
    assert witness.first_offset == (-1e100, 0.0, 0.0) and witness.first_point == (0.0, 0.0, 0.0)


def test_unrepresentable_plane_gap_has_no_published_witness():
    a, b = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
    model = ph.CollisionModel(
        [
            ph.CollisionShape(a, "plane", (1,), normal=(1, 1, 0)),
            ph.CollisionShape(b, "sphere", (1,), 0.125),
        ]
    )
    maximum = np.finfo(np.float64).max
    (row,) = detect_one(
        model, {a: ph.CollisionMotion([[maximum, maximum, 0]]), b: ph.CollisionMotion([[0, 0, 0]])}
    ).pairs
    assert row.candidate and row.status == "unresolved" and row.reason == "witness_outside_fp64"
    assert row.witness is None
