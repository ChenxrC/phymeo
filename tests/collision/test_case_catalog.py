"""M0 baselines for the first independent RR, RD and DD catalog cases."""

import json
from copy import deepcopy
from math import asin, cos, hypot, pi, radians, sin
from pathlib import Path

import numpy as np
import pytest
from case_oracles import crossing_edges, rotating_capsule, sphere_face

from phymeo import closest_point_triangle, closest_segment_segment, linear_feature_sweep
from phymeo.geometry.capsule import closest_segment_points

CATALOG = Path(__file__).resolve().parents[2] / "docs/benchmarks/p65-general-collision-cases.json"
# Fixed before observing the geometry measurements. Coordinates are in metres.
GAP_M = 1e-5
TIME_T = 1e-4
RUNS = {
    "RR04": ("base", "clear", "holdout"),
    "RD01": ("base", "clear", "permuted", "holdout", "holdout_permuted"),
    "DD02": ("base", "clear", "swap_reverse", "holdout"),
}


def catalog_entries():
    document = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert document["schema"] == "phymeo-general-collision-cases/1"
    entries = {row["id"]: row for row in document["cases"]}
    assert len(entries) == len(document["cases"])
    expected = {f"{family}{i:02d}" for family in ("RR", "RD", "DD") for i in range(1, 6)}
    assert expected <= entries.keys()
    return entries


@pytest.mark.parametrize("case_id", RUNS)
def test_catalog_binds_executable_controls_and_holdouts(case_id):
    case = catalog_entries()[case_id]
    variants = case.get("m0_variants", [])
    assert [v["name"] for v in variants] == list(RUNS[case_id])
    assert case["family"] == case_id[:2]
    assert variants[0]["changes"] == {}
    assert all(v["relation"] in {"hit", "clear"} for v in variants)
    assert {v["relation"] for v in variants} == {"hit", "clear"}


def geometry_for(case_id, variant_name):
    case = catalog_entries()[case_id]
    variant = next(v for v in case["m0_variants"] if v["name"] == variant_name)
    geometry = deepcopy(case["geometry"])
    for side, fields in variant["changes"].items():
        assert fields.keys() <= geometry[side].keys()
        geometry[side].update(fields)
    for side in ("A", "B"):
        if order := variant.get("order_" + side):
            assert sorted(order) == [0, 1, 2]
            for field in ("vertices", "vertices_start", "vertices_end"):
                if field in geometry[side]:
                    geometry[side][field] = [geometry[side][field][i] for i in order]
    if variant.get("swap"):
        geometry["A"], geometry["B"] = geometry["B"], geometry["A"]
    return case, variant, geometry


def assert_expected(variant, reference, expected_time):
    if variant["relation"] == "clear":
        assert reference.time is None and reference.separation_lower_m > 0
    else:
        assert reference.time == pytest.approx(expected_time, abs=1e-12, rel=0)


def samples(hit):
    times = [0.0, 0.5, 1.0]
    if hit is not None:
        times.extend([hit - TIME_T, hit, hit + TIME_T])
    return sorted(set(times))


def save_observation(tmp_path, case_id, variant, geometry, reference, observed):
    record = {
        "case": case_id,
        "variant": variant["name"],
        "backend": "host",
        "geometry": geometry,
        "reference": {
            "relation": variant["relation"],
            "first_t": reference.time,
            "point_m": reference.point,
            "weights": reference.weights,
            "separation_lower_m": reference.separation_lower_m,
        },
        "observed": observed,
    }
    (tmp_path / "observation.json").write_text(
        json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )


@pytest.mark.parametrize("name", RUNS["RR04"])
def test_rr04_rotating_capsule(name, tmp_path):
    _, variant, geom = geometry_for("RR04", name)
    a, b = geom["A"], geom["B"]
    ref = rotating_capsule(a, b)
    ratio = 1 / 6 if name == "holdout" else 1 / 8
    assert_expected(variant, ref, (pi / 3 - asin(ratio)) / (2 * pi / 3))

    rotation = a["rotation"]
    start, end = (radians(rotation[key]) for key in ("angle_start_deg", "angle_end_deg"))
    center = np.array(b["center"])
    radii = a["radius"] + b["radius"]
    rows = []
    for time in samples(ref.time):
        angle = (1 - time) * start + time * end
        matrix = np.array([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]])
        p0, p1 = np.asarray(a["local_axis"]) @ matrix.T
        p, q = closest_segment_points(p0, p1, center, center)
        gap = np.linalg.norm(q - p) - radii
        expected = hypot(center[1] * cos(angle) - center[0] * sin(angle), center[2]) - radii
        assert gap == pytest.approx(expected, abs=GAP_M, rel=0)
        if time == ref.time:
            point = p + a["radius"] * (q - p) / np.linalg.norm(q - p)
            np.testing.assert_allclose(point, ref.point, atol=GAP_M, rtol=0)
        rows.append({"t": time, "gap_m": gap, "error_m": abs(gap - expected)})
    by_time = {row["t"]: row["gap_m"] for row in rows}
    assert by_time[0] > 0 and by_time[1] > 0
    if ref.time is not None:
        assert by_time[ref.time - TIME_T] > 0 > by_time[ref.time + TIME_T]
        assert by_time[0.5] < 0
    else:
        assert min(by_time.values()) >= ref.separation_lower_m - GAP_M
    save_observation(
        tmp_path,
        "RR04",
        variant,
        geom,
        ref,
        {
            "entry": "closest_segment_points",
            "samples": rows,
            "pair_first_t": None,
            "coverage": "static_queries_on_prescribed_arc",
        },
    )


@pytest.mark.parametrize("name", RUNS["RD01"])
def test_rd01_sphere_and_whole_face(name, tmp_path):
    case, variant, geom = geometry_for("RD01", name)
    a, b = geom["A"], geom["B"]
    ref = sphere_face(a, b)
    assert_expected(variant, ref, case["oracle"]["first_t"])
    tri = np.asarray(a["vertices"], dtype=float)
    c0, c1 = np.asarray(b["center_start"], dtype=float), np.asarray(b["center_end"], dtype=float)
    radius = b["radius"]
    sweep = linear_feature_sweep(c0[None], c1[None], tri, tri, clearance_m=radius)
    assert sweep.status == ("certified" if ref.time is None else "violation")
    rows = []
    witness_error = None
    for time in samples(ref.time):
        center = c0 + time * (c1 - c0)
        pair = closest_point_triangle(center, tri)
        gap = pair.distance - radius
        rows.append({"t": time, "gap_m": gap})
        if time == ref.time:
            assert min(np.linalg.norm(tri - center, axis=1)) > radius
            np.testing.assert_allclose(pair.second, ref.point, atol=GAP_M, rtol=0)
            np.testing.assert_allclose(pair.second_weights, ref.weights, atol=1e-10, rtol=0)
            witness_error = float(np.linalg.norm(pair.second - ref.point))
            # Pin the original material IDs as well as the current vertex order.
            expected = [0.25, 0.25, 0.5] if name.startswith("holdout") else [1 / 3] * 3
            order = variant.get("order_A", [0, 1, 2])
            np.testing.assert_allclose(pair.second_weights, np.array(expected)[order], atol=1e-10)
    gaps = {row["t"]: row["gap_m"] for row in rows}
    assert gaps[0] > 0 and gaps[1] > 0
    if ref.time is None:
        assert min(gaps.values()) >= ref.separation_lower_m - GAP_M
        assert sweep.witness_time is None
    else:
        assert abs(gaps[ref.time]) <= GAP_M
        assert gaps[ref.time - TIME_T] > 0 > gaps[ref.time + TIME_T]
        assert sweep.witness_time is not None and sweep.witness_time > ref.time
        assert abs((c0 + sweep.witness_time * (c1 - c0))[2]) < radius
    save_observation(
        tmp_path,
        "RD01",
        variant,
        geom,
        ref,
        {
            "entry": "closest_point_triangle + linear_feature_sweep",
            "samples": rows,
            "sweep_status": sweep.status,
            "sweep_witness_t": sweep.witness_time,
            "pair_first_t": None,
            "point_error_m": witness_error,
            "coverage": "static_face_and_whole_path_positive_clearance",
        },
    )


@pytest.mark.parametrize("name", RUNS["DD02"])
def test_dd02_two_edges_without_vertex_hits(name, tmp_path):
    case, variant, geom = geometry_for("DD02", name)
    a, b = geom["A"], geom["B"]
    ref = crossing_edges(a, b)
    assert_expected(variant, ref, 0.6 if name == "holdout" else case["oracle"]["first_t"])
    first = np.array([a["vertices_start"], a["vertices_end"]])
    second = np.array([b["vertices_start"], b["vertices_end"]])
    with pytest.raises(ValueError, match="clearance_m must be finite and positive") as rejected:
        linear_feature_sweep(*first[:, :2], *second[:, :2], clearance_m=0.0)

    # Inspect all nine edge pairs and six vertex/face pairs at the event.
    # For the clear control the terminal pose attains the path's positive lower bound.
    time = ref.time if ref.time is not None else 1.0
    x, y = ((1 - time) * path[0] + time * path[1] for path in (first, second))
    edges = [[0, 1], [1, 2], [2, 0]]
    ee = [closest_segment_segment(x[i], y[j]) for i in edges for j in edges]
    vf = [closest_point_triangle(v, face) for verts, face in ((x, y), (y, x)) for v in verts]
    assert min(p.distance for p in vf) > GAP_M
    hits = [i for i, p in enumerate(ee) if p.distance <= GAP_M]
    edge = ee[0]
    if ref.time is None:
        assert not hits
        assert min(p.distance for p in ee + vf) == pytest.approx(ref.separation_lower_m, abs=GAP_M)
    else:
        assert hits == [0]
        for point in (edge.first, edge.second):
            np.testing.assert_allclose(point, ref.point, atol=GAP_M, rtol=0)
        params = [edge.first_weights[1], edge.second_weights[1]]
        np.testing.assert_allclose(params, ref.weights, atol=1e-10, rtol=0)
        np.testing.assert_allclose(params, case["oracle"]["edge_parameters"], atol=1e-10, rtol=0)
    save_observation(
        tmp_path,
        "DD02",
        variant,
        geom,
        ref,
        {
            "entry": "closest_segment_segment + closest_point_triangle + linear_feature_sweep",
            "evaluated_t": time,
            "edge_pair_distances_m": [p.distance for p in ee],
            "vertex_face_distances_m": [p.distance for p in vf],
            "edge_hit_indices": hits,
            "zero_clearance_response": str(rejected.value),
            "pair_first_t": None,
            "coverage": "static_triangle_features_only",
        },
    )
