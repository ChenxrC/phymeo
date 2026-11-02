"""M0 static contact regions, capsule bodies and shared-material counterexamples."""

import json
from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest
from case_catalog import catalog_entries, geometry_for
from contact_region_oracles import (
    box_slide,
    capsule_face,
    last_face_time,
    shared_faces,
    skew_edge_time,
)

from phymeo import closest_point_triangle, linear_feature_sweep
from phymeo.constraints.cloth_self_contact import ClothSelfContactTopology
from phymeo.geometry._prism_clip import hull, intersect
from phymeo.geometry.capsule import closest_segment_points
from phymeo.geometry.cloth_self_candidates import candidate_self_pairs

RUNS = {
    "RR03": ("base", "permuted", "clear_gap", "edge_edge", "holdout"),
    "RD02": ("base", "clear", "cap_touch", "holdout", "reverse_holdout"),
    "DD03": ("base", "flat", "hinge", "renumbered", "holdout"),
}


@pytest.mark.parametrize("case_id", RUNS)
def test_second_batch_variants(case_id):
    case = catalog_entries()[case_id]
    variants = case.get("m0_variants", [])
    assert tuple(row["name"] for row in variants) == RUNS[case_id]
    assert variants[0]["changes"] == {}
    assert {row["relation"] for row in variants} == {"hit", "clear"}


def record(tmp_path, case_id, variant, geometry, reference, observed):
    row = dict(
        case=case_id,
        variant=variant,
        geometry=geometry,
        backend="host",
        reference=reference,
        observed=observed,
    )
    (tmp_path / "observation.json").write_text(
        json.dumps(row, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )


def box_vertices(shape, time, order):
    if "center" in shape:
        center = tuple(map(F, shape["center"]))
    else:
        start, end = shape["center_start"], shape["center_end"]
        center = tuple((1 - time) * F(u) + time * F(v) for u, v in zip(start, end, strict=True))
    axes = shape.get("half_axes")
    if axes is None:
        axes = [[h if i == j else 0 for j in range(3)] for i, h in enumerate(shape["half"])]
    corners = [
        tuple(
            center[j] + sum(k * F(axis[j]) for k, axis in zip(signs, axes, strict=True))
            for j in range(3)
        )
        for signs in product((-1, 1), repeat=3)
    ]
    assert sorted(order) == list(range(8))
    return [corners[i] for i in order]


def projected_features(a, b, time, order):
    # This adapter supplies +z from the fixture. The Phymeo clipper consumes
    # unordered support vertices, not a complete pair of general 3D shapes.
    x, y = [box_vertices(shape, time, order) for shape in (a, b)]
    z_a, z_b = max(p[2] for p in x), min(p[2] for p in y)
    fa = hull([p[:2] for p in x if p[2] == z_a])
    fb = hull([p[:2] for p in y if p[2] == z_b])
    return z_a, z_b, intersect(fa, fb)


def region_area(poly):
    return abs(sum(p[0] * q[1] - p[1] * q[0] for p, q in zip(poly, poly[1:] + poly[:1]))) / 2


@pytest.mark.parametrize("name", RUNS["RR03"])
def test_rr03_projected_box_features(name, tmp_path):
    case, variant, geom = geometry_for("RR03", name)
    a, b = geom["A"], geom["B"]
    order = variant.get("hull_order", list(range(8)))
    eps_t = F(1, 10000)
    observed = []
    if "half_axes" in a:
        time, height = skew_edge_time(a, b)
        assert float(time) == pytest.approx(0.5, abs=1e-12, rel=0)
        for t in (time - eps_t, time, time + eps_t):
            za, zb, region = projected_features(a, b, t, order)
            assert region == ((0, 0),) and za == height
            if t < time:
                assert zb > za
            elif t == time:
                assert zb == za
            else:
                assert zb < za
                # A point in both open hulls proves overlap following contact.
                middle = np.array([0.0, 0.0, float((za + zb) / 2)])
                for shape, center in (
                    (a, a["center"]),
                    (b, [0, 0, float(zb + sum(abs(F(v[2])) for v in b["half_axes"]))]),
                ):
                    coords = np.linalg.solve(np.array(shape["half_axes"]).T, middle - center)
                    assert np.max(np.abs(coords)) < 1
            observed.append(
                {
                    "t": float(t),
                    "support_height_gap_m": float(zb - za),
                    "projected_vertices_xy_m": [list(map(float, p)) for p in region],
                }
            )
        ref = {"kind": "skew_edge_touch", "first_t": float(time), "point_m": [0, 0, float(height)]}
    else:
        time = last_face_time(a, b)
        expected_t = 0.7 if name == "holdout" else case["oracle"]["edge_only_t"]
        assert float(time) == pytest.approx(expected_t, abs=1e-12, rel=0)
        ref_rows = []
        for t in (F(0), F(1, 5), time - eps_t, time, time + eps_t, F(1)):
            corners, area, gap, dim = box_slide(a, b, t)
            za, zb, poly = projected_features(a, b, t, order)
            assert set(poly) == corners and region_area(poly) == area
            if variant["relation"] == "clear":
                assert dim == -1 and gap > 0 and zb > za
                if t <= time:
                    assert gap < variant["prediction_margin_m"]
            else:
                assert za == zb and dim == (2 if t < time else 1 if t == time else -1)
                if t == time:
                    assert len(poly) == 2
                    assert abs(poly[0][1] - poly[1][1]) == (F(1, 2) if name == "holdout" else 1)
            ref_rows.append({"t": float(t), "hard_dimension": dim, "hard_gap_m": gap})
            observed.append(
                {
                    "t": float(t),
                    "projected_area_m2": float(region_area(poly)),
                    "support_height_gap_m": float(zb - za),
                    "projected_vertices_xy_m": [list(map(float, p)) for p in poly],
                }
            )
        ref = {"kind": "sliding_boxes", "projected_edge_t": float(time), "samples": ref_rows}
    record(
        tmp_path,
        "RR03",
        name,
        geom,
        ref,
        {
            "entry": "_prism_clip.hull/intersect",
            "axis_supplied": [0, 0, 1],
            "shape_pair_manifold": None,
            "samples": observed,
            "hull_order": order,
        },
    )


@pytest.mark.parametrize("name", RUNS["RD02"])
def test_rd02_capsule_axis_through_soft_face(name, tmp_path):
    _, variant, geom = geometry_for("RD02", name)
    tri = np.array(geom["A"]["vertices"], dtype=float)
    axis = np.array(geom["B"]["axis"], dtype=float)
    radius = geom["B"]["radius"]
    ref = capsule_face(tri, axis, radius)
    assert (ref["kind"] == "separated") == (variant["relation"] == "clear")
    # The capsule is the union of radius-r balls on this *static* segment.
    # Here the sweep variable parametrizes the axis; it is not a physical TOI.
    sweep = linear_feature_sweep(axis[:1], axis[1:], tri, tri, clearance_m=radius)
    cap_gaps = [closest_point_triangle(p, tri).distance - radius for p in axis]
    vertex_gaps = []
    for vertex in tri:
        first, second = closest_segment_points(*axis, vertex, vertex)
        vertex_gaps.append(float(np.linalg.norm(first - second) - radius))
    assert min(vertex_gaps) > 0
    witness = None
    ref_gap = None
    if ref["kind"] == "axis_pierces":
        assert sweep.status == "violation" and min(cap_gaps) > 0
        s = sweep.witness_time
        sample = axis[0] + s * (axis[1] - axis[0])
        pair = closest_point_triangle(sample, tri)
        assert pair.distance < radius and 0 < s < 1
        witness = {
            "axis_parameter": s,
            "axis_point_m": sample.tolist(),
            "face_point_m": pair.second.tolist(),
            "face_weights": pair.second_weights.tolist(),
            "axis_distance_minus_radius_m": pair.distance - radius,
        }
    elif ref["kind"] == "cap_tangent":
        assert sweep.status == "certified" and abs(min(cap_gaps)) < 1e-12
        assert ref["axis_parameter"] == 0
    else:
        assert sweep.status == "certified" and min(cap_gaps) >= ref["gap_lower_m"] - 1e-12
    if "axis_parameter" in ref:
        s = ref["axis_parameter"]
        pair = closest_point_triangle(axis[0] + s * (axis[1] - axis[0]), tri)
        np.testing.assert_allclose(pair.second, ref["point_m"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(pair.second_weights, ref["face_weights"], atol=1e-12, rtol=0)
        ref_gap = pair.distance - radius
        if ref["kind"] == "axis_pierces":
            expected_s = {"base": 0.5, "holdout": 0.25, "reverse_holdout": 0.75}[name]
            assert s == pytest.approx(expected_s, abs=1e-12, rel=0)
            assert ref_gap == pytest.approx(-radius, abs=1e-12, rel=0)
    record(
        tmp_path,
        "RD02",
        name,
        geom,
        ref,
        {
            "entry": "linear_feature_sweep + closest_point_triangle",
            "axis_status": sweep.status,
            "axis_witness": witness,
            "cap_gaps_m": cap_gaps,
            "cloth_vertex_gaps_m": vertex_gaps,
            "gap_at_reference_parameter_m": ref_gap,
            "physical_toi": None,
            "shape_pair_manifold": None,
        },
    )


@pytest.mark.parametrize("name", RUNS["DD03"])
def test_dd03_shared_material_and_remote_intersection(name, tmp_path):
    case, variant, geom = geometry_for("DD03", name)
    mesh = geom["A"]
    original_vertices = np.array(mesh["current_vertices"], dtype=float)
    order = variant.get("node_order", list(range(len(original_vertices))))
    assert sorted(order) == list(range(len(original_vertices)))
    vertices = original_vertices[order]
    faces = np.argsort(order)[np.asarray(mesh["faces"])]
    target = tuple(geom["pair_faces"])
    pair = faces[list(target)]
    ref = shared_faces(vertices, pair)
    is_remote = ref["kind"] == "distinct_material_intersection"
    assert is_remote == (variant["relation"] == "hit")
    measured = []
    if is_remote:
        p = np.array(ref["point_m"])
        material_weights = []
        for face, weights in zip(pair, ref["weights"], strict=True):
            result = closest_point_triangle(p, vertices[face])
            np.testing.assert_allclose(result.second, p, atol=1e-12, rtol=0)
            np.testing.assert_allclose(result.second_weights, weights, atol=1e-12, rtol=0)
            original_ids = [order[i] for i in face]
            material_weights.append(dict(zip(original_ids, weights, strict=True)))
            measured.append(
                {
                    "node_ids": original_ids,
                    "point_m": result.second.tolist(),
                    "weights": result.second_weights.tolist(),
                    "distance_m": result.distance,
                }
            )
        assert material_weights[0] != material_weights[1]
        if name in ("base", "renumbered"):
            oracle = case["oracle"]
            np.testing.assert_allclose(p, oracle["point"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(
                ref["weights"],
                [oracle["barycentric_first"], oracle["barycentric_second"]],
                atol=1e-12,
                rtol=0,
            )
        else:
            np.testing.assert_allclose(p, [0.625, 0.25, 0], atol=1e-12, rtol=0)
    elif name == "hinge":
        assert ref["kind"] == "shared_edge_only"
    else:
        assert ref["kind"] == "shared_vertex_only"

    with pytest.raises(ValueError, match="strictly positive") as zero_rejected:
        ClothSelfContactTopology("sheet", faces, np.zeros(len(faces)))
    # Positive thickness is a separate diagnostic of the old broad phase.
    # Duplicating vertex IDs in the control preserves the geometry and its bounds.
    rho = 1 / 64
    topology = ClothSelfContactTopology("sheet", faces, np.full(len(faces), rho))
    candidates = candidate_self_pairs(topology, vertices)
    disconnected = ClothSelfContactTopology("copies", np.arange(6).reshape(2, 3), np.full(2, rho))
    control = candidate_self_pairs(disconnected, vertices[pair].reshape(6, 3))
    assert topology.incidence_policy == "exclude_shared_vertex"
    assert target not in candidates and control == ((0, 1),)
    record(
        tmp_path,
        "DD03",
        name,
        geom,
        ref,
        {
            "entry": "candidate_self_pairs + closest_point_triangle",
            "node_order": order,
            "nominal_zero_thickness": str(zero_rejected.value),
            "supplied_witness_checks": measured,
            "separate_diagnostic_half_thickness_m": rho,
            "legacy_incidence_policy": topology.incidence_policy,
            "selected_pair": target,
            "legacy_candidate_pairs": candidates,
            "split_id_candidate_pairs": control,
            "shape_pair_manifold": None,
        },
    )
