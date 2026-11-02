"""Final M0 host case baselines, before the shared tyx collision API."""

import json
from fractions import Fraction
from math import sqrt

import numpy as np
import pytest
from case_catalog import catalog_entries, geometry_for
from contact_region_oracles import barycentric
from matrix_case_oracles import (
    aligned_capsules,
    ball_event,
    ball_gap,
    cloth_tet_patch,
    corner_clearance_reference,
    first_vertex_on_face,
    plate_path,
    rounded_point,
)

from phymeo import (
    CollisionPlane,
    LinkCapsule,
    RobotContactSystem,
    RobotModel,
    RobotState,
    SolverFailure,
    SphereShapes,
    closest_point_triangle,
    linear_feature_sweep,
    particle_sphere_geometry,
)
from phymeo.geometry._prism_clip import hull, intersect
from phymeo.geometry.capsule import closest_segment_points
from phymeo.geometry.plane import Plane
from phymeo.geometry.robot_patch_sweep import _closest
from phymeo.geometry.shell_boundary import _tet_complex

RUNS = {
    "RR01": ("base", "clear", "filtered", "sphere_pair", "axis_holdout"),
    "RR02": ("base", "tangent", "overlap", "endcaps", "parallel_holdout", "coincident"),
    "RD03": ("base", "large_face", "rim_bypass", "through_plate", "corner_holdout"),
    "RD05": ("base", "clear", "outside_box", "rigid_transform", "tip_holdout"),
    "DD01": ("base", "clear", "reused_ids", "swapped", "tip_holdout"),
    "DD04": ("base", "thick", "lateral_clear", "reversed_thick", "moving_tet_holdout"),
}


@pytest.mark.parametrize("case_id", RUNS)
def test_last_batch_binds_variants(case_id):
    case = catalog_entries()[case_id]
    variants = case.get("m0_variants", [])
    assert tuple(v["name"] for v in variants) == RUNS[case_id]
    assert variants[0]["changes"] == {}
    assert {"hit", "clear"} <= {v["relation"] for v in variants}


def record(tmp_path, case, variant, geometry, reference, observed):
    data = dict(
        case=case,
        variant=variant["name"],
        backend="host",
        geometry=geometry,
        settings={k: v for k, v in variant.items() if k not in ("geometry", "changes")},
        reference=reference,
        observed=observed,
    )
    (tmp_path / "observation.json").write_text(
        json.dumps(data, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )


def lerp(a, b, t):
    return (1 - t) * a + t * b


def times(first):
    samples = {0.0, 0.5, 1.0}
    if first is not None:
        samples.update([max(0.0, first - 1e-4), first, min(1.0, first + 1e-4)])
    return sorted(samples)


def expect_time(actual, expected):
    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected, abs=1e-12, rel=0)


def slider(origin, direction):
    direction = np.array(direction, dtype=float)
    length = np.linalg.norm(direction)
    unit = direction / length if length else np.array([1.0, 0.0, 0.0])
    return RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.array([origin], dtype=float),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        unit[None],
        np.ones(1),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )


def link(key, axis, r):
    return LinkCapsule(key, key, 0, *np.asarray(axis, dtype=float), r)


def state(q=0.0, speed=0.0):
    return RobotState(np.array([q]), np.array([speed]))


def patch_clip(a, b, axes):
    def project(points):
        return hull([tuple(Fraction(float(p[i])) for i in axes) for p in points])

    poly = intersect(project(a), project(b))
    twice = sum(p[0] * q[1] - p[1] * q[0] for p, q in zip(poly, poly[1:] + poly[:1], strict=True))
    return [[float(x) for x in p] for p in poly], float(abs(twice) / 2)


@pytest.mark.parametrize("name", RUNS["RR01"])
def test_rr01_sphere_pairs_and_filter(name, tmp_path):
    _, v, geom = geometry_for("RR01", name)
    a, b = geom["A"], geom["B"]
    ref = ball_event(a, b)
    e = v["expected"]
    expect_time(ref["first_t"], e.get("geometric_first_t", e.get("first_t")))
    if "point_m" in e:
        np.testing.assert_allclose(ref["point_m"], e["point_m"], atol=1e-12, rtol=0)
    for key, index in (("start_gap", 0), ("end_gap", 1)):
        if key in e:
            assert ref["endpoint_gap_m"][index] == pytest.approx(e[key], abs=1e-12)
    c0, c1 = (np.asarray(b[k], dtype=float) for k in ("center_start", "center_end"))
    speed = float(np.linalg.norm(c1 - c0))
    models = {"B": slider(c0, c1 - c0)}
    caps = [link("B", np.zeros((2, 3)), b["radius"])]
    planes = []
    free_plane = None
    if a["type"] == "plane":
        planes.append(CollisionPlane("A", np.array(a["normal"], dtype=float), a["offset"]))
        if a["normal"] == [0, 1, 0]:
            free_plane = list(Plane(a["offset"], tuple(a["normal"])).normal)
        else:
            with pytest.raises(ValueError, match="only normal") as rejected:
                Plane(a["offset"], tuple(a["normal"]))
            free_plane = str(rejected.value)
    else:
        models["A"] = slider(a["center"], [1, 0, 0])
        caps.insert(0, link("A", np.zeros((2, 3)), a["radius"]))
    masked = v.get("exclude_pair", False)
    system = RobotContactSystem(
        models, capsules=caps, planes=planes, excluded_pairs=[("A", "B")] if masked else []
    )
    rows = []
    for t in times(ref["first_t"]):
        c = lerp(c0, c1, t)
        reference_gap = ball_gap(a, c, b["radius"])
        st = {key: state() for key in models}
        st["B"] = state(t * speed, speed)
        out = system.evaluate(st)
        assert out["checked_pairs"] == (0 if masked else 1)
        gaps = [r["gap_m"] for r in out["contacts"]]
        if masked or reference_gap > 1e-12:
            assert gaps == []
        elif reference_gap < -1e-12:
            assert len(gaps) == 1
        if gaps:
            assert gaps == pytest.approx([reference_gap], abs=1e-12, rel=0)
        point_gap = None
        if a["type"] == "sphere":
            # Minkowski radius shift maps sphere/sphere to point/sphere for this static query.
            query = particle_sphere_geometry(
                c[None],
                np.array([a["center"]], dtype=float),
                SphereShapes(np.array([a["radius"] + b["radius"]])),
            )
            point_gap = float(query.gap[0])
            assert point_gap == pytest.approx(reference_gap, abs=1e-12, rel=0)
        rows.append(
            dict(
                t=t,
                checked_pairs=out["checked_pairs"],
                response_gaps_m=gaps,
                expanded_point_sphere_gap_m=point_gap,
            )
        )
    split = None
    if "split_t" in v:
        cut = v["split_t"]
        mid = lerp(c0, c1, cut)
        left = ball_event(a, {**b, "center_end": mid})
        right = ball_event(a, {**b, "center_start": mid})
        assert left["first_t"] is None
        mapped = cut + (1 - cut) * right["first_t"]
        expect_time(mapped, ref["first_t"])
        split = dict(cut=cut, mapped_reference_t=mapped)
    record(
        tmp_path,
        "RR01",
        v,
        geom,
        ref,
        dict(
            entry="RobotContactSystem.evaluate",
            samples=rows,
            named_exclusion=system.excluded_pairs,
            free_plane_input=free_plane,
            split=split,
            public_pair_status=None,
            pair_first_t=None,
        ),
    )


@pytest.mark.parametrize("name", RUNS["RR02"])
def test_rr02_capsule_witnesses_and_ties(name, tmp_path):
    _, v, g = geometry_for("RR02", name)
    a, b = g["A"], g["B"]
    ref = aligned_capsules(a, b)
    aa, bb = (np.array(s["axis"], dtype=float) for s in (a, b))
    p, q = closest_segment_points(*aa, *bb)
    length = np.linalg.norm(q - p)
    gap = float(length - a["radius"] - b["radius"])
    assert gap == pytest.approx(ref["gap_m"], abs=1e-12)
    assert gap == pytest.approx(v["expected"]["gap_m"], abs=1e-12)
    parameters = [
        float((pt - seg[0]) @ (seg[1] - seg[0]) / np.sum((seg[1] - seg[0]) ** 2))
        for pt, seg in ((p, aa), (q, bb))
    ]
    assert all(0 <= x <= 1 for x in parameters)
    if name in ("base", "tangent", "overlap"):
        np.testing.assert_allclose(parameters, [0.5, 0.5], atol=1e-12)
    if name == "endcaps":
        np.testing.assert_allclose(parameters, [1, 0], atol=1e-12)
    if "parallel_overlap_x" in v["expected"]:
        lo, hi = v["expected"]["parallel_overlap_x"]
        assert lo <= p[0] == q[0] <= hi
    system = RobotContactSystem(
        {k: slider([0, 0, 0], [1, 0, 0]) for k in ("A", "B")},
        capsules=(link("A", aa, a["radius"]), link("B", bb, b["radius"])),
    )
    states = {k: state() for k in ("A", "B")}
    response = []
    rejection = None
    if ref["normal"] is None:
        assert v["expected"]["normal"] is None and length == 0
        with pytest.raises(SolverFailure, match="coincident") as rejected:
            system.evaluate(states)
        rejection = str(rejected.value)
    else:
        n = (q - p) / length
        np.testing.assert_allclose(n, ref["normal"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(n, v["expected"]["normal"], atol=1e-12, rtol=0)
        out = system.evaluate(states)
        assert out["checked_pairs"] == 1
        response = [r["gap_m"] for r in out["contacts"]]
        if gap < -1e-12:
            assert response == pytest.approx([gap], abs=1e-12, rel=0)
            np.testing.assert_allclose(out["contacts"][0]["normal"], -n, atol=1e-12, rtol=0)
        else:
            assert not response
    record(
        tmp_path,
        "RR02",
        v,
        g,
        ref,
        dict(
            axis_points_m=[p.tolist(), q.tolist()],
            gap_m=gap,
            axis_parameters=parameters,
            response_gaps_m=response,
            normal_rejection=rejection,
        ),
    )


def plate_probe(polygon, radius, point):
    weights, closest = _closest(polygon, point)
    center = np.array(closest, dtype=float)
    delta = point - center
    length = np.linalg.norm(delta)
    normal = (delta / length).tolist() if length else None
    ref = rounded_point(polygon, radius, point)
    np.testing.assert_allclose(center, ref["middle_m"], atol=1e-12, rtol=0)
    assert length - radius == pytest.approx(ref["gap_m"], abs=1e-12)
    if normal is not None:
        np.testing.assert_allclose(normal, ref["normal"], atol=1e-12, rtol=0)
    return dict(
        middle_m=center.tolist(),
        weights=list(map(float, weights)),
        normal=normal,
        gap_m=float(length - radius),
    ), ref


@pytest.mark.parametrize("name", RUNS["RD03"])
def test_rd03_finite_rounded_plate(name, tmp_path):
    _, v, g = geometry_for("RD03", name)
    a, b = g["A"], g["B"]
    poly, r = np.array(a["polygon"], dtype=float), a["radius"]
    e = v["expected"]
    if b["type"] == "cloth_material_vertex_queries":
        points = list(b["points"])
        if "corner_point_expression" in b:
            assert b["corner_point_expression"] == "(0.5+0.15/sqrt(2),0.5+0.15/sqrt(2),0)"
            points.append([0.5 + 0.15 / sqrt(2), 0.5 + 0.15 / sqrt(2), 0])
        probes = [plate_probe(poly, r, np.array(p)) for p in points]
        actual, refs = zip(*probes, strict=True)
        assert [p["feature"] for p in refs] == (
            e["features"] if "features" in e else [e["feature"]]
        )
        gaps = [p["gap_m"] for p in actual]
        assert gaps == pytest.approx(e["gaps_m"] if "gaps_m" in e else [e["gap_m"]], abs=1e-12)
        if name == "base":
            np.testing.assert_allclose(
                [p["normal"] for p in actual],
                [[0, 0, 1], [0, 0, -1], [1, 0, 0], [1 / sqrt(2), 1 / sqrt(2), 0]],
                atol=1e-12,
                rtol=0,
            )
        record(tmp_path, "RD03", v, g, refs, dict(samples=actual, pair_manifold=None))
        return

    ref = plate_path(poly, r, b)
    expect_time(ref["first_t"], e["first_t"])
    if b["type"] == "cloth_triangle":
        start, end = (np.array(b[k]) for k in ("vertices_start", "vertices_end"))

        def corner_sweeps(s0, s1):
            return [linear_feature_sweep(p[None], p[None], s0, s1, clearance_m=r) for p in poly]

        sweeps = corner_sweeps(start, end)
        geometry = lerp(start, end, ref["first_t"])
        region, area = patch_clip(poly, geometry, (0, 1))
        assert area == pytest.approx(e["area_m2"])
        assert len(region) == 4 and all(q.status == "violation" for q in sweeps)
        assert all(q.witness_time > ref["first_t"] for q in sweeps)
        vertex_gaps = [plate_probe(poly, r, p)[0]["gap_m"] for p in geometry]
        assert min(vertex_gaps) > 0
        cut = v["split_t"]
        mid = lerp(start, end, cut)
        parts = [corner_sweeps(start, mid), corner_sweeps(mid, end)]
        assert all(q.status == "certified" for q in parts[0])
        assert all(q.status == "violation" for q in parts[1])
        witnesses = [cut + (1 - cut) * q.witness_time for q in parts[1]]
        assert all(w > ref["first_t"] for w in witnesses)
        obs = dict(
            projected_region=region,
            area_m2=area,
            cloth_vertex_gaps_m=vertex_gaps,
            corner_witness_t=[q.witness_time for q in sweeps],
            split_t=cut,
            split_status=[[q.status for q in part] for part in parts],
            split_mapped_witness_t=witnesses,
            pair_first_t=None,
        )
    else:
        start, end = (np.array(b[k]) for k in ("start", "end"))
        sweeps = [
            linear_feature_sweep(start[None], end[None], tri, tri, clearance_m=r)
            for tri in poly[[[0, 1, 2], [0, 2, 3]]]
        ]
        expect = "certified" if ref["first_t"] is None else "violation"
        assert all(q.status == expect for q in sweeps)
        checks = [
            dict(t=u, **plate_probe(poly, r, lerp(start, end, u))[0]) for u in times(ref["first_t"])
        ]
        if ref["first_t"] is None:
            assert min(s["gap_m"] for s in checks) == pytest.approx(e["min_gap_m"], abs=1e-12)
        else:
            assert min(s["gap_m"] for s in checks) == pytest.approx(e["mid_gap_m"])
            assert all(q.witness_time > ref["first_t"] for q in sweeps)
        obs = dict(
            samples=checks,
            triangle_status=[q.status for q in sweeps],
            violation_witness_t=[q.witness_time for q in sweeps],
            pair_first_t=None,
        )
    record(tmp_path, "RD03", v, g, ref, obs)


def box_faces(box, rotation, translation):
    x, y, z = box["half"]
    vertices = (
        np.array(
            [
                [-x, -y, -z],
                [x, -y, -z],
                [x, y, -z],
                [-x, y, -z],
                [-x, -y, z],
                [x, -y, z],
                [x, y, z],
                [-x, y, z],
            ]
        )
        + box["center"]
    )
    vertices = vertices @ rotation.T + translation
    quads = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    triangles = [vertices[[a, b, c]] for a, b, c, _ in quads] + [
        vertices[[a, c, d]] for a, _, c, d in quads
    ]
    return vertices, vertices[quads[1]], triangles


def sample_tips(start, end, fixed_faces, ref):
    observed = []
    for t in times(ref["first_t"]):
        vertices = lerp(start, end, t)
        nearest = [
            min((closest_point_triangle(p, face) for face in fixed_faces), key=lambda q: q.distance)
            for p in vertices
        ]
        d = [q.distance for q in nearest]
        observed.append(dict(t=t, vertex_face_distances_m=d))
        if t == ref["first_t"]:
            tip = ref["vertex"]
            assert [i for i, dist in enumerate(d) if dist < 1e-10] == [tip]
            np.testing.assert_allclose(nearest[tip].second, ref["point_m"], atol=1e-12, rtol=0)
            observed[-1].update(
                face_point_m=nearest[tip].second.tolist(),
                face_weights=nearest[tip].second_weights.tolist(),
            )
        elif ref["first_t"] is None:
            assert min(d) >= ref["gap_lower_m"] - 1e-12
    return observed


@pytest.mark.parametrize("name", RUNS["RD05"])
def test_rd05_deforming_tip_and_rigid_box(name, tmp_path):
    _, v, g = geometry_for("RD05", name)
    rotation, translation = (
        np.array(v.get("rotation", np.eye(3))),
        np.array(v.get("translation", [0, 0, 0])),
    )
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3))
    assert np.linalg.det(rotation) == 1
    corners, top, faces = box_faces(g["A"], rotation, translation)
    start, end = (
        np.array(g["B"][k]) @ rotation.T + translation for k in ("vertices_start", "vertices_end")
    )
    ref = first_vertex_on_face(top, corners, start, end)
    e = v["expected"]
    expect_time(ref["first_t"], e["first_t"])
    if ref["first_t"] is None:
        assert ref["gap_lower_m"] == pytest.approx(e["gap_lower_m"])
    else:
        np.testing.assert_allclose(ref["point_m"], e["point_m"], atol=1e-12, rtol=0)
        assert v.get("order_B", [0, 1, 2])[ref["vertex"]] == e["original_vertex"]
    rows = sample_tips(start, end, faces, ref)
    record(
        tmp_path, "RD05", v, g, ref, dict(samples=rows, box_triangles=len(faces), pair_first_t=None)
    )


@pytest.mark.parametrize("name", RUNS["DD01"])
def test_dd01_independent_bodies_and_material_ids(name, tmp_path):
    _, v, g = geometry_for("DD01", name)
    face_role = "A" if "vertices" in g["A"] else "B"
    tip_role = "B" if face_role == "A" else "A"
    face = np.array(g[face_role]["vertices"], dtype=float)
    start, end = (np.array(g[tip_role][k]) for k in ("vertices_start", "vertices_end"))
    ref = first_vertex_on_face(face, face, start, end)
    e = v["expected"]
    expect_time(ref["first_t"], e["first_t"])
    rows = sample_tips(start, end, [face], ref)
    ids = v.get("node_ids", [[0, 1, 2], [0, 1, 2]])
    domains = v.get("object_ids", ["cloth-left", "cloth-right"])
    # These owner addresses are supplied by the adapter. The feature API has no body-ID field.
    face_ids = [ids[0][i] for i in v.get("order_A", [0, 1, 2])]
    material = None
    if ref["first_t"] is not None:
        p = np.array(ref["point_m"])
        np.testing.assert_allclose(p, e["point_m"], atol=1e-12, rtol=0)
        weights = barycentric(face, p).tolist()
        observed = next(row for row in rows if row["t"] == ref["first_t"])
        np.testing.assert_allclose(observed["face_weights"], weights, atol=1e-12, rtol=0)
        index = ref["vertex"]
        tip_id = ids[1][index]
        assert tip_id in face_ids and domains[0] != domains[1]
        material = dict(
            face_role=face_role,
            tip_role=tip_role,
            face_domain=domains[0],
            face_node_ids=face_ids,
            vertex_domain=domains[1],
            vertex_id=tip_id,
            weights=weights,
        )
    else:
        assert ref["gap_lower_m"] == pytest.approx(e["gap_lower_m"])
    with pytest.raises(ValueError, match="clearance_m must be finite and positive") as rejected:
        linear_feature_sweep(start[:1], end[:1], face, face, clearance_m=0.0)
    record(
        tmp_path,
        "DD01",
        v,
        g,
        ref,
        dict(
            samples=rows,
            adapter_material=material,
            zero_clearance_reply=str(rejected.value),
            pair_first_t=None,
        ),
    )


def corner_sweeps(start, end, tet0, tet1, faces, rho):
    rows = []
    for f in faces:
        statuses, witnesses, references = [], [], []
        for p, q in zip(start, end):
            reference = corner_clearance_reference(tet0, tet1, p, q, f, rho)
            references.append(reference)
            sweep = linear_feature_sweep(p[None], q[None], tet0[f], tet1[f], clearance_m=rho)
            assert sweep.status == reference["status"]
            if sweep.witness_time is not None:
                u = sweep.witness_time
                assert (
                    closest_point_triangle(lerp(p, q, u), lerp(tet0[f], tet1[f], u)).distance < rho
                )
            statuses.append(sweep.status)
            witnesses.append(sweep.witness_time)
        rows.append(
            dict(face=f.tolist(), status=statuses, witness_t=witnesses, reference=references)
        )
    return rows


@pytest.mark.parametrize("name", RUNS["DD04"])
def test_dd04_sheet_area_and_derived_thickness(name, tmp_path):
    _, v, g = geometry_for("DD04", name)
    a, b = g["A"], g["B"]
    tet0, tet1 = (
        np.asarray(x, dtype=float) for x in (a["vertices"], a.get("vertices_end", a["vertices"]))
    )
    start, end = (np.array(b[k]) for k in ("vertices_start", "vertices_end"))
    rho = v.get("half_thickness_m", 0.0)
    ref = cloth_tet_patch(tet0, tet1, start, end, rho)
    e = v["expected"]
    expect_time(ref["first_t"], e["first_t"])
    boundary, _ = _tet_complex({0: tuple(a["tets"][0])})
    faces = np.asarray(boundary)
    assert len(faces) == 4
    (selected,) = [f for f in faces if set(f) == {0, 2, 3}]
    t = ref["first_t"] if ref["first_t"] is not None else 0.5
    tet, sheet = lerp(tet0, tet1, t), lerp(start, end, t)
    n = np.cross(sheet[1] - sheet[0], sheet[2] - sheet[0])
    n /= np.linalg.norm(n)
    assert abs(n[0]) == pytest.approx(1)
    side = 1 if n[0] > 0 else -1
    points = sheet + side * rho * n
    region, area = patch_clip(tet[selected], points, (1, 2))
    samples = []
    if ref["first_t"] is None:
        assert not region and area == 0 and ref["gap_lower_m"] == pytest.approx(e["gap_lower_m"])
    else:
        assert len(region) == 3 and area == pytest.approx(e["area_m2"], abs=1e-12)
        assert area == pytest.approx(ref["area_m2"], abs=1e-12)
        np.testing.assert_allclose(
            sorted(region), sorted([p[1:] for p in ref["points_m"]]), atol=1e-12, rtol=0
        )
        for p, expected_point, weights in zip(
            points, ref["points_m"], ref["tet_weights"], strict=True
        ):
            pair = closest_point_triangle(p, tet[selected])
            full = np.zeros(4)
            full[selected] = pair.second_weights
            np.testing.assert_allclose(pair.second, expected_point, atol=1e-12, rtol=0)
            np.testing.assert_allclose(full, weights, atol=1e-12, rtol=0)
            samples.append(dict(point_m=pair.second.tolist(), tet_weights=full.tolist()))
    split = None
    zero_reply = None
    sweeps = []
    if rho:
        sweeps = corner_sweeps(start, end, tet0, tet1, faces, rho)
        first_face = next(row for row in sweeps if row["face"] == selected.tolist())
        assert first_face["status"] == ["violation"] * 3
        assert all(w > ref["first_t"] for w in first_face["witness_t"])
        if name == "moving_tet_holdout":
            oblique = next(row for row in sweeps if set(row["face"]) == {1, 2, 3})
            assert oblique["status"] == ["certified", "certified", "violation"]
            later = oblique["reference"][2]
            assert later["proof"] == "terminal_layer_intersection"
            np.testing.assert_allclose(
                later["point_m"], [3 / 14, 3 / 16, 9 / 16], atol=1e-12, rtol=0
            )
        if "split_t" in v:
            cut = v["split_t"]
            middle, tm = lerp(start, end, cut), lerp(tet0, tet1, cut)
            left = corner_sweeps(start, middle, tet0, tm, faces, rho)
            right = corner_sweeps(middle, end, tm, tet1, faces, rho)
            assert all(r["status"] == ["certified"] * 3 for r in left)
            assert [r["status"] for r in right] == [r["status"] for r in sweeps]
            segment_w = next(r["witness_t"] for r in right if r["face"] == selected.tolist())
            mapped = [cut + (1 - cut) * u for u in segment_w]
            assert min(mapped) > ref["first_t"]
            split = dict(cut=cut, left=left, right=right, mapped_witness_t=mapped)
    else:
        with pytest.raises(ValueError, match="clearance_m must be finite and positive") as rejected:
            linear_feature_sweep(
                start[:1], end[:1], tet0[selected], tet1[selected], clearance_m=0.0
            )
        zero_reply = str(rejected.value)
    record(
        tmp_path,
        "DD04",
        v,
        g,
        ref,
        dict(
            supplied_face=selected.tolist(),
            owner=0,
            layer_side=side if rho else 0,
            half_thickness_m=rho,
            region_yz_m=region,
            area_m2=area,
            corner_samples=samples,
            positive_clearance=sweeps,
            zero_clearance_reply=zero_reply,
            split=split,
            pair_first_t=None,
        ),
    )
