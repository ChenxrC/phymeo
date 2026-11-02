"""M0 closed mesh, material boundary and volume-occupancy baselines."""

import json

import numpy as np
import pytest
from case_catalog import catalog_entries, geometry_for
from solid_case_oracles import (
    deforming_touch,
    plane_weights,
    sphere_bipyramid,
    static_tets,
    supporting_faces,
    u_solid_path,
)

from phymeo import TetrahedralModel, closest_point_triangle, linear_feature_sweep
from phymeo.geometry.shell_boundary import _surface, _tet_complex

RUNS = {
    "RR05": ("base", "flip_diagonal", "clear_notch", "permuted_hull", "left_arm_holdout"),
    "RD04": ("base", "reindexed", "clear", "internal_face_trap", "lower_holdout"),
    "DD05": ("base", "clear", "collapsed", "swapped", "external_touch", "deforming_holdout"),
}


@pytest.mark.parametrize("case_id", RUNS)
def test_third_batch_binds_variants(case_id):
    variants = catalog_entries()[case_id].get("m0_variants", [])
    assert tuple(v["name"] for v in variants) == RUNS[case_id]
    assert variants[0]["changes"] == {}
    assert {v["relation"] for v in variants} >= {"hit", "clear"}


def record(tmp_path, case_id, name, geometry, reference, observed):
    value = dict(
        case=case_id,
        variant=name,
        backend="host",
        geometry=geometry,
        reference=reference,
        observed=observed,
    )
    (tmp_path / "observation.json").write_text(
        json.dumps(value, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )


def triangulate_u(mesh, flip):
    ring = mesh["ring_xy"]
    assert len(ring) == 8
    vertices = np.array([[x, y, z] for z in mesh["z"] for x, y in ring], dtype=float)

    def split(i, j, k, last):
        return ((i, j, last), (j, k, last)) if flip else ((i, j, k), (i, k, last))

    # Conforming cap quads for this concave ring, including both re-entrant corners.
    cap = [tri for quad in ((1, 2, 3, 4), (0, 1, 4, 5), (0, 5, 6, 7)) for tri in split(*quad)]
    faces = [tri[::-1] for tri in cap] + [tuple(i + 8 for i in tri) for tri in cap]
    for i in range(8):
        j = (i + 1) % 8
        faces.extend(split(i, j, j + 8, i + 8))
    _surface(tuple(faces), "sphere")
    return vertices, np.array(faces)


def nearest_face(point, triangles):
    pairs = [closest_point_triangle(point, face) for face in triangles]
    face_id = min(range(len(pairs)), key=lambda i: pairs[i].distance)
    return face_id, pairs[face_id]


@pytest.mark.parametrize("name", RUNS["RR05"])
def test_rr05_concave_notch_and_containment(name, tmp_path):
    _, variant, geom = geometry_for("RR05", name)
    mesh, body = geom["A"], geom["B"]
    ref = u_solid_path(mesh, body)
    mesh_vertices, faces = triangulate_u(mesh, variant.get("flip_diagonal", False))
    triangles = mesh_vertices[faces]
    volume = sum(np.dot(x, np.cross(y, z)) for x, y, z in triangles) / 6
    assert len(faces) == 28 and volume == pytest.approx(ref["volume_m3"])
    assert volume == pytest.approx(10)
    if variant["relation"] == "clear":
        assert ref["first_t"] is None and [r["kind"] for r in ref["samples"]] == ["separated"] * 2
    else:
        expected = 8 / 11 if name != "left_arm_holdout" else 1 / 1.25
        assert ref["first_t"] == pytest.approx(expected, abs=1e-12, rel=0)
        assert [r["kind"] for r in ref["samples"]] == ["separated", "touch", "cut", "contained"]
    v = np.asarray(body["vertices"])
    start, end = np.array(body["translation_start"]), np.array(body["translation_end"])
    observations = []
    for row in ref["samples"]:
        t = row["t"]
        posed = v + start + t * (end - start)
        box_overlap = np.all(posed.min(axis=0) <= mesh_vertices.max(axis=0)) and np.all(
            posed.max(axis=0) >= mesh_vertices.min(axis=0)
        )
        assert box_overlap
        samples = [nearest_face(p, triangles) for p in posed]
        index = min(range(len(v)), key=lambda i: samples[i][1].distance)
        face_id, closest = samples[index]
        if row["kind"] == "cut":
            # A convex edge straddles the wall although no vertex itself touches it.
            low, high = posed[np.argmin(posed[:, 0])], posed[np.argmax(posed[:, 0])]
            u = (ref["wall_x"] - low[0]) / (high[0] - low[0])
            assert 0 < u < 1 and closest.distance > 1e-5
            cut_point = low + u * (high - low)
            _, pair = nearest_face(cut_point, triangles)
            assert pair.distance < 1e-12
        else:
            assert closest.distance == pytest.approx(row["boundary_lower_m"], abs=1e-12, rel=0)
            cut_point = None
        observations.append(
            dict(
                phase=row["phase"],
                t=t,
                whole_aabb_overlap=bool(box_overlap),
                closest_convex_vertex=index,
                mesh_face_id=face_id,
                mesh_point_m=closest.second.tolist(),
                face_weights=closest.second_weights.tolist(),
                vertex_surface_min_m=closest.distance,
                supplied_cut_point_m=cut_point.tolist() if cut_point is not None else None,
            )
        )
    record(
        tmp_path,
        "RR05",
        name,
        geom,
        ref,
        dict(
            boundary_triangles=len(faces),
            signed_mesh_volume_m3=volume,
            samples=observations,
            pair_first_t=None,
            pair_solid_classification=None,
        ),
    )


def tet_model(vertices, cells):
    return TetrahedralModel(
        vertices, cells, 1.0, 1.0, 0.25, np.zeros(len(vertices), dtype=bool), np.zeros(3)
    )


def exterior(model):
    cells = {i: tuple(map(int, tet)) for i, tet in enumerate(model.tetrahedra)}
    faces, _ = _tet_complex(cells)
    return list(faces)


@pytest.mark.parametrize("name", RUNS["RD04"])
def test_rd04_sphere_on_tet_exterior(name, tmp_path):
    case, variant, geom = geometry_for("RD04", name)
    soft, ball = geom["A"], geom["B"]
    vertices, cells = np.array(soft["vertices"], dtype=float), np.array(soft["tets"])
    expected = supporting_faces(vertices, cells)
    node_order = np.array(variant.get("node_order", range(len(vertices))))
    cell_order = np.array(variant.get("cell_order", range(len(cells))))
    model = tet_model(vertices[node_order], np.argsort(node_order)[cells[cell_order]])
    faces = exterior(model)
    original_keys = [tuple(sorted(int(node_order[i]) for i in face)) for face in faces]
    internal = tuple(case["oracle"]["excluded_internal_face"])
    assert set(original_keys) == set(expected) and internal not in original_keys
    assert len(faces) == len(expected) == case["oracle"]["exterior_triangles"]
    ref = sphere_bipyramid(ball)
    assert (ref["kind"] == "separated") == (variant["relation"] == "clear")
    start, end = (np.array(ball[key], dtype=float) for key in ("center_start", "center_end"))
    radius = ball["radius"]
    rows, hits = [], []
    for face, key in zip(faces, original_keys, strict=True):
        ids = np.array(face)
        triangle = model.rest_position[ids]
        owners = np.flatnonzero(np.sum(np.isin(model.tetrahedra, ids), axis=1) == 3)
        assert len(owners) == 1
        local_owner = int(owners[0])
        owner = int(cell_order[local_owner])
        assert owner == expected[key]
        (other,) = set(model.tetrahedra[local_owner]) - set(face)
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        assert normal @ (model.rest_position[other] - triangle[0]) < 0
        sweep = linear_feature_sweep(start[None], end[None], triangle, triangle, clearance_m=radius)
        assert sweep.status in {"certified", "violation"}
        row = dict(
            original_face=list(key),
            original_owner=owner,
            status=sweep.status,
            violation_witness_t=sweep.witness_time,
        )
        if sweep.status == "violation":
            assert ref["kind"] == "external_touch" and sweep.witness_time > ref["first_t"]
            p = start + ref["first_t"] * (end - start)
            probe = closest_point_triangle(p, triangle)
            assert probe.distance == pytest.approx(radius, abs=1e-12, rel=0)
            np.testing.assert_allclose(probe.second, ref["point_m"], atol=1e-12, rtol=0)
            weights = np.zeros(len(vertices))
            weights[node_order[ids]] = probe.second_weights
            np.testing.assert_allclose(weights, ref["node_weights"], atol=1e-12, rtol=0)
            hits.append(owner)
            row.update(observed_face_point_m=probe.second.tolist(), node_weights=weights.tolist())
        rows.append(row)
    extra = None
    if ref["kind"] == "external_touch":
        assert hits == [ref["owner"]]
        expected_t = 0.8 if name == "lower_holdout" else case["oracle"]["first_t"]
        assert ref["first_t"] == pytest.approx(expected_t, abs=1e-12, rel=0)
    else:
        assert not hits and ref["boundary_gap_lower_m"] > 0
        if ref["kind"] == "ball_contained":
            triangle = vertices[list(internal)]
            false_candidate = linear_feature_sweep(
                start[None], end[None], triangle, triangle, clearance_m=radius
            )
            assert false_candidate.status == "violation"
            extra = {
                "input_face": list(internal),
                "status": false_candidate.status,
                "witness_t": false_candidate.witness_time,
            }
    record(
        tmp_path,
        "RD04",
        name,
        geom,
        ref,
        dict(
            exterior_faces=rows,
            node_order=node_order.tolist(),
            cell_order=cell_order.tolist(),
            excluded_internal_face=list(internal),
            internal_face_negative_control=extra,
            pair_first_t=None,
            pair_solid_classification=None,
        ),
    )


def material_weights(model, current, points):
    # Pull back through the represented geometry, keeping the material inverse at rest.
    tet = current[model.tetrahedra[0]]
    inverse = model.rest_inverse[0]
    deformation = (tet[1:] - tet[0]).T @ inverse
    rest_delta = np.linalg.solve(deformation, (points - tet[0]).T)
    weights = (inverse @ rest_delta).T
    return np.column_stack((1 - weights.sum(axis=1), weights))


@pytest.mark.parametrize("name", RUNS["DD05"])
def test_dd05_volume_overlap_and_first_contact(name, tmp_path):
    _, variant, geom = geometry_for("DD05", name)
    bodies = [geom["A"], geom["B"]]
    vertices = [np.asarray(body["vertices"], dtype=float) for body in bodies]
    cells = [np.asarray(body["tets"]) for body in bodies]
    assert all(c.shape == (1, 4) for c in cells)
    tets = [v[c[0]] for v, c in zip(vertices, cells, strict=True)]
    if "vertices_end" in bodies[0]:
        ends = [np.asarray(body["vertices_end"]) for body in bodies]
        paths = [
            np.array([v[c[0]], e[c[0]]]) for v, e, c in zip(vertices, ends, cells, strict=True)
        ]
        ref = deforming_touch(*paths)
    else:
        ref = static_tets(*tets)
    if ref["kind"] == "invalid_geometry":
        assert variant["relation"] == "invalid"
        (bad,) = [i for i, v in enumerate(ref["volumes_m3"]) if v <= 0]
        with pytest.raises(ValueError, match="positive and nondegenerate") as rejected:
            tet_model(vertices[bad], cells[bad])
        record(
            tmp_path,
            "DD05",
            name,
            geom,
            ref,
            dict(
                invalid_body="AB"[bad],
                model_rejection=str(rejected.value),
                collision_query_submitted=False,
            ),
        )
        return

    assert (ref["kind"] == "separated") == (variant["relation"] == "clear")
    models = [tet_model(v, c) for v, c in zip(vertices, cells, strict=True)]
    faces = [np.asarray(exterior(model)) for model in models]
    observed = dict(
        reference_volumes_m3=[float(m.volume[0]) for m in models],
        pair_first_t=None,
        pair_solid_classification=None,
    )
    if ref["kind"] == "external_touch":
        t = ref["first_t"]
        assert t == pytest.approx(0.5 if name == "deforming_holdout" else 0.75, abs=1e-12, rel=0)
        current = [v + t * (e - v) for v, e in zip(vertices, ends, strict=True)]
        tip = current[1][cells[1][0, 0]]
        face_id, pair = nearest_face(tip, current[0][faces[0]])
        assert set(faces[0][face_id]) == {0, 2, 3} and pair.distance < 1e-12
        np.testing.assert_allclose(pair.second, ref["point_m"], atol=1e-12, rtol=0)
        weights = material_weights(models[0], current[0], tip[None])[0]
        np.testing.assert_allclose(weights, ref["a_weights"], atol=1e-12, rtol=0)
        end_w = [
            material_weights(m, e, np.array([ref["end_interior_point_m"]]))[0]
            for m, e in zip(models, ends, strict=True)
        ]
        assert all(min(row) > 0 for row in end_w)
        observed.update(
            supplied_event_face=faces[0][face_id].tolist(),
            face_point_m=pair.second.tolist(),
            tip_to_face_distance_m=pair.distance,
            volume_weights=weights.tolist(),
            end_interior_weights=[r.tolist() for r in end_w],
        )
    else:
        maps, unsigned = [], []
        for i, j in ((0, 1), (1, 0)):
            w = material_weights(models[i], vertices[i], tets[j])
            np.testing.assert_allclose(w, plane_weights(tets[i], tets[j]), atol=1e-12, rtol=0)
            maps.append(w)
            unsigned.append(
                min(nearest_face(p, vertices[i][faces[i]])[1].distance for p in tets[j])
            )
        assert min(unsigned) > 0
        if ref["kind"] == "contained_overlap":
            outer = "AB".index(ref["outer"])
            assert np.min(maps[outer]) > 0
            assert models[1 - outer].volume[0] == pytest.approx(ref["overlap_volume_m3"], rel=1e-12)
        else:
            assert min(unsigned) >= ref["separating_gap_lower_m"] - 1e-12
        observed.update(
            vertex_face_min_m=unsigned, minimum_mapped_weights=[float(w.min()) for w in maps]
        )
    record(tmp_path, "DD05", name, geom, ref, observed)
