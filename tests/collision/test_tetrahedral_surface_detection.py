"""Independent checks of volume exterior ownership through the public detector."""

import json
from dataclasses import asdict, replace
from fractions import Fraction

import numpy as np
import pytest
from case_catalog import geometry_for
from solid_case_oracles import sphere_bipyramid, supporting_faces
from tetrahedral_surface_cases import CASES, partners, scene

import phymeo as ph


def detect(model, paths):
    with ph.CollisionWorkspace(model) as work:
        return ph.detect(model, paths, ph.CollisionPolicy(), work)


def observed(tmp_path, out, key):
    (tmp_path / "observation.json").write_text(
        json.dumps([asdict(r) for r in partners(out, key)], allow_nan=False)
    )


@pytest.mark.parametrize("name", CASES["RD04"])
@pytest.mark.parametrize("reverse", [False, True])
def test_sphere_hits_only_the_mesh_exterior(name, reverse, tmp_path):
    model, paths, key = scene("RD04", name, reverse=reverse)
    _, _, g = geometry_for("RD04", name)
    ref = sphere_bipyramid(g["B"])
    rows = partners(detect(model, paths), key)
    assert all(r.status in {"hit", "surface_clear", "separated"} for r in rows), rows
    hits = [r for r in rows if r.status == "hit"]
    if ref["kind"] == "external_touch":
        (hit,) = hits
        owner = hit.second if reverse else hit.first
        assert owner.shape == ref["owner"]
        lo, hi = hit.sweep.first_t
        assert lo <= ref["first_t"] + 1e-15 and hi >= ref["first_t"] - 1e-15 and hi - lo <= 2**-40
        (f,) = [f for f in hit.facets if f.sweep.first_t is not None]
        assert set(f.face_ids) == {0, 2, 3 if not owner.shape else 4}
        w = f.sweep.witness
        weights = dict(w.second_weights if reverse else w.first_weights)
        assert set(weights) == set(next(s.vertex_ids for s in model.shapes if s.key == owner))
        assert all(weights[n] == pytest.approx(ref["node_weights"][n]) for n in weights)
        np.testing.assert_allclose(
            w.second_point if reverse else w.first_point, ref["point_m"], atol=1e-12, rtol=0
        )
    else:
        assert not hits
        if name == "internal_face_trap":
            assert all(
                r.status == "surface_clear"
                and r.witness.gap_m >= ref["boundary_gap_lower_m"] - 1e-12
                for r in rows
            )
            assert all(set(f.face_ids) != {0, 1, 2} for r in rows for f in r.facets)
    (tmp_path / "observation.json").write_text(
        json.dumps([asdict(r) for r in rows], allow_nan=False)
    )


def test_exterior_is_oriented_owned_and_immutable():
    model, _, _ = scene("RD04", "reindexed")
    _, _, geom = geometry_for("RD04", "base")
    verts = np.array(geom["A"]["vertices"])
    cells = geom["A"]["tets"]
    expected = supporting_faces(verts, cells)
    with ph.CollisionWorkspace(model) as work:
        faces = work.exterior_faces
        assert sum(map(len, faces.values())) == len(expected) == 6
        for key, loops in faces.items():
            for face in loops:
                assert expected[tuple(sorted(face))] == key.shape
                (opposite,) = set(cells[key.shape]) - set(face)
                a, b, c = verts[list(face)]
                assert np.cross(b - a, c - a) @ (verts[opposite] - a) < 0
        with pytest.raises(TypeError):
            faces[next(iter(faces))] = ()


@pytest.mark.parametrize("name", CASES["DD04"])
@pytest.mark.parametrize("reverse", [False, True])
def test_cloth_first_patch_and_later_oblique_facet(name, reverse, tmp_path):
    model, paths, key = scene("DD04", name, reverse=reverse)
    _, v, g = geometry_for("DD04", name)
    out = detect(model, paths)
    (row,) = partners(out, key)
    if name == "lateral_clear":
        assert row.status == "separated"
        observed(tmp_path, out, key)
        return
    assert row.status == "hit" and len(row.facets) == 4
    tm = v["expected"]["first_t"]
    lo, hi = row.sweep.first_t
    assert lo <= tm + 1e-15 and hi >= tm - 1e-15 and hi - lo <= 2**-40
    (patch,) = [f for f in row.facets if set(f.face_ids) == {0, 2, 3}]
    exact = Fraction(*patch.event_time_ratio)
    assert lo <= exact <= hi and float(exact) == pytest.approx(tm)
    region = patch.event_witness.intersection_points
    assert len(region) == 3
    vertices = np.array([p.point for p in region])
    assert np.linalg.norm(
        np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    ) / 2 == pytest.approx(v["expected"]["area_m2"])
    cloth = (1 - tm) * paths[key].start + tm * paths[key].end
    n = np.cross(cloth[1] - cloth[0], cloth[2] - cloth[0])
    n /= np.linalg.norm(n)
    cell = row.second if reverse else row.first
    tet = (1 - tm) * paths[cell].start + tm * paths[cell].end
    for p in region:
        wb = dict(p.first_weights if reverse else p.second_weights)
        wa = dict(p.second_weights if reverse else p.first_weights)
        height = p.first_height_m if reverse else p.second_height_m
        assert height == pytest.approx(
            v.get("half_thickness_m", 0) * (-1 if name == "reversed_thick" else 1)
        )
        assert set(wa) == {0, 1, 2, 3} and wa[1] == 0
        np.testing.assert_allclose(sum(w * tet[i] for i, w in wa.items()), p.point, atol=1e-12)
        np.testing.assert_allclose(
            sum(wb[i] * x for i, x in zip(v.get("order_B", range(3)), cloth, strict=True))
            + height * n,
            p.point,
            atol=1e-12,
        )
    if name == "moving_tet_holdout":
        (oblique,) = [f for f in row.facets if set(f.face_ids) == {1, 2, 3}]
        assert oblique.sweep.first_t[0] > hi
        assert oblique.witness.gap_m == 0 and oblique.witness.intersection_points
        point = np.array([3 / 14, 3 / 16, 9 / 16])
        assert any(
            np.linalg.norm(p.point - point) < 1e-12 for p in oblique.witness.intersection_points
        )
    observed(tmp_path, out, key)


def test_tetrahedral_topology_rejects_duplicate_equal_winding_and_three_owners():
    def build(cells):
        shapes = [
            ph.CollisionShape(ph.CollisionKey(3, 4, i), "tetrahedron", cell)
            for i, cell in enumerate(cells)
        ]
        with ph.CollisionWorkspace(ph.CollisionModel(shapes)):
            pass

    for cells in [
        [(0, 1, 2, 3), (1, 0, 2, 3)],
        [(0, 1, 2, 3), (0, 1, 2, 4)],
        [(0, 1, 2, 3), (0, 2, 1, 4), (0, 1, 2, 5)],
    ]:
        with pytest.raises(ValueError, match="duplicate tetrahedron|opposite owners"):
            build(cells)


def test_nonpositive_neighbor_invalidates_internal_cancellation():
    model, paths, key = scene("RD04", "base")
    lower = ph.CollisionKey(0, 0, 1)
    x = paths[lower].start
    y = x.copy()
    y[3, 2] = 1.0
    paths[lower] = ph.CollisionMotion(x, y)
    targeted = partners(detect(model, paths), key)
    assert any(r.candidate for r in targeted)
    assert all(r.reason == "tetrahedron_nonpositive_on_path" for r in targeted if r.candidate)


def test_valid_endpoints_cannot_hide_cell_collapse():
    model, paths, key = scene("DD04", "base")
    k = ph.CollisionKey(0, 0, 0)
    x = paths[k].start
    paths[k] = ph.CollisionMotion(x, x * [-1, -1, 1])
    (p,) = partners(detect(model, paths), key)
    assert p.status == "unresolved" and p.reason == "tetrahedron_nonpositive_on_path"


def test_cell_face_fully_inside_thick_sheet_at_entry():
    face, tet = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 1)
    model = ph.CollisionModel(
        (
            ph.CollisionShape(face, "triangle", (10, 20, 30), 1.0),
            ph.CollisionShape(tet, "tetrahedron", (0, 1, 2, 3)),
        )
    )
    x = np.array([[0, 0, 0], [0.125, 0, 0], [0, 0.125, 0], [0, 0, 0.125]])
    tri = [[-1, -1, 0], [3, -1, 0], [-1, 3, 0]]
    paths = {face: ph.CollisionMotion(tri), tet: ph.CollisionMotion(x, x + [0, 0, 0.25])}
    (p,) = detect(model, paths).pairs
    assert p.status == "hit" and p.sweep.first_t == (0.0, 0.0)
    assert all(len(f.event_witness.intersection_points) == 3 for f in p.facets)
    heights = {
        point.first_height_m for f in p.facets for point in f.event_witness.intersection_points
    }
    assert heights == {0.0, 0.125}


def test_changing_layer_normal_is_explicit_and_thin_variant_is_supported():
    model, paths, key = scene("DD04", "thick")
    x = paths[key]
    end = x.end.copy()
    end[2, 0] += 0.125
    paths[key] = ph.CollisionMotion(x.start, end)
    (row,) = partners(detect(model, paths), key)
    assert row.status == "unsupported" and row.reason == "unsupported_cloth_layer_director"
    thin, _, _ = scene("DD04", "base")
    (row,) = partners(detect(thin, paths), key)
    assert row.status == "hit"


def test_split_moving_tet_and_cloth_step_keeps_the_first_and_later_hits():
    model, paths, key = scene("DD04", "moving_tet_holdout")
    halves = []
    for left, right in ((0.0, 0.5), (0.5, 1.0)):
        sub = {
            k: ph.CollisionMotion(
                p.start + left * (p.end - p.start), p.start + right * (p.end - p.start)
            )
            for k, p in paths.items()
        }
        halves.append(partners(detect(model, sub), key)[0])
    assert halves[0].status in {"surface_clear", "separated"}
    assert halves[1].status == "hit"
    lo, hi = halves[1].sweep.first_t
    assert 0.5 + 0.5 * lo <= 9 / 14 <= 0.5 + 0.5 * hi
    assert {frozenset(f.face_ids) for f in halves[1].facets if f.sweep.first_t is not None} == {
        frozenset((0, 2, 3)),
        frozenset((1, 2, 3)),
    }


def test_masked_neighbor_still_cancels_its_shared_face():
    model, paths, ball = scene("RD04", "internal_face_trap")
    model = ph.CollisionModel(
        [replace(s, mask=0) if s.key == ph.CollisionKey(0, 0, 1) else s for s in model.shapes]
    )
    rows = partners(detect(model, paths), ball)
    assert [r.status for r in rows] == ["surface_clear", "filtered"]
    assert len(rows[0].facets) == 3


def test_rotated_rational_normal_layer_maps_material_to_the_original_cell():
    model, paths, key = scene("DD04", "moving_tet_holdout", reverse=True)
    # Dyadic similarity of a rational rotation, with a common translating frame.
    linear = np.array([[3, -4, 0], [4, 3, 0], [0, 0, 5.0]])
    shift = np.array([2.0, -1.0, 3.0])
    drift = np.array([0.5, -0.25, 0.125])
    shapes = [ph.CollisionShape(s.key, s.kind, s.vertex_ids, s.radius * 5) for s in model.shapes]
    model = ph.CollisionModel(tuple(reversed(shapes)))
    transformed = {
        k: ph.CollisionMotion(p.start @ linear.T + shift, p.end @ linear.T + shift + drift)
        for k, p in paths.items()
    }
    (row,) = partners(detect(model, transformed), key)
    assert row.status == "hit"
    assert row.sweep.first_t[0] <= 9 / 14 <= row.sweep.first_t[1]
    (f,) = [f for f in row.facets if set(f.face_ids) == {0, 2, 3}]
    assert Fraction(*f.event_time_ratio) == Fraction(9, 14)
    assert len(f.event_witness.intersection_points) == 3
    assert all(p.first_height_m == 5 / 32 for p in f.event_witness.intersection_points)


def test_exterior_classification_keeps_the_sign_when_reported_gap_rounds_to_zero():
    k, s = ph.CollisionKey(0, 0, 0), ph.CollisionKey(0, 1, 0)
    model = ph.CollisionModel(
        (
            ph.CollisionShape(k, "tetrahedron", (0, 1, 2, 3)),
            ph.CollisionShape(s, "sphere", (0,), 1.0),
        )
    )
    # Closest vertex is the origin. Squared center distance minus radius² is
    # exactly 2^-1080, while the surface gap rounds to zero in binary64.
    paths = {
        k: ph.CollisionMotion([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        s: ph.CollisionMotion([[-1.0, -(2.0**-540), 0.0]]),
    }
    (row,) = detect(model, paths).pairs
    assert row.witness.gap_m == 0.0
    assert row.status == "surface_clear"


def test_side_wall_enters_when_both_cloth_caps_stay_clear():
    tet, sheet = ph.CollisionKey(0, 0, 0), ph.CollisionKey(0, 1, 0)
    model = ph.CollisionModel(
        (
            ph.CollisionShape(tet, "tetrahedron", (0, 1, 2, 3)),
            ph.CollisionShape(sheet, "triangle", (7, 8, 9), 0.25),
        )
    )
    vertices = [
        [0, -0.125, 0.4375],
        [0.125, -0.125, 0.4375],
        [0, 0.125, 0.4375],
        [0, -0.125, 0.5625],
    ]
    cloth = np.array([[-1.0, -1, 0.5], [-0.25, -1, 0.5], [-0.25, 1, 0.5]])
    paths = {
        tet: ph.CollisionMotion(vertices),
        sheet: ph.CollisionMotion(cloth, cloth + [0.5, 0, 0]),
    }
    (row,) = detect(model, paths).pairs
    assert row.sweep.first_t == (0.5, 0.5)
    (f,) = [f for f in row.facets if set(f.face_ids) == {0, 2, 3}]
    points = f.event_witness.intersection_points
    assert len(points) == 3
    assert all(dict(p.second_weights)[7] == 0 and abs(p.second_height_m) <= 0.0625 for p in points)
    assert {p.point for p in points} == {tuple(vertices[i]) for i in (0, 2, 3)}
