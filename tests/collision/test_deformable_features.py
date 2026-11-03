"""The complete material VT/EE families and their independent Newton/analytic references."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph

DD = Path(__file__).resolve().parents[1] / "reference_newton" / "dd"
CAT = {r["id"]: r for r in json.loads((DD / "cases.json").read_text("utf-8"))["cases"]}


def register(case, *, world=0, mirrored=False):
    shapes = []
    motion = {}
    for sheet in case["inputs"]["sheets"]:
        pos = {id_: np.asarray(x, dtype=float) for id_, *x in sheet["vertices"]}
        for i, face in enumerate(sheet["faces"]):
            if mirrored:
                face = list(reversed(face))
            s = ph.CollisionShape(ph.CollisionKey(world, sheet["body"], i), "triangle", tuple(face))
            shapes.append(s)
            p = np.array([pos[id_] for id_ in face])
            if mirrored:
                p = p * [-1, 1, 1] + [1 / 4, -1 / 2, 1 / 8]
            motion[s.key] = ph.CollisionMotion(p)
    return shapes, motion


def query(model, paths, work, *, search=3 / 32, force=1 / 16, **bounds):
    return ph.query_deformable_features(
        model, paths, work, search_distance_m=search, contact_distance_m=force, **bounds
    )


def key(row):
    a, b = tuple(x[2] for x in row.first_nodes), tuple(x[2] for x in row.second_nodes)
    return row.kind, a, b


def material_check(row, by_node):
    for lead, other in row.support_face_pairs:
        assert (lead.world, lead.body) == row.first_nodes[0][:2]
        assert (other.world, other.body) == row.second_nodes[0][:2]
    for q in row.closest:
        for group, weights, p in (
            (row.first_nodes, q.first_weights, q.first_point),
            (row.second_nodes, q.second_weights, q.second_point),
        ):
            coords = {n[-1]: by_node[n] for n in group}
            assert tuple(i for i, _ in weights) == tuple(sorted(coords))
            assert all(w >= 0 for _, w in weights) and sum(w for _, w in weights) == pytest.approx(
                1, abs=3e-14
            )
            np.testing.assert_allclose(
                p, sum(w * coords[i] for i, w in weights), rtol=0, atol=3e-13
            )
        d = np.asarray(q.second_point) - q.first_point
        assert np.linalg.norm(d) == pytest.approx(row.distance_m, abs=2e-14)
        if row.normal is None:
            assert row.distance_m == 0
        else:
            np.testing.assert_allclose(row.normal, d / row.distance_m, rtol=0, atol=1e-12)


def readbacks(rows, device):
    # Native directed EE records may choose any point on a parallel closest span.
    # Check the same interpolation parameter for both bodies and their weights.
    ref = json.loads((DD / "native_fold_features.json").read_text("utf-8"))["devices"][device][
        "records"
    ]
    seen = set()
    for kind, native, a, b, wa, wb, pa, pb in ref:
        if kind == "EE" and tuple(sorted(a)) > tuple(sorted(b)):
            a, b, wa, wb, pa, pb = b, a, wb, wa, pb, pa
        k = kind, tuple(sorted(a)), tuple(sorted(b))
        seen.add(k)
        row = rows[k]

        def vector(first, second, points_a, points_b):
            return np.asarray(
                [
                    *(v for _, v in sorted(first)),
                    *(v for _, v in sorted(second)),
                    *points_a,
                    *points_b,
                ]
            )

        actual = vector(zip(a, wa, strict=True), zip(b, wb, strict=True), pa, pb)
        ends = [
            vector(q.first_weights, q.second_weights, q.first_point, q.second_point)
            for q in row.closest
        ]
        d = ends[-1] - ends[0]
        t = 0.0 if not np.any(d) else np.clip((actual - ends[0]) @ d / (d @ d), 0, 1)
        np.testing.assert_allclose(
            ends[0] + t * d, actual, rtol=0, atol=3e-6, err_msg=str((kind, native))
        )
    assert seen == rows.keys()


def test_fold_features_match_every_record_from_native_cpu_and_cuda(tmp_path):
    case = CAT["DD-REMOTE-FOLD"]
    shapes, paths = register(case)
    model = ph.CollisionModel(tuple(shapes))
    with ph.CollisionWorkspace(model, "host") as work:
        last = ph.detect(model, paths, ph.CollisionPolicy(), work)
        obs = query(model, paths, work)
        assert work.last_result is last
    assert obs.status == "ok" and obs.required_features == 32
    rows = {key(r): r for r in obs.contacts}
    assert len(rows) == 32 and sum(r.kind == "VT" for r in obs.contacts) == 8
    assert all(
        r.force_eligible and r.inside_force_distance and r.distance_m == 3 / 64
        for r in obs.contacts
    )
    shallow = sum(128 * (1 / 16 - r.distance_m) for r in rows.values() if r.force_eligible)
    assert shallow == 64.0
    for device in ("cpu", "cuda:0"):
        readbacks(rows, device)
    by_node = {
        (s.key.world, s.key.body, v): p
        for s in shapes
        for v, p in zip(s.vertex_ids, paths[s.key].start, strict=True)
    }
    for row in rows.values():
        material_check(row, by_node)
    (tmp_path / "features.json").write_text(
        json.dumps(
            dict(
                vt=8,
                ee=24,
                broad_face_pairs=obs.broad_face_pairs,
                feature_tests=obs.feature_tests,
                shallow_linear_sum_n=shallow,
                parallel_spans=sum(len(x.closest) == 2 for x in rows.values()),
            )
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("id_", list(CAT))
def test_frozen_feature_identity_radii_and_independent_exhaustive_oracle(id_):
    case = CAT[id_]
    shapes, paths = register(case)
    model = ph.CollisionModel(shapes)
    with ph.CollisionWorkspace(model, "host") as work:
        result = query(model, paths, work)
    vt, ee = runpy.run_path(str(DD / "oracles.py"))["feature_sets"](case["inputs"])
    # The independent oracle's legacy IDs are globally unique but its EE keys
    # use only those local numbers. The public feature protocol qualifies each
    # endpoint by (world,body,id) before ordering either side.
    owners = {n: (s.key.world, s.key.body, n) for s in shapes for n in s.vertex_ids}
    expected = {("VT", (n,), face): d for (n, face), d in vt.items()}
    for pair, distance in ee.items():
        e, f = sorted(pair, key=lambda face: tuple(owners[i] for i in face))
        expected["EE", e, f] = distance
    assert result.status == "ok" and result.required_features == len(expected)
    assert {key(r) for r in result.contacts} == expected.keys()
    for row in result.contacts:
        assert abs(row.distance_m - expected[key(row)]) <= 1e-12
        assert row.inside_force_distance == (row.distance_m < 1 / 16)
        assert row.force_eligible == (0 < row.distance_m < 1 / 16)
        assert row.gap_to_force_m == pytest.approx(row.distance_m - 1 / 16, abs=1e-14)


def fixture(items):
    shapes = []
    paths = {}
    for world, body, index, ids, xyz in items:
        s = ph.CollisionShape(ph.CollisionKey(world, body, index), "triangle", ids)
        shapes.append(s)
        paths[s.key] = ph.CollisionMotion(xyz)
    return ph.CollisionModel(shapes), paths


def test_parallel_span_singleton_overlap_zero_normal_and_same_ids_on_other_body():
    ids = (7, 13, 19)
    first = ((-1, 0, 0), (1, 0, 0), (0, -1, 0))
    second = ((-0.5, 0, 1 / 32), (1.5, 0, 1 / 32), (0, 1, 1 / 32))
    model, paths = fixture([(0, 2, 0, ids, first), (0, 4, 0, ids, second), (1, 2, 0, ids, first)])
    with ph.CollisionWorkspace(model, "host") as work:
        obs = query(model, paths, work)
        row = next(r for r in obs.contacts if key(r) == ("EE", (7, 13), (7, 13)))
        assert len(row.closest) == 2
        assert row.first_nodes == ((0, 2, 7), (0, 2, 13)) and row.second_nodes == (
            (0, 4, 7),
            (0, 4, 13),
        )
        np.testing.assert_array_equal(
            [q.first_point for q in row.closest], [[-0.5, 0, 0], [1, 0, 0]]
        )
        assert row.normal == (0.0, 0.0, 1.0)
        actual = dict(paths)
        actual[ph.CollisionKey(0, 4, 0)] = ph.CollisionMotion(np.asarray(second) - [0, 0, 1 / 32])
        crossing = query(model, actual, work)
        zero = next(r for r in crossing.contacts if key(r) == key(row))
        assert (
            zero.distance_m == 0
            and zero.inside_force_distance
            and not zero.force_eligible
            and zero.normal is None
        )
    assert all(r.first_nodes[0][0] == r.second_nodes[0][0] == 0 for r in obs.contacts)


def test_shared_endpoint_filters_and_one_ring_exclusion_are_distinct():
    a = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    b = [a[1], a[0], (0.25, 0.25, 1 / 32)]
    m, p = fixture([(0, 4, 0, (10, 20, 30), a), (0, 4, 1, (20, 10, 40), b)])
    with ph.CollisionWorkspace(m, "host") as work:
        one = query(m, p, work, topology_rings=1)
        two = query(m, p, work, topology_rings=2)
    assert ("VT", (40,), (10, 20, 30)) in {key(r) for r in one.contacts}
    assert two.contacts == () and two.status == "ok"
    for r in one.contacts:
        assert set(r.first_nodes).isdisjoint(r.second_nodes)


def test_ownership_capacity_degeneracy_motion_and_filter_statuses():
    shapes, p = register(CAT["DD-REMOTE-FOLD"])
    m = ph.CollisionModel(shapes)
    with ph.CollisionWorkspace(m, "host") as work:
        result = query(m, p, work, feature_capacity=31)
        assert (
            result.status == "capacity_exceeded"
            and result.reason == "feature_capacity"
            and result.required_features == 32
            and result.contacts == ()
        )
        budget = query(m, p, work, max_feature_tests=1)
        assert (
            budget.status == "capacity_exceeded"
            and budget.reason == "feature_test_capacity"
            and budget.contacts == ()
            and budget.required_features is None
        )
        bad = dict(p)
        face = shapes[0]
        xyz = bad[face.key].start.copy()
        xyz[1] += [0.125, 0, 0]
        bad[face.key] = ph.CollisionMotion(xyz)
        bad_row = query(m, bad, work)
        assert (
            bad_row.status == "invalid_geometry"
            and bad_row.reason == "inconsistent_material_position"
            and not bad_row.contacts
        )
        moved = dict(p)
        moved[face.key] = ph.CollisionMotion(p[face.key].start, p[face.key].start + [0, 0, 0.03125])
        assert query(m, moved, work).reason == "discrete_pose_required"
    tri = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
    m, p = fixture(
        [(0, 1, 0, (1, 2, 3), tri), (0, 2, 0, (4, 5, 6), [(0, 0, 1), (1, 0, 1), (2, 0, 1)])]
    )
    with ph.CollisionWorkspace(m, "host") as work:
        failed = query(m, p, work)
        assert (
            failed.status == "invalid_geometry"
            and failed.reason == "degenerate_face"
            and failed.invalid_faces == (m.shapes[1].key,)
        )
    _, original = register(CAT["DD-REMOTE-FOLD"])
    both = ph.CollisionModel(shapes[:2], excluded=((shapes[0].key, shapes[1].key),))
    with ph.CollisionWorkspace(both, "host") as work:
        restricted = query(both, {s.key: original[s.key] for s in shapes[:2]}, work)
    assert restricted.broad_face_pairs == 0 and restricted.contacts == ()


def test_reorder_and_reflection_keep_canonical_material_coordinates():
    case = CAT["DD-REMOTE-FOLD"]
    observations = []
    for mirror in (False, True):
        shapes, p = register(case, mirrored=mirror)
        m = ph.CollisionModel(tuple(reversed(shapes)))
        with ph.CollisionWorkspace(m, "host") as work:
            observations.append(query(m, p, work))
    plain, mirror = observations
    assert [key(r) for r in plain.contacts] == [key(r) for r in mirror.contacts]
    for a, b in zip(plain.contacts, mirror.contacts, strict=True):
        assert a.distance_m == b.distance_m and a.support_face_pairs == b.support_face_pairs
        np.testing.assert_allclose(b.normal, np.asarray(a.normal) * [-1, 1, 1], rtol=0, atol=2e-14)
        for x, y in zip(a.closest, b.closest, strict=True):
            assert x.first_weights == y.first_weights and x.second_weights == y.second_weights
            np.testing.assert_allclose(
                np.asarray(x.first_point) * [-1, 1, 1] + [0.25, -0.5, 0.125],
                y.first_point,
                rtol=0,
                atol=2e-13,
            )


def test_same_edge_two_registered_faces_is_one_pair_with_eligible_supports_only():
    a = [(-1, 0, 0), (1, 0, 0), (0, -0.75, -0.75)]
    side = [a[1], a[0], (0, 0.75, -0.75)]
    b = [(0, -1, 3 / 64), (0, 1, 3 / 64), (0.75, 0, 0.75 + 3 / 64)]
    m, p = fixture(
        [(0, 7, 0, (11, 23, 37), a), (0, 7, 1, (23, 11, 41), side), (0, 13, 0, (53, 67, 79), b)]
    )
    target = ("EE", (11, 23), (53, 67))
    faces = [s.key for s in m.shapes]
    for excluded, wanted in (
        ((), 2),
        (((faces[0], faces[2]),), 1),
        (((faces[0], faces[2]), (faces[1], faces[2])), 0),
    ):
        filtered = ph.CollisionModel(m.shapes, excluded)
        with ph.CollisionWorkspace(filtered, "host") as work:
            out = query(filtered, p, work)
        found = [r for r in out.contacts if key(r) == target]
        assert len(found) == bool(wanted)
        if found:
            assert len(found[0].support_face_pairs) == wanted and found[0].force_eligible


def test_sub_ulp_thresholds_are_compared_before_scalar_rounding():
    shapes, p = register(CAT["DD-VT-TWO-SHEETS"])
    model = ph.CollisionModel(shapes)
    source = shapes[1].key
    target = ("VT", (53,), (11, 23, 37))
    with ph.CollisionWorkspace(model, "host") as work:
        for threshold in (1 / 16, 3 / 32):
            for h in (np.nextafter(threshold, 0.0), threshold, np.nextafter(threshold, 1.0)):
                q = p[source].start.copy()
                q[0, 2] = h
                out = query(model, {**p, source: ph.CollisionMotion(q)}, work)
                found = {key(r): r for r in out.contacts}
                assert (target in found) == (h < 3 / 32)
                if target in found:
                    v = found[target]
                    assert (
                        v.force_eligible == (h < 1 / 16)
                        and v.distance_bounds_m[0] <= h <= v.distance_bounds_m[1]
                    )
                    assert v.gap_to_force_bounds_m[0] <= h - 1 / 16 <= v.gap_to_force_bounds_m[1]


def test_nearly_parallel_but_skew_segments_have_one_closest_material_pair():
    e = 2.0**-23
    a = [(-1, 0, 0), (1, 0, 0), (0, -1, -0.75)]
    b = [(-1, -e, 1 / 32), (1, e, 1 / 32), (0, 1, 0.75)]
    model, paths = fixture([(0, 1, 0, (11, 23, 37), a), (0, 2, 0, (53, 67, 79), b)])
    with ph.CollisionWorkspace(model, "host") as work:
        out = query(model, paths, work)
    row = next(x for x in out.contacts if key(x) == ("EE", (11, 23), (53, 67)))
    assert row.distance_m == 1 / 32 and row.normal == (0.0, 0.0, 1.0) and len(row.closest) == 1
    q = row.closest[0]
    assert q.first_weights == ((11, 0.5), (23, 0.5)) and q.second_weights == ((53, 0.5), (67, 0.5))
    assert q.first_point == (0.0, 0.0, 0.0) and q.second_point == (0.0, 0.0, 1 / 32)


def test_duplicate_face_nonmanifold_edge_and_cross_owner_reuse_are_explicit():
    abc = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    reversed_ = [abc[1], abc[0], abc[2]]
    m, p = fixture([(0, 2, 0, (4, 8, 16), abc), (0, 2, 1, (8, 4, 16), reversed_)])
    with ph.CollisionWorkspace(m, "host") as work:
        out = query(m, p, work)
    assert out.reason == "duplicate_material_face" and out.invalid_faces == tuple(
        s.key for s in m.shapes
    )
    m, p = fixture(
        [
            (0, 2, 0, (4, 8, 16), abc),
            (0, 2, 1, (8, 4, 17), [abc[1], abc[0], (0, -1, 0)]),
            (0, 2, 2, (8, 4, 18), [abc[1], abc[0], (0, 0, 1)]),
        ]
    )
    with ph.CollisionWorkspace(m, "host") as work:
        out = query(m, p, work)
    assert out.reason == "nonmanifold_material_edge" and out.invalid_faces == tuple(
        s.key for s in m.shapes
    )
    assert not out.contacts and out.required_features is None
    independent, p = fixture(
        [(0, 2, 0, (4, 8, 16), abc), (0, 3, 0, (4, 8, 16), np.asarray(abc) + [0, 0, 1 / 32])]
    )
    with ph.CollisionWorkspace(independent, "host") as work:
        out = query(independent, p, work)
    assert out.status == "ok" and out.contacts


def test_empty_single_face_and_return_capacity_zero_keep_complete_counts():
    shapes, paths = register(CAT["DD-VT-TWO-SHEETS"])
    models = (ph.CollisionModel(()), ph.CollisionModel(shapes[:1]), ph.CollisionModel(shapes))
    for m, n in zip(models, (0, 0, 1), strict=True):
        with ph.CollisionWorkspace(m, "host") as work:
            result = query(m, {s.key: paths[s.key] for s in m.shapes}, work, feature_capacity=0)
            assert result.required_features == n and result.contacts == ()
            assert result.status == ("capacity_exceeded" if n else "ok")
            with pytest.raises(ValueError, match="contact_distance_m"):
                query(m, {s.key: paths[s.key] for s in m.shapes}, work, search=0.1, force=0.2)
