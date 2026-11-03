"""Discrete material observations have independent analytic point, chart and coverage oracles."""

import numpy as np
import pytest

import phymeo as ph


def shape(kind, ids, *, world=0, body=1, index=0, radius=0, mask=2**31 - 1):
    return ph.CollisionShape(
        ph.CollisionKey(world, body, index), kind, ids, radius=radius, mask=mask
    )


def grid(values, *, origin=(0.0, 0.0, 0.0), step=(1.0, 1.0, 1.0), dims=(2, 2, 2), band=None):
    z, y, x = np.indices(dims[::-1], dtype=float)
    v = values(origin[0] + step[0] * x, origin[1] + step[1] * y, origin[2] + step[2] * z)
    return ph.SignedDistanceGrid(
        np.broadcast_to(v, dims[::-1]), origin_m=origin, spacing_m=step, band_m=band
    )


def setup(field, *, pose=None, solids=(), materials=(), excluded=()):
    rigid = shape("sphere", (70,), body=0, radius=1)
    model = ph.CollisionModel((rigid, *solids, *materials), excluded)
    frame = ph.SdfFieldPose(rigid.key, field, **(pose or {}))
    return model, frame


def one(batch):
    assert len(batch.results) == 1
    return batch.results[0]


def check_material(row, shape, points):
    by_id = dict(zip(shape.vertex_ids, points, strict=True))
    for obs in row.samples:
        assert tuple(sorted(obs.weights)) == obs.weights
        assert sum(w for _, w in obs.weights) == pytest.approx(1, abs=2e-14)
        assert all(w > 0 for _, w in obs.weights)
        p = sum(w * np.array(by_id[i]) for i, w in obs.weights)
        np.testing.assert_allclose(obs.world_point_m, p, rtol=0, atol=4e-14)
        assert not hasattr(obs, "mesh_face") and not hasattr(obs, "rigid_weights")


def analytic_triangle():
    # On x+y+z=1, xy+xz+yz is at most 1/3. On each edge it is at most 1/4.
    g = grid(lambda x, y, z: 5 / 16 - x * y - y * z - z * x)
    s = shape("triangle", (47, 5, 29))
    p = np.eye(3)
    return g, s, p


def test_nonvertex_interior_contact_with_positive_entire_boundary():
    g, s, p = analytic_triangle()
    model, f = setup(g, materials=(s,))
    row = one(ph.sample_sdf_contacts(model, f, {s.key: p}))
    assert row.field_status == "contact" and row.source_geometry_status == "unbound"
    assert row.field_gap_bounds_m[0] <= -1 / 48 <= row.field_gap_bounds_m[1] + 1e-15
    inner = min((v for v in row.samples if v.status == "valid"), key=lambda q: q.field_gap_m)
    assert inner.field_gap_m == pytest.approx(-1 / 48, abs=2e-15)
    np.testing.assert_allclose(inner.world_point_m, [1 / 3] * 3, atol=1e-15)
    assert [i for i, _ in inner.weights] == [5, 29, 47]
    np.testing.assert_allclose(inner.gradient_world, [-2 / 3] * 3, atol=2e-15)
    assert inner.projected_surface_estimate_world_m is not None
    check_material(row, s, p)


def test_exact_triangle_clear_requires_a_surface_bound_and_budget_is_explicit():
    s = shape("triangle", (1, 5, 9))
    p = np.eye(3)
    g = grid(lambda x, y, z: 13 / 32 - x * y - y * z - z * x)
    model, f = setup(g, materials=(s,))
    whole = one(ph.sample_sdf_contacts(model, f, {s.key: p}, max_depth=3))
    assert whole.field_status == "clear"
    exact = 13 / 32 - 1 / 3
    assert 0 < whole.field_gap_bounds_m[0] <= exact <= whole.field_gap_bounds_m[1] + 1e-15
    loose = one(ph.sample_sdf_contacts(model, f, {s.key: p}, max_depth=0))
    assert loose.field_status == "unresolved" and "field_precision" in loose.reasons
    limited = one(ph.sample_sdf_contacts(model, f, {s.key: p}, max_samples=1))
    assert limited.field_status == "unresolved" and "sample_capacity" in limited.reasons


def test_discrete_radius_world_filters_masks_and_projection_are_decoupled():
    g = grid(lambda x, y, z: x - 0.5)
    a = shape("particle", (7,), index=0)
    b = shape("rounded_particle", (7,), index=1, radius=0.25)
    far = shape("rounded_particle", (7,), world=1, radius=0.25)
    masked = shape("particle", (7,), body=2, mask=0)
    excluded = shape("particle", (7,), body=3)
    rigid = shape("sphere", (70,), body=0, radius=1)
    model, f = setup(
        g, materials=(a, b, far, masked, excluded), excluded=((rigid.key, excluded.key),)
    )
    at = [[0.625, 0.5, 0.5]]
    results = ph.sample_sdf_contacts(model, f, {s.key: at for s in (a, b, far, masked, excluded)})
    rows = {r.material_key: r for r in results.results}
    assert rows[a.key].field_status == "clear" and rows[b.key].field_status == "contact"
    q = rows[b.key].samples[0]
    assert q.field_distance_m == 0.125 and q.field_gap_m == -0.125 and q.weights == ((7, 1.0),)
    assert q.projected_surface_estimate_world_m == (0.5, 0.5, 0.5)
    assert rows[far.key].reasons == ("different_world",) and rows[masked.key].reasons == ("mask",)
    assert rows[excluded.key].reasons == ("excluded",) and not rows[far.key].samples
    assert ph.sample_sdf_contacts(model, f, {}).results == ()
    with pytest.raises((TypeError, ValueError), match="discrete"):
        ph.sample_sdf_contacts(model, f, {a.key: ph.CollisionMotion(at, [[0.125, 0.5, 0.5]])})


def test_partial_domain_and_band_do_not_certify_whole_face_clear():
    a = shape("triangle", (1, 5, 8))
    p = np.array([[-0.5, 0.5, 0.5], [0.75, 0.25, 0.5], [0.75, 0.75, 0.5]])
    g = grid(lambda x, y, z: x + 1)
    model, f = setup(g, materials=(a,))
    partial = one(ph.sample_sdf_contacts(model, f, {a.key: p}))
    assert partial.field_status == "unresolved" and "outside_grid" in partial.reasons
    assert partial.field_gap_bounds_m[0] is None
    _, neg = setup(grid(lambda x, y, z: x - 0.25), materials=(a,))
    hit = one(ph.sample_sdf_contacts(model, neg, {a.key: p}))
    assert hit.field_status == "contact" and "outside_grid" in hit.reasons
    assert hit.field_gap_bounds_m[0] is None and hit.field_gap_bounds_m[1] < 0
    _, band = setup(grid(lambda x, y, z: x - 0.5, band=0.45), materials=(a,))
    cut = one(ph.sample_sdf_contacts(model, band, {a.key: p + 0.125}))
    assert cut.field_status == "unresolved" and "outside_band" in cut.reasons
    outside = one(ph.sample_sdf_contacts(model, f, {a.key: p + [2, 0, 0]}))
    assert outside.field_status == "unresolved" and outside.field_gap_bounds_m == (None, None)


def test_multicell_edge_feature_full_pose_and_material_reordering():
    g = grid(
        lambda x, y, z: np.maximum.reduce([np.abs(x), np.abs(y), np.abs(z)]) - 0.25,
        origin=(-1, -1, -1),
        step=(0.25, 0.25, 0.5),
        dims=(9, 9, 5),
    )
    s = shape("triangle", (29, 3, 17))
    p = np.array([[-1, 0, 0], [1, 0, 0], [1, 1, 0.5]])
    model, f = setup(g, materials=(s,))
    row = one(ph.sample_sdf_contacts(model, f, {s.key: p}))
    assert row.field_status == "contact" and row.cells_covered > 1
    edge = next(q for q in row.samples if np.array_equal(q.local_point_m, (0, 0, 0)))
    assert edge.weights == ((3, 0.5), (29, 0.5)) and edge.field_distance_m == -0.25
    turned = shape("triangle", (17, 3, 29))
    shift = np.array([2.0, -3.0, 0.5])
    pose = dict(translation_m=shift, orientation_xyzw=(0.5, 0.5, 0.5, 0.5))
    m2, f2 = setup(g, materials=(turned,), pose=pose)
    wp = np.roll(p[::-1], 1, axis=1) + shift
    other = one(ph.sample_sdf_contacts(m2, f2, {turned.key: wp}))
    assert other.field_status == "contact" and other.field_gap_bounds_m == row.field_gap_bounds_m
    same = next(q for q in other.samples if q.weights == edge.weights)
    np.testing.assert_allclose(same.gradient_world, np.roll(edge.gradient_world, 1), atol=1e-14)
    check_material(other, turned, wp)
    capped = one(ph.sample_sdf_contacts(model, f, {s.key: p}, max_cells=1))
    assert capped.field_status == "unresolved" and "cell_capacity" in capped.reasons


def test_baked_source_requires_registered_topology_vertices_and_pose():
    from test_sdf_grid import asset

    mesh, v = asset("u")
    s = shape("triangle", (33, 15, 21), body=1)
    model = ph.CollisionModel((mesh, s))
    frame = ph.SdfFieldPose.bake(
        mesh,
        v,
        shape_xyz=(13, 13, 7),
        origin_m=(-3, -3, -1.5),
        spacing_m=(0.5, 0.5, 0.5),
        translation_m=(2, -1, 0.5),
        orientation_xyzw=(0.5, 0.5, 0.5, 0.5),
    )

    def tx(p):
        return np.roll(p, 1, axis=1) + [2, -1, 0.5]

    rigid = tx(v)
    p = tx(np.array([[-0.125, 0, 0], [0.125, 0, 0], [0, 0.25, 0.0]]))
    row = one(ph.sample_sdf_contacts(model, frame, {s.key: p}, rigid_vertices_world_m=rigid))
    assert (
        row.field_status == row.source_geometry_status == "clear" and row.source_gap_bounds_m[0] > 0
    )
    no_rigid = one(ph.sample_sdf_contacts(model, frame, {s.key: p}))
    assert no_rigid.field_status == "clear" and no_rigid.source_geometry_status == "pose_unverified"
    alternate = rigid.copy()
    alternate[0, 0] += 0.03125
    mismatch = one(
        ph.sample_sdf_contacts(model, frame, {s.key: p}, rigid_vertices_world_m=alternate)
    )
    assert (
        mismatch.source_geometry_status == "pose_unverified"
        and mismatch.source_gap_bounds_m == (None, None)
    )
    wrong = ph.CollisionShape(
        mesh.key, "mesh", mesh.vertex_ids, triangles=mesh.triangles, revision=1
    )
    changed = one(
        ph.sample_sdf_contacts(
            ph.CollisionModel((wrong, s)), frame, {s.key: p}, rigid_vertices_world_m=rigid
        )
    )
    assert changed.source_geometry_status == "pose_unverified"
    straight = ph.SdfFieldPose(
        mesh.key,
        frame.grid,
        translation_m=frame.translation_m,
        orientation_xyzw=frame.orientation_xyzw,
    )
    assert (
        one(
            ph.sample_sdf_contacts(model, straight, {s.key: p}, rigid_vertices_world_m=rigid)
        ).source_geometry_status
        == "unbound"
    )


def test_coarse_thin_wall_may_be_clear_in_field_and_uncertain_for_source():
    from test_sdf_grid import asset

    mesh, v = asset("thin")
    q = shape("particle", (91,))
    m = ph.CollisionModel((mesh, q))
    f = ph.SdfFieldPose.bake(
        mesh, v, shape_xyz=(9, 9, 9), origin_m=(-1, -1, -1), spacing_m=(0.25,) * 3
    )
    row = one(ph.sample_sdf_contacts(m, f, {q.key: [[0.125, 0, 0]]}, rigid_vertices_world_m=v))
    assert row.field_status == "clear" and row.samples[0].field_distance_m == 7 / 64
    assert (
        row.samples[0].gradient_world == (0, 0, 0)
        and row.samples[0].projected_surface_estimate_world_m is None
    )
    assert row.source_geometry_status == "unresolved"
    lo, hi = row.source_gap_bounds_m
    assert lo <= -1 / 64 <= hi


def test_degenerate_material_and_nondiscrete_or_unknown_keys():
    s = shape("triangle", (1, 2, 3))
    t = shape("triangle", (1, 2, 3), body=2, radius=0.125)
    g = grid(lambda x, y, z: x)
    m, f = setup(g, materials=(s, t))
    p = np.array([[0, 0, 0], [0, 0, 0.5], [0, 0, 1.0]])
    result = ph.sample_sdf_contacts(m, f, {s.key: p, t.key: p})
    assert result.results[0].reasons == ("degenerate_triangle",)
    assert result.results[1].field_status == "unsupported"
    with pytest.raises(ValueError, match="registered"):
        ph.sample_sdf_contacts(m, f, {ph.CollisionKey(0, 9, 0): p})


def test_isolated_edge_contact_and_disjoint_coincident_worlds():
    g = grid(lambda x, y, z: (x - 0.5) * (z - 0.5) + y)
    wall = shape("sphere", (88,), body=0, radius=1)
    other = shape("sphere", (88,), world=1, body=0, radius=1)
    empty = shape("sphere", (88,), world=2, body=0, radius=1)
    s = shape("triangle", (5, 11, 17))
    copy = shape("triangle", (5, 11, 17), world=1)
    model = ph.CollisionModel((wall, other, empty, s, copy))
    p = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 0.0]])
    points = {s.key: p, copy.key: p}
    first = ph.SdfFieldPose(wall.key, g)
    second = ph.SdfFieldPose(other.key, g, translation_m=(0, -0.125, 0))
    zero, foreign = ph.sample_sdf_contacts(model, first, points).results
    assert zero.field_status == "contact" and foreign.reasons == ("different_world",)
    touching = [q for q in zero.samples if q.status == "valid" and q.field_gap_m <= 0]
    assert touching and all(q.weights == ((5, 0.5), (11, 0.5)) for q in touching)
    foreign, partial = ph.sample_sdf_contacts(model, second, points).results
    assert foreign.reasons == ("different_world",)
    assert partial.field_status == "unresolved" and "outside_grid" in partial.reasons
    assert partial.field_gap_bounds_m[1] > 0
    empty_batch = ph.sample_sdf_contacts(model, ph.SdfFieldPose(empty.key, g), points)
    assert empty_batch.samples_evaluated == 0 and all(
        q.field_status == "filtered" for q in empty_batch.results
    )


def test_reposing_a_bound_grid_keeps_a_confirmed_penetrating_material():
    from test_sdf_grid import asset

    mesh, v = asset("box")
    q = shape("rounded_particle", (391,), radius=0.125)
    model = ph.CollisionModel((mesh, q))
    src = ph.SdfFieldPose.bake(
        mesh, v, shape_xyz=(9, 9, 9), origin_m=(-1, -1, -1), spacing_m=(0.25,) * 3
    )
    shift = np.array([2.0, -1.0, 0.25])
    new = src.at_pose(translation_m=shift, orientation_xyzw=(0, 0, 1, 0))
    assert new.grid is src.grid
    posed = v * np.array([-1, -1, 1]) + shift
    row = one(ph.sample_sdf_contacts(model, new, {q.key: [shift]}, rigid_vertices_world_m=posed))
    assert row.field_status == row.source_geometry_status == "contact"
    assert row.source_gap_bounds_m[1] < 0 and row.samples[0].field_gap_m == -0.375


def _octahedral_minimum(vertices):
    # Independent LP oracle: each orthant is affine. Its extrema on the
    # material polygon occur where two triangle/orthant boundary lines meet.
    from fractions import Fraction as F
    from itertools import combinations

    a, b, c = [tuple(F(float(x)) for x in p) for p in vertices]
    lines = [(F(0), F(1), F(0)), (F(0), F(0), F(1)), (F(1), F(-1), F(-1))]
    axes = [(a[k], b[k] - a[k], c[k] - a[k]) for k in range(3)]
    candidates = []
    for (p, px, py), (q, qx, qy) in combinations([*lines, *axes], 2):
        det = px * qy - py * qx
        if det:
            u, v = (py * q - p * qy) / det, (p * qx - px * q) / det
            if u >= 0 and v >= 0 and u + v <= 1:
                xyz = [abs(o + u * du + v * dv) for o, du, dv in axes]
                candidates.append(xyz[0] + 2 * xyz[1] + 3 * xyz[2] - F(3, 8))
    return min(candidates)


@pytest.mark.parametrize(
    "vertices",
    [
        ((-0.875, -0.75, 0.125), (0.875, -0.375, 0.125), (-0.25, 0.875, 0.125)),
        ((-0.875, -0.75, 0.25), (0.875, -0.375, 0.25), (-0.25, 0.875, 0.25)),
        ((0.75, 0.25, 0.625), (-0.875, 0.5, -0.375), (0.125, -0.75, 0.25)),
        ((-0.75, 0.25, 0.625), (0.875, 0.5, -0.375), (-0.125, -0.75, 0.25)),
    ],
)
def test_whole_triangle_bounds_equal_independent_piecewise_affine_minimum(vertices, tmp_path):
    import json
    from fractions import Fraction as F

    g = grid(
        lambda x, y, z: abs(x) + 2 * abs(y) + 3 * abs(z) - 0.375,
        origin=(-1.0, -1.0, -1.0),
        step=(0.25, 0.5, 0.25),
        dims=(9, 5, 9),
    )
    s = shape("triangle", (23, 5, 11))
    m, f = setup(g, materials=(s,))
    row = one(ph.sample_sdf_contacts(m, f, {s.key: vertices}))
    target = _octahedral_minimum(vertices)
    low, high = row.field_gap_bounds_m
    assert F(low) <= target <= F(high) and high - low < 1e-15
    assert row.field_status == ("contact" if target <= 0 else "clear")
    check_material(row, s, vertices)
    (tmp_path / "sdf-material-whole-face.json").write_text(
        json.dumps(
            dict(
                oracle=str(target),
                status=row.field_status,
                bounds_m=(low, high),
                aabb_cells=row.cells_examined,
                nonempty_cells=row.cells_covered,
                samples=len(row.samples),
            )
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("kind", ["box", "u", "thin"])
def test_material_ids_reuse_the_native_texture_free_positions(kind, tmp_path):
    import json

    from test_sdf_grid import REF, asset

    case = json.loads((REF / "rr" / "sdf_grid_points.json").read_text("utf-8"))[kind]
    mesh, vertices = asset(kind)
    frame = ph.SdfFieldPose.bake(
        mesh,
        vertices,
        shape_xyz=case["shape_xyz"],
        origin_m=case["origin_m"],
        spacing_m=case["spacing_m"],
    )
    particles = [shape("particle", (103 + i * 13,), index=i) for i in range(len(case["points_m"]))]
    model = ph.CollisionModel((mesh, *particles))
    xyz = {s.key: [p] for s, p in zip(particles, case["points_m"], strict=True)}
    rows = ph.sample_sdf_contacts(model, frame, xyz, rigid_vertices_world_m=vertices).results
    refs = frame.grid.sample_local(case["points_m"])
    for s, row, ref in zip(particles, rows, refs, strict=True):
        point = row.samples[0]
        assert row.material_key == s.key and point.weights == ((s.vertex_ids[0], 1.0),)
        assert point.field_distance_m == ref.distance_m and point.status == ref.status
        assert point.gradient_world == ref.gradient
    assert rows[-1].field_status == rows[-1].source_geometry_status == "unresolved"
    if kind == "thin":
        assert rows[0].field_status == "clear" and rows[0].source_geometry_status == "unresolved"
    (tmp_path / "sdf-material-point-catalog.json").write_text(
        json.dumps(
            dict(
                kind=kind,
                points_m=case["points_m"],
                rows=[
                    dict(
                        material=list(x.samples[0].weights),
                        field=x.field_status,
                        source=x.source_geometry_status,
                        scalar_m=x.samples[0].field_distance_m,
                        bounds_m=x.source_gap_bounds_m,
                    )
                    for x in rows
                ],
            ),
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def test_host_numeric_failure_marks_only_the_affected_material():
    from numpy import broadcast_to

    g = ph.SignedDistanceGrid(
        broadcast_to(np.array([0.0, 1.0, 1e308, -1e308]), (2, 2, 4)),
        origin_m=(0, 0, 0),
        spacing_m=(1, 1, 1),
    )
    a, b = shape("particle", (7,), index=0), shape("particle", (7,), index=1)
    m, f = setup(g, materials=(a, b))
    good, bad = ph.sample_sdf_contacts(
        m, f, {a.key: [[0.5, 0.5, 0.5]], b.key: [[2.5, 0.5, 0.5]]}
    ).results
    assert good.field_status == "clear" and good.samples[0].field_gap_m == 0.5
    assert bad.field_status == "unresolved" and bad.reasons == ("numeric_precision",)
    assert bad.samples[0].field_gap_m is None and bad.field_gap_bounds_m == (None, None)
