"""Analytic curved solids are queried through the mixed public collision pipeline."""

import numpy as np
import pytest

import phymeo as ph


CYCLIC = (0.5, 0.5, 0.5, 0.5)  # Exact double cover of (x, y, z) -> (z, x, y).
IDENTITY = (0, 0, 0, 1)


def solid(kind, dimensions, orientation=IDENTITY):
    assert hasattr(ph, "CurvedSolid"), "public analytic curved-solid geometry is missing"
    return ph.CurvedSolid(kind, dimensions, orientation)


def scene(asset, curved_center, other, other_center, *, reverse=False, radius=0, normal=None):
    a, b = ph.CollisionKey(4, 1, 5), ph.CollisionKey(4, 2, 8)
    c, o = (b, a) if reverse else (a, b)
    curved = ph.CollisionShape(c, asset.kind, (47,), curved=asset)
    second = ph.CollisionShape(o, other, (91,), radius, normal=normal)
    model = ph.CollisionModel((second, curved))
    return model, {c: ph.CollisionMotion([curved_center]), o: ph.CollisionMotion([other_center])}, c, o


def query(model, paths, margin=2):
    with ph.CollisionWorkspace(model) as work:
        out = ph.detect(model, paths, ph.CollisionPolicy(margin=margin), work)
        assert out.status == "ok" and work.last_result is out
        return out


def oriented(row, c):
    w = row.witness
    assert row.family == "RR" and row.candidate and row.curved.owner == c and row.sweep is None
    if row.first == c:
        pc, po, wc, wo, nc = w.first_point, w.second_point, w.first_weights, w.second_weights, w.normal
    else:
        pc, po, wc, wo = w.second_point, w.first_point, w.second_weights, w.first_weights
        nc = None if w.normal is None else -np.asarray(w.normal)
    assert wc == ((47, 1.0),) and wo == ((91, 1.0),)
    return np.asarray(pc), np.asarray(po), nc


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("center,near,normal,gap,features", [
    ((0.625, 0, -0.125), (0.5, 0, -0.125), (1, 0, 0), -0.0625, ("side",)),
    ((0.125, 0.125, -0.5), (0.125, 0.125, -0.25), (0, 0, -1), 0.0625, ("bottom_cap",)),
    ((0.59375, 0, 0.375), (0.5, 0, 0.25), (0.6, 0, 0.8), -0.03125, ("side", "top_cap")),
    ((0.375, 0, 0), (0.5, 0, 0), (1, 0, 0), -0.3125, ("side",)),
])
def test_cylinder_sphere_side_cap_rim_and_inside(center, near, normal, gap, features, reverse):
    origin = np.array([2, -3, 4])
    ball = origin + np.roll(center, 1)
    points = scene(solid("cylinder", (0.5, 0.25), CYCLIC), origin, "sphere", ball,
                   reverse=reverse, radius=0.1875)
    model, paths, c, _ = points
    (pair,) = query(model, paths).pairs
    assert pair.status == ("overlap" if gap < 0 else "separated")
    assert pair.witness.gap_m == pytest.approx(gap, abs=1e-12)
    pc, po, nc = oriented(pair, c)
    expected_normal = np.roll(normal, 1)
    np.testing.assert_allclose(nc, expected_normal, rtol=0, atol=1e-12)
    np.testing.assert_allclose(pc, origin + np.roll(near, 1), rtol=0, atol=1e-12)
    np.testing.assert_allclose(po, ball - 0.1875 * expected_normal, rtol=0, atol=1e-12)
    assert pair.curved.features == features and pair.curved.locus == "point"


@pytest.mark.parametrize("dims,point,features,locus", [
    ((0.5, 0.5), (0.25, 0, 0.25), ("side", "top_cap"), "multiple"),
    ((0.5, 0.75), (0, 0, 0.125), ("side",), "circle"),
    ((0.5, 0.25), (0.125, 0, 0), ("bottom_cap", "top_cap"), "multiple"),
    ((0.5, 0.25), (0.5, 0, -0.25), ("side", "bottom_cap"), "normal_cone"),
])
def test_cylinder_nonunique_normals_keep_actual_boundary_representatives(dims, point, features, locus):
    model, paths, c, _ = scene(solid("cylinder", dims), [0, 0, 0], "sphere", point, radius=0.0625)
    (pair,) = query(model, paths).pairs
    assert pair.status == "overlap" and pair.witness.normal is None
    assert pair.curved.owner == c and pair.curved.features == features and pair.curved.locus == locus
    assert pair.curved.points_m
    for x, y, z in pair.curved.points_m:
        radial = np.hypot(x, y)
        assert radial <= dims[0] and abs(z) <= dims[1]
        assert radial == pytest.approx(dims[0]) or abs(z) == pytest.approx(dims[1])
        assert -np.linalg.norm(np.array([x, y, z]) - point) - 0.0625 == pytest.approx(pair.witness.gap_m)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("n,features,locus,extremes", [
    ((-3, 0, -4), ("apex",), "point", [(0, 0, 0.25)]),
    ((0, 3, 4), ("base_rim",), "point", [(0, -0.5, -0.25)]),
    ((1, 0, -1), ("apex", "base_rim"), "segment", [(0, 0, 0.25), (-0.5, 0, -0.25)]),
    ((0, 0, 2), ("base_disk",), "disk", [(0, 0, -0.25), (0.5, 0, -0.25), (-0.5, 0, -0.25)]),
])
def test_cone_supports_arbitrary_halfspaces(n, features, locus, extremes, reverse):
    center, plane = np.array([1.0, -0.5, -0.125]), np.array([-0.25, 0.75, 0.125])
    world_n = np.roll(n, 1)
    unit = world_n / np.linalg.norm(world_n)
    model, paths, c, _ = scene(solid("cone", (0.5, 0.25), CYCLIC), center, "plane", plane,
                                reverse=reverse, normal=world_n)
    (pair,) = query(model, paths).pairs
    assert pair.curved.features == features and pair.curved.locus == locus
    pc, pp, normal = oriented(pair, c)
    expected = [center + np.roll(v, 1) for v in extremes]
    gap = unit @ (expected[0] - plane)
    for p in expected:
        assert any(np.allclose(p, q, atol=1e-12, rtol=0) for q in pair.curved.points_m)
    np.testing.assert_allclose(normal, -unit, atol=1e-12, rtol=0)
    assert pair.witness.gap_m == pytest.approx(gap, abs=1e-12)
    assert unit @ (pp - plane) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(np.subtract(pc, pp), gap * unit, atol=1e-12, rtol=0)


@pytest.mark.parametrize("reverse", [False, True])
def test_ellipsoid_surface_and_tangent_plane_own_distinct_points(reverse):
    axes = np.array([0.25, 0.5, 1.0])
    center, anchor = np.array([0.75, 0.25, -0.5]), np.array([1.125, -0.5, 0.25])
    local_n = np.array([3, -4, 7])
    n = np.roll(local_n, 1)
    model, paths, c, _ = scene(solid("ellipsoid", axes, CYCLIC), center, "plane", anchor,
                                reverse=reverse, normal=n)
    (pair,) = query(model, paths).pairs
    pc, pp, normal = oriented(pair, c)
    local = np.roll(pc - center, -1)
    assert sum((local / axes) ** 2) == pytest.approx(1.0)
    gradient = local / axes**2
    np.testing.assert_allclose(gradient / np.linalg.norm(gradient), -local_n / np.linalg.norm(n), atol=1e-12)
    gap = (n @ (center - anchor) - np.linalg.norm(axes * local_n)) / np.linalg.norm(n)
    assert pair.witness.gap_m == pytest.approx(gap, abs=1e-12)
    np.testing.assert_allclose(normal, -n / np.linalg.norm(n), atol=1e-12)
    np.testing.assert_allclose(pp - pc, gap * np.array(normal), atol=1e-12)
    assert (pp - anchor) @ n == pytest.approx(0.0, abs=1e-12)


def test_bounds_include_rotated_shape_and_motion_has_no_manufactured_toi():
    ell = solid("ellipsoid", (0.25, 0.5, 1.0), CYCLIC)
    model, paths, c, o = scene(ell, [2, 0, 0], "sphere", [0, 0, 0], radius=0.0625)
    out = query(model, paths, margin=0)
    bounds = next(b for b in out.bounds if b.key == c)
    assert bounds.lower == (1.0, -0.25, -0.5) and bounds.upper == (3.0, 0.25, 0.5)
    assert not out.pairs[0].candidate and out.pairs[0].status == "separated"
    model, paths, c, o = scene(solid("cylinder", (0.5, 0.25)), [-2, 0, 0], "sphere",
                                [0, 0, 0], radius=0.0625)
    for movement in (
        ph.CollisionMotion([[-2, 0, 0]], [[2, 0, 0]]),
        ph.CollisionMotion([[-2, 0, 0]], turn=ph.RigidTurn((0, 0, 0), (0, 0, 0), (0, 0, 1), np.pi)),
    ):
        (pair,) = query(model, {c: movement, o: paths[o]}, margin=0).pairs
        assert pair.candidate and pair.status == "unsupported" and pair.sweep is None and pair.witness is None


def test_curved_geometry_validation_and_unhandled_pair():
    for kind, dims, quat in (("cone", (1, 0), IDENTITY), ("sphere", (1, 2), IDENTITY),
                            ("ellipsoid", (1, 2), IDENTITY), ("cylinder", (1, 2), (0, 0, 0, 2)),
                            ("ellipsoid", (1, np.nan, 1), IDENTITY)):
        with pytest.raises(ValueError):
            solid(kind, dims, quat)
    c = ph.CollisionKey(0, 0, 0)
    asset = solid("cone", (1, 2))
    for kind, ids, r in (("cylinder", (0,), 0), ("cone", (0, 1), 0),
                         ("cone", (0,), 0.25), ("sphere", (0,), 0.5)):
        with pytest.raises(ValueError):
            ph.CollisionShape(c, kind, ids, r, curved=asset)
    model, paths, _, _ = scene(asset, [0, 0, 0], "sphere", [0, 0, 0.5], radius=0.25)
    (pair,) = query(model, paths).pairs
    assert pair.candidate and pair.status == "unsupported" and pair.witness is None
