"""Whole-body prism geometry: exact supports, clipping, and path certificates."""

from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest

import phymeo as ph
from phymeo.core.scene import fingerprint

TRI = np.array([[-1.0, 0.125, -1.0], [0.0, 0.125, 2.0], [1.0, 0.125, -1.0]])
SQUARE = np.array([[-0.5, 0.0, -0.5], [0.5, 0.0, -0.5], [0.5, 0.0, 0.5], [-0.5, 0.0, 0.5]])


def model():
    return ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.zeros((1, 3)),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[0.0, 1.0, 0.0]]),
        np.array([2.0]),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )


def patch(vertices=SQUARE):
    assert hasattr(ph, "RobotContactPatch"), "missing zero-thickness finite patch API"
    return ph.RobotContactPatch("patch", 0, vertices)


def manifold(x=TRI, vertices=SQUARE, budget=0.0):
    assert hasattr(ph, "triangle_prism_patch_manifold"), "missing whole convex prism API"
    return ph.triangle_prism_patch_manifold(
        model(), np.zeros(1), patch(vertices), x, half_thickness_m=0.125, geometric_budget_m=budget
    )


def test_owned_surface_and_patch_schema():
    assert hasattr(ph, "TrianglePrismSurface"), "missing owned prism schema"
    s = ph.TrianglePrismSurface("cloth", "nodes", [0, 1, 2], half_thickness_m=0.125)
    assert s.node_indices == (0, 1, 2)
    for kw in (
        {"key": ""},
        {"domain": False},
        {"node_indices": [0, 0, 2]},
        {"node_indices": [False, 1, 2]},
        {"half_thickness_m": 1},
        {"half_thickness_m": 0.0},
        {"half_thickness_m": float("nan")},
    ):
        args = dict(key="cloth", domain="nodes", node_indices=[0, 1, 2], half_thickness_m=0.125)
        args.update(kw)
        with pytest.raises((ValueError, TypeError)):
            ph.TrianglePrismSurface(**args)
    vertices = SQUARE.copy()
    p = patch(vertices)
    sig = fingerprint(p)
    vertices[:] = 100
    assert fingerprint(p) == sig and not p.local_vertices.flags.writeable
    assert not hasattr(p, "radius_m")
    for v in (SQUARE[::-1], np.roll(SQUARE, 1, axis=0)):
        assert fingerprint(patch(v)) == sig
    with pytest.raises(TypeError):
        ph.RobotContactPatch("p", 0, SQUARE, radius_m=0.125)
    for v in (SQUARE.astype(np.float32), SQUARE[[0, 2, 1, 3]], np.vstack((SQUARE, SQUARE[0]))):
        with pytest.raises((ValueError, TypeError)):
            patch(v)
    with pytest.raises(ValueError, match="coordinate"):
        ph.triangle_prism_patch_manifold(
            replace(model(), joint_axis=np.array([[0.6, 0.8, 0.0]])),
            np.zeros(1),
            patch(),
            TRI,
            half_thickness_m=0.125,
            geometric_budget_m=0.0,
        )


def test_patch_inside_triangle_has_complete_four_corner_manifold():
    r = manifold()
    assert r.status == "contact" and r.gap_bounds == (0.0, 0.0)
    assert len(r.points) == 4 and len(r.axes) > 0
    np.testing.assert_array_equal(r.normal, [0.0, 1.0, 0.0])
    assert {tuple(p.robot_point) for p in r.points} == {tuple(p) for p in SQUARE}
    for p in r.points:
        assert sum(p.material[:3]) == 1 and min(p.material[:3]) >= 0
        assert p.material[3] == F(-1, 8)
        np.testing.assert_allclose(
            np.array(p.material[:3], float) @ TRI + [0.0, -0.125, 0.0], p.prism_point, atol=1e-15
        )
    # Entire triangle is outside the finite square, so vertex-only discovery misses it.
    assert all(abs(x) > 0.5 or abs(z) > 0.5 for x, y, z in TRI)


def test_side_edge_vertex_and_true_edge_edge():
    # An independent full-vertex SAT oracle compares all face/physical-edge axes.
    cases = [
        TRI + np.array([1.5, 0.0, 0.0]),
        TRI + np.array([2.0, 0.0, 0.0]),
        TRI + np.array([0.0, 0.02, 0.0]),
        TRI * np.array([1.0, 1.0, 0.2]),
    ]
    for x in cases:
        r = manifold(x)
        n = np.cross(x[1] - x[0], x[2] - x[0])
        n = n / np.linalg.norm(n)
        av = np.vstack((x - 0.125 * n, x + 0.125 * n))
        bv = SQUARE
        ae = [x[(i + 1) % 3] - x[i] for i in range(3)] + [n]
        be = [bv[(i + 1) % 4] - bv[i] for i in range(4)]
        axes = [n, np.cross(be[0], be[1])] + [np.cross(e, n) for e in ae[:3]]
        axes += [np.cross(a, b) for a in ae for b in be]
        gaps = [
            min(av @ d) - max(bv @ d)
            for a in axes
            if np.linalg.norm(a) > 0
            for d in (a / np.linalg.norm(a), -a / np.linalg.norm(a))
        ]
        expected = max(gaps)
        assert r.gap_bounds[0] - 1e-15 <= expected <= r.gap_bounds[1] + 1e-15
        if r.status == "contact":
            for p in r.points:
                assert np.min(av @ r.normal) >= p.prism_point @ r.normal - 1e-14
                assert np.max(bv @ r.normal) <= p.robot_point @ r.normal + 1e-14
    # Two nonparallel thin supports meet on actual physical edges, no cap interior overlap.
    v = np.array([[-2.0, -0.25, -0.3], [2.0, 0.25, -0.3], [2.0, 0.25, 0.3], [-2.0, -0.25, 0.3]])
    r = manifold(TRI + np.array([0.0, 0.125, 0.0]), v, budget=0.5)
    assert r.status == "contact" and r.points


def test_quadratic_field_sign_and_negative_gap_order():
    from phymeo.geometry._prism_exact import Quadratic, compare_gaps

    root = Quadratic(0, 1, F(2))
    assert root > F(7, 5) and root < F(3, 2)
    assert (root * root - 2).sign() == 0 and (root - root).sign() == 0
    assert (1 / root - root / 2).sign() == 0
    assert Quadratic(F(1, 3), 0, F(2)).a == F(1, 3)
    assert (
        compare_gaps(
            Quadratic(-1, 0, 2), Quadratic(4, 0, 2), Quadratic(-1, 0, 2), Quadratic(1, 0, 2)
        )
        > 0
    )
    lo, hi = root.bounds()
    assert lo * lo <= 2 <= hi * hi
    x = np.array(TRI, dtype=object)
    x[:, 1] = F(1, 8) + F(1, 2**60)
    r = manifold(x)
    assert r.exact_gap_bounds == (F(1, 2**60),) * 2


def sweep(x0, x1, **kw):
    assert hasattr(ph, "triangle_prism_patch_sweep"), "missing whole-prism CCD API"
    return ph.triangle_prism_patch_sweep(
        model(),
        np.zeros(1),
        np.zeros(1),
        patch(),
        x0,
        x1,
        half_thickness_m=0.125,
        penetration_budget_m=kw.pop("budget", 0.0),
        **kw,
    )


def test_translation_contact_certifies_whole_prism():
    r = sweep(TRI, TRI + np.array([0.25, 0.0, 0.0]))
    assert r.status == "certified" and r.checked_intervals == 1
    assert r.segments[0].start == 0 and r.segments[-1].end == 1
    assert r.segments[0].gap_lower_m == 0.0


def test_changing_normal_safe_path_and_interior_crossing():
    a = TRI + np.array([0.0, 2.0, 0.0])
    b = a.copy()
    b[1, 1] += 1.0
    r = sweep(a, b)
    assert r.status == "certified"
    # Both endpoints separated; a complete prism sweeps across the zero-thickness face.
    r = sweep(TRI + np.array([0.0, 0.5, 0.0]), TRI - np.array([0.0, 0.75, 0.0]))
    assert r.status == "violation" and isinstance(r.witness_time, F)
    t = r.witness_time
    x = (1 - float(t)) * (TRI + [0.0, 0.5, 0.0]) + float(t) * (TRI - [0.0, 0.75, 0.0])
    assert abs(x[0, 1]) < 0.125


def test_area_collapse_and_budget_exhaustion_are_not_safe():
    b = TRI[[2, 1, 0]]
    r = sweep(TRI, b)
    assert r.status == "unresolved" and "area" in r.reason
    a = TRI + np.array([0.0, 2.0, 0.0])
    b = a.copy()
    b[1, 1] += 8.0
    r = sweep(a, b, max_intervals=1, max_depth=1)
    assert r.status in ("certified", "unresolved")
    # This safe arch has no single fixed separating axis at this coarse curvature bound.
    a = TRI.copy()
    b = TRI.copy()
    b[1, 1] += 0.2
    r = sweep(a, b, max_intervals=1, max_depth=1)
    assert r.status == "unresolved"


def test_exact_time_and_candidate_fingerprint():
    from phymeo.geometry.prism_patch_sweep import area_polynomial

    a = np.array([[F(v) for v in row] for row in TRI], dtype=object)
    a[:, 1] = F(1, 8) + F(1, 2**60)
    b = a.copy()
    b[:, 0] += F(1, 3)
    r = sweep(a, b)
    assert r.status == "certified"
    assert r.segments[0].exact_gap_lower == F(1, 2**60)
    assert (
        r.input_fingerprint != sweep(a, b + np.array([F(1, 2**60), 0, 0], object)).input_fingerprint
    )
    c = area_polynomial(a, b)
    assert all(v == 0 for row in c[1:] for v in row)
    t = F(1, 3)
    x = (1 - t) * a + t * b
    np.testing.assert_array_equal(np.cross(x[1] - x[0], x[2] - x[0]), np.array(c[0], object))
