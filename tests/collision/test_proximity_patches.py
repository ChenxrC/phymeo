"""Opt-in separated support measures, independent of the single distance witness."""

from itertools import product

import numpy as np
import pytest

import phymeo as ph


def sample(kind, a, b):
    shapes = [
        ph.CollisionShape(ph.CollisionKey(0, i, 0), kind, tuple(range(20 * i, 20 * i + len(p))))
        for i, p in enumerate((a, b))
    ]
    model = ph.CollisionModel(shapes)
    poses = {s.key: ph.CollisionMotion(p) for s, p in zip(shapes, (a, b), strict=True)}
    with ph.CollisionWorkspace(model) as ws:
        default = ph.detect(model, poses, ph.CollisionPolicy(margin=0.1), ws).pairs[0]
        full = ph.detect(
            model, poses, ph.CollisionPolicy(margin=0.1, proximity_patches=True), ws
        ).pairs[0]
    assert default.status == full.status == "separated"
    assert default.witness.gap_m == full.witness.gap_m
    assert not hasattr(default.witness, "proximity_patch")
    if default.convex is not None:
        assert not default.convex.end.patches
    patch = full.witness.proximity_patch
    for p in patch.points:
        for weights, coords, shape, point in zip(
            (p.first_weights, p.second_weights),
            (a, b),
            shapes,
            (p.first_point, p.second_point),
            strict=True,
        ):
            assert sum(w for _, w in weights) == pytest.approx(1.0)
            mapping = dict(zip(shape.vertex_ids, coords, strict=True))
            np.testing.assert_allclose(sum(w * mapping[k] for k, w in weights), point, atol=1e-14)
    return patch


def test_disjoint_box_diagonal_keeps_complete_nearest_edge():
    a = np.array(list(product((-0.125, 0.125), repeat=3)))
    patch = sample("hull", a, a + (0.28125, 0.28125, 0.0))
    assert patch.dimension == 1 and patch.area_m2 == 0.0
    assert sorted(p.first_point for p in patch.points) == [
        (0.125, 0.125, -0.125),
        (0.125, 0.125, 0.125),
    ]
    assert sorted(p.second_point for p in patch.points) == [
        (0.15625, 0.15625, -0.125),
        (0.15625, 0.15625, 0.125),
    ]
    np.testing.assert_allclose(patch.normal, [2**-0.5, 2**-0.5, 0.0])


def test_opposed_triangle_overhang_clips_at_new_material_points():
    a = np.array([(0.0, 0, 0), (1.0, 0, 0), (0.0, 1, 0)])
    b = np.array([(-0.25, 0.25, 1 / 32), (0.75, 0.25, 1 / 32), (0.25, 0.75, 1 / 32)])
    patch = sample("triangle", a, b)
    assert patch.dimension == 2 and patch.area_m2 == pytest.approx(7 / 32)
    assert sorted(p.first_point for p in patch.points) == [
        (0.0, 0.25, 0.0),
        (0.0, 0.5, 0.0),
        (0.25, 0.75, 0.0),
        (0.75, 0.25, 0.0),
    ]
