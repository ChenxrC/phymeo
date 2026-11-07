"""Independent finite convex patch geometry and material reconstruction."""
from dataclasses import replace

import numpy as np
import pytest
from convex_family_cases import PAIRS, convex_case

from phymeo.geometry.contact_patch_adapter import contact_patch_sample


def sample(kinds, gap, *, reverse=False, matrix=None):
    shapes, paths = convex_case(*kinds, gap)
    a, b = shapes
    x, y = paths[a.key].start, paths[b.key].start
    if matrix is not None:
        shift = np.array((2.0, -3.0, 0.25))
        x, y = x @ matrix.T + shift, y @ matrix.T + shift
    if reverse:
        a, b, x, y = b, a, y, x
    return contact_patch_sample(a, b, x, y), (a, b), (x, y)


def reconstruct(shape, points, weights, offset):
    rows = dict(zip(shape.vertex_ids, points, strict=True))
    assert all(i in rows and w >= 0 for i, w in weights)
    assert sum(w for _, w in weights) == pytest.approx(1, abs=1e-12)
    value = sum(w * rows[i] for i, w in weights)
    return value if offset is None else value + offset


@pytest.mark.parametrize("kinds", PAIRS)
@pytest.mark.parametrize("gap", (0.02, 0.0, -0.01))
def test_ten_families_have_true_finite_material_patches(kinds, gap):
    result, shapes, points = sample(kinds, gap)
    assert result.sites and sum(s.weight for s in result.sites) == pytest.approx(1)
    dimension = 0 if "sphere" in kinds else 1 if "capsule" in kinds else 2
    assert result.dimension == dimension
    for site in result.sites:
        w = site.witness
        np.testing.assert_allclose(w.normal, (0, 0, 1), atol=1e-10)
        assert w.gap_m == pytest.approx(gap, abs=1e-10)
        for shape, coords, weights, offset, point in (
            (shapes[0], points[0], w.first_weights, w.first_offset, w.first_point),
            (shapes[1], points[1], w.second_weights, w.second_offset, w.second_point),
        ):
            np.testing.assert_allclose(reconstruct(shape, coords, weights, offset), point,
                                       atol=1e-10, rtol=0)
        np.testing.assert_allclose(site.load_point,
                                   (np.asarray(w.first_point) + w.second_point) / 2,
                                   atol=1e-12, rtol=0)


@pytest.mark.parametrize("kinds", PAIRS)
def test_exchange_and_rigid_frame_preserve_patch_and_normal(kinds):
    rotation = np.array(((0, 0, 1), (1, 0, 0), (0, 1, 0)))
    forward, _, _ = sample(kinds, -0.01)
    reverse, _, _ = sample(kinds, -0.01, reverse=True, matrix=rotation)
    assert reverse.dimension == forward.dimension
    assert len(reverse.sites) == len(forward.sites)
    np.testing.assert_allclose(sum(s.weight * np.asarray(s.load_point) for s in reverse.sites),
                               rotation @ sum(s.weight * np.asarray(s.load_point) for s in forward.sites)
                               + (2, -3, 0.25), atol=1e-10, rtol=0)
    for site in reverse.sites:
        np.testing.assert_allclose(site.witness.normal, (-1, 0, 0), atol=1e-10)
        assert site.witness.gap_m == pytest.approx(-0.01, abs=1e-10)


def test_capsule_uses_finite_shaft_and_never_an_infinite_line():
    shapes, paths = convex_case("capsule", "box", -0.01)
    a, b = shapes
    x = paths[a.key].start + (1.2, 0, 0)
    patch = contact_patch_sample(a, b, x, paths[b.key].start)
    assert patch.dimension == 0
    assert all(s.witness.gap_m > 0 for s in patch.sites)
    assert all(s.witness.first_weights == ((100, 1.0), (101, 0.0)) for s in patch.sites)


def test_coplanar_vertex_refinement_preserves_resultant_centroid():
    shapes, paths = convex_case("box", "box", -0.01)
    a, b = shapes
    x, y = paths[a.key].start, paths[b.key].start
    base = contact_patch_sample(a, b, x, y)
    more = replace(a, vertex_ids=(*a.vertex_ids, 999))
    refined = contact_patch_sample(more, b, np.vstack((x, (0, 0, 0.1))), y)
    for patch in (base, refined):
        assert sum(s.weight for s in patch.sites) == pytest.approx(1)
        np.testing.assert_allclose(sum(s.weight * np.asarray(s.load_point) for s in patch.sites),
                                   (0, 0, 0.095), atol=1e-12)


def test_ambiguous_deep_overlap_and_malformed_frames_are_explicit():
    shapes, paths = convex_case("sphere", "sphere", 0)
    a, b = shapes
    with pytest.raises(ValueError, match="normal|coincident|ambiguous"):
        contact_patch_sample(a, b, paths[a.key].start, paths[a.key].start)
    with pytest.raises(ValueError):
        contact_patch_sample(a, b, [[float("nan"), 0, 0]], paths[b.key].start)
    with pytest.raises(ValueError, match="world"):
        contact_patch_sample(a, replace(b, key=replace(b.key, world=5)),
                             paths[a.key].start, paths[b.key].start)
