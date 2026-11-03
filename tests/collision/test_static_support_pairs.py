"""Independent common-normal controls for bounded convex rigid primitives."""

from itertools import product
from math import hypot

import numpy as np
import pytest

import phymeo as ph


def geometry(kind, parameters, centers, ids=None, orientation=(0, 0, 0, 1)):
    return (kind, parameters, np.asarray(centers, dtype=np.float64), ids, orientation)


def control(name, gap):
    if name == "two-cylinders":
        a = geometry("cylinder", (0.5, 0.375), [[0, 0, 0]])
        b = geometry("cylinder", (0.25, 0.25), [[0.75 + gap, 0, 0]])
        return a, b, np.array([1, 0, 0]), 0.5
    if name == "ellipsoid-sphere":
        a = geometry("ellipsoid", (0.5, 0.375, 0.25), [[0, 0, 0]])
        b = geometry("sphere", 0.125, [[0, 0, 0.375 + gap]])
        return a, b, np.array([0, 0, 1]), 0.25
    if name == "cone-capsule":
        a = geometry("cone", (0.5, 0.375), [[0, 0, 0]])
        b = geometry("capsule", 0.125, [[0, 0, 0.5 + gap], [0, 0, 0.75 + gap]], (13, 9))
        return a, b, np.array([0, 0, 1]), 0.375
    if name == "cone-cylinder":
        a = geometry("cone", (0.5, 0.375), [[0, 0, 0]])
        b = geometry("cylinder", (0.75, 0.25), [[0, 0, 0.625 + gap]])
        return a, b, np.array([0, 0, 1]), 0.375
    # For the rotated ellipsoid the bottom is offset sideways from its center.
    # Place that point over the interior of the box's top face.
    corners = list(product((-0.5, 0.5), (-0.5, 0.5), (-0.25, 0.25)))
    a = geometry("hull", (), corners, (23, 8, 11, 5, 2, 14, 63, 3))
    sin_y, cos_y = 24 / 25, 7 / 25
    reach = hypot(3 * sin_y / 8, cos_y / 8)
    side = ((3 / 8) ** 2 - (1 / 8) ** 2) * sin_y * cos_y / reach
    b = geometry("ellipsoid", (0.375, 0.25, 0.125), [[-side, 0.125, 0.25 + reach + gap]],
                 orientation=(0.0, 0.6, 0.0, 0.8))
    return a, b, np.array([0, 0, 1]), 0.25


def shape_and_motion(key, config, shift):
    kind, parameters, vertices, ids, rotation = config
    ids = tuple(range(len(vertices))) if ids is None else ids
    if kind in ("sphere", "capsule"):
        shape = ph.CollisionShape(key, kind, ids, parameters)
    elif kind == "hull":
        shape = ph.CollisionShape(key, kind, ids)
    else:
        shape = ph.CollisionShape(key, kind, ids, curved=ph.CurvedSolid(kind, parameters, rotation))
    return shape, ph.CollisionMotion(vertices + shift)


def query(name, gap, swapped=False):
    a, b, n, support = control(name, gap)
    shift = np.array([0.375, -0.25, 0.5])
    first, second = ph.CollisionKey(4, 1, 0), ph.CollisionKey(4, 2, 0)
    ka, kb = (second, first) if swapped else (first, second)
    sa, ma = shape_and_motion(ka, a, shift)
    sb, mb = shape_and_motion(kb, b, shift)
    model = ph.CollisionModel((sb, sa))
    paths = {ka: ma, kb: mb}
    with ph.CollisionWorkspace(model) as work:
        detection = ph.detect(model, paths, ph.CollisionPolicy(margin=0.25), work)
    assert detection.status == "ok" and detection.required_candidates == 1
    (row,) = detection.pairs
    assert row.family == "RR" and row.candidate
    return row, {sa.key: sa, sb.key: sb}, paths, n, support + n @ shift


@pytest.mark.parametrize("name", ["two-cylinders", "ellipsoid-sphere", "cone-capsule", "cone-cylinder", "oblique-hull"])
@pytest.mark.parametrize("gap", [1 / 16, 0.0, -1 / 32])
@pytest.mark.parametrize("swapped", [False, True])
def test_support_pairs_publish_distance_and_material_owners(name, gap, swapped):
    row, shapes, paths, n, h = query(name, gap, swapped)
    assert row.status == ("separated" if gap > 0 else "overlap" if gap < 0 else "touching"), row.reason
    w, trace = row.witness, row.support_map
    assert w.source == "host-convex-support" and trace.support_calls > 0
    assert trace.gap_lower_m - 1e-10 <= gap <= trace.gap_upper_m + 1e-10
    assert trace.gap_upper_m - trace.gap_lower_m <= 2e-5
    assert abs(w.gap_m - gap) <= 2e-5
    np.testing.assert_allclose(w.normal, -n if swapped else n, rtol=0, atol=3e-3)
    assert abs((np.asarray(w.second_point) - w.first_point) @ w.normal - w.gap_m) < 2e-5
    pa, pb = (w.second_point, w.first_point) if swapped else (w.first_point, w.second_point)
    assert abs(np.dot(pa, n) - h) < 3e-5
    assert abs(np.dot(pb, n) - (h + gap)) < 3e-5
    for key, weights, offset, point in (
        (row.first, w.first_weights, w.first_offset, w.first_point),
        (row.second, w.second_weights, w.second_offset, w.second_point),
    ):
        vertices = dict(zip(shapes[key].vertex_ids, paths[key].start, strict=True))
        assert all(i in vertices and x >= 0 for i, x in weights)
        assert sum(x for _, x in weights) == pytest.approx(1.0, abs=1e-12)
        anchor = sum(x * vertices[i] for i, x in weights)
        np.testing.assert_allclose(anchor + (0 if offset is None else offset), point, rtol=0, atol=1e-9)


def test_static_support_does_not_claim_a_swept_toi():
    a, b, _, _ = control("two-cylinders", 1 / 16)
    ka, kb = ph.CollisionKey(4, 1, 0), ph.CollisionKey(4, 2, 0)
    sa, ma = shape_and_motion(ka, a, np.zeros(3))
    sb, mb = shape_and_motion(kb, b, np.zeros(3))
    model = ph.CollisionModel((sa, sb))
    with ph.CollisionWorkspace(model) as work:
        result = ph.detect(model, {ka: ma, kb: ph.CollisionMotion(mb.start, mb.start - [0.5, 0, 0])},
                           ph.CollisionPolicy(), work)
    assert result.pairs[0].status == "unsupported" and result.pairs[0].sweep is None
