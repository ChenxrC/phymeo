"""Independent common-normal controls for bounded convex rigid primitives."""

import json
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
    b = geometry(
        "ellipsoid",
        (0.375, 0.25, 0.125),
        [[-side, 0.125, 0.25 + reach + gap]],
        orientation=(0.0, 0.6, 0.0, 0.8),
    )
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


def check_material_surface(shape, nodes, point):
    if shape.kind == "hull":
        # This control uses an axis-aligned box; its corners are registered in arbitrary order.
        low, high = np.min(nodes, axis=0), np.max(nodes, axis=0)
        assert np.all(point >= low - 2e-5) and np.all(point <= high + 2e-5)
        assert np.min(np.minimum(abs(point - low), abs(point - high))) < 2e-5
        return
    if shape.kind == "capsule":
        axis = nodes[1] - nodes[0]
        t = np.clip((point - nodes[0]) @ axis / (axis @ axis), 0, 1)
        assert abs(np.linalg.norm(point - (nodes[0] + t * axis)) - shape.radius) < 2e-5
        return
    offset = point - nodes[0]
    if shape.kind == "sphere":
        assert abs(np.linalg.norm(offset) - shape.radius) < 2e-5
        return
    # Rotate backward independently with a unit quaternion (XYZW).
    x, y, z, s = shape.curved.orientation_xyzw
    vector = -np.array((x, y, z))
    local = offset + 2 * np.cross(vector, s * offset + np.cross(vector, offset))
    dims = shape.curved.dimensions_m
    if shape.kind == "ellipsoid":
        assert abs(np.linalg.norm(local / dims) - 1) * max(dims) < 2e-5
        return
    radius, half_height = dims
    radial = hypot(*local[:2])
    side = (
        radius
        if shape.kind == "cylinder"
        else radius * (half_height - local[2]) / (2 * half_height)
    )
    assert abs(local[2]) <= half_height + 2e-5 and radial <= side + 2e-5
    cap = (
        min(abs(local[2] + half_height), abs(local[2] - half_height))
        if shape.kind == "cylinder"
        else abs(local[2] + half_height)
    )
    assert min(abs(radial - side), cap) < 2e-5


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


@pytest.mark.parametrize(
    "name", ["two-cylinders", "ellipsoid-sphere", "cone-capsule", "cone-cylinder", "oblique-hull"]
)
@pytest.mark.parametrize("gap", [1 / 16, 0.0, -1 / 32])
@pytest.mark.parametrize("swapped", [False, True])
def test_support_pairs_publish_distance_and_material_owners(name, gap, swapped, tmp_path):
    row, shapes, paths, n, h = query(name, gap, swapped)
    assert row.status == ("separated" if gap > 0 else "overlap" if gap < 0 else "touching"), (
        row.reason
    )
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
        check_material_surface(shapes[key], paths[key].start, np.asarray(point))
        vertices = dict(zip(shapes[key].vertex_ids, paths[key].start, strict=True))
        assert all(i in vertices and x >= 0 for i, x in weights)
        assert sum(x for _, x in weights) == pytest.approx(1.0, abs=1e-12)
        anchor = sum(x * vertices[i] for i, x in weights)
        np.testing.assert_allclose(
            anchor + (0 if offset is None else offset), point, rtol=0, atol=1e-9
        )

    readback = {
        "pair": name,
        "input_gap_m": gap,
        "swapped": swapped,
        "status": row.status,
        "gap_m": w.gap_m,
        "lower_m": trace.gap_lower_m,
        "upper_m": trace.gap_upper_m,
        "normal": w.normal,
        "first_point": w.first_point,
        "second_point": w.second_point,
        "phase": trace.phase,
        "support_calls": trace.support_calls,
        "iterations": trace.iterations,
    }
    (tmp_path / "convex-result.json").write_text(
        json.dumps(readback, allow_nan=False), encoding="utf-8"
    )


def test_static_support_does_not_claim_a_rotating_toi():
    a, b, _, _ = control("two-cylinders", 1 / 16)
    ka, kb = ph.CollisionKey(4, 1, 0), ph.CollisionKey(4, 2, 0)
    sa, ma = shape_and_motion(ka, a, np.zeros(3))
    sb, mb = shape_and_motion(kb, b, np.zeros(3))
    model = ph.CollisionModel((sa, sb))
    with ph.CollisionWorkspace(model) as work:
        result = ph.detect(
            model,
            {
                ka: ma,
                kb: ph.CollisionMotion(
                    mb.start, turn=ph.RigidTurn((0, 0, 0), (0, 0, 0), (0, 1, 0), 0.5)
                ),
            },
            ph.CollisionPolicy(),
            work,
        )
    assert result.pairs[0].status == "unsupported" and result.pairs[0].sweep is None


def test_rotating_both_static_bodies_transports_the_common_normal():
    cycled = (0.5, 0.5, 0.5, 0.5)
    ka, kb = ph.CollisionKey(2, 12, 5), ph.CollisionKey(2, 40, 8)
    a = geometry("cylinder", (0.5, 0.375), [[0, 0, 0]], orientation=cycled)
    b = geometry("cylinder", (0.25, 0.25), [[0, 23 / 32, 0]], orientation=cycled)
    shift = np.array([-4.125, 3.25, 1.0])
    sa, ma = shape_and_motion(ka, a, shift)
    sb, mb = shape_and_motion(kb, b, shift)
    model = ph.CollisionModel((sb, sa))
    with ph.CollisionWorkspace(model) as work:
        (row,) = ph.detect(model, {ka: ma, kb: mb}, ph.CollisionPolicy(), work).pairs
    assert row.status == "overlap" and row.witness.gap_m == pytest.approx(-1 / 32, abs=2e-5)
    np.testing.assert_allclose(row.witness.normal, [0, 1, 0], rtol=0, atol=3e-3)


@pytest.mark.tyx
def test_static_support_uses_real_partition_broad_phase(tyx_mode, tmp_path):
    models, paths, policies = {}, {}, {}
    for key, gap in ((18, 1 / 16), (4, -1 / 32)):
        a, b, _, _ = control("oblique-hull", gap)
        ka, kb = ph.CollisionKey(key, 0, 0), ph.CollisionKey(key, 1, 0)
        sa, ma = shape_and_motion(ka, a, np.zeros(3))
        sb, mb = shape_and_motion(kb, b, np.zeros(3))
        models[key], paths[key] = ph.CollisionModel([sb, sa]), {ka: ma, kb: mb}
        policies[key] = ph.CollisionPolicy(margin=0.25)
    with ph.WorldCollisionWorkspace(models.items(), tyx_mode, cache_root=tmp_path) as batch:
        result = batch.detect(paths, policies)
        for key, gap in ((18, 1 / 16), (4, -1 / 32)):
            (row,) = result[key].pairs
            assert row.status == ("overlap" if gap < 0 else "separated")
            assert row.witness.source == "host-convex-support"
            assert abs(row.witness.gap_m - gap) <= 2e-5
        stats = batch.stats()
        assert stats["host_pair_table_slots"] == 2 and stats["completed_launches"] > 0
        if tyx_mode == "cuda":
            assert stats["device_pair_table_slots"] == 2 and stats["synchronizations"] > 0


def test_hull_extent_is_not_published_as_an_unregistered_rounded_material_surface():
    hull, cur, _, _ = control("oblique-hull", 1 / 16)
    ka, kb = ph.CollisionKey(0, 5, 0), ph.CollisionKey(0, 9, 0)
    a = ph.CollisionShape(ka, "hull", hull[3], radius=1 / 32)
    b, mb = shape_and_motion(kb, cur, np.zeros(3))
    model = ph.CollisionModel((a, b))
    with ph.CollisionWorkspace(model) as work:
        (row,) = ph.detect(
            model, {ka: ph.CollisionMotion(hull[2]), kb: mb}, ph.CollisionPolicy(margin=1 / 4), work
        ).pairs
    assert row.candidate and row.status == "unsupported" and row.witness is None
