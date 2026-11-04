"""Independent half-space supports through ordinary detect and finite material impulse rows."""

import json
from dataclasses import asdict
from itertools import product

import numpy as np
import pytest

import phymeo as ph

CORNERS = np.array(list(product((-0.5, 0.5), repeat=3)))
Q = np.array([0.0, 0.0, 0.0, 1.0])


def fixture(normal=(0, 0, 1), *, lift=0.0, reverse=False, transformed=False, hull_radius=0.0):
    plane, hull = (ph.CollisionKey(5, int(reverse), 0), ph.CollisionKey(5, int(not reverse), 0))
    frame = np.array(((0, 0, 1), (1, 0, 0), (0, 1, 0))) if transformed else np.eye(3)
    origin = np.array([2.25, -1.5, 0.375]) if transformed else np.zeros(3)
    canonical = np.vstack((CORNERS, [0.0, 0.0, -0.5], [0.0, 0.0, 0.0], [0.0, -0.5, -0.5]))
    ids = tuple(37 + 7 * i for i in range(len(canonical)))
    order = list(reversed(range(len(ids)))) if reverse else list(range(len(ids)))
    n = np.asarray(normal, dtype=float)
    anchor = CORNERS[0].copy()
    if tuple(normal) == (0, 0, 1):
        anchor += [1.25, -0.75, 0]
    center = np.array([0.0, 0.0, lift])
    positions = (canonical + center) @ frame.T + origin
    plane_anchor = frame @ anchor + origin
    shapes = ph.CollisionModel(
        (
            ph.CollisionShape(plane, "plane", (37,), normal=tuple(frame @ n)),
            ph.CollisionShape(hull, "hull", tuple(ids[i] for i in order), radius=hull_radius),
        )
    )
    paths = {plane: ph.CollisionMotion([plane_anchor]), hull: ph.CollisionMotion(positions[order])}
    n = frame @ (n / np.linalg.norm(n))
    projected = CORNERS @ np.asarray(normal, dtype=float)
    extreme = np.flatnonzero(projected == projected.min())
    return shapes, paths, plane, hull, n, {ids[i]: positions[i] for i in extreme}


def detect(model, paths, *, backend="host", cache_root=None):
    with ph.CollisionWorkspace(model, backend, cache_root=cache_root) as work:
        result = ph.detect(model, paths, ph.CollisionPolicy(margin=2), work)
        return result.pairs[0], work.stats()


@pytest.mark.parametrize(
    "normal,dim,lift",
    [
        ((0, 0, 1), 2, 0.0),
        ((0, 0, 1), 2, 0.125),
        ((0, 0, 1), 2, -0.25),
        ((3, 0, 4), 1, 0.0),
        ((2**-37, 0, 1), 1, 0.0),
        ((1, 2, 3), 0, 0.0),
    ],
)
@pytest.mark.parametrize("reverse,transformed", [(False, False), (True, True)])
def test_signed_halfspace_gap_and_all_actual_support_materials(
    normal, dim, lift, reverse, transformed, tmp_path
):
    model, paths, plane, hull, n, extreme = fixture(
        normal, lift=lift, reverse=reverse, transformed=transformed
    )
    row, _ = detect(model, paths)
    assert row.status == ("separated" if lift > 0 else "overlap" if lift < 0 else "touching"), row
    assert row.candidate and row.family == "RR" and row.sweep is None
    assert row.witness.gap_m == pytest.approx(lift, abs=1e-14)
    (patch,) = row.convex.end.patches
    sign = 1 if row.first == plane else -1
    np.testing.assert_allclose(patch.normal, sign * n, atol=1e-14, rtol=0)
    assert patch.dimension == dim and patch.area_m2 == pytest.approx(1.0 if dim == 2 else 0)
    expected = dict(extreme)
    for corner in patch.points:
        pp, hp, pw, hw, offset = (
            corner.first_point,
            corner.second_point,
            corner.first_weights,
            corner.second_weights,
            corner.first_offset,
        )
        if row.first != plane:
            pp, hp, pw, hw, offset = (
                corner.second_point,
                corner.first_point,
                corner.second_weights,
                corner.first_weights,
                corner.second_offset,
            )
        assert pw == ((37, 1.0),) and len(hw) == 1 and hw[0][1] == 1
        np.testing.assert_allclose(expected.pop(hw[0][0]), hp, atol=1e-14, rtol=0)
        np.testing.assert_allclose(np.asarray(hp) - lift * n, pp, atol=1e-14, rtol=0)
        np.testing.assert_allclose(paths[plane].start[0] + offset, pp, atol=1e-14, rtol=0)
        assert abs(n @ offset) < 1e-13
    assert not expected
    if not lift:
        for site, corner in zip(row.witness.intersection_points, patch.points, strict=True):
            assert site.point == corner.first_point == corner.second_point
            assert (
                site.first_offset == corner.first_offset
                and site.second_offset == corner.second_offset
            )
    (tmp_path / "plane-hull.json").write_text(json.dumps(asdict(row)), encoding="utf8")


def test_motion_and_rounded_hulls_do_not_publish_a_partial_static_result():
    model, paths, _, hull, _, _ = fixture()
    p = paths[hull].start
    paths[hull] = ph.CollisionMotion(p, p + [0, 0, 0.125])
    row, _ = detect(model, paths)
    assert row.status == "unsupported" and row.sweep is None and row.witness is None
    model, paths, *_ = fixture(hull_radius=0.125)
    assert detect(model, paths)[0].status == "unsupported"


def test_unrepresentable_anchor_offset_cannot_claim_the_plane_material_point():
    model, paths, plane, _, _, _ = fixture()
    paths[plane] = ph.CollisionMotion([[float(2**54), 0, -0.5]])
    row, _ = detect(model, paths)
    assert row.status == "unresolved" and row.witness is None


def test_polygon_carries_eccentric_gravity_without_false_roll_and_allows_separation(tmp_path):
    model, paths, plane, hull, _, _ = fixture()
    pair, _ = detect(model, paths)
    assert pair.status == "touching"
    patch = pair.convex.end.patches[0]
    xy = np.array([p.first_point for p in patch.points])
    ids = np.zeros(4, dtype=np.int64)
    body = ph.RigidBodyModel(np.ones(1), np.ones((1, 3)), np.array([0.0, 0.0, -8.0]))
    arm = ph.FloatingRobotModel(body, np.array([0.1875, 0.0, 0.0]))
    state = ph.FloatingRobotState(np.r_[np.zeros(3), Q], np.zeros(6))
    inputs = ph.FloatingRobotInputs(np.empty(0), np.zeros((1, 3)), np.zeros((1, 3)))
    floor = ph.RigidBodyModel(np.zeros(1), np.zeros((1, 3)), np.zeros(3))
    resting = ph.RigidBodyState(np.zeros((1, 3)), Q[None, :], np.zeros((1, 3)), np.zeros((1, 3)))
    rows = [(pair, {hull: ("box", i), plane: ("floor", i)}, i) for i in range(4)]
    attachments = {"box": (ids, xy), "floor": (ids, xy)}
    h = 1 / 64

    def step(start, ordered):
        return ph.step_floating_contacts(
            {"box": (arm, start, inputs)},
            attachments,
            ordered,
            [0.4] * len(ordered),
            [0.3] * len(ordered),
            h,
            fixed_bodies={"floor": (floor, resting)},
            local_vertices={hull: paths[hull].start, plane: paths[plane].start},
            local_normals={plane: (0, 0, 1)},
        )

    out = step(state, rows)
    assert out.report.accepted, out.report.reason
    j = out.impulses.world_impulses
    np.testing.assert_allclose(j.sum(0), [0, 0, 8 * h], atol=2e-9, rtol=0)
    np.testing.assert_allclose(
        np.cross(xy - arm.root_com_m, j).sum(0), np.zeros(3), atol=2e-9, rtol=0
    )
    assert (xy[:, 0] @ j[:, 2]) / j[:, 2].sum() == pytest.approx(0.1875, abs=2e-8)
    np.testing.assert_allclose(out.states["box"].v, np.zeros(6), atol=2e-9, rtol=0)
    incomplete = step(state, rows[:-1])
    assert not incomplete.report.accepted and incomplete.impulses is None
    upward = ph.FloatingRobotState(state.q, np.array([0, 0, 1, 0, 0, 0], dtype=float))
    exited = step(upward, rows)
    assert exited.report.accepted and not np.any(exited.impulses.world_impulses)
    assert exited.states["box"].q[2] > 0
    (tmp_path / "eccentric.json").write_text(
        json.dumps(
            {
                "world_impulses": j.tolist(),
                "center_of_pressure_x_m": (xy[:, 0] @ j[:, 2]) / j[:, 2].sum(),
                "remaining_v": out.states["box"].v.tolist(),
            }
        ),
        encoding="utf8",
    )


@pytest.mark.tyx
def test_imported_box_on_infinite_plane_after_real_device_candidate(tyx_mode, tmp_path):
    from pathlib import Path

    asset = ph.load_mjcf(
        Path(__file__).resolve().parents[1]
        / "reference_newton"
        / "robot"
        / "joint_interval_arm.xml"
    )
    cube = next(c.shape for c in asset.collisions if c.name == "carriage_tag")
    assert cube.kind == "hull" and len(cube.vertex_ids) == 8
    q = np.zeros(len(asset.joint_names))
    path = asset.collision_motion(q)[cube.key]
    key = ph.CollisionKey(cube.key.world, 99, 0)
    ground = ph.CollisionShape(key, "plane", (19,), normal=(0, 0, 1))
    model = ph.CollisionModel([cube, ground])
    z = -0.15
    assert np.min(path.start[:, 2]) == z
    paths = {cube.key: path, key: ph.CollisionMotion([[1.5, -0.25, z]])}
    row, stats = detect(model, paths, backend=tyx_mode, cache_root=tmp_path / "cache")
    assert row.status == "touching" and row.witness.source == "host-plane-hull"
    patch = row.convex.end.patches[0]
    assert len(patch.points) == 4 and patch.area_m2 == pytest.approx(0.02)
    assert stats["completed_launches"] > 0
    if tyx_mode == "cuda":
        assert stats["downloads"] > 0
    (tmp_path / "imported.json").write_text(
        json.dumps(
            {
                "device": tyx_mode,
                "points": len(patch.points),
                "area_m2": patch.area_m2,
                "launches": stats["completed_launches"],
                "downloads": stats["downloads"],
            }
        ),
        encoding="utf8",
    )


@pytest.mark.parametrize("normal,count", [((3, 0, 4), 2), ((1, 2, 3), 1)])
def test_all_edge_materials_or_the_single_point_accept_an_opening_impulse(normal, count):
    model, paths, plane, hull, n, _ = fixture(normal)
    pair, _ = detect(model, paths)
    assert pair.status == "touching"
    pts = np.array([p.first_point for p in pair.convex.end.patches[0].points])
    assert len(pts) == count
    ports = {}
    zeros = np.zeros((1, 3))
    ids = np.zeros(count, dtype=np.int64)
    for key, domain in ((plane, "floor"), (hull, "box")):
        mass = np.zeros(1) if key == plane else np.ones(1)
        model = ph.RigidBodyModel(mass, np.broadcast_to(mass[:, None], (1, 3)).copy(), np.zeros(3))
        state = ph.RigidBodyState(zeros, Q[None, :], zeros if key == plane else n[None, :], zeros)
        ports[domain] = ph.RigidPointPort(model, state, ids, pts, domain=domain)
    local = {k: path.start for k, path in paths.items()}
    rows = [(pair, {plane: ("floor", i), hull: ("box", i)}, i) for i in range(count)]
    if count == 1:
        rows = [rows[0][:2]]
    args = ([0.4] * count, [0.2] * count)
    out = ph.solve_witness_impulses(
        ports, rows, *args, local_vertices=local, local_normals={plane: n}
    )
    np.testing.assert_array_equal(out.world_impulses, np.zeros((count, 3)))
    shifted = {**local, plane: local[plane] + np.cross(n, (1, 0, 0)) / 8}
    with pytest.raises((ValueError, ph.SolverFailure), match="material|pose|polygon"):
        ph.solve_witness_impulses(
            ports, rows, *args, local_vertices=shifted, local_normals={plane: n}
        )


def test_invalid_coplanar_hull_and_world_filter_are_distinct():
    plane, hull = ph.CollisionKey(7, 0, 0), ph.CollisionKey(7, 1, 0)
    p = ph.CollisionShape(plane, "plane", (0,), normal=(0, 0, 1))
    h = ph.CollisionShape(hull, "hull", (0, 1, 2, 3))
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    paths = {plane: ph.CollisionMotion([[0, 0, 0]]), hull: ph.CollisionMotion(vertices)}
    row, _ = detect(ph.CollisionModel([p, h]), paths)
    assert row.status == "invalid_geometry" and row.witness is None
    alt = ph.CollisionShape(ph.CollisionKey(8, 1, 0), "hull", h.vertex_ids)
    row, _ = detect(ph.CollisionModel([p, alt]), {plane: paths[plane], alt.key: paths[hull]})
    assert row.status == "filtered" and row.reason == "different_world" and not row.candidate


def test_oblique_irregular_pyramid_support_is_the_true_pentagon():
    ring = np.array([(0, 0), (2, 0), (3, 1), (1, 3), (-1, 1)], dtype=float)
    base = np.column_stack((ring, -0.5 * ring[:, 0] - ring[:, 1]))
    vertices = np.vstack((base, [1, 1, 2], [1, 1, -1.5]))
    nodes = (42, 18, 7, 91, 35, 60, 22)
    floor, hull = ph.CollisionKey(0, 2, 2), ph.CollisionKey(0, 3, 7)
    m = ph.CollisionModel(
        [
            ph.CollisionShape(floor, "plane", (7,), normal=(1, 2, 2)),
            ph.CollisionShape(hull, "hull", nodes),
        ]
    )
    paths = {floor: ph.CollisionMotion([[2, -1, 0]]), hull: ph.CollisionMotion(vertices)}
    row, _ = detect(m, paths)
    assert row.status == "touching" and row.sweep is None
    patch = row.convex.end.patches[0]
    assert patch.dimension == 2 and patch.area_m2 == pytest.approx(10.5)
    assert {p.second_weights[0][0] for p in patch.points} == set(nodes[:5])
    polygon = np.array([v.first_point for v in patch.points])
    area_vector = (
        np.cross(polygon - polygon[0], np.roll(polygon, -1, axis=0) - polygon[0]).sum(0) / 2
    )
    np.testing.assert_allclose(area_vector, 10.5 * np.array([1, 2, 2]) / 3, rtol=0, atol=1e-12)


def test_reversing_plane_orientation_reverses_the_occupied_halfspace():
    model, paths, plane, hull, _, _ = fixture(lift=1 / 8)
    registered = next(shape for shape in model.shapes if shape.key == hull)
    opposite = ph.CollisionShape(plane, "plane", (37,), normal=(0, 0, -7))
    row, _ = detect(ph.CollisionModel((registered, opposite)), paths)
    assert row.status == "overlap" and row.witness.gap_m == -9 / 8
    assert row.witness.normal == (0.0, 0.0, -1.0)
    assert {p.second_point[2] for p in row.convex.end.patches[0].points} == {5 / 8}
