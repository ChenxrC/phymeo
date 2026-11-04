"""Actual rotating-solid queries against the previously frozen finite-shaft oracles."""

import json
import runpy
from dataclasses import asdict
from math import acos, cos, pi, sin
from pathlib import Path

import numpy as np
import pytest
from test_mesh_mesh_contact import box, detect

import phymeo as ph

REF = Path(__file__).resolve().parents[1] / "reference_newton" / "rr"
CASES = json.loads((REF / "rotating_mesh_capsule_cases.json").read_text("utf8"))
CYCLE = np.array(((0, 0, 1), (1, 0, 0), (0, 1, 0)))


def spin(points, theta):
    c, s = cos(theta), sin(theta)
    return np.asarray(points) @ np.array(((c, s, 0), (-s, c, 0), (0, 0, 1)))


def scene(asset, spine, radius, theta, *, swap=False, moved=False, worlds=(0, 0), drift=(0, 0, 0)):
    local, faces = asset
    ids = tuple(41 + 7 * i for i in range(len(local)))
    mesh = ph.CollisionShape(
        ph.CollisionKey(worlds[0], int(swap), 0),
        "mesh",
        ids,
        triangles=tuple(tuple(ids[i] for i in f) for f in faces),
    )
    cap = ph.CollisionShape(
        ph.CollisionKey(worlds[1], int(not swap), 0), "capsule", (41, 93), radius
    )
    basis = CYCLE if moved else np.eye(3)
    origin = np.array((2.25, -0.75, 0.375)) if moved else np.zeros(3)
    turn = ph.RigidTurn(origin, origin + basis @ drift, basis @ (0, 0, 1), theta)
    paths = {
        mesh.key: ph.CollisionMotion(np.asarray(local) @ basis.T + origin, turn=turn),
        cap.key: ph.CollisionMotion(np.asarray(spine) @ basis.T + origin),
    }
    return ph.CollisionModel([mesh, cap]), paths, mesh, cap, basis


def materials(row, pose, mesh, cap, paths, basis, t):
    turn = paths[mesh.key].turn
    pivot = np.asarray(turn.pivot_start)
    travel = np.subtract(turn.pivot_end, pivot)
    local = (paths[mesh.key].start - pivot) @ basis
    xyz = spin(local, t * turn.angle) @ basis.T + pivot + t * travel
    lookup = dict(zip(mesh.vertex_ids, xyz, strict=True))
    axis = paths[cap.key].start
    if paths[cap.key].end is not None:
        axis = axis + t * (paths[cap.key].end - axis)
    spans = []
    for support in (*pose.supports, *pose.boundary):
        w = support.witness
        assert support.face in mesh.triangles
        mp, cp, mw, cw, radial = (
            w.first_point,
            w.second_point,
            w.first_weights,
            w.second_weights,
            w.second_offset,
        )
        if row.first != mesh.key:
            mp, cp, mw, cw, radial = (
                w.second_point,
                w.first_point,
                w.second_weights,
                w.first_weights,
                w.first_offset,
            )
        assert set(i for i, x in mw if x) <= set(support.face)
        np.testing.assert_allclose(sum(x * lookup[i] for i, x in mw), mp, atol=2e-9, rtol=0)
        anchor = sum(x * axis[cap.vertex_ids.index(i)] for i, x in cw)
        np.testing.assert_allclose(anchor + radial, cp, atol=2e-9, rtol=0)
        assert np.linalg.norm(radial) == pytest.approx(cap.radius, abs=2e-9)
        if w.normal is not None:
            assert np.dot(np.subtract(w.second_point, w.first_point), w.normal) == pytest.approx(
                w.gap_m, abs=2e-9
            )
        if w.capsule_axis_span is None:
            continue
        low, high = w.capsule_axis_span
        for c in (low, high):
            u = c.axis_parameter
            np.testing.assert_allclose(
                c.axis_point, (1 - u) * axis[0] + u * axis[1], atol=2e-9, rtol=0
            )
            np.testing.assert_allclose(
                sum(v * lookup[i] for i, v in c.face_weights), c.face_point, atol=2e-9, rtol=0
            )
        spans.append((low.axis_parameter, high.axis_parameter))
    return sorted(spans)


@pytest.mark.parametrize("case", CASES["cases"], ids=[c["id"] for c in CASES["cases"]])
def test_public_rotating_capsule_against_frozen_midshaft_and_clear_cases(case, tmp_path):
    if case["mesh"] == "bar":
        asset = box(*CASES["bar_bounds_m"])
    else:
        vertices, triangles = runpy.run_path(str(REF / "baseline.py"))["_u_geometry"](
            [1, 1, 1], False
        )
        asset = vertices, triangles.reshape(-1, 3)
    spine = [(0, case["axis_y_m"], z) for z in CASES["axis_ends_z_m"]]
    model, paths, mesh, cap, basis = scene(
        asset,
        spine,
        CASES["capsule_radius_m"],
        case["angle_rad"],
        swap=case["body_order"] == "capsule_first",
        moved=case["placement"] == "cyclic_translation",
        worlds=case["worlds"],
    )
    row = detect(model, paths)
    (tmp_path / "rotating-shaft.json").write_text(json.dumps(asdict(row)), encoding="utf8")
    if mesh.key.world != cap.key.world:
        assert row.status == "filtered" and row.reason == "different_world" and not row.candidate
        return
    if CASES["analytic_targets"][case["id"]]["status"] == "clear":
        assert row.candidate and row.status == "separated" and row.sweep.first_t is None
        assert row.mesh_round.start.relation == row.mesh_round.end.relation == "separated"
        return
    assert row.status == "hit", row
    trace = row.mesh_round
    assert trace.start.relation == trace.end.relation == "separated"
    event = trace.first_contact
    root = acos(1 / 6) / pi
    t = event.witness_time
    assert event.interval[0] <= root <= t <= event.interval[1]
    assert event.interval[1] - event.interval[0] <= 2**-27
    assert abs(case["axis_y_m"]) * cos(pi * t) - 0.125 <= cap.radius + 1e-14
    assert trace.start.axis_gap_m > 0 and trace.end.axis_gap_m > 0 and event.pose.axis_gap_m <= 0
    spans = materials(row, event.pose, mesh, cap, paths, basis, t)
    left, end = spans[0]
    for lo, hi in spans[1:]:
        assert lo <= end + 1e-10
        end = max(end, hi)
    np.testing.assert_allclose([left, end], [3 / 8, 5 / 8], atol=1e-10, rtol=0)
    for p, m in ((trace.start, 0), (trace.end, 1)):
        materials(row, p, mesh, cap, paths, basis, m)


def test_oblique_capsule_endcap_on_a_separately_translating_mesh_pivot():
    asset = box((-0.75, -0.75, -0.25), (0.75, 0.75, 0.25))
    spine = [(0, -0.25, 0.625), (0.25, 0.25, 1.125)]
    model, paths, mesh, cap, basis = scene(asset, spine, 0.125, -1.25, drift=(0, 0, 5 / 16))
    row = detect(model, paths)
    assert row.status == "hit", row
    event = row.mesh_round.first_contact
    assert event.interval[0] <= 0.8 <= event.witness_time <= event.interval[1]
    spans = materials(row, event.pose, mesh, cap, paths, basis, event.witness_time)
    assert spans and all(abs(s) < 1e-10 for pair in spans for s in pair)
    assert event.pose.witness.gap_m <= 0


def test_rotating_mesh_with_separately_rotating_capsule_remains_unavailable():
    asset = box((1, -0.125, -0.25), (2, 0.125, 0.25))
    model, paths, _, cap, _ = scene(asset, [(0, 1.5, -1), (0, 1.5, 1)], 0.125, pi)
    paths[cap.key] = ph.CollisionMotion(
        paths[cap.key].start, turn=ph.RigidTurn((0, 0, 0), (0, 0, 0), (1, 0, 0), 0.125)
    )
    row = detect(model, paths)
    assert row.status == "unsupported" and row.sweep is None


def test_isolated_shaft_corner_tangency_cannot_be_reported_clear():
    asset = box((0.25, 0.25, -0.125), (0.75, 1.0, 0.125))
    model, paths, _, _, _ = scene(asset, [(0, 1.5, -1), (0, 1.5, 1)], 0.25, 1.5)
    # Maximum xy radius of the single edge is 5/4. Its only tangency is at atan(3/4).
    row = detect(model, paths)
    assert row.candidate and row.status in ("unresolved", "hit")
    if row.status == "unresolved":
        assert row.sweep is None
    else:
        lo, hi = row.sweep.first_t
        expected = np.arctan(3 / 4) / 1.5
        assert lo <= expected <= hi and abs(row.sweep.witness_time - expected) < 1e-9


@pytest.mark.tyx
def test_swept_device_bounds_then_certified_host_shaft_in_an_affine_frame(tyx_mode, tmp_path):
    drift = np.array([0.125, -0.25, 0.0625])
    model, paths, mesh, cap, basis = scene(
        box(*CASES["bar_bounds_m"]), [(0, 1.5, -1), (0, 1.5, 1)], 0.125, pi, drift=drift
    )
    start = paths[cap.key].start
    paths[cap.key] = ph.CollisionMotion(start, start + drift)
    with ph.CollisionWorkspace(
        model, tyx_mode, broad_phase="bvh", cache_root=tmp_path / "cache"
    ) as work:
        result = ph.detect(model, paths, ph.CollisionPolicy(), work)
        (row,) = result.pairs
        assert result.backend == tyx_mode and row.status == "hit" and row.candidate
        event = row.mesh_round.first_contact
        assert event.interval[0] <= acos(1 / 6) / pi <= event.witness_time
        assert row.mesh_round.axis_samples > 0 and row.mesh_round.axis_speed_upper_m >= 4.71238898
        assert event.pose.witness.source == "host-closed-mesh-round"
        spans = materials(row, event.pose, mesh, cap, paths, basis, event.witness_time)
        assert min(x for x, _ in spans) == pytest.approx(3 / 8) and max(
            y for _, y in spans
        ) == pytest.approx(5 / 8)
        stats = work.stats()
        assert stats["completed_launches"] > 0
        if tyx_mode == "cuda":
            assert stats["downloads"] > 0
        (tmp_path / "shaft-device.json").write_text(
            json.dumps(
                dict(
                    device=tyx_mode,
                    launches=stats["completed_launches"],
                    downloads=stats["downloads"],
                    interval=event.interval,
                    sample=event.witness_time,
                    axis_samples=row.mesh_round.axis_samples,
                    spans=spans,
                )
            ),
            encoding="utf8",
        )


def test_finite_axis_remains_transversely_through_real_faces_at_the_end():
    asset = box((1, -0.125, -0.25), (2, 0.125, 0.25))
    radius, d, theta = 0.125, 1.375, 1.55
    model, paths, mesh, cap, basis = scene(asset, [(0, d, -1), (0, d, 1)], radius, theta)
    pair = detect(model, paths)
    assert pair.status == "hit", pair
    event = pair.mesh_round.first_contact
    assert event.interval[0] <= acos(0.25 / d) / theta <= event.witness_time
    assert pair.mesh_round.end.relation == "boundary_overlap"
    assert (
        pair.mesh_round.end.axis_location == "on_boundary"
        and pair.mesh_round.end.axis_gap_m == -radius
    )
    assert pair.mesh_round.observed_axis_error_upper_m < 1e-12
    materials(pair, pair.mesh_round.end, mesh, cap, paths, basis, 1)


def test_axis_extension_only_intersects_beyond_the_capsule_ends():
    asset = box((1, -0.125, -0.25), (2, 0.125, 0.25))
    model, paths, mesh, cap, basis = scene(asset, [(0, 1.5, 0.625), (0, 1.5, 0.875)], 0.125, pi)
    row = detect(model, paths)
    # Its infinite supporting line traverses the strip. Every finite spine point stays
    # at least .375 m above the upper face, leaving at least .25 m of skin clearance.
    assert row.candidate and row.status == "separated" and row.sweep.first_t is None
    for pose, t in ((row.mesh_round.start, 0), (row.mesh_round.end, 1)):
        assert pose.boundary_gap_m >= 0.25
        materials(row, pose, mesh, cap, paths, basis, t)


def test_two_coincident_material_ends_keep_their_finite_capsule_radius():
    asset = box((1, -0.125, -0.25), (2, 0.125, 0.25))
    model, paths, mesh, cap, basis = scene(asset, [(0, 1.5, 0.0625)] * 2, 0.125, pi)
    row = detect(model, paths)
    assert row.status == "hit", row
    event = row.mesh_round.first_contact
    assert event.interval[0] <= acos(1 / 6) / pi <= event.witness_time
    spans = materials(row, event.pose, mesh, cap, paths, basis, event.witness_time)
    assert spans and all(low == 0 and high == 1 for low, high in spans)


@pytest.mark.parametrize("lift", [0.0, 1 / 64])
def test_rotating_face_contains_coplanar_axis_but_not_a_lifted_axis(lift):
    asset = box((1, -0.125, -0.25), (2, 0.125, 0.25))
    spine = [(1.375, 0, 0.25 + lift), (1.625, 0, 0.25 + lift)]
    model, paths, mesh, cap, basis = scene(asset, spine, 0.125, 1 / 16)
    pair = detect(model, paths)
    assert pair.status == "hit" and pair.sweep.first_t == (0, 0), pair
    pose = pair.mesh_round.end
    assert pose.relation == "boundary_overlap"
    assert pose.axis_location == ("on_boundary" if not lift else "outside")
    assert pose.axis_gap_m == pytest.approx(lift - 0.125, abs=1e-12)
    materials(pair, pose, mesh, cap, paths, basis, 1)


@pytest.mark.parametrize("below", [0.0, 1 / 128])
def test_axis_meets_the_lower_material_edge_without_crossing_an_open_face(below):
    vertices, faces = box((-0.5, 0, 0), (0.5, 1, 1))
    # This prism meets z=0 only along the finite edge y=0, -1/2 <= x <= 1/2.
    x, y, z = vertices.T
    wedge = np.column_stack([x, y - z, y + z]), faces
    endpoints = [(0.125, -0.25, -below), (0.125, 0.25, -below)]
    model, paths, mesh, cap, basis = scene(wedge, endpoints, 1 / 16, 0.4)
    pair = detect(model, paths)
    assert pair.status == "hit" and pair.sweep.first_t == (0, 0), pair
    pose = pair.mesh_round.end
    assert pose.axis_location == ("on_boundary" if not below else "outside")
    assert pose.axis_gap_m == pytest.approx(below - 1 / 16, abs=1e-12)
    materials(pair, pose, mesh, cap, paths, basis, 1)
