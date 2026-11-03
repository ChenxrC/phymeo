"""Public solid CCD against a rotating closed mesh, with independent box and U oracles."""

import json
from dataclasses import asdict
from math import acos, atan2, cos, pi, sin

import numpy as np
import pytest
from test_mesh_mesh_contact import box, detect
from test_mesh_round import CYCLE, REF

import phymeo as ph


def scene(
    vertices, faces, center, radius, angle, *, reverse=False, transformed=False, drift=(0, 0, 0)
):
    matrix = CYCLE if transformed else np.eye(3)
    origin = np.array((2.25, -0.75, 0.375)) if transformed else np.zeros(3)
    drift = matrix @ drift
    ids = tuple(41 + 7 * i for i in range(len(vertices)))
    mesh = ph.CollisionShape(
        ph.CollisionKey(0, int(reverse), 1),
        "mesh",
        ids,
        triangles=tuple(tuple(ids[i] for i in face) for face in faces),
    )
    sphere = ph.CollisionShape(ph.CollisionKey(0, int(not reverse), 1), "sphere", (41,), radius)
    vertices = np.asarray(vertices) @ matrix.T + origin
    center = matrix @ center + origin
    turn = ph.RigidTurn(origin, origin + drift, matrix @ (0, 0, 1), angle)
    paths = {
        mesh.key: ph.CollisionMotion(vertices, turn=turn),
        sphere.key: ph.CollisionMotion([center], [center + drift]),
    }
    return ph.CollisionModel([mesh, sphere]), paths, mesh, sphere


def record(tmp_path, pair):
    trace = None if pair.mesh_round is None else asdict(pair.mesh_round)
    sweep = None if pair.sweep is None else asdict(pair.sweep)
    data = {"status": pair.status, "reason": pair.reason, "trace": trace, "sweep": sweep}
    (tmp_path / "mesh-sphere.json").write_text(json.dumps(data), encoding="utf-8")


def rot_z(point, angle):
    x, y, z = point
    return np.array([cos(angle) * x - sin(angle) * y, sin(angle) * x + cos(angle) * y, z])


def check_materials(pose, pair, mesh, sphere, paths, t, transformed=False):
    m, s = paths[mesh.key], paths[sphere.key]
    matrix = CYCLE if transformed else np.eye(3)
    pivot = np.asarray(m.turn.pivot_start)
    travel = np.subtract(m.turn.pivot_end, pivot)
    center = s.start[0] + t * (s.end[0] - s.start[0])
    node = {
        i: pivot + t * travel + matrix @ rot_z(matrix.T @ (p - pivot), t * m.turn.angle)
        for i, p in zip(mesh.vertex_ids, m.start, strict=True)
    }
    assert pose.supports
    for support in (*pose.supports, *pose.boundary):
        w = support.witness
        sides = [
            (w.first_weights, w.first_offset, w.first_point),
            (w.second_weights, w.second_offset, w.second_point),
        ]
        (wa, oa, pa), (wb, ob, pb) = sides if pair.first == mesh.key else sides[::-1]
        assert support.face in mesh.triangles and {i for i, x in wa if x} <= set(support.face)
        assert sum(x for _, x in wa) == pytest.approx(1) and min(x for _, x in wa) >= 0
        assert oa == (0, 0, 0) and wb == ((sphere.vertex_ids[0], 1),)
        np.testing.assert_allclose(sum(x * node[i] for i, x in wa), pa, atol=2e-10, rtol=0)
        np.testing.assert_allclose(center + ob, pb, atol=2e-10, rtol=0)
        assert np.linalg.norm(ob) == pytest.approx(sphere.radius, abs=2e-10)
        if w.normal is not None:
            assert np.linalg.norm(w.normal) == pytest.approx(1)
            delta = np.subtract(w.second_point, w.first_point)
            assert delta @ w.normal == pytest.approx(w.gap_m, abs=2e-10)


@pytest.mark.parametrize("reverse,transformed,sign", [(False, False, 1), (True, True, -1)])
def test_rotating_bar_enters_finite_sphere_between_separated_endpoints(
    reverse, transformed, sign, tmp_path
):
    # In the initial mesh frame the ball first meets the y side when 1.5*cos(theta)=.25.
    vertices, faces = box((1, -0.125, -0.25), (2, 0.125, 0.25), flip=reverse)
    model, paths, mesh, sphere = scene(
        vertices,
        faces,
        (0, sign * 1.5, 0.0625),
        0.125,
        sign * pi,
        reverse=reverse,
        transformed=transformed,
        drift=(0.25, -0.375, 0.125),
    )
    pair = detect(model, paths)
    record(tmp_path, pair)
    assert pair.status == "hit", pair
    trace = pair.mesh_round
    assert trace.start.relation == trace.end.relation == "separated"
    event = trace.first_contact
    lo, hi = event.interval
    assert lo <= acos(1 / 6) / pi <= hi and hi - lo <= 2**-27
    t = event.witness_time
    assert lo <= t <= hi and 1.5 * cos(pi * t) - 0.125 <= sphere.radius + 1e-14
    assert event.pose.relation in ("touching", "boundary_overlap")
    assert pair.sweep.first_t == (lo, hi) and pair.sweep.witness_time == t
    assert pair.sweep.witness == event.pose.witness and pair.witness == trace.end.witness
    for pose, moment in ((trace.start, 0), (event.pose, t), (trace.end, 1)):
        check_materials(pose, pair, mesh, sphere, paths, moment, transformed)


def test_rotating_concave_notch_proves_clear_for_the_whole_turn(tmp_path):
    import runpy

    vertices, faces = runpy.run_path(str(REF / "baseline.py"))["_u_geometry"]([1, 1, 1], False)
    model, paths, mesh, sphere = scene(vertices, faces.reshape(-1, 3), (0, 0.25, 0), 0.125, 2 * pi)
    # Every center lies at radius .25. Every U wall is >=1 from the pivot in the xy plane.
    pair = detect(model, paths)
    record(tmp_path, pair)
    assert pair.candidate and pair.status == "separated" and pair.sweep.first_t is None
    for pose, moment in ((pair.mesh_round.start, 0), (pair.mesh_round.end, 1)):
        assert pose.relation == "separated" and pose.boundary_gap_m >= 0.625
        check_materials(pose, pair, mesh, sphere, paths, moment)


def test_start_inside_the_closed_rotating_solid_is_immediate_occupancy(tmp_path):
    vertices, faces = box((1, -0.125, -0.25), (2, 0.125, 0.25))
    model, paths, mesh, sphere = scene(vertices, faces, (1.5, 0, 0), 0.0625, 0.75)
    pair = detect(model, paths)
    record(tmp_path, pair)
    assert pair.status == "hit" and pair.sweep.first_t == (0, 0)
    start = pair.mesh_round.first_contact.pose
    assert start.relation == "mesh_contains_round" and start.axis_location == "inside"
    assert start.boundary_gap_m == pytest.approx(0.0625)
    check_materials(start, pair, mesh, sphere, paths, 0)


@pytest.mark.parametrize("extra", [0, 1 / 32])
def test_single_corner_graze_is_not_discarded_or_replaced_by_a_dyadic_sample(extra, tmp_path):
    vertices, faces = box((0.25, 0.25, -0.125), (0.75, 1, 0.125))
    model, paths, _, _ = scene(vertices, faces, (0, 1.5 + extra, 0), 0.25, 1.5)
    # The (3/4,1) corner alone reaches radius 5/4. All other points have smaller xy radius.
    pair = detect(model, paths)
    record(tmp_path, pair)
    assert pair.candidate
    if extra:
        assert pair.status == "separated" and pair.sweep.first_t is None
    elif pair.status == "hit":
        lo, hi = pair.sweep.first_t
        assert lo <= atan2(0.75, 1) / 1.5 <= hi
        t = pair.sweep.witness_time
        separation = np.linalg.norm(rot_z((0.75, 1, 0), 1.5 * t) - (0, 1.5, 0))
        assert separation == pytest.approx(0.25, abs=1e-14)
    else:
        assert pair.status == "unresolved" and pair.sweep is None


def test_a_full_turn_over_budget_does_not_publish_an_unchecked_clear_sweep(tmp_path):
    vertices, faces = box((1, -0.125, -0.25), (2, 0.125, 0.25))
    model, paths, _, _ = scene(vertices, faces, (0, 1.5, 0), 0.125, 128)
    pair = detect(model, paths)
    record(tmp_path, pair)
    assert pair.candidate and pair.status == "unresolved" and pair.sweep is None


def test_independently_moving_pivot_reaches_a_stationary_sphere_with_known_time(tmp_path):
    vertices, faces = box((-0.5, -0.5, -0.25), (0.5, 0.5, 0.25))
    model, paths, mesh, sphere = scene(
        vertices, faces, (0.25, 0, 0.625), 0.125, -1.75, drift=(0, 0, 0.625)
    )
    center = paths[sphere.key].start
    paths[sphere.key] = ph.CollisionMotion(center, center)
    # The in-plane orbit remains inside the top square. Only the affine top z determines entry.
    pair = detect(model, paths)
    record(tmp_path, pair)
    assert pair.status == "hit" and pair.mesh_round.end.relation == "mesh_contains_round"
    event = pair.mesh_round.first_contact
    assert event.interval[0] <= 0.4 <= event.interval[1] and event.witness_time >= 0.4
    check_materials(event.pose, pair, mesh, sphere, paths, event.witness_time)


@pytest.mark.tyx
def test_device_classifies_candidates_and_host_certifies_the_actual_rotating_contact(
    tyx_mode, tmp_path
):
    vertices, faces = box((1, -0.125, -0.25), (2, 0.125, 0.25))
    model, paths, mesh, sphere = scene(vertices, faces, (0, 1.5, 0.0625), 0.125, pi)
    with ph.CollisionWorkspace(
        model, tyx_mode, broad_phase="bvh", cache_root=tmp_path / "cache"
    ) as work:
        report = ph.detect(model, paths, ph.CollisionPolicy(), work)
        (row,) = report.pairs
        record(tmp_path, row)
        assert report.backend == tyx_mode and row.candidate and row.status == "hit"
        event = row.mesh_round.first_contact
        assert event.interval[0] <= acos(1 / 6) / pi <= event.interval[1]
        assert event.pose.witness.source == "host-closed-mesh-round"
        check_materials(event.pose, row, mesh, sphere, paths, event.witness_time)
        stats = work.stats()
        assert stats["completed_launches"] > 0
        if tyx_mode == "cuda":
            assert stats["downloads"] > 0
        print(
            "device",
            tyx_mode,
            "launches",
            stats["completed_launches"],
            "downloads",
            stats["downloads"],
            "entry",
            event.interval,
            "sample",
            event.witness_time,
            "center samples",
            row.mesh_round.center_samples,
        )
