"""Carry a measured joint trajectory through the moving site and public collision APIs."""

import json
from math import atan, cos, hypot, pi, sin, sqrt

import numpy as np
import pytest

import phymeo as ph


def robot_motion(model, speed, backend, cache_root):
    drive = ph.DrivenRobotProgram(
        model, kp=np.zeros(1), kd=np.zeros(1), effort_limit=np.zeros(1)
    )
    entry = ph.RobotState(np.zeros(1), np.array([speed]))
    with drive.stepper(ph.StepSettings(substeps=1)) as stepper:
        step = stepper.step(drive.initial_state(entry), drive.inputs(np.zeros(1)), 1.0)
    assert step.report.accepted, step.report.reason
    end = step.state.domains["robot"]
    np.testing.assert_allclose(end.q, [speed], rtol=0, atol=1e-12)
    np.testing.assert_allclose(end.v, [speed], rtol=0, atol=1e-12)

    center = ph.RobotSite(0)
    world = ph.RobotSite(None)
    marker = ph.RobotSite(None, (1 / 4, 0, 3 / 8))
    first, last = (
        ph.measure_relative_site(model, state, world, center, backend=backend, cache_root=cache_root)
        for state in (entry, end)
    )
    reading = ph.measure_relative_site(model, end, center, marker, backend=backend, cache_root=cache_root)
    tolerance = 2e-12 if backend == "host" else 2e-5
    angle = float(end.q[0] - entry.q[0])
    np.testing.assert_allclose(first.rotation, np.eye(3), rtol=0, atol=tolerance)
    np.testing.assert_allclose(last.position_m, first.position_m, rtol=0, atol=tolerance)
    np.testing.assert_allclose(
        reading.position_m, (cos(angle) / 4, 0, sin(angle) / 4), rtol=0, atol=tolerance
    )
    np.testing.assert_allclose(
        reading.linear_velocity_m_s,
        (-speed * sin(angle) / 4, 0, speed * cos(angle) / 4),
        rtol=0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        reading.angular_velocity_rad_s, (0, -speed, 0), rtol=0, atol=tolerance
    )
    if backend != "host":
        assert reading.kernel_stats["completed_launches"] > 0
    turn = ph.RigidTurn(first.position_m, last.position_m, first.rotation[:, 1], angle)
    observations = {
        "q0_rad": float(entry.q[0]),
        "q1_rad": float(end.q[0]),
        "v1_rad_s": float(end.v[0]),
        "marker_in_link_m": reading.position_m.tolist(),
        "marker_in_link_m_s": reading.linear_velocity_m_s.tolist(),
        "marker_in_link_rad_s": reading.angular_velocity_rad_s.tolist(),
        "sensor_completed_launches": int(reading.kernel_stats.get("completed_launches", 0)),
    }
    return ph.CollisionMotion([first.position_m], turn=turn), observations


def trajectory_scene(world, path):
    plane_key, body_key = ph.CollisionKey(world, 1, 0), ph.CollisionKey(world, 2, 0)
    plane = ph.CollisionShape(plane_key, "plane", (91,), normal=(0, 0, 1))
    asset = ph.CurvedSolid("ellipsoid", (1 / 2, 1 / 4, 1 / 4))
    body = ph.CollisionShape(body_key, "ellipsoid", (47,), curved=asset)
    return ph.CollisionModel((body, plane)), {
        body_key: path,
        plane_key: ph.CollisionMotion([[0, 0, 0]]),
    }


def check_events(moving, still):
    assert still.status == "separated" and still.sweep is None
    assert still.witness.gap_m == pytest.approx(1 / 8)
    assert moving.status == "hit"
    assert moving.sweep.start_gap_m > 0 and moving.sweep.end_gap_m > 0
    alpha = atan(sqrt(5 / 7))
    first, last = moving.curved.first_contact, moving.curved.last_contact
    assert moving.sweep.first_t == first.interval and moving.sweep.witness == first.witness
    for target, event in ((alpha / 3, first), ((pi - alpha) / 3, last)):
        lo, hi = event.interval
        assert lo <= target <= hi and hi - lo <= 1e-6
        assert lo <= event.witness_time <= hi
        theta = 3 * event.witness_time
        reach = hypot(sin(theta) / 2, cos(theta) / 4)
        x = 3 * sin(theta) * cos(theta) / (16 * reach)
        w = event.witness
        np.testing.assert_allclose(w.first_point, (x, 0, 0), rtol=0, atol=1e-6)
        np.testing.assert_allclose(w.second_point, (x, 0, 3 / 8 - reach), rtol=0, atol=1e-6)
        np.testing.assert_allclose(w.normal, (0, 0, 1), rtol=0, atol=1e-5)
        assert w.first_weights == ((91, 1.0),) and w.second_weights == ((47, 1.0),)
        np.testing.assert_allclose(w.first_offset, w.first_point, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            np.array([0, 0, 3 / 8]) + w.second_offset, w.second_point, rtol=0, atol=1e-6
        )


def observe(backend, cache_root, *, partitioned):
    robot = ph.RobotModel(
        np.array([-1]),
        ("revolute",),
        np.array([[0, 0, 3 / 8]]),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[0.0, 1.0, 0.0]]),
        np.ones(1),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )
    bindings, paths, singles, states = [], {}, {}, {}
    for world, speed in ((17, 3.0), (5, 0.0)):
        path, states[world] = robot_motion(robot, speed, backend, cache_root)
        model, paths[world] = trajectory_scene(world, path)
        bindings.append((world, model))
        with ph.CollisionWorkspace(model, backend, cache_root=cache_root) as work:
            singles[world] = ph.detect(model, paths[world], ph.CollisionPolicy(), work)
    hit, clear = singles[17].pairs[0], singles[5].pairs[0]
    check_events(hit, clear)
    record = {
        "backend": backend,
        "states": states,
        "first": hit.curved.first_contact.interval,
        "last": hit.curved.last_contact.interval,
        "entry_plane_point_m": hit.curved.first_contact.witness.first_point,
        "entry_ellipsoid_point_m": hit.curved.first_contact.witness.second_point,
        "moving_end_gap_m": hit.sweep.end_gap_m,
        "stationary_gap_m": clear.witness.gap_m,
    }
    if partitioned:
        policy = {world: ph.CollisionPolicy() for world, _ in bindings}
        with ph.WorldCollisionWorkspace(bindings, backend, cache_root=cache_root) as work:
            result = work.detect(paths, policy)
            assert list(result) == [17, 5] and result == singles
            assert all(work.last_result(world) is result[world] for world in result)
            stats = work.stats()
            record["host_pair_slots"] = int(stats["host_pair_table_slots"])
            record["device_pair_slots"] = int(stats["device_pair_table_slots"])
            record["broad_completed_launches"] = int(stats["completed_launches"])
            assert stats["host_pair_table_slots"] == 2
            if backend == "cuda":
                assert stats["device_pair_table_slots"] == 2
                assert stats["completed_launches"] == 2 and stats["synchronizations"] == 1
    (cache_root / "driven-rotation.json").write_text(json.dumps(record, allow_nan=False), encoding="utf-8")


def test_measured_rotation_generates_two_contact_events(tmp_path):
    observe("host", tmp_path, partitioned=False)


def test_partitioned_motion_agrees_with_single_world_queries(tmp_path):
    observe("host", tmp_path, partitioned=True)


@pytest.mark.tyx
def test_measured_rotation_and_partitioned_geometry_on_tyx(tyx_mode, tmp_path):
    observe(tyx_mode, tmp_path, partitioned=True)
