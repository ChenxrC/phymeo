"""Real sphere contact witnesses must bind the current floating material sites."""

from math import fsum

import numpy as np
import pytest

import phymeo as ph


def _sphere_model():
    keys = (ph.CollisionKey(7, 1, 0), ph.CollisionKey(7, 2, 0))
    return ph.CollisionModel(
        tuple(ph.CollisionShape(k, "sphere", (0,), radius=1.0) for k in keys)
    )


def _touch(model, centers):
    motion = {
        shape.key: ph.CollisionMotion([center])
        for shape, center in zip(model.shapes, centers, strict=True)
    }
    with ph.CollisionWorkspace(model, "host") as workspace:
        row, = ph.detect(model, motion, ph.CollisionPolicy(), workspace).pairs
    assert row.candidate and row.status == "touching" and row.witness.gap_m == 0
    return row


def _ports(centers, normal, orientation, lateral_offset=0.0):
    root = ph.RigidBodyModel(np.ones(1), np.ones((1, 3)), np.zeros(3))
    model = ph.FloatingRobotModel(root, np.zeros(3))
    ports, states = {}, {}
    for name, center, sign in zip(("left", "right"), centers, (1, -1), strict=True):
        state = ph.FloatingRobotState(np.r_[center, orientation], np.r_[sign * normal, np.zeros(3)])
        local = sign * np.array([[1.0, lateral_offset, 0.0]])
        ports[name] = ph.FloatingRobotPointPort(model, state, np.array([0]), local, domain=name)
        states[name] = state
    return ports, states


def _solve(row, ports, states):
    initial = {k: (s.q.copy(), s.v.copy()) for k, s in states.items()}
    binding = {row.first: ("left", 0), row.second: ("right", 0)}
    try:
        return ph.solve_witness_impulses(
            ports, [(row, binding)], [0.0], [0.0], position_tolerance=0.0
        )
    finally:
        for k, (q, v) in initial.items():
            snapshot = ports[k].snapshot()
            np.testing.assert_array_equal(states[k].q, q)
            np.testing.assert_array_equal(states[k].v, v)
            np.testing.assert_array_equal(snapshot.q, q)
            np.testing.assert_array_equal(snapshot.v, v)


def _check_points(row, ports, expected):
    np.testing.assert_array_equal(row.witness.first_point, expected)
    np.testing.assert_array_equal(row.witness.second_point, expected)
    for name in ("left", "right"):
        np.testing.assert_array_equal(ports[name].positions()[0], expected)


def _check_central_impulse(result, normal):
    np.testing.assert_allclose(result.world_impulses[0], normal, atol=2e-12, rtol=0)
    assert result.normal_impulses[0] == pytest.approx(1.0, abs=2e-12)
    for name in ("left", "right"):
        np.testing.assert_allclose(result.states[name].v, np.zeros(6), atol=2e-12, rtol=0)


def test_zero_tolerance_rejects_physically_distinct_sites_aliased_in_world_fp64():
    model = _sphere_model()
    origin = np.array([0.0, 2.0**54, 0.0])
    normal = np.array([1.0, 0.0, 0.0])
    centers = (origin - normal, origin + normal)
    row = _touch(model, centers)
    ports, states = _ports(centers, normal, (0, 0, 0, 1), lateral_offset=0.25)
    _check_points(row, ports, origin)
    # Cancel the common world origin without losing either actual local offset.
    assert fsum((centers[0][1], 0.25, -origin[1])) == 0.25
    assert fsum((centers[1][1], -0.25, -origin[1])) == -0.25
    try:
        result = _solve(row, ports, states)
    except (ValueError, ph.SolverFailure):
        return
    # Both spin increments point along +z. The two COMs and linear impulses
    # are on the x-axis relative to origin, so their orbital change is zero.
    total_angular_z = sum(result.states[k].v[5] for k in ("left", "right"))
    pytest.fail(
        f"accepted sites 0.5 m apart at zero tolerance: impulse={result.world_impulses[0]}, "
        f"total angular change={total_angular_z} kg m2/s (analytic alias gives 8/17)"
    )


def test_true_coincident_large_world_sites_receive_the_unit_central_impulse():
    model = _sphere_model()
    origin = np.array([0.0, 2.0**54, 0.0])
    normal = np.array([1.0, 0.0, 0.0])
    centers = (origin - normal, origin + normal)
    row = _touch(model, centers)
    ports, states = _ports(centers, normal, (0, 0, 0, 1))
    _check_points(row, ports, origin)
    _check_central_impulse(_solve(row, ports, states), normal)


def test_old_normal_is_rejected_at_the_same_material_point_after_a_common_rotation():
    model = _sphere_model()
    x, y = np.eye(3)[:2]
    origin = np.zeros(3)
    old = _touch(model, (-x, x))
    centers = (-y, y)
    current = _touch(model, centers)
    # The exact coordinate cycle maps the same registered local sphere sites
    # (+1,0,0) and (-1,0,0) to the same world origin after rotating X into Y.
    ports, states = _ports(centers, y, (0.5, 0.5, 0.5, 0.5))
    _check_points(old, ports, origin)
    _check_points(current, ports, origin)
    np.testing.assert_array_equal(old.witness.normal, x)
    np.testing.assert_array_equal(current.witness.normal, y)
    _check_central_impulse(_solve(current, ports, states), y)
    try:
        result = _solve(old, ports, states)
    except (ValueError, ph.SolverFailure):
        return
    pytest.fail(
        f"accepted stale +X normal at the same point: impulse={result.world_impulses[0]}, "
        "the actual fresh +Y contact gives (0,1,0) Ns"
    )
