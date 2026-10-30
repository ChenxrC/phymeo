"""Authored G1 contract tests; execution is deferred to the serial queue."""

import numpy as np
import pytest

from phymeo.builders.parallel_jaw_gripper import gripper_inputs, make_parallel_jaw_gripper
from phymeo.dynamics.robot_kinematics import forward_kinematics, point_jacobians


def build(**changes):
    values = dict(
        half_gap_m=0.5,
        patch_half_extent_m=np.array([0.25, 0.125]),
        carriage_mass_kg=2.0,
        finger_mass_kg=np.array([1.0, 3.0]),
        inertia_diagonal_kg_m2=np.full((3, 3), 0.1),
        effort_limits_N=np.array([8.0, 2.0, 4.0]),
    )
    values.update(changes)
    return make_parallel_jaw_gripper(**values)


def test_massive_fingers_have_shared_finite_carriage():
    g = build()
    np.testing.assert_array_equal(g.model.parents, [-1, 0, 0])
    np.testing.assert_array_equal(g.model.mass, [2, 1, 3])
    np.testing.assert_array_equal(g.initial_state.q, np.zeros(3))
    np.testing.assert_array_equal(g.initial_state.v, np.zeros(3))
    assert g.model.joint_types == ("prismatic",) * 3
    assert g.opening_m(g.initial_state.q) == 1.0
    assert len(g.patches) == 2
    assert [p.link_index for p in g.patches] == [1, 2]


def test_closing_and_lifting_use_actual_coordinate_joint_geometry():
    g = build()
    q = np.array([0.25, 0.125, 0.0625])
    k = forward_kinematics(g.model, q)
    np.testing.assert_array_equal(
        k.position, [[0, 0, 0.25], [0, -0.375, 0.25], [0, 0.4375, 0.25]]
    )
    assert g.opening_m(q) == 0.8125
    linear, angular = point_jacobians(g.model, q, np.array([1, 2]), np.zeros((2, 3)))
    expected = np.zeros((2, 3, 3))
    expected[:, 2, 0] = 1
    expected[0, 1, 1] = 1
    expected[1, 1, 2] = -1
    np.testing.assert_array_equal(linear, expected)
    np.testing.assert_array_equal(angular, np.zeros_like(angular))
    np.testing.assert_array_equal(g.initial_state.q, np.zeros(3))


def test_patch_rectangles_are_finite_owned_and_not_radius_colliders():
    g = build()
    for p in g.patches:
        assert p.local_vertices.shape == (4, 3)
        np.testing.assert_array_equal(p.local_vertices[:, 1], np.zeros(4))
        assert set(p.local_vertices[:, 0]) == {-0.25, 0.25}
        assert set(p.local_vertices[:, 2]) == {-0.125, 0.125}
        assert not p.local_vertices.flags.writeable
        assert not hasattr(p, "radius_m")


def test_symmetric_finger_scalar_and_different_finite_dimensions():
    g = build(finger_mass_kg=0.75, patch_half_extent_m=[0.5, 0.25])
    np.testing.assert_array_equal(g.model.mass, [2, 0.75, 0.75])
    assert np.max(g.patches[0].local_vertices[:, 0]) == 0.5
    assert np.max(g.patches[1].local_vertices[:, 2]) == 0.25


def test_efforts_are_finite_bounded_requests_with_no_state_publication():
    g = build()
    q0, v0 = g.initial_state.q.copy(), g.initial_state.v.copy()
    closing = np.array([-2.0, 4.0])
    request = gripper_inputs(g, lift_effort_N=8.0, closing_effort_N=closing)
    np.testing.assert_array_equal(request.effort, [8, -2, 4])
    np.testing.assert_array_equal(request.force, np.zeros((3, 3)))
    np.testing.assert_array_equal(request.torque, np.zeros((3, 3)))
    closing[:] = 0
    np.testing.assert_array_equal(request.effort, [8, -2, 4])
    np.testing.assert_array_equal(g.initial_state.q, q0)
    np.testing.assert_array_equal(g.initial_state.v, v0)


@pytest.mark.parametrize("lift,closing", [(9.0, [0, 0]), (-9.0, [0, 0]), (0.0, [3, 0]), (0.0, [0, -5])])
def test_effort_limit_rejects_without_clipping(lift, closing):
    g = build()
    with pytest.raises(ValueError, match="limit"):
        gripper_inputs(g, lift_effort_N=lift, closing_effort_N=closing)
    np.testing.assert_array_equal(g.initial_state.q, np.zeros(3))


def test_inputs_and_distinct_builder_states_do_not_alias():
    extents, masses, limits, inertia = (
        np.array([0.25, 0.125]), np.array([1.0, 3.0]),
        np.array([8.0, 2.0, 4.0]), np.full((3, 3), 0.1),
    )
    g = build(patch_half_extent_m=extents, finger_mass_kg=masses,
              effort_limits_N=limits, inertia_diagonal_kg_m2=inertia)
    extents[:] = masses[:] = limits[:] = 99
    inertia[:] = 99
    np.testing.assert_array_equal(g.model.mass, [2, 1, 3])
    np.testing.assert_array_equal(g.effort_limits_N, [8, 2, 4])
    np.testing.assert_array_equal(g.model.inertia_diagonal, np.full((3, 3), 0.1))
    assert not g.effort_limits_N.flags.writeable
    other = build()
    g.initial_state.q[:] = 0.1
    np.testing.assert_array_equal(other.initial_state.q, np.zeros(3))


@pytest.mark.parametrize("changes", [
    {"half_gap_m": 0}, {"half_gap_m": True}, {"half_gap_m": float("nan")},
    {"patch_half_extent_m": [1, 0]}, {"patch_half_extent_m": [1, 2, 3]},
    {"carriage_mass_kg": -1}, {"finger_mass_kg": [1, 0]},
    {"finger_mass_kg": True}, {"finger_mass_kg": [1, 2, 3]},
    {"inertia_diagonal_kg_m2": np.tile([1.0, 1.0, 3.0], (3, 1))},
    {"effort_limits_N": [1, 2, float("inf")]}, {"effort_limits_N": [1, 0, 1]},
    {"gravity_m_s2": [0, 0, complex(0, 1)]},
])
def test_invalid_model_parameters_reject(changes):
    with pytest.raises((TypeError, ValueError)):
        build(**changes)


@pytest.mark.parametrize("lift,closing", [(True, [0, 0]), (float("inf"), [0, 0]), (0.0, [float("nan"), 0]), (0.0, [1])])
def test_invalid_effort_requests_reject(lift, closing):
    with pytest.raises((TypeError, ValueError)):
        gripper_inputs(build(), lift_effort_N=lift, closing_effort_N=closing)


def test_opening_is_a_diagnostic_not_a_hidden_joint_stop():
    g = build()
    q = np.array([0.0, 0.75, 0.75])
    assert g.opening_m(q) == -0.5
    np.testing.assert_array_equal(q, [0, 0.75, 0.75])
    with pytest.raises(ValueError):
        g.opening_m([0, 0])
