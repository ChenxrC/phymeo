"""Force tables from real finite surface contacts, including inactive and disjoint worlds."""

from dataclasses import replace

import numpy as np
import pytest
from test_robot_contact_wrench import layer_fixture

import phymeo as ph


def test_world_slots_and_totals_have_different_denominators():
    system, state, _, keys = layer_fixture(worlds=2, friction=0.5)
    (a0, b0), (a1, b1) = keys
    with ph.CollisionWorkspace(system.model) as work:
        loads = system.sample(state, work)
    result = ph.measure_contact_matrix(loads, [a0, b0, a1], [b1, b0])
    expected = ph.measure_contact_wrench(loads, a0)
    np.testing.assert_allclose(result.total_force_n[0], [-1, 0, 2])
    np.testing.assert_allclose(result.total_force_n[1], -expected.force_n)
    np.testing.assert_allclose(result.force_n[0, 1], expected.force_n)
    np.testing.assert_allclose(result.friction_n[0, 1], expected.friction_force_n)
    np.testing.assert_array_equal(result.force_n[0, 0], np.zeros(3))
    np.testing.assert_array_equal(result.force_n[1], np.zeros((2, 3)))
    assert result.force_n[2, 0, 2] > 0
    point = loads.contacts[0].sites[0].material
    midpoint = (np.array(point.first_point) + point.second_point) / 2
    np.testing.assert_allclose(result.position_m[0, 1], midpoint)
    assert result.weight_n[0, 1] == pytest.approx(np.linalg.norm(expected.force_n))
    assert not result.total_force_n.flags.writeable
    empty = ph.measure_contact_matrix(loads, [a0], [])
    assert empty.force_n.shape == (1, 0, 3)
    np.testing.assert_array_equal(empty.total_force_n[0], result.total_force_n[0])


def test_counterpart_position_weights_survive_net_force_cancellation():
    system, state, _, keys = layer_fixture()
    a, b = keys[0]
    with ph.CollisionWorkspace(system.model) as work:
        loads = system.sample(state, work)
    pair = loads.contacts[0]
    s = pair.sites[0]
    one = replace(
        s,
        material=replace(s.material, first_point=(0, 0, 0), second_point=(0, 2, 0)),
        force_on_second_n=(0, 0, 3),
    )
    two = replace(
        s,
        material=replace(s.material, first_point=(2, 0, 0), second_point=(2, 2, 0)),
        force_on_second_n=(0, 0, -3),
    )
    table = ph.measure_contact_matrix(
        replace(loads, contacts=(replace(pair, sites=(one, two)),)), [a], [b]
    )
    np.testing.assert_array_equal(table.force_n[0, 0], np.zeros(3))
    np.testing.assert_array_equal(table.position_m[0, 0], [1, 1, 0])
    assert table.weight_n[0, 0] == 6
    with pytest.raises(ValueError, match="overlap"):
        ph.measure_contact_matrix(loads, [a, ph.ContactSelector(a.world, a.body)], [b])
    with pytest.raises(ValueError, match="cross"):
        ph.measure_contact_matrix(
            replace(
                loads,
                contacts=(replace(pair, second=ph.CollisionKey(b.world + 1, b.body, b.shape)),),
            ),
            [a],
            [b],
        )


def test_rejected_offset_rotation_does_not_publish_sensor_data():
    system, state, models, keys = layer_fixture()
    a, b = keys[0]
    with pytest.raises(ValueError, match="published"):
        ph.measure_contact_matrix(system, [a], [b])
    with ph.CollisionWorkspace(system.model) as work:
        solvers = (ph.SemiImplicitSolver(), ph.RigidSemiImplicitSolver())
        zero = ph.CoupledInputs.zeros(3, 1)
        # The off-centre load rotates this sphere attachment. The producer cannot
        # certify that continuous motion and must not publish a candidate.
        with pytest.raises(ph.SolverFailure, match="rotating attached shape"):
            system.step(state, zero, models, solvers, 1 / 1024, work)
        assert system.last_frame is None
        with pytest.raises(ValueError, match="published"):
            ph.measure_contact_matrix(system, [a], [b])
