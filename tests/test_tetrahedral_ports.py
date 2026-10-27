"""Lumped-mass solid ports obey the same owned impulse exchange protocol."""

import importlib

import numpy as np
import pytest
from test_tetrahedral_material import unit_tetra

from phymeo.core.capabilities import SolverFailure
from phymeo.core.model import ParticleModel
from phymeo.core.state import ParticleState
from phymeo.coupling.ports import ParticlePointPort, exchange_impulses


def port_api():
    return importlib.import_module("phymeo.coupling.tetrahedral_port").TetrahedralPointPort


def port(indices=None):
    model = unit_tetra()
    state = ParticleState(model.rest_position, np.zeros_like(model.rest_position))
    return port_api()(model, state, [0, 1, 1, 2] if indices is None else indices)


def test_solid_port_sums_repeated_nodes_and_keeps_fixed_support_motion_zero():
    point = port()
    impulses = np.array([[99.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [0.0, 1, 0]])
    expected = [[0, 0, 0], [12, 0, 0], [12, 0, 0], [0, 4, 0]]
    np.testing.assert_allclose(point.response(impulses), expected, atol=1e-12)
    result = point.apply_impulses(impulses)
    np.testing.assert_allclose(
        result.velocity, [[0, 0, 0], [12, 0, 0], [0, 4, 0], [0, 0, 0]], atol=1e-12
    )
    np.testing.assert_array_equal(result.position, unit_tetra().rest_position)
    np.testing.assert_array_equal(point.snapshot().velocity, np.zeros((4, 3)))


def test_solid_particle_exchange_preserves_actual_momentum_and_equal_opposite_loads():
    solid = port([1])
    particle = ParticlePointPort(
        ParticleModel(np.array([2.0]), np.zeros(3)),
        ParticleState(np.array([[1.0, 0, 0]]), np.zeros((1, 3))),
        [0],
    )
    result = exchange_impulses(particle, solid, np.array([[0.0, 0.5, 0]]))
    np.testing.assert_allclose(result.states["particles"].velocity, [[0, 1, 0]], atol=1e-12)
    np.testing.assert_allclose(result.states["soft"].velocity[1], [0, -2, 0], atol=1e-12)
    np.testing.assert_allclose(
        0.5 * result.states["particles"].velocity[0] + 0.25 * result.states["soft"].velocity[1],
        0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(result.impulses["particles"], -result.impulses["soft"])


def test_solid_port_owns_inputs_model_indices_and_queries():
    model = unit_tetra()
    state, indices = ParticleState(model.rest_position, np.zeros((4, 3))), np.array([1])
    point = port_api()(model, state, indices)
    state.position[:] = 42
    indices[:] = 0
    model.inverse_mass.flags.writeable = True
    model.inverse_mass[:] = 99
    point.snapshot().position[:] = 12
    point.positions()[:] = 88
    np.testing.assert_array_equal(point.positions(), [[1.0, 0, 0]])
    np.testing.assert_allclose(point.response(np.array([[1.0, 0, 0]])), [[4, 0, 0]], atol=1e-12)


@pytest.mark.parametrize("indices", [[-1], [4], [True], [1.0], [[1]]])
def test_bad_solid_port_indices_rejected(indices):
    with pytest.raises((ValueError, TypeError)):
        port(indices)


def test_solid_port_empty_and_canceling_repeated_loads():
    empty = port(np.array([], dtype=int))
    assert empty.response(np.empty((0, 3))).shape == (0, 3)
    repeated = port([1, 1, 1])
    np.testing.assert_allclose(
        repeated.apply_impulses(np.array([[1e16, 0, 0], [1.0, 0, 0], [-1e16, 0, 0]])).velocity[1],
        [4, 0, 0],
        atol=1e-12,
    )


def test_solid_port_rejects_configuration_changes_and_fixed_velocity():
    point = port()
    changed = point.snapshot()
    changed.position[1, 0] += 0.01
    with pytest.raises(ValueError, match="configuration"):
        point.validate_candidate(changed)
    changed = point.snapshot()
    changed.velocity[0, 0] = 1
    with pytest.raises(ValueError, match="fixed"):
        point.validate_candidate(changed)


def test_solid_port_unrepresentable_impulse_cannot_corrupt_original():
    point = port([1])
    with pytest.raises(SolverFailure):
        point.apply_impulses(np.array([[1e308, 0, 0]]))
    np.testing.assert_array_equal(point.snapshot().velocity, np.zeros((4, 3)))
