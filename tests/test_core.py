"""Validate the public data boundary before any solver can consume it."""

import numpy as np
import pytest

import phymeo


def api(name):
    implementation = getattr(phymeo, name, None)
    assert implementation is not None, f"Missing public implementation: {name}"
    return implementation


@pytest.mark.parametrize("mass", [[-1.0], [float("nan")], [float("inf")], [[1.0]]])
def test_model_rejects_invalid_inverse_mass(mass):
    with pytest.raises(ValueError):
        api("ParticleModel")(inverse_mass=np.asarray(mass))


@pytest.mark.parametrize("gravity", [[0.0, -9.81], [0, float("nan"), 0]])
def test_model_rejects_invalid_gravity(gravity):
    with pytest.raises(ValueError):
        api("ParticleModel")(inverse_mass=np.ones(2), gravity=np.asarray(gravity))


def test_model_owns_frozen_arrays_without_mutating_caller():
    source = np.array([0.0, 2.0], dtype=np.float32)
    model = api("ParticleModel")(inverse_mass=source)
    source[1] = 3.0
    np.testing.assert_array_equal(model.inverse_mass, [0.0, 2.0])
    assert source.flags.writeable
    with pytest.raises(ValueError):
        model.inverse_mass[1] = 4.0
    assert model.particle_count == 2


def test_empty_system_has_valid_shape_and_force_storage():
    model = api("ParticleModel")(inverse_mass=np.empty(0, dtype=np.float32))
    state = api("ParticleState")(np.empty((0, 3)), np.empty((0, 3)))
    inputs = api("StepInputs").zeros(model.particle_count, dtype=np.float64)
    assert state.particle_count == 0
    assert inputs.force.shape == (0, 3)
    assert inputs.force.dtype == np.float64


@pytest.mark.parametrize(
    "position,velocity",
    [
        (np.zeros((2, 2)), np.zeros((2, 2))),
        (np.zeros((2, 3)), np.zeros((1, 3))),
        ([[0.0, np.nan, 0.0]], [[0.0, 0.0, 0.0]]),
        ([[0.0, 0.0, 0.0]], [[0.0, np.inf, 0.0]]),
    ],
)
def test_state_rejects_invalid_shape_or_nonfinite_values(position, velocity):
    with pytest.raises(ValueError):
        api("ParticleState")(np.asarray(position), np.asarray(velocity))


def test_state_owns_input_and_copy_has_independent_mutable_storage():
    position = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    velocity = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    state = api("ParticleState")(position, velocity)
    position[0, 0] = 10.0
    np.testing.assert_array_equal(state.position, [[1.0, 2.0, 3.0]])
    clone = state.copy()
    clone.position[0, 1] = -20.0
    clone.velocity[0, 1] = -30.0
    assert state.position[0, 1] == 2.0
    assert state.velocity[0, 1] == 5.0
    assert clone.position.dtype == np.float32


@pytest.mark.parametrize("values", [np.array([[1, 2, 3]]), np.ones((1, 3), dtype=complex)])
def test_state_rejects_unsupported_storage_types(values):
    with pytest.raises((TypeError, ValueError)):
        api("ParticleState")(values, values)


def test_state_rejects_mixed_precision():
    with pytest.raises(ValueError):
        api("ParticleState")(np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3)))


def test_inputs_validate_shape_and_finite_values():
    inputs_type = api("StepInputs")
    for invalid in (np.ones((2, 2)), np.array([[0, np.nan, 0]])):
        with pytest.raises(ValueError):
            inputs_type(force=invalid)
    valid = inputs_type.zeros(2, dtype=np.float32)
    np.testing.assert_array_equal(valid.force, np.zeros((2, 3)))
    with pytest.raises(ValueError):
        inputs_type.zeros(-1)
