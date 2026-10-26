"""Independent discrete solutions catch integration order and boundary errors."""

import numpy as np
import pytest

import phymeo
from phymeo import ParticleModel, ParticleState, StepInputs


def solver():
    implementation = getattr(phymeo, "SemiImplicitSolver", None)
    assert implementation is not None, "SemiImplicitSolver is not implemented"
    return implementation()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_free_fall_matches_discrete_solution_over_ten_steps(dtype):
    model = ParticleModel(np.ones(1, dtype=dtype), np.array([0.0, -10.0, 0.0]))
    state = ParticleState(np.array([[0.0, 10.0, 0.0]], dtype=dtype), np.zeros((1, 3), dtype=dtype))
    inputs = StepInputs.zeros(1, dtype=dtype)
    integrator = solver()
    for _ in range(10):
        result = integrator.step(model, state, inputs, 0.1)
        assert result.report.accepted
        state = result.state
    np.testing.assert_allclose(state.position, [[0.0, 4.5, 0.0]], atol=1e-6)
    np.testing.assert_allclose(state.velocity, [[0.0, -10.0, 0.0]], atol=1e-6)
    assert state.position.dtype == dtype


def test_force_is_mass_weighted_and_new_velocity_updates_position():
    model = ParticleModel(np.array([0.5]), np.array([0.0, -10.0, 0.0]))
    state = ParticleState(np.array([[1.0, 2.0, 3.0]]), np.array([[0.5, 0.0, -1.0]]))
    result = solver().step(model, state, StepInputs(np.array([[2.0, 4.0, 6.0]])), 0.25)
    np.testing.assert_allclose(result.state.velocity, [[0.75, -2.0, -0.25]], atol=1e-14)
    np.testing.assert_allclose(result.state.position, [[1.1875, 1.5, 2.9375]], atol=1e-14)


def test_fixed_particle_stays_fixed_and_has_zero_velocity():
    model = ParticleModel(np.array([0.0]))
    state = ParticleState(np.array([[1.0, 2.0, 3.0]]), np.array([[20.0, -30.0, 40.0]]))
    result = solver().step(model, state, StepInputs(np.full((1, 3), 100.0)), 0.25)
    np.testing.assert_array_equal(result.state.position, [[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(result.state.velocity, [[0.0, 0.0, 0.0]])


def test_step_leaves_input_unchanged_and_output_unaliased():
    model = ParticleModel(np.ones(1))
    state = ParticleState(np.zeros((1, 3)), np.zeros((1, 3)))
    inputs = StepInputs.zeros(1)
    result = solver().step(model, state, inputs, 0.1)
    result.state.position[:] = 123.0
    result.state.velocity[:] = 456.0
    np.testing.assert_array_equal(state.position, np.zeros((1, 3)))
    np.testing.assert_array_equal(state.velocity, np.zeros((1, 3)))
    np.testing.assert_array_equal(inputs.force, np.zeros((1, 3)))


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, np.inf, True, [0.1]])
def test_rejects_invalid_time_step(dt):
    integrator = solver()
    with pytest.raises((TypeError, ValueError)):
        integrator.step(
            ParticleModel(np.ones(1)),
            ParticleState(np.zeros((1, 3)), np.zeros((1, 3))),
            StepInputs.zeros(1),
            dt,
        )


@pytest.mark.parametrize("kind", ["model_count", "force_count", "model_dtype", "force_dtype"])
def test_rejects_mismatched_data_contract(kind):
    integrator = solver()
    model = ParticleModel(
        np.ones(
            2 if kind == "model_count" else 1,
            dtype=np.float32 if kind == "model_dtype" else np.float64,
        )
    )
    inputs = StepInputs.zeros(
        2 if kind == "force_count" else 1, dtype=np.float32 if kind == "force_dtype" else np.float64
    )
    state = ParticleState(np.zeros((1, 3)), np.zeros((1, 3)))
    with pytest.raises(ValueError):
        integrator.step(model, state, inputs, 0.1)


def test_revalidates_state_mutated_after_construction():
    integrator = solver()
    state = ParticleState(np.zeros((1, 3)), np.zeros((1, 3)))
    state.position[0, 0] = np.nan
    with pytest.raises(ValueError):
        integrator.step(ParticleModel(np.ones(1)), state, StepInputs.zeros(1), 0.1)


def test_empty_system_returns_empty_accepted_state():
    model = ParticleModel(np.empty(0))
    state = ParticleState(np.empty((0, 3)), np.empty((0, 3)))
    result = solver().step(model, state, StepInputs.zeros(0), 0.1)
    assert result.report.accepted
    assert result.state.position.shape == (0, 3)


def test_numerical_overflow_is_rejected_without_destroying_original_state():
    model = ParticleModel(np.array([1.0]), np.zeros(3))
    state = ParticleState(np.array([[1e308, 0.0, 0.0]]), np.array([[1e308, 0.0, 0.0]]))
    result = solver().step(model, state, StepInputs.zeros(1), 10.0)
    assert not result.report.accepted
    assert result.report.reason
    np.testing.assert_array_equal(result.state.position, state.position)
    assert np.isfinite(state.position).all()
