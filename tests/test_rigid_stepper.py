"""The common runtime must own and validate every rigid field, including torque."""

from dataclasses import replace

import numpy as np
import pytest

from phymeo import (
    ParticleModel,
    ParticleState,
    RigidBodyInputs,
    RigidBodyModel,
    RigidBodyState,
    RigidSemiImplicitSolver,
    SemiImplicitSolver,
    StepInputs,
    Stepper,
    StepReport,
    StepResult,
)

STATE_FIELDS = ("position", "orientation", "velocity", "angular_velocity")


def scene(dtype=np.float64, count=1):
    return (
        RigidBodyModel(np.ones(count, dtype), np.ones((count, 3), dtype), np.zeros(3)),
        RigidBodyState(
            np.zeros((count, 3), dtype),
            np.tile(np.array([0, 0, 0, 1], dtype), (count, 1)),
            np.zeros((count, 3), dtype),
            np.tile(np.array([0, 0, 1], dtype), (count, 1)),
        ),
        RigidBodyInputs.zeros(count, dtype=dtype),
    )


class Candidate:
    """Deliberately untrusted test provider; this is not a learned solver."""

    def __init__(self, behavior):
        self.behavior, self.calls = behavior, 0
        self.capabilities = RigidSemiImplicitSolver().capabilities
        self.signature = RigidSemiImplicitSolver().problem_signature()

    def problem_signature(self):
        return self.signature

    def step(self, model, state, inputs, dt):
        self.calls += 1
        return self.behavior(model, state, inputs, dt)


def assert_state_equal(actual, expected):
    for field in STATE_FIELDS:
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_real_rigid_solver_is_accepted_with_owned_rotation(dtype):
    model, state, inputs = scene(dtype)
    inputs.force[0, 0] = 2
    result = Stepper(RigidSemiImplicitSolver(), required_features={"rotation"}).step(
        model, state, inputs, 0.25
    )
    assert result.report.accepted and not result.report.fallback_used
    np.testing.assert_allclose(result.state.position, [[0.125, 0, 0]], atol=1e-7)
    np.testing.assert_allclose(
        result.state.orientation, [[0, 0, np.sin(0.125), np.cos(0.125)]], atol=1e-7
    )
    for field in STATE_FIELDS:
        assert not np.shares_memory(getattr(result.state, field), getattr(state, field))
    np.testing.assert_array_equal(state.orientation, [[0, 0, 0, 1]])


def test_empty_rigid_state_preserves_all_array_shapes():
    model, state, inputs = scene(count=0)
    result = Stepper(RigidSemiImplicitSolver()).step(model, state, inputs, 0.1)
    assert result.report.accepted
    assert_state_equal(result.state, state)
    assert result.state.orientation.shape == (0, 4)


@pytest.mark.parametrize(
    "bad",
    [
        "quaternion_norm",
        "orientation_nan",
        "angular_inf",
        "orientation_shape",
        "angular_dtype",
        "body_count",
        "dtype",
        "particle_state",
    ],
)
def test_invalid_rigid_candidate_rejects_and_restores_every_original_field(bad):
    model, state, inputs = scene()

    def corrupt(model, working, inputs, dt):
        if bad == "quaternion_norm":
            working.orientation *= 2
        elif bad == "orientation_nan":
            working.orientation[:] = np.nan
        elif bad == "angular_inf":
            working.angular_velocity[:] = np.inf
        elif bad == "orientation_shape":
            working.orientation = np.zeros((1, 3))
        elif bad == "angular_dtype":
            working.angular_velocity = working.angular_velocity.astype(np.float32)
        elif bad == "body_count":
            working = scene(count=0)[1]
        elif bad == "dtype":
            working = scene(np.float32)[1]
        else:
            working = ParticleState(working.position, working.velocity)
        return StepResult(working, StepReport(accepted=True))

    result = Stepper(Candidate(corrupt)).step(model, state, inputs, 0.1)
    assert not result.report.accepted and result.report.reason
    assert_state_equal(result.state, state)


@pytest.mark.parametrize("field", STATE_FIELDS)
def test_fixed_body_checks_position_orientation_and_both_velocities(field):
    model, state, inputs = scene()
    model = RigidBodyModel(np.zeros(1), np.zeros((1, 3)), np.zeros(3))
    state.angular_velocity[:] = 0

    def move_fixed(model, working, inputs, dt):
        if field == "orientation":
            working.orientation *= -1  # even an equivalent sign must preserve fixed stored state
        else:
            getattr(working, field)[0, 0] = 1
        return StepResult(working, StepReport(accepted=True))

    result = Stepper(Candidate(move_fixed)).step(model, state, inputs, 0.1)
    assert not result.report.accepted and "fixed" in result.report.reason
    assert_state_equal(result.state, state)


def test_rigid_fallback_receives_original_orientation_angular_velocity_and_torque():
    model, state, inputs = scene()
    inputs.force[0, 0], inputs.torque[0, 2] = 2, 3
    before, original_inputs = state.copy(), inputs.copy()

    def corrupt(model, working, working_inputs, dt):
        for field in STATE_FIELDS:
            getattr(working, field)[:] = 99
        working_inputs.force[:] = 88
        working_inputs.torque[:] = 77
        return StepResult(working, StepReport(accepted=False, reason="rejected proposal"))

    def recover(model, working, working_inputs, dt):
        assert_state_equal(working, before)
        np.testing.assert_array_equal(working_inputs.force, original_inputs.force)
        np.testing.assert_array_equal(working_inputs.torque, original_inputs.torque)
        return RigidSemiImplicitSolver().step(model, working, working_inputs, dt)

    primary, fallback = Candidate(corrupt), Candidate(recover)
    result = Stepper(primary, fallback).step(model, state, inputs, 0.25)
    assert result.report.accepted and result.report.fallback_used
    assert result.report.primary_failure == "rejected proposal"
    assert primary.calls == fallback.calls == 1
    assert_state_equal(state, before)
    np.testing.assert_array_equal(inputs.force, original_inputs.force)
    np.testing.assert_array_equal(inputs.torque, original_inputs.torque)
    np.testing.assert_allclose(result.state.angular_velocity, [[0, 0, 1.75]])
    np.testing.assert_allclose(result.state.orientation, [[0, 0, np.sin(0.21875), np.cos(0.21875)]])


def test_result_does_not_alias_a_provider_retained_rigid_buffer():
    model, state, inputs = scene()
    retained = []

    def remember(model, working, inputs, dt):
        retained.append(working)
        return StepResult(working, StepReport(accepted=True))

    result = Stepper(Candidate(remember)).step(model, state, inputs, 0.1)
    for field in STATE_FIELDS:
        getattr(retained[0], field)[:] = 55
    assert result.report.accepted
    assert_state_equal(result.state, state)


def test_both_failed_rigid_attempts_return_original_with_both_reasons():
    model, state, inputs = scene()

    def reject(label):
        def behavior(model, working, inputs, dt):
            working.angular_velocity[:] = 123
            return StepResult(working, StepReport(accepted=False, reason=label))

        return Candidate(behavior)

    primary, fallback = reject("primary"), reject("fallback")
    result = Stepper(primary, fallback).step(model, state, inputs, 0.1)
    assert not result.report.accepted and result.report.fallback_used
    assert result.report.reason == "fallback" and result.report.primary_failure == "primary"
    assert primary.calls == fallback.calls == 1
    assert_state_equal(result.state, state)


@pytest.mark.parametrize("part", ["model", "state", "inputs"])
def test_particle_types_are_rejected_before_a_rigid_provider_is_called(part):
    model, state, inputs = scene()
    if part == "model":
        model = ParticleModel(np.ones(1))
    elif part == "state":
        state = ParticleState(np.zeros((1, 3)), np.zeros((1, 3)))
    else:
        inputs = StepInputs.zeros(1)
    provider = Candidate(lambda *args: pytest.fail("wrong type must never reach solver"))
    stepper = Stepper(provider)
    with pytest.raises(TypeError, match="type"):
        stepper.step(model, state, inputs, 0.1)
    assert provider.calls == 0


def test_rigid_force_and_torque_inputs_are_revalidated_before_attempt():
    model, state, inputs = scene()
    inputs.torque[0, 0] = np.nan
    provider = Candidate(lambda *args: pytest.fail("invalid torque must never reach solver"))
    stepper = Stepper(provider)
    with pytest.raises(ValueError):
        stepper.step(model, state, inputs, 0.1)
    assert provider.calls == 0


def test_solver_state_kind_cannot_change_after_stepper_construction():
    provider = Candidate(lambda *args: pytest.fail("changed state kind must not run"))
    stepper = Stepper(provider)
    provider.capabilities = replace(provider.capabilities, state_kind="particles")
    with pytest.raises(ValueError, match="state"):
        stepper.step(*scene(), 0.1)


def test_different_kinds_reject_even_when_signatures_are_spoofed_equal():
    primary = Candidate(lambda *args: None)
    fallback = Candidate(lambda *args: None)
    fallback.capabilities = replace(fallback.capabilities, state_kind="particles")
    with pytest.raises(ValueError, match="state"):
        Stepper(primary, fallback)


def test_particle_input_copy_is_owned_and_revalidates_mutated_values():
    inputs = StepInputs(np.array([[1.0, 2.0, 3.0]]))
    copy = inputs.copy()
    inputs.force[:] = 0
    np.testing.assert_array_equal(copy.force, [[1, 2, 3]])
    inputs.force[0, 0] = np.nan
    with pytest.raises(ValueError):
        inputs.copy()


@pytest.mark.parametrize("kind", ["particles", "rigid_bodies"])
@pytest.mark.parametrize("bad", [False, True])
def test_provider_subclass_copy_override_cannot_bypass_validation_or_ownership(kind, bad):
    if kind == "rigid_bodies":
        model, state, inputs = scene()
        solver, fields = RigidSemiImplicitSolver(), STATE_FIELDS
    else:
        model = ParticleModel(np.ones(1), np.zeros(3))
        state, inputs = ParticleState(np.zeros((1, 3)), np.zeros((1, 3))), StepInputs.zeros(1)
        solver, fields = SemiImplicitSolver(), ("position", "velocity")

    class CachedState(type(state)):
        def copy(self):
            return self

    retained = CachedState(**{field: getattr(state, field) for field in fields})
    if bad:
        retained.velocity[:] = np.nan
        if kind == "rigid_bodies":
            retained.orientation *= 2
            retained.angular_velocity[:] = np.nan
    provider = Candidate(lambda *args: StepResult(retained, StepReport(accepted=True)))
    provider.capabilities, provider.signature = solver.capabilities, solver.problem_signature()
    result = Stepper(provider).step(model, state, inputs, 0.1)
    assert result.report.accepted == (not bad)
    assert result.state is not retained
    retained.position[:] = 999
    for field in fields:
        np.testing.assert_array_equal(getattr(result.state, field), getattr(state, field))


@pytest.mark.parametrize("kind", ["particles", "rigid_bodies"])
def test_input_subclass_copy_overrides_cannot_leak_mutations_to_caller_or_fallback(kind):
    if kind == "rigid_bodies":
        model, original, original_inputs = scene()
        solver, fields, input_fields = RigidSemiImplicitSolver(), STATE_FIELDS, ("force", "torque")
    else:
        model = ParticleModel(np.ones(1), np.zeros(3))
        original = ParticleState(np.zeros((1, 3)), np.zeros((1, 3)))
        original_inputs = StepInputs.zeros(1)
        solver, fields, input_fields = SemiImplicitSolver(), ("position", "velocity"), ("force",)

    class CachedState(type(original)):
        def copy(self):
            return self

    class CachedInputs(type(original_inputs)):
        def copy(self):
            return self

    state = CachedState(**{field: getattr(original, field) for field in fields})
    inputs = CachedInputs(**{field: getattr(original_inputs, field) for field in input_fields})

    def corrupt(model, working, working_inputs, dt):
        working.position[:] = 999
        for field in input_fields:
            getattr(working_inputs, field)[:] = 888
        return StepResult(working, StepReport(accepted=False, reason="retry"))

    provider = Candidate(corrupt)
    provider.capabilities, provider.signature = solver.capabilities, solver.problem_signature()
    expected = solver.step(model, original, original_inputs, 0.1)
    result = Stepper(provider, solver).step(model, state, inputs, 0.1)
    assert result.report.accepted and result.report.fallback_used
    for field in fields:
        np.testing.assert_array_equal(getattr(state, field), getattr(original, field))
        np.testing.assert_array_equal(getattr(result.state, field), getattr(expected.state, field))
    for field in input_fields:
        np.testing.assert_array_equal(getattr(inputs, field), getattr(original_inputs, field))
