"""Independent physics validation for untrusted numerical candidate reports."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

import phymeo
from phymeo import (
    ParticleModel,
    ParticleState,
    Plane,
    ProblemSignature,
    SemiImplicitSolver,
    SolverFailure,
    StepInputs,
    Stepper,
    StepReport,
    StepResult,
    XPBDSolver,
)


def api(name):
    result = getattr(phymeo, name, None)
    assert result is not None, f"missing independent verification API: {name}"
    return result


def scene(dtype=np.float64):
    model = ParticleModel(np.array([0.5, 0], dtype), np.array([0, -10, 0], dtype))
    state = ParticleState(
        np.array([[1, 2, 3], [4, 5, 6]], dtype), np.array([[1, 2, 3], [9, 8, 7]], dtype)
    )
    inputs = StepInputs(np.array([[4, 0, 0], [999, 999, 999]], dtype))
    exact = ParticleState(
        np.array([[1.375, 1.875, 3.75], [4, 5, 6]], dtype),
        np.array([[1.5, -0.5, 3], [0, 0, 0]], dtype),
    )
    return model, state, inputs, exact


class Proposal:
    def __init__(self, candidate, *, learned=True):
        self.candidate = candidate
        self.capabilities = replace(
            SemiImplicitSolver.capabilities,
            backend="example:proposal",
            provider_kind="learned" if learned else "mathematical",
        )

    def problem_signature(self):
        return ProblemSignature()

    def step(self, model, state, inputs, dt):
        return StepResult(self.candidate, StepReport(True, backend=self.capabilities.backend))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_independent_free_particle_equations_match_hand_computed_solution(dtype):
    model, state, inputs, exact = scene(dtype)
    verifier = api("FreeParticleVerifier")(tolerance=0)
    report = verifier.verify(model, state, inputs, 0.25, exact)
    assert report.accepted
    assert report.max_residual == report.max_constraint_error == 0
    assert verifier.state_kind == "particles"
    assert verifier.problem_signature() == ProblemSignature()


def test_position_and_velocity_errors_have_distinct_physical_scales():
    model, state, inputs, candidate = scene()
    candidate.position[0, :2] += [0.003, 0.004]
    candidate.velocity[0, :2] += [0.03, 0.04]
    report = api("FreeParticleVerifier")(
        tolerance=0.006, reference_length=2, reference_speed=10
    ).verify(model, state, inputs, 0.25, candidate)
    assert report.accepted
    assert report.max_constraint_error == pytest.approx(0.005)
    assert report.max_residual == pytest.approx(0.005)


@pytest.mark.parametrize("name", ["reference_length", "reference_speed", "tolerance"])
@pytest.mark.parametrize("value", [True, "1", np.inf, np.nan, -1])
def test_verifier_rejects_invalid_settings(name, value):
    with pytest.raises(ValueError):
        api("FreeParticleVerifier")(**{name: value})


@pytest.mark.parametrize("name", ["reference_length", "reference_speed"])
def test_verifier_requires_positive_scales(name):
    with pytest.raises(ValueError):
        api("FreeParticleVerifier")(**{name: 0})


@pytest.mark.parametrize("value", ["neural", "", None, True])
def test_provider_kind_is_explicit_and_validated(value):
    with pytest.raises(ValueError, match="provider_kind"):
        replace(SemiImplicitSolver.capabilities, provider_kind=value)


def test_learned_primary_or_fallback_requires_independent_verifier():
    _, _, _, candidate = scene()
    learned = Proposal(candidate)
    for primary, fallback in [(learned, None), (SemiImplicitSolver(), learned)]:
        with pytest.raises(ValueError, match="learned.*verifier"):
            Stepper(primary, fallback)


def test_learned_correct_candidate_is_accepted_with_independent_verifier():
    model, state, inputs, candidate = scene()
    result = Stepper(Proposal(candidate), verifier=api("FreeParticleVerifier")()).step(
        model, state, inputs, 0.25
    )
    assert result.report.accepted and not result.report.fallback_used
    assert result.report.backend == "example:proposal"
    assert not np.shares_memory(result.state.position, candidate.position)
    np.testing.assert_array_equal(result.state.position, candidate.position)


def test_zero_self_report_cannot_hide_wrong_dynamics_and_fallback_restarts_original():
    model, state, inputs, candidate = scene()
    expected = candidate.copy()
    candidate.position[0, 0] += 1
    result = Stepper(
        Proposal(candidate), SemiImplicitSolver(), verifier=api("FreeParticleVerifier")()
    ).step(model, state, inputs, 0.25)
    assert result.report.accepted and result.report.fallback_used
    assert "independent verification" in result.report.primary_failure
    assert result.report.backend == "numpy"
    np.testing.assert_array_equal(result.state.position, expected.position)
    np.testing.assert_array_equal(state.position, [[1, 2, 3], [4, 5, 6]])


def test_wrong_mathematical_fallback_is_also_independently_rejected():
    model, state, inputs, candidate = scene()
    candidate.velocity[0, 0] += 1
    result = Stepper(
        Proposal(candidate),
        Proposal(candidate, learned=False),
        verifier=api("FreeParticleVerifier")(),
    ).step(model, state, inputs, 0.25)
    assert not result.report.accepted and result.report.fallback_used
    assert "independent verification" in result.report.reason
    np.testing.assert_array_equal(result.state.position, state.position)
    np.testing.assert_array_equal(result.state.velocity, state.velocity)


def test_verifier_cannot_repair_or_corrupt_candidate_by_mutating_its_copies():
    model, state, inputs, candidate = scene()
    expected = candidate.copy()

    class MutatingVerifier(api("FreeParticleVerifier")):
        def verify(self, model, state, inputs, dt, candidate):
            state.position[:] = 111
            inputs.force[:] = 222
            candidate.position[:] = 333
            return api("VerificationReport")(accepted=True)

    result = Stepper(Proposal(candidate), verifier=MutatingVerifier()).step(
        model, state, inputs, 0.25
    )
    assert result.report.accepted
    np.testing.assert_array_equal(result.state.position, expected.position)
    np.testing.assert_array_equal(candidate.position, expected.position)
    np.testing.assert_array_equal(state.position, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(inputs.force, [[4, 0, 0], [999, 999, 999]])


@pytest.mark.parametrize("change", ["kind", "signature"])
def test_verifier_must_match_frozen_state_and_physical_identity(change):
    class WrongVerifier(api("FreeParticleVerifier")):
        state_kind = "rigid_bodies" if change == "kind" else "particles"

        def problem_signature(self):
            return (
                ProblemSignature(semantics="different")
                if change == "signature"
                else ProblemSignature()
            )

    with pytest.raises(ValueError, match="verifier"):
        Stepper(SemiImplicitSolver(), verifier=WrongVerifier())
    with pytest.raises(ValueError, match="verifier"):
        Stepper(XPBDSolver(plane=Plane()), verifier=api("FreeParticleVerifier")())


def test_verifier_identity_is_rechecked_before_each_step():
    class ChangingVerifier(api("FreeParticleVerifier")):
        changed = False

        def problem_signature(self):
            return ProblemSignature(semantics="changed") if self.changed else ProblemSignature()

    verifier = ChangingVerifier()
    stepper = Stepper(SemiImplicitSolver(), verifier=verifier)
    object.__setattr__(verifier, "changed", True)
    model, state, inputs, _ = scene()
    with pytest.raises(ValueError, match="verifier"):
        stepper.step(model, state, inputs, 0.25)


@pytest.mark.parametrize("bad", ["type", "flag", "nan", "negative", "bool_error"])
def test_malformed_verification_cannot_commit(bad):
    model, state, inputs, candidate = scene()

    class BadVerifier(api("FreeParticleVerifier")):
        def verify(self, *args):
            if bad == "type":
                return StepReport(True)
            options = {
                "flag": {"accepted": 1},
                "nan": {"max_residual": np.nan},
                "negative": {"max_constraint_error": -1},
                "bool_error": {"max_residual": True},
            }[bad]
            return api("VerificationReport")(**{"accepted": True, **options})

    result = Stepper(Proposal(candidate), verifier=BadVerifier()).step(model, state, inputs, 0.25)
    assert not result.report.accepted
    assert "verification" in result.report.reason
    np.testing.assert_array_equal(result.state.position, state.position)


def test_stepper_threshold_applies_to_independent_residual_even_if_verifier_accepts():
    model, state, inputs, candidate = scene()
    candidate.position[0, 0] += 1e-3
    result = Stepper(Proposal(candidate), verifier=api("FreeParticleVerifier")(tolerance=0.1)).step(
        model, state, inputs, 0.25
    )
    assert not result.report.accepted
    assert result.report.max_residual == pytest.approx(1e-3)


def test_success_report_contains_independent_residual_without_losing_provenance():
    model, state, inputs, candidate = scene()
    candidate.position[0, 0] += 1e-6
    result = Stepper(Proposal(candidate), verifier=api("FreeParticleVerifier")()).step(
        model, state, inputs, 0.25
    )
    assert result.report.accepted
    assert result.report.max_residual == pytest.approx(1e-6)
    assert result.report.max_constraint_error == pytest.approx(1e-6)
    assert result.report.backend == "example:proposal"


@pytest.mark.parametrize(
    "error", [SolverFailure("out of validation domain"), RuntimeError("program bug")]
)
def test_only_expected_validation_failures_become_rejections(error):
    class ThrowingVerifier(api("FreeParticleVerifier")):
        def verify(self, *args):
            raise error

    model, state, inputs, candidate = scene()
    stepper = Stepper(Proposal(candidate), verifier=ThrowingVerifier())
    if isinstance(error, SolverFailure):
        result = stepper.step(model, state, inputs, 0.25)
        assert not result.report.accepted and "out of validation domain" in result.report.reason
    else:
        with pytest.raises(RuntimeError, match="program bug"):
            stepper.step(model, state, inputs, 0.25)


def test_verification_report_is_immutable():
    report = api("VerificationReport")(accepted=True)
    with pytest.raises(FrozenInstanceError):
        report.accepted = False


def test_empty_scene_passes_independent_equations():
    model = ParticleModel(np.empty(0))
    state = ParticleState(np.empty((0, 3)), np.empty((0, 3)))
    report = api("FreeParticleVerifier")(tolerance=0).verify(
        model, state, StepInputs.zeros(0), 0.1, state
    )
    assert report.accepted and report.max_residual == 0


def test_small_positive_velocity_error_never_normalizes_to_zero():
    model = ParticleModel(np.ones(1), np.zeros(3))
    state = ParticleState(np.zeros((1, 3)), np.zeros((1, 3)))
    candidate = state.copy()
    candidate.velocity[0, 0] = np.nextafter(0.0, 1.0)
    report = api("FreeParticleVerifier")(tolerance=0, reference_speed=2).verify(
        model, state, StepInputs.zeros(1), 0.1, candidate
    )
    assert not report.accepted and report.max_residual > 0


def test_nonfinite_reference_is_rejected_not_hidden_by_a_finite_candidate():
    model = ParticleModel(np.ones(1), np.zeros(3))
    state = ParticleState(np.zeros((1, 3)), np.full((1, 3), 1e308))
    report = api("FreeParticleVerifier")().verify(model, state, StepInputs.zeros(1), 10, state)
    assert not report.accepted and np.isinf(report.max_residual)


@pytest.mark.parametrize("large", ["position", "velocity"])
def test_large_base_values_cannot_cancel_representable_physical_error(large):
    model = ParticleModel(np.ones(1), np.zeros(3))
    state = ParticleState(np.zeros((1, 3)), np.zeros((1, 3)))
    inputs = StepInputs.zeros(1)
    if large == "position":
        state.position[0, 0] = 1e16
        state.velocity[0, 0] = 0.5
        candidate = state.copy()
    else:
        state.velocity[0, 0] = 1e16
        inputs.force[0, 0] = 0.5
        candidate = state.copy()
        candidate.position[0, 0] = 1e16
    report = api("FreeParticleVerifier")(tolerance=1e-6).verify(model, state, inputs, 1, candidate)
    assert not report.accepted
    assert report.max_residual == report.max_constraint_error == 0.5


def test_underflow_before_large_dt_cannot_accept_static_learned_candidate():
    model = ParticleModel(np.array([1e-300]), np.zeros(3))
    state = ParticleState(np.zeros((1, 3)), np.zeros((1, 3)))
    inputs = StepInputs(np.array([[1e-100, 0, 0]]))
    verifier = api("FreeParticleVerifier")()
    report = verifier.verify(model, state, inputs, 1e300, state)
    assert not report.accepted and report.max_residual > 0
    result = Stepper(Proposal(state), verifier=verifier).step(model, state, inputs, 1e300)
    assert not result.report.accepted


def test_inexact_nonzero_subnormal_product_cannot_be_amplified_into_false_success():
    model = ParticleModel(np.array([1e-300]), np.zeros(3))
    state = ParticleState(np.zeros((1, 3)), np.zeros((1, 3)))
    inputs = StepInputs(np.array([[5e-24, 0, 0]]))
    h = 1e162
    with np.errstate(under="ignore"):
        rounded_acceleration = model.inverse_mass[0] * inputs.force[0, 0]
    assert rounded_acceleration > 0
    candidate = state.copy()
    candidate.velocity[0, 0] = h * rounded_acceleration
    candidate.position[0, 0] = h * candidate.velocity[0, 0]
    report = api("FreeParticleVerifier")().verify(model, state, inputs, h, candidate)
    assert not report.accepted and report.max_residual > 0


def test_exact_subnormal_reference_product_is_retained():
    smallest = np.nextafter(0.0, 1.0)
    model = ParticleModel(np.array([smallest]), np.zeros(3))
    state = ParticleState(np.zeros((1, 3)), np.zeros((1, 3)))
    inputs = StepInputs(np.array([[1.0, 0, 0]]))
    candidate = state.copy()
    candidate.position[0, 0] = candidate.velocity[0, 0] = smallest
    report = api("FreeParticleVerifier")(tolerance=0).verify(model, state, inputs, 1, candidate)
    assert report.accepted and report.max_residual == 0
