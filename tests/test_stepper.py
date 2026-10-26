"""Transactional acceptance, problem identity, and bounded fallback behavior."""

from dataclasses import replace

import numpy as np
import pytest

import phymeo
from phymeo import (
    DistanceConstraints,
    ParticleModel,
    ParticleState,
    Plane,
    SemiImplicitSolver,
    StepInputs,
    StepReport,
    StepResult,
    XPBDSolver,
)


def api(name):
    result = getattr(phymeo, name, None)
    assert result is not None, f"{name} is not implemented"
    return result


def scene(dtype=np.float64):
    return (
        ParticleModel(np.ones(1, dtype=dtype)),
        ParticleState(np.array([[1.0, 2.0, 3.0]], dtype=dtype), np.zeros((1, 3), dtype=dtype)),
        StepInputs.zeros(1, dtype=dtype),
    )


class Candidate:
    """Test-only foreign solver: transaction checks must not trust its buffers."""

    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = 0
        self.capabilities = api("SolverCapabilities")(
            backend="test-candidate",
            dtypes=frozenset(["float32", "float64"]),
            features=frozenset(["particles", "free_dynamics"]),
        )
        self.signature = SemiImplicitSolver().problem_signature()

    def problem_signature(self):
        return self.signature

    def step(self, model, state, inputs, dt):
        self.calls += 1
        return self.behavior(model, state, inputs, dt)


def test_primary_acceptance_returns_independent_state_without_fallback():
    model, state, inputs = scene()
    fallback = Candidate(lambda *args: pytest.fail("fallback must not run"))
    result = api("Stepper")(SemiImplicitSolver(), fallback).step(model, state, inputs, 0.1)
    assert result.report.accepted
    assert not result.report.fallback_used
    assert fallback.calls == 0
    np.testing.assert_allclose(result.state.position, [[1.0, 1.9019, 3.0]])
    np.testing.assert_array_equal(state.position, [[1.0, 2.0, 3.0]])


@pytest.mark.parametrize(
    "bad",
    [
        "nan",
        "inf",
        "shape",
        "dtype",
        "unconverged",
        "residual",
        "nan_residual",
        "negative_error",
        "negative_iterations",
    ],
)
def test_invalid_candidate_is_rejected_and_original_is_returned(bad):
    model, state, inputs = scene()

    def behavior(model, working, inputs, dt):
        report = StepReport(accepted=True)
        if bad == "nan":
            working.position[0, 0] = np.nan
        elif bad == "inf":
            working.velocity[0, 0] = np.inf
        elif bad == "shape":
            working = ParticleState(np.empty((0, 3)), np.empty((0, 3)))
        elif bad == "dtype":
            working = ParticleState(
                working.position.astype(np.float32), working.velocity.astype(np.float32)
            )
        elif bad == "unconverged":
            report = StepReport(accepted=False, reason="not converged")
        elif bad == "residual":
            report = StepReport(accepted=True, max_residual=0.2)
        elif bad == "nan_residual":
            report = StepReport(accepted=True, max_residual=np.nan)
        elif bad == "negative_error":
            report = StepReport(accepted=True, max_constraint_error=-1.0)
        else:
            report = StepReport(accepted=True, iterations=-1)
        return StepResult(working, report)

    result = api("Stepper")(Candidate(behavior)).step(model, state, inputs, 0.1)
    assert not result.report.accepted
    assert result.report.reason
    np.testing.assert_array_equal(result.state.position, state.position)
    np.testing.assert_array_equal(result.state.velocity, state.velocity)
    assert not np.shares_memory(result.state.position, state.position)


def test_fallback_receives_original_state_and_force_after_primary_corruption():
    model, state, inputs = scene()
    inputs.force[0, 0] = 2.0

    def corrupt(model, working, force, dt):
        working.position[:] = 999.0
        force.force[:] = 888.0
        return StepResult(working, StepReport(accepted=False, reason="candidate rejected"))

    def recover(model, working, force, dt):
        np.testing.assert_array_equal(working.position, state.position)
        np.testing.assert_array_equal(force.force, inputs.force)
        return SemiImplicitSolver().step(model, working, force, dt)

    primary, fallback = Candidate(corrupt), Candidate(recover)
    result = api("Stepper")(primary, fallback).step(model, state, inputs, 0.1)
    assert result.report.accepted
    assert result.report.fallback_used
    assert "candidate rejected" in result.report.primary_failure
    assert primary.calls == fallback.calls == 1
    np.testing.assert_allclose(result.state.position, [[1.02, 1.9019, 3.0]])
    np.testing.assert_array_equal(state.position, [[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(inputs.force, [[2.0, 0.0, 0.0]])


def test_both_failures_are_visible_and_retry_budget_is_one():
    def reject(label):
        return Candidate(
            lambda model, state, inputs, dt: StepResult(
                state, StepReport(accepted=False, reason=label)
            )
        )

    primary, fallback = reject("primary refused"), reject("fallback exhausted")
    model, state, inputs = scene()
    result = api("Stepper")(primary, fallback).step(model, state, inputs, 0.1)
    assert not result.report.accepted
    assert result.report.fallback_used
    assert "primary refused" in result.report.primary_failure
    assert "fallback exhausted" in result.report.reason
    assert primary.calls == fallback.calls == 1
    np.testing.assert_array_equal(result.state.position, state.position)


def test_controlled_failure_can_use_fallback_but_programming_errors_propagate():
    def unavailable(*args):
        raise api("SolverFailure")("outside model domain")

    model, state, inputs = scene()
    result = api("Stepper")(Candidate(unavailable), SemiImplicitSolver()).step(
        model, state, inputs, 0.1
    )
    assert result.report.accepted and result.report.fallback_used
    assert "outside model domain" in result.report.primary_failure

    def broken(*args):
        raise RuntimeError("implementation bug")

    with pytest.raises(RuntimeError, match="implementation bug"):
        api("Stepper")(Candidate(broken), SemiImplicitSolver()).step(model, state, inputs, 0.1)


@pytest.mark.parametrize("kind", ["position", "velocity"])
def test_fixed_particle_invariant_is_independently_checked(kind):
    model, state, inputs = scene()
    model = ParticleModel(np.zeros(1))

    def move_fixed(model, working, inputs, dt):
        getattr(working, kind)[0, 0] += 1.0
        return StepResult(working, StepReport(accepted=True))

    result = api("Stepper")(Candidate(move_fixed)).step(model, state, inputs, 0.1)
    assert not result.report.accepted
    assert "fixed" in result.report.reason


def chain(iterations):
    return XPBDSolver(
        DistanceConstraints(np.array([[0, 1], [1, 2]]), np.ones(2), np.zeros(2)),
        iterations=iterations,
        tolerance=1e-9,
    )


def test_real_xpbd_can_fall_back_to_same_problem_with_more_iterations():
    stepper = api("Stepper")(chain(1), chain(100), required_features={"distance"})
    model = ParticleModel(np.array([0.0, 1.0, 1.0]), np.zeros(3))
    state = ParticleState(
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]), np.zeros((3, 3))
    )
    result = stepper.step(model, state, StepInputs.zeros(3), 0.1)
    assert result.report.accepted and result.report.fallback_used
    assert result.report.max_residual < 1e-9
    np.testing.assert_allclose(result.state.position[:, 0], [0.0, 1.0, 2.0], atol=2e-9)
    np.testing.assert_array_equal(state.position[:, 0], [0.0, 2.0, 4.0])


def test_constrained_problem_cannot_fall_back_to_unconstrained_integrator():
    stepper = api("Stepper")
    with pytest.raises(ValueError, match="physical problem"):
        stepper(chain(1), SemiImplicitSolver())


@pytest.mark.parametrize("change", ["indices", "rest", "compliance", "plane"])
def test_problem_signature_includes_physics_parameters(change):
    stepper = api("Stepper")
    original = chain(1)
    indices = np.array([[0, 1], [1, 2]]) if change != "indices" else np.array([[0, 2], [1, 2]])
    modified = XPBDSolver(
        DistanceConstraints(
            indices,
            np.full(2, 2.0 if change == "rest" else 1.0),
            np.full(2, 0.1 if change == "compliance" else 0.0),
        ),
        plane=Plane() if change == "plane" else None,
    )
    with pytest.raises(ValueError, match="physical problem"):
        stepper(original, modified)


def test_signature_ignores_iterations_tolerance_and_reference_scale():
    api("Stepper")
    solver = chain(2)
    other = replace(solver, iterations=70, tolerance=1e-4, reference_length=0.01)
    assert solver.problem_signature() == other.problem_signature()


def test_capability_mismatch_rejects_before_call():
    candidate = Candidate(lambda *args: pytest.fail("must not execute"))
    with pytest.raises(ValueError, match="capabilit"):
        api("Stepper")(candidate, required_features={"distance"})
    with pytest.raises(ValueError, match="gradient"):
        api("Stepper")(candidate, gradient="smooth")
    candidate.capabilities = replace(candidate.capabilities, history="learned_hidden_state")
    with pytest.raises(ValueError, match="history"):
        api("Stepper")(candidate)


def test_unsupported_precision_rejects_before_either_solver_executes():
    candidate = Candidate(lambda *args: pytest.fail("must not execute"))
    candidate.capabilities = replace(candidate.capabilities, dtypes=frozenset(["float32"]))
    stepper = api("Stepper")(SemiImplicitSolver(), candidate)
    with pytest.raises(ValueError, match="dtype"):
        stepper.step(*scene(), 0.1)


@pytest.mark.parametrize("threshold", [-1.0, np.nan, np.inf, True])
def test_invalid_acceptance_threshold_rejected(threshold):
    stepper = api("Stepper")
    with pytest.raises(ValueError):
        stepper(SemiImplicitSolver(), max_residual=threshold)


@pytest.mark.tyx
def test_tyx_and_numpy_share_free_dynamics_problem_identity(tyx_mode, tmp_path):
    stepper = api("Stepper")
    with phymeo.TyxSemiImplicitSolver(mode=tyx_mode, cache_root=tmp_path) as solver:
        assert solver.problem_signature() == SemiImplicitSolver().problem_signature()
        result = stepper(solver, SemiImplicitSolver()).step(*scene(np.float32), 0.1)
        assert result.report.accepted and not result.report.fallback_used


def test_problem_changed_after_construction_is_rejected_before_execution():
    candidate = Candidate(lambda *args: pytest.fail("must not execute"))
    stepper = api("Stepper")(candidate)
    candidate.signature = replace(candidate.signature, plane=(1.0, "y-up-frictionless-e0-v1"))
    with pytest.raises(ValueError, match="physical problem"):
        stepper.step(*scene(), 0.1)


def test_invalid_input_does_not_trigger_a_fallback():
    candidate = Candidate(lambda *args: pytest.fail("must not execute"))
    stepper = api("Stepper")(candidate, candidate)
    with pytest.raises(ValueError, match="dt"):
        stepper.step(*scene(), -0.1)
    assert candidate.calls == 0


def test_accepted_output_is_detached_from_solver_owned_storage():
    model, state, inputs = scene()
    retained = state.copy()
    candidate = Candidate(lambda *args: StepResult(retained, StepReport(accepted=True)))
    result = api("Stepper")(candidate).step(model, state, inputs, 0.1)
    retained.position[:] = 999.0
    np.testing.assert_array_equal(result.state.position, state.position)
