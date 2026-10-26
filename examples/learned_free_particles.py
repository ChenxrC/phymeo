"""A tiny data-fit provider with independent physical acceptance and fallback.

This two-coefficient regression demonstrates the extension contract. It is not
a general AI physics solver, neural network training system, or speed benchmark.
"""

import json
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np

from phymeo import (
    FreeParticleVerifier,
    ParticleModel,
    ParticleState,
    ProblemSignature,
    SemiImplicitSolver,
    SolverFailure,
    StepInputs,
    Stepper,
    StepReport,
    StepResult,
)


@dataclass(frozen=True, eq=False)
class LinearResponseProvider:
    coefficients: np.ndarray
    model_version: ClassVar[str] = "example-linear-response-v1"
    dt_range: ClassVar[tuple[float, float]] = (0.005, 0.05)
    capabilities = replace(
        SemiImplicitSolver.capabilities,
        backend="numpy:linear-response",
        dtypes=frozenset(["float64"]),
        provider_kind="learned",
    )

    def __post_init__(self):
        coefficients = np.array(self.coefficients, dtype=np.float64, copy=True)
        if coefficients.shape != (2,) or not np.isfinite(coefficients).all():
            raise ValueError("two finite gravity/force response coefficients are required")
        coefficients.flags.writeable = False
        object.__setattr__(self, "coefficients", coefficients)

    def problem_signature(self):
        return ProblemSignature()

    def step(self, model, state, inputs, dt):
        if not self.dt_range[0] <= dt <= self.dt_range[1]:
            raise SolverFailure("time step is outside the fitted provider's declared dt interval")
        candidate = state.copy()
        dynamic = model.inverse_mass > 0
        gravity_response, force_response = self.coefficients
        candidate.velocity[dynamic] += dt * (
            gravity_response * model.gravity
            + force_response * model.inverse_mass[dynamic, None] * inputs.force[dynamic]
        )
        candidate.position[dynamic] += dt * candidate.velocity[dynamic]
        candidate.velocity[~dynamic] = 0
        # This proposal does not know its physical error. Stepper must verify it.
        return StepResult(candidate, StepReport(True, backend=self.capabilities.backend))


def sample(rng, *, fixed=False):
    count = 8
    weights = rng.uniform(0.25, 2.0, count)
    if fixed:
        weights[0] = 0
    model = ParticleModel(weights, rng.uniform(-10, 10, 3))
    state = ParticleState(rng.uniform(-1, 1, (count, 3)), rng.uniform(-2, 2, (count, 3)))
    inputs = StepInputs(rng.uniform(-10, 10, (count, 3)))
    dt = rng.uniform(*LinearResponseProvider.dt_range)
    return model, state, inputs, dt


def fit():
    rng, features, targets = np.random.default_rng(20261027), [], []
    teacher = SemiImplicitSolver()
    for _ in range(64):
        model, state, inputs, dt = sample(rng)
        label = teacher.step(model, state, inputs, dt)
        if not label.report.accepted:
            raise RuntimeError("training teacher failed")
        features.append(
            np.column_stack(
                [
                    np.broadcast_to(dt * model.gravity, state.position.shape).ravel(),
                    (dt * model.inverse_mass[:, None] * inputs.force).ravel(),
                ]
            )
        )
        targets.append((label.state.velocity - state.velocity).ravel())
    coefficients, _, rank, _ = np.linalg.lstsq(
        np.concatenate(features), np.concatenate(targets), rcond=None
    )
    if rank != 2:
        raise RuntimeError("training data did not identify both response coefficients")
    return LinearResponseProvider(coefficients)


def main():
    provider = fit()
    verifier = FreeParticleVerifier(tolerance=1e-10)
    stepper = Stepper(provider, SemiImplicitSolver(), verifier=verifier, max_residual=1e-10)
    rng, max_residual, max_error = np.random.default_rng(20261103), 0.0, 0.0
    for _ in range(64):
        model, state, inputs, dt = sample(rng, fixed=True)
        result = stepper.step(model, state, inputs, dt)
        if not result.report.accepted or result.report.fallback_used:
            raise RuntimeError(f"held-out candidate did not pass: {result.report}")
        expected = SemiImplicitSolver().step(model, state, inputs, dt)
        error = float(
            np.max(np.hypot.reduce(result.state.position - expected.state.position, axis=1))
        )
        max_error = max(max_error, error)
        max_residual = max(max_residual, result.report.max_residual)
    corrupted = Stepper(
        LinearResponseProvider(provider.coefficients * 1.25),
        SemiImplicitSolver(),
        verifier=verifier,
        max_residual=1e-10,
    ).step(model, state, inputs, dt)
    outside = stepper.step(model, state, inputs, 0.1)
    for name, result, h in [("corrupted", corrupted, dt), ("outside_dt", outside, 0.1)]:
        if not result.report.accepted or not result.report.fallback_used:
            raise RuntimeError(f"{name}: expected a successful mathematical fallback")
        expected = SemiImplicitSolver().step(model, state, inputs, h)
        np.testing.assert_array_equal(result.state.position, expected.state.position)
        np.testing.assert_array_equal(result.state.velocity, expected.state.velocity)
    print(
        json.dumps(
            {
                "model_version": provider.model_version,
                "training_seed": 20261027,
                "held_out_seed": 20261103,
                "training_scenes": 64,
                "held_out_scenes": 64,
                "particles_per_scene": 8,
                "units": "SI; gravity/force response coefficients are dimensionless",
                "fitted_coefficients": provider.coefficients.tolist(),
                "dt_interval_seconds": provider.dt_range,
                "independent_residual_tolerance": 1e-10,
                "held_out_max_residual": max_residual,
                "held_out_max_position_difference_m": max_error,
                "held_out_fallbacks": 0,
                "corrupted_model_fallback": corrupted.report.fallback_used,
                "corrupted_model_reason": corrupted.report.primary_failure,
                "outside_dt_fallback": outside.report.fallback_used,
                "outside_dt_reason": outside.report.primary_failure,
                "scope": "two-coefficient data regression; no neural model, history, gradients or speed claim",
            }
        )
    )


if __name__ == "__main__":
    main()
