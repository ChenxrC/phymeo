"""Independent FP64 verification of the velocity-first free-particle equations."""

from dataclasses import dataclass
from math import fsum
from typing import ClassVar

import numpy as np

from ..core.capabilities import ProblemSignature
from ..core.numerics import normalized_error
from ..core.state import ParticleState
from ..core.validation import finite_real, validate_step
from ..core.verification import VerificationReport


def _product(left, right):
    # IEEE underflow also covers an inexact, nonzero subnormal. Such lost
    # precision can be amplified by a later large dt, so reject the signal.
    with np.errstate(under="raise"):
        value = left * right
    if not np.isfinite(value).all():
        raise FloatingPointError("non-finite independent reference product")
    if np.any((left != 0) & (right != 0) & (value == 0)):
        raise FloatingPointError("independent reference product underflow")
    return value


def _sum_terms(*terms):
    try:
        value = np.fromiter(
            (fsum(row) for row in zip(*(term.ravel() for term in terms), strict=True)),
            dtype=np.float64,
            count=terms[0].size,
        ).reshape(terms[0].shape)
    except (OverflowError, ValueError) as error:
        raise FloatingPointError("non-finite independent reference sum") from error
    if not np.isfinite(value).all():
        raise FloatingPointError("non-finite independent reference sum")
    return value


@dataclass(frozen=True)
class FreeParticleVerifier:
    """Check position and velocity in explicit length/speed reference scales.

    Recomputes the declared discrete equations without calling a solver. This is
    a correctness reference, not an acceleration path or general contact check.
    """

    tolerance: float = 1e-5
    reference_length: float = 1.0
    reference_speed: float = 1.0
    state_kind: ClassVar[str] = "particles"

    def __post_init__(self):
        for name in ("tolerance", "reference_length", "reference_speed"):
            object.__setattr__(self, name, finite_real(getattr(self, name), name))
        if self.tolerance < 0 or self.reference_length <= 0 or self.reference_speed <= 0:
            raise ValueError("tolerance must be non-negative and reference scales positive")

    def problem_signature(self):
        return ProblemSignature()

    def verify(self, model, state_in, inputs, dt, candidate):
        snapshot, force, h = validate_step(model, state_in, inputs, dt)
        if not isinstance(candidate, ParticleState):
            raise TypeError("candidate must be ParticleState")
        candidate = ParticleState(candidate.position, candidate.velocity)
        if (
            candidate.position.shape != snapshot.position.shape
            or candidate.position.dtype != snapshot.position.dtype
        ):
            raise ValueError("candidate shape and dtype must match the original state")
        dynamic = model.inverse_mass > 0
        old_position = snapshot.position.astype(np.float64)
        old_velocity = snapshot.velocity.astype(np.float64)
        proposed_position = candidate.position.astype(np.float64)
        proposed_velocity = candidate.velocity.astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
            # Keep force/gravity increments separate. Subtracting two rounded
            # final states can hide finite motion at large coordinates/velocities.
            try:
                gravity = np.broadcast_to(
                    model.gravity.astype(np.float64), old_position[dynamic].shape
                )
                acceleration = _product(
                    model.inverse_mass[dynamic, None].astype(np.float64),
                    force[dynamic].astype(np.float64),
                )
                dv_gravity, dv_force = _product(float(h), gravity), _product(float(h), acceleration)
                drift = _product(float(h), old_velocity[dynamic])
                dx_gravity, dx_force = _product(float(h), dv_gravity), _product(float(h), dv_force)
                # Also require the actual reference states to be representable.
                _sum_terms(old_velocity[dynamic], dv_gravity, dv_force)
                _sum_terms(old_position[dynamic], drift, dx_gravity, dx_force)
                position_delta = proposed_position - old_position
                velocity_delta = proposed_velocity.copy()
                position_delta[dynamic] = _sum_terms(
                    proposed_position[dynamic],
                    -old_position[dynamic],
                    -drift,
                    -dx_gravity,
                    -dx_force,
                )
                velocity_delta[dynamic] = _sum_terms(
                    proposed_velocity[dynamic], -old_velocity[dynamic], -dv_gravity, -dv_force
                )
            except FloatingPointError as error:
                return VerificationReport(False, float("inf"), float("inf"), str(error))
            position_error = float(
                np.max(
                    np.hypot.reduce(position_delta, axis=1),
                    initial=0.0,
                )
            )
            velocity_error = float(
                np.max(
                    np.hypot.reduce(velocity_delta, axis=1),
                    initial=0.0,
                )
            )
            residual = max(
                float(normalized_error(position_error, self.reference_length)),
                float(normalized_error(velocity_error, self.reference_speed)),
            )
        if not (
            np.isfinite(position_error) and np.isfinite(velocity_error) and np.isfinite(residual)
        ):
            return VerificationReport(
                False, float("inf"), float("inf"), "non-finite independent reference or error"
            )
        accepted = residual <= self.tolerance
        return VerificationReport(
            accepted,
            residual,
            position_error,
            None if accepted else "free-particle equation residual exceeds tolerance",
        )
