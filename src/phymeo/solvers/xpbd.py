"""Independent serial XPBD reference implementation for particles and edges.

Based on the mathematical formulation in Macklin et al., XPBD (2016).
This baseline implements neither friction nor continuous collision detection.
"""

from dataclasses import dataclass

import numpy as np

from ..constraints.distance import DistanceConstraints, distance_residual, project_distances
from ..core.capabilities import ProblemSignature, SolverCapabilities
from ..core.model import ParticleModel
from ..core.result import StepReport, StepResult
from ..core.state import ParticleState, StepInputs
from ..core.validation import finite_real, validate_step
from ..dynamics.integrate import predict
from ..geometry.plane import Plane


@dataclass(frozen=True, eq=False)
class XPBDSolver:
    """Distance projection with per-step multipliers and optional y-plane contact.

    A residual is normalized by max(rest length, reference_length), with plane
    penetration normalized by reference_length. The default reference length is 1 m.
    A finite unconverged candidate is returned with accepted=False for inspection.
    """

    constraints: DistanceConstraints | None = None
    iterations: int = 20
    tolerance: float = 1e-6
    plane: Plane | None = None
    reference_length: float = 1.0

    def __post_init__(self):
        if isinstance(self.iterations, (bool, np.bool_)) or not isinstance(
            self.iterations, (int, np.integer)
        ):
            raise ValueError("iterations must be a positive integer")
        if self.iterations < 1:
            raise ValueError("iterations must be a positive integer")
        for name, value in [
            ("tolerance", self.tolerance),
            ("reference_length", self.reference_length),
        ]:
            object.__setattr__(self, name, finite_real(value, name))
        if self.tolerance < 0 or self.reference_length <= 0:
            raise ValueError("tolerance must be non-negative and reference_length positive")
        if self.constraints is not None and not isinstance(self.constraints, DistanceConstraints):
            raise TypeError("constraints must be DistanceConstraints or None")
        if self.plane is not None and not isinstance(self.plane, Plane):
            raise TypeError("plane must be Plane or None")

    @property
    def capabilities(self):
        features = {"particles", "free_dynamics"}
        if self.constraints is not None and len(self.constraints.indices):
            features.add("distance")
        if self.plane is not None:
            features.add("plane_contact")
        return SolverCapabilities(
            backend="numpy", dtypes=frozenset(["float32", "float64"]), features=frozenset(features)
        )

    def problem_signature(self) -> ProblemSignature:
        distance = ()
        if self.constraints is not None:
            distance = tuple(
                (int(i), int(j), float(rest), float(compliance))
                for (i, j), rest, compliance in zip(
                    self.constraints.indices,
                    self.constraints.rest_length,
                    self.constraints.compliance,
                )
            )
        plane = None if self.plane is None else (self.plane.height, "y-up-frictionless-e0-v1")
        return ProblemSignature(distance=distance, plane=plane)

    def step(
        self, model: ParticleModel, state_in: ParticleState, inputs: StepInputs, dt: float
    ) -> StepResult:
        candidate, force, h = validate_step(model, state_in, inputs, dt)
        constraints = self.constraints
        if constraints is not None:
            if np.any(constraints.indices >= model.particle_count):
                raise ValueError("constraint index exceeds the model particle count")
            if constraints.rest_length.dtype != candidate.position.dtype:
                raise ValueError("constraints and state must have the same dtype")
        if not predict(model, candidate, force, h):
            return self._numerical_failure(state_in, 0)
        dynamic = model.inverse_mass > 0
        alpha = np.zeros(0 if constraints is None else len(constraints.indices), dtype=np.float64)
        multipliers = np.zeros_like(alpha)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            if constraints is not None:
                # Successive divisions avoid h*h overflowing when the final ratio is finite.
                np.divide(
                    constraints.compliance,
                    np.float64(h),
                    out=alpha,
                    where=constraints.compliance > 0,
                )
                np.divide(alpha, np.float64(h), out=alpha)
            if not np.isfinite(alpha).all():
                return self._numerical_failure(state_in, 0)
            residual, geometric = 0.0, 0.0
            for iteration in range(1, self.iterations + 1):
                if constraints is not None:
                    project_distances(
                        candidate.position, model.inverse_mass, constraints, alpha, multipliers
                    )
                if self.plane is not None:
                    candidate.position[dynamic, 1] = np.maximum(
                        candidate.position[dynamic, 1], self.plane.height
                    )
                residual, geometric = self._residual(candidate.position, alpha, multipliers)
                if not (
                    np.isfinite(candidate.position).all()
                    and np.isfinite(multipliers).all()
                    and np.isfinite(residual)
                    and np.isfinite(geometric)
                ):
                    return self._numerical_failure(state_in, iteration)
                if residual <= self.tolerance:
                    break
            candidate.velocity[dynamic] = (
                candidate.position[dynamic] - state_in.position[dynamic]
            ) / h
            candidate.velocity[~dynamic] = 0
            if self.plane is not None:
                active = dynamic & (candidate.position[:, 1] <= self.plane.height)
                candidate.velocity[active, 1] = np.maximum(candidate.velocity[active, 1], 0.0)
        if not np.isfinite(candidate.velocity).all():
            return self._numerical_failure(state_in, iteration)
        converged = residual <= self.tolerance
        return StepResult(
            candidate,
            StepReport(
                accepted=converged,
                iterations=iteration,
                max_residual=residual,
                max_constraint_error=geometric,
                reason=None if converged else "iteration budget exhausted",
            ),
        )

    def _residual(self, position, alpha, multipliers):
        residual, geometric = 0.0, 0.0
        if self.constraints is not None:
            residual, geometric = distance_residual(
                position, self.constraints, alpha, multipliers, self.reference_length
            )
        if self.plane is not None:
            penetration = float(
                np.max(np.maximum(self.plane.height - position[:, 1], 0.0), initial=0.0)
            )
            residual = max(residual, penetration / self.reference_length)
            geometric = max(geometric, penetration)
        return residual, geometric

    @staticmethod
    def _numerical_failure(state_in, iterations):
        return StepResult(
            state_in.copy(),
            StepReport(
                accepted=False,
                iterations=iterations,
                max_residual=float("inf"),
                max_constraint_error=float("inf"),
                reason="non-finite XPBD result",
            ),
        )
