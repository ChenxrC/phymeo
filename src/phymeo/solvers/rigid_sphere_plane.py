"""Discrete, frictionless, zero-restitution contact for COM-centered spheres."""

from dataclasses import dataclass, replace

import numpy as np

from ..core.result import StepReport, StepResult
from ..core.validation import finite_real, validate_rigid_step
from ..geometry.plane import Plane
from ..geometry.sphere import SpherePlaneSignature, SphereShapes
from .rigid_semi_implicit import RigidSemiImplicitSolver


@dataclass(frozen=True, eq=False)
class RigidSpherePlaneSolver:
    """Free rigid prediction followed by sphere/plane stabilization and normal impulse.

    A normal impulse through the COM changes no angular velocity. Position
    stabilization is separate from velocity: it must not create a launch speed.
    This is discrete end-step contact, not continuous collision detection.
    """

    shapes: SphereShapes
    plane: Plane = Plane()
    tolerance: float = 1e-6
    reference_length: float = 1.0

    def __post_init__(self):
        if not isinstance(self.shapes, SphereShapes):
            raise TypeError("shapes must be SphereShapes")
        if not isinstance(self.plane, Plane):
            raise TypeError("plane must be Plane")
        for name, value in (
            ("tolerance", self.tolerance),
            ("reference_length", self.reference_length),
        ):
            object.__setattr__(self, name, finite_real(value, name))
        if self.tolerance < 0 or self.reference_length <= 0:
            raise ValueError("tolerance must be non-negative and reference_length positive")

    @property
    def capabilities(self):
        free = RigidSemiImplicitSolver.capabilities
        return replace(free, features=free.features | {"sphere_plane_contact"})

    def problem_signature(self):
        return SpherePlaneSignature(
            plane=(self.plane.height, "sphere-y-up-frictionless-e0-v1"),
            radii=tuple(float(radius) for radius in self.shapes.radii),
        )

    def step(self, model, state_in, inputs, dt) -> StepResult:
        snapshot, owned_inputs, h = validate_rigid_step(model, state_in, inputs, dt)
        radii, dtype = self.shapes.radii, snapshot.position.dtype
        if len(radii) != model.body_count:
            raise ValueError("sphere radii must match the body count")
        if radii.dtype != dtype:
            raise ValueError("sphere radii and state must have the same dtype")
        radii64 = radii.astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
            floor = (self.plane.height + radii64).astype(dtype)
            # Checking (height_at_center - plane_height) - radius also detects a
            # tiny radius lost by the initial addition, including FP64 rounding.
            rounded_gap = (floor.astype(np.float64) - self.plane.height) - radii64
            np.nextafter(floor, dtype.type(np.inf), out=floor, where=rounded_gap < 0)
            rounded_gap = (floor.astype(np.float64) - self.plane.height) - radii64
            if (
                not np.isfinite(floor).all()
                or not np.isfinite(rounded_gap).all()
                or np.any(rounded_gap < 0)
            ):
                return self._failure(snapshot, "sphere contact height is not representable")
            predicted = RigidSemiImplicitSolver().step(model, snapshot, owned_inputs, h)
            if not predicted.report.accepted:
                return predicted
            candidate = predicted.state
            dynamic = model.inverse_mass > 0
            candidate.position[dynamic, 1] = np.maximum(
                candidate.position[dynamic, 1], floor[dynamic]
            )
            active = dynamic & (candidate.position[:, 1] <= floor)
            candidate.velocity[active, 1] = np.maximum(candidate.velocity[active, 1], 0.0)
            gap = (candidate.position[:, 1].astype(np.float64) - self.plane.height) - radii64
            if not np.isfinite(gap).all():
                return self._failure(snapshot, "non-finite sphere contact gap")
            penetration = float(np.max(np.maximum(-gap, 0.0), initial=0.0))
            residual = penetration / self.reference_length
            if not np.isfinite(residual):
                return self._failure(snapshot, "non-finite sphere contact residual")
        accepted = residual <= self.tolerance
        return StepResult(
            candidate,
            StepReport(
                accepted=accepted,
                iterations=1,
                max_residual=residual,
                max_constraint_error=penetration,
                reason=None if accepted else "contact penetration exceeds tolerance",
            ),
        )

    @staticmethod
    def _failure(snapshot, reason):
        return StepResult(
            snapshot.copy(),
            StepReport(
                accepted=False,
                max_residual=float("inf"),
                max_constraint_error=float("inf"),
                reason=reason,
            ),
        )
