"""Joint distance/point-sphere reference integration with whole-frame rollback."""

from dataclasses import dataclass
from math import fsum

import numpy as np

from ..constraints.distance import DistanceConstraints, distance_residual, project_distances
from ..constraints.particle_sphere import (
    particle_sphere_residual,
    particle_sphere_velocity_residual,
    project_particle_sphere_velocities,
    project_particle_spheres,
)
from ..core.capabilities import SolverFailure
from ..core.coupled import (
    CoupledInputs,
    CoupledState,
    CoupledStepReport,
    CoupledStepResult,
    CoupledSubstepReport,
)
from ..core.coupling import ImpulseLedger
from ..core.model import ParticleModel
from ..core.rigid import RigidBodyModel
from ..core.validation import finite_real, validate_dt
from ..geometry.particle_sphere import particle_sphere_geometry
from ..geometry.sphere import SphereShapes
from .rigid_semi_implicit import RigidSemiImplicitSolver
from .semi_implicit import SemiImplicitSolver


def _impulse_ledger(transfers, divisor, particle_count, body_count):
    values = []
    for field, count in (("particle", particle_count), ("rigid", body_count)):
        total = np.zeros((count, 3), dtype=np.float64)
        for i in range(count):
            for axis in range(3):
                total[i, axis] = fsum(getattr(item, field)[i, axis] for item in transfers)
        values.append(total / divisor)
    return ImpulseLedger(values[0], values[1], np.zeros((body_count, 3)))


def _reconstruct(candidate, predicted, initial, inverse_mass, h):
    # An untouched predicted coordinate can round back to a large initial
    # coordinate. Preserve its predicted velocity rather than erase that motion.
    corrected = np.any(candidate.position != predicted.position, axis=1) & (inverse_mass > 0)
    candidate.velocity[corrected] = (
        (candidate.position[corrected].astype(np.float64) - initial.position[corrected]) / h
    ).astype(candidate.velocity.dtype)
    candidate.velocity[inverse_mass == 0] = 0


@dataclass(frozen=True, eq=False)
class CoupledXPBDSolver:
    particle_model: ParticleModel
    rigid_model: RigidBodyModel
    constraints: DistanceConstraints | None
    spheres: SphereShapes
    substeps: int = 4
    iterations: int = 200
    tolerance: float = 1e-7
    reference_length: float = 1.0
    reference_speed: float = 1.0
    record_substeps: bool = False

    def __post_init__(self):
        if not isinstance(self.particle_model, ParticleModel) or not isinstance(
            self.rigid_model, RigidBodyModel
        ):
            raise TypeError("coupled solver requires particle and rigid models")
        if not isinstance(self.spheres, SphereShapes):
            raise TypeError("spheres must be SphereShapes")
        dtype = self.particle_model.inverse_mass.dtype
        if dtype != self.rigid_model.inverse_mass.dtype or dtype != self.spheres.radii.dtype:
            raise ValueError("coupled models and geometry must have the same dtype")
        if len(self.spheres.radii) != self.rigid_model.body_count:
            raise ValueError("each rigid body must have one sphere radius")
        if self.constraints is not None:
            if not isinstance(self.constraints, DistanceConstraints):
                raise TypeError("constraints must be DistanceConstraints or None")
            if self.constraints.rest_length.dtype != dtype:
                raise ValueError("constraints and coupled models must have the same dtype")
            if np.any(self.constraints.indices >= self.particle_model.particle_count):
                raise ValueError("distance constraint exceeds particle count")
        for name in ("substeps", "iterations"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        for name in ("tolerance", "reference_length", "reference_speed"):
            value = finite_real(getattr(self, name), name)
            if value < 0 or (name != "tolerance" and value == 0):
                raise ValueError(f"{name} has an invalid sign or zero scale")
            object.__setattr__(self, name, value)
        if not isinstance(self.record_substeps, (bool, np.bool_)):
            raise TypeError("record_substeps must be a boolean")

    def step(self, state: CoupledState, inputs: CoupledInputs, dt: float) -> CoupledStepResult:
        if not isinstance(state, CoupledState) or not isinstance(inputs, CoupledInputs):
            raise TypeError("step requires CoupledState and CoupledInputs")
        original = CoupledState(state.particles, state.rigid)
        owned_inputs = CoupledInputs(inputs.particles, inputs.rigid)
        dtype = self.particle_model.inverse_mass.dtype
        if (
            original.particles.position.dtype != dtype
            or owned_inputs.particles.force.dtype != dtype
        ):
            raise ValueError("states, inputs and models must have the same dtype")
        if (
            original.particles.particle_count != self.particle_model.particle_count
            or original.rigid.body_count != self.rigid_model.body_count
        ):
            raise ValueError("state and model counts must match")
        if (
            owned_inputs.particles.force.shape != original.particles.position.shape
            or owned_inputs.rigid.force.shape != original.rigid.position.shape
        ):
            raise ValueError("force and state counts must match")
        h = validate_dt(float(validate_dt(dt, dtype)) / self.substeps, dtype)
        current = CoupledState(original.particles, original.rigid)
        reports, snapshots = [], []
        for _ in range(self.substeps):
            candidate, report = self._substep(current, owned_inputs, h)
            reports.append(report)
            if not report.accepted:
                return CoupledStepResult(
                    CoupledState(original.particles, original.rigid),
                    CoupledStepReport(False, len(reports) - 1, tuple(reports), report.reason),
                )
            current = candidate
            if self.record_substeps:
                snapshots.append(CoupledState(current.particles, current.rigid))
        return CoupledStepResult(
            current, CoupledStepReport(True, self.substeps, tuple(reports)), tuple(snapshots)
        )

    def _substep(self, initial, inputs, h):
        iteration = velocity_iteration = 0
        distance = contact = velocity = float("inf")
        geometric = penetration = 0.0

        def reject(reason, *, numerical=False):
            residuals = (float("inf"),) * 3 if numerical else (distance, contact, velocity)
            return None, CoupledSubstepReport(
                False,
                iteration,
                velocity_iteration,
                *residuals,
                geometric,
                penetration,
                reason=reason,
            )

        try:
            p = SemiImplicitSolver().step(
                self.particle_model, initial.particles, inputs.particles, h
            )
            b = RigidSemiImplicitSolver().step(self.rigid_model, initial.rigid, inputs.rigid, h)
            if not p.report.accepted or not b.report.accepted:
                return reject("non-finite coupled free prediction", numerical=True)
            candidate = CoupledState(p.state, b.state)
            wp, wb = self.particle_model.inverse_mass, self.rigid_model.inverse_mass
            constraints = self.constraints
            edge_count = 0 if constraints is None else len(constraints.indices)
            alpha, edge_lambda = np.zeros(edge_count), np.zeros(edge_count)
            contact_lambda = np.zeros(len(wp) * len(wb))
            position_transfers, velocity_transfers = [], []
            with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
                if constraints is not None:
                    np.divide(
                        constraints.compliance,
                        np.float64(h),
                        out=alpha,
                        where=constraints.compliance > 0,
                    )
                    np.divide(alpha, np.float64(h), out=alpha)
                for iteration in range(1, self.iterations + 1):
                    if constraints is not None:
                        project_distances(
                            candidate.particles.position, wp, constraints, alpha, edge_lambda
                        )
                    position_transfers.append(
                        project_particle_spheres(
                            candidate.particles.position,
                            candidate.rigid.position,
                            wp,
                            wb,
                            self.spheres,
                            contact_lambda,
                        )
                    )
                    distance, geometric = (
                        (0.0, 0.0)
                        if constraints is None
                        else distance_residual(
                            candidate.particles.position,
                            constraints,
                            alpha,
                            edge_lambda,
                            self.reference_length,
                        )
                    )
                    contact = particle_sphere_residual(
                        candidate.particles.position,
                        candidate.rigid.position,
                        wp,
                        wb,
                        self.spheres,
                        contact_lambda,
                        reference_length=self.reference_length,
                    )
                    if not np.isfinite([distance, geometric, contact]).all():
                        return reject("non-finite coupled position residual", numerical=True)
                    if max(distance, contact) <= self.tolerance:
                        break
                else:
                    return reject("coupled position iteration budget exhausted")
                _reconstruct(candidate.particles, p.state, initial.particles, wp, float(h))
                _reconstruct(candidate.rigid, b.state, initial.rigid, wb, float(h))
                geometry = particle_sphere_geometry(
                    candidate.particles.position, candidate.rigid.position, self.spheres
                )
                penetration = float(np.max(np.maximum(-geometry.gap, 0), initial=0))
                impulses = np.zeros_like(contact_lambda)
                band = np.float64(self.tolerance) * self.reference_length
                for velocity_iteration in range(1, self.iterations + 1):
                    velocity_transfers.append(
                        project_particle_sphere_velocities(
                            candidate.particles.velocity,
                            candidate.rigid.velocity,
                            wp,
                            wb,
                            geometry,
                            impulses,
                            contact_tolerance=band,
                        )
                    )
                    velocity = particle_sphere_velocity_residual(
                        candidate.particles.velocity,
                        candidate.rigid.velocity,
                        wp,
                        wb,
                        geometry,
                        impulses,
                        contact_tolerance=band,
                        reference_speed=self.reference_speed,
                    )
                    if velocity <= self.tolerance:
                        break
                else:
                    return reject("coupled velocity iteration budget exhausted")
                position_impulses = _impulse_ledger(position_transfers, float(h), len(wp), len(wb))
                velocity_impulses = _impulse_ledger(velocity_transfers, 1.0, len(wp), len(wb))
            return candidate, CoupledSubstepReport(
                True,
                iteration,
                velocity_iteration,
                distance,
                contact,
                velocity,
                geometric,
                penetration,
                position_impulses,
                velocity_impulses,
            )
        except (SolverFailure, FloatingPointError, OverflowError) as error:
            return reject(str(error), numerical=True)
