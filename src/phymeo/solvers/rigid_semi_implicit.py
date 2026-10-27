"""First-order free rigid-body dynamics with a rotation-group pose update."""

import numpy as np

from ..core.capabilities import ProblemSignature, SolverCapabilities
from ..core.result import StepReport, StepResult
from ..core.rigid import RigidBodyInputs, RigidBodyModel, RigidBodyState
from ..core.validation import validate_rigid_step
from ..dynamics.integrate import predict
from ..dynamics.rotation import advance_world_orientation, quaternion_matrix


class RigidSemiImplicitSolver:
    """Update world velocities, then translation and quaternion pose.

    This first-order baseline includes gyroscopic acceleration but is not an
    energy/momentum-preserving integrator for arbitrary anisotropic inertia.
    """

    capabilities = SolverCapabilities(
        backend="numpy",
        dtypes=frozenset(["float32", "float64"]),
        features=frozenset(["rigid_bodies", "free_dynamics", "rotation"]),
        state_kind="rigid_bodies",
    )

    def problem_signature(self):
        return ProblemSignature(semantics="rigid-force-torque-gravity-v1")

    def step(
        self, model: RigidBodyModel, state_in: RigidBodyState, inputs: RigidBodyInputs, dt: float
    ) -> StepResult[RigidBodyState]:
        candidate, owned_inputs, h = validate_rigid_step(model, state_in, inputs, dt)
        if not predict(model, candidate, owned_inputs.force, h):
            return self._failure(state_in)
        dynamic = model.inverse_mass > 0
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            rotation = quaternion_matrix(candidate.orientation[dynamic])
            omega_world = candidate.angular_velocity[dynamic].astype(np.float64)
            omega_body = np.einsum("nji,nj->ni", rotation, omega_world)
            torque_body = np.einsum("nji,nj->ni", rotation, owned_inputs.torque[dynamic])
            inertia = model.inertia_diagonal[dynamic].astype(np.float64)
            acceleration_body = (torque_body - np.cross(omega_body, inertia * omega_body)) / inertia
            candidate.angular_velocity[dynamic] = omega_world + float(h) * np.einsum(
                "nij,nj->ni", rotation, acceleration_body
            )
            candidate.angular_velocity[~dynamic] = 0.0
            candidate.orientation[dynamic] = advance_world_orientation(
                candidate.orientation[dynamic], candidate.angular_velocity[dynamic], h
            )
        if not (
            np.isfinite(candidate.angular_velocity).all()
            and np.isfinite(candidate.orientation).all()
        ):
            return self._failure(state_in)
        return StepResult(candidate, StepReport(accepted=True))

    @staticmethod
    def _failure(state_in):
        return StepResult(
            RigidBodyState.copy(state_in),
            StepReport(accepted=False, reason="non-finite or underflowed rigid integration result"),
        )
