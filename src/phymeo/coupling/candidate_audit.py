"""Validation-only FP64 kinematics and stable kinetic-energy differences.

These routines inspect actual candidate states. They never produce a replacement
state, impulse response, or backend fallback answer.
"""

from dataclasses import dataclass
from math import fsum

import numpy as np

from ..core._arrays import float_array
from ..core.capabilities import SolverFailure
from ..dynamics.robot_dynamics import mass_matrix
from ..dynamics.robot_kinematics import point_jacobians
from ..dynamics.rotation import quaternion_matrix
from .ports import _numerical


@dataclass(frozen=True, eq=False)
class PointCandidateAudit:
    """World point velocities and full-domain kinetic change, counted once."""

    initial_velocities: np.ndarray
    candidate_velocities: np.ndarray
    energy_change: float

    def __post_init__(self):
        for name in ("initial_velocities", "candidate_velocities"):
            value = float_array(getattr(self, name), name, ndim=2, columns=3)
            if value.dtype != np.float64:
                raise TypeError("candidate audit requires float64 validation")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if self.initial_velocities.shape != self.candidate_velocities.shape:
            raise ValueError("candidate audit point counts must match")
        energy = float(self.energy_change)
        if not np.isfinite(energy):
            raise SolverFailure("candidate energy is not representable")
        object.__setattr__(self, "energy_change", energy)


def _linear_energy(model, before, after):
    dynamic = model.inverse_mass > 0
    a, b = before[dynamic].astype(np.float64), after[dynamic].astype(np.float64)
    # Divide before multiplication; do not subtract two large squared energies.
    momentum_delta = (b - a) / model.inverse_mass[dynamic, None]
    return fsum((momentum_delta * (b * 0.5 + a * 0.5)).flat)


@_numerical
def particle_audit(model, before, after, indices):
    return PointCandidateAudit(
        before.velocity[indices].astype(np.float64),
        after.velocity[indices].astype(np.float64),
        _linear_energy(model, before.velocity, after.velocity),
    )


@_numerical
def robot_audit(model, before, after, indices, local):
    jacobian = point_jacobians(model, before.q, indices, local)[0]
    mass = mass_matrix(model, before.q)
    energy = float((after.v - before.v) @ mass @ (after.v * 0.5 + before.v * 0.5))
    return PointCandidateAudit(
        np.einsum("nki,i->nk", jacobian, before.v),
        np.einsum("nki,i->nk", jacobian, after.v),
        energy,
    )


@_numerical
def rigid_audit(model, before, after, indices, local):
    rotation = quaternion_matrix(before.orientation)
    arms = np.einsum("nij,nj->ni", rotation[indices], local.astype(np.float64))
    a = before.velocity[indices].astype(np.float64) + np.cross(
        before.angular_velocity[indices], arms
    )
    b = after.velocity[indices].astype(np.float64) + np.cross(after.angular_velocity[indices], arms)
    dynamic = model.inverse_mass > 0
    body_a = np.einsum("nji,nj->ni", rotation[dynamic], before.angular_velocity[dynamic])
    body_b = np.einsum("nji,nj->ni", rotation[dynamic], after.angular_velocity[dynamic])
    angular = fsum(
        ((body_b - body_a) * model.inertia_diagonal[dynamic] * (body_b * 0.5 + body_a * 0.5)).flat
    )
    return PointCandidateAudit(
        a, b, fsum([_linear_energy(model, before.velocity, after.velocity), angular])
    )
