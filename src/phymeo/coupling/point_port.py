"""Instantaneous same-world-point particle/rigid impulse exchange."""

from dataclasses import dataclass
from math import fsum

import numpy as np

from ..core._arrays import float_array
from ..core.capabilities import SolverFailure
from ..core.coupling import ImpulseExchangeResult, ImpulseLedger
from ..core.model import ParticleModel
from ..core.rigid import RigidBodyModel, RigidBodyState
from ..core.state import ParticleState
from ..dynamics.rotation import quaternion_matrix


def _group_sum(indices, values, count):
    total = np.zeros((count, 3), dtype=np.float64)
    for index in np.unique(indices):
        rows = values[indices == index]
        total[index] = [fsum(rows[:, component]) for component in range(3)]
    return total


def _skew(vector):
    result = np.zeros((len(vector), 3, 3), dtype=np.float64)
    x, y, z = vector.T
    result[:, 0, 1], result[:, 0, 2] = -z, y
    result[:, 1, 0], result[:, 1, 2] = z, -x
    result[:, 2, 0], result[:, 2, 1] = -y, x
    return result


@dataclass(frozen=True, eq=False)
class ParticleRigidPorts:
    """Each row exchanges J/-J at that particle's current world position.

    These are impulse response ports, not a contact detector or constraint solver.
    A caller must integrate force over time before supplying an impulse.
    """

    particle_indices: np.ndarray
    body_indices: np.ndarray

    def __post_init__(self):
        for name in ("particle_indices", "body_indices"):
            indices = np.asarray(getattr(self, name))
            if indices.ndim != 1 or indices.dtype.kind not in "iu":
                raise TypeError(f"{name} must be a rank-one integer array")
            if np.any(indices < 0) or np.any(indices > np.iinfo(np.int64).max):
                raise ValueError(f"{name} must contain non-negative int64 indices")
            indices = np.array(indices, dtype=np.int64, order="C", copy=True)
            indices.flags.writeable = False
            object.__setattr__(self, name, indices)
        if self.particle_indices.shape != self.body_indices.shape:
            raise ValueError("particle and body port counts must match")

    def _snapshot(self, particle_model, particle_state, rigid_model, rigid_state):
        if not isinstance(particle_model, ParticleModel) or not isinstance(
            particle_state, ParticleState
        ):
            raise TypeError("particle ports require ParticleModel and ParticleState")
        if not isinstance(rigid_model, RigidBodyModel) or not isinstance(
            rigid_state, RigidBodyState
        ):
            raise TypeError("rigid ports require RigidBodyModel and RigidBodyState")
        particles = ParticleState(particle_state.position, particle_state.velocity)
        bodies = RigidBodyState(
            rigid_state.position,
            rigid_state.orientation,
            rigid_state.velocity,
            rigid_state.angular_velocity,
        )
        if (
            particle_model.particle_count != particles.particle_count
            or rigid_model.body_count != bodies.body_count
        ):
            raise ValueError("each model and state must have matching counts")
        dtype = particles.position.dtype
        if any(
            value != dtype
            for value in (
                bodies.position.dtype,
                particle_model.inverse_mass.dtype,
                rigid_model.inverse_mass.dtype,
            )
        ):
            raise ValueError("both state families and models must have the same dtype")
        if np.any(self.particle_indices >= particles.particle_count) or np.any(
            self.body_indices >= bodies.body_count
        ):
            raise ValueError("port index is outside its state family")
        return particles, bodies

    def effective_inverse_mass(self, particle_model, particle_state, rigid_model, rigid_state):
        """Per-port self response [1/kg]; shared objects also create cross terms."""
        particles, bodies = self._snapshot(particle_model, particle_state, rigid_model, rigid_state)
        count = len(self.particle_indices)
        if count == 0:
            return np.empty((0, 3, 3), dtype=np.float64)
        p, b = self.particle_indices, self.body_indices
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
                weights = particle_model.inverse_mass[p].astype(
                    np.float64
                ) + rigid_model.inverse_mass[b].astype(np.float64)
                dynamic = rigid_model.inverse_mass[b] > 0
                # A fixed rigid body has no rotational response, regardless of
                # point location. Do not evaluate an unused lever arm.
                arm = np.zeros((count, 3), dtype=np.float64)
                arm[dynamic] = particles.position[p[dynamic]].astype(np.float64) - bodies.position[
                    b[dynamic]
                ].astype(np.float64)
                rotation = quaternion_matrix(bodies.orientation[b[dynamic]])
                cross = _skew(arm)
                # -S R I^-1 R^T S = A A^T, A = S R I^-1/2.
                # Scaling R first avoids forming an overflowing inverse inertia
                # when a zero/small lever arm still has a finite response.
                scaled_rotation = (
                    rotation
                    / np.sqrt(rigid_model.inertia_diagonal[b[dynamic]].astype(np.float64))[
                        :, None, :
                    ]
                )
                factor = np.zeros((count, 3, 3), dtype=np.float64)
                factor[dynamic] = cross[dynamic] @ scaled_rotation
                effective = weights[:, None, None] * np.eye(3) + factor @ factor.swapaxes(1, 2)
                if not np.isfinite(effective).all():
                    raise FloatingPointError("non-finite effective inverse mass")
        except (FloatingPointError, OverflowError) as error:
            raise SolverFailure(
                f"point effective inverse mass is not representable: {error}"
            ) from error
        return effective

    def apply_impulses(self, particle_model, particle_state, rigid_model, rigid_state, impulses):
        particles, bodies = self._snapshot(particle_model, particle_state, rigid_model, rigid_state)
        impulses = float_array(impulses, "impulses", ndim=2, columns=3)
        if len(impulses) != len(self.particle_indices):
            raise ValueError("impulses must contain one row per port")
        if impulses.dtype != particles.position.dtype:
            raise ValueError("impulses and both states must have the same dtype")
        p, b = self.particle_indices, self.body_indices
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
                values = impulses.astype(np.float64)
                active = np.any(values != 0, axis=1)
                arm = particles.position[p[active]].astype(np.float64) - bodies.position[
                    b[active]
                ].astype(np.float64)
                angular = np.zeros_like(values)
                angular[active] = np.cross(arm, -values[active])
                particle_load = _group_sum(p, values, particles.particle_count)
                rigid_load = _group_sum(b, -values, bodies.body_count)
                rigid_angular = _group_sum(b, angular, bodies.body_count)
                if not all(
                    np.isfinite(value).all() for value in (particle_load, rigid_load, rigid_angular)
                ):
                    raise FloatingPointError("non-finite accumulated impulse")
                p_dynamic, b_dynamic = particle_model.inverse_mass > 0, rigid_model.inverse_mass > 0
                p_velocity = particles.velocity.astype(np.float64)
                b_velocity = bodies.velocity.astype(np.float64)
                b_angular = bodies.angular_velocity.astype(np.float64)
                p_velocity[p_dynamic] += (
                    particle_model.inverse_mass[p_dynamic, None].astype(np.float64)
                    * particle_load[p_dynamic]
                )
                b_velocity[b_dynamic] += (
                    rigid_model.inverse_mass[b_dynamic, None].astype(np.float64)
                    * rigid_load[b_dynamic]
                )
                changed = b_dynamic & np.any(rigid_angular != 0, axis=1)
                rotation = quaternion_matrix(bodies.orientation[changed])
                body_impulse = np.einsum("nji,nj->ni", rotation, rigid_angular[changed])
                body_delta = body_impulse / rigid_model.inertia_diagonal[changed].astype(np.float64)
                b_angular[changed] += np.einsum("nij,nj->ni", rotation, body_delta)
                p_velocity[~p_dynamic] = 0
                b_velocity[~b_dynamic] = b_angular[~b_dynamic] = 0
                dtype = particles.position.dtype
                p_velocity, b_velocity, b_angular = (
                    value.astype(dtype) for value in (p_velocity, b_velocity, b_angular)
                )
                if not all(
                    np.isfinite(value).all() for value in (p_velocity, b_velocity, b_angular)
                ):
                    raise FloatingPointError("non-finite impulse response")
        except (FloatingPointError, OverflowError) as error:
            return ImpulseExchangeResult(
                particles,
                bodies,
                False,
                reason=f"point impulse response is not representable: {error}",
            )
        return ImpulseExchangeResult(
            ParticleState(particles.position, p_velocity),
            RigidBodyState(bodies.position, bodies.orientation, b_velocity, b_angular),
            True,
            ImpulseLedger(particle_load, rigid_load, rigid_angular),
        )
