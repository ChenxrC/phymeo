"""All-pair point/COM-sphere geometry, without contact response or broadphase."""

from dataclasses import dataclass

import numpy as np

from ..core._arrays import float_array
from ..core.capabilities import SolverFailure
from .sphere import SphereShapes


@dataclass(frozen=True, eq=False)
class ParticleSphereGeometry:
    """Owned readonly rows; outward unit normals and signed gaps use FP64.

    A negative gap means point penetration. Coincident points use +X and set
    degenerate=True; that convention does not select a unique physical normal.
    Evaluation returns every pair in particle-major order, including separation,
    so iterative solvers can retain multiplier identity while contacts release.
    """

    particle_indices: np.ndarray
    body_indices: np.ndarray
    normal: np.ndarray
    gap: np.ndarray
    degenerate: np.ndarray

    def __post_init__(self):
        for name in ("particle_indices", "body_indices"):
            value = np.asarray(getattr(self, name))
            if value.ndim != 1 or value.dtype.kind not in "iu":
                raise TypeError(f"{name} must be a rank-one integer array")
            if np.any(value < 0) or np.any(value > np.iinfo(np.int64).max):
                raise ValueError(f"{name} must contain non-negative int64 indices")
            owned = np.array(value, dtype=np.int64, order="C", copy=True)
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)
        for name, ndim, columns in (("normal", 2, 3), ("gap", 1, None)):
            owned = float_array(getattr(self, name), name, ndim=ndim, columns=columns).astype(
                np.float64
            )
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)
        flag = np.asarray(self.degenerate)
        if flag.ndim != 1 or flag.dtype != np.dtype(bool):
            raise TypeError("degenerate must be a rank-one boolean array")
        flag = np.array(flag, copy=True, order="C")
        flag.flags.writeable = False
        object.__setattr__(self, "degenerate", flag)
        count = len(self.particle_indices)
        if any(len(value) != count for value in (self.body_indices, self.normal, self.gap, flag)):
            raise ValueError("all geometry fields must have the same row count")
        if not np.allclose(np.hypot.reduce(self.normal, axis=1), 1.0, rtol=0, atol=1e-12):
            raise ValueError("normal rows must have unit length")


def particle_sphere_geometry(particle_position, rigid_position, spheres):
    """Evaluate every point/sphere pair with stable identity and FP64 arithmetic.

    Both position arrays and radii must have the same FP32/FP64 storage dtype.
    This small-scene reference costs O(Np*Nb) memory and work. It is discrete
    vertex geometry, not triangle collision, a swept query or a response solver.
    """
    if not isinstance(spheres, SphereShapes):
        raise TypeError("spheres must be SphereShapes")
    particles = float_array(particle_position, "particle_position", ndim=2, columns=3)
    bodies = float_array(rigid_position, "rigid_position", ndim=2, columns=3)
    if particles.dtype != bodies.dtype or particles.dtype != spheres.radii.dtype:
        raise ValueError("positions and sphere radii must have the same dtype")
    if len(bodies) != len(spheres.radii):
        raise ValueError("each rigid position must have one sphere radius")
    particle_indices = np.repeat(np.arange(len(particles), dtype=np.int64), len(bodies))
    body_indices = np.tile(np.arange(len(bodies), dtype=np.int64), len(particles))
    normal, gap, degenerate = _sphere_values(
        particles[particle_indices], bodies[body_indices], spheres.radii[body_indices]
    )
    return ParticleSphereGeometry(particle_indices, body_indices, normal, gap, degenerate)


def _sphere_values(particles, bodies, radii):
    """Paired geometric values for validated arrays, also used during live sweeps."""
    count = len(particles)
    normal = np.zeros((count, 3), dtype=np.float64)
    normal[:, 0] = 1.0
    if count == 0:
        return normal, np.empty(0), np.empty(0, dtype=bool)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            delta = particles.astype(np.float64) - bodies.astype(np.float64)
            scale = np.max(np.abs(delta), axis=1)
            distinct = scale > 0
            # Normalize before taking a norm: squaring coordinates can overflow
            # or underflow even when the distance and normal are representable.
            scaled = delta[distinct] / scale[distinct, None]
            unit_length = np.hypot.reduce(scaled, axis=1)
            normal[distinct] = scaled / unit_length[:, None]
            distance = np.zeros(count, dtype=np.float64)
            # Inexact subnormal distances can erase a real signed gap. Reject
            # that lost evidence; exact subnormal axis distances remain valid.
            with np.errstate(under="raise"):
                distance[distinct] = scale[distinct] * unit_length
                gap = distance - radii.astype(np.float64)
            if not (np.isfinite(normal).all() and np.isfinite(gap).all()):
                raise FloatingPointError("non-finite geometry")
    except (FloatingPointError, OverflowError) as error:
        raise SolverFailure(f"particle/sphere geometry is not representable: {error}") from error
    return normal, gap, np.logical_not(distinct)
