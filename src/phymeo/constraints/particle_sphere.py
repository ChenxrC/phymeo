"""Deterministic unilateral point/sphere sweeps on caller-owned candidates.

Positions share a live Gauss-Seidel workspace. Velocity sweeps instead keep
geometry frozen and retain normal impulses until that velocity solve finishes.
These primitives do not commit a simulation step: on SolverFailure discard the
candidate and all multiplier/load workspaces. The mixed solver owns rollback.
"""

from dataclasses import dataclass
from math import fsum

import numpy as np

from ..core._arrays import float_array
from ..core.capabilities import SolverFailure
from ..core.numerics import normalized_error
from ..core.validation import finite_real
from ..geometry.particle_sphere import (
    ParticleSphereGeometry,
    _sphere_values,
    particle_sphere_geometry,
)


@dataclass(frozen=True, eq=False)
class ContactTransfer:
    """Assigned opposite loads, including reactions on zero-inverse-mass objects.

    kind='position' has units kg*m; division by substep duration yields the
    effective impulse ALREADY included in reconstructed velocity. kind='velocity'
    has units kg*m/s and records only this sweep's additional velocity impulses.
    Sphere normal loads have zero torque about their COM.
    """

    particle: np.ndarray
    rigid: np.ndarray
    kind: str

    def __post_init__(self):
        if self.kind not in ("position", "velocity"):
            raise ValueError("contact transfer kind must be position or velocity")
        for name in ("particle", "rigid"):
            value = float_array(getattr(self, name), name, ndim=2, columns=3).astype(np.float64)
            value.flags.writeable = False
            object.__setattr__(self, name, value)


def _array(value, name, shape, dtype=None):
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"{name} must use float32 or float64")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} has inconsistent dtype")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _validate(p, b, wp, wb, multipliers, row_count, *, mutate):
    for value, name in ((p, "particle values"), (b, "rigid values")):
        if not isinstance(value, np.ndarray) or value.ndim != 2:
            raise TypeError(f"{name} must be a rank-two numpy array")
        _array(value, name, (len(value), 3), p.dtype)
    _array(wp, "particle inverse mass", (len(p),), p.dtype)
    _array(wb, "rigid inverse mass", (len(b),), p.dtype)
    _array(multipliers, "multipliers", (row_count,), np.dtype(np.float64))
    if any(np.any(value < 0) for value in (wp, wb, multipliers)):
        raise ValueError("inverse masses and multipliers must be non-negative")
    if mutate:
        writable = (p, b, multipliers)
        if any(not value.flags.writeable for value in writable):
            raise ValueError("candidate arrays and multipliers must be writable")
        arrays = (p, b, multipliers, wp, wb)
        for i in range(3):
            if any(np.shares_memory(arrays[i], arrays[j]) for j in range(i + 1, len(arrays))):
                raise ValueError("mutable contact workspaces must not alias other inputs")


def _scale(value, name, *, allow_zero=False):
    value = finite_real(value, name)
    if value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}")
    return value


def _weights(wp, wb, geometry):
    with np.errstate(over="raise", invalid="raise"):
        return wp[geometry.particle_indices].astype(np.float64) + wb[geometry.body_indices].astype(
            np.float64
        )


def _new_multiplier(old, value, weight):
    # Branching at zero avoids a needless, potentially overflowing division for
    # separated constraints with no multiplier to release.
    if value >= 0 and old == 0:
        return np.float64(0)
    return np.maximum(0, old - np.float64(value) / weight)


def _apply(p, b, wp, wb, i, j, load):
    if wp[i] > 0:
        p[i] = (p[i].astype(np.float64) + np.float64(wp[i]) * load).astype(p.dtype)
    if wb[j] > 0:
        b[j] = (b[j].astype(np.float64) - np.float64(wb[j]) * load).astype(b.dtype)


def _load_sums(indices, values, count):
    total = np.zeros((count, 3), dtype=np.float64)
    for index in np.unique(indices):
        rows = values[indices == index]
        total[index] = [fsum(rows[:, component]) for component in range(3)]
    return total


def _transfer(geometry, loads, np_count, nb_count, kind):
    return ContactTransfer(
        _load_sums(geometry.particle_indices, loads, np_count),
        _load_sums(geometry.body_indices, -loads, nb_count),
        kind,
    )


def _complementarity_errors(value, multipliers, weights):
    # abs(max(0, lambda-value/w)-lambda)*w = abs(min(value, w*lambda)).
    # This form does not subtract a small correction from a huge lambda.
    errors = np.maximum(-value, 0)
    positive = (value >= 0) & (multipliers > 0) & (weights > 0)
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        product = weights[positive] * multipliers[positive]
        product = np.where(product == 0, np.nextafter(0.0, 1.0), product)
        errors[positive] = np.minimum(value[positive], product)
    return errors


def _residual(errors, scale):
    result = float(np.max(normalized_error(errors, scale), initial=0))
    if not np.isfinite(result):
        raise SolverFailure("contact residual is not representable")
    return result


def project_particle_spheres(
    particle_position,
    rigid_position,
    particle_inverse_mass,
    rigid_inverse_mass,
    spheres,
    multipliers,
):
    """One live position sweep; retain multipliers across sweeps in a substep."""
    p, b, wp, wb = particle_position, rigid_position, particle_inverse_mass, rigid_inverse_mass
    _validate(p, b, wp, wb, multipliers, len(p) * len(b), mutate=True)
    geometry = particle_sphere_geometry(p, b, spheres)
    if any(np.shares_memory(spheres.radii, value) for value in (p, b, multipliers)):
        raise ValueError("sphere radii must not alias mutable contact workspaces")
    loads = np.zeros((len(multipliers), 3), dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            weights = _weights(wp, wb, geometry)
            for k, (i, j) in enumerate(zip(geometry.particle_indices, geometry.body_indices)):
                if weights[k] == 0:
                    continue
                normal, gap, _ = _sphere_values(
                    p[i : i + 1], b[j : j + 1], spheres.radii[j : j + 1]
                )
                new = _new_multiplier(multipliers[k], gap[0], weights[k])
                loads[k] = (new - multipliers[k]) * normal[0]
                _apply(p, b, wp, wb, i, j, loads[k])
                multipliers[k] = new
            return _transfer(geometry, loads, len(p), len(b), "position")
    except (FloatingPointError, OverflowError) as error:
        raise SolverFailure(f"contact position sweep is not representable: {error}") from error


def particle_sphere_residual(
    particle_position,
    rigid_position,
    particle_inverse_mass,
    rigid_inverse_mass,
    spheres,
    multipliers,
    *,
    reference_length=1.0,
):
    """Dimensionless projected complementarity residual, including fixed pairs."""
    p, b, wp, wb = particle_position, rigid_position, particle_inverse_mass, rigid_inverse_mass
    reference_length = _scale(reference_length, "reference_length")
    _validate(p, b, wp, wb, multipliers, len(p) * len(b), mutate=False)
    geometry = particle_sphere_geometry(p, b, spheres)
    try:
        return _residual(
            _complementarity_errors(geometry.gap, multipliers, _weights(wp, wb, geometry)),
            reference_length,
        )
    except (FloatingPointError, OverflowError) as error:
        raise SolverFailure(f"contact residual is not representable: {error}") from error


def _velocity_inputs(pv, bv, wp, wb, geometry, impulses, mutate, contact_tolerance):
    if not isinstance(geometry, ParticleSphereGeometry):
        raise TypeError("geometry must be ParticleSphereGeometry")
    tolerance = _scale(contact_tolerance, "contact_tolerance", allow_zero=True)
    _validate(pv, bv, wp, wb, impulses, len(geometry.gap), mutate=mutate)
    if np.any(geometry.particle_indices >= len(pv)) or np.any(geometry.body_indices >= len(bv)):
        raise ValueError("geometry indices exceed the velocity state")
    if mutate and any(
        np.shares_memory(value, data)
        for value in (pv, bv, impulses)
        for data in (geometry.normal, geometry.gap)
    ):
        raise ValueError("geometry must not alias mutable velocity workspaces")
    return geometry.gap <= tolerance


def _normal_velocity(pv, bv, normal):
    # Small terms between cancelling large products must survive; an inexact
    # subnormal product is insufficient evidence for a zero residual.
    with np.errstate(over="raise", invalid="raise", under="raise"):
        products = (pv.astype(np.float64) - bv.astype(np.float64)) * normal
    return fsum(products)


def project_particle_sphere_velocities(
    particle_velocity,
    rigid_velocity,
    particle_inverse_mass,
    rigid_inverse_mass,
    geometry,
    impulses,
    *,
    contact_tolerance=0.0,
):
    """One plastic normal-velocity sweep on frozen geometry; no tangential load.

    Zero inverse mass preserves the supplied velocity, allowing prescribed
    boundary velocities. A physical fixed-body integrator supplies zero velocity.
    """
    pv, bv, wp, wb = particle_velocity, rigid_velocity, particle_inverse_mass, rigid_inverse_mass
    active = _velocity_inputs(pv, bv, wp, wb, geometry, impulses, True, contact_tolerance)
    loads = np.zeros((len(impulses), 3), dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            weights = _weights(wp, wb, geometry)
            for k, (i, j) in enumerate(zip(geometry.particle_indices, geometry.body_indices)):
                if weights[k] == 0:
                    continue
                new = np.float64(0)
                if active[k]:
                    vn = _normal_velocity(pv[i], bv[j], geometry.normal[k])
                    new = _new_multiplier(impulses[k], vn, weights[k])
                loads[k] = (new - impulses[k]) * geometry.normal[k]
                _apply(pv, bv, wp, wb, i, j, loads[k])
                impulses[k] = new
            return _transfer(geometry, loads, len(pv), len(bv), "velocity")
    except (FloatingPointError, OverflowError) as error:
        raise SolverFailure(f"contact velocity sweep is not representable: {error}") from error


def particle_sphere_velocity_residual(
    particle_velocity,
    rigid_velocity,
    particle_inverse_mass,
    rigid_inverse_mass,
    geometry,
    impulses,
    *,
    contact_tolerance=0.0,
    reference_speed=1.0,
):
    """Dimensionless velocity complementarity error; length band is independent."""
    pv, bv, wp, wb = particle_velocity, rigid_velocity, particle_inverse_mass, rigid_inverse_mass
    speed = _scale(reference_speed, "reference_speed")
    active = _velocity_inputs(pv, bv, wp, wb, geometry, impulses, False, contact_tolerance)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            vn = np.zeros(len(impulses), dtype=np.float64)
            for k in np.flatnonzero(active):
                vn[k] = _normal_velocity(
                    pv[geometry.particle_indices[k]],
                    bv[geometry.body_indices[k]],
                    geometry.normal[k],
                )
            weights = _weights(wp, wb, geometry)
            errors = _complementarity_errors(vn, impulses, weights)
            inactive = ~active
            product = weights[inactive] * impulses[inactive]
            product = np.where(
                (weights[inactive] > 0) & (impulses[inactive] > 0) & (product == 0),
                np.nextafter(0.0, 1.0),
                product,
            )
            errors[inactive] = product
            return _residual(errors, speed)
    except (FloatingPointError, OverflowError) as error:
        raise SolverFailure(f"contact velocity residual is not representable: {error}") from error
