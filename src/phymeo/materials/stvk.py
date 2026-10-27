"""Bounded 3D St. Venant–Kirchhoff elasticity with full nodal internal forces."""

from functools import wraps

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.tetrahedral import TetrahedralModel, tetra_array


def _finite(value):
    if not np.isfinite(value).all():
        raise FloatingPointError("non-finite tetrahedral result")
    return value


def _numerical(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
                return _finite(function(*args, **kwargs))
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
            raise SolverFailure(f"tetrahedral material failed: {error}") from error

    return checked


@_numerical
def deformation_gradient(model, position):
    if not isinstance(model, TetrahedralModel):
        raise TypeError("material requires TetrahedralModel")
    position = tetra_array(position, "position", ndim=2, columns=3)
    if position.shape != model.rest_position.shape:
        raise ValueError("position must match the tetrahedral node count")
    points = position[model.tetrahedra]
    ds = (points[:, 1:] - points[:, :1]).transpose(0, 2, 1)
    gradient = _finite(ds @ model.rest_inverse)
    determinant = _finite(np.linalg.det(gradient))
    singular = _finite(np.linalg.svd(gradient, compute_uv=False))
    if np.any(determinant <= 0.2) or np.any(singular < 0.6) or np.any(singular > 1.4):
        raise SolverFailure(
            "tetrahedron outside StVK domain: det(F)>.2 and singular values in [.6,1.4]"
        )
    return gradient


def _auxetic_parts(model, strain):
    # Negative lambda can cancel O(mu) hydrostatic terms. Compute bulk stiffness
    # directly from E/nu, and diagonal deviators from differences so equal strains
    # remain exactly zero even when mu is very large.
    bulk = model.young_modulus / (3 * (1 - 2 * model.poisson_ratio))
    if not np.isfinite(bulk) or bulk <= 0:
        raise SolverFailure("material bulk stiffness is not representable")
    diagonal = np.diagonal(strain, axis1=1, axis2=2)
    deviator = strain.copy()
    for i, j, k in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
        deviator[:, i, i] = (
            (diagonal[:, i] - diagonal[:, j]) + (diagonal[:, i] - diagonal[:, k])
        ) / 3
    return deviator, bulk


@_numerical
def elastic_energy(model, position):
    gradient = deformation_gradient(model, position)
    strain = (gradient.transpose(0, 2, 1) @ gradient - np.eye(3)) / 2
    trace = np.trace(strain, axis1=1, axis2=2)
    if model.lame_lambda < 0:
        deviator, bulk = _auxetic_parts(model, strain)
        density = _finite(
            model.lame_mu * np.einsum("nij,nij->n", deviator, deviator) + bulk / 2 * trace**2
        )
    else:
        density = _finite(
            model.lame_mu * np.einsum("nij,nij->n", strain, strain)
            + model.lame_lambda / 2 * trace**2
        )
    if np.any(density < 0):
        raise SolverFailure("material energy lost numerical positivity")
    return float(np.sum(model.volume * density))


@_numerical
def elastic_forces(model, position):
    gradient = deformation_gradient(model, position)
    strain = (gradient.transpose(0, 2, 1) @ gradient - np.eye(3)) / 2
    trace = np.trace(strain, axis1=1, axis2=2)
    if model.lame_lambda < 0:
        deviator, bulk = _auxetic_parts(model, strain)
        second_piola = _finite(
            2 * model.lame_mu * deviator + bulk * trace[:, None, None] * np.eye(3)
        )
    else:
        second_piola = _finite(
            2 * model.lame_mu * strain + model.lame_lambda * trace[:, None, None] * np.eye(3)
        )
    piola = _finite(gradient @ second_piola)
    h = _finite(-model.volume[:, None, None] * (piola @ model.rest_inverse.transpose(0, 2, 1)))
    local = np.empty((len(model.tetrahedra), 4, 3))
    local[:, 1:] = h.transpose(0, 2, 1)
    local[:, 0] = -h.sum(axis=2)
    force = np.zeros_like(model.rest_position)
    # Fixed-node forces are intentionally retained for support/load audits.
    np.add.at(force, model.tetrahedra.reshape(-1), local.reshape(-1, 3))
    return force
