"""Independent FP64 material acceptance; never supplies replacement device forces."""

import numpy as np

from ..core.capabilities import SolverFailure


def validate_device_material(model, position, force, energy):
    """Reject errors beyond frozen P6 per-component absolute/relative budgets.

    Derive the energy gradient directly from reference shape gradients and the
    volumetric/deviatoric energy, without calling the production material solver.
    All calculation here is validation only: the accepted output remains the
    original device result, including fixed-node forces.
    """
    reference_force = np.zeros_like(model.rest_position)
    reference_energy = 0.0
    bulk = model.young_modulus / (3 * (1 - 2 * model.poisson_ratio))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            for nodes, inverse, volume in zip(
                model.tetrahedra, model.rest_inverse, model.volume, strict=True
            ):
                deformed_edges = (position[nodes[1:]] - position[nodes[0]]).T
                gradient = deformed_edges @ inverse
                strain = (gradient.T @ gradient - np.eye(3)) / 2
                trace = float(np.trace(strain))
                deviator = strain.copy()
                # Differences preserve a small deviator beside a large bulk
                # component, including nearly -1 Poisson ratios.
                for axis in range(3):
                    deviator[axis, axis] = (
                        (strain[axis, axis] - strain[(axis + 1) % 3, (axis + 1) % 3])
                        + (strain[axis, axis] - strain[(axis + 2) % 3, (axis + 2) % 3])
                    ) / 3
                stress = 2 * model.lame_mu * deviator + bulk * trace * np.eye(3)
                reference_energy += float(volume) * (
                    model.lame_mu * float(np.sum(deviator * deviator)) + bulk * trace * trace / 2
                )
                # Each column is the force on a non-origin tetrahedron node;
                # the fourth gradient is minus their sum, enforcing balance.
                nodal = -float(volume) * gradient @ stress @ inverse.T
                for local in range(3):
                    reference_force[nodes[local + 1]] += nodal[:, local]
                reference_force[nodes[0]] -= np.sum(nodal, axis=1)
    except (FloatingPointError, OverflowError) as error:
        raise SolverFailure(f"independent device material validation failed: {error}") from error
    if (
        not np.isfinite(force).all()
        or not np.isfinite(energy)
        or not np.isfinite(reference_force).all()
        or not np.isfinite(reference_energy)
        or np.any(np.abs(force - reference_force) > 2e-4 + 5e-4 * np.abs(reference_force))
        or abs(energy - reference_energy) > 2e-4 + 5e-4 * abs(reference_energy)
    ):
        raise SolverFailure("independent device material force/energy error exceeds frozen budget")
