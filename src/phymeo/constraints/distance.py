"""Distance constraint data and a serial XPBD projection sweep."""

from dataclasses import dataclass

import numpy as np

from ..core._arrays import float_array


@dataclass(frozen=True, eq=False)
class DistanceConstraints:
    """Owned edge indices (m,2), rest lengths [m], and compliance [m/N]."""

    indices: np.ndarray
    rest_length: np.ndarray
    compliance: np.ndarray

    def __post_init__(self):
        indices = np.asarray(self.indices)
        if indices.ndim != 2 or indices.shape[1] != 2:
            raise ValueError("indices must have shape (m, 2)")
        if indices.dtype.kind not in "iu":
            raise TypeError("indices must be integers")
        if np.any(indices < 0) or np.any(indices > np.iinfo(np.int64).max):
            raise ValueError("indices must be non-negative int64 values")
        if np.any(indices[:, 0] == indices[:, 1]):
            raise ValueError("an edge must connect two different particles")
        indices = np.array(indices, dtype=np.int64, order="C", copy=True)
        rest = float_array(self.rest_length, "rest_length", ndim=1)
        compliance = float_array(self.compliance, "compliance", ndim=1)
        if rest.shape != (len(indices),) or compliance.shape != rest.shape:
            raise ValueError("rest_length and compliance must contain one value per edge")
        if rest.dtype != compliance.dtype:
            raise ValueError("rest_length and compliance must have the same dtype")
        if np.any(rest < 0) or np.any(compliance < 0):
            raise ValueError("rest_length and compliance must be non-negative")
        for name, value in [
            ("indices", indices),
            ("rest_length", rest),
            ("compliance", compliance),
        ]:
            value.flags.writeable = False
            object.__setattr__(self, name, value)


def project_distances(position, inverse_mass, constraints, alpha, multipliers):
    """One deterministic Gauss-Seidel sweep; multipliers persist within this step."""
    for k, (i, j) in enumerate(constraints.indices):
        delta = position[i] - position[j]
        length = float(np.hypot.reduce(delta))
        # Coincident endpoints have no unique gradient. Choose the first world axis.
        direction = (
            delta / length if length > 0 else np.array([1.0, 0.0, 0.0], dtype=position.dtype)
        )
        wi, wj = float(inverse_mass[i]), float(inverse_mass[j])
        denominator = wi + wj + alpha[k]
        if denominator == 0:
            continue  # A fixed, inconsistent hard edge is rejected by the residual check.
        c = length - float(constraints.rest_length[k])
        dlambda = (-c - alpha[k] * multipliers[k]) / denominator
        multipliers[k] += dlambda
        if wi > 0:
            position[i] += wi * dlambda * direction
        if wj > 0:
            position[j] -= wj * dlambda * direction


def distance_residual(position, constraints, alpha, multipliers, reference_length):
    """Return dimensionless constitutive residual and geometric error in metres."""
    delta = position[constraints.indices[:, 0]] - position[constraints.indices[:, 1]]
    c = np.hypot.reduce(delta, axis=1) - constraints.rest_length
    # A finite Python scale may exceed FP32 range; keep normalization in FP64.
    scale = np.maximum(constraints.rest_length.astype(np.float64), reference_length)
    residual = np.abs(c + alpha * multipliers) / scale
    return float(np.max(residual, initial=0.0)), float(np.max(np.abs(c), initial=0.0))
