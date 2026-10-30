"""Immutable observations of one discrete signed-dihedral projection.

These records describe actual FP64 endpoints and the gradient evaluated before
an update. They do not certify a continuous path, energy balance or torque.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..core.validation import finite_real


def _owned_float64(value, name, shape):
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must use float64 storage")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    array = np.array(array, dtype=np.float64, order="C", copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class BendingProjectionTrace:
    """Complete local observation; lambda and delta_lambda have units kg*m^2.

    ``smooth`` means both recorded endpoints satisfy the B1 angle/error domain
    and their angle difference is less than pi. It is not a segment certificate.
    """

    element_index: int
    node_indices: np.ndarray
    positions_before: np.ndarray
    positions_after: np.ndarray
    gradient: np.ndarray
    theta_before: float
    theta_after: float
    constraint_before: float
    constraint_after: float
    lambda_before: float
    lambda_after: float
    delta_lambda: float
    alpha: float
    denominator: float
    branch_status: str

    def __post_init__(self):
        if (
            isinstance(self.element_index, (bool, np.bool_))
            or not isinstance(self.element_index, Integral)
            or self.element_index < 0
        ):
            raise ValueError("element_index must be a non-negative integer")
        object.__setattr__(self, "element_index", int(self.element_index))
        nodes = np.asarray(self.node_indices)
        if nodes.dtype.kind not in "iu":
            raise TypeError("node_indices must be integers")
        if nodes.shape != (4,):
            raise ValueError("node_indices must have shape (4,)")
        if np.any(nodes < 0) or np.any(nodes > np.iinfo(np.int64).max):
            raise ValueError("node_indices must be non-negative int64 values")
        if len(set(map(int, nodes))) != 4:
            raise ValueError("node_indices must contain four distinct nodes")
        nodes = np.array(nodes, dtype=np.int64, order="C", copy=True)
        nodes.flags.writeable = False
        object.__setattr__(self, "node_indices", nodes)
        for name in ("positions_before", "positions_after", "gradient"):
            object.__setattr__(self, name, _owned_float64(getattr(self, name), name, (4, 3)))
        for name in (
            "theta_before", "theta_after", "constraint_before", "constraint_after",
            "lambda_before", "lambda_after", "delta_lambda", "alpha", "denominator",
        ):
            object.__setattr__(self, name, finite_real(getattr(self, name), name))
        if self.alpha <= 0 or self.denominator < self.alpha:
            raise ValueError("alpha must be positive and denominator must be at least alpha")
        if self.branch_status != "smooth":
            raise ValueError("branch_status must be smooth for a published trace")
        if any(
            abs(value) >= np.pi
            for value in (
                self.theta_before, self.theta_after, self.constraint_before, self.constraint_after
            )
        ) or abs(self.theta_after - self.theta_before) >= np.pi:
            raise ValueError("trace endpoints must lie on the supported smooth branch")
