"""Signed-dihedral elastic bending and an atomic host FP64 XPBD sweep.

This additive B1 primitive does not advance time or modify the existing distance
law. Native bending, composite material acceptance and continuous correction-path
geometry are separate integration work.
"""

from dataclasses import dataclass
from math import atan2, fsum

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.numerics import normalized_error
from ..core.validation import finite_real
from .bending_trace import BendingProjectionTrace, _owned_float64


@dataclass(frozen=True, eq=False)
class BendingConstraints:
    """Owned oriented hinges (a,b,c,d), rest radians and compliance [1/(N*m)].

    The faces are (a,b,c) and (b,a,d). Both endpoint order and opposite-vertex
    order are retained. Hinge identity ignores either order for duplicate checks.
    Zero compliance is outside this soft-bending operator's support.
    """

    indices: np.ndarray
    rest_angle: np.ndarray
    compliance: np.ndarray

    def __post_init__(self):
        indices = np.asarray(self.indices)
        if indices.ndim != 2 or indices.shape[1] != 4:
            raise ValueError("indices must have shape (m, 4)")
        if indices.dtype.kind not in "iu":
            raise TypeError("indices must be integers")
        if np.any(indices < 0) or np.any(indices > np.iinfo(np.int64).max):
            raise ValueError("indices must be non-negative int64 values")
        identities = set()
        for row in indices:
            a, b, c, d = map(int, row)
            if len({a, b, c, d}) != 4:
                raise ValueError("each bending hinge must contain four distinct indices")
            identity = (min(a, b), max(a, b), min(c, d), max(c, d))
            if identity in identities:
                raise ValueError("duplicate bending hinge")
            identities.add(identity)
        indices = np.array(indices, dtype=np.int64, order="C", copy=True)
        rest = _owned_float64(self.rest_angle, "rest_angle", (len(indices),))
        compliance = _owned_float64(self.compliance, "compliance", (len(indices),))
        if np.any(np.abs(rest) >= np.pi):
            raise ValueError("rest_angle must be strictly inside (-pi, pi)")
        if np.any(compliance <= 0):
            raise ValueError("compliance must be strictly positive")
        indices.flags.writeable = False
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "rest_angle", rest)
        object.__setattr__(self, "compliance", compliance)


def _array(value, name, shape):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must be a float64 ndarray")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _position(position, constraints):
    if not isinstance(constraints, BendingConstraints):
        raise TypeError("constraints must be BendingConstraints")
    if not isinstance(position, np.ndarray) or position.ndim != 2:
        raise TypeError("position must be a rank-two float64 ndarray")
    _array(position, "position", (len(position), 3))
    if np.any(constraints.indices >= len(position)):
        raise ValueError("bending index outside position array")


def _constitutive_arrays(constraints, alpha, multipliers):
    count = len(constraints.indices)
    _array(alpha, "alpha", (count,))
    _array(multipliers, "multipliers", (count,))
    if np.any(alpha <= 0):
        raise ValueError("alpha must be strictly positive")


def _writable_storage(value, name):
    if not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    # Accept disjoint C/F/strided views, but not self-overlapping publication
    # targets. A rank-two target has exactly three columns here.
    if len(value) > 1 and abs(value.strides[0]) < value.itemsize:
        raise ValueError(f"{name} must not have overlapping elements")
    if value.ndim == 2:
        for i in range(value.shape[1]):
            for j in range(i + 1, value.shape[1]):
                if np.shares_memory(value[:, i], value[:, j]):
                    raise ValueError(f"{name} must not have overlapping elements")


def _face(first, second):
    scale = float(max(np.max(np.abs(first)), np.max(np.abs(second))))
    if scale == 0:
        raise SolverFailure("degenerate bending face")
    u, v = first / scale, second / scale
    cross = np.cross(u, v)
    area = float(np.hypot.reduce(cross))
    if area <= 1e-10:
        raise SolverFailure("degenerate or ill-conditioned bending face")
    return cross / area, area, scale, u, v


def bending_angle_gradient(positions) -> tuple[float, np.ndarray]:
    """Return the exact stated oriented atan2 convention and analytic dtheta/dx.

    Geometry is evaluated after per-face scaling. The support threshold is the
    dimensionless scaled cross magnitude > 1e-10, as in the offset point port.
    A branch-cut angle, ill-conditioned face or unrepresentable evaluation raises
    SolverFailure. The owned returned gradient is read-only and has units 1/m.
    """
    x = _array(positions, "positions", (4, 3))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            a, b, c, d = x
            e, u, v = b - a, c - a, d - b
            left, left_area, left_scale, el, ul = _face(e, u)
            right, right_area, right_scale, minus_er, vr = _face(-e, v)
            edge_length = float(np.hypot.reduce(e))
            if edge_length == 0:
                raise SolverFailure("degenerate bending edge")
            axis = e / edge_length
            theta = atan2(float(np.dot(axis, np.cross(left, right))), float(np.dot(left, right)))
            if not -np.pi < theta < np.pi:
                raise SolverFailure("bending angle lies on the atan2 branch cut")
            # dtheta/dc and dtheta/dd have the SAME negative-normal sign for
            # nL=e cross(c-a), nR=(-e) cross(d-b). Scale before dividing by area.
            gc = -((edge_length / left_scale) / left_area) * (left / left_scale)
            gd = -((edge_length / right_scale) / right_area) * (right / right_scale)
            er = -minus_er
            tc = float(np.dot(ul, el) / np.dot(el, el))
            td = 1.0 + float(np.dot(vr, er) / np.dot(er, er))
            ga = (tc - 1.0) * gc + (td - 1.0) * gd
            gb = -tc * gc - td * gd
            gradient = np.array([ga, gb, gc, gd], dtype=np.float64)
            if not np.isfinite(gradient).all():
                raise SolverFailure("bending gradient is not representable")
    except (FloatingPointError, OverflowError) as error:
        raise SolverFailure(f"bending geometry is not representable: {error}") from error
    gradient.flags.writeable = False
    return theta, gradient


def _evaluate(nodes, rest):
    theta, gradient = bending_angle_gradient(nodes)
    constraint = theta - float(rest)
    if abs(constraint) >= np.pi:
        raise SolverFailure("bending angular error is outside the supported smooth branch")
    return theta, gradient, constraint


def bending_energy(position, constraints) -> float:
    """Elastic potential in Joules: sum((theta-rest)^2/(2*compliance))."""
    _position(position, constraints)
    terms = []
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            for row, rest, compliance in zip(
                constraints.indices, constraints.rest_angle, constraints.compliance
            ):
                _, _, constraint = _evaluate(position[row], rest)
                terms.append(float(0.5 * np.square(np.float64(constraint)) / compliance))
            energy = fsum(terms)
    except (FloatingPointError, OverflowError) as error:
        raise SolverFailure(f"bending energy is not representable: {error}") from error
    if not np.isfinite(energy):
        raise SolverFailure("bending energy is not representable")
    return energy


def project_bending(
    position, inverse_mass, constraints, alpha, multipliers
) -> tuple[BendingProjectionTrace, ...]:
    """One atomic, ordered Gauss-Seidel sweep over caller-owned FP64 state.

    alpha=compliance/h^2 is supplied by the integrator. A full private sweep,
    including trace construction and every endpoint check, precedes publication.
    Returned ``smooth`` traces certify only their recorded endpoint domains and
    absence of an explicit >=pi angle jump, not the intervening affine segment.
    """
    _position(position, constraints)
    _array(inverse_mass, "inverse_mass", (len(position),))
    _constitutive_arrays(constraints, alpha, multipliers)
    if np.any(inverse_mass < 0):
        raise ValueError("inverse_mass must be non-negative")
    _writable_storage(position, "position")
    _writable_storage(multipliers, "multipliers")
    inputs = (
        position,
        multipliers,
        inverse_mass,
        alpha,
        constraints.indices,
        constraints.rest_angle,
        constraints.compliance,
    )
    for i in range(2):
        if any(np.shares_memory(inputs[i], other) for other in inputs[i + 1 :]):
            raise ValueError("mutable bending arrays must not alias other inputs")
    trial_position, trial_lambda = position.copy(), multipliers.copy()
    traces = []
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            for k, row in enumerate(constraints.indices):
                before = trial_position[row].copy()
                theta_before, gradient, constraint_before = _evaluate(
                    before, constraints.rest_angle[k]
                )
                weights = inverse_mass[row]
                weighted_gradient = np.sqrt(weights)[:, None] * gradient
                denominator = float(np.sum(np.square(weighted_gradient)) + alpha[k])
                if not np.isfinite(denominator) or denominator <= 0:
                    raise SolverFailure("bending denominator is not representable")
                old_lambda = float(trial_lambda[k])
                delta_lambda = float((-constraint_before - alpha[k] * old_lambda) / denominator)
                new_lambda = old_lambda + delta_lambda
                after = before.copy()
                mobile = weights > 0
                after[mobile] += (weights[mobile] * delta_lambda)[:, None] * gradient[mobile]
                if not np.isfinite(after).all() or not np.isfinite(new_lambda):
                    raise SolverFailure("bending update is not representable")
                theta_after, _, constraint_after = _evaluate(after, constraints.rest_angle[k])
                if abs(theta_after - theta_before) >= np.pi:
                    raise SolverFailure("bending update has an explicit atan2 branch jump")
                traces.append(
                    BendingProjectionTrace(
                        element_index=k,
                        node_indices=row,
                        positions_before=before,
                        positions_after=after,
                        gradient=gradient,
                        theta_before=theta_before,
                        theta_after=theta_after,
                        constraint_before=constraint_before,
                        constraint_after=constraint_after,
                        lambda_before=old_lambda,
                        lambda_after=new_lambda,
                        delta_lambda=delta_lambda,
                        alpha=float(alpha[k]),
                        denominator=denominator,
                        branch_status="smooth",
                    )
                )
                trial_position[row] = after
                trial_lambda[k] = new_lambda
    except (FloatingPointError, OverflowError) as error:
        raise SolverFailure(f"bending sweep is not representable: {error}") from error
    result = tuple(traces)
    position[...] = trial_position
    multipliers[...] = trial_lambda
    return result


def bending_residual(
    position, constraints, alpha, multipliers, reference_angle=1.0
) -> tuple[float, float]:
    """Return max |C+alpha*lambda|/max(|rest|,reference_angle), max |C| [rad]."""
    _position(position, constraints)
    _constitutive_arrays(constraints, alpha, multipliers)
    reference = finite_real(reference_angle, "reference_angle")
    if reference <= 0:
        raise ValueError("reference_angle must be positive")
    residual, error = 0.0, 0.0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            for k, row in enumerate(constraints.indices):
                _, _, constraint = _evaluate(position[row], constraints.rest_angle[k])
                constitutive = abs(constraint + alpha[k] * multipliers[k])
                scale = max(abs(float(constraints.rest_angle[k])), reference)
                normalized = float(normalized_error(constitutive, scale))
                if not np.isfinite(normalized):
                    raise SolverFailure("bending residual is not representable")
                residual = max(residual, normalized)
                error = max(error, abs(constraint))
    except (FloatingPointError, OverflowError) as failure:
        raise SolverFailure(f"bending residual is not representable: {failure}") from failure
    return residual, error
