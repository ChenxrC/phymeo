"""Bounded actuator effort evaluated at an implicit free-motion prediction."""

from dataclasses import dataclass
from functools import wraps
from numbers import Integral

import numpy as np

from ..core._arrays import float_array
from ..core.capabilities import SolverFailure
from ..core.validation import finite_real


def _numerical(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
                return function(*args, **kwargs)
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
            raise SolverFailure(f"bounded control is not representable: {error}") from error

    return checked


def _array(value, name, shape):
    result = float_array(value, name, ndim=len(shape))
    if result.dtype != np.float64 or result.shape != shape:
        raise ValueError(f"{name} requires FP64 shape {shape}")
    return result


def _finite(value):
    if not np.isfinite(value).all():
        raise SolverFailure("non-finite bounded-control arithmetic")
    return value


@dataclass(frozen=True, eq=False)
class BoundedPDResult:
    """Effort is the command; predictions must never replace a backend state."""

    effort: np.ndarray
    predicted_velocity: np.ndarray
    predicted_configuration: np.ndarray
    iterations: int
    residual_effort: float
    equation_residual: float
    mass_asymmetry: float

    def __post_init__(self):
        for name in ("effort", "predicted_velocity", "predicted_configuration"):
            value = float_array(getattr(self, name), name, ndim=1)
            if value.dtype != np.float64:
                raise TypeError("control results require FP64")
            value.flags.writeable = False
            object.__setattr__(self, name, value)


@_numerical
def solve_bounded_pd(
    mass,
    bias,
    q,
    velocity,
    target,
    target_velocity,
    kp,
    kd,
    effort_limit,
    h,
    *,
    feedforward=None,
    effort_tolerance=1e-10,
    mass_tolerance=1e-12,
    max_iterations=512,
):
    """Solve M(u-v)=h(tau-b), tau=clip(kp(q*-q-hu)+kd(v*-u)+ff).

    This free-motion predictor chooses held effort. Contact corrections, target
    history, joint limits and physical state publication belong to the caller.
    """
    q = float_array(q, "q", ndim=1)
    count = len(q)
    if not count or q.dtype != np.float64:
        raise ValueError("q requires a nonempty FP64 vector")
    mass = _array(mass, "mass", (count, count))
    arrays = [
        _array(value, name, (count,))
        for name, value in (
            ("bias", bias),
            ("velocity", velocity),
            ("target", target),
            ("target_velocity", target_velocity),
            ("kp", kp),
            ("kd", kd),
            ("effort_limit", effort_limit),
        )
    ]
    bias, velocity, target, target_velocity, kp, kd, limit = arrays
    ff = np.zeros(count) if feedforward is None else _array(feedforward, "feedforward", (count,))
    if np.any(kp < 0) or np.any(kd < 0) or np.any(limit < 0):
        raise ValueError("gains and effort limits must be nonnegative")
    h = finite_real(h, "h")
    effort_tolerance = finite_real(effort_tolerance, "effort_tolerance")
    mass_tolerance = finite_real(mass_tolerance, "mass_tolerance")
    if h <= 0:
        raise ValueError("h must be positive")
    if not 0 < effort_tolerance <= 1e-3 or not 0 < mass_tolerance <= 1e-3:
        raise ValueError("control tolerances must be in (0,1e-3]")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, Integral):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    scale = float(np.max(np.abs(mass)))
    if scale == 0:
        raise SolverFailure("mass must be positive definite")
    asymmetry = float(np.max(np.abs(mass / scale - mass.T / scale)))
    if asymmetry > mass_tolerance:
        raise SolverFailure("mass asymmetry exceeds declared tolerance")
    symmetric = mass * 0.5 + mass.T * 0.5
    spectrum = np.linalg.eigvalsh(symmetric / scale)
    if spectrum[-1] <= 0 or spectrum[0] <= 1e-12 * spectrum[-1]:
        raise SolverFailure("mass must be positive definite and well-conditioned")
    mobility = _finite(np.linalg.solve(symmetric, np.eye(count)))
    free = _finite(velocity - h * (mobility @ bias))
    a = _finite(kp * (target - q) + kd * target_velocity + ff)
    d = _finite(h * kp + kd)
    active = np.flatnonzero(d > 0)
    fixed = np.flatnonzero(d == 0)
    effort = np.zeros(count)
    effort[fixed] = np.clip(a[fixed], -limit[fixed], limit[fixed])
    if len(active):
        matrix = _finite(np.diag(1 / d[active]) + h * mobility[np.ix_(active, active)])
        rhs = _finite(
            a[active] / d[active]
            - free[active]
            - h * mobility[np.ix_(active, fixed)] @ effort[fixed]
        )
        local = np.zeros(len(active))
    for iteration in range(1, max_iterations + 1):
        if len(active):
            for index in range(len(active)):
                change = (rhs[index] - matrix[index] @ local) / matrix[index, index]
                local[index] = np.clip(
                    local[index] + change, -limit[active[index]], limit[active[index]]
                )
            effort[active] = local
        u = _finite(free + h * (mobility @ effort))
        expected = _finite(np.clip(a - d * u, -limit, limit))
        residual = float(np.max(np.abs(effort - expected)))
        if residual > effort_tolerance:
            continue
        delta = _finite(u - velocity)
        force_step = _finite(h * (effort - bias))
        equation = _finite(mass @ delta - force_step)
        equation_scale = max(
            1.0,
            float(np.linalg.norm(mass, np.inf) * np.max(np.abs(delta))),
            float(np.max(np.abs(force_step))),
        )
        if not np.isfinite(equation_scale):
            raise SolverFailure("bounded-control equation scale is not representable")
        equation_residual = float(np.max(np.abs(equation)) / equation_scale)
        if equation_residual > mass_tolerance:
            raise SolverFailure("original mass equation disagrees with control prediction")
        return BoundedPDResult(
            effort,
            u,
            _finite(q + h * u),
            iteration if len(active) else 0,
            residual,
            equation_residual,
            asymmetry,
        )
    raise SolverFailure("bounded control did not converge within iteration budget")
