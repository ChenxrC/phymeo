"""Host FP64 local Coulomb disk solve at a fixed tangent basis and normal impulse."""

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
            raise SolverFailure(f"friction solve is not representable: {error}") from error

    return checked


def _norm(value):
    result = float(np.hypot.reduce(value))
    if not np.isfinite(result):
        raise SolverFailure("non-finite friction norm")
    return result


@dataclass(frozen=True, eq=False)
class FrictionImpulseResult:
    impulse: np.ndarray
    velocity_after: np.ndarray
    regime: str
    lagrange_multiplier: float | None
    iterations: int
    energy_change: float

    def __post_init__(self):
        for name in ("impulse", "velocity_after"):
            value = float_array(getattr(self, name), name, ndim=1)
            if value.shape != (2,) or value.dtype != np.float64:
                raise ValueError("friction result must have two FP64 components")
            value.flags.writeable = False
            object.__setattr__(self, name, value)


def _verified(k, v, impulse, regime, eta, iterations, radius, tolerance):
    delta = k @ impulse
    after = v + delta
    if not np.isfinite(after).all() or not np.isfinite(impulse).all():
        raise SolverFailure("non-finite friction candidate")
    if regime == "sticking":
        if _norm(after) > tolerance * max(_norm(v), 1e-12):
            raise SolverFailure("sticking velocity did not converge")
        if _norm(impulse) > radius:
            raise SolverFailure("sticking impulse outside static disk")
    elif eta is not None:
        dual = eta * impulse
        residual = _norm(after + dual)
        if residual > tolerance * max(_norm(v), _norm(delta), _norm(dual), 1e-12):
            raise SolverFailure("sliding KKT residual did not converge")
        if abs(_norm(impulse) - radius) > tolerance * max(radius, 1e-12):
            raise SolverFailure("sliding disk radius did not converge")
    work, quadratic = float(v @ impulse), float(0.5 * impulse @ delta)
    energy = work + quadratic
    if not np.isfinite(energy) or energy > tolerance * max(abs(work), abs(quadratic), 1e-24):
        raise SolverFailure("friction candidate increases kinetic energy")
    return FrictionImpulseResult(impulse, after, regime, eta, iterations, energy)


@_numerical
def solve_friction_disk(
    response,
    velocity,
    normal_impulse,
    *,
    mu_static,
    mu_dynamic,
    tolerance=1e-10,
    max_iterations=96,
):
    """Solve one fixed-normal tangent problem; does not advance a contact graph.

    response is m/s per N*s, velocity is m/s, normal_impulse and the returned
    tangent impulse are N*s. The returned quadratic energy change is joules.
    """
    k = float_array(response, "response", ndim=2)
    v = float_array(velocity, "velocity", ndim=1)
    if k.shape != (2, 2) or v.shape != (2,):
        raise ValueError("friction response/velocity shapes must be (2,2)/(2,)")
    if k.dtype != np.float64 or v.dtype != np.float64:
        raise TypeError("friction arrays must use float64 storage")
    normal = finite_real(normal_impulse, "normal_impulse")
    static, dynamic = finite_real(mu_static, "mu_static"), finite_real(mu_dynamic, "mu_dynamic")
    tolerance = finite_real(tolerance, "tolerance")
    if normal < 0 or dynamic < 0 or static < dynamic:
        raise ValueError("require normal_impulse >= 0 and 0 <= mu_dynamic <= mu_static")
    if tolerance <= 0 or tolerance > 1e-3:
        raise ValueError("tolerance must be in (0,1e-3]")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, Integral):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    scale = float(np.max(np.abs(k)))
    if scale == 0:
        raise SolverFailure("friction response is singular")
    if np.max(np.abs(k - k.T)) > 1e-12 * scale:
        raise ValueError("friction response must be symmetric")
    k = k * 0.5 + k.T * 0.5
    values, basis = np.linalg.eigh(k / scale)
    if values[-1] <= 0 or values[0] <= 1e-12 * values[-1]:
        raise SolverFailure("friction response is not reliably positive definite")
    if normal == 0:
        return _verified(k, v, np.zeros(2), "open", None, 0, 0, tolerance)
    static_radius, radius = static * normal, dynamic * normal
    if not np.isfinite(static_radius) or not np.isfinite(radius):
        raise SolverFailure("friction disk radius is not representable")
    rhs = (basis.T @ v) / scale
    stick = -(basis @ (rhs / values))
    if _norm(stick) <= static_radius:
        return _verified(k, v, stick, "sticking", 0.0, 0, static_radius, tolerance)
    if radius == 0:
        return _verified(k, v, np.zeros(2), "sliding", None, 0, 0, tolerance)
    lower, upper = 0.0, _norm(rhs) / radius
    if not np.isfinite(upper):
        raise SolverFailure("friction root interval is not representable")
    for iteration in range(1, max_iterations + 1):
        eta_scaled = lower * 0.5 + upper * 0.5
        impulse = -(basis @ (rhs / (values + eta_scaled)))
        length = _norm(impulse)
        if abs(length - radius) <= tolerance * max(radius, 1e-12):
            return _verified(
                k, v, impulse, "sliding", eta_scaled * scale, iteration, radius, tolerance
            )
        if length > radius:
            lower = eta_scaled
        else:
            upper = eta_scaled
    raise SolverFailure("friction disk solve did not converge within iteration budget")
