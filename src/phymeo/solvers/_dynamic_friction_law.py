"""Exact stored-world cone and dissipation audits; never generate native states."""

from fractions import Fraction

import numpy as np

from ..core.capabilities import SolverFailure
from ._dynamic_normal_geometry import normal_projection_bounds
from ._robot_replay_ledger import dot, rational, upper_float

ZERO = Fraction(0)


def invariants(normal, impulse, velocity):
    n2 = dot(normal, normal)
    if n2 <= 0:
        raise SolverFailure("invalid friction normal")
    ns, nu = dot(normal, impulse), dot(normal, velocity)
    t2 = dot(impulse, impulse) - ns * ns / n2
    u2 = dot(velocity, velocity) - nu * nu / n2
    if t2 < 0 or u2 < 0:
        raise SolverFailure("negative exact tangential squared norm")
    return t2, u2, dot(impulse, velocity) - ns * nu / n2


def audit(normal, impulse, velocity, capacity, kappa, speed):
    from ._contact_replay_audit import sqrt_bounds

    t2, u2, work = invariants(normal, impulse, velocity)
    tl, th = sqrt_bounds(t2)
    ul, uh = sqrt_bounds(u2)
    cl, ch = capacity
    if cl < 0 or ch < cl or normal_projection_bounds(normal, impulse)[0] < 0:
        raise SolverFailure("negative friction normal or capacity")
    k, v = rational(kappa), rational(speed)
    return (
        dict(
            cone=upper_float(k * max(th - cl, ZERO) / v),
            dissipation=upper_float(
                k * max(abs(work + c * u) for c in (cl, ch) for u in (ul, uh)) / (v * v)
            ),
            positive_work=upper_float(k * max(work, ZERO) / (v * v)),
        ),
        (cl, ch),
        work,
    )


def capacity(normal, world, mu):
    lo, hi = normal_projection_bounds(normal, world)
    return rational(mu) * lo, rational(mu) * hi


def combined_capacity(normal, held, terminal, mu):
    """Sum stored components exactly, before any normal projection or display."""
    return capacity(
        normal, [rational(p) + rational(e) for p, e in zip(held, terminal, strict=True)], mu
    )


def projected(vector, normal, radius):
    if radius < 0 or not np.isfinite(radius):
        raise SolverFailure("invalid friction projection capacity")
    value = vector - normal * (float(normal @ vector) / float(normal @ normal))
    length = np.hypot.reduce(value)
    if length > radius:
        value *= radius / length
    return value


def nonnegative_world(normal, world):
    """Closest one-component representation on the nonnegative-normal side.

    Projection rounding can give a pure tangent a negative exact dot product.
    This changes only the largest-normal component, then the caller audits its
    entire construction error in the original velocity budget. No state or
    latent impulse is modified, and the actual-normal sign gate stays strict.
    """
    result = np.array(world, dtype=np.float64, copy=True)
    if dot(normal, result) >= 0:
        return result
    axis = int(np.argmax(np.abs(normal)))
    nk = rational(normal[axis])
    if nk == 0:
        raise SolverFailure("zero normal cannot represent contact world")
    boundary = (
        -sum((rational(normal[k]) * rational(result[k]) for k in range(3) if k != axis), ZERO) / nk
    )
    try:
        rounded = float(boundary)
    except (OverflowError, ValueError) as error:
        raise SolverFailure("friction world boundary is unrepresentable") from error
    if not np.isfinite(rounded):
        raise SolverFailure("friction world boundary is unrepresentable")
    if (rational(rounded) - boundary) * nk < 0:
        with np.errstate(over="ignore", invalid="ignore"):
            rounded = np.nextafter(rounded, np.inf if nk > 0 else -np.inf)
    if not np.isfinite(rounded):
        raise SolverFailure("friction world boundary is unrepresentable")
    result[axis] = rounded
    if dot(normal, result) < 0:
        raise SolverFailure("friction world representation remains negative")
    return result
