"""Exact stored-basis contact laws with rational square-root enclosures."""

from fractions import Fraction
from math import isqrt

from ._robot_replay_ledger import dot, rational, upper_float


def sqrt_bounds(value):
    """Enclose sqrt(nonnegative rational), with 256 significant binary bits."""
    if value < 0:
        raise ValueError("square root requires nonnegative value")
    if value == 0:
        return Fraction(0), Fraction(0)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    shift = 256 - exponent // 2
    if shift >= 0:
        root = isqrt((value.numerator << (2 * shift)) // value.denominator)
        unit = Fraction(1, 1 << shift)
    else:
        root = isqrt(value.numerator // (value.denominator << (-2 * shift)))
        unit = Fraction(1 << -shift)
    lower = root * unit
    return lower, lower if lower * lower == value else lower + unit


def contact_residual(relative, world_impulse, basis, kn, kt, mu, speed, tolerance):
    """Audit actual world p and full unrounded relative v, not solver iterates."""
    u = [
        sum((rational(basis[k, a]) * relative[k] for k in range(3)), Fraction(0)) for a in range(3)
    ]
    p = [dot(basis[:, a], world_impulse) for a in range(3)]
    kn, kt, mu, scale = map(rational, (kn, kt, mu, speed))
    projected = max(Fraction(0), p[0] - u[0] / kn)
    normal = max(abs(p[0] - projected) * kn, -p[0] * kn, -u[0], Fraction(0)) / scale
    radius = mu * max(p[0], Fraction(0))
    tangent2 = sum(v * v for v in p[1:])
    _, tangent_upper = sqrt_bounds(tangent2)
    friction = max(Fraction(0), tangent_upper - radius) * kt / scale
    speed2 = sum(v * v for v in u[1:])
    if speed2 > (rational(tolerance) * scale) ** 2:
        lower, upper = sqrt_bounds(speed2)
        mixed = p[1] * u[1] + p[2] * u[2]
        denominator = upper if mixed <= 0 else lower
        # Sign selects the conservative bound of the mixed term.
        error2 = max(Fraction(0), tangent2 + radius * radius + 2 * radius * mixed / denominator)
        _, error_upper = sqrt_bounds(error2)
        friction = max(friction, error_upper * kt / scale)
    return upper_float(normal), upper_float(friction)
