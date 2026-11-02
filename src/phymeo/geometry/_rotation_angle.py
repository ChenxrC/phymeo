"""Rational enclosures for changing between angle and half-turn coordinates."""

from dataclasses import dataclass
from fractions import Fraction as F
from math import factorial

from ..core.capabilities import SolverFailure
from ._round_sweep import _GRID

# Rational bounds for pi. Only the parameter-to-time conversion uses them.
_PI_LO = F(3141592653589793238462643383279502884197, 10**39)
_PI_HI = _PI_LO + F(1, 10**39)


def _atan_small(x):
    """Alternating enclosure for |x| <= 1/2."""
    power = x
    total = F(0)
    for k in range(28):
        total += power / (2 * k + 1)
        power *= -x * x
    other = total + power / 57
    return min(total, other), max(total, other)


def _atan_positive(x):
    if x > 1:
        lo, hi = _atan_positive(1 / x)
        return _PI_LO / 2 - hi, _PI_HI / 2 - lo
    if x > F(1, 2):
        lo, hi = _atan_small((x - 1) / (x + 1))
        return _PI_LO / 4 + lo, _PI_HI / 4 + hi
    return _atan_small(x)


@dataclass(frozen=True)
class _AngleChart:
    half: int
    axis_length: tuple
    angle: F

    def time(self, index, upper):
        s = F(index, _GRID)
        pi = _PI_HI if upper else _PI_LO
        if s == 1:
            phase = pi
        else:
            x = self.axis_length[upper] * s / (1 - s)
            phase = 2 * _atan_positive(x)[upper]
        return (self.half * pi + phase) / self.angle


def sin_cos(angle):
    """Dyadic proposals for sin and cos, with one absolute error for both."""
    scale = 1 << 72
    half_pi = (_PI_LO + _PI_HI) / 4
    k = round(angle / half_pi)
    rest = angle - k * half_pi
    x = F(round(rest * scale), scale)
    uncertainty = abs(x - rest) + abs(k) * (_PI_HI - _PI_LO) / 4
    if abs(x) > 1 or uncertainty > F(1, 1 << 40):
        raise SolverFailure("rotation_angle_reduction_limit")

    def polynomial(odd):
        term = x if odd else F(1)
        total = term
        for j in range(1, 11):
            n = 2 * j + odd
            term *= -x * x / (n * (n - 1))
            total += term
        return F(round(total * scale), scale)

    s, c = polynomial(1), polynomial(0)
    s, c = ((s, c), (c, -s), (-s, -c), (-c, s))[k % 4]
    return s, c, uncertainty + F(1, scale) + F(1, factorial(21))
