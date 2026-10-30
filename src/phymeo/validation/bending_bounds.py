"""Independent rational enclosures of the ideal signed dihedral and elastic law.

No production angle, gradient, energy or floating transcendental is an oracle.
Binary input coordinates are exact rationals; square roots and atan use outward
rational enclosures. Fractions remain internal, not runtime diagnostic payloads.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import isqrt

import numpy as np

from ..constraints.cloth_material import CompositeMaterialAuditSettings
from ..core.capabilities import SolverFailure
from ._material_path_math import q, interval_float


@dataclass(frozen=True)
class Interval:
    low: Fraction
    high: Fraction

    def __post_init__(self):
        object.__setattr__(self, "low", q(self.low))
        object.__setattr__(self, "high", q(self.high))
        if self.low > self.high:
            raise ValueError("inverted interval")

    @classmethod
    def point(cls, value):
        return cls(q(value), q(value))

    def __add__(self, other):
        other = interval(other)
        return Interval(self.low + other.low, self.high + other.high)

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.high, -self.low)

    def __sub__(self, other):
        return self + -interval(other)

    def __rsub__(self, other):
        return interval(other) + -self

    def __mul__(self, other):
        other = interval(other)
        values = (self.low * other.low, self.low * other.high,
                  self.high * other.low, self.high * other.high)
        return Interval(min(values), max(values))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = interval(other)
        if other.low <= 0 <= other.high:
            raise SolverFailure("independent interval divisor contains zero")
        return self * Interval(1 / other.high, 1 / other.low)

    def square(self):
        low = 0 if self.low <= 0 <= self.high else min(self.low**2, self.high**2)
        return Interval(low, max(self.low**2, self.high**2))

    @property
    def width(self):
        return self.high - self.low

    @property
    def magnitude(self):
        return max(abs(self.low), abs(self.high))

    def floats(self):
        return interval_float((self.low, self.high))

    def rounded(self, bits):
        """Outward absolute dyadic rounding, only to bound proof arithmetic size."""
        scale = 1 << bits
        a = self.low * scale
        b = self.high * scale
        return Interval(Fraction(a.numerator // a.denominator, scale),
                        Fraction(-(-b.numerator // b.denominator), scale))


def interval(value):
    return value if isinstance(value, Interval) else Interval.point(value)


def sqrt_interval(value, bits=256):
    value = interval(value)
    if value.low < 0:
        raise SolverFailure("negative independent square root domain")

    def bounds(x):
        if x == 0:
            return Fraction(0), Fraction(0)
        exponent = x.numerator.bit_length() - x.denominator.bit_length()
        shift = bits - exponent // 2
        if shift >= 0:
            root = isqrt((x.numerator << (2 * shift)) // x.denominator)
            unit = Fraction(1, 1 << shift)
        else:
            root = isqrt(x.numerator // (x.denominator << (-2 * shift)))
            unit = Fraction(1 << -shift)
        low = root * unit
        return low, low if low * low == x else low + unit

    return Interval(bounds(value.low)[0], bounds(value.high)[1])


def _series(x, terms, bits):
    """Alternating atan series plus a proved first-omitted-term remainder."""
    if x.low < 0 or x.high >= 1:
        raise SolverFailure("atan series argument outside [0,1)")
    square = x.square().rounded(bits)
    power, total = x, Interval.point(0)
    for k in range(terms):
        term = power / (2 * k + 1)
        total = (total + (term if k % 2 == 0 else -term)).rounded(bits)
        power = (power * square).rounded(bits)
    remainder = power.high / (2 * terms + 1)
    return total + (Interval(0, remainder) if terms % 2 == 0 else Interval(-remainder, 0))


@lru_cache(maxsize=32)
def pi_bounds(bits=256, terms=64):
    # Machin's identity, with independent rational series/remainder bounds.
    return 16 * _series(Interval.point(Fraction(1, 5)), terms, bits) - 4 * _series(
        Interval.point(Fraction(1, 239)), terms, bits)


def _atan_point(value, policy):
    value = q(value)
    if value < 0:
        return -_atan_point(-value, policy)
    if value == 0:
        return Interval.point(0)
    if value > 1:
        return pi_bounds(policy.sqrt_bits, policy.angle_terms) / 2 - _atan_point(1 / value, policy)
    reduced = Interval.point(value)
    for _ in range(2):
        reduced = (reduced / (1 + sqrt_interval(1 + reduced.square(), policy.sqrt_bits))).rounded(
            policy.sqrt_bits)
    return 4 * _series(reduced, policy.angle_terms, policy.sqrt_bits)


def atan_bounds(value, policy=None):
    policy = CompositeMaterialAuditSettings() if policy is None else policy
    value = interval(value)
    return Interval(_atan_point(value.low, policy).low, _atan_point(value.high, policy).high)


def atan2_bounds(sine, cosine, policy=None):
    """Enclose one continuous principal branch over a rectangle; unknown rejects."""
    policy = CompositeMaterialAuditSettings() if policy is None else policy
    s, a = interval(sine), interval(cosine)
    pi = pi_bounds(policy.sqrt_bits, policy.angle_terms)
    if a.low > 0:
        return atan_bounds(s / a, policy)
    if s.low > 0:
        return pi / 2 - atan_bounds(a / s, policy)
    if s.high < 0:
        return -pi / 2 - atan_bounds(a / s, policy)
    raise SolverFailure("angle rectangle contains branch cut or unresolved origin")


def dot(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), 0)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def subtract(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


@dataclass(frozen=True)
class BendingBounds:
    angle: Interval
    constraint: Interval
    gradient: tuple
    energy: Interval


def bending_bounds(nodes, rest_angle, compliance, *, policy=None):
    policy = CompositeMaterialAuditSettings() if policy is None else policy
    nodes = np.asarray(nodes)
    if nodes.dtype != np.float64 or nodes.shape != (4, 3) or not np.isfinite(nodes).all():
        raise ValueError("independent hinge bounds require finite FP64 (4,3)")
    rest, compliance = q(rest_angle), q(compliance)
    if compliance <= 0:
        raise ValueError("bending compliance must be positive")
    x = tuple(tuple(q(v) for v in row) for row in nodes)
    e, u, v = (subtract(x[k], x[0]) for k in (1, 2, 3))
    right_v = subtract(x[3], x[1])
    nl, nr = cross(e, u), cross(v, e)
    e2, l2, r2 = dot(e, e), dot(nl, nl), dot(nr, nr)
    threshold = q(1e-10)**2
    for normal2, first, second in ((l2, e, u), (r2, e, right_v)):
        scale = max(abs(z) for z in (*first, *second))
        if scale == 0 or normal2 <= threshold * scale**4:
            raise SolverFailure("independent bending face domain rejected")
    length = sqrt_interval(e2, policy.sqrt_bits)
    determinant = dot(e, cross(u, v))
    cosine = dot(e, u) * dot(e, v) - e2 * dot(u, v)
    theta = atan2_bounds(-length * determinant, cosine, policy)
    constraint = theta - rest
    pi = pi_bounds(policy.sqrt_bits, policy.angle_terms)
    if not (-pi.low < theta.low <= theta.high < pi.low and constraint.magnitude < pi.low):
        raise SolverFailure("independent strict angle/error branch not certified")
    gc = tuple(-length * z / l2 for z in nl)
    gd = tuple(-length * z / r2 for z in nr)
    tc, td = dot(u, e) / e2, dot(v, e) / e2
    ga = tuple((tc - 1) * c + (td - 1) * d for c, d in zip(gc, gd, strict=True))
    gb = tuple(-tc * c - td * d for c, d in zip(gc, gd, strict=True))
    return BendingBounds(theta, constraint, (ga, gb, gc, gd), constraint.square() / (2 * compliance))


def elastic_bounds(position, material, *, policy=None):
    policy = CompositeMaterialAuditSettings() if policy is None else policy
    total = Interval.point(0)
    if material.distance is not None:
        for ids, rest, compliance in zip(material.distance.indices, material.distance.rest_length,
                                          material.distance.compliance, strict=True):
            if compliance == 0:
                continue
            delta = subtract(tuple(map(q, position[ids[0]])), tuple(map(q, position[ids[1]])))
            c = sqrt_interval(dot(delta, delta), policy.sqrt_bits) - q(rest)
            total += c.square() / (2 * q(compliance))
    if material.bending is not None:
        for ids, rest, compliance in zip(material.bending.indices, material.bending.rest_angle,
                                          material.bending.compliance, strict=True):
            total += bending_bounds(position[ids], rest, compliance, policy=policy).energy
    return total
