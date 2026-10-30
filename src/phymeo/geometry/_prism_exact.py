"""Single-prism quadratic field; strict signs and rational outward bounds."""

from dataclasses import dataclass
from fractions import Fraction as F
from functools import total_ordering
from math import isqrt

import numpy as np

from ..solvers._contact_replay_audit import sqrt_bounds


def rational(value):
    if isinstance(value, F):
        return value
    if isinstance(value, (int, np.integer)):
        return F(int(value))
    return F(float(value))


@total_ordering
@dataclass(frozen=True, eq=False)
class Quadratic:
    a: F
    b: F = F(0)
    c: F = F(1)

    def __post_init__(self):
        a, b, c = map(rational, (self.a, self.b, self.c))
        if c <= 0:
            raise ValueError("quadratic radicand must be positive")
        if not b:
            c = F(1)
        elif c == 1:
            a, b = a + b, F(0)
        else:
            n, d = isqrt(c.numerator), isqrt(c.denominator)
            if n * n == c.numerator and d * d == c.denominator:
                a, b, c = a + b * F(n, d), F(0), F(1)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "c", c)

    @staticmethod
    def _canonical(a, b, c):
        """Internal Fraction coefficients in an already validated canonical field.

        Arithmetic cannot change the non-square radicand. Cancellation alone
        removes that field; public construction still validates every input.
        """
        value = object.__new__(Quadratic)
        object.__setattr__(value, "a", a)
        object.__setattr__(value, "b", b)
        object.__setattr__(value, "c", c if b else F(1))
        return value

    def pair(self, other):
        if not isinstance(other, Quadratic):
            other = Quadratic(other)
        if self.b and other.b and self.c != other.c:
            raise ValueError("different quadratic fields")
        return other, self.c if self.b else other.c

    def __add__(self, other):
        o, c = self.pair(other)
        b = o.b if not self.b else self.b if not o.b else self.b + o.b
        return self._canonical(self.a + o.a, b, c)

    __radd__ = __add__

    def __neg__(self):
        return self._canonical(-self.a, -self.b, self.c)

    def __sub__(self, other):
        o, c = self.pair(other)
        b = -o.b if not self.b else self.b if not o.b else self.b - o.b
        return self._canonical(self.a - o.a, b, c)

    def __rsub__(self, other):
        return -self + other

    @staticmethod
    def coerce(x):
        return x if isinstance(x, Quadratic) else Quadratic(x)

    def __mul__(self, other):
        o, c = self.pair(other)
        if not self.b:
            return self._canonical(self.a * o.a, self.a * o.b if o.b else F(0), c)
        if not o.b:
            return self._canonical(self.a * o.a, self.b * o.a, c)
        return self._canonical(self.a * o.a + self.b * o.b * c, self.a * o.b + self.b * o.a, c)

    __rmul__ = __mul__

    def __truediv__(self, other):
        o, c = self.pair(other)
        if not o.b:
            if not o.a:
                raise ZeroDivisionError("quadratic division by zero")
            return self._canonical(self.a / o.a, self.b / o.a if self.b else F(0), c)
        den = o.a * o.a - o.b * o.b * c
        if not den:
            raise ZeroDivisionError("quadratic division by zero")
        if not self.b:
            return self._canonical(self.a * o.a / den, -self.a * o.b / den, c)
        return self._canonical(
            (self.a * o.a - self.b * o.b * c) / den,
            (self.b * o.a - self.a * o.b) / den,
            c,
        )

    def __rtruediv__(self, other):
        return Quadratic.coerce(other) / self

    def __pow__(self, power):
        if power < 0:
            return (1 / self) ** (-power)
        result = Quadratic(1)
        base = self
        while power:
            if power % 2:
                result = result * base
            base = base * base
            power //= 2
        return result

    def sign(self):
        if not self.b:
            return (self.a > 0) - (self.a < 0)
        if not self.a or (self.a > 0) == (self.b > 0):
            return 1 if self.b > 0 else -1
        difference = self.a * self.a - self.b * self.b * self.c
        return ((difference > 0) - (difference < 0)) * (1 if self.a > 0 else -1)

    def __eq__(self, other):
        try:
            return (self - other).sign() == 0
        except (TypeError, ValueError):
            return False

    def __lt__(self, other):
        return (self - other).sign() < 0

    def __bool__(self):
        return self.sign() != 0

    def __abs__(self):
        return -self if self.sign() < 0 else self

    def __hash__(self):
        return hash((self.a, self.b, self.c))

    def bounds(self):
        if not self.b:
            return self.a, self.a
        lo, hi = sqrt_bounds(self.c)
        return (
            (self.a + self.b * lo, self.a + self.b * hi)
            if self.b >= 0
            else (self.a + self.b * hi, self.a + self.b * lo)
        )

    def __float__(self):
        lo, hi = self.bounds()
        return float((lo + hi) / 2)


def compare_gaps(a, aa, b, bb):
    """Compare a/sqrt(aa) with b/sqrt(bb), including negative gaps."""
    a, aa, b, bb = map(Quadratic.coerce, (a, aa, b, bb))
    sa, sb = a.sign(), b.sign()
    if sa != sb:
        return (sa > sb) - (sa < sb)
    if not sa:
        return 0
    return (a * a * bb - b * b * aa).sign() * sa


def normalized_bounds(gap, norm2):
    gap, norm2 = map(Quadratic.coerce, (gap, norm2))
    gl, gh = gap.bounds()
    nl, nh = norm2.bounds()
    if nl <= 0:
        raise ValueError("unresolved quadratic axis norm")
    a, b = sqrt_bounds(nl)[0], sqrt_bounds(nh)[1]
    values = (gl / a, gl / b, gh / a, gh / b)
    return min(values), max(values)


def outward_float(bounds):
    lo, hi = bounds
    a, b = float(lo), float(hi)
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("unrepresentable exact geometry")
    if rational(a) > lo:
        a = float(np.nextafter(a, -np.inf))
    if rational(b) < hi:
        b = float(np.nextafter(b, np.inf))
    return a, b


def dot(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), Quadratic(0))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


def add(a, b):
    return tuple(x + y for x, y in zip(a, b, strict=True))


def scale(a, k):
    return tuple(x * k for x in a)


def cross(a, b):
    return a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]


def vector(row):
    return tuple(Quadratic(rational(x)) for x in row)
