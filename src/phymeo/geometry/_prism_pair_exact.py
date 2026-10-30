"""Reduced, pair-local exact fields for two finite-thickness triangle normals.

No predicate uses float equality. A bounded sign computation may be unresolved;
that is a proof obligation, not zero. This module does not change legacy Q.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite, isqrt


@dataclass(frozen=True)
class PrismPairProofLimits:
    initial_sqrt_bits: int = 64
    max_sqrt_bits: int = 4096
    max_time_cells: int = 4096
    max_time_depth: int = 32

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.initial_sqrt_bits > self.max_sqrt_bits:
            raise ValueError("initial precision exceeds maximum precision")

    @property
    def token(self):
        return tuple(getattr(self, name) for name in self.__dataclass_fields__)


class SignUnresolved(ArithmeticError):
    """The exact value is nonzero, but its finite sign budget is exhausted."""

    def __init__(self, value, bits):
        self.value_token = value.token
        self.bits = bits
        super().__init__(f"nonzero sign unresolved after {bits} sqrt bits")


def rational(value):
    if isinstance(value, Fraction):
        return value
    if type(value) is int:
        return Fraction(value)
    if type(value) is float and isfinite(value):
        return Fraction.from_float(value)
    raise TypeError("expected a finite Python float, integer or Fraction")


def square_root(value):
    """Return a rational square root, or None; no integer factorization."""
    value = rational(value)
    if value < 0:
        return None
    p, q = isqrt(value.numerator), isqrt(value.denominator)
    return Fraction(p, q) if p * p == value.numerator and q * q == value.denominator else None


def sqrt_bounds(value, bits=64):
    """Exact rational outward enclosure with relative binary precision."""
    value = rational(value)
    if value < 0 or type(bits) is not int or bits <= 0:
        raise ValueError("nonnegative radicand and positive integer bits required")
    if not value:
        return Fraction(0), Fraction(0)
    n, d = value.numerator, value.denominator
    shift = bits - (n.bit_length() - d.bit_length()) // 2
    if shift >= 0:
        root = isqrt((n << (2 * shift)) // d)
        unit = Fraction(1, 1 << shift)
    else:
        root = isqrt(n // (d << (-2 * shift)))
        unit = Fraction(1 << -shift)
    lo = root * unit
    return lo, lo if lo * lo == value else lo + unit


@dataclass(frozen=True, init=False)
class PairField:
    """Ordered reduced basis (1,u,v,uv); input generator order is irrelevant.

    Context identity is the reduced rational radicand tuple. Rational square
    factors between the two supplied roots are removed before zero detection.
    Unrelated contexts are never implicitly converted.
    """

    roots: tuple
    limits: PrismPairProofLimits

    def __init__(self, ca, cb, limits=PrismPairProofLimits()):
        ca, cb = rational(ca), rational(cb)
        if ca <= 0 or cb <= 0:
            raise ValueError("normal squared areas must be positive")
        if not isinstance(limits, PrismPairProofLimits):
            raise TypeError("proof_limits must be PrismPairProofLimits")
        roots = sorted(set(x for x in (ca, cb) if square_root(x) is None))
        if len(roots) == 2 and square_root(roots[1] / roots[0]) is not None:
            roots.pop()
        object.__setattr__(self, "roots", tuple(roots))
        object.__setattr__(self, "limits", limits)

    def scalar(self, value):
        return PairNumber(self, (rational(value),) + (Fraction(0),) * ((1 << len(self.roots)) - 1))

    def element(self, *coefficients):
        if len(coefficients) != 1 << len(self.roots):
            raise ValueError("coefficient count does not match the reduced field")
        return PairNumber(self, tuple(rational(x) for x in coefficients))

    def sqrt(self, value):
        value = rational(value)
        if value < 0:
            raise ValueError("negative radicand")
        for mask in range(1 << len(self.roots)):
            basis_squared = Fraction(1)
            for i, root in enumerate(self.roots):
                if mask & (1 << i):
                    basis_squared *= root
            factor = square_root(value / basis_squared)
            if factor is not None:
                coefficients = [Fraction(0)] * (1 << len(self.roots))
                coefficients[mask] = factor
                return self.element(*coefficients)
        raise ValueError("radical is outside this field")


@dataclass(frozen=True, eq=False)
class PairNumber:
    field: PairField
    coefficients: tuple

    def __post_init__(self):
        if not isinstance(self.field, PairField):
            raise TypeError("PairField required")
        if len(self.coefficients) != 1 << len(self.field.roots):
            raise ValueError("coefficient count does not match field")
        object.__setattr__(self, "coefficients", tuple(rational(x) for x in self.coefficients))

    @property
    def token(self):
        def token(value):
            return value.numerator, value.denominator
        return tuple(map(token, self.field.roots)), tuple(map(token, self.coefficients))

    @property
    def is_zero(self):
        return not any(self.coefficients)

    def _other(self, other):
        if isinstance(other, PairNumber):
            if self.field.roots != other.field.roots:
                raise ValueError("unrelated pair fields require explicit conversion")
            return other
        return self.field.scalar(other)

    def __add__(self, other):
        other = self._other(other)
        return self.field.element(*(a + b for a, b in zip(self.coefficients, other.coefficients, strict=True)))

    __radd__ = __add__

    def __neg__(self):
        return self.field.element(*(-x for x in self.coefficients))

    def __sub__(self, other):
        return self + -self._other(other)

    def __rsub__(self, other):
        return self._other(other) + -self

    def __mul__(self, other):
        other = self._other(other)
        result = [Fraction(0)] * len(self.coefficients)
        for i, a in enumerate(self.coefficients):
            for j, b in enumerate(other.coefficients):
                term = a * b
                shared = i & j
                for k, root in enumerate(self.field.roots):
                    if shared & (1 << k):
                        term *= root
                result[i ^ j] += term
        return self.field.element(*result)

    __rmul__ = __mul__

    def inverse(self):
        if self.is_zero:
            raise ZeroDivisionError("exact zero field denominator")
        size = len(self.coefficients)
        columns = []
        for i in range(size):
            basis = [Fraction(0)] * size
            basis[i] = Fraction(1)
            columns.append((self * self.field.element(*basis)).coefficients)
        rows = [[columns[j][i] for j in range(size)] + [Fraction(i == 0)] for i in range(size)]
        for col in range(size):
            pivot = next(i for i in range(col, size) if rows[i][col])
            rows[col], rows[pivot] = rows[pivot], rows[col]
            scale = rows[col][col]
            rows[col] = [x / scale for x in rows[col]]
            for i in range(size):
                if i != col:
                    factor = rows[i][col]
                    rows[i] = [a - factor * b for a, b in zip(rows[i], rows[col], strict=True)]
        return self.field.element(*(row[-1] for row in rows))

    def __truediv__(self, other):
        return self * self._other(other).inverse()

    def __rtruediv__(self, other):
        return self._other(other) * self.inverse()

    def __pow__(self, exponent):
        if type(exponent) is not int:
            return NotImplemented
        if exponent < 0:
            return self.inverse() ** -exponent
        result, base = self.field.scalar(1), self
        while exponent:
            if exponent & 1:
                result *= base
            base *= base
            exponent //= 2
        return result

    def bounds(self, bits=None):
        bits = self.field.limits.initial_sqrt_bits if bits is None else bits
        roots = [sqrt_bounds(root, bits) for root in self.field.roots]
        lo = hi = Fraction(0)
        for mask, coefficient in enumerate(self.coefficients):
            a = b = Fraction(1)
            for i, (left, right) in enumerate(roots):
                if mask & (1 << i):
                    a *= left
                    b *= right
            lo += coefficient * (a if coefficient >= 0 else b)
            hi += coefficient * (b if coefficient >= 0 else a)
        return lo, hi

    def signed_bounds(self):
        if self.is_zero:
            return 0, (Fraction(0), Fraction(0))
        bits = self.field.limits.initial_sqrt_bits
        while True:
            lo, hi = self.bounds(bits)
            if lo > 0:
                return 1, (lo, hi)
            if hi < 0:
                return -1, (lo, hi)
            if bits == self.field.limits.max_sqrt_bits:
                raise SignUnresolved(self, bits)
            bits = min(2 * bits, self.field.limits.max_sqrt_bits)

    def sign(self):
        return self.signed_bounds()[0]

    def __eq__(self, other):
        if isinstance(other, PairNumber):
            return self.field.roots == other.field.roots and self.coefficients == other.coefficients
        if isinstance(other, (int, Fraction)) and not isinstance(other, bool):
            return self.coefficients[0] == other and not any(self.coefficients[1:])
        return False

    def __hash__(self):
        if not any(self.coefficients[1:]):
            return hash(self.coefficients[0])
        return hash((self.field.roots, self.coefficients))

    def __lt__(self, other):
        return (self - other).sign() < 0

    def __le__(self, other):
        return (self - other).sign() <= 0

    def __gt__(self, other):
        return (self - other).sign() > 0

    def __ge__(self, other):
        return (self - other).sign() >= 0


def compare_gaps(a, qa, b, qb):
    """Compare a/sqrt(qa) and b/sqrt(qb), without adjoining more roots."""
    if qa <= 0 or qb <= 0:
        raise ValueError("positive squared norms required")
    sa, sb = a.sign(), b.sign()
    if sa != sb:
        return (sa > sb) - (sa < sb)
    if not sa:
        return 0
    return (a * a * qb - b * b * qa).sign() * sa


def normalized_bounds(value, norm_squared):
    _, (lo, hi) = value.signed_bounds()
    a, b = sqrt_bounds(norm_squared, value.field.limits.initial_sqrt_bits)
    if a <= 0:
        raise ArithmeticError("normal norm enclosure is not positive")
    return lo / (b if lo >= 0 else a), hi / (a if hi >= 0 else b)
