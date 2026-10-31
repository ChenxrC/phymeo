"""Bounded rational outward arithmetic; no geometry or solver policy lives here."""

from dataclasses import dataclass
from fractions import Fraction as F
from math import comb, isfinite, isqrt

import numpy as np

from ..constraints.conforming_shell import ShellProofLimits, ShellUnresolved


@dataclass(frozen=True)
class Interval:
    lo: F
    hi: F

    def __post_init__(self):
        if type(self.lo) is not F or type(self.hi) is not F or self.lo > self.hi:
            raise ValueError("ordered normalized Fraction interval endpoints required")


@dataclass(frozen=True)
class Publication:
    intervals: tuple
    values: np.ndarray
    errors: tuple
    shape: tuple

    def __post_init__(self):
        if type(self.values) is not np.ndarray or self.values.dtype != np.dtype("float64"):
            raise TypeError("publication requires an actual FP64 proposal array")
        object.__setattr__(
            self,
            "values",
            np.frombuffer(self.values.tobytes(), dtype=np.float64).reshape(self.values.shape),
        )
        object.__setattr__(self, "intervals", tuple(self.intervals))
        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(self, "shape", tuple(self.shape))


class Arithmetic:
    """One counter for an entire producer or independent verifier invocation."""

    def __init__(self, limits, *, bits=None):
        if type(limits) is not ShellProofLimits:
            raise TypeError("ShellProofLimits required")
        limits.__post_init__()
        self.limits = limits
        self.operations = 0
        self.created_cells = 0
        self.bits = limits.initial_sqrt_bits if bits is None else bits
        self.set_bits(self.bits)

    def set_bits(self, bits):
        if (
            type(bits) is not int
            or not self.limits.initial_sqrt_bits <= bits <= self.limits.max_sqrt_bits
        ):
            raise ShellUnresolved("precision_exhausted", "illegal sqrt precision")
        legal = self.limits.initial_sqrt_bits
        while legal < bits:
            legal = min(2 * legal, self.limits.max_sqrt_bits)
        if legal != bits:
            raise ShellUnresolved("precision_exhausted", "precision is not a legal doubling step")
        self.bits = bits

    def tick(self):
        self.operations += 1
        if self.operations > self.limits.max_scalar_operations:
            raise ShellUnresolved("operation_budget_exhausted", "rational scalar operations")

    def check_integer(self, value):
        if type(value) is not int or abs(value).bit_length() > self.limits.max_fraction_bits:
            raise ShellUnresolved("fraction_bits_exhausted", "integer input/intermediate")
        return value

    def check(self, value):
        if type(value) is not F:
            raise ShellUnresolved("invalid_interval", "normalized Fraction required")
        self.check_integer(value.numerator)
        self.check_integer(value.denominator)
        return value

    def rational(self, value):
        if type(value) is int:
            return self.check(F(self.check_integer(value)))
        return self.check(value)

    def input_float(self, value):
        if type(value) not in (float, np.float64) or not isfinite(float(value)):
            raise ShellUnresolved("invalid_input", "finite FP64 scalar required")
        return self.check(F.from_float(float(value)))

    def exact(self, value):
        value = self.rational(value)
        return Interval(value, value)

    def interval(self, value):
        if type(value) is not Interval:
            raise ShellUnresolved("invalid_interval", "Interval required")
        self.check(value.lo)
        self.check(value.hi)
        if value.lo > value.hi:
            raise ShellUnresolved("invalid_interval", "reversed interval")
        return value

    def _binary(self, a, b, operation):
        self.check(a)
        self.check(b)
        self.tick()
        if operation in ("add", "sub"):
            if operation == "sub" and a == b:
                return F(0)
            if a == 0:
                return b if operation == "add" else -b
            if b == 0:
                return a
            if a.denominator == b.denominator:
                estimate = max(abs(a.numerator).bit_length(), abs(b.numerator).bit_length()) + 1
            else:
                estimate = (
                    max(
                        abs(a.numerator).bit_length() + b.denominator.bit_length(),
                        abs(b.numerator).bit_length() + a.denominator.bit_length(),
                    )
                    + 1
                )
                estimate = max(estimate, a.denominator.bit_length() + b.denominator.bit_length())
            if estimate > self.limits.max_fraction_bits:
                raise ShellUnresolved("fraction_bits_exhausted", "addition intermediate")
            return self.check(a + b if operation == "add" else a - b)
        if operation == "div":
            if b == 0:
                raise ShellUnresolved("division_by_zero", "exact scalar")
            b = F(b.denominator, b.numerator)
        if a == 0 or b == 0:
            return F(0)
        if (
            abs(a.numerator).bit_length() + abs(b.numerator).bit_length()
            > self.limits.max_fraction_bits
            or a.denominator.bit_length() + b.denominator.bit_length()
            > self.limits.max_fraction_bits
        ):
            raise ShellUnresolved("fraction_bits_exhausted", "multiplication intermediate")
        return self.check(a * b)

    def qadd(self, a, b):
        return self._binary(a, b, "add")

    def qneg(self, value):
        self.check(value)
        self.tick()
        return self.check(-value)

    def qabs(self, value):
        self.check(value)
        return self.qneg(value) if value < 0 else value

    def qsub(self, a, b):
        return self._binary(a, b, "sub")

    def qmul(self, a, b):
        return self._binary(a, b, "mul")

    def qdiv(self, a, b):
        return self._binary(a, b, "div")

    def qpow(self, value, exponent):
        if type(exponent) is not int or not 0 <= exponent <= 16:
            raise ShellUnresolved("polynomial_budget", "exponent")
        result = F(1)
        for _ in range(exponent):
            result = self.qmul(result, value)
        return result

    def add(self, a, b):
        a, b = self.interval(a), self.interval(b)
        return Interval(self.qadd(a.lo, b.lo), self.qadd(a.hi, b.hi))

    def neg(self, value):
        value = self.interval(value)
        return Interval(self.qneg(value.hi), self.qneg(value.lo))

    def sub(self, a, b):
        return self.add(a, self.neg(b))

    def mul(self, a, b):
        a, b = self.interval(a), self.interval(b)
        products = tuple(self.qmul(x, y) for x in (a.lo, a.hi) for y in (b.lo, b.hi))
        return Interval(min(products), max(products))

    def div(self, a, b):
        a, b = self.interval(a), self.interval(b)
        if b.lo <= 0 <= b.hi:
            raise ShellUnresolved("division_interval_contains_zero", "denominator not separated")
        reciprocal = Interval(self.qdiv(F(1), b.hi), self.qdiv(F(1), b.lo))
        return self.mul(a, reciprocal)

    def square(self, value):
        value = self.interval(value)
        ends = (self.qmul(value.lo, value.lo), self.qmul(value.hi, value.hi))
        return Interval(F(0) if value.lo <= 0 <= value.hi else min(ends), max(ends))

    def _isqrt(self, value):
        self.check_integer(value)
        if value < 0:
            raise ShellUnresolved("sqrt_negative", "integer")
        self.tick()
        return self.check_integer(isqrt(value))

    def _sqrt_endpoint(self, value):
        self.check(value)
        if value < 0:
            raise ShellUnresolved("sqrt_negative", "rational endpoint")
        numerator_root = self._isqrt(value.numerator)
        denominator_root = self._isqrt(value.denominator)
        n_square = self.qmul(F(numerator_root), F(numerator_root))
        d_square = self.qmul(F(denominator_root), F(denominator_root))
        if n_square == value.numerator and d_square == value.denominator:
            root = self.qdiv(F(numerator_root), F(denominator_root))
            return root, root
        if (
            value.numerator.bit_length() + 2 * self.bits > self.limits.max_fraction_bits
            or self.bits + 1 > self.limits.max_fraction_bits
        ):
            raise ShellUnresolved("fraction_bits_exhausted", "sqrt shift intermediate")
        shifted = self.check_integer(value.numerator << (2 * self.bits))
        self.tick()
        quotient = self.check_integer(shifted // value.denominator)
        lower_integer = self._isqrt(quotient)
        denominator = self.check_integer(1 << self.bits)
        lower = self.qdiv(F(lower_integer), F(denominator))
        upper_integer = self.check_integer(lower_integer + 1)
        self.tick()
        upper = self.qdiv(F(upper_integer), F(denominator))
        if self.qmul(lower, lower) > value or self.qmul(upper, upper) < value:
            raise ShellUnresolved("sqrt_witness_invalid", "integer enclosure")
        return lower, upper

    def sqrt(self, value):
        value = self.interval(value)
        if value.lo < 0:
            raise ShellUnresolved("sqrt_negative", "interval endpoint")
        lower = self._sqrt_endpoint(value.lo)[0]
        upper = self._sqrt_endpoint(value.hi)[1]
        return Interval(lower, upper)

    def sum(self, values):
        result = self.exact(0)
        for value in values:
            result = self.add(result, value)
        return result

    def vadd(self, a, b):
        return tuple(self.add(x, y) for x, y in zip(a, b, strict=True))

    def vsub(self, a, b):
        return tuple(self.sub(x, y) for x, y in zip(a, b, strict=True))

    def scale(self, scalar, vector):
        return tuple(self.mul(scalar, value) for value in vector)

    def dot(self, a, b):
        return self.sum(self.mul(x, y) for x, y in zip(a, b, strict=True))

    def cross(self, a, b):
        return tuple(
            self.sub(self.mul(a[j], b[k]), self.mul(a[k], b[j]))
            for j, k in ((1, 2), (2, 0), (0, 1))
        )

    def norm(self, vector):
        return self.sqrt(self.sum(self.square(value) for value in vector))

    def cross_matrix(self, vector):
        x, y, z = vector
        zero = self.exact(0)
        return ((zero, self.neg(z), y), (z, zero, self.neg(x)), (self.neg(y), x, zero))

    def matmul(self, left, right):
        return tuple(
            tuple(self.dot(row, column) for column in zip(*right, strict=True)) for row in left
        )

    def determinant(self, matrix):
        return self.dot(matrix[0], self.cross(matrix[1], matrix[2]))

    def inverse3(self, matrix):
        columns = tuple(zip(*matrix, strict=True))
        crosses = (
            self.cross(columns[1], columns[2]),
            self.cross(columns[2], columns[0]),
            self.cross(columns[0], columns[1]),
        )
        determinant = self.dot(columns[0], crosses[0])
        if determinant.lo <= 0 <= determinant.hi:
            raise ShellUnresolved("reference_inverse_unknown", "determinant includes zero")
        return tuple(
            tuple(self.div(entry, determinant) for entry in row) for row in crosses
        ), determinant

    def poly(self, values):
        if type(values) not in (tuple, list) or not 1 <= len(values) <= 9:
            raise ShellUnresolved("polynomial_budget", "degree")
        result = [self.rational(value) for value in values]
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        return tuple(result)

    def poly_add(self, a, b):
        a, b = self.poly(a), self.poly(b)
        return self.poly(
            tuple(
                self.qadd(a[i] if i < len(a) else F(0), b[i] if i < len(b) else F(0))
                for i in range(max(len(a), len(b)))
            )
        )

    def poly_sub(self, a, b):
        return self.poly_add(a, tuple(self.qneg(v) for v in self.poly(b)))

    def poly_mul(self, a, b):
        a, b = self.poly(a), self.poly(b)
        if len(a) + len(b) - 1 > 9:
            raise ShellUnresolved("polynomial_budget", "product degree")
        result = [F(0)] * (len(a) + len(b) - 1)
        for i, left in enumerate(a):
            for j, right in enumerate(b):
                result[i + j] = self.qadd(result[i + j], self.qmul(left, right))
        return self.poly(result)

    def poly_vsub(self, a, b):
        return tuple(self.poly_sub(x, y) for x, y in zip(a, b, strict=True))

    def poly_vadd(self, a, b):
        return tuple(self.poly_add(x, y) for x, y in zip(a, b, strict=True))

    def poly_cross(self, a, b):
        return tuple(
            self.poly_sub(self.poly_mul(a[j], b[k]), self.poly_mul(a[k], b[j]))
            for j, k in ((1, 2), (2, 0), (0, 1))
        )

    def poly_dot(self, a, b):
        result = (F(0),)
        for x, y in zip(a, b, strict=True):
            result = self.poly_add(result, self.poly_mul(x, y))
        return result

    def polynomial_bounds(self, polynomial, left, right):
        polynomial = self.poly(polynomial)
        self.check(left)
        self.check(right)
        if left > right:
            raise ShellUnresolved("invalid_time_cell", "reversed interval")
        degree = len(polynomial) - 1
        width = self.qsub(right, left)
        powers = []
        for j in range(degree + 1):
            value = F(0)
            for k in range(j, degree + 1):
                term = self.qmul(polynomial[k], F(comb(k, j)))
                term = self.qmul(term, self.qpow(left, k - j))
                term = self.qmul(term, self.qpow(width, j))
                value = self.qadd(value, term)
            powers.append(value)
        coefficients = []
        for i in range(degree + 1):
            value = F(0)
            for j in range(i + 1):
                value = self.qadd(
                    value, self.qmul(powers[j], self.qdiv(F(comb(i, j)), F(comb(degree, j))))
                )
            coefficients.append(value)
        return Interval(min(coefficients), max(coefficients))

    def poly_vector_bounds(self, vector, left, right):
        return tuple(self.polynomial_bounds(value, left, right) for value in vector)

    def publish(self, intervals, shape):
        if (
            type(shape) not in (tuple, list)
            or not shape
            or any(type(v) is not int or v <= 0 for v in shape)
        ):
            raise ShellUnresolved("invalid_publication", "shape")
        size = 1
        for dimension in shape:
            size *= dimension
        if size != len(intervals) or size > self.limits.max_nodes * 18:
            raise ShellUnresolved("shape_budget", "publication")
        values, errors = [], []
        for interval in intervals:
            interval = self.interval(interval)
            midpoint = self.qdiv(self.qadd(interval.lo, interval.hi), F(2))
            try:
                proposal = float(midpoint)
            except (ValueError, OverflowError) as error:
                raise ShellUnresolved("publication_not_representable", "FP64 conversion") from error
            if not isfinite(proposal):
                raise ShellUnresolved("publication_not_representable", "nonfinite FP64")
            stored = self.input_float(proposal)
            error = max(
                self.qabs(self.qsub(stored, interval.lo)), self.qabs(self.qsub(stored, interval.hi))
            )
            if error >= F(1, 2**40):
                raise ShellUnresolved(
                    "publication_error", "point/J storage error is not below 2^-40"
                )
            values.append(proposal)
            errors.append(error)
        return Publication(
            tuple(intervals),
            np.array(values, dtype=np.float64).reshape(shape),
            tuple(errors),
            tuple(shape),
        )

    def visit_cell(self, depth):
        if type(depth) is not int or depth < 0 or depth > self.limits.max_time_depth:
            raise ShellUnresolved("time_depth_exhausted", "cell depth")
        self.created_cells += 1
        if self.created_cells > self.limits.max_time_cells:
            raise ShellUnresolved("time_cells_exhausted", "all visited/created cells")

    def qvsub(self, left, right):
        return tuple(self.qsub(a, b) for a, b in zip(left, right, strict=True))

    def qvadd(self, left, right):
        return tuple(self.qadd(a, b) for a, b in zip(left, right, strict=True))

    def qdot(self, left, right):
        value = F(0)
        for a, b in zip(left, right, strict=True):
            value = self.qadd(value, self.qmul(a, b))
        return value

    def qcross(self, left, right):
        return tuple(
            self.qsub(self.qmul(left[j], right[k]), self.qmul(left[k], right[j]))
            for j, k in ((1, 2), (2, 0), (0, 1))
        )

    def qcross2(self, left, right):
        return self.qsub(self.qmul(left[0], right[1]), self.qmul(left[1], right[0]))
