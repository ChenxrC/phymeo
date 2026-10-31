"""Checked STATIC multiquadratic primitives; no topology or field-admission oracle."""

from fractions import Fraction as F
from math import comb, isqrt

from ..constraints.conforming_shell import ShellUnresolved
from ..constraints.shell_boundary import BoundaryProofLimits
from ._shell_intervals import Arithmetic, Interval


class Meter(Arithmetic):
    def __init__(self, limits, *, bits=None):
        self.polynomial_ceiling = 8
        self.expression_ceiling = 4096
        self.peak_fraction_bits = 0
        super().__init__(limits, bits=bits)

    def check_integer(self, value):
        result = super().check_integer(value)
        self.peak_fraction_bits = max(self.peak_fraction_bits, abs(value).bit_length())
        return result

    def poly(self, values):
        if (
            type(values) not in (tuple, list)
            or not values
            or len(values) > self.polynomial_ceiling + 1
            or len(values) > self.expression_ceiling
        ):
            raise ShellUnresolved("polynomial_budget", "rational intermediate shape")
        return super().poly(values)

    def poly_mul(self, a, b):
        left, right = self.poly(a), self.poly(b)
        size = len(left) + len(right) - 1
        if size > self.polynomial_ceiling + 1 or size > self.expression_ceiling:
            raise ShellUnresolved("polynomial_budget", "rational intermediate product")
        return super().poly_mul(left, right)

    def perfect_square(self, value):
        """Exact nonnegative rational square test, including all integer-root work."""
        self.check(value)
        if value < 0:
            return None
        self.tick()
        n = self.check_integer(isqrt(self.check_integer(value.numerator)))
        self.tick()
        d = self.check_integer(isqrt(self.check_integer(value.denominator)))
        ns = self.qmul(F(n), F(n))
        ds = self.qmul(F(d), F(d))
        if ns != value.numerator or ds != value.denominator:
            return None
        return self.qdiv(F(n), F(d))


class FieldArithmetic:
    """Caller independently authenticates basis independence and branch dependencies.

    This class checks dimensions/types/guards and performs only algebra. Constructing
    it is not proof that its generators form a field. Producer and verifier each
    perform their OWN square-class admission before using inversion/sign decisions.
    """

    def __init__(self, generators, meter, limits=BoundaryProofLimits()):
        if type(limits) is not BoundaryProofLimits or not isinstance(meter, Meter):
            raise TypeError("bounded policy and checked Meter required")
        limits.__post_init__()
        if type(generators) not in (tuple, list) or len(generators) > limits.max_field_rank:
            raise ShellUnresolved("unsupported_algebraic_rank", "static basis")
        self.generators = tuple(meter.check(q) for q in generators)
        if any(q <= 0 for q in self.generators):
            raise ShellUnresolved("branch_or_endpoint_mismatch", "positive generators")
        if meter.limits != limits.strict_limits:
            raise TypeError("one original policy-bound counter required")
        meter.polynomial_ceiling = min(meter.polynomial_ceiling, limits.max_polynomial_degree)
        meter.expression_ceiling = min(meter.expression_ceiling, limits.max_expression_nodes)
        self.meter, self.limits = meter, limits
        self.rank = len(self.generators)
        self.width = 1 << self.rank
        if self.width > limits.max_expression_nodes:
            raise ShellUnresolved("polynomial_budget", "field coefficient expression size")
        self.zero = (F(0),) * self.width
        self.one = (F(1),) + (F(0),) * (self.width - 1)
        products = [F(1)] * self.width
        for mask in range(1, self.width):
            bit = mask & -mask
            index = bit.bit_length() - 1
            products[mask] = (
                self.generators[index]
                if mask == bit
                else meter.qmul(products[mask ^ bit], self.generators[index])
            )
        self.products = tuple(products)
        self._basis_intervals = {}

    def element(self, values):
        if type(values) not in (tuple, list) or len(values) != self.width:
            raise ShellUnresolved("invalid_shape_or_identity", "field mask dimension")
        return tuple(self.meter.check(v) for v in values)

    def rational(self, value):
        if type(value) not in (int, F):
            raise ShellUnresolved("invalid_shape_or_identity", "exact rational required")
        return (self.meter.rational(value),) + (F(0),) * (self.width - 1)

    def add(self, a, b):
        a, b = self.element(a), self.element(b)
        return tuple(self.meter.qadd(x, y) for x, y in zip(a, b, strict=True))

    def sub(self, a, b):
        a, b = self.element(a), self.element(b)
        return tuple(self.meter.qsub(x, y) for x, y in zip(a, b, strict=True))

    def neg(self, a):
        return tuple(self.meter.qneg(q) for q in self.element(a))

    def scale(self, q, a):
        self.meter.check(q)
        return tuple(self.meter.qmul(q, v) for v in self.element(a))

    def _multiply(self, a, b):
        width = len(a)
        if width == 1:
            return (self.meter.qmul(a[0], b[0]),)
        result = [F(0)] * width
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                value = self.meter.qmul(x, y)
                if i & j:
                    value = self.meter.qmul(value, self.products[i & j])
                result[i ^ j] = self.meter.qadd(result[i ^ j], value)
        return tuple(result)

    def mul(self, a, b):
        return self._multiply(self.element(a), self.element(b))

    def _inverse(self, value):
        nonzero = [i for i, q in enumerate(value) if q]
        if not nonzero:
            raise ShellUnresolved("singular_expression", "zero inverse")
        if len(nonzero) == 1:
            mask = nonzero[0]
            denominator = value[mask]
            if mask:
                denominator = self.meter.qmul(denominator, self.products[mask])
            result = [F(0)] * len(value)
            result[mask] = self.meter.qdiv(F(1), denominator)
            return tuple(result)
        half = len(value) // 2
        a, b = value[:half], value[half:]
        aa, bb = self._multiply(a, a), self._multiply(b, b)
        generator = self.generators[half.bit_length() - 1]
        norm = tuple(
            self.meter.qsub(x, self.meter.qmul(generator, y)) for x, y in zip(aa, bb, strict=True)
        )
        inverse = self._inverse(norm)
        left, right = self._multiply(a, inverse), self._multiply(b, inverse)
        return left + tuple(self.meter.qneg(v) for v in right)

    def inverse(self, value):
        return self.element(self._inverse(self.element(value)))

    def div(self, a, b):
        return self.mul(a, self.inverse(b))

    def root_from_dependency(self, radicand, coefficient, mask):
        self.meter.check(radicand)
        self.meter.check(coefficient)
        if type(mask) is not int or not 0 <= mask < self.width or coefficient <= 0:
            raise ShellUnresolved("branch_or_endpoint_mismatch", "positive root dependency")
        square = self.meter.qmul(self.meter.qmul(coefficient, coefficient), self.products[mask])
        if square != radicand:
            raise ShellUnresolved("branch_or_endpoint_mismatch", "root square identity")
        value = [F(0)] * self.width
        value[mask] = coefficient
        return tuple(value)

    def _interval_basis(self):
        bits = self.meter.bits
        if bits not in self._basis_intervals:
            basis = [Interval(F(1), F(1))] * self.width
            for mask in range(1, self.width):
                bit = mask & -mask
                i = bit.bit_length() - 1
                if mask == bit:
                    basis[mask] = self.meter.sqrt(self.meter.exact(self.generators[i]))
                else:
                    basis[mask] = self.meter.mul(basis[mask ^ bit], basis[bit])
            self._basis_intervals[bits] = tuple(basis)
        return self._basis_intervals[bits]

    def enclose(self, value):
        value = self.element(value)
        if self.rank == 0:
            return Interval(value[0], value[0])
        lo, hi = F(0), F(0)
        for q, basis in zip(value, self._interval_basis(), strict=True):
            lower, upper = (basis.lo, basis.hi) if q >= 0 else (basis.hi, basis.lo)
            lo = self.meter.qadd(lo, self.meter.qmul(q, lower))
            hi = self.meter.qadd(hi, self.meter.qmul(q, upper))
        return Interval(lo, hi)

    def refine(self):
        maximum = self.meter.limits.max_sqrt_bits
        if self.meter.bits == maximum:
            raise ShellUnresolved("unresolved_sign", "legal precision schedule exhausted")
        self.meter.set_bits(min(2 * self.meter.bits, maximum))

    def sign(self, value):
        value = self.element(value)
        if value == self.zero:
            return 0
        while True:
            interval = self.enclose(value)
            if interval.lo > 0:
                return 1
            if interval.hi < 0:
                return -1
            self.refine()

    def signed(self, value):
        sign = self.sign(value)
        return sign, self.enclose(value)

    def vadd(self, a, b):
        return tuple(self.add(x, y) for x, y in zip(a, b, strict=True))

    def vsub(self, a, b):
        return tuple(self.sub(x, y) for x, y in zip(a, b, strict=True))

    def dot(self, a, b):
        result = self.zero
        for x, y in zip(a, b, strict=True):
            result = self.add(result, self.mul(x, y))
        return result

    def cross(self, a, b):
        return tuple(
            self.sub(self.mul(a[j], b[k]), self.mul(a[k], b[j]))
            for j, k in ((1, 2), (2, 0), (0, 1))
        )

    def determinant(self, a, b, c):
        return self.dot(a, self.cross(b, c))

    def poly(self, values):
        if (
            type(values) not in (tuple, list)
            or not values
            or len(values) > self.limits.max_polynomial_degree + 1
            or len(values) * self.width > self.limits.max_expression_nodes
        ):
            raise ShellUnresolved("polynomial_budget", "bounded static-field expression")
        result = [self.element(q) for q in values]
        while len(result) > 1 and result[-1] == self.zero:
            result.pop()
        return tuple(result)

    def poly_add(self, a, b):
        a, b = self.poly(a), self.poly(b)
        return self.poly(
            tuple(
                self.add(a[i] if i < len(a) else self.zero, b[i] if i < len(b) else self.zero)
                for i in range(max(len(a), len(b)))
            )
        )

    def poly_sub(self, a, b):
        a, b = self.poly(a), self.poly(b)
        return self.poly(
            tuple(
                self.sub(a[i] if i < len(a) else self.zero, b[i] if i < len(b) else self.zero)
                for i in range(max(len(a), len(b)))
            )
        )

    def poly_mul(self, a, b):
        a, b = self.poly(a), self.poly(b)
        size = len(a) + len(b) - 1
        if (
            size > self.limits.max_polynomial_degree + 1
            or size * self.width > self.limits.max_expression_nodes
        ):
            raise ShellUnresolved("polynomial_budget", "intermediate product degree")
        result = [self.zero] * size
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                result[i + j] = self.add(result[i + j], self.mul(x, y))
        return self.poly(result)

    def poly_evaluate(self, polynomial, time):
        self.meter.check(time)
        result = self.zero
        for coefficient in reversed(self.poly(polynomial)):
            result = self.add(self.scale(time, result), coefficient)
        return result

    def bernstein(self, polynomial, left, right):
        polynomial = self.poly(polynomial)
        self.meter.check(left)
        self.meter.check(right)
        if left > right:
            raise ShellUnresolved("incomplete_time_cover", "reversed span")
        d = len(polynomial) - 1
        width = self.meter.qsub(right, left)
        transformed = []
        for j in range(d + 1):
            value = self.zero
            for k in range(j, d + 1):
                term = self.scale(F(comb(k, j)), polynomial[k])
                term = self.scale(self.meter.qpow(left, k - j), term)
                term = self.scale(self.meter.qpow(width, j), term)
                value = self.add(value, term)
            transformed.append(value)
        result = []
        for i in range(d + 1):
            value = self.zero
            for j in range(i + 1):
                factor = self.meter.qdiv(F(comb(i, j)), F(comb(d, j)))
                value = self.add(value, self.scale(factor, transformed[j]))
            result.append(value)
        return tuple(result)

    def polynomial_bounds(self, polynomial, left, right):
        intervals = tuple(self.enclose(v) for v in self.bernstein(polynomial, left, right))
        return Interval(min(v.lo for v in intervals), max(v.hi for v in intervals))

    def rational_function(self, numerator, denominator):
        p, q = self.poly(numerator), self.poly(denominator)
        if q == (self.zero,):
            raise ShellUnresolved("singular_expression", "zero polynomial denominator")
        if (len(p) + len(q)) * self.width > self.limits.max_expression_nodes:
            raise ShellUnresolved("polynomial_budget", "rational expression nodes")
        return p, q

    def function_bounds(self, function, left, right):
        p, q = self.rational_function(*function)
        numerator = self.polynomial_bounds(p, left, right)
        denominator = self.polynomial_bounds(q, left, right)
        if denominator.lo <= 0 <= denominator.hi:
            raise ShellUnresolved("singular_expression", "denominator not separated on closed span")
        return self.meter.div(numerator, denominator)

    def function_equal(self, left, right):
        a, b = self.rational_function(*left)
        c, d = self.rational_function(*right)
        return self.poly_sub(self.poly_mul(a, d), self.poly_mul(c, b)) == (self.zero,)

    def factor_endpoint(self, polynomial, endpoint, multiplicity):
        if (
            type(multiplicity) is not int
            or not 0 <= multiplicity <= self.limits.max_factor_exponent
        ):
            raise ShellUnresolved("polynomial_budget", "factor exponent")
        self.meter.check(endpoint)
        value = self.poly(polynomial)
        for _ in range(multiplicity):
            if len(value) == 1:
                raise ShellUnresolved("branch_or_endpoint_mismatch", "false endpoint factor")
            quotient = [self.zero] * (len(value) - 1)
            quotient[-1] = value[-1]
            for i in range(len(quotient) - 2, -1, -1):
                quotient[i] = self.add(value[i + 1], self.scale(endpoint, quotient[i + 1]))
            remainder = self.add(value[0], self.scale(endpoint, quotient[0]))
            if remainder != self.zero:
                raise ShellUnresolved("branch_or_endpoint_mismatch", "nonzero factor remainder")
            value = self.poly(quotient)
        return value

    def publish(self, values, shape):
        values = tuple(self.element(v) for v in values)
        while True:
            intervals = tuple(self.enclose(v) for v in values)
            try:
                return self.meter.publish(intervals, shape)
            except ShellUnresolved as error:
                if error.category != "publication_error":
                    raise
                self.refine()
