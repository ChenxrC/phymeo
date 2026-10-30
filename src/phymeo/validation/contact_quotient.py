"""Bounded exact quotient evidence, unconnected to physical contact admission.

A certificate chooses a zero-nullspace algebraic representative. It is not a
measured owner pressure, a friction capacity, a state update or external work.
Only fixed, translational coordinates and a positive frozen inverse mass belong
here. Production dependencies are standard library only.
"""

from dataclasses import dataclass, fields
from fractions import Fraction
from hashlib import sha256
from operator import add, mul, sub, truediv

_ZERO = Fraction(0)
_ONE = Fraction(1)
_CONDITION_CEILING = Fraction(1_000_000_000_000)
_SCHEMA = "exact-contact-quotient-v1"
_STATUS = "exact_algebra_certified"
_POLICY_CEILINGS = (
    ("max_owner_rows", 192),
    ("max_coordinates", 256),
    ("max_fraction_bits", 4096),
    ("max_scalar_operations", 2_000_000),
)
Matrix = tuple[tuple[Fraction, ...], ...]
Vector = tuple[Fraction, ...]


class QuotientUnresolved(ValueError):
    """A short failure category; no partially certified evidence is returned."""


@dataclass(frozen=True)
class QuotientProofLimits:
    max_owner_rows: int = 192
    max_coordinates: int = 256
    max_fraction_bits: int = 4096
    max_scalar_operations: int = 2_000_000

    def __post_init__(self):
        for name, ceiling in _POLICY_CEILINGS:
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= ceiling:
                raise QuotientUnresolved("policy")


def _own_limits(limits):
    if type(limits) is not QuotientProofLimits:
        raise QuotientUnresolved("policy")
    return QuotientProofLimits(**{name: getattr(limits, name) for name, _ in _POLICY_CEILINGS})


def _container(value, length=None):
    if type(value) not in (list, tuple):
        raise QuotientUnresolved("shape")
    if length is not None and len(value) != length:
        raise QuotientUnresolved("shape")


def _matrix_shape(value, rows, columns):
    _container(value, rows)
    for row in value:
        _container(row, columns)


def _fraction(value, limits):
    if type(value) not in (int, Fraction):
        raise QuotientUnresolved("scalar")
    numerator = value if type(value) is int else value.numerator
    denominator = 1 if type(value) is int else value.denominator
    if (
        numerator.bit_length() > limits.max_fraction_bits
        or denominator.bit_length() > limits.max_fraction_bits
    ):
        raise QuotientUnresolved("bits")
    return Fraction(value) if type(value) is int else value


def _vector(value, length, limits):
    _container(value, length)
    return tuple(_fraction(item, limits) for item in value)


def _matrix(value, rows, columns, limits):
    _matrix_shape(value, rows, columns)
    return tuple(_vector(row, columns, limits) for row in value)


def _text(value, max_bytes):
    if type(value) is not str or not value or len(value) > max_bytes:
        raise QuotientUnresolved("identity")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise QuotientUnresolved("identity") from error
    if len(encoded) > max_bytes:
        raise QuotientUnresolved("identity")
    return value


def _ids(values, length):
    _container(values, length)
    result = tuple(_text(value, 256) for value in values)
    if len(set(result)) != length:
        raise QuotientUnresolved("identity")
    return result


@dataclass(frozen=True)
class _Inputs:
    difference_map: Matrix
    inverse_mass: Matrix
    delivered_impulse: Vector
    impulse_uncertainty: Vector
    owner_ids: tuple[str, ...]
    coordinate_ids: tuple[str, ...]
    limits: QuotientProofLimits


def _inputs(b, w, ell, e, owner_ids, coordinate_ids, limits):
    limits = _own_limits(limits)
    _container(b)
    if not 0 < len(b) <= limits.max_owner_rows:
        raise QuotientUnresolved("shape")
    _container(b[0])
    m, n = len(b), len(b[0])
    if not 0 < n <= limits.max_coordinates:
        raise QuotientUnresolved("shape")
    # Check all shapes and IDs before copying any scalar input containers.
    _matrix_shape(b, m, n)
    _matrix_shape(w, n, n)
    _container(ell, n)
    _container(e, n)
    owners, coordinates = _ids(owner_ids, m), _ids(coordinate_ids, n)
    exact_e = _vector(e, n, limits)
    if any(value < _ZERO for value in exact_e):
        raise QuotientUnresolved("uncertainty")
    return _Inputs(
        _matrix(b, m, n, limits),
        _matrix(w, n, n, limits),
        _vector(ell, n, limits),
        exact_e,
        owners,
        coordinates,
        limits,
    )


@dataclass(frozen=True)
class ContactQuotientCertificate:
    """Deep-owned algebraic evidence; construction alone does not verify it."""

    owner_ids: tuple[str, ...]
    coordinate_ids: tuple[str, ...]
    difference_map: Matrix
    inverse_mass: Matrix
    delivered_impulse: Vector
    impulse_uncertainty: Vector
    limits: QuotientProofLimits
    input_fingerprint: str
    rank: int
    nullspace_basis: Matrix
    response_matrix: Matrix
    nullspace_projector: Matrix
    response_scale: Fraction
    gauged_response: Matrix
    gauged_inverse: Matrix
    pressure_map: Matrix
    pressure: Vector
    remainder: Vector
    pressure_uncertainty: Vector
    remainder_uncertainty: Vector
    condition_number: Fraction
    schema: str = _SCHEMA
    status: str = _STATUS

    def __post_init__(self):
        owned = _inputs(
            self.difference_map,
            self.inverse_mass,
            self.delivered_impulse,
            self.impulse_uncertainty,
            self.owner_ids,
            self.coordinate_ids,
            self.limits,
        )
        for item in fields(_Inputs):
            object.__setattr__(self, item.name, getattr(owned, item.name))
        m, n = len(owned.owner_ids), len(owned.coordinate_ids)
        if type(self.rank) is not int or not 0 < self.rank <= min(m, n):
            raise QuotientUnresolved("rank")
        for name, rows, columns in (
            ("nullspace_basis", m, m - self.rank),
            ("response_matrix", m, m),
            ("nullspace_projector", m, m),
            ("gauged_response", m, m),
            ("gauged_inverse", m, m),
            ("pressure_map", m, n),
        ):
            object.__setattr__(self, name, _matrix(getattr(self, name), rows, columns, self.limits))
        for name, length in (
            ("pressure", m),
            ("remainder", n),
            ("pressure_uncertainty", m),
            ("remainder_uncertainty", n),
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), length, self.limits))
        for name in ("response_scale", "condition_number"):
            object.__setattr__(self, name, _fraction(getattr(self, name), self.limits))
        _text(self.schema, 128)
        _text(self.status, 128)
        _text(self.input_fingerprint, 64)
        if len(self.input_fingerprint) != 64:
            raise QuotientUnresolved("fingerprint")


class _Budget:
    """Every rational binary operation passes here, including proof checking."""

    def __init__(self, limits):
        self.limits = limits
        self.operations = 0

    def _apply(self, operation, left, right):
        if self.operations >= self.limits.max_scalar_operations:
            raise QuotientUnresolved("operations")
        self.operations += 1
        if operation is truediv and right == _ZERO:
            raise QuotientUnresolved("singular")
        return _fraction(operation(left, right), self.limits)

    def add(self, left, right):
        return self._apply(add, left, right)

    def sub(self, left, right):
        return self._apply(sub, left, right)

    def mul(self, left, right):
        return self._apply(mul, left, right)

    def div(self, left, right):
        return self._apply(truediv, left, right)


def _zero(rows, columns):
    return tuple(tuple(_ZERO for _ in range(columns)) for _ in range(rows))


def _identity(size):
    return tuple(tuple(_ONE if i == j else _ZERO for j in range(size)) for i in range(size))


def _transpose(matrix):
    return tuple(tuple(row[j] for row in matrix) for j in range(len(matrix[0])))


def _dot(left, right, budget):
    total = _ZERO
    for a, b in zip(left, right, strict=True):
        total = budget.add(total, budget.mul(a, b))
    return total


def _mm(left, right, budget):
    columns = _transpose(right)
    return tuple(tuple(_dot(row, column, budget) for column in columns) for row in left)


def _mv(matrix, vector, budget):
    return tuple(_dot(row, vector, budget) for row in matrix)


def _scale(matrix, factor, budget):
    return tuple(tuple(budget.mul(value, factor) for value in row) for row in matrix)


def _combine(left, right, operation):
    return tuple(
        tuple(operation(a, b) for a, b in zip(row, other, strict=True))
        for row, other in zip(left, right, strict=True)
    )


def _vsub(left, right, budget):
    return tuple(budget.sub(a, b) for a, b in zip(left, right, strict=True))


def _absolute(matrix):
    return tuple(tuple(abs(value) for value in row) for row in matrix)


def _norm_inf(matrix, budget):
    maximum = _ZERO
    for row in matrix:
        total = _ZERO
        for value in row:
            total = budget.add(total, abs(value))
        maximum = max(maximum, total)
    return maximum


def _rref(matrix, budget, pivot_limit=None):
    """Deterministic first-nonzero row pivots, with every arithmetic step metered."""
    rows = [list(row) for row in matrix]
    height, width = len(rows), len(rows[0])
    limit = width if pivot_limit is None else pivot_limit
    pivot_row = 0
    pivots = []
    for column in range(limit):
        selected = next((i for i in range(pivot_row, height) if rows[i][column] != _ZERO), None)
        if selected is None:
            continue
        rows[pivot_row], rows[selected] = rows[selected], rows[pivot_row]
        divisor = rows[pivot_row][column]
        for j in range(column, width):
            rows[pivot_row][j] = budget.div(rows[pivot_row][j], divisor)
        for i in range(height):
            if i == pivot_row or rows[i][column] == _ZERO:
                continue
            factor = rows[i][column]
            for j in range(column, width):
                rows[i][j] = budget.sub(rows[i][j], budget.mul(factor, rows[pivot_row][j]))
        pivots.append(column)
        pivot_row += 1
        if pivot_row == height:
            break
    return tuple(tuple(row) for row in rows), tuple(pivots)


def _nullspace(difference_map, budget):
    reduced, pivots = _rref(_transpose(difference_map), budget)
    rank = len(pivots)
    if rank == 0:
        raise QuotientUnresolved("zero-rank")
    owners = len(difference_map)
    free = tuple(i for i in range(owners) if i not in pivots)
    basis = [[_ZERO for _ in free] for _ in range(owners)]
    for j, column in enumerate(free):
        basis[column][j] = _ONE
        for row, pivot in enumerate(pivots):
            basis[pivot][j] = -reduced[row][column]
    return rank, tuple(tuple(row) for row in basis)


def _inverse(matrix, budget):
    size = len(matrix)
    identity = _identity(size)
    augmented = tuple(row + unit for row, unit in zip(matrix, identity, strict=True))
    reduced, pivots = _rref(augmented, budget, pivot_limit=size)
    if pivots != tuple(range(size)):
        raise QuotientUnresolved("singular")
    if tuple(row[:size] for row in reduced) != identity:
        raise QuotientUnresolved("inverse")
    return tuple(row[size:] for row in reduced)


def _positive_ldl(matrix, budget):
    if matrix != _transpose(matrix):
        raise QuotientUnresolved("symmetric")
    size = len(matrix)
    lower = [list(row) for row in _identity(size)]
    diagonal = []
    for j in range(size):
        pivot = matrix[j][j]
        for k in range(j):
            product = budget.mul(budget.mul(lower[j][k], lower[j][k]), diagonal[k])
            pivot = budget.sub(pivot, product)
        if pivot <= _ZERO:
            raise QuotientUnresolved("positive")
        diagonal.append(pivot)
        for i in range(j + 1, size):
            value = matrix[i][j]
            for k in range(j):
                product = budget.mul(budget.mul(lower[i][k], lower[j][k]), diagonal[k])
                value = budget.sub(value, product)
            lower[i][j] = budget.div(value, pivot)


def _fingerprint(inputs):
    """Length-framed UTF-8 tokens; exact numerator/denominator pairs, never floats."""
    digest = sha256()

    def token(value):
        encoded = value.encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)

    def rational(value):
        token(str(value.numerator))
        token(str(value.denominator))

    token(_SCHEMA)
    token(_STATUS)
    token("translational-fixed-map-constant-mass")
    token("coordinates:m;B:1;W:1/kg;ell:N*s;e:N*s")
    for name, _ in _POLICY_CEILINGS:
        token(name)
        token(str(getattr(inputs.limits, name)))
    for name in ("owner_ids", "coordinate_ids"):
        values = getattr(inputs, name)
        token(name)
        token(str(len(values)))
        for value in values:
            token(value)
    for name in ("difference_map", "inverse_mass"):
        matrix = getattr(inputs, name)
        token(name)
        token(str(len(matrix)))
        token(str(len(matrix[0])))
        for row in matrix:
            for value in row:
                rational(value)
    for name in ("delivered_impulse", "impulse_uncertainty"):
        values = getattr(inputs, name)
        token(name)
        token(str(len(values)))
        for value in values:
            rational(value)
    return digest.hexdigest()


def _require_equal(left, right, category="identity"):
    if left != right:
        raise QuotientUnresolved(category)


def _verify_evidence(certificate, inputs, budget):
    """Check submitted identities; never regenerate a producer certificate.

    The producer also invokes this checker within its existing budget before
    publishing. The public verifier supplies a fresh counter and independently
    owned external inputs. Alternate complete nullspace bases are accepted.
    """
    c = certificate
    _require_equal(c.schema, _SCHEMA, "schema")
    _require_equal(c.status, _STATUS, "status")
    for item in fields(_Inputs):
        _require_equal(getattr(c, item.name), getattr(inputs, item.name), "binding")
    _require_equal(c.input_fingerprint, _fingerprint(inputs), "fingerprint")
    b, w = inputs.difference_map, inputs.inverse_mass
    ell, e = inputs.delivered_impulse, inputs.impulse_uncertainty
    m, n = len(b), len(w)
    bt, im, inn = _transpose(b), _identity(m), _identity(n)
    _positive_ldl(w, budget)
    _, pivots = _rref(bt, budget)
    rank = len(pivots)
    if rank == 0:
        raise QuotientUnresolved("zero-rank")
    _require_equal(c.rank, rank, "rank")
    nullity = m - rank
    if nullity:
        _, null_pivots = _rref(c.nullspace_basis, budget)
        _require_equal(len(null_pivots), nullity, "nullspace")
        _require_equal(_mm(bt, c.nullspace_basis, budget), _zero(n, nullity), "nullspace")
    else:
        _require_equal(c.nullspace_basis, _zero(m, 0), "nullspace")

    a = _mm(_mm(b, w, budget), bt, budget)
    _require_equal(c.response_matrix, a, "response")
    pn = c.nullspace_projector
    _require_equal(pn, _transpose(pn), "projector")
    _require_equal(_mm(pn, pn, budget), pn, "projector")
    _require_equal(_mm(pn, b, budget), _zero(m, n), "projector")
    _require_equal(_mm(a, pn, budget), _zero(m, m), "projector")
    if nullity:
        _require_equal(_mm(pn, c.nullspace_basis, budget), c.nullspace_basis, "projector")
    else:
        _require_equal(pn, _zero(m, m), "projector")
    # Identity on a complete nullspace plus annihilation of range(B), checked
    # above, characterizes its projector independently of the producer's basis.
    pr = _combine(im, pn, budget.sub)
    scale = _norm_inf(a, budget)
    if scale <= _ZERO:
        raise QuotientUnresolved("scale")
    _require_equal(c.response_scale, scale, "scale")
    reciprocal = budget.div(_ONE, scale)
    h = _combine(_scale(a, reciprocal, budget), pn, budget.add)
    _require_equal(c.gauged_response, h, "gauge")
    _positive_ldl(h, budget)
    hi = c.gauged_inverse
    _require_equal(_mm(h, hi, budget), im, "inverse")
    _require_equal(_mm(hi, h, budget), im, "inverse")
    condition = budget.mul(_norm_inf(h, budget), _norm_inf(hi, budget))
    if condition >= _CONDITION_CEILING:
        raise QuotientUnresolved("condition")
    _require_equal(c.condition_number, condition, "condition")
    pressure_map = _scale(_mm(_mm(hi, b, budget), w, budget), reciprocal, budget)
    _require_equal(c.pressure_map, pressure_map, "pressure-map")
    _require_equal(_mm(pressure_map, bt, budget), pr, "range")

    pressure = _mv(pressure_map, ell, budget)
    represented = _mv(bt, pressure, budget)
    remainder = _vsub(ell, represented, budget)
    _require_equal(c.pressure, pressure, "pressure")
    _require_equal(c.remainder, remainder, "remainder")
    _require_equal(_mv(pn, pressure, budget), tuple(_ZERO for _ in range(m)), "null-pressure")
    _require_equal(
        _mv(b, _mv(w, remainder, budget), budget),
        tuple(_ZERO for _ in range(m)),
        "orthogonality",
    )
    q = _mm(bt, pressure_map, budget)
    _require_equal(_mm(q, q, budget), q, "projection")
    _require_equal(_mm(_transpose(q), w, budget), _mm(w, q, budget), "mass-adjoint")
    _require_equal(
        c.pressure_uncertainty, _mv(_absolute(pressure_map), e, budget), "pressure-error"
    )
    _require_equal(
        c.remainder_uncertainty,
        _mv(_absolute(_combine(inn, q, budget.sub)), e, budget),
        "remainder-error",
    )
    full_energy = _dot(ell, _mv(w, ell, budget), budget)
    represented_energy = _dot(represented, _mv(w, represented, budget), budget)
    residual_energy = _dot(remainder, _mv(w, remainder, budget), budget)
    if residual_energy < _ZERO:
        raise QuotientUnresolved("energy")
    _require_equal(full_energy, budget.add(represented_energy, residual_energy), "energy")


def certify_contact_quotient(
    difference_map,
    inverse_mass,
    delivered_impulse,
    impulse_uncertainty,
    *,
    owner_ids,
    coordinate_ids,
    limits=QuotientProofLimits(),
) -> ContactQuotientCertificate:
    """Produce exact fixed-map evidence, without accepting physical contact.

    Every owner row survives. A zero-rank map is unsupported even when unloaded.
    Negative representative components and off-range remainders remain visible.
    All construction and final identity checking share one bounded counter.
    """
    inputs = _inputs(
        difference_map,
        inverse_mass,
        delivered_impulse,
        impulse_uncertainty,
        owner_ids,
        coordinate_ids,
        limits,
    )
    budget = _Budget(inputs.limits)
    b, w = inputs.difference_map, inputs.inverse_mass
    ell, e = inputs.delivered_impulse, inputs.impulse_uncertainty
    m, n = len(b), len(w)
    _positive_ldl(w, budget)
    rank, basis = _nullspace(b, budget)
    bt = _transpose(b)
    a = _mm(_mm(b, w, budget), bt, budget)
    if rank < m:
        nt = _transpose(basis)
        gram = _mm(nt, basis, budget)
        pn = _mm(_mm(basis, _inverse(gram, budget), budget), nt, budget)
    else:
        pn = _zero(m, m)
    scale = _norm_inf(a, budget)
    if scale <= _ZERO:
        raise QuotientUnresolved("scale")
    reciprocal = budget.div(_ONE, scale)
    h = _combine(_scale(a, reciprocal, budget), pn, budget.add)
    _positive_ldl(h, budget)
    hi = _inverse(h, budget)
    condition = budget.mul(_norm_inf(h, budget), _norm_inf(hi, budget))
    if condition >= _CONDITION_CEILING:
        raise QuotientUnresolved("condition")
    pressure_map = _scale(_mm(_mm(hi, b, budget), w, budget), reciprocal, budget)
    pressure = _mv(pressure_map, ell, budget)
    remainder = _vsub(ell, _mv(bt, pressure, budget), budget)
    q = _mm(bt, pressure_map, budget)
    certificate = ContactQuotientCertificate(
        owner_ids=inputs.owner_ids,
        coordinate_ids=inputs.coordinate_ids,
        difference_map=b,
        inverse_mass=w,
        delivered_impulse=ell,
        impulse_uncertainty=e,
        limits=inputs.limits,
        input_fingerprint=_fingerprint(inputs),
        rank=rank,
        nullspace_basis=basis,
        response_matrix=a,
        nullspace_projector=pn,
        response_scale=scale,
        gauged_response=h,
        gauged_inverse=hi,
        pressure_map=pressure_map,
        pressure=pressure,
        remainder=remainder,
        pressure_uncertainty=_mv(_absolute(pressure_map), e, budget),
        remainder_uncertainty=_mv(_absolute(_combine(_identity(n), q, budget.sub)), e, budget),
        condition_number=condition,
    )
    _verify_evidence(certificate, inputs, budget)
    return certificate


def verify_contact_quotient(
    certificate,
    difference_map,
    inverse_mass,
    delivered_impulse,
    impulse_uncertainty,
    *,
    owner_ids,
    coordinate_ids,
    limits=QuotientProofLimits(),
) -> bool:
    """Independently check bounded submitted evidence against external inputs.

    This never invokes the producer or requires its specific nullspace basis.
    No certificate arithmetic count is trusted. Failure raises QuotientUnresolved;
    True means the exact algebra only, never unilateral/Coulomb/physical admission.
    """
    inputs = _inputs(
        difference_map,
        inverse_mass,
        delivered_impulse,
        impulse_uncertainty,
        owner_ids,
        coordinate_ids,
        limits,
    )
    if type(certificate) is not ContactQuotientCertificate:
        raise QuotientUnresolved("certificate")
    try:
        submitted = {
            item.name: getattr(certificate, item.name)
            for item in fields(ContactQuotientCertificate)
        }
    except AttributeError as error:
        raise QuotientUnresolved("certificate") from error
    # Check a possibly corrupted frozen policy before using it to bound any copy.
    _require_equal(_own_limits(submitted["limits"]), inputs.limits, "binding")
    owned = ContactQuotientCertificate(**submitted)
    budget = _Budget(inputs.limits)
    _verify_evidence(owned, inputs, budget)
    return True
