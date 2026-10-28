"""Exact stored-value arithmetic for material audits; no state generation."""

from fractions import Fraction
from numbers import Real

import numpy as np

from ..core.capabilities import SolverFailure
from ..solvers._contact_replay_audit import sqrt_bounds

ZERO = Fraction(0)


def q(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise SolverFailure("material ledger requires finite real values")
    value = float(value)
    if not np.isfinite(value):
        raise SolverFailure("material ledger requires finite real values")
    return Fraction(value)


def array(value, shape, name):
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float64
        or value.shape != shape
        or not np.isfinite(value).all()
    ):
        raise SolverFailure(f"invalid material ledger {name} storage")
    return value


def matrix(value):
    return [[q(v) for v in row] for row in value]


def zero_matrix(count):
    return [[ZERO] * 3 for _ in range(count)]


def compute(value):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.asarray(value, dtype=np.float32).astype(np.float64)
    if not np.isfinite(result).all():
        raise SolverFailure("material input is outside FP32 range")
    return result


def norm_bounds(row):
    return sqrt_bounds(sum((v * v for v in row), ZERO))


def maxabs(bounds):
    return max(abs(bounds[0]), abs(bounds[1]))


def direction_bounds(delta):
    low, high = norm_bounds(delta)
    if high == 0:
        return (low, high), [(Fraction(i == 0), Fraction(i == 0)) for i in range(3)]
    result = []
    for value in delta:
        a, b = value / low, value / high
        result.append((min(a, b), max(a, b)))
    return (low, high), result


def display(value):
    result = float(value)
    if not np.isfinite(result) or (value != 0 and result == 0):
        raise SolverFailure("material report is not representable in FP64")
    return result


def display_array(values):
    source = np.asarray(values, dtype=object)
    return np.array([display(v) for v in source.flat]).reshape(source.shape)


def interval_float(bounds):
    result = []
    for value, direction in zip(bounds, (-np.inf, np.inf), strict=True):
        out = float(value)
        if not np.isfinite(out):
            raise SolverFailure("material report interval overflow")
        if (direction < 0 and q(out) > value) or (direction > 0 and q(out) < value):
            out = float(np.nextafter(out, direction))
        if not np.isfinite(out):
            raise SolverFailure("material report interval overflow")
        result.append(out)
    return tuple(result)


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def elastic(model, constraints, position):
    low = high = ZERO
    if constraints is not None:
        for (i, j), rest, compliance in zip(
            constraints.indices, constraints.rest_length, constraints.compliance, strict=True
        ):
            if compliance == 0:
                continue
            a, b = norm_bounds([position[i][k] - position[j][k] for k in range(3)])
            a, b = a - q(rest), b - q(rest)
            lower = ZERO if a <= 0 <= b else min(a * a, b * b)
            low += lower / (2 * q(compliance))
            high += max(a * a, b * b) / (2 * q(compliance))
    return low, high


def bookkeeping(model, constraints, initial, final, effective, original, expected, load, h):
    x0, x1, v0, v1 = map(
        matrix, (initial.position, final.position, initial.velocity, final.velocity)
    )
    k0 = k1 = g0 = g1 = work = oldwork = requestwork = ZERO
    pchange, external, material, angular, extangular, matangular = ([ZERO] * 3 for _ in range(6))
    impulse = [[v / h for v in row] for row in load]
    support = zero_matrix(len(load))
    for i, inverse in enumerate(model.inverse_mass):
        if inverse == 0:
            support[i] = [-v for v in impulse[i]]
            continue
        w = q(inverse)
        ext = [h * (q(effective.force[i, k]) + q(model.gravity[k]) / w) for k in range(3)]
        for k in range(3):
            k0 += v0[i][k] ** 2 / (2 * w)
            k1 += v1[i][k] ** 2 / (2 * w)
            g0 -= q(model.gravity[k]) * x0[i][k] / w
            g1 -= q(model.gravity[k]) * x1[i][k] / w
            dx = x1[i][k] - x0[i][k]
            work += q(effective.force[i, k]) * dx
            oldwork += q(original.force[i, k]) * dx
            requestwork += expected[i][k] * dx / h
            pchange[k] += (v1[i][k] - v0[i][k]) / w
            external[k] += ext[k]
            material[k] += impulse[i][k]
        before = cross(x0[i], [v / w for v in v0[i]])
        after = cross(x1[i], [v / w for v in v1[i]])
        te, tm = cross(x0[i], ext), cross(x0[i], impulse[i])
        for k in range(3):
            angular[k] += after[k] - before[k]
            extangular[k] += te[k]
            matangular[k] += tm[k]
    u0, u1 = elastic(model, constraints, x0), elastic(model, constraints, x1)
    e0 = (k0 + g0 + u0[0], k0 + g0 + u0[1])
    e1 = (k1 + g1 + u1[0], k1 + g1 + u1[1])
    energy = dict(
        kinetic_before=display(k0),
        kinetic_after=display(k1),
        elastic_before=interval_float(u0),
        elastic_after=interval_float(u1),
        gravity_before=display(g0),
        gravity_after=display(g1),
        total_before=interval_float(e0),
        total_after=interval_float(e1),
        endpoint_work=display(work),
        original_input_work=display(oldwork),
        requested_input_work=display(requestwork),
        input_work_discrepancy=display(work - oldwork - requestwork),
        change_minus_work=interval_float((e1[0] - e0[1] - work, e1[1] - e0[0] - work)),
    )
    momentum = {
        k: display_array(v)
        for k, v in dict(
            linear_change=pchange,
            external_impulse=external,
            material_impulse=material,
            linear_balance_error=[pchange[k] - external[k] - material[k] for k in range(3)],
            angular_change=angular,
            external_angular_impulse=extangular,
            material_angular_impulse=matangular,
            angular_balance_error=[angular[k] - extangular[k] - matangular[k] for k in range(3)],
        ).items()
    }
    return dict(
        material_impulses=display_array(impulse),
        support_impulses=display_array(support),
        energy=energy,
        momentum=momentum,
    )
