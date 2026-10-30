"""Independent exact identities and deterministic redundant-root cost controls."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

from phymeo.geometry import _prism_exact as exact

Q = exact.Quadratic


def coefficients(v):
    assert all(isinstance(x, F) for x in (v.a, v.b, v.c))
    return v.a, v.b, v.c


@pytest.mark.parametrize(
    "args,expected",
    [
        ((F(1, 3), F(2, 5), F(9, 16)), (F(19, 30), F(0), F(1))),
        ((2, 0, 17), (F(2), F(0), F(1))),
        ((0, 3, 4), (F(6), F(0), F(1))),
        ((1, -2, 2), (F(1), F(-2), F(2))),
    ],
)
def test_public_canonical_forms(args, expected):
    assert coefficients(Q(*args)) == expected


@pytest.mark.parametrize(
    "value",
    [
        0,
        np.int64(-3),
        F(1, 3),
        0.1,
        np.float32(0.1),
        float.fromhex("0x0.0000000000001p-1022"),
        float.fromhex("0x1.fffffffffffffp+1023"),
        "3.5",
    ],
)
def test_public_rational_conversions_and_float_bits(value):
    expected = (
        value
        if isinstance(value, F)
        else F(int(value))
        if isinstance(value, (int, np.integer))
        else F(float(value))
    )
    v = Q(value)
    assert coefficients(v) == (expected, F(0), F(1))
    assert v.bounds() == (expected, expected)
    assert float(v).hex() == float(expected).hex()


@pytest.mark.parametrize("c", [0, -1, F(-1, 2)])
@pytest.mark.parametrize("b", [0, 1])
def test_nonpositive_radicand_rejected_even_when_rational(b, c):
    with pytest.raises(ValueError, match="radicand"):
        Q(1, b, c)


@pytest.mark.parametrize(
    "value,error", [(None, TypeError), (float("inf"), OverflowError), (float("nan"), ValueError)]
)
def test_invalid_public_conversion(value, error):
    with pytest.raises(error):
        Q(value)


def test_independent_quadratic_identities_and_cancellation():
    x, y = Q(1, 2, 2), Q(3, -1, 2)
    assert coefficients(x * y) == (F(-1), F(5), F(2))
    assert coefficients(x / y) == (F(1), F(1), F(2))
    assert coefficients(x + y) == (F(4), F(1), F(2))
    assert coefficients(x - y) == (F(-2), F(3), F(2))
    assert coefficients(-x) == (F(-1), F(-2), F(2))
    assert coefficients(x - x) == (F(0), F(0), F(1))
    assert coefficients(Q(0, 1, 2) ** 2) == (F(2), F(0), F(1))
    assert coefficients(Q(1, 1, 2) ** -1) == (F(-1), F(1), F(2))
    assert coefficients(F(3, 7) * Q(0, 7, 2)) == (F(0), F(3), F(2))
    assert coefficients(Q(0, 7, 2) / F(7, 3)) == (F(0), F(3), F(2))
    assert coefficients(2 - Q(1, 1, 2)) == (F(1), F(-1), F(2))
    assert coefficients(2 / Q(0, 1, 2)) == (F(0), F(1), F(2))
    assert Q(1, 1, 2) ** 0 == Q(1)


@pytest.mark.parametrize(
    "operation",
    [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: x * y,
        lambda x, y: x / y,
        lambda x, y: x < y,
    ],
)
def test_different_nonzero_fields_rejected(operation):
    with pytest.raises(ValueError, match="different quadratic fields"):
        operation(Q(0, 1, 2), Q(0, 1, 3))
    assert not (Q(0, 1, 2) == Q(0, 1, 3))


@pytest.mark.parametrize("v", [Q(0), Q(2, 0, 3), Q(0, 1, 2)])
def test_division_by_zero_and_frozen_storage(v):
    with pytest.raises(ZeroDivisionError):
        v / Q(0)
    with pytest.raises(AttributeError):
        v.a = F(5)


def test_exact_sign_order_hash_and_near_cancellation():
    n = 2**200
    below = F(math.isqrt(2 * n * n), n)
    assert below * below < 2
    v = Q(-below, 1, 2)
    assert v.sign() == 1 and v > 0 and (-v).sign() == -1
    assert Q(1, -1, 2) < 0 < Q(-1, 1, 2)
    assert hash(Q(0, 2, 2)) == hash((F(0), F(2), F(2)))
    assert len({Q(1), Q(0, 1, 1)}) == 1


@pytest.mark.parametrize("a,b", [(F(1, 3), F(2, 7)), (F(-1, 3), F(-2, 7)), (F(0), F(1))])
def test_independent_outward_bounds(a, b):
    lo, hi = Q(a, b, 2).bounds()
    rootlo, roothi = ((lo - a) / b, (hi - a) / b) if b > 0 else ((hi - a) / b, (lo - a) / b)
    assert rootlo * rootlo <= 2 <= roothi * roothi and rootlo > 0
    assert float(Q(0, 1, 2)).hex() == math.sqrt(2).hex()
    flo, fhi = exact.outward_float((F(1, 10), F(1, 10)))
    assert F(flo) <= F(1, 10) <= F(fhi)
    assert np.nextafter(flo, np.inf) == fhi
    assert exact.compare_gaps(-1, 4, -1, 1) == 1
    assert exact.normalized_bounds(Q(-1), Q(4)) == (F(-1, 2), F(-1, 2))


@pytest.mark.parametrize("kind", ["rational", "canonical"])
def test_redundant_root_work_is_eliminated(monkeypatch, kind):
    x, y = Q(1, 2, 2), Q(3, -1, 2)
    calls = {"classification": 0, "bounds": 0}
    oldroot, oldbounds = exact.isqrt, exact.sqrt_bounds

    def root(value):
        calls["classification"] += 1
        return oldroot(value)

    def bounds(value):
        calls["bounds"] += 1
        return oldbounds(value)

    monkeypatch.setattr(exact, "isqrt", root)
    monkeypatch.setattr(exact, "sqrt_bounds", bounds)
    if kind == "rational":
        value = Q(F(3, 7), 0, F(17))
        assert value.bounds() == (F(3, 7), F(3, 7))
    else:
        for v in (x + y, x - y, x * y, x / y, -x, x - x, x * 0, x / 2):
            assert isinstance(v, Q)
    assert calls == {"classification": 0, "bounds": 0}, calls


def test_non_square_fields_are_not_square_factor_normalized():
    assert coefficients(Q(0, 1, 8)) == (F(0), F(1), F(8))
    assert not (Q(0, 2, 2) == Q(0, 1, 8))
    for operation in (
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
    ):
        with pytest.raises(ValueError, match="different quadratic fields"):
            operation(Q(0, 2, 2), Q(0, 1, 8))
