"""Independent algebraic equations; no expectations sampled from geometry."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from math import isqrt

import pytest

from phymeo.geometry._prism_pair_exact import (
    PairField,
    PrismPairProofLimits,
    SignUnresolved,
    compare_gaps,
    normalized_bounds,
    sqrt_bounds,
)


@pytest.mark.parametrize("ca,cb,degree", [(4, 9, 1), (2, 8, 2), (F(2, 9), F(8, 25), 2), (2, 3, 4)])
def test_degree_reduction_and_exact_generator_squares(ca, cb, degree):
    field = PairField(ca, cb)
    assert len(field.scalar(0).coefficients) == degree
    assert field.sqrt(ca) ** 2 == ca
    assert field.sqrt(cb) ** 2 == cb


def test_common_radical_cancellation_is_exact_zero():
    field = PairField(2, 8)
    value = field.sqrt(8) - 2 * field.sqrt(2)
    assert value.is_zero
    assert value.sign() == 0
    assert value.bounds() == (0, 0)


def test_independent_roots_inverse_and_cross_term():
    field = PairField(2, 3)
    u, v = field.sqrt(2), field.sqrt(3)
    assert (u + v).inverse() == v - u
    assert (u * v) ** 2 == 6
    assert (1 + u + v + u * v) / (1 + u) == 1 + v
    assert (u + v) ** -1 == v - u


def test_ordered_equivalent_contexts_and_hashes():
    a, b = PairField(2, 3), PairField(3, 2)
    assert a.roots == b.roots
    assert a.sqrt(2) + a.sqrt(3) == b.sqrt(3) + b.sqrt(2)
    assert hash(a.sqrt(6)) == hash(b.sqrt(6))
    assert PairField(8, 2).roots == PairField(2, 8).roots
    assert len({a.scalar(1), 1}) == 1


def test_unrelated_contexts_cannot_silently_coerce_even_rational_elements():
    a, b = PairField(2, 3), PairField(2, 5)
    with pytest.raises(ValueError, match="unrelated"):
        _ = a.scalar(1) + b.scalar(1)
    assert a.scalar(1) + 2 == 3
    with pytest.raises(ValueError, match="outside"):
        a.sqrt(5)


def test_exact_zero_denominator_and_context_ownership():
    field = PairField(2, 3)
    with pytest.raises(ZeroDivisionError):
        _ = 1 / (field.sqrt(2) ** 2 - 2)
    with pytest.raises(FrozenInstanceError):
        field.roots = (F(7),)


def test_narrow_nonzero_sign_around_two_to_minus_1024():
    # r^2 < 2*2^2048 < (r+1)^2; irrational sqrt(2) is strictly above r/2^1024.
    r = isqrt(2 << 2048)
    field = PairField(2, 3)
    positive = field.sqrt(2) - F(r, 1 << 1024)
    assert not positive.is_zero
    assert positive.sign() == 1
    assert (-positive).sign() == -1
    lo, hi = positive.signed_bounds()[1]
    assert 0 < lo <= hi < F(1, 1 << 1024)


def test_sign_budget_exhaustion_is_typed_not_equality():
    field = PairField(2, 3, PrismPairProofLimits(8, 8))
    r = isqrt(2 << 2048)
    value = field.sqrt(2) - F(r, 1 << 1024)
    assert not value.is_zero
    with pytest.raises(SignUnresolved) as caught:
        value.sign()
    assert caught.value.bits == 8
    assert caught.value.value_token == value.token


@pytest.mark.parametrize("value", [F(0), F(2), F(3, 7), F(1, 1 << 2100), F(1 << 2200)])
def test_root_enclosures_are_outward_for_extreme_scales(value):
    low, high = sqrt_bounds(value, 37)
    assert low >= 0 and low * low <= value <= high * high
    if low == high:
        assert low * low == value


def test_normalized_comparison_reverses_negative_squares():
    field = PairField(2, 3)
    assert compare_gaps(field.scalar(-2), F(4), field.scalar(-3), F(4)) > 0
    assert compare_gaps(field.sqrt(2), F(2), field.sqrt(3), F(3)) == 0
    assert compare_gaps(-field.sqrt(2), F(2), -field.sqrt(3), F(3)) == 0
    assert compare_gaps(field.scalar(0), F(2), field.scalar(-1), F(3)) > 0
    lo, hi = normalized_bounds(field.sqrt(2), F(2))
    assert lo <= 1 <= hi


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_sqrt_bits": 0},
        {"max_sqrt_bits": True},
        {"max_time_cells": -1},
        {"max_time_depth": 1.5},
        {"initial_sqrt_bits": 9, "max_sqrt_bits": 8},
    ],
)
def test_invalid_proof_limits_reject(kwargs):
    with pytest.raises(ValueError):
        PrismPairProofLimits(**kwargs)


@pytest.mark.parametrize("ca,cb", [(0, 1), (-1, 2), (float("inf"), 2)])
def test_invalid_field_radicands_reject(ca, cb):
    with pytest.raises((ValueError, TypeError)):
        PairField(ca, cb)
