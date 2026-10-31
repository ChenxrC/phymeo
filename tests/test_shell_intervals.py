"""Unexecuted exact interval controls with independent integer inequalities."""

from dataclasses import replace
from fractions import Fraction as F

import pytest

from phymeo.constraints.conforming_shell import ShellProofLimits, ShellUnresolved
from phymeo.geometry._shell_intervals import Arithmetic, Interval


def test_sqrt_exact_squares_and_nonsquare_integer_bounds():
    a = Arithmetic(ShellProofLimits())
    for value, root in [(F(0), F(0)), (F(4, 9), F(2, 3)), (F(1, 2**200), F(1, 2**100))]:
        result = a.sqrt(a.exact(value))
        assert result == Interval(root, root)
    for value in (F(2), F(3, 7), F(1, 3 * 2**200)):
        result = a.sqrt(a.exact(value))
        assert result.lo**2 <= value <= result.hi**2
        assert result.lo >= 0
        assert result.hi - result.lo <= F(1, 2**64)
    assert a.operations > 0


def test_interval_arithmetic_and_denominator_sign():
    a = Arithmetic(ShellProofLimits())
    x, y = Interval(F(-2), F(3)), Interval(F(4), F(5))
    assert a.add(x, y) == Interval(F(2), F(8))
    assert a.mul(x, y) == Interval(F(-10), F(15))
    assert a.div(x, y) == Interval(F(-1, 2), F(3, 4))
    with pytest.raises(ShellUnresolved, match="division"):
        a.div(y, x)
    with pytest.raises(ShellUnresolved, match="sqrt"):
        a.sqrt(x)


def test_polynomial_bernstein_encloses_interior_extremum():
    a = Arithmetic(ShellProofLimits())
    # t*(1-t) vanishes at both endpoints but reaches 1/4 inside.
    bound = a.polynomial_bounds((F(0), F(1), F(-1)), F(0), F(1))
    assert bound.lo <= 0 and bound.hi >= F(1, 4)
    assert bound.hi > 0
    narrowed = a.polynomial_bounds((F(0), F(1), F(-1)), F(1, 4), F(3, 4))
    assert narrowed.lo <= F(3, 16) and narrowed.hi >= F(1, 4)


def test_publication_has_exact_storage_error_not_correct_rounding_claim():
    a = Arithmetic(ShellProofLimits())
    interval = Interval(F(1, 3), F(1, 3) + F(1, 2**80))
    pub = a.publish((interval,), (1,))
    stored = F.from_float(float(pub.values[0]))
    assert pub.errors == (max(abs(stored - interval.lo), abs(stored - interval.hi)),)
    assert pub.intervals == (interval,)
    with pytest.raises(ValueError):
        pub.values.flags.writeable = True


def test_operation_and_intermediate_bit_budgets_are_enforced():
    a = Arithmetic(replace(ShellProofLimits(), max_scalar_operations=1))
    a.qadd(F(1), F(2))
    with pytest.raises(ShellUnresolved, match="operation"):
        a.qmul(F(1), F(2))
    b = Arithmetic(replace(ShellProofLimits(), max_fraction_bits=128))
    with pytest.raises(ShellUnresolved, match="bits"):
        b.qmul(F(2**100), F(2**100))
    with pytest.raises(ShellUnresolved, match="bits"):
        b.sqrt(b.exact(F(2)))


def test_created_cells_count_failed_parents_and_depth():
    a = Arithmetic(replace(ShellProofLimits(), max_time_cells=2, max_time_depth=1))
    a.visit_cell(0)
    a.visit_cell(1)
    with pytest.raises(ShellUnresolved, match="cells"):
        a.visit_cell(1)
    b = Arithmetic(replace(ShellProofLimits(), max_time_depth=1))
    with pytest.raises(ShellUnresolved, match="depth"):
        b.visit_cell(2)


def test_unary_fraction_negation_and_negative_absolute_value_are_metered():
    a = Arithmetic(ShellProofLimits())
    before = a.operations
    assert a.qneg(F(2, 3)) == F(-2, 3)
    assert a.operations == before + 1
    before = a.operations
    assert a.neg(Interval(F(-2), F(3))) == Interval(F(-3), F(2))
    assert a.operations == before + 2
    before = a.operations
    assert a.qabs(F(-2, 3)) == F(2, 3)
    assert a.operations == before + 1
    before = a.operations
    assert a.poly_sub((F(0),), (F(2), F(-3))) == (F(-2), F(3))
    assert a.operations == before + 4  # two coefficient signs and two additions
    b = Arithmetic(replace(ShellProofLimits(), max_scalar_operations=1))
    assert b.qsub(F(0), F(1)) == -1  # its internal sign is charged once
    with pytest.raises(ShellUnresolved, match="operation"):
        b.qneg(F(0))
