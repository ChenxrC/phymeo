"""Independent exact algebra controls, authored and not executed."""

from fractions import Fraction as F

import pytest

from phymeo.constraints.conforming_shell import ShellProofLimits, ShellUnresolved
from phymeo.constraints.shell_boundary import BoundaryProofLimits
from phymeo.geometry._shell_contact_field import FieldArithmetic, Meter


def field(generators=(F(2), F(5)), limits=None):
    limits = BoundaryProofLimits() if limits is None else limits
    meter = Meter(limits.strict_limits)
    return FieldArithmetic(generators, meter, limits)


def test_independent_two_root_inverse_zero_and_sign():
    a = field()
    r2 = a.element((F(0), F(1), F(0), F(0)))
    r5 = a.element((F(0), F(0), F(1), F(0)))
    u = a.add(r2, r5)
    inverse = a.scale(F(1, 3), a.sub(r5, r2))
    assert a.mul(u, inverse) == a.one
    assert a.mul(a.inverse(u), u) == a.one
    assert a.sub(a.mul(r2, r2), a.rational(F(2))) == a.zero
    assert a.sign(a.sub(r5, r2)) == 1
    assert a.sign(a.sub(r2, r5)) == -1
    assert a.sign(a.zero) == 0
    enclosure = a.enclose(u)
    assert enclosure.lo > F(3) and enclosure.hi < F(4)


def test_complete_rank_two_primitive_charges_even_zero_terms():
    a = field()
    before = a.meter.operations
    a.add(a.one, a.zero)
    assert a.meter.operations - before == 4
    before = a.meter.operations
    a.mul(a.one, a.zero)
    assert a.meter.operations - before == 39
    before = a.meter.operations
    a.neg(a.zero)
    assert a.meter.operations - before == 4


def test_static_polynomial_identity_endpoint_factor_and_rational_denominator():
    a = field(())
    p = (a.rational(F(1)), a.rational(F(-2)), a.one)
    q = (a.one, a.rational(F(-1)))
    assert a.poly_mul(q, q) == p
    assert a.poly_evaluate(p, F(1)) == a.zero
    assert a.polynomial_bounds(p, F(0), F(1)).lo == 0
    ratio = a.rational_function((a.one,), (a.one, a.one))
    interval = a.function_bounds(ratio, F(0), F(1))
    assert interval.lo <= F(1, 2) and interval.hi >= 1
    with pytest.raises(ShellUnresolved):
        a.function_bounds(a.rational_function((a.one,), q), F(0), F(1))


def test_monomial_sqrt_reconstruction_and_branch():
    a = field()
    root = a.root_from_dependency(F(20), F(2), 2)
    assert a.mul(root, root) == a.rational(F(20))
    assert a.sign(root) == 1
    with pytest.raises(ShellUnresolved):
        a.root_from_dependency(F(20), F(-2), 2)
    with pytest.raises(ShellUnresolved):
        a.root_from_dependency(F(20), F(1), 2)


@pytest.mark.parametrize("case", ("operations", "bits", "degree", "rank", "precision", "singular"))
def test_all_resource_and_expression_failures_are_explicit(case):
    if case == "operations":
        a = field(
            limits=BoundaryProofLimits(strict_limits=ShellProofLimits(max_scalar_operations=1))
        )
        with pytest.raises(ShellUnresolved):
            a.mul(a.one, a.one)
    elif case == "bits":
        a = field((), BoundaryProofLimits(strict_limits=ShellProofLimits(max_fraction_bits=16)))
        with pytest.raises(ShellUnresolved):
            a.rational(F(1 << 17))
    elif case == "degree":
        a = field(())
        with pytest.raises(ShellUnresolved):
            a.poly_mul((a.one,) * 6, (a.one,) * 5)
    elif case == "rank":
        with pytest.raises(ShellUnresolved):
            field(tuple(map(F, (2, 3, 5, 7, 11))))
    elif case == "precision":
        a = field(
            (F(2),),
            BoundaryProofLimits(
                strict_limits=ShellProofLimits(initial_sqrt_bits=1, max_sqrt_bits=1)
            ),
        )
        value = a.sub(a.element((F(0), F(1))), a.rational(F(7, 5)))
        with pytest.raises(ShellUnresolved):
            a.sign(value)
    else:
        a = field()
        with pytest.raises(ShellUnresolved):
            a.inverse(a.zero)


def test_coercive_types_and_widened_limits_reject():
    with pytest.raises((TypeError, ValueError)):
        BoundaryProofLimits(max_field_rank=True)
    with pytest.raises((TypeError, ValueError)):
        BoundaryProofLimits(max_field_rank=5)
    a = field()
    for value in (True, 1.0, "1"):
        with pytest.raises((TypeError, ValueError, ShellUnresolved)):
            a.rational(value)


def test_intermediate_degree_and_expression_caps_apply_before_cancellation():
    a = field((), BoundaryProofLimits(max_polynomial_degree=2))
    p = (a.one, a.one, a.one)
    with pytest.raises(ShellUnresolved):
        a.poly_mul(p, p)
    # A numerator product exceeding the cap is not legalized by a subsequent
    # subtraction of the same expression in a rational-function equality.
    function = a.rational_function(p, p)
    with pytest.raises(ShellUnresolved):
        a.function_equal(function, function)
    with pytest.raises(ShellUnresolved):
        field((F(2), F(5)), BoundaryProofLimits(max_expression_nodes=2))


def test_rational_unary_minus_and_polynomial_subtraction_are_metered():
    a = field(())
    start = a.meter.operations
    assert a.meter.qneg(F(0)) == 0
    assert a.meter.operations == start + 1
    start = a.meter.operations
    assert a.meter.poly_sub((F(3),), (F(1),)) == (F(2),)
    assert a.meter.operations == start + 2  # qneg and qadd, including a zero case.
