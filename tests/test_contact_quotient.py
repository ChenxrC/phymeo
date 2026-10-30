"""Independent exact oracles; authored code-first, execution initially deferred."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import pytest

from phymeo.validation import contact_quotient as quotient
from phymeo.validation.contact_quotient import (
    ContactQuotientCertificate,
    QuotientProofLimits,
    QuotientUnresolved,
    certify_contact_quotient,
    verify_contact_quotient,
)


def issue(b, w, ell, e=None, *, owners=None, coordinates=None, limits=None):
    """Supply IDs only; expected algebra is always written explicitly below."""
    if e is None:
        e = [0] * len(ell)
    kwargs = {
        "owner_ids": owners or [f"owner-{i}" for i in range(len(b))],
        "coordinate_ids": coordinates or [f"coordinate-{i}" for i in range(len(ell))],
    }
    if limits is not None:
        kwargs["limits"] = limits
    return certify_contact_quotient(b, w, ell, e, **kwargs)


def check(certificate, **overrides):
    values = {
        "difference_map": certificate.difference_map,
        "inverse_mass": certificate.inverse_mass,
        "delivered_impulse": certificate.delivered_impulse,
        "impulse_uncertainty": certificate.impulse_uncertainty,
        "owner_ids": certificate.owner_ids,
        "coordinate_ids": certificate.coordinate_ids,
        "limits": certificate.limits,
    }
    values.update(overrides)
    return verify_contact_quotient(certificate, **values)


def duplicate():
    return issue(((1,), (1,)), ((1,),), (2,), (F(1, 4),))


def forged(certificate, field, value):
    # Simulate externally corrupted frozen evidence, not the supported mutation API.
    result = replace(certificate)
    object.__setattr__(result, field, value)
    return result


def test_full_rank_diagonal_independent_oracle():
    c = issue(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((1, 0, 0), (0, 2, 0), (0, 0, 4)),
        (1, -2, 3),
        (F(1, 16), F(1, 8), F(1, 4)),
    )
    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert isinstance(c, ContactQuotientCertificate)
    assert c.schema == "exact-contact-quotient-v1"
    assert c.status == "exact_algebra_certified"
    assert c.rank == 3
    assert c.nullspace_basis == ((), (), ())
    assert c.pressure_map == identity
    assert c.pressure == (1, -2, 3)
    assert c.remainder == (0, 0, 0)
    assert c.pressure_uncertainty == (F(1, 16), F(1, 8), F(1, 4))
    assert c.remainder_uncertainty == (0, 0, 0)
    assert c.condition_number == 4
    assert check(c) is True


def test_duplicate_rows_preserve_both_owners_and_exact_projector():
    c = duplicate()
    assert c.owner_ids == ("owner-0", "owner-1")
    assert c.difference_map == ((1,), (1,))
    assert c.rank == 1
    assert c.nullspace_projector == ((F(1, 2), -F(1, 2)), (-F(1, 2), F(1, 2)))
    assert c.response_scale == 2
    assert c.gauged_response == ((1, 0), (0, 1))
    assert c.gauged_inverse == ((1, 0), (0, 1))
    assert c.pressure_map == ((F(1, 2),), (F(1, 2),))
    assert c.pressure == (1, 1)
    assert c.remainder == (0,)
    assert c.pressure_uncertainty == (F(1, 8), F(1, 8))
    assert c.remainder_uncertainty == (0,)
    assert check(c)


def test_opposed_rows_do_not_fit_positive_self_stress():
    c = issue(((1,), (-1,)), ((1,),), (2,))
    assert c.pressure == (1, -1)
    assert c.nullspace_projector == ((F(1, 2), F(1, 2)), (F(1, 2), F(1, 2)))
    for external_proposal in ((2, 0), (12, 10)):
        assert external_proposal[0] - external_proposal[1] == 2
        assert issue(((1,), (-1,)), ((1,),), (2,)) == c
    with pytest.raises(TypeError):
        certify_contact_quotient(
            ((1,), (-1,)), ((1,),), (2,), (0,),
            owner_ids=("a", "b"), coordinate_ids=("x",), proposal_pressure=(12, 10),
        )
    assert check(c)


def test_off_range_remainder_and_uncertainty_are_retained():
    c = issue(((1, 0),), ((1, 0), (0, 1)), (2, 3), (F(1, 4), F(1, 8)))
    assert c.pressure_map == ((1, 0),)
    assert c.pressure == (2,)
    assert c.remainder == (0, 3)
    assert c.pressure_uncertainty == (F(1, 4),)
    assert c.remainder_uncertainty == (0, F(1, 8))
    assert (c.remainder[0] ** 2 + c.remainder[1] ** 2) / 2 == F(9, 2)
    assert check(c)


def test_full_inverse_mass_cross_terms_and_quadratic_identity():
    c = issue(((1, 0),), ((2, 1), (1, 2)), (0, 2), (F(1, 4), F(1, 8)))
    assert c.response_matrix == ((2,),)
    assert c.pressure_map == ((1, F(1, 2)),)
    assert c.pressure == (1,)
    assert c.remainder == (-1, 2)
    assert c.pressure_uncertainty == (F(5, 16),)
    assert c.remainder_uncertainty == (F(1, 16), F(1, 8))
    # Hand-expanded independent identity: 8 = 2 + (2 - 4 + 8).
    assert 8 == 2 + 2 * c.remainder[0] ** 2 + 2 * c.remainder[0] * c.remainder[1] + 2 * c.remainder[1] ** 2
    assert check(c)
    with pytest.raises(QuotientUnresolved):
        check(c, inverse_mass=((2, 0), (0, 2)))


@pytest.mark.parametrize("k", [10**12, 10**12 + 1], ids=["equality", "above"])
def test_strict_condition_ceiling_rejects(k):
    with pytest.raises(QuotientUnresolved, match="condition"):
        issue(((1, 0), (0, 1)), ((1, 0), (0, F(1, k))), (0, 0))


def test_condition_immediately_below_ceiling_succeeds():
    k = 10**12 - 1
    c = issue(((1, 0), (0, 1)), ((1, 0), (0, F(1, k))), (0, 0))
    assert c.condition_number == k
    assert check(c)


@pytest.mark.parametrize("scale", [F(1, 7), F(8)], ids=["fractional", "integer"])
def test_uniform_inverse_mass_scaling_preserves_quotient(scale):
    c = issue(((1, 0),), ((2 * scale, scale), (scale, 2 * scale)), (0, 2))
    assert c.gauged_response == ((1,),)
    assert c.pressure_map == ((1, F(1, 2)),)
    assert c.pressure == (1,)
    assert c.remainder == (-1, 2)
    assert check(c)


def test_nontrivial_owner_and_coordinate_permutations_with_ids():
    original = issue(
        ((1, 0), (1, 0), (0, 1)), ((2, 0), (0, 3)), (4, -2),
        owners=("a", "b", "c"), coordinates=("x", "y"),
    )
    permuted = issue(
        ((1, 0), (0, 1), (0, 1)), ((3, 0), (0, 2)), (-2, 4),
        owners=("c", "a", "b"), coordinates=("y", "x"),
    )
    assert original.pressure == (2, 2, -2)
    assert permuted.pressure == (-2, 2, 2)
    assert permuted.pressure_map == ((1, 0), (0, F(1, 2)), (0, F(1, 2)))
    assert original.condition_number == permuted.condition_number == F(4, 3)
    assert original.input_fingerprint != permuted.input_fingerprint
    assert check(original) and check(permuted)


def test_verifier_accepts_complete_scaled_noncanonical_nullspace():
    c = duplicate()
    changed = replace(c, nullspace_basis=((-3,), (3,)))
    assert changed.nullspace_basis != c.nullspace_basis
    assert check(changed)


def test_verifier_is_independent_of_producer(monkeypatch):
    c = duplicate()

    def forbidden(*args, **kwargs):
        raise AssertionError("verifier called producer")

    monkeypatch.setattr(quotient, "certify_contact_quotient", forbidden)
    assert check(c) is True
    assert check(c) is True  # A fresh budget and verification on every invocation.


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rank", 0),
        ("nullspace_basis", ((), ())),
        ("nullspace_basis", ((1,), (1,))),
        ("response_matrix", ((1, 1), (1, 2))),
        ("nullspace_projector", ((0, 0), (0, 0))),
        ("response_scale", F(3)),
        ("gauged_response", ((2, 0), (0, 1))),
        ("gauged_inverse", ((1, 0), (0, 2))),
        ("pressure_map", ((1,), (0,))),
        ("pressure", (2, 0)),
        ("remainder", (1,)),
        ("pressure_uncertainty", (0, 0)),
        ("remainder_uncertainty", (1,)),
        ("condition_number", F(2)),
        ("input_fingerprint", "0" * 64),
        ("owner_ids", ("owner-1", "owner-0")),
        ("coordinate_ids", ("other",)),
        ("schema", "exact-contact-quotient-v0"),
        ("status", "physical_contact_accepted"),
    ],
    ids=[
        "rank", "incomplete-basis", "wrong-basis", "A", "projector", "scale",
        "H", "inverse", "L", "pressure", "remainder", "pressure-error",
        "remainder-error", "condition", "fingerprint", "owners", "coordinates",
        "schema", "physical-status",
    ],
)
def test_forged_certificate_rejects(field, value):
    c = duplicate()
    bad = forged(c, field, value)
    with pytest.raises(QuotientUnresolved):
        verify_contact_quotient(
            bad, c.difference_map, c.inverse_mass, c.delivered_impulse, c.impulse_uncertainty,
            owner_ids=c.owner_ids, coordinate_ids=c.coordinate_ids,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"difference_map": ((2,), (2,))},
        {"inverse_mass": ((2,),)},
        {"delivered_impulse": (3,)},
        {"impulse_uncertainty": (F(1, 2),)},
        {"owner_ids": ("renamed", "owner-1")},
        {"coordinate_ids": ("renamed",)},
        {"limits": QuotientProofLimits(max_owner_rows=2)},
    ],
    ids=["B", "W", "delivery", "uncertainty", "owner-id", "coordinate-id", "policy"],
)
def test_external_binding_changes_reject(overrides):
    with pytest.raises(QuotientUnresolved):
        check(duplicate(), **overrides)

@pytest.mark.parametrize("scalar", [True, False, 1.0, float("nan"), float("inf"), "1", None])
def test_nonexact_scalar_inputs_reject(scalar):
    with pytest.raises(QuotientUnresolved, match="scalar"):
        issue(((scalar,),), ((1,),), (0,))


@pytest.mark.parametrize(
    ("b", "w", "ell", "e"),
    [
        ((), ((1,),), (0,), (0,)),
        (((1,), (1, 0)), ((1,),), (0,), (0,)),
        (((1, 0),), ((1,),), (0, 0), (0, 0)),
        (((1,),), ((1,),), (0, 0), (0,)),
        (((1,),), ((1,),), (0,), (0, 0)),
        (((1,),), ((1,),), (0,), (-1,)),
        (((1, 0),), ((1, 1), (0, 1)), (0, 0), (0, 0)),
        (((1, 0),), ((1, 1), (1, 1)), (0, 0), (0, 0)),
        (((1,),), ((-1,),), (0,), (0,)),
        (((0,), (0,)), ((1,),), (0,), (0,)),
    ],
    ids=["empty", "ragged", "mass-shape", "impulse-shape", "error-shape",
         "negative-error", "asymmetric", "singular-mass", "negative-mass", "zero-rank"],
)
def test_invalid_shapes_mass_uncertainty_and_zero_rank_reject(b, w, ell, e):
    with pytest.raises(QuotientUnresolved):
        issue(b, w, ell, e)


def test_generators_and_implicit_conversion_are_not_consumed():
    class Coercible:
        def __iter__(self):
            raise AssertionError("implicit iteration")

        def __int__(self):
            raise AssertionError("implicit scalar conversion")

    for invalid in (Coercible(), iter(((1,),)), "1", 1):
        with pytest.raises(QuotientUnresolved):
            certify_contact_quotient(
                invalid, ((1,),), (0,), (0,), owner_ids=("a",), coordinate_ids=("x",)
            )
    with pytest.raises(QuotientUnresolved):
        issue(((Coercible(),),), ((1,),), (0,))


@pytest.mark.parametrize(
    "owner_ids",
    [("a", "a"), ("", "b"), ("a",), ("a", 2), ("a", "x" * 257), ("a", "é" * 129)],
    ids=["duplicate", "empty", "shape", "type", "ascii-bytes", "utf8-bytes"],
)
def test_invalid_owner_identity_rejects(owner_ids):
    with pytest.raises(QuotientUnresolved):
        certify_contact_quotient(
            ((1,), (1,)), ((1,),), (2,), (0,),
            owner_ids=owner_ids, coordinate_ids=("x",),
        )


def test_coordinate_ids_and_utf8_boundary_are_bound():
    c = issue(((1,),), ((1,),), (0,), owners=("é" * 128,), coordinates=("x",))
    assert check(c)
    with pytest.raises(QuotientUnresolved):
        issue(((1, 0),), ((1, 0), (0, 1)), (0, 0), coordinates=("x", "x"))
    with pytest.raises(QuotientUnresolved):
        issue(((1,),), ((1,),), (0,), coordinates=("\ud800",))


@pytest.mark.parametrize(
    "values",
    [
        {"max_owner_rows": 0}, {"max_coordinates": -1}, {"max_fraction_bits": True},
        {"max_scalar_operations": 1.0}, {"max_owner_rows": 193},
        {"max_coordinates": 257}, {"max_fraction_bits": 4097},
        {"max_scalar_operations": 2_000_001},
    ],
    ids=["zero", "negative", "bool", "float", "rows", "columns", "bits", "operations"],
)
def test_invalid_or_expanded_resource_policy_rejects(values):
    with pytest.raises(QuotientUnresolved, match="policy"):
        QuotientProofLimits(**values)


def test_shape_limits_precede_scalar_copying():
    with pytest.raises(QuotientUnresolved, match="shape"):
        issue(((object(),), (object(),)), ((1,),), (0,), limits=QuotientProofLimits(max_owner_rows=1))
    with pytest.raises(QuotientUnresolved, match="shape"):
        issue(((1, 0),), ((1, 0), (0, 1)), (0, 0), limits=QuotientProofLimits(max_coordinates=1))


def test_input_and_intermediate_fraction_bit_exhaustion_reject():
    with pytest.raises(QuotientUnresolved, match="bits"):
        issue(((8,),), ((1,),), (0,), limits=QuotientProofLimits(max_fraction_bits=3))
    # Inputs fit four bits, but A = 3 * 3 * 3 = 27 needs five bits.
    with pytest.raises(QuotientUnresolved, match="bits"):
        issue(((3,),), ((3,),), (0,), limits=QuotientProofLimits(max_fraction_bits=4))
    with pytest.raises(QuotientUnresolved, match="bits"):
        check(forged(duplicate(), "nullspace_basis", ((-(1 << 4096),), (1 << 4096,))))


def test_scalar_operation_exhaustion_is_fail_closed():
    with pytest.raises(QuotientUnresolved, match="operations"):
        issue(((1,),), ((1,),), (0,), limits=QuotientProofLimits(max_scalar_operations=1))


def test_certificate_and_policy_deep_ownership_and_frozen_fields():
    b, w, ell, e = [[1], [1]], [[1]], [2], [F(1, 4)]
    owners, coordinates = ["a", "b"], ["x"]
    c = issue(b, w, ell, e, owners=owners, coordinates=coordinates)
    b[0][0], w[0][0], ell[0], e[0], owners[0], coordinates[0] = 9, 9, 9, 9, "bad", "bad"
    assert c.difference_map == ((F(1),), (F(1),))
    assert c.inverse_mass == ((F(1),),)
    assert c.delivered_impulse == (F(2),)
    assert c.impulse_uncertainty == (F(1, 4),)
    assert c.owner_ids == ("a", "b") and c.coordinate_ids == ("x",)
    assert type(c.difference_map[0][0]) is F
    derived = [[-3], [3]]
    copied = replace(c, nullspace_basis=derived)
    derived[0][0] = 99
    assert copied.nullspace_basis == ((F(-3),), (F(3),))
    assert check(copied)
    with pytest.raises(FrozenInstanceError):
        c.rank = 2
    with pytest.raises(FrozenInstanceError):
        c.limits.max_owner_rows = 1


def test_equivalent_fraction_spellings_have_identical_binding():
    a = issue([[1], [1]], [[1]], [2], [F(1, 4)])
    b = issue(((F(2, 2),), (F(3, 3),)), ((F(5, 5),),), (F(4, 2),), (F(2, 8),))
    assert a.input_fingerprint == b.input_fingerprint
    assert a == b
    assert check(a) and check(b)

def test_incomplete_nullspace_with_valid_forged_shapes_rejects():
    c = issue(((1, 0), (1, 0), (1, 0)), ((1, 0), (0, 1)), (6, 3))
    bad = replace(c, rank=2, nullspace_basis=((-1,), (1,), (0,)))
    with pytest.raises(QuotientUnresolved, match="rank"):
        check(bad)


def test_verifier_accepts_mixed_complete_nullspace_basis():
    c = issue(((1, 0), (1, 0), (1, 0)), ((1, 0), (0, 1)), (6, 3))
    alternate = replace(c, nullspace_basis=((-2, -1), (1, 2), (1, -1)))
    assert c.pressure == (2, 2, 2)
    assert c.remainder == (0, 3)
    assert c.gauged_response == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert check(alternate)


def test_verifier_has_a_fresh_metered_budget_each_time(monkeypatch):
    c = duplicate()
    original = quotient._Budget
    counters = []

    class TrackingBudget(original):
        def __init__(self, limits):
            super().__init__(limits)
            counters.append(self)

    monkeypatch.setattr(quotient, "_Budget", TrackingBudget)
    assert check(c)
    assert len(counters) == 1
    assert 0 < counters[0].operations <= c.limits.max_scalar_operations
    first_count = counters[0].operations
    assert check(c)
    assert len(counters) == 2
    assert counters[0] is not counters[1]
    assert counters[1].operations == first_count


@pytest.mark.parametrize("limits", [None, {}, 1], ids=["none", "mapping", "integer"])
def test_nonpolicy_objects_reject(limits):
    with pytest.raises(QuotientUnresolved, match="policy"):
        certify_contact_quotient(
            ((1,),), ((1,),), (0,), (0,),
            owner_ids=("a",), coordinate_ids=("x",), limits=limits,
        )