"""Hand-derived affine paths frozen in the lane report before execution."""

from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest

from phymeo.geometry._prism_pair_exact import PrismPairProofLimits
from phymeo.geometry.prism_prism_manifold import PrismPairIdentity
from phymeo.geometry.prism_prism_sweep import prism_prism_sweep, verify_prism_prism_sweep

LIMITS = PrismPairProofLimits()
RHO = np.array([0.25, 0.25], dtype=np.float64)


def identity(swapped=False):
    faces = np.array([11, 29], dtype=np.int64)
    nodes = np.array([[2, 7, 9], [12, 17, 19]], dtype=np.int64)
    return PrismPairIdentity(
        "cloth", faces[::-1] if swapped else faces, nodes[::-1] if swapped else nodes
    )


def safe_path_one():
    a = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        ]
    )
    b = np.array(
        [
            [[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]],
            [[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 4.0]],
        ]
    )
    return a, b, RHO.copy()


def safe_path_two():
    a = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        ]
    )
    b = np.array(
        [
            [[0.0, 0.0, 8.0], [1.0, 0.0, 8.0], [0.0, 1.0, 9.0]],
            [[1.0, 0.0, 8.0], [2.0, 0.0, 8.0], [1.0, 1.0, 7.0]],
        ]
    )
    return a, b, np.array([0.125, 0.25], dtype=np.float64)


def split_required_path():
    a = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )
    return a, a + np.array([0.0, 0.0, 5.0]), RHO.copy()


def run(a, b, rho, *, ids=None, limits=LIMITS, budget=0.0, h=0.125):
    return prism_prism_sweep(
        a,
        b,
        identity=identity() if ids is None else ids,
        half_thickness_m=rho,
        h=h,
        penetration_budget_m=budget,
        proof_limits=limits,
    )


def verify(certificate, a, b, rho, *, ids=None, limits=LIMITS, budget=0.0, h=0.125):
    return verify_prism_prism_sweep(
        certificate,
        a,
        b,
        identity=identity() if ids is None else ids,
        half_thickness_m=rho,
        h=h,
        penetration_budget_m=budget,
        proof_limits=limits,
    )


def assert_complete_partition(certificate):
    assert certificate.status == "certified"
    assert not certificate.pending and certificate.obligation is None
    cursor = F(0)
    for cell in certificate.segments:
        assert cell.start == cursor and cell.end > cursor
        assert cell.gap_lower >= 0
        assert cell.area_a.norm_lower > 0 and cell.area_b.norm_lower > 0
        cursor = cell.end
    assert cursor == 1


@pytest.mark.parametrize("fixture", [safe_path_one, safe_path_two])
def test_two_independent_nontrivial_moving_safe_paths_certify(fixture):
    a, b, rho = fixture()
    result = run(a, b, rho)
    assert_complete_partition(result)
    assert verify(result, a, b, rho)
    assert all(all(term > 0 for term in cell.normal_contributions) for cell in result.segments)
    # The report proves stronger actual vertical lower bounds 5/2 and45/8.
    # A certificate may be more conservative; no produced value is the oracle.


def test_two_curvature_terms_and_lower_inequality_are_independently_reconstructed():
    a, b, rho = safe_path_one()
    result = run(a, b, rho)
    assert_complete_partition(result)
    for cell in result.segments:
        for area in (cell.area_a, cell.area_b):
            # c'=unit vector and c''=0 for these explicitly written paths.
            assert area.derivative_norm_squared_upper == 1
            assert area.second_derivative_norm_upper == 0
            assert area.curvature_upper == 3 / (area.norm_lower**2)
            assert area.projection_minimum > 0
            assert (
                area.norm_lower**2 * sum(v * v for v in area.projection_direction)
                <= area.projection_minimum**2
            )
        expected = (F(1, 4) * cell.area_a.curvature_upper, F(1, 4) * cell.area_b.curvature_upper)
        assert cell.normal_contributions == expected
        norm_squared = sum(v * v for v in cell.canonical_direction)
        # curvature/norm is exactly the sum; outward norm may exceed true norm.
        assert cell.curvature_upper**2 >= norm_squared * sum(expected) ** 2
        assert (
            cell.unnormalized_lower
            == min(cell.endpoint_gap_lowers)
            - cell.curvature_upper * (cell.end - cell.start) ** 2 / 8
        )


@pytest.mark.parametrize("omitted", [0, 1])
def test_independent_verifier_rejects_either_missing_curvature_contribution(omitted):
    a, b, rho = safe_path_one()
    result = run(a, b, rho)
    assert verify(result, a, b, rho)
    cell = result.segments[0]
    contributions = list(cell.normal_contributions)
    contributions[omitted] = F(0)
    norm_upper = cell.curvature_upper / sum(cell.normal_contributions)
    curvature = norm_upper * sum(contributions)
    lower = min(cell.endpoint_gap_lowers) - curvature * (cell.end - cell.start) ** 2 / 8
    forged = replace(
        cell,
        normal_contributions=tuple(contributions),
        curvature_upper=curvature,
        unnormalized_lower=lower,
        gap_lower=lower / norm_upper,
    )
    assert not verify(replace(result, segments=(forged,) + result.segments[1:]), a, b, rho)


def test_endpoints_separated_but_interior_crossing_is_not_certified():
    triangle = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    a = np.stack((triangle, triangle))
    b = np.stack((triangle + [0.0, 0.0, -2.0], triangle + [0.0, 0.0, 2.0]))
    result = run(a, b, RHO)
    assert result.status == "violation"
    assert result.witness_time == F(1, 2)
    assert result.witness_manifold.sat_complete
    assert result.witness_manifold.exact_gap_bounds == (-F(1, 2), -F(1, 2))
    assert not verify(result, a, b, RHO)


def test_both_rotating_normals_have_safe_endpoints_and_unsafe_midpoint():
    a = safe_path_two()[0]
    b = a + np.array([0.0, 0.0, 18.0])
    rho = np.array([10.0, 10.0])
    result = run(a, b, rho)
    assert result.status == "violation"
    assert result.witness_time == F(1, 2)
    lo, hi = result.witness_manifold.exact_gap_bounds
    # The hand-derived complete SAT maximum is -1/sqrt(2).
    assert lo < 0 and hi < 0 and lo * lo >= F(1, 2) >= hi * hi
    assert not verify(result, a, b, rho)


def test_area_collapse_rejects_despite_spatial_separation():
    a = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ]
    )
    b0 = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]])
    b = np.stack((b0, b0))
    result = run(a, b, RHO)
    assert result.status == "unresolved"
    assert result.witness_time == F(1, 2)
    assert "degeneracy" in result.obligation.reason
    assert [(p.start, p.end) for p in result.pending] == [(0, 1)]
    assert not verify(result, a, b, RHO)


def test_exact_contact_without_epsilon_or_chord_substitution():
    triangle = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    a = np.stack((triangle, triangle))
    b = a + [0.0, 0.0, 0.5]
    result = run(a, b, RHO)
    assert_complete_partition(result)
    assert len(result.segments) == 1
    assert result.segments[0].gap_lower == 0
    assert result.segments[0].normal_contributions == (0, 0)
    assert verify(result, a, b, RHO)


def test_budget_exhaustion_preserves_both_pending_siblings():
    a, b, rho = split_required_path()
    limits = PrismPairProofLimits(max_time_cells=1)
    result = run(a, b, rho, limits=limits)
    assert result.status == "unresolved" and result.checked_cells == 1
    assert [(p.start, p.end) for p in result.pending] == [(0, F(1, 2)), (F(1, 2), 1)]
    assert not result.segments
    assert not verify(result, a, b, rho, limits=limits)
    complete = run(a, b, rho)
    assert_complete_partition(complete)
    assert len(complete.segments) == 2
    assert verify(complete, a, b, rho)


def test_verifier_rejects_incomplete_partition_altered_bindings_and_claims():
    a, b, rho = split_required_path()
    result = run(a, b, rho)
    assert verify(result, a, b, rho)
    assert not verify(replace(result, segments=result.segments[:1]), a, b, rho)
    assert not verify(replace(result, segments=tuple(reversed(result.segments))), a, b, rho)
    assert not verify(replace(result, checked_cells=1), a, b, rho)
    assert not verify(replace(result, fingerprint="unbound"), a, b, rho)
    assert not verify(result, a, b + [0.0, 0.0, 1.0], rho)
    assert not verify(result, a, b, rho, h=0.25)
    assert not verify(result, a, b, rho, budget=0.125)
    assert not verify(result, a, b, rho, limits=PrismPairProofLimits(max_time_depth=31))
    cell = result.segments[0]
    bad_area = replace(cell.area_b, curvature_upper=F(0), norm_lower=F(1000))
    assert not verify(
        replace(result, segments=(replace(cell, area_b=bad_area),) + result.segments[1:]), a, b, rho
    )


def test_caller_exchange_and_cyclic_path_reorder_preserve_canonical_evidence():
    a, b, rho = safe_path_two()
    forward = run(a, b, rho)
    backward = run(b, a, rho[::-1], ids=identity(True))
    assert forward.fingerprint == backward.fingerprint
    assert forward.caller_fingerprint != backward.caller_fingerprint
    assert forward.area_coefficients == backward.area_coefficients
    assert forward.caller_to_canonical == (0, 1) and backward.caller_to_canonical == (1, 0)
    assert verify(backward, b, a, rho[::-1], ids=identity(True))
    for left, right in zip(forward.segments, backward.segments, strict=True):
        assert left.canonical_direction == right.canonical_direction
        assert left.direction == tuple(-v for v in right.direction)
    ids = PrismPairIdentity(
        "cloth",
        np.array([11, 29], dtype=np.int64),
        np.array([[7, 9, 2], [12, 17, 19]], dtype=np.int64),
    )
    reordered = run(a[:, [1, 2, 0]], b, rho, ids=ids)
    assert reordered.fingerprint == forward.fingerprint
    assert verify(reordered, a[:, [1, 2, 0]], b, rho, ids=ids)


def test_sweep_owns_receipts_and_shared_node_endpoint_correspondence():
    a, b, rho = safe_path_one()
    result = run(a, b, rho)
    fingerprint = result.fingerprint
    a[:] = 100
    rho[:] = 100
    assert result.fingerprint == fingerprint
    shared = PrismPairIdentity(
        "cloth", np.array([1, 2], dtype=np.int64), np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int64)
    )
    with pytest.raises(ValueError, match="shared node"):
        run(a, b, rho, ids=shared)


@pytest.mark.parametrize(
    "h,budget",
    [(0.0, 0.0), (-1.0, 0.0), (float("nan"), 0.0), (1.0, -0.1), (1.0, float("inf")), (True, 0.0)],
)
def test_invalid_duration_or_budget_reject(h, budget):
    a, b, rho = safe_path_one()
    with pytest.raises((ValueError, TypeError)):
        run(a, b, rho, h=h, budget=budget)


def test_invalid_path_shape_dtype_and_nonfinite_reject():
    a, b, rho = safe_path_one()
    for bad in (a[0], a.astype(np.float32), np.full_like(a, np.nan)):
        with pytest.raises((ValueError, TypeError)):
            run(bad, b, rho)


def test_explicit_penetration_budget_is_signed_and_fingerprinted():
    triangle = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    a = np.stack((triangle, triangle))
    result = run(a, a.copy(), RHO, budget=0.5)
    assert result.status == "certified"
    assert result.segments[0].gap_lower == -F(1, 2)
    assert verify(result, a, a.copy(), RHO, budget=0.5)
    assert not verify(result, a, a.copy(), RHO, budget=0.0)


def test_negative_single_axis_is_not_a_violation_witness():
    triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    a = np.stack((triangle, triangle))
    b = a + [3.0, 0.0, 0.0]
    result = run(a, b, RHO)
    assert_complete_partition(result)
    assert verify(result, a, b, RHO)
