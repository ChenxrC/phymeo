"""Strict continuous-domain controls, including safe endpoints with bad interiors."""

from fractions import Fraction as F

import numpy as np
import pytest

from phymeo.constraints.bending import BendingConstraints
from phymeo.constraints.cloth_material import ClothMaterial, CompositeMaterialAuditSettings
from phymeo.core.capabilities import SolverFailure
from phymeo.validation.bending_domain import Polynomial, certify_bending_segment
from test_bending_bounds import flat_nodes


def material(rest=0.):
    return ClothMaterial(bending=BendingConstraints(np.array([[0, 1, 2, 3]]), np.array([rest]), np.ones(1)))


def test_bernstein_certificate_bounds_exact_polynomial():
    # (2t-1)^2 has a zero at 1/2; whole-interval Bernstein lower bound is negative.
    polynomial = Polynomial((1, -4, 4))
    bound = polynomial.bounds(F(0), F(1))
    assert bound.low <= 0 and bound.high >= 1
    assert polynomial.bounds(F(0), F(1, 4)).low > 0


def test_translation_segment_has_complete_strict_partition():
    nodes = flat_nodes()
    certificate = certify_bending_segment(nodes, nodes + [3/8, -1/4, 1/8], material())
    assert certificate.status == "certified"
    assert certificate.partition == (("0", "1"),)
    assert not certificate.unresolved


def test_nontrivial_fold_segment_is_certified():
    start = flat_nodes()
    end = start.copy()
    end[0, 2], end[2, 2], end[3, 2] = 1/32, -1/64, -1/64
    proof = certify_bending_segment(start, end, material(0.25))
    assert proof.require().status == "certified"
    cells = [(F(a), F(b)) for a, b in proof.partition]
    assert cells[0][0] == 0 and cells[-1][1] == 1
    assert all(cells[i][1] == cells[i+1][0] for i in range(len(cells)-1))


def test_valid_endpoints_do_not_certify_interior_area_collapse():
    start = flat_nodes()
    end = start.copy()
    end[[2, 3], 1] *= -1
    proof = certify_bending_segment(start, end, material())
    assert proof.status in ("unknown", "rejected")
    with pytest.raises(SolverFailure):
        proof.require()


def test_branch_cut_and_error_branch_reject_even_without_degeneracy():
    start = flat_nodes()
    start[3] = (0., 1., 0.)
    assert certify_bending_segment(start, start, material()).status != "certified"
    start[3] = (0., 0., -1.)
    assert certify_bending_segment(start, start, material(-2.)).status != "certified"


def test_fixed_budget_exhaustion_is_unknown_and_fingerprinted():
    start = flat_nodes()
    end = start.copy()
    end[[2, 3], 1] *= -1
    one = certify_bending_segment(start, end, material(), policy=CompositeMaterialAuditSettings(max_domain_intervals=1))
    two = certify_bending_segment(start, end, material(), policy=CompositeMaterialAuditSettings(max_domain_intervals=2))
    assert one.status == two.status == "unknown"
    assert one.unresolved and one.processed_intervals == 1
    assert one.policy_fingerprint != two.policy_fingerprint


def test_conversion_domain_loss_cannot_be_hidden_by_valid_original_nodes():
    start = flat_nodes() + 2**26
    end = start.astype(np.float32).astype(np.float64)
    assert certify_bending_segment(start, end, material()).status != "certified"


def test_distance_row_must_certify_every_affected_hinge():
    start = flat_nodes()
    end = start.copy()
    end[2] = end[0]
    assert certify_bending_segment(start, end, material()).status != "certified"
