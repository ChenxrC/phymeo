"""Independent analytic fixtures for rational ideal-law enclosures (not producer parity)."""

from fractions import Fraction as F
from math import atan2, sqrt

import numpy as np
import pytest

from phymeo.constraints.cloth_material import CompositeMaterialAuditSettings
from phymeo.core.capabilities import SolverFailure
from phymeo.validation.bending_bounds import Interval, atan_bounds, atan2_bounds, bending_bounds, pi_bounds, sqrt_interval


def flat_nodes():
    return np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., -1., 0.]])


def independent_energy(nodes, rest=0.25, compliance=0.125):
    # Determinant form, written independently of production normalized normals.
    e, u, v = nodes[1]-nodes[0], nodes[2]-nodes[0], nodes[3]-nodes[0]
    determinant = e[0]*(u[1]*v[2]-u[2]*v[1])-e[1]*(u[0]*v[2]-u[2]*v[0])+e[2]*(u[0]*v[1]-u[1]*v[0])
    cosine = float(e @ u) * float(e @ v) - float(e @ e) * float(u @ v)
    theta = atan2(-sqrt(float(e @ e))*determinant, cosine)
    return (theta-rest)**2/(2*compliance)


def test_sqrt_enclosures_are_rational_proofs_not_float_roots():
    for value in (F(0), F(2), F(1, 3), F(1, 2**1024), F(2**1024)):
        bounded = sqrt_interval(value)
        assert bounded.low**2 <= value <= bounded.high**2
        assert bounded.width >= 0


def test_pi_machin_and_quarter_angle_enclose_independent_decimal_brackets():
    low = F("3.141592653589793238462643383279502884197169399375105820974944592307816406286")
    high = low + F(1, 10**75)
    pi = pi_bounds()
    assert low < pi.low <= pi.high < high
    angle = atan_bounds(1)
    assert low/4 < angle.low <= angle.high < high/4
    assert pi.width < F(1, 2**240)


@pytest.mark.parametrize("s,a,quarter", [(1, 1, 1), (1, -1, 3), (-1, -1, -3), (-1, 1, -1), (1, 0, 2), (-1, 0, -2)])
def test_explicit_atan_quadrants(s, a, quarter):
    bounds = atan2_bounds(s, a)
    pi = pi_bounds()
    expected = pi * F(quarter, 4)
    assert bounds.low <= expected.high and expected.low <= bounds.high


def test_zero_cut_and_unknown_rectangles_are_explicit():
    assert atan2_bounds(0, 1) == Interval.point(0)
    for sine, cosine in ((0, -1), (0, 0), (Interval(-1, 1), Interval(-2, -1))):
        with pytest.raises(SolverFailure):
            atan2_bounds(sine, cosine)


def test_unit_flat_hinge_literal_gradient_and_energy():
    result = bending_bounds(flat_nodes(), 0.25, 0.125)
    assert result.angle == Interval.point(0)
    assert result.constraint == Interval.point(F(-1, 4))
    assert result.energy == Interval.point(F(1, 4))
    expected = ((0, 0, 2), (0, 0, 0), (0, 0, -1), (0, 0, -1))
    for row, literal in zip(result.gradient, expected, strict=True):
        for bounded, value in zip(row, literal, strict=True):
            assert bounded == Interval.point(value)


def test_oblique_energy_gradient_has_independent_directional_derivative():
    nodes = flat_nodes()
    nodes[2, 2], nodes[3, 2] = 1/8, -1/16
    direction = np.array([[1., 2., -1.], [-2., 1., 0.], [0., -1., 2.], [1., 0., -1.]])/8
    result = bending_bounds(nodes, 0.25, 0.125)
    derivative = sum((result.constraint/F(1, 8)*g*F(float(direction[i, c]))
                      for i, row in enumerate(result.gradient) for c, g in enumerate(row)), Interval.point(0))
    step = 2**-18
    finite_difference = (independent_energy(nodes+step*direction)-independent_energy(nodes-step*direction))/(2*step)
    assert abs(finite_difference-float((derivative.low+derivative.high)/2)) < 2e-7
    assert derivative.width < F(1, 2**220)


def test_permutations_obey_axis_and_face_orientation_independently():
    nodes = flat_nodes()
    nodes[3] = (0., 0., -1.)
    original = bending_bounds(nodes, 0., 1.)
    reverse_axis = bending_bounds(nodes[[1, 0, 2, 3]], 0., 1.)
    reverse_faces = bending_bounds(nodes[[0, 1, 3, 2]], 0., 1.)
    full_exchange = bending_bounds(nodes[[1, 0, 3, 2]], 0., 1.)
    pi = pi_bounds()
    assert original.angle.low > pi.low/3  # independently +pi/2.
    assert reverse_axis.angle.high < -pi.low/3
    assert reverse_faces.angle.high < -pi.low/3
    assert full_exchange.angle.low > pi.low/3


def test_degeneracy_and_rest_error_cut_rejected():
    nodes = flat_nodes()
    nodes[2] = nodes[0]
    with pytest.raises(SolverFailure, match="face"):
        bending_bounds(nodes, 0., 1.)
    nodes = flat_nodes()
    nodes[3] = (0., 0., -1.)
    with pytest.raises(SolverFailure, match="branch"):
        bending_bounds(nodes, -2., 1.)


def test_proof_policy_validation_and_resource_width_is_visible():
    for kwargs in ({"angle_terms": 0}, {"sqrt_bits": True}, {"max_domain_depth": -1}):
        with pytest.raises(ValueError):
            CompositeMaterialAuditSettings(**kwargs)
    coarse = atan_bounds(F(1, 2), CompositeMaterialAuditSettings(sqrt_bits=8, angle_terms=1))
    fine = atan_bounds(F(1, 2))
    assert coarse.low <= fine.low <= fine.high <= coarse.high
    assert coarse.width > F(1, 1000000)
