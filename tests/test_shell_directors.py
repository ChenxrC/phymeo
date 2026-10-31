"""Unexecuted analytic full-star maps; no finite-difference or physical pass claim."""

from fractions import Fraction as F

import numpy as np
import pytest
from test_conforming_shell import f1_positions, f1_reference, f2_reference

from phymeo.constraints.conforming_shell import ShellMaterialAddress, ShellUnresolved
from phymeo.geometry.shell_directors import shell_director_geometry, shell_material_point


def assert_publication(pub, expected):
    expected = np.asarray(expected)
    assert pub.values.shape == expected.shape
    for interval, error, ideal, stored in zip(pub.intervals, pub.errors, expected.flat, pub.values.flat, strict=True):
        rational = F.from_float(float(ideal))
        assert interval.lo <= rational <= interval.hi
        assert error < F(1, 2**40)
        actual = F.from_float(float(stored))
        assert abs(actual - rational) <= error


def top_center(ref):
    for tet in ref.tetrahedra:
        if (0, 1) in tet.vertices:
            counts = tuple(2**52 if vertex == (0, 1) else 0 for vertex in tet.vertices)
            return ShellMaterialAddress(ref.reference_id, tet.tetra_id, counts, 0)
    raise AssertionError("top-center must be a reference vertex")


@pytest.mark.parametrize("variant", ["identity", "rotation", "translation", "permutation"])
def test_f2_complete_five_node_jacobian_and_exact_independent_response(variant):
    receipts = []
    for _ in range(2):
        ref, rotation, shift = f2_reference(variant)
        point = shell_material_point(ref, ref.reference_positions, top_center(ref))
        blocks = np.zeros((5, 3, 3))
        blocks[0] = np.eye(3)
        blocks[1, 0, 2], blocks[3, 0, 2] = -1 / 8, 1 / 8
        blocks[2, 1, 2], blocks[4, 1, 2] = -1 / 8, 1 / 8
        blocks = np.array([rotation @ block @ rotation.T for block in blocks])
        assert point.node_ids == (0, 1, 2, 3, 4)
        assert_publication(point.point, rotation @ [0, 0, 1 / 4] + shift)
        assert_publication(point.jacobian, blocks)
        assert np.array_equal(point.jacobian.values.sum(axis=0), np.eye(3))
        w = np.eye(3) + sum(block @ block.T for block in point.jacobian.values)
        assert np.array_equal(w, rotation @ np.diag([65 / 32, 65 / 32, 2]) @ rotation.T)
        receipts.append((point.canonical_address, point.point.intervals, point.jacobian.intervals))
    assert receipts[0] == receipts[1]


@pytest.mark.parametrize("u,tangent,slip,energy", [(F(65, 256), F(-1, 8), F(0), F(-1089, 4096)), (F(65, 64), F(-1, 4), F(65, 128), F(-451, 1024))])
def test_f2_independent_impulse_squares_virtual_work_and_reaction(u, tangent, slip, energy):
    ref, _, _ = f2_reference()
    point = shell_material_point(ref, ref.reference_positions, top_center(ref))
    p = np.array([float(tangent), 0, 1 / 2])
    shell_load = -np.einsum("nij,j->ni", point.jacobian.values.transpose(0, 2, 1), p)
    expected = np.zeros((5, 3))
    expected[0] = -p
    expected[1, 2], expected[3, 2] = float(tangent / 8), float(-tangent / 8)
    assert np.array_equal(shell_load, expected)
    assert np.array_equal(shell_load.sum(axis=0) + p, np.zeros(3))
    moment = np.cross(ref.reference_positions, shell_load).sum(axis=0)
    assert np.array_equal(moment, [0, float(-tangent / 4), 0])
    assert np.array_equal(moment + np.cross(point.point.values, p), np.zeros(3))
    virtual = np.array([[1, 2, 3], [2, -1, 3], [-4, 2, 1], [3, 1, -2], [1, 0, 4]], dtype=np.float64)
    velocity = np.einsum("nij,nj->i", point.jacobian.values, virtual)
    assert np.sum(shell_load * virtual) == -p @ velocity
    kappa = F(65, 32)
    assert u + kappa * tangent == slip
    assert -F(1, 2) + F(1, 2) * (2 * F(1, 4)) + tangent * u + kappa * tangent**2 / 2 == energy
    original_ke = (u*u + 1) / 2
    outgoing_ke = ((u+tangent)**2 + F(1, 4) + tangent**2 + F(1, 4) + 2*(tangent/8)**2) / 2
    assert outgoing_ke - original_ke == energy


def test_shared_material_parent_address_does_not_change_point_or_star():
    ref, _, _ = f2_reference()
    receipts = []
    for tet in ref.tetrahedra:
        if (0, 1) not in tet.vertices:
            continue
        address = ShellMaterialAddress(ref.reference_id, tet.tetra_id, tuple(2**52 if v == (0, 1) else 0 for v in tet.vertices), 3)
        point = shell_material_point(ref, ref.reference_positions, address)
        receipts.append((point.canonical_address, point.node_ids, point.point.intervals, point.jacobian.intervals))
    assert len(receipts) > 1 and all(value == receipts[0] for value in receipts)


@pytest.mark.parametrize("variant", ["identity", "rotation", "translation", "permutation"])
@pytest.mark.parametrize("q", [F(1, 8), F(3, 16), F(1, 4)])
def test_f1_nonflat_full_map_along_independently_differentiated_free_tangent(q, variant):
    ref, rotation, shift = f1_reference(variant)
    positions = f1_positions(ref, float(q), rotation, shift)
    tet = ref.tetrahedra[0]
    address = ShellMaterialAddress(ref.reference_id, tet.tetra_id, (0, 0, 0, 2**52), 0)
    point = shell_material_point(ref, positions, address)
    # y_top,2=(0,1-rho*q/s,q+rho/s), s=sqrt(1+q*q).
    # dy/dq=(0,-rho/s^3,1-rho*q/s^3), qdot=1.
    from phymeo.geometry._shell_intervals import Arithmetic
    from phymeo.constraints.conforming_shell import ShellProofLimits
    a = Arithmetic(ShellProofLimits())
    s = a.sqrt(a.exact(1 + q*q))
    inverse_cube = a.div(a.exact(1), a.mul(a.mul(s, s), s))
    expected = (a.exact(0), a.mul(a.exact(-F(1, 16)), inverse_cube), a.sub(a.exact(1), a.mul(a.exact(q / 16), inverse_cube)))
    expected = tuple(a.sum(a.mul(a.exact(F.from_float(float(rotation[i, j]))), expected[j]) for j in range(3)) for i in range(3))
    # Add only the known two outer-node z columns, not producer response output.
    blocks = dict(zip(point.node_ids, np.array(point.jacobian.intervals, dtype=object).reshape((-1, 3, 3)), strict=True))
    for component in range(3):
        value = a.exact(0)
        for node in (2, 3):
            if node in blocks:
                value = a.add(value, blocks[node][component, 2])
        assert value.lo <= expected[component].hi and value.hi >= expected[component].lo
        assert value.hi - value.lo < F(1, 2**40)
    # Independent squared exact relation also checks the sign and rational target.
    y = rotation.T @ (point.point.values - shift)
    assert y[1] < 1 and y[2] > float(q)


def test_geometry_only_status_storage_errors_and_vanishing_sum_rejection():
    ref, _, _ = f1_reference()
    geometry = shell_director_geometry(ref, f1_positions(ref, 1 / 8))
    assert geometry.status == "director_geometry_only"
    assert geometry.precision == "ideal-from-stored-fp64"
    assert max(geometry.lifted_publication.errors) < F(1, 2**40)
    bad = np.array(ref.reference_positions, copy=True)
    bad[3] = bad[2]
    with pytest.raises(ShellUnresolved, match="director"):
        shell_director_geometry(ref, bad)