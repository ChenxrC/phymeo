"""Unexecuted independent strict-embedding and complete reference-chart controls."""

from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest
from test_conforming_shell import f1_positions, f1_reference, f2_reference

from phymeo.constraints.conforming_shell import ConformingShellReference, ShellProofLimits, ShellUnresolved
from phymeo.geometry.shell_directors import shell_director_geometry
from phymeo.geometry.shell_embedding import certify_shell_embedding


def fan_reference():
    ring = [(2, 0, 0), (1, 1, 0), (-1, 1, 0), (-2, 0, 0), (-1, -1, 0), (1, -1, 0)]
    faces = np.array([[0, j, 1 if j == 6 else j + 1] for j in range(1, 7)], dtype=np.int64)
    return ConformingShellReference("fan", np.arange(7, dtype=np.int64), np.arange(6, dtype=np.int64), faces, np.array([(0, 0, 0)] + ring, dtype=np.float64), np.full(7, 1 / 16))


def disconnected_reference():
    x = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, -1], [1, 0, -1], [0, 1, -1]], dtype=np.float64)
    return ConformingShellReference("two", np.arange(6, dtype=np.int64), np.arange(2, dtype=np.int64), np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64), x, np.full(6, 1 / 8))


def test_f1_static_all_tetra_pair_coverage_and_strict_bound():
    ref, _, _ = f1_reference()
    certificate = certify_shell_embedding(ref, f1_positions(ref, 1 / 4))
    assert certificate.status == "strict_geometry_certified"
    assert certificate.cell.k_max < F(33, 64)
    assert len(certificate.cell.tetrahedra) == 6
    assert len(certificate.cell.pair_incidence) == 15
    assert all(t.determinant.lo > 0 for t in certificate.cell.tetrahedra)


@pytest.mark.parametrize("ring", [
    [(1, 0, 0), (-1, 1, 0), (-1, -1, 0), (2, 0, 0), (-2, 2, 0), (-2, -2, 0)],
    [(1, 0, 0), (-1, 1, 0), (-1, -1, 0)] * 2,
])
def test_n1_n2_positive_local_geometry_does_not_certify_overlapping_star(ring):
    ref = fan_reference()
    x = np.array([(0, 0, 0)] + ring, dtype=np.float64)
    geometry = shell_director_geometry(ref, x)
    assert all(v[2].lo > 0 for v in geometry.areas)
    # Every planar triangle's positive area times common positive thickness gives positive tets.
    assert all(v[0].lo == v[0].hi == v[1].lo == v[1].hi == 0 for v in geometry.areas)
    with pytest.raises(ShellUnresolved, match="embedding_not_certified"):
        certify_shell_embedding(ref, x)


@pytest.mark.parametrize("rho", [5 / 8, 3 / 4])
def test_n4_exact_degenerate_or_inverted_tetra_despite_valid_directors(rho):
    old, _, _ = f1_reference()
    ref = ConformingShellReference(old.domain, old.node_ids, old.face_ids, old.oriented_nodes, old.reference_positions, np.full(4, rho))
    x = np.array([[-1, 0, 0], [1, 0, 0], [0, 3 / 8, 9 / 32], [0, -3 / 8, 9 / 32]], dtype=np.float64)
    shell_director_geometry(ref, x)
    assert 4 * F.from_float(rho) * (F(3, 8) - 3 * F.from_float(rho) / 5) <= 0
    with pytest.raises(ShellUnresolved):
        certify_shell_embedding(ref, x)


@pytest.mark.parametrize("mode", ["thickness", "nonplanar"])
def test_general_reference_remains_owned_but_unsupported_chart(mode):
    old, _, _ = f1_reference()
    x, rho = old.reference_positions.copy(), old.half_thickness_m.copy()
    if mode == "thickness":
        rho[2] *= 2
    else:
        x[2, 2] = 1 / 8
    ref = ConformingShellReference(old.domain, old.node_ids, old.face_ids, old.oriented_nodes, x, rho)
    shell_director_geometry(ref, x)
    with pytest.raises(ShellUnresolved, match="unsupported_reference_chart"):
        certify_shell_embedding(ref, x)
    assert len(ref.tetrahedra) == 6


def test_ordered_collinear_boundary_subdivision_is_valid():
    x = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], dtype=np.float64)
    ref = ConformingShellReference("collinear", np.arange(5, dtype=np.int64), np.arange(3, dtype=np.int64), np.array([[4, 0, 1], [4, 1, 2], [4, 2, 3]], dtype=np.int64), x, np.full(5, 1 / 16))
    result = certify_shell_embedding(ref, x)
    assert result.cell.k_max < F(1, 2**40)
    assert len(result.reference.charts[0].boundary) == 5


def test_wrong_reference_point_incidence_rejects_even_without_positive_area_overlap():
    # Two components sharing a world boundary point are not one material vertex.
    ref = disconnected_reference()
    x = ref.reference_positions.copy()
    x[3:] += [1, 0, 2]
    touching = ConformingShellReference(ref.domain, ref.node_ids, ref.face_ids, ref.oriented_nodes, x, ref.half_thickness_m)
    with pytest.raises(ShellUnresolved):
        certify_shell_embedding(touching, x)


def test_strict_remote_component_boundary_touch_excluded():
    ref = disconnected_reference()
    x = ref.reference_positions.copy()
    x[:3, 2], x[3:, 2] = 1 / 8, -1 / 8
    with pytest.raises(ShellUnresolved):
        certify_shell_embedding(ref, x)


def test_resource_limits_are_not_silently_reset_in_reference_proof():
    ref, _, _ = f2_reference()
    with pytest.raises(ShellUnresolved, match="operation"):
        certify_shell_embedding(ref, ref.reference_positions, limits=replace(ShellProofLimits(), max_scalar_operations=4))