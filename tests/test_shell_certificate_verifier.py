"""Unexecuted forged receipts and independently valid alternative witnesses."""

import copy
from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest
from test_conforming_shell import f1_positions, f1_reference, f2_reference
from test_shell_directors import top_center
from test_shell_embedding import disconnected_reference

from phymeo.constraints.conforming_shell import ShellMaterialAddress, ShellUnresolved
from phymeo.geometry._shell_intervals import Interval
from phymeo.geometry.shell_directors import shell_director_geometry, shell_material_point
from phymeo.geometry.shell_embedding import certify_shell_embedding
from phymeo.geometry.shell_sweep import TimeCell, certify_shell_sweep
from phymeo.validation.conforming_shell import (
    verify_shell_director_geometry,
    verify_shell_embedding,
    verify_shell_material_point,
    verify_shell_sweep,
)


def test_verifiers_do_not_call_producers_or_constructor_topology(monkeypatch):
    ref, _, _ = f2_reference()
    x = ref.reference_positions
    address = top_center(ref)
    geometry = shell_director_geometry(ref, x)
    point = shell_material_point(ref, x, address)
    static = certify_shell_embedding(ref, x)
    path = np.stack((x, x))
    sweep = certify_shell_sweep(ref, path, h=1 / 16)
    def forbidden(*args, **kwargs):
        raise AssertionError("independent verifier called a producer")
    monkeypatch.setattr("phymeo.geometry.shell_directors.shell_director_geometry", forbidden)
    monkeypatch.setattr("phymeo.geometry.shell_directors.shell_material_point", forbidden)
    monkeypatch.setattr("phymeo.geometry.shell_embedding.certify_shell_embedding", forbidden)
    monkeypatch.setattr("phymeo.geometry.shell_sweep.certify_shell_sweep", forbidden)
    monkeypatch.setattr("phymeo.constraints.conforming_shell.ConformingShellReference.__init__", forbidden)
    assert verify_shell_director_geometry(geometry, ref, x)
    assert verify_shell_material_point(point, ref, x, address)
    assert verify_shell_embedding(static, ref, x)
    assert verify_shell_sweep(sweep, ref, path, h=1 / 16)


@pytest.mark.parametrize("field", ["node", "zero_j", "opposite", "point", "error", "identity", "generation"])
def test_forged_full_star_material_point_rejects(field):
    ref, _, _ = f2_reference()
    address = top_center(ref)
    point = shell_material_point(ref, ref.reference_positions, address)
    if field == "node":
        point = replace(point, node_ids=point.node_ids[:-1])
    elif field in ("zero_j", "opposite"):
        intervals = list(point.jacobian.intervals)
        index = 0 if field == "zero_j" else 3 * 9 + 2
        intervals[index] = Interval(F(0), F(0))
        point = replace(point, jacobian=replace(point.jacobian, intervals=tuple(intervals)))
    elif field == "point":
        values = point.point.values.copy()
        values[0] = 1
        point = replace(point, point=replace(point.point, values=values))
    elif field == "error":
        errors = list(point.point.errors)
        errors[0] = F(-1)
        point = replace(point, point=replace(point.point, errors=tuple(errors)))
    elif field == "identity":
        point = replace(point, reference_id="0" * 64)
    else:
        address = ShellMaterialAddress(address.reference_id, address.tetra_id, address.counts, 1)
    with pytest.raises(ShellUnresolved):
        verify_shell_material_point(point, ref, ref.reference_positions, address)


def test_wider_containing_interval_is_valid_but_merely_overlapping_is_not():
    ref, _, _ = f1_reference()
    x = f1_positions(ref, 1 / 8)
    tet = ref.tetrahedra[0]
    address = ShellMaterialAddress(ref.reference_id, tet.tetra_id, (0, 0, 0, 2**52), 0)
    point = shell_material_point(ref, x, address)
    eps = F(1, 2**60)
    widened = tuple(Interval(v.lo-eps, v.hi+eps) for v in point.point.intervals)
    larger = replace(point, point=replace(point.point, intervals=widened, errors=tuple(e+eps for e in point.point.errors)))
    assert verify_shell_material_point(larger, ref, x, address)
    original = point.point.intervals[1]
    assert original.lo < original.hi
    midpoint = (original.lo+original.hi)/2
    forged = list(point.point.intervals)
    forged[1] = Interval(midpoint, original.hi+eps)
    with pytest.raises(ShellUnresolved):
        verify_shell_material_point(replace(point, point=replace(point.point, intervals=tuple(forged))), ref, x, address)


@pytest.mark.parametrize("field", ["tetra", "pair", "component", "boundary", "state", "status", "bits"])
def test_static_coverage_topology_and_binding_forgeries_reject(field):
    ref, _, _ = f1_reference()
    x = f1_positions(ref, 1 / 8)
    cert = certify_shell_embedding(ref, x)
    if field == "tetra":
        cert = replace(cert, cell=replace(cert.cell, tetrahedra=cert.cell.tetrahedra[:-1]))
    elif field == "pair":
        cert = replace(cert, cell=replace(cert.cell, pair_incidence=cert.cell.pair_incidence[:-1]))
    elif field == "component":
        cert = replace(cert, cell=replace(cert.cell, components=()))
    elif field == "boundary":
        ref = copy.copy(ref)
        object.__setattr__(ref, "boundary_facets", ref.boundary_facets[:-1])
    elif field == "state":
        x = x.copy()
        x[2, 2] += 1 / 64
    elif field == "status":
        cert = replace(cert, status="physical_accepted")
    else:
        cert = replace(cert, cell=replace(cert.cell, bits=8192))
    with pytest.raises(ShellUnresolved):
        verify_shell_embedding(cert, ref, x)


def test_alternative_rational_separator_not_producer_byte_identity():
    ref = disconnected_reference()
    cert = certify_shell_embedding(ref, ref.reference_positions)
    sep = cert.cell.separators[0]
    twice = lambda interval: Interval(2*interval.lo, 2*interval.hi)
    other = replace(sep, axis=tuple(2*v for v in sep.axis), left_support=twice(sep.left_support), right_support=twice(sep.right_support))
    altered = replace(cert, cell=replace(cert.cell, separators=(other,)))
    assert verify_shell_embedding(altered, ref, ref.reference_positions)


def test_alternative_complete_dyadic_partition_and_missing_cells():
    ref, _, _ = f2_reference()
    path = np.stack((ref.reference_positions, ref.reference_positions))
    cert = certify_shell_sweep(ref, path, h=1 / 16)
    assert len(cert.cells) == 1
    left = replace(cert.cells[0], start=F(0), end=F(1, 2))
    right = replace(cert.cells[0], start=F(1, 2), end=F(1))
    tree = (TimeCell(F(0), F(1), 0, False), TimeCell(F(0), F(1, 2), 1, True), TimeCell(F(1, 2), F(1), 1, True))
    alternate = replace(cert, cells=(left, right), tree=tree)
    assert verify_shell_sweep(alternate, ref, path, h=1 / 16)
    with pytest.raises(ShellUnresolved):
        verify_shell_sweep(replace(alternate, cells=(left,)), ref, path, h=1 / 16)
    with pytest.raises(ShellUnresolved):
        verify_shell_sweep(cert, ref, path, h=1 / 32)
    with pytest.raises(ShellUnresolved):
        verify_shell_sweep(replace(cert, tree=()), ref, path, h=1 / 16)