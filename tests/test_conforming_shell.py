"""Authored first, unexecuted: independent oriented material-complex controls."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import numpy as np
import pytest

from phymeo.constraints.conforming_shell import (
    ConformingShellReference,
    ShellDirectorLaw,
    ShellMaterialAddress,
    ShellProofLimits,
    ShellUnresolved,
)


def f1_reference(variant="identity"):
    ids = np.arange(4, dtype=np.int64)
    faces = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int64)
    face_ids = np.array([10, 20], dtype=np.int64)
    x = np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float64)
    rotation = np.eye(3)
    shift = np.zeros(3)
    if variant == "rotation":
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    if variant == "translation":
        shift = np.array([4, -2, 1], dtype=np.float64)
    x = x @ rotation.T + shift
    if variant == "permutation":
        order = [3, 1, 0, 2]
        ids, x = ids[order], x[order]
        face_ids, faces = face_ids[::-1], faces[::-1]
    ref = ConformingShellReference(
        "cloth", ids, face_ids, faces, x, np.full(4, 1 / 16, dtype=np.float64)
    )
    return ref, rotation, shift


def f1_positions(ref, q, rotation=None, shift=None):
    rotation = np.eye(3) if rotation is None else rotation
    shift = np.zeros(3) if shift is None else shift
    original = {0: (-1, 0, 0), 1: (1, 0, 0), 2: (0, 1, q), 3: (0, -1, q)}
    return np.array([original[int(i)] for i in ref.node_ids], dtype=np.float64) @ rotation.T + shift


def f2_reference(variant="identity"):
    ids = np.arange(5, dtype=np.int64)
    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]], dtype=np.int64)
    face_ids = np.array([10, 20, 30, 40], dtype=np.int64)
    r = np.eye(3)
    shift = np.zeros(3)
    if variant == "rotation":
        r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    if variant == "translation":
        shift = np.array([4, -2, 1], dtype=np.float64)
    x = x @ r.T + shift
    if variant == "permutation":
        order = [3, 0, 4, 1, 2]
        ids, x = ids[order], x[order]
        faces, face_ids = faces[[3, 1, 0, 2]], face_ids[[3, 1, 0, 2]]
    ref = ConformingShellReference("star", ids, face_ids, faces, x, np.full(5, 1 / 4))
    return ref, r, shift


def test_f1_exact_six_oriented_tetrahedra_and_shared_diagonal():
    ref, _, _ = f1_reference()
    expected = (
        ((0, -1), (1, -1), (2, -1), (2, 1)),
        ((0, -1), (1, -1), (2, 1), (1, 1)),
        ((0, -1), (0, 1), (1, 1), (2, 1)),
        ((0, -1), (1, -1), (3, 1), (3, -1)),
        ((0, -1), (1, -1), (1, 1), (3, 1)),
        ((0, -1), (0, 1), (3, 1), (1, 1)),
    )
    assert tuple(t.vertices for t in ref.tetrahedra) == expected
    assert tuple(t.tetra_id for t in ref.tetrahedra) == tuple(
        (face, stair) for face in (10, 20) for stair in range(3)
    )
    assert len(ref.boundary_facets) == 12
    internal_edge_side = {((0, -1), (1, -1), (1, 1)), ((0, -1), (0, 1), (1, 1))}
    assert not internal_edge_side.intersection(tuple(sorted(f.vertices)) for f in ref.boundary_facets)
    assert ref.components == ((0, 1, 2, 3),)


def test_reference_owns_arrays_with_immutable_backing_and_canonical_identity():
    ref, _, _ = f1_reference()
    changed, _, _ = f1_reference("permutation")
    assert ref.reference_id == changed.reference_id
    assert ref.input_fingerprint != changed.input_fingerprint
    assert ref.tetrahedra == changed.tetrahedra
    for value in (ref.node_ids, ref.face_ids, ref.oriented_nodes, ref.reference_positions, ref.half_thickness_m):
        with pytest.raises(ValueError):
            value.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        ref.domain = "changed"
    source = np.array(ref.reference_positions, copy=True)
    own = ConformingShellReference(ref.domain, ref.node_ids, ref.face_ids, ref.oriented_nodes, source, ref.half_thickness_m)
    source[:] = 17
    assert own.reference_id == ref.reference_id


@pytest.mark.parametrize("mutation", ["dtype", "bool", "shape", "nan", "unused", "duplicate", "winding", "thickness", "domain"])
def test_rejects_malformed_owned_inputs(mutation):
    ref, _, _ = f1_reference()
    data = [ref.domain, ref.node_ids.copy(), ref.face_ids.copy(), ref.oriented_nodes.copy(), ref.reference_positions.copy(), ref.half_thickness_m.copy()]
    if mutation == "dtype":
        data[1] = data[1].astype(np.int32)
    elif mutation == "bool":
        data[3] = data[3].astype(bool)
    elif mutation == "shape":
        data[4] = data[4].ravel()
    elif mutation == "nan":
        data[4][0, 0] = np.nan
    elif mutation == "unused":
        data[3][1] = [0, 1, 2]
    elif mutation == "duplicate":
        data[2][1] = data[2][0]
    elif mutation == "winding":
        data[3][1] = [0, 1, 3]
    elif mutation == "thickness":
        data[5][0] = 0
    else:
        data[0] = "x" * 257
    with pytest.raises((ValueError, TypeError)):
        ConformingShellReference(*data)


def test_disconnected_vertex_link_rejects_bow_tie():
    with pytest.raises(ShellUnresolved, match="topology"):
        ConformingShellReference("bow", np.arange(5, dtype=np.int64), np.arange(2, dtype=np.int64), np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int64), np.zeros((5, 3), dtype=np.float64), np.ones(5))


def test_nonmanifold_edge_rejects_before_geometry():
    with pytest.raises(ShellUnresolved, match="topology"):
        ConformingShellReference("edge", np.arange(5, dtype=np.int64), np.arange(3, dtype=np.int64), np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64), np.zeros((5, 3)), np.ones(5))


def test_law_limits_and_address_are_exact_owned_values():
    assert ShellDirectorLaw().projected_ratio_min == F(1, 4)
    limits = ShellProofLimits()
    with pytest.raises((TypeError, ValueError)):
        replace(limits, max_nodes=True)
    with pytest.raises(ValueError):
        replace(limits, max_nodes=65)
    with pytest.raises(ValueError):
        replace(limits, initial_sqrt_bits=65, max_sqrt_bits=64)
    ref, _, _ = f1_reference()
    counts = [0, 0, 0, 2**52]
    address = ShellMaterialAddress(ref.reference_id, (10, 0), counts, 0)
    counts[-1] = 0
    assert address.counts[-1] == 2**52
    with pytest.raises(ValueError):
        ShellMaterialAddress(ref.reference_id, (10, 0), [0, 0, 0, 1], 0)
    with pytest.raises(TypeError):
        ShellMaterialAddress(ref.reference_id, (10, 0), [True, 0, 0, 2**52 - 1], 0)

def test_cyclic_face_order_preserves_canonical_identity_but_binds_input_order():
    ref, _, _ = f1_reference()
    cycled = np.roll(ref.oriented_nodes, 1, axis=1)
    other = ConformingShellReference(ref.domain, ref.node_ids, ref.face_ids, cycled, ref.reference_positions, ref.half_thickness_m)
    assert other.reference_id == ref.reference_id
    assert other.input_fingerprint != ref.input_fingerprint
    assert other.tetrahedra == ref.tetrahedra


def test_constructor_shape_ceiling_precedes_owned_copy():
    ref, _, _ = f1_reference()
    with pytest.raises(ShellUnresolved, match="shape"):
        ConformingShellReference(ref.domain, ref.node_ids, ref.face_ids, ref.oriented_nodes, ref.reference_positions, ref.half_thickness_m, limits=replace(ShellProofLimits(), max_nodes=3))