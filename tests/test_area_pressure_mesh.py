"""Independent bounded-mesh admission tests for the frozen I1 contract.

Local affine fixtures are not complete boundary-zero pressure assets. Geometry
is prescribed here; no new clipping implementation is used as an oracle.
"""

import numpy as np
import pytest

from phymeo.geometry.area_pressure_intersection import (
    AreaPressureIntersectionError,
    RigidPressureMesh,
    TetraPressureMesh,
)


def _unit(**changes):
    values = dict(
        node_ids=("o", "x", "y", "z"),
        world_vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
        cells=((0, 1, 2, 3),), nodal_pressure=(0, 0, 0, 1), cell_ids=("unit",),
        pressure_domain="local",
    )
    values.update(changes)
    return TetraPressureMesh(**values)


def _full(**changes):
    values = dict(
        node_ids=("o", "x", "y", "z", "inside"),
        world_vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1 / 4,) * 3),
        cells=((0, 1, 2, 4), (0, 2, 3, 4), (0, 3, 1, 4), (1, 3, 2, 4)),
        nodal_pressure=(0, 0, 0, 0, 1), cell_ids=("base", "x-side", "y-side", "slant"),
        pressure_domain="full",
    )
    values.update(changes)
    return TetraPressureMesh(**values)


def _rigid(**changes):
    values = dict(
        node_ids=("a", "b", "c"),
        world_vertices=((0, 0, 1 / 4), (1, 0, 1 / 4), (0, 1, 1 / 4)),
        faces=((0, 1, 2),), face_ids=("face",),
    )
    values.update(changes)
    return RigidPressureMesh(**values)


def _reject(category, factory):
    with pytest.raises(AreaPressureIntersectionError) as caught:
        factory()
    assert caught.value.category == category


def _readonly(array, dtype):
    assert isinstance(array, np.ndarray) and array.dtype == np.dtype(dtype)
    owner = array
    while isinstance(owner, np.ndarray):
        assert not owner.flags.writeable
        with pytest.raises(ValueError):
            owner.setflags(write=True)
        owner = owner.base
    while isinstance(owner, memoryview):
        assert owner.readonly
        owner = owner.obj
    assert isinstance(owner, bytes)
    with pytest.raises(ValueError):
        array.flat[0] = 99


def test_full_hat_asset_and_local_affine_cell_are_distinct_domains():
    # Four equal-volume cells join at an interior node. All external faces
    # interpolate zero; the interior nodal pressure is strictly positive.
    full = _full()
    assert full.pressure_domain == "full"
    np.testing.assert_array_equal(full.nodal_pressure, (0, 0, 0, 0, 1))
    local = _unit()
    assert local.pressure_domain == "local"
    _reject("invalid_mesh", lambda: _unit(pressure_domain="full"))
    _reject("invalid_mesh", lambda: _full(nodal_pressure=(0, 0, 0, 0, 0)))
    _reject("invalid_mesh", lambda: _full(nodal_pressure=(0, 0, 0, 1, 1)))
    # Local bypasses only the full-pressure boundary condition, not conformity.
    assert _full(nodal_pressure=(0, 0, 0, 1, 1), pressure_domain="local").pressure_domain == "local"
    with pytest.raises(ValueError):
        _unit(pressure_domain="manufactured-but-unlabelled")


def test_tetra_vertex_permutation_is_not_a_deformation_inversion_claim():
    original = _unit()
    permuted = _unit(cells=((3, 1, 2, 0),))
    assert original.pressure_domain == permuted.pressure_domain == "local"
    assert set(permuted.cells[0].tolist()) == {0, 1, 2, 3}


def test_disconnected_conforming_volume_and_open_rigid_components_are_allowed():
    vertices = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                (3, 0, 0), (4, 0, 0), (3, 1, 0), (3, 0, 1))
    mesh = _unit(node_ids=tuple(f"n-{i}" for i in range(8)), world_vertices=vertices,
                 cells=((0, 1, 2, 3), (4, 5, 6, 7)), nodal_pressure=(0, 0, 0, 1) * 2,
                 cell_ids=("left", "right"))
    assert len(mesh.cells) == 2
    rigid = _rigid(node_ids=tuple(f"r-{i}" for i in range(6)),
                   world_vertices=(vertices[0], vertices[1], vertices[2],
                                   vertices[4], vertices[5], vertices[6]),
                   faces=((0, 1, 2), (3, 4, 5)), face_ids=("left", "right"))
    assert len(rigid.faces) == 2


def test_shared_rigid_edge_requires_opposite_directed_incidence():
    values = dict(node_ids=("a", "b", "c", "d"),
                  world_vertices=((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
                  faces=((0, 1, 2), (0, 2, 3)), face_ids=("one", "two"))
    assert len(_rigid(**values).faces) == 2
    values["faces"] = ((0, 1, 2), (0, 3, 2))
    _reject("invalid_mesh", lambda: _rigid(**values))


def test_mesh_arrays_are_owned_bytes_backed_and_indices_are_int64():
    vertices = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=float)
    cells = np.array(((0, 1, 2, 3),), dtype=np.int32)
    pressure = np.array((0, 0, 0, 1), dtype=float)
    originals = [array.copy() for array in (vertices, cells, pressure)]
    mesh = _unit(world_vertices=vertices, cells=cells, nodal_pressure=pressure)
    vertices[:] = cells[:] = pressure[:] = 99
    for name, expected, dtype in zip(("world_vertices", "cells", "nodal_pressure"),
                                      originals, (np.float64, np.int64, np.float64)):
        np.testing.assert_array_equal(getattr(mesh, name), expected)
        _readonly(getattr(mesh, name), dtype)
    r_vertices = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
    faces = np.array(((0, 1, 2),), dtype=np.int32)
    rigid = _rigid(world_vertices=r_vertices, faces=faces)
    r_vertices[:] = faces[:] = -1
    np.testing.assert_array_equal(rigid.world_vertices, ((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    np.testing.assert_array_equal(rigid.faces, ((0, 1, 2),))
    _readonly(rigid.world_vertices, np.float64)
    _readonly(rigid.faces, np.int64)


@pytest.mark.parametrize("changes", [
    {"world_vertices": ((0, 0), (1, 0), (0, 1), (0, 0))},
    {"world_vertices": ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, np.nan))},
    {"cells": ((True, 1, 2, 3),)}, {"cells": ((0.0, 1.0, 2.0, 3.0),)},
    {"cells": ((0, 1, 2, 4),)}, {"nodal_pressure": (0, 0, 0, -1)},
    {"node_ids": ("o", "x", "x", "z")}, {"cell_ids": ("c" * 129,)},
])
def test_representative_input_errors_are_value_errors(changes):
    with pytest.raises(ValueError):
        _unit(**changes)


def test_empty_meshes_unused_nodes_and_coincident_distinct_ids_are_rejected():
    with pytest.raises(ValueError):
        TetraPressureMesh((), np.empty((0, 3)), np.empty((0, 4), dtype=int), (), (),
                          pressure_domain="local")
    with pytest.raises(ValueError):
        RigidPressureMesh((), np.empty((0, 3)), np.empty((0, 3), dtype=int), ())
    with pytest.raises(ValueError):
        _unit(node_ids=("o", "x", "y", "z", "unused"),
              world_vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 2, 2)),
              nodal_pressure=(0, 0, 0, 1, 0))
    _reject("invalid_mesh", lambda: _unit(
        node_ids=("o", "x", "y", "z", "different-id-same-position"),
        world_vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)),
        cells=((0, 1, 2, 3), (4, 1, 2, 3)), nodal_pressure=(0, 0, 0, 1, 0),
        cell_ids=("one", "two"),
    ))


@pytest.mark.parametrize("kind", ("duplicate", "same-side", "overlap", "contained", "nonmatching"))
def test_tet_closed_intersection_must_be_exactly_the_shared_node_simplex(kind):
    first = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    if kind == "duplicate":
        vertices, cells = first, ((0, 1, 2, 3), (3, 2, 1, 0))
    elif kind == "same-side":
        vertices, cells = first + ((0, 0, 1 / 2),), ((0, 1, 2, 3), (0, 1, 2, 4))
    else:
        second = {
            "overlap": ((1 / 4, 1 / 4, 1 / 4), (5 / 4, 1 / 4, 1 / 4),
                        (1 / 4, 5 / 4, 1 / 4), (1 / 4, 1 / 4, 5 / 4)),
            "contained": ((1 / 8, 1 / 8, 1 / 8), (1 / 4, 1 / 8, 1 / 8),
                          (1 / 8, 1 / 4, 1 / 8), (1 / 8, 1 / 8, 1 / 4)),
            "nonmatching": ((1 / 4, 1 / 4, 0), (1 / 2, 1 / 4, 0),
                            (1 / 4, 1 / 2, 0), (1 / 4, 1 / 4, -1 / 4)),
        }[kind]
        vertices, cells = first + second, ((0, 1, 2, 3), (4, 5, 6, 7))
    # Zero pressure is a valid local field; geometry alone rejects these pairs.
    _reject("invalid_mesh", lambda: _unit(
        node_ids=tuple(f"n-{i}" for i in range(len(vertices))), world_vertices=vertices,
        cells=cells, nodal_pressure=(0,) * len(vertices), cell_ids=("one", "two"),
    ))


@pytest.mark.parametrize("kind", ("duplicate", "coplanar-overlap", "nonshared-crossing"))
def test_rigid_closed_intersection_cannot_hide_coverage_or_crossings(kind):
    if kind == "duplicate":
        vertices = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
        faces = ((0, 1, 2), (0, 1, 2))
    elif kind == "coplanar-overlap":
        vertices = ((0, 0, 0), (2, 0, 0), (0, 2, 0),
                    (1 / 4, 1 / 4, 0), (1, 1 / 4, 0), (1 / 4, 1, 0))
        faces = ((0, 1, 2), (3, 4, 5))
    else:
        vertices = ((-1, -1, 0), (1, -1, 0), (0, 1, 0),
                    (0, -1 / 2, -1), (0, -1 / 2, 1), (0, 1 / 2, 0))
        faces = ((0, 1, 2), (3, 4, 5))
    _reject("invalid_mesh", lambda: _rigid(
        node_ids=tuple(f"r-{i}" for i in range(len(vertices))), world_vertices=vertices,
        faces=faces, face_ids=("one", "two"),
    ))


def test_resource_caps_reject_before_conformity_work():
    # Thirty-three conforming wedges around one common edge, only 36 nodes.
    vertices = ((0, 0, 0), (0, 0, 1)) + tuple((1, i, 0) for i in range(34))
    _reject("resource_limit", lambda: _unit(
        node_ids=tuple(f"n-{i}" for i in range(36)), world_vertices=vertices,
        cells=tuple((0, 1, i + 2, i + 3) for i in range(33)),
        nodal_pressure=(0, 1) + (0,) * 34, cell_ids=tuple(f"cell-{i}" for i in range(33)),
    ))
    # Sixty-five nonoverlapping triangles in an open fan, only 67 nodes.
    r_vertices = ((0, 1, 0),) + tuple((i, 0, 0) for i in range(66))
    _reject("resource_limit", lambda: _rigid(
        node_ids=tuple(f"r-{i}" for i in range(67)), world_vertices=r_vertices,
        faces=tuple((0, i + 1, i + 2) for i in range(65)),
        face_ids=tuple(f"face-{i}" for i in range(65)),
    ))
    # Forty-three disconnected triangles are within the face cap but use 129 nodes.
    r_vertices = tuple(point for i in range(43)
                       for point in ((3 * i, 0, 0), (3 * i + 1, 0, 0), (3 * i, 1, 0)))
    _reject("resource_limit", lambda: _rigid(
        node_ids=tuple(f"r-{i}" for i in range(129)), world_vertices=r_vertices,
        faces=tuple((3 * i, 3 * i + 1, 3 * i + 2) for i in range(43)),
        face_ids=tuple(f"face-{i}" for i in range(43)),
    ))


@pytest.mark.parametrize("coordinate", (2.0**256, 2.0**-256))
def test_stored_float_exact_ratio_has_a_finite_bit_budget(coordinate):
    _reject("resource_limit", lambda: _unit(
        world_vertices=((0, 0, 0), (coordinate, 0, 0), (0, 1, 0), (0, 0, 1)),
    ))


def test_zero_volume_tet_and_zero_area_rigid_face_are_invalid_meshes():
    _reject("invalid_mesh", lambda: _unit(
        world_vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)),
    ))
    _reject("invalid_mesh", lambda: _rigid(
        world_vertices=((0, 0, 0), (1, 0, 0), (2, 0, 0)),
    ))
