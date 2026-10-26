"""Mesh topology and area-mass conservation for the independent cloth builder."""

import numpy as np
import pytest

import phymeo
from phymeo import Plane, StepInputs, XPBDSolver


def build(*args, **kwargs):
    builder = getattr(phymeo, "cloth_grid", None)
    assert builder is not None, "cloth_grid is not implemented"
    return builder(*args, **kwargs)


def test_single_cell_has_upward_winding_and_exact_lumped_mass():
    patch = build(2, 2, spacing=1.0, density=6.0, origin=(0.0, 0.0, 0.0))
    np.testing.assert_array_equal(patch.triangles, [[0, 2, 1], [1, 2, 3]])
    np.testing.assert_array_equal(
        patch.initial_state.position,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
    )
    np.testing.assert_allclose(patch.vertex_mass, [1.0, 2.0, 2.0, 1.0])
    np.testing.assert_allclose(patch.model.inverse_mass, [1.0, 0.5, 0.5, 1.0])
    assert patch.structural_edge_count == 4
    assert len(patch.constraints.indices) == 6


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rectangular_grid_conserves_area_mass_and_has_unique_constraints(dtype):
    rows, columns, spacing, density = 3, 5, 0.2, 1.7
    patch = build(rows, columns, spacing=spacing, density=density, dtype=dtype)
    assert patch.model.particle_count == rows * columns
    assert patch.triangles.shape == (2 * (rows - 1) * (columns - 1), 3)
    edges = patch.constraints.indices
    assert len(edges) == rows * (columns - 1) + columns * (rows - 1) + 2 * (rows - 1) * (
        columns - 1
    )
    assert len({tuple(sorted(edge)) for edge in edges}) == len(edges)
    assert patch.vertex_mass.sum() == pytest.approx(
        (rows - 1) * (columns - 1) * spacing**2 * density, rel=1e-6
    )
    positions = patch.initial_state.position
    triangles = positions[patch.triangles].astype(np.float64)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert np.all(normals[:, 1] > 0.0)
    np.testing.assert_allclose(normals[:, [0, 2]], 0.0, atol=0.0)
    geometric_length = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    np.testing.assert_allclose(geometric_length, patch.constraints.rest_length, rtol=1e-6)
    assert (
        positions.dtype == patch.vertex_mass.dtype == patch.constraints.rest_length.dtype == dtype
    )


def test_structural_and_shear_edges_have_separate_compliance():
    patch = build(2, 2, spacing=0.5, structural_compliance=0.01, shear_compliance=0.03)
    split = patch.structural_edge_count
    np.testing.assert_allclose(patch.constraints.compliance[:split], 0.01)
    np.testing.assert_allclose(patch.constraints.compliance[split:], 0.03)
    np.testing.assert_allclose(patch.constraints.rest_length[:split], 0.5)
    np.testing.assert_allclose(patch.constraints.rest_length[split:], np.sqrt(0.5))


def test_pins_preserve_auditable_physical_mass_and_are_idempotent():
    patch = build(2, 2, spacing=1.0, density=6.0, pinned=[0, 3, 0])
    np.testing.assert_allclose(patch.vertex_mass, [1.0, 2.0, 2.0, 1.0])
    np.testing.assert_allclose(patch.model.inverse_mass, [0.0, 0.5, 0.5, 0.0])
    assert patch.vertex_mass.sum() == pytest.approx(6.0)


def test_builder_outputs_do_not_alias_caller_or_other_patches():
    origin = np.array([1.0, 2.0, 3.0])
    patch = build(2, 2, origin=origin)
    origin[:] = 100.0
    np.testing.assert_array_equal(patch.initial_state.position[0], [1.0, 2.0, 3.0])
    assert not patch.triangles.flags.writeable
    assert not patch.vertex_mass.flags.writeable
    other = build(2, 2, origin=(1.0, 2.0, 3.0))
    patch.initial_state.position[:] = 900.0
    np.testing.assert_array_equal(other.initial_state.position[0], [1.0, 2.0, 3.0])


def test_unpinned_patch_falls_under_uniform_gravity_without_initial_stretch():
    patch = build(3, 3, spacing=0.1)
    before = patch.initial_state.copy()
    result = XPBDSolver(patch.constraints).step(patch.model, before, StepInputs.zeros(9), 0.1)
    assert result.report.accepted
    np.testing.assert_allclose(result.state.position[:, 1], before.position[:, 1] - 0.0981)
    np.testing.assert_array_equal(patch.initial_state.position, before.position)


def test_flat_patch_plane_contact_is_finite_and_does_not_penetrate():
    patch = build(3, 3, spacing=0.1, origin=(0.0, 0.1, 0.0))
    result = XPBDSolver(patch.constraints, plane=Plane()).step(
        patch.model, patch.initial_state, StepInputs.zeros(9), 0.2
    )
    assert result.report.accepted
    np.testing.assert_allclose(result.state.position[:, 1], 0.0, atol=1e-14)
    np.testing.assert_allclose(result.state.velocity[:, 1], 0.0, atol=1e-14)


@pytest.mark.parametrize("rows, columns", [(1, 2), (2, 1), (0, 2), (-1, 3), (True, 3), (2.5, 3)])
def test_invalid_grid_dimensions_are_rejected(rows, columns):
    builder = getattr(phymeo, "cloth_grid", None)
    assert builder is not None, "cloth_grid is not implemented"
    with pytest.raises((TypeError, ValueError)):
        builder(rows, columns)


@pytest.mark.parametrize(
    "options",
    [
        {"spacing": 0.0},
        {"spacing": -1.0},
        {"spacing": np.inf},
        {"spacing": True},
        {"density": 0.0},
        {"density": -1.0},
        {"density": np.nan},
        {"structural_compliance": -1.0},
        {"shear_compliance": np.inf},
        {"dtype": np.int32},
        {"origin": [0.0, 1.0]},
        {"origin": [0.0, np.nan, 0.0]},
    ],
)
def test_invalid_grid_parameters_are_rejected(options):
    builder = getattr(phymeo, "cloth_grid", None)
    assert builder is not None, "cloth_grid is not implemented"
    with pytest.raises((TypeError, ValueError)):
        builder(2, 2, **options)


@pytest.mark.parametrize("pinned", [[-1], [4], [1.0], [True]])
def test_invalid_pins_are_rejected(pinned):
    builder = getattr(phymeo, "cloth_grid", None)
    assert builder is not None, "cloth_grid is not implemented"
    with pytest.raises((TypeError, ValueError)):
        builder(2, 2, pinned=pinned)


@pytest.mark.parametrize(
    "options",
    [
        {"dtype": np.float32, "structural_compliance": 1e-50},
        {"dtype": np.float32, "shear_compliance": 1e40},
        {"dtype": np.float32, "density": 1e-40},
        {"dtype": np.float32, "origin": (1e20, 0.0, 1e20)},
        {"dtype": np.float64, "spacing": 1e200},
    ],
)
def test_unrepresentable_physics_or_degenerate_geometry_is_rejected(options):
    builder = getattr(phymeo, "cloth_grid", None)
    assert builder is not None, "cloth_grid is not implemented"
    with pytest.raises(ValueError):
        builder(2, 2, **options)
