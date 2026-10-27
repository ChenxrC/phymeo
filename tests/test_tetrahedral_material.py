"""Continuum energy/force, objectivity and valid volume, independently checked."""

import importlib
from dataclasses import fields

import numpy as np
import pytest

from phymeo.core.capabilities import SolverFailure


def model_api():
    return importlib.import_module("phymeo.core.tetrahedral").TetrahedralModel


def material():
    return importlib.import_module("phymeo.materials.stvk")


def tetra_parameters():
    return dict(
        rest_position=np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]]),
        tetrahedra=np.array([[0, 1, 2, 3]]),
        density=6.0,
        young_modulus=1000.0,
        poisson_ratio=0.3,
        fixed=np.array([True, False, False, False]),
        gravity=np.zeros(3),
    )


def unit_tetra(**overrides):
    values = tetra_parameters()
    values.update(overrides)
    return model_api()(**values)


def test_unit_tetra_owns_reference_material_mass_and_fixed_support():
    values = tetra_parameters()
    model = model_api()(**values)
    assert model.particle_count == 4
    np.testing.assert_allclose(model.volume, [1 / 6], atol=1e-15)
    np.testing.assert_allclose(model.mass, [0.25] * 4, atol=1e-15)
    np.testing.assert_allclose(model.inverse_mass, [0.0, 4, 4, 4], atol=1e-14)
    np.testing.assert_allclose(model.rest_inverse, [np.eye(3)], atol=1e-15)
    assert model.lame_lambda == pytest.approx(1000 * 0.3 / (1.3 * 0.4))
    assert model.lame_mu == pytest.approx(1000 / 2.6)
    for field in fields(model):
        value = getattr(model, field.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
            if field.name in values:
                assert not np.shares_memory(value, values[field.name])
    values["rest_position"][:] = 42
    np.testing.assert_array_equal(model.rest_position, tetra_parameters()["rest_position"])


@pytest.mark.parametrize("angle", [0.0, 0.4, 1.7])
def test_rigid_motion_has_zero_energy_force_and_rotational_objectivity(angle):
    model, law = unit_tetra(), material()
    # Rodrigues around a non-coordinate axis, plus translation.
    axis = np.array([1.0, 2, 3]) / np.sqrt(14)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    rotation = (
        np.eye(3) * np.cos(angle)
        + (1 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * skew
    )
    points = model.rest_position @ rotation.T + np.array([0.7, -0.3, 1.2])
    assert abs(law.elastic_energy(model, points)) <= 1e-20
    assert np.max(np.abs(law.elastic_forces(model, points))) <= 1e-10
    np.testing.assert_allclose(law.deformation_gradient(model, points), [rotation], atol=1e-14)
    stretched = model.rest_position @ np.diag([1.1, 0.95, 1.02])
    original_force = law.elastic_forces(model, stretched)
    transformed = stretched @ rotation.T + [2.0, -0.4, 0.3]
    np.testing.assert_allclose(
        law.elastic_forces(model, transformed), original_force @ rotation.T, atol=1e-10
    )
    assert law.elastic_energy(model, transformed) == pytest.approx(
        law.elastic_energy(model, stretched), rel=1e-12, abs=1e-12
    )


def test_uniform_extension_has_analytic_stress_energy_and_support_force():
    model, law = unit_tetra(), material()
    stretch, young, nu, volume = 1.1, 1000.0, 0.3, 1 / 6
    mu, lam = young / (2 * (1 + nu)), young * nu / ((1 + nu) * (1 - 2 * nu))
    strain = (stretch**2 - 1) / 2
    expected_energy = volume * (3 * mu * strain**2 + lam / 2 * (3 * strain) ** 2)
    traction = stretch * (2 * mu + 3 * lam) * strain
    force = np.array([[1.0, 1, 1], [-1.0, 0, 0], [0.0, -1, 0], [0.0, 0, -1]]) * volume * traction
    points = model.rest_position * stretch
    assert law.elastic_energy(model, points) == pytest.approx(expected_energy, rel=1e-12)
    np.testing.assert_allclose(law.elastic_forces(model, points), force, atol=1e-12)
    assert np.linalg.norm(force[model.fixed]) > 1  # Support load is retained before BCs.


@pytest.mark.parametrize("nu", [-0.9, -1 + 1e-10, np.nextafter(-1.0, 0.0)])
@pytest.mark.parametrize("stretch", [0.75, 1.1, 1.25])
def test_auxetic_hydrostatic_response_does_not_cancel_to_false_equilibrium(nu, stretch):
    model, law = unit_tetra(poisson_ratio=nu), material()
    strain = (stretch * stretch - 1) / 2
    # Independent bulk modulus expression avoids the implementation's Lame terms.
    bulk = 1000.0 / (3 * (1 - 2 * nu))
    expected_energy = (1 / 6) * bulk / 2 * (3 * strain) ** 2
    traction = stretch * (1000.0 / (1 - 2 * nu)) * strain
    expected_force = np.array([[1.0, 1, 1], [-1.0, 0, 0], [0.0, -1, 0], [0.0, 0, -1]]) * (
        traction / 6
    )
    position = model.rest_position * stretch
    assert law.elastic_energy(model, position) == pytest.approx(expected_energy, rel=1e-12)
    np.testing.assert_allclose(law.elastic_forces(model, position), expected_force, rtol=1e-12)


def test_auxetic_stretched_tetra_runtime_must_not_accept_zero_response():
    from phymeo import (
        ParticleState,
        SceneInputs,
        SceneState,
        StepInputs,
        make_tetrahedral_runtime,
    )

    model = unit_tetra(poisson_ratio=np.nextafter(-1.0, 0.0))
    state = ParticleState(model.rest_position * 1.25, np.zeros((4, 3)))
    dt = 1e-12
    with make_tetrahedral_runtime(model) as runtime:
        result = runtime.step(
            SceneState({"soft": state}), SceneInputs({"soft": StepInputs.zeros(4)}), dt
        )
    assert result.report.accepted
    expected_v = np.zeros((4, 3))
    expected_v[1:] = -np.eye(3) * 19.53125 * 4 * dt
    np.testing.assert_allclose(
        result.state.domains["soft"].velocity, expected_v, rtol=1e-12, atol=0
    )
    assert material().elastic_energy(model, state.position) == pytest.approx(6.591796875)


@pytest.mark.parametrize("nu", [-0.95, -0.4, 0.3])
def test_isochoric_simple_shear_has_nonzero_independent_energy_and_forces(nu):
    model, law = unit_tetra(poisson_ratio=nu), material()
    shear = 0.2
    transform = np.array([[1.0, shear, 0], [0.0, 1, 0], [0.0, 0, 1]])
    points = model.rest_position @ transform.T
    mu, lam = 1000 / (2 * (1 + nu)), 1000 * nu / ((1 + nu) * (1 - 2 * nu))
    expected_energy = (mu * (shear**2 / 2 + shear**4 / 4) + lam * shear**4 / 8) / 6
    # Explicit Piola components from simple shear, not the tested tensor function.
    p = np.array(
        [
            [(mu + lam / 2) * shear**2, mu * shear + (mu + lam / 2) * shear**3, 0.0],
            [mu * shear, (mu + lam / 2) * shear**2, 0.0],
            [0.0, 0.0, lam * shear**2 / 2],
        ]
    )
    expected = np.vstack([p.sum(axis=1), -p.T]) / 6
    assert np.linalg.det(transform) == 1
    assert law.elastic_energy(model, points) == pytest.approx(expected_energy, rel=1e-12)
    np.testing.assert_allclose(law.elastic_forces(model, points), expected, atol=1e-12)


@pytest.mark.parametrize("nu", [-0.95, -0.4, 0.3])
def test_internal_force_matches_energy_gradient_and_balances_force_and_torque(nu):
    model, law = unit_tetra(poisson_ratio=nu), material()
    transform = np.array([[1.07, 0.1, 0.02], [0.03, 0.92, -0.04], [0.01, 0.06, 1.03]])
    points = model.rest_position @ transform.T + [0.1, -0.2, 0.3]
    original = points.copy()
    force, eps = law.elastic_forces(model, points), 1e-6
    derivative = np.zeros_like(points)
    for row in range(4):
        for axis in range(3):
            delta = np.zeros_like(points)
            delta[row, axis] = eps
            derivative[row, axis] = -(
                law.elastic_energy(model, points + delta)
                - law.elastic_energy(model, points - delta)
            ) / (2 * eps)
    np.testing.assert_allclose(force, derivative, atol=2e-7, rtol=2e-6)
    scale = max(1.0, np.linalg.norm(force))
    assert np.linalg.norm(force.sum(axis=0)) <= 1e-12 * scale
    assert np.linalg.norm(np.cross(points, force).sum(axis=0)) <= 1e-12 * scale
    np.testing.assert_array_equal(points, original)
    force[:] = 42
    assert not np.all(law.elastic_forces(model, points) == 42)


@pytest.mark.parametrize(
    "field,value",
    [
        ("density", 0),
        ("density", True),
        ("density", np.inf),
        ("young_modulus", -1),
        ("young_modulus", False),
        ("young_modulus", np.nan),
        ("poisson_ratio", -0.9999999999999999 - 1e-10),
        ("poisson_ratio", 0.5),
        ("poisson_ratio", True),
        ("tetrahedra", np.empty((0, 4), dtype=int)),
        ("tetrahedra", np.array([[0.0, 1, 2, 3]])),
        ("tetrahedra", np.array([[0, 1, 1, 3]])),
        ("tetrahedra", np.array([[0, 1, 2, 4]])),
        ("tetrahedra", np.array([[0, 1, 2, -1]])),
        ("tetrahedra", np.array([[0, 1, 2, 3], [0, 2, 3, 1]])),
        ("tetrahedra", np.array([[0, 2, 1, 3]])),
        ("tetrahedra", np.array([0, 1, 2, 3])),
        ("rest_position", np.zeros((4, 3))),
        ("rest_position", np.zeros((4, 3), np.float32)),
        ("rest_position", np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1e-14]])),
        ("fixed", np.array([1, 0, 0, 0])),
        ("fixed", np.array([True])),
        ("gravity", np.zeros(3, np.float32)),
        ("gravity", np.zeros(2)),
    ],
)
def test_invalid_reference_topology_and_material_rejected(field, value):
    with pytest.raises((ValueError, TypeError)):
        unit_tetra(**{field: value})


def test_unreferenced_node_is_rejected_instead_of_receiving_zero_mass():
    points = np.vstack([tetra_parameters()["rest_position"], [2.0, 2, 2]])
    with pytest.raises(ValueError):
        unit_tetra(rest_position=points, fixed=np.zeros(5, bool))


@pytest.mark.parametrize(
    "transform",
    [
        np.diag([-1.0, 1, 1]),
        np.diag([0.5, 1, 1]),
        np.diag([1.5, 1, 1]),
        np.array([[1.0, 2, 0], [0.0, 1, 0], [0.0, 0, 1]]),
    ],
)
@pytest.mark.parametrize("function", ["deformation_gradient", "elastic_energy", "elastic_forces"])
def test_inverted_or_out_of_domain_material_never_returns_finite_success(transform, function):
    model = unit_tetra()
    with pytest.raises(SolverFailure):
        getattr(material(), function)(model, model.rest_position @ transform.T)


@pytest.mark.parametrize(
    "points", [np.zeros((4, 3), np.float32), np.zeros((5, 3)), np.full((4, 3), np.nan)]
)
def test_invalid_position_storage_rejected(points):
    with pytest.raises((TypeError, ValueError)):
        material().elastic_forces(unit_tetra(), points)


def test_extreme_finite_geometry_material_and_forces_fail_loudly():
    law = material()
    with pytest.raises(SolverFailure):
        unit_tetra(rest_position=tetra_parameters()["rest_position"] * 1e200)
    with pytest.raises(SolverFailure):
        unit_tetra(young_modulus=1e308, poisson_ratio=0.49)
    # E stays finite at construction; finite admissible F produces unrepresentable P.
    model = unit_tetra(
        young_modulus=1e308,
        poisson_ratio=0.3,
        rest_position=tetra_parameters()["rest_position"] * 10,
    )
    with pytest.raises(SolverFailure):
        law.elastic_forces(model, model.rest_position * 1.3)


def test_minimal_block_is_conforming_positive_and_has_correct_lumped_mass():
    builder = importlib.import_module("phymeo.builders.tetrahedral").tetrahedral_block
    model, state = builder(origin=(0.1, -0.1, -0.05), size=(0.1, 0.1, 0.1))
    assert model.rest_position.shape == (18, 3) and model.tetrahedra.shape == (24, 4)
    assert np.all(model.volume > 0) and np.count_nonzero(model.fixed) == 9
    assert model.volume.sum() == pytest.approx(0.001, abs=1e-15)
    assert model.mass.sum() == pytest.approx(0.2, abs=1e-14)
    np.testing.assert_array_equal(model.fixed, model.rest_position[:, 1] == -0.1)
    np.testing.assert_array_equal(state.position, model.rest_position)
    assert not np.shares_memory(state.position, model.rest_position)
    assert np.max(np.abs(material().elastic_forces(model, state.position))) <= 1e-10
    # Each internal face has two incident cells; boundary faces tile all six sides.
    face_counts = {}
    for cell in model.tetrahedra:
        for omitted in range(4):
            key = tuple(sorted(np.delete(cell, omitted)))
            face_counts[key] = face_counts.get(key, 0) + 1
    assert set(face_counts.values()) == {1, 2}
    boundary = [
        model.rest_position[list(face)] for face, count in face_counts.items() if count == 1
    ]
    assert len(boundary) == 32
    assert sum(
        np.linalg.norm(np.cross(face[1] - face[0], face[2] - face[0])) / 2 for face in boundary
    ) == pytest.approx(0.06, abs=1e-14)


def test_shared_node_material_assembly_matches_block_energy_gradient():
    builder = importlib.import_module("phymeo.builders.tetrahedral").tetrahedral_block
    model, state = builder(origin=(0.0, 0.0, 0.0), size=(0.1, 0.1, 0.1))
    # Affine strain is identical in every element; shared-node loads must add.
    transform = np.array([[1.04, 0.08, 0.0], [0.0, 0.97, 0.03], [0.0, 0.0, 1.01]])
    points = state.position @ transform.T
    force, eps = material().elastic_forces(model, points), 1e-6
    gradient = np.zeros_like(points)
    for node in range(len(points)):
        for axis in range(3):
            delta = np.zeros_like(points)
            delta[node, axis] = eps
            gradient[node, axis] = -(
                material().elastic_energy(model, points + delta)
                - material().elastic_energy(model, points - delta)
            ) / (2 * eps)
    np.testing.assert_allclose(force, gradient, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.cross(points, force).sum(axis=0), 0.0, atol=1e-12)
    assert np.max(np.abs(force[model.fixed])) > 0.01


@pytest.mark.parametrize(
    "shape,size",
    [
        ((1, 2, 3), (0.1, 0.1, 0.1)),
        ((3, True, 3), (0.1, 0.1, 0.1)),
        ((3, 2), (0.1, 0.1, 0.1)),
        ((3, 2, 3), (0.1, 0, 0.1)),
        ((3, 2, 3), (0.1, np.inf, 0.1)),
    ],
)
def test_block_rejects_invalid_shape_and_extent(shape, size):
    builder = importlib.import_module("phymeo.builders.tetrahedral").tetrahedral_block
    with pytest.raises((TypeError, ValueError)):
        builder(origin=(0.0, 0, 0), size=size, shape=shape)
