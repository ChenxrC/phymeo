"""Device node operators against analytic mechanics and independent energies."""

import importlib
import importlib.util

import numpy as np
import pytest
from test_tetrahedral_material import unit_tetra

from phymeo.builders.tetrahedral import tetrahedral_block
from phymeo.constraints.distance import DistanceConstraints, project_distances
from phymeo.core.capabilities import SolverFailure
from phymeo.materials import stvk

pytestmark = pytest.mark.tyx


def implementation():
    name = "phymeo.backends.tyx_node_math"
    assert importlib.util.find_spec(name) is not None, "P6 node device mathematics missing"
    return importlib.import_module(name).TyxNodeMath


def test_predict_and_reconstruct_once_with_exact_fixed_entry(tyx_mode, tmp_path):
    x = np.array([[0.123456789, 0.2, 0.3], [1.0, 2.0, 3.0]])
    v = np.array([[0.0, 0.0, 0.0], [0.3, 0.4, -0.2]])
    force, weights, gravity = np.ones((2, 3)), np.array([0.0, 2.0]), np.array([0.0, -9.8, 0.0])
    with implementation()(tyx_mode, cache_root=tmp_path) as engine:
        px, pv = engine.predict(x, v, force, weights, gravity, 0.01)
        np.testing.assert_array_equal(px[0], x[0])
        expected_v = v[1] + 0.01 * (2 * force[1] + gravity)
        np.testing.assert_allclose(pv[1], expected_v, atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(px[1], x[1] + 0.01 * expected_v, atol=2e-6, rtol=2e-5)
        changed = px.copy()
        changed[1] += [0.01, -0.02, 0.03]
        rebuilt = engine.reconstruct(px, pv, changed, weights, 0.01)
        np.testing.assert_allclose(rebuilt[1], pv[1] + [1.0, -2.0, 3.0], atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(rebuilt[0], 0.0)
        np.testing.assert_array_equal(x[0], [0.123456789, 0.2, 0.3])
        assert engine.stats()["completed_launches"] >= 2


def test_duplicate_node_loads_and_zero_response_are_ordered(tyx_mode, tmp_path):
    x = np.array([[0.123456789, 0.0, 0.0], [0.2, 0.3, 0.4]])
    indices, loads = (
        np.array([1, 0, 1]),
        np.array([[0.1, 0.2, 0.3], [99.0, 99.0, 99.0], [-0.1, -0.2, -0.3]]),
    )
    with implementation()(tyx_mode, cache_root=tmp_path) as engine:
        answer = engine.apply_load(x, indices, loads, np.array([0.0, 2.0]))
        np.testing.assert_array_equal(answer[0], x[0])
        np.testing.assert_allclose(answer[1], x[1], atol=2e-7)
        np.testing.assert_array_equal(
            engine.apply_load(x, np.array([], dtype=int), np.zeros((0, 3)), np.ones(2)), x
        )


def test_xpbd_serial_order_and_multiplier_match_reference(tyx_mode, tmp_path):
    constraints = DistanceConstraints(
        np.array([[0, 1], [1, 2], [0, 2]]), np.array([1.0, 1.0, 2.0]), np.zeros(3)
    )
    x = np.array([[0.123456789, 0.0, 0.0], [1.4, 0.2, 0.0], [2.7, 0.1, 0.0]])
    weights, alpha, multiplier = np.array([0.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]), np.zeros(3)
    expected, expected_l = x.copy(), multiplier.copy()
    with implementation()(tyx_mode, cache_root=tmp_path) as engine:
        for _ in range(5):
            project_distances(expected, weights, constraints, alpha, expected_l)
            x, multiplier = engine.project_distances(x, weights, constraints, alpha, multiplier)
        np.testing.assert_allclose(x, expected, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(multiplier, expected_l, atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(x[0], expected[0])


@pytest.mark.parametrize("nu", [0.3, -0.4, -0.9])
@pytest.mark.parametrize("stretch", [0.75, 1.1, 1.25])
def test_material_hydrostatic_analytic_force_energy_and_support(nu, stretch, tyx_mode, tmp_path):
    model = unit_tetra(poisson_ratio=nu)
    strain, volume = (stretch**2 - 1) / 2, 1 / 6
    bulk = 1000 / (3 * (1 - 2 * nu))
    expected_energy = volume * bulk / 2 * (3 * strain) ** 2
    expected_force = (
        np.array([[1.0, 1.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        * volume
        * stretch
        * 3
        * bulk
        * strain
    )
    with implementation()(tyx_mode, cache_root=tmp_path) as engine:
        force, energy = engine.material(model, model.rest_position * stretch)
        np.testing.assert_allclose(force, expected_force, atol=2e-4, rtol=5e-4)
        assert energy == pytest.approx(expected_energy, abs=2e-4, rel=5e-4)
        assert np.linalg.norm(force[model.fixed]) > 0


@pytest.mark.parametrize("nu", [0.3, -0.4, -0.9])
def test_shear_and_shared_node_24_tetrahedra_match_independent_energy_gradient(
    nu, tyx_mode, tmp_path
):
    model, state = tetrahedral_block([0.0, 0.0, 0.0], [0.3, 0.2, 0.3], poisson_ratio=nu)
    x = state.position @ np.array([[1.02, 0.03, 0.0], [0.0, 0.98, 0.01], [0.0, 0.0, 1.01]])
    expected_force, expected_energy = stvk.elastic_forces(model, x), stvk.elastic_energy(model, x)
    with implementation()(tyx_mode, cache_root=tmp_path) as engine:
        force, energy = engine.material(model, x)
        np.testing.assert_allclose(force, expected_force, atol=2e-4, rtol=5e-4)
        assert energy == pytest.approx(expected_energy, abs=2e-4, rel=5e-4)
        eps = 1e-6
        plus, minus = x.copy(), x.copy()
        plus[8, 1] += eps
        minus[8, 1] -= eps
        gradient = -(stvk.elastic_energy(model, plus) - stvk.elastic_energy(model, minus)) / (
            2 * eps
        )
        assert force[8, 1] == pytest.approx(gradient, abs=2e-4, rel=5e-4)
        np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=2e-5)


@pytest.mark.parametrize("angle", [0.0, 0.4, 1.7])
def test_material_rigid_motion(tyx_mode, tmp_path, angle):
    model = unit_tetra()
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    x = model.rest_position @ rotation.T + [0.2, -0.1, 0.3]
    with implementation()(tyx_mode, cache_root=tmp_path) as engine:
        force, energy = engine.material(model, x)
        np.testing.assert_allclose(force, 0.0, atol=2e-4)
        assert abs(energy) < 2e-4


@pytest.mark.parametrize("stretch", [-1.0, 0.5, 1.5])
def test_material_domain_is_not_relaxed(tyx_mode, tmp_path, stretch):
    model = unit_tetra()
    with implementation()(tyx_mode, cache_root=tmp_path) as engine:
        with pytest.raises(SolverFailure, match="domain"):
            engine.material(model, model.rest_position * stretch)


def test_device_material_and_projection_do_not_use_cpu_force_or_sweep(
    tyx_mode, tmp_path, monkeypatch
):
    model = unit_tetra()
    expected = stvk.elastic_forces(model, model.rest_position * 1.1)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU numerical fallback")

    monkeypatch.setattr(stvk, "elastic_forces", forbidden)
    monkeypatch.setattr(stvk, "elastic_energy", forbidden)
    distance_module = importlib.import_module("phymeo.constraints.distance")
    monkeypatch.setattr(distance_module, "project_distances", forbidden)
    constraints = DistanceConstraints(np.array([[0, 1]]), np.array([1.0]), np.zeros(1))
    with implementation()(tyx_mode, cache_root=tmp_path) as engine:
        np.testing.assert_allclose(
            engine.material(model, model.rest_position * 1.1)[0], expected, atol=2e-4, rtol=5e-4
        )
        x, multiplier = engine.project_distances(
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.ones(2),
            constraints,
            np.zeros(1),
            np.zeros(1),
        )
        np.testing.assert_allclose(x, [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]], atol=2e-5)
        np.testing.assert_allclose(multiplier, [-0.5], atol=2e-5)


@pytest.mark.parametrize("nu", [-1 + 1e-10, np.nextafter(-1.0, 0.0)])
def test_extreme_auxetic_pure_volume_keeps_bulk_response(nu, tyx_mode, tmp_path):
    model = unit_tetra(poisson_ratio=nu)
    stretch = 1.25
    expected = 1000 / (3 * (1 - 2 * nu)) / 12 * (3 * (stretch**2 - 1) / 2) ** 2
    with implementation()(tyx_mode, cache_root=tmp_path) as engine:
        force, energy = engine.material(model, model.rest_position * stretch)
        assert energy == pytest.approx(expected, abs=2e-4, rel=5e-4)
        assert np.linalg.norm(force[0]) > 1


def test_empty_operators_and_zero_motion_do_not_invent_load(tyx_mode, tmp_path):
    with implementation()(tyx_mode, cache_root=tmp_path) as engine:
        empty = np.zeros((0, 3))
        for value in engine.predict(empty, empty, empty, np.zeros(0), np.zeros(3), 0.01):
            assert value.shape == (0, 3)
        assert engine.reconstruct(empty, empty, empty, np.zeros(0), 0.01).shape == (0, 3)
        constraints = DistanceConstraints(np.zeros((0, 2), dtype=int), np.zeros(0), np.zeros(0))
        x, multipliers = engine.project_distances(
            empty, np.zeros(0), constraints, np.zeros(0), np.zeros(0)
        )
        assert x.shape == (0, 3) and multipliers.shape == (0,)
        model = unit_tetra()
        force, energy = engine.material(model, model.rest_position)
        np.testing.assert_array_equal(force, 0.0)
        assert energy == 0.0
        x, v = engine.predict(
            model.rest_position, np.zeros((4, 3)), force, model.inverse_mass, np.zeros(3), 0.01
        )
        np.testing.assert_array_equal(x, model.rest_position)
        np.testing.assert_array_equal(v, 0.0)


def test_coincident_edge_and_fixed_inconsistent_edge_match_reference(tyx_mode, tmp_path):
    constraints = DistanceConstraints(np.array([[0, 1]]), np.array([1.0]), np.zeros(1))
    with implementation()(tyx_mode, cache_root=tmp_path) as engine:
        for weights in (np.ones(2), np.zeros(2)):
            x, multiplier = engine.project_distances(
                np.zeros((2, 3)), weights, constraints, np.zeros(1), np.zeros(1)
            )
            expected, lambdas = np.zeros((2, 3)), np.zeros(1)
            project_distances(expected, weights, constraints, np.zeros(1), lambdas)
            np.testing.assert_array_equal(x, expected)
            np.testing.assert_array_equal(multiplier, lambdas)


@pytest.mark.parametrize(
    "weights,indices,loads",
    [
        (np.array([-1.0, 1.0]), np.array([0]), np.ones((1, 3))),
        (np.ones(2), np.array([2]), np.ones((1, 3))),
        (np.ones(2), np.array([0]), np.full((1, 3), 1e-50)),
        (np.ones(2), np.array([0]), np.full((1, 3), np.inf)),
    ],
)
def test_bad_loads_reject_without_mutating_caller(weights, indices, loads, tyx_mode, tmp_path):
    x = np.ones((2, 3))
    with implementation()(tyx_mode, cache_root=tmp_path) as engine:
        with pytest.raises(ValueError):
            engine.apply_load(x, indices, loads, weights)
    np.testing.assert_array_equal(x, np.ones((2, 3)))


def test_material_keeps_resolved_response_with_tiny_transverse_perturbation(tyx_mode, tmp_path):
    model, state = tetrahedral_block([-0.05, 0.0, -0.05], [0.1, 0.1, 0.1])
    x = state.position.copy()
    x[~model.fixed, 1] *= 0.98
    x[8, 0] += 1e-24
    # Node 8 has x=0 in the builder; the perturbation survives quantization.
    assert x[8, 0] != state.position[8, 0]
    expected_force, expected_energy = stvk.elastic_forces(model, x), stvk.elastic_energy(model, x)
    with implementation()(tyx_mode, cache_root=tmp_path) as engine:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            force, energy = engine.material(model, x)
        np.testing.assert_allclose(force, expected_force, atol=2e-4, rtol=5e-4)
        assert energy == pytest.approx(expected_energy, abs=2e-4, rel=5e-4)


def test_material_rejects_scale_with_unsafe_omitted_term_error_bound(tyx_mode, tmp_path):
    model = unit_tetra(young_modulus=1e35)
    with implementation()(tyx_mode, cache_root=tmp_path) as engine:
        with pytest.raises(SolverFailure, match="error budget"):
            engine.material(model, model.rest_position * 1.1)


@pytest.mark.parametrize("strict", [False, True])
def test_material_rejects_volume_amplified_omitted_products(tyx_mode, tmp_path, strict):
    from dataclasses import replace

    base = unit_tetra(young_modulus=1e-25)
    model = replace(base, rest_position=base.rest_position * 3e12)
    position = model.rest_position * 1.1
    expected = stvk.elastic_forces(model, position)
    assert np.max(np.abs(expected)) > 0.04
    before = position.copy()
    with implementation()(tyx_mode, cache_root=tmp_path) as engine:
        with np.errstate(under="raise" if strict else "ignore"):
            with pytest.raises(SolverFailure, match="error budget"):
                engine.material(model, position)
    np.testing.assert_array_equal(position, before)


def test_material_rejects_stiff_rigid_motion_spurious_device_force(tyx_mode, tmp_path):
    model = unit_tetra(young_modulus=1e6)
    c, s = np.cos(0.4), np.sin(0.4)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    position = model.rest_position @ rotation.T + np.array([0.2, -0.1, 0.3])
    assert np.max(np.abs(stvk.elastic_forces(model, position))) < 1e-8
    before = position.copy()
    with implementation()(tyx_mode, cache_root=tmp_path) as engine:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            with pytest.raises(SolverFailure, match="independent.*material"):
                engine.material(model, position)
        assert engine.stats()["completed_launches"] > 0
    np.testing.assert_array_equal(position, before)


@pytest.mark.parametrize("field", ["force", "energy", "nonfinite_force", "nonfinite_energy"])
def test_material_rejects_finite_but_wrong_kernel_output(tyx_mode, tmp_path, monkeypatch, field):
    model = unit_tetra()
    position = model.rest_position.copy()
    position[1, 0] *= 1.02
    with implementation()(tyx_mode, cache_root=tmp_path) as engine:
        call = engine.call

        def corrupted(*args, **kwargs):
            values = call(*args, **kwargs)
            error = np.nan if field.startswith("nonfinite") else 1.0
            if field.endswith("force"):
                values[0][0, 0] += error
            else:
                values[1][0] += error
            return values

        monkeypatch.setattr(engine, "call", corrupted)
        with pytest.raises(SolverFailure, match="independent.*material"):
            engine.material(model, position)
        assert engine.stats()["completed_launches"] > 0
