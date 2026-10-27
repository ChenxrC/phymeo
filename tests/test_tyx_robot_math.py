"""Analytic and independent FP64 checks of actual FP32 robot kernels."""

import importlib
import importlib.util

import numpy as np
import pytest
from test_robot_dynamics import planar_bias, planar_mass, polar_model
from test_robot_kinematics import branched_model, robot_model

from phymeo.core.capabilities import SolverFailure
from phymeo.core.robot import RobotInputs, RobotState
from phymeo.dynamics import robot_dynamics as reference
from phymeo.dynamics import robot_kinematics as geometry_reference

pytestmark = pytest.mark.tyx


def implementation():
    name = "phymeo.backends.tyx_robot_math"
    assert importlib.util.find_spec(name) is not None, "P6 robot device mathematics missing"
    return importlib.import_module(name).TyxRobotMath


@pytest.mark.parametrize(
    "q,v", [([0.0, 0.0], [0.0, 0.0]), ([0.3, -0.5], [0.7, 0.2]), ([-0.6, 0.4], [-0.2, 0.5])]
)
def test_device_two_rods_match_planar_equations(q, v, tyx_mode, tmp_path):
    q, v = np.array(q), np.array(v)
    loads = RobotInputs.zeros(2)
    loads.effort[:] = [0.1, -0.2]
    with implementation()(robot_model(), tyx_mode, cache_root=tmp_path) as engine:
        np.testing.assert_allclose(engine.mass(q), planar_mass(q), atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(engine.bias(q, v), planar_bias(q, v), atol=2e-4, rtol=2e-4)
        expected = np.linalg.solve(planar_mass(q), loads.effort - planar_bias(q, v))
        np.testing.assert_allclose(engine.acceleration(q, v, loads), expected, atol=2e-4, rtol=2e-4)
        assert engine.stats()["launches"] > 0


def test_device_moving_prismatic_coriolis(tyx_mode, tmp_path):
    theta, radius, omega, radial = 0.4, 0.7, 0.6, -0.2
    with implementation()(polar_model(), tyx_mode, cache_root=tmp_path) as engine:
        q, v = np.array([theta, radius]), np.array([omega, radial])
        np.testing.assert_allclose(
            engine.mass(q), np.diag([2 + 3 * radius**2, 3]), atol=2e-4, rtol=2e-4
        )
        np.testing.assert_allclose(
            engine.bias(q, v),
            [6 * radius * omega * radial, -3 * radius * omega**2],
            atol=2e-4,
            rtol=2e-4,
        )


@pytest.mark.parametrize("model_factory", [lambda: robot_model(1), robot_model, branched_model])
def test_seeded_geometry_mass_bias_acceleration(tyx_mode, tmp_path, model_factory):
    model = model_factory()
    rng = np.random.default_rng(681)
    with implementation()(model, tyx_mode, cache_root=tmp_path) as engine:
        for _ in range(32):
            q = rng.uniform(-0.6, 0.6, model.dof_count)
            v = rng.uniform(-0.4, 0.4, model.dof_count)
            state = RobotState(q, v)
            inputs = RobotInputs(
                rng.uniform(-0.2, 0.2, model.dof_count),
                rng.uniform(-0.3, 0.3, (model.dof_count, 3)),
                rng.uniform(-0.1, 0.1, (model.dof_count, 3)),
            )
            actual, expected = engine.geometry(q), geometry_reference.forward_kinematics(model, q)
            for key in ("position", "rotation", "axis", "com_position"):
                np.testing.assert_allclose(
                    getattr(actual, key), getattr(expected, key), atol=2e-5, rtol=2e-5
                )
            points = rng.uniform(-0.2, 0.2, (model.dof_count, 3))
            indices = np.arange(model.dof_count)
            for a, b in zip(
                engine.jacobians(q, indices, points),
                geometry_reference.point_jacobians(model, q, indices, points),
                strict=True,
            ):
                np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(
                engine.mass(q), reference.mass_matrix(model, q), atol=2e-4, rtol=2e-4
            )
            np.testing.assert_allclose(
                engine.bias(q, v), reference.bias_forces(model, state), atol=2e-4, rtol=2e-4
            )
            np.testing.assert_allclose(
                engine.generalized(q, inputs.effort, inputs.force, inputs.torque),
                reference.generalized_forces(model, state, inputs),
                atol=2e-4,
                rtol=2e-4,
            )
            np.testing.assert_allclose(
                engine.acceleration(q, v, inputs),
                reference.forward_dynamics(model, state, inputs),
                atol=2e-4,
                rtol=2e-4,
            )


def test_no_cpu_robot_math_supplies_device_results(tyx_mode, tmp_path, monkeypatch):
    model = robot_model()
    q, v = np.array([0.3, -0.5]), np.array([0.7, 0.2])
    expected = np.linalg.solve(planar_mass(q), -planar_bias(q, v))

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU robot operator used by device implementation")

    with implementation()(model, tyx_mode, cache_root=tmp_path) as engine:
        for module, names in (
            (
                reference,
                (
                    "mass_matrix",
                    "bias_forces",
                    "inverse_dynamics",
                    "generalized_forces",
                    "forward_dynamics",
                    "solve_mass",
                    "forward_kinematics",
                    "point_jacobians",
                ),
            ),
            (geometry_reference, ("forward_kinematics", "point_jacobians")),
        ):
            for name in names:
                monkeypatch.setattr(module, name, forbidden)
        np.testing.assert_allclose(
            engine.acceleration(q, v, RobotInputs.zeros(2)), expected, atol=2e-4, rtol=2e-4
        )


def test_device_shared_ancestor_jacobian_and_owned_results(tyx_mode, tmp_path):
    model = robot_model()
    q = np.zeros(2)
    indices = np.array([0, 1, 1])
    points = np.tile([0.0, -0.5, 0.0], (3, 1))
    with implementation()(model, tyx_mode, cache_root=tmp_path) as engine:
        linear, angular = engine.jacobians(q, indices, points)
        np.testing.assert_allclose(linear[:, 0], [[0.5, 0], [1.0, 0.5], [1.0, 0.5]], atol=2e-5)
        load = np.array([0.7, -0.2])
        answer = engine.solve(q, load)
        np.testing.assert_allclose(planar_mass(q) @ answer, load, atol=2e-5)
        linear[:] = 999
        np.testing.assert_allclose(
            engine.jacobians(q, indices, points)[0][:, 0],
            [[0.5, 0], [1.0, 0.5], [1.0, 0.5]],
            atol=2e-5,
        )
        assert angular.dtype == np.float64


@pytest.mark.parametrize("q", [np.array([np.inf, 0.0]), np.zeros(3), np.array([1e40, 0.0])])
def test_invalid_configuration_does_not_return_candidate(q, tyx_mode, tmp_path):
    with implementation()(robot_model(), tyx_mode, cache_root=tmp_path) as engine:
        with pytest.raises((ValueError, TypeError)):
            engine.geometry(q)


def test_point_jacobian_matches_independent_position_difference(tyx_mode, tmp_path):
    model = branched_model()
    q, v = np.array([0.3, -0.2, 0.4, 0.1]), np.array([-0.1, 0.2, 0.3, -0.4])
    indices = np.array([3, 1, 3])
    points = np.array([[0.1, -0.2, 0.3], [-0.1, 0.2, 0.05], [0.1, -0.2, 0.3]])
    positions = []
    for sign in (-1, 1):
        g = geometry_reference.forward_kinematics(model, q + sign * 1e-6 * v)
        positions.append(g.position[indices] + np.einsum("nij,nj->ni", g.rotation[indices], points))
    independent_velocity = (positions[1] - positions[0]) / 2e-6
    with implementation()(model, tyx_mode, cache_root=tmp_path) as engine:
        linear, _ = engine.jacobians(q, indices, points)
        np.testing.assert_allclose(linear @ v, independent_velocity, atol=2e-5, rtol=2e-5)
        # Opposite impulses at duplicate attachments cancel on every ancestor.
        loads = np.array([[0.3, -0.4, 0.2], [0.0, 0.0, 0.0], [-0.3, 0.4, -0.2]])
        projected = engine.matmul(linear.reshape(-1, model.dof_count).T, loads.reshape(-1, 1))
        np.testing.assert_allclose(projected, 0.0, atol=2e-7, rtol=0)


def test_world_loads_match_independent_polar_power(tyx_mode, tmp_path):
    q, v = np.array([0.0, 0.7]), np.array([0.6, -0.2])
    effort = np.array([0.1, 0.2])
    force = np.array([[9.0, 8.0, 0.0], [2.0, 3.0, 0.0]])
    torque = np.array([[0.0, 0.0, 0.4], [0.0, 0.0, -0.1]])
    with implementation()(polar_model(), tyx_mode, cache_root=tmp_path) as engine:
        load = engine.generalized(q, effort, force, torque)
        expected_power = 0.1 * 0.6 + 0.2 * (-0.2) + 2 * (-0.2) + 3 * (0.7 * 0.6) + (0.4 - 0.1) * 0.6
        assert load @ v == pytest.approx(expected_power, abs=2e-5, rel=2e-5)


def test_quantized_mass_loses_spd_and_must_not_be_regularized(tyx_mode, tmp_path):
    model = robot_model(
        com=np.zeros((2, 3)),
        joint_position=np.zeros((2, 3)),
        inertia_diagonal=np.array([[1e-10] * 3, [1e10] * 3]),
    )
    with implementation()(model, tyx_mode, cache_root=tmp_path) as engine:
        with pytest.raises(SolverFailure, match="positive definite"):
            engine.solve(np.zeros(2), np.array([1.0, -1.0]))


def test_constructor_conversion_failure_closes_runtime(tyx_mode, tmp_path, monkeypatch):
    cls = implementation()
    closed = []
    original_close = cls.close

    def close(self):
        closed.append(self)
        return original_close(self)

    monkeypatch.setattr(cls, "close", close)
    model = robot_model(1, mass=np.array([1e40]))
    with pytest.raises(ValueError, match="mass"):
        cls(model, tyx_mode, cache_root=tmp_path)
    assert len(closed) == 1
    with pytest.raises(RuntimeError):
        closed[0].geometry(np.zeros(1))
