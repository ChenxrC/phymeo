"""Whole-graph Signorini, Coulomb, reaction and energy checks at fixed geometry."""

import importlib

import numpy as np
import pytest

from phymeo import (
    MultiDomainPointOperator,
    ParticleModel,
    ParticlePointPort,
    ParticleState,
    SolverFailure,
)


def solve(graph, normals, mu=0.5, **kwargs):
    k = len(normals)
    return importlib.import_module("phymeo.constraints.contact_impulse").solve_contact_impulses(
        graph,
        np.asarray(normals, dtype=np.float64),
        np.full(k, float(mu)),
        np.full(k, float(mu)),
        **kwargs,
    )


def test_contact_impulse_public_exports():
    import phymeo

    module = importlib.import_module("phymeo.constraints.contact_impulse")
    assert phymeo.solve_contact_impulses is module.solve_contact_impulses
    assert phymeo.ContactImpulseResult is module.ContactImpulseResult


def test_installed_surface_contact_smoke_uses_public_physical_apis():
    from test_articulated_measurement import script

    result = script("smoke_installed").check_surface_contact()
    np.testing.assert_allclose(result["velocity"], [[1.5, 0, 0]], atol=2e-9)
    np.testing.assert_allclose(result["impulse"], [[-1, 2, 0]], atol=2e-9)


def node(name, inverse_mass, velocity, position=None):
    position = np.zeros((1, 3)) if position is None else np.array([position], dtype=float)
    return ParticlePointPort(
        ParticleModel(np.array([float(inverse_mass)]), np.zeros(3)),
        ParticleState(position, np.array([velocity], dtype=float)),
        [0],
        domain=name,
    )


def plane_graph(velocity):
    ports = {"body": node("body", 0.5, velocity), "floor": node("floor", 0, [0, 0, 0])}
    return MultiDomainPointOperator(ports, [("body", 0)], [("floor", 0)])


@pytest.mark.parametrize(
    "velocity,expected_p,expected_v,regime",
    [
        ([2.0, -1, 0], [-1.0, 2, 0], [1.5, 0, 0], "sliding"),
        ([0.2, -1, 0], [-0.4, 2, 0], [0.0, 0, 0], "sticking"),
        ([2.0, 1, 0], [0.0, 0, 0], [2.0, 1, 0], "open"),
    ],
)
def test_plane_analytic_stick_slide_and_separation(velocity, expected_p, expected_v, regime):
    graph = plane_graph(velocity)
    result = solve(graph, [[0.0, 1, 0]])
    np.testing.assert_allclose(result.world_impulses, [expected_p], atol=2e-9)
    np.testing.assert_allclose(result.states["body"].velocity, [expected_v], atol=2e-9)
    np.testing.assert_allclose(result.relative_velocity_after, [expected_v], atol=2e-9)
    assert result.regimes == (regime,)
    assert result.residual_mps <= 1e-10
    expected_energy = np.dot(expected_v, expected_v) - np.dot(velocity, velocity)
    assert result.energy_change == pytest.approx(expected_energy, abs=1e-9)
    np.testing.assert_array_equal(result.states["floor"].velocity, [[0, 0, 0]])
    np.testing.assert_array_equal(graph.snapshot()["body"].velocity, [velocity])


def pinch_graph():
    from test_robot_kinematics import robot_model

    from phymeo import RobotPointPort, RobotState

    model = robot_model(
        1,
        joint_types=("prismatic",),
        joint_axis=np.array([[1.0, 0, 0]]),
        mass=np.array([1.0]),
        gravity=np.zeros(3),
    )
    ports = {
        name: RobotPointPort(
            model, RobotState(np.zeros(1), np.array([velocity])), [0], np.zeros((1, 3)), domain=name
        )
        for name, velocity in (("A", 0.2), ("B", -0.2))
    }
    ports["cloth"] = node("cloth", 4, [0, -0.05, 0])
    return MultiDomainPointOperator(ports, [("A", 0), ("B", 0)], [("cloth", 0), ("cloth", 0)])


@pytest.mark.parametrize("mu,expected_vertical", [(0.8, 0.0), (0.0001, -0.04984)])
def test_two_dynamic_jaws_share_cloth_response_and_low_friction_slips(mu, expected_vertical):
    graph = pinch_graph()
    result = solve(graph, [[-1.0, 0, 0], [1.0, 0, 0]], mu)
    np.testing.assert_allclose(result.normal_impulses, [0.2, 0.2], atol=2e-9)
    np.testing.assert_allclose(
        result.states["cloth"].velocity, [[0, expected_vertical, 0]], atol=2e-9
    )
    for name in ("A", "B"):
        np.testing.assert_allclose(result.states[name].v, 0, atol=2e-9)
        np.testing.assert_array_equal(result.states[name].q, [0])
    # Fixed robot bases absorb vertical reaction; this is an open system.
    cloth_momentum = 0.25 * (expected_vertical + 0.05)
    assert np.sum(result.world_impulses[:, 1]) == pytest.approx(-cloth_momentum, abs=2e-9)
    after_energy = sum(
        float(result.states[name].v @ result.states[name].v) / 2 for name in ("A", "B")
    )
    after_energy += float(np.sum(result.states["cloth"].velocity ** 2)) / 8
    before_energy = 0.2**2 + 0.05**2 / 8
    assert result.energy_change == pytest.approx(after_energy - before_energy, abs=2e-9)
    assert result.energy_change < 0
    assert result.residual_mps <= 1e-10


def test_anisotropic_rigid_response_has_known_coupled_normal_tangent_solution():
    from phymeo import RigidBodyModel, RigidBodyState, RigidPointPort

    offset = np.array([0.2, 0.3, 0.4])
    k = np.array(
        [
            [1 + 0.09 / 4 + 0.16 / 3, -0.06 / 4, -0.08 / 3],
            [-0.06 / 4, 1 + 0.16 / 2 + 0.04 / 4, -0.12 / 2],
            [-0.08 / 3, -0.12 / 2, 1 + 0.09 / 2 + 0.04 / 3],
        ]
    )
    impulse, after = np.array([-0.6, -0.8, 1.0]), np.array([1.2, 1.6, 0.0])
    free = after - k @ impulse
    model = RigidBodyModel(np.ones(1), np.array([[2.0, 3, 4]]), np.zeros(3))
    state = RigidBodyState(
        np.zeros((1, 3)), np.array([[0.0, 0, 0, 1]]), free[None], np.zeros((1, 3))
    )
    body = RigidPointPort(model, state, [0], offset[None], domain="body")
    floor = node("floor", 0, [0, 0, 0], offset)
    graph = MultiDomainPointOperator({"body": body, "floor": floor}, [("body", 0)], [("floor", 0)])
    result = solve(graph, [[0.0, 0, 1]], mu=1)
    np.testing.assert_allclose(result.world_impulses, [impulse], atol=3e-9)
    np.testing.assert_allclose(result.relative_velocity_after, [after], atol=3e-9)
    np.testing.assert_allclose(result.states["body"].velocity, [free + impulse], atol=3e-9)
    expected_angular = np.cross(offset, impulse) / [2, 3, 4]
    np.testing.assert_allclose(
        result.states["body"].angular_velocity, [expected_angular], atol=3e-9
    )
    assert result.regimes == ("sliding",)


def test_low_iteration_budget_does_not_return_uncoupled_partial_solution():
    with pytest.raises(SolverFailure, match="converg"):
        solve(pinch_graph(), [[-1.0, 0, 0], [1.0, 0, 0]], 0.8, max_iterations=1)


def test_result_arrays_are_owned_and_total_impulse_is_applied_once():
    graph = plane_graph([2.0, -1, 0])
    result = solve(graph, [[0.0, 1, 0]])
    once = graph.apply_impulses(result.world_impulses)
    np.testing.assert_array_equal(result.states["body"].velocity, once["body"].velocity)
    for value in (
        result.world_impulses,
        result.relative_velocity_after,
        result.contact_basis,
        result.normal_impulses,
        result.tangent_impulses,
    ):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value[:] = 42
    result.states["body"].velocity[:] = 10
    np.testing.assert_array_equal(graph.snapshot()["body"].velocity, [[2, -1, 0]])
    with pytest.raises(TypeError):
        result.states["body"] = once["body"]


def test_empty_graph_and_zero_friction():
    result = solve(MultiDomainPointOperator({}, [], []), np.empty((0, 3)))
    assert result.world_impulses.shape == (0, 3)
    assert dict(result.states) == {}
    assert result.iterations == 0 and result.residual_mps == 0
    result = solve(plane_graph([2.0, -1, 0]), [[0.0, 1, 0]], mu=0)
    np.testing.assert_allclose(result.world_impulses, [[0, 2, 0]], atol=1e-12)
    np.testing.assert_allclose(result.relative_velocity_after, [[2, 0, 0]], atol=1e-12)


@pytest.mark.parametrize(
    "normals",
    [
        np.array([[0.0, 2, 0]]),
        np.array([[0.0, 0, 0]]),
        np.array([[0.0, np.nan, 0]]),
        np.ones((2, 3)),
    ],
)
def test_invalid_normal_geometry_rejected(normals):
    with pytest.raises((ValueError, TypeError)):
        solve(plane_graph([0, -1, 0]), normals)


@pytest.mark.parametrize(
    "settings",
    [
        dict(velocity_tolerance=0),
        dict(velocity_tolerance=1),
        dict(response_tolerance=0),
        dict(response_tolerance=1),
        dict(position_tolerance=-1),
        dict(max_iterations=True),
        dict(max_iterations=0),
        dict(max_iterations=2.5),
    ],
)
def test_invalid_settings_rejected(settings):
    with pytest.raises((ValueError, TypeError)):
        solve(plane_graph([0, -1, 0]), [[0.0, 1, 0]], **settings)


def test_separated_points_cannot_be_treated_as_closed_contact():
    graph = MultiDomainPointOperator(
        {"a": node("a", 1, [0, -1, 0], [0, 1, 0]), "b": node("b", 0, [0, 0, 0])},
        [("a", 0)],
        [("b", 0)],
    )
    with pytest.raises(ValueError, match="point|closed|position"):
        solve(graph, [[0.0, 1, 0]])


def test_actual_final_response_must_match_the_solved_physics():
    class Inconsistent(MultiDomainPointOperator):
        calls = 0
        candidates = 0

        def response(self, p):
            self.calls += 1
            return super().response(p) if self.calls <= 3 else np.zeros_like(p)

        def apply_impulses(self, p):
            self.candidates += 1
            return super().apply_impulses(p)

    graph = Inconsistent(
        {"a": node("a", 1, [0, -1, 0]), "b": node("b", 0, [0, 0, 0])}, [("a", 0)], [("b", 0)]
    )
    with pytest.raises(SolverFailure):
        solve(graph, [[0.0, 1, 0]], max_iterations=3)
    assert graph.candidates == 0


def test_last_candidate_failure_does_not_change_any_port():
    class Broken(ParticlePointPort):
        def apply_impulses(self, p):
            raise SolverFailure("final candidate failed")

    a = node("a", 1, [0, -1, 0])
    z = Broken(
        ParticleModel(np.zeros(1), np.zeros(3)),
        ParticleState(np.zeros((1, 3)), np.zeros((1, 3))),
        [0],
        domain="z",
    )
    graph = MultiDomainPointOperator({"a": a, "z": z}, [("a", 0)], [("z", 0)])
    with pytest.raises(SolverFailure, match="final candidate"):
        solve(graph, [[0.0, 1, 0]])
    np.testing.assert_array_equal(a.snapshot().velocity, [[0, -1, 0]])


def test_large_finite_response_keeps_representable_normal_solution():
    graph = MultiDomainPointOperator(
        {"a": node("a", 1e308, [0, -1e308, 0]), "b": node("b", 0, [0, 0, 0])},
        [("a", 0)],
        [("b", 0)],
    )
    result = solve(graph, [[0.0, 1, 0]])
    np.testing.assert_array_equal(result.world_impulses, [[0, 1, 0]])
    np.testing.assert_array_equal(result.states["a"].velocity, [[0, 0, 0]])
    assert result.energy_change == pytest.approx(-5e307)


@pytest.mark.parametrize(
    "matrix", [np.zeros((3, 3)), -np.eye(3), np.array([[1.0, 0.1, 0], [0, 1, 0], [0, 0, 1]])]
)
def test_singular_negative_or_asymmetric_graph_response_rejected(matrix):
    class BadResponse(MultiDomainPointOperator):
        def response(self, p):
            return p @ matrix.T

    graph = BadResponse(
        {"a": node("a", 1, [0, -1, 0]), "b": node("b", 0, [0, 0, 0])}, [("a", 0)], [("b", 0)]
    )
    with pytest.raises(SolverFailure):
        solve(graph, [[0.0, 1, 0]])


@pytest.mark.parametrize(
    "normal,static,dynamic",
    [
        (np.array([[0.0, 1, 0]], np.float32), np.ones(1), np.ones(1)),
        (np.array([[0.0, 1, 0]]), np.ones(1, np.float32), np.ones(1)),
        (np.array([[0.0, 1, 0]]), np.ones(2), np.ones(1)),
        (np.array([[0.0, 1, 0]]), np.ones(1), np.array([-1.0])),
        (np.array([[0.0, 1, 0]]), np.ones(1), np.array([2.0])),
        (np.array([[0.0, 1, 0]]), np.array([np.inf]), np.ones(1)),
    ],
)
def test_raw_contact_array_contract_rejects_wrong_precision_or_coefficients(
    normal, static, dynamic
):
    function = importlib.import_module("phymeo.constraints.contact_impulse").solve_contact_impulses
    with pytest.raises((ValueError, TypeError)):
        function(plane_graph([0, -1, 0]), normal, static, dynamic)
