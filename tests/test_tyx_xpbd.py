"""Actual colored tyx projection checked by independent analytical solutions."""

import numpy as np
import pytest

import phymeo
from phymeo import (
    DistanceConstraints,
    ParticleModel,
    ParticleState,
    Plane,
    StepInputs,
    Stepper,
    XPBDSolver,
)

pytestmark = pytest.mark.tyx


def implementation():
    cls = getattr(phymeo, "TyxXPBDSolver", None)
    assert cls is not None, "TyxXPBDSolver is not implemented"
    return cls


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("colored-xpbd")


def pair(*, compliance=0.0, rest=1.0, dtype=np.float32):
    model = ParticleModel(np.array([1.0, 3.0], dtype=dtype), np.zeros(3))
    state = ParticleState(
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=dtype),
        np.zeros((2, 3), dtype=dtype),
    )
    edges = DistanceConstraints(
        np.array([[0, 1]]), np.array([rest], dtype=dtype), np.array([compliance], dtype=dtype)
    )
    return model, state, StepInputs.zeros(2, dtype=dtype), edges


def test_hard_edge_uses_inverse_mass_and_reconstructs_velocity(tyx_mode, cache):
    model, state, inputs, edges = pair()
    before = state.copy()
    with implementation()(edges, mode=tyx_mode, cache_root=cache, iterations=1) as solver:
        result = solver.step(model, state, inputs, 0.25)
        assert solver.problem_signature() == XPBDSolver(edges).problem_signature()
        assert solver.capabilities.dtypes == frozenset(["float32"])
        assert "distance" in solver.capabilities.features
    assert result.report.accepted
    assert result.report.backend == f"tyx:{tyx_mode}"
    np.testing.assert_allclose(result.state.position[:, 0], [0.25, 1.25], atol=1e-6)
    np.testing.assert_allclose(result.state.velocity[:, 0], [1, -3], atol=1e-6)
    np.testing.assert_array_equal(state.position, before.position)
    np.testing.assert_array_equal(state.velocity, before.velocity)
    assert not np.shares_memory(result.state.position, state.position)


@pytest.mark.parametrize("dt, expected", [(0.1, [0.2, 1.4]), (0.2, [4 / 17, 22 / 17])])
def test_compliance_and_dt_match_analytic_solution(tyx_mode, cache, dt, expected):
    model, state, inputs, edges = pair(compliance=0.01)
    with implementation()(edges, mode=tyx_mode, cache_root=cache) as solver:
        result = solver.step(model, state, inputs, dt)
    assert result.report.accepted
    assert result.report.max_constraint_error > 0.01
    assert result.report.max_residual < 1e-6
    np.testing.assert_allclose(result.state.position[:, 0], expected, atol=3e-7)


def test_coupled_chain_and_repeat_step_reset_multipliers(tyx_mode, cache):
    edges = DistanceConstraints(
        np.array([[0, 1], [1, 2]]), np.ones(2, np.float32), np.full(2, 0.01, np.float32)
    )
    model = ParticleModel(np.ones(3, np.float32), np.zeros(3))
    state = ParticleState(
        np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]], np.float32), np.zeros((3, 3), np.float32)
    )
    with implementation()(
        edges, mode=tyx_mode, cache_root=cache, iterations=100, tolerance=1e-6
    ) as solver:
        result = solver.step(model, state, StepInputs.zeros(3, dtype=np.float32), 0.1)
        replay = solver.step(model, state, StepInputs.zeros(3, dtype=np.float32), 0.1)
    assert result.report.accepted and replay.report.accepted
    # Solve [[3,-1],[-1,3]] lambda=[-1,-1] independently.
    np.testing.assert_allclose(result.state.position[:, 0], [0.5, 2.0, 3.5], atol=2e-6)
    np.testing.assert_array_equal(result.state.position, replay.state.position)


def test_many_disjoint_edges_execute_together_without_shared_writers(tyx_mode, cache):
    pair_count = 64
    count = 2 * pair_count
    edges = DistanceConstraints(
        np.arange(count).reshape(pair_count, 2),
        np.ones(pair_count, np.float32),
        np.zeros(pair_count, np.float32),
    )
    positions = np.zeros((count, 3), np.float32)
    positions[1::2, 0] = 2
    positions[:, 1] = np.repeat(np.arange(pair_count), 2)
    model = ParticleModel(np.tile(np.array([1, 3], np.float32), pair_count), np.zeros(3))
    state = ParticleState(positions, np.zeros_like(positions))
    expected = positions.copy()
    expected[::2, 0] = 0.25
    expected[1::2, 0] = 1.25
    with implementation()(edges, mode=tyx_mode, cache_root=cache, iterations=1) as solver:
        assert solver.color_count == 1
        for _ in range(3):
            result = solver.step(model, state, StepInputs.zeros(count, dtype=np.float32), 0.25)
            assert result.report.accepted
            np.testing.assert_allclose(result.state.position, expected, atol=1e-6)


def test_colored_chain_matches_independent_dense_compliant_system(tyx_mode, cache):
    count = 6
    edges = DistanceConstraints(
        np.column_stack([np.arange(count - 1), np.arange(1, count)]),
        np.ones(count - 1, np.float32),
        np.full(count - 1, 0.01, np.float32),
    )
    positions = np.zeros((count, 3), np.float32)
    positions[:, 0] = 2 * np.arange(count)
    model = ParticleModel(np.ones(count, np.float32), np.zeros(3))
    state = ParticleState(positions, np.zeros_like(positions))
    jacobian = np.zeros((count - 1, count))
    jacobian[np.arange(count - 1), np.arange(count - 1)] = -1
    jacobian[np.arange(count - 1), np.arange(1, count)] = 1
    multipliers = np.linalg.solve(jacobian @ jacobian.T + np.eye(count - 1), -np.ones(count - 1))
    expected = positions[:, 0] + jacobian.T @ multipliers
    with implementation()(
        edges, mode=tyx_mode, cache_root=cache, iterations=100, tolerance=2e-6
    ) as solver:
        assert solver.color_count == 2
        result = solver.step(model, state, StepInputs.zeros(count, dtype=np.float32), 0.1)
    assert result.report.accepted
    np.testing.assert_allclose(result.state.position[:, 0], expected, atol=4e-6)


def test_unsatisfiable_fixed_hard_edge_exhausts_budget(tyx_mode, cache):
    _, state, inputs, edges = pair()
    model = ParticleModel(np.zeros(2, np.float32), np.zeros(3))
    with implementation()(edges, mode=tyx_mode, cache_root=cache, iterations=3) as solver:
        result = solver.step(model, state, inputs, 0.1)
    assert not result.report.accepted
    assert result.report.iterations == 3
    assert result.report.max_residual == pytest.approx(1)
    np.testing.assert_array_equal(result.state.position, state.position)


@pytest.mark.parametrize("rest", [0.0, 1.0])
def test_coincident_points_are_finite_and_deterministic(tyx_mode, cache, rest):
    model, state, inputs, edges = pair(rest=rest)
    state.position[:] = 0
    with implementation()(edges, mode=tyx_mode, cache_root=cache) as solver:
        result = solver.step(model, state, inputs, 0.1)
    assert result.report.accepted
    np.testing.assert_allclose(result.state.position[:, 0], [rest / 4, -3 * rest / 4], atol=1e-6)


def test_fixed_anchor_and_plane_preserve_tangent_and_reject_fixed_penetration(tyx_mode, cache):
    model = ParticleModel(np.array([1, 0], np.float32), np.zeros(3))
    state = ParticleState(
        np.array([[0, 1, 0], [3, -1, 4]], np.float32),
        np.array([[5, -20, 3], [7, 8, 9]], np.float32),
    )
    with implementation()(mode=tyx_mode, cache_root=cache, plane=Plane(), iterations=2) as solver:
        result = solver.step(model, state, StepInputs.zeros(2, dtype=np.float32), 0.1)
    assert not result.report.accepted
    assert result.report.max_residual == pytest.approx(1.0)
    np.testing.assert_allclose(result.state.position[0], [0.5, 0, 0.3], atol=1e-6)
    np.testing.assert_allclose(result.state.velocity[0], [5, 0, 3], atol=1e-6)
    np.testing.assert_array_equal(result.state.position[1], state.position[1])
    np.testing.assert_array_equal(result.state.velocity[1], [0, 0, 0])


def test_nonrepresentable_plane_rounds_up_and_is_checked_against_true_height(tyx_mode, cache):
    model = ParticleModel(np.ones(1, np.float32), np.zeros(3))
    state = ParticleState(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32))
    height = 0.7  # nearest FP32 is below the physical plane
    with implementation()(
        mode=tyx_mode, cache_root=cache, plane=Plane(height=height), tolerance=0
    ) as solver:
        result = solver.step(model, state, StepInputs.zeros(1, dtype=np.float32), 1.0)
    assert result.report.accepted
    assert float(result.state.position[0, 1]) >= height
    assert result.report.max_residual == 0


@pytest.mark.parametrize("count", [0, 2])
@pytest.mark.parametrize("empty_constraints", [False, True])
def test_empty_buffers_skip_zero_length_device_allocations(
    tyx_mode, cache, count, empty_constraints
):
    constraints = None
    if empty_constraints:
        constraints = DistanceConstraints(
            np.empty((0, 2), np.int64), np.empty(0, np.float32), np.empty(0, np.float32)
        )
    model = ParticleModel(np.ones(count, np.float32), np.zeros(3))
    state = ParticleState(np.ones((count, 3), np.float32), np.zeros((count, 3), np.float32))
    with implementation()(constraints, mode=tyx_mode, cache_root=cache, plane=Plane()) as solver:
        result = solver.step(model, state, StepInputs.zeros(count, dtype=np.float32), 0.1)
    assert result.report.accepted
    np.testing.assert_array_equal(result.state.position, state.position)


def test_unconverged_primary_can_fall_back_to_same_physics_numpy(tyx_mode, cache):
    constraints = DistanceConstraints(
        np.array([[0, 1], [1, 2]]), np.ones(2, np.float32), np.zeros(2, np.float32)
    )
    model = ParticleModel(np.array([0, 1, 1], np.float32), np.zeros(3))
    state = ParticleState(
        np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]], np.float32), np.zeros((3, 3), np.float32)
    )
    with implementation()(
        constraints, mode=tyx_mode, cache_root=cache, iterations=1, tolerance=1e-6
    ) as primary:
        fallback = XPBDSolver(constraints, iterations=100, tolerance=1e-6)
        result = Stepper(primary, fallback).step(
            model, state, StepInputs.zeros(3, dtype=np.float32), 0.1
        )
    assert result.report.accepted and result.report.fallback_used
    assert result.report.primary_failure
    np.testing.assert_allclose(result.state.position[:, 0], [0, 1, 2], atol=2e-6)


@pytest.mark.parametrize("compliance,dt", [(1.0, 1e-30), (1e-30, 1e30)])
def test_scaled_compliance_outside_fp32_range_rejects_atomically(tyx_mode, cache, compliance, dt):
    model, state, inputs, edges = pair(compliance=compliance)
    with implementation()(edges, mode=tyx_mode, cache_root=cache) as solver:
        result = solver.step(model, state, inputs, dt)
    assert not result.report.accepted and result.report.reason
    np.testing.assert_array_equal(result.state.position, state.position)
    np.testing.assert_array_equal(result.state.velocity, state.velocity)


def test_large_finite_norm_does_not_overflow_due_to_squaring(tyx_mode, cache):
    model, state, inputs, edges = pair(rest=1e20)
    state.position[1, 0] = 2e20
    with implementation()(edges, mode=tyx_mode, cache_root=cache, tolerance=1e-6) as solver:
        result = solver.step(model, state, inputs, 1e20)
    assert result.report.accepted
    np.testing.assert_allclose(result.state.position[:, 0] / 1e20, [0.25, 1.25], atol=2e-6)


@pytest.mark.parametrize("scenario", ["prediction", "difference", "denominator", "plane"])
def test_nonfinite_intermediate_is_not_hidden_by_later_projection(tyx_mode, cache, scenario):
    model, state, inputs, edges = pair()
    plane = None
    dt = 0.1
    if scenario == "prediction":
        state.velocity[:] = -3e38
        dt, plane = 10.0, Plane()
    elif scenario == "difference":
        state.position[:, 0] = [3e38, -3e38]
    elif scenario == "denominator":
        model = ParticleModel(np.full(2, 2e38, np.float32), np.zeros(3))
    else:
        plane = Plane(height=1e100)
    with implementation()(edges, mode=tyx_mode, cache_root=cache, plane=plane) as solver:
        result = solver.step(model, state, inputs, dt)
    assert not result.report.accepted and result.report.reason
    np.testing.assert_array_equal(result.state.position, state.position)
    np.testing.assert_array_equal(result.state.velocity, state.velocity)


def test_fp64_and_invalid_indices_are_rejected(tyx_mode, cache):
    model, state, inputs, edges = pair(dtype=np.float64)
    with implementation()(mode=tyx_mode, cache_root=cache) as solver:
        with pytest.raises(ValueError, match="FP32"):
            solver.step(model, state, inputs, 0.1)
    model, state, inputs, edges = pair()
    with implementation()(edges, mode=tyx_mode, cache_root=cache) as solver:
        with pytest.raises(ValueError, match="index"):
            solver.step(
                ParticleModel(np.ones(1, np.float32)),
                ParticleState(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32)),
                StepInputs.zeros(1, dtype=np.float32),
                0.1,
            )


def test_closed_solver_rejects_even_empty_step(tyx_mode, cache):
    solver = implementation()(mode=tyx_mode, cache_root=cache)
    solver.close()
    solver.close()
    model, state, inputs, _ = pair()
    with pytest.raises(RuntimeError, match="closed"):
        solver.step(model, state, inputs, 0.1)


def test_distance_and_plane_alternation_satisfies_both(tyx_mode, cache):
    model = ParticleModel(np.array([0, 1], np.float32), np.zeros(3))
    state = ParticleState(
        np.array([[0, 1, 0], [2, -1, 0]], np.float32), np.zeros((2, 3), np.float32)
    )
    edges = DistanceConstraints(
        np.array([[0, 1]]), np.array([2], np.float32), np.zeros(1, np.float32)
    )
    with implementation()(
        edges, mode=tyx_mode, cache_root=cache, plane=Plane(), iterations=100, tolerance=1e-6
    ) as solver:
        result = solver.step(model, state, StepInputs.zeros(2, dtype=np.float32), 0.1)
        assert solver.problem_signature() == XPBDSolver(edges, plane=Plane()).problem_signature()
    assert result.report.accepted
    assert np.all(result.state.position[:, 1] >= 0)
    assert np.linalg.norm(result.state.position[1] - result.state.position[0]) == pytest.approx(
        2, abs=2e-6
    )
    assert result.state.velocity[1, 1] >= 0
    np.testing.assert_array_equal(result.state.position[0], state.position[0])


def test_valid_noncontiguous_state_uses_validated_owned_snapshot(tyx_mode, cache):
    model, state, inputs, edges = pair()
    state.position = state.position[::-1]
    state.velocity = state.velocity[::-1]
    inputs.force = inputs.force[::-1]
    assert not state.position.flags.c_contiguous
    before = state.copy()
    reference = XPBDSolver(edges).step(model, state, inputs, 0.1)
    with implementation()(edges, mode=tyx_mode, cache_root=cache) as solver:
        result = solver.step(model, state, inputs, 0.1)
    assert result.report.accepted and reference.report.accepted
    np.testing.assert_allclose(result.state.position, reference.state.position, atol=1e-6)
    np.testing.assert_allclose(result.state.velocity, reference.state.velocity, atol=1e-6)
    np.testing.assert_array_equal(state.position, before.position)
    assert result.state.position.flags.c_contiguous
