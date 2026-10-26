"""XPBD is checked against small independent solutions and failure cases."""

import numpy as np
import pytest

import phymeo
from phymeo import ParticleModel, ParticleState, StepInputs


def api(name):
    implementation = getattr(phymeo, name, None)
    assert implementation is not None, f"{name} is not implemented"
    return implementation


def edge(compliance=0.0, rest=1.0, dtype=np.float64):
    return api("DistanceConstraints")(
        np.array([[0, 1]]),
        np.array([rest], dtype=dtype),
        np.array([compliance], dtype=dtype),
    )


def pair(dtype=np.float64):
    return (
        ParticleModel(np.array([1.0, 3.0], dtype=dtype), np.zeros(3)),
        ParticleState(
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=dtype), np.zeros((2, 3), dtype=dtype)
        ),
        StepInputs.zeros(2, dtype=dtype),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hard_distance_uses_inverse_mass_and_reconstructs_velocity(dtype):
    model, state, inputs = pair(dtype)
    result = api("XPBDSolver")(edge(dtype=dtype), iterations=1).step(model, state, inputs, 0.25)
    assert result.report.accepted
    np.testing.assert_allclose(result.state.position[:, 0], [0.25, 1.25], atol=1e-6)
    np.testing.assert_allclose(result.state.velocity[:, 0], [1.0, -3.0], atol=1e-6)
    np.testing.assert_array_equal(state.position[:, 0], [0.0, 2.0])
    assert result.state.position.dtype == dtype


@pytest.mark.parametrize("dt, expected", [(0.1, [0.2, 1.4]), (0.2, [4 / 17, 22 / 17])])
def test_compliance_scales_by_dt_squared_and_nonzero_stretch_is_accepted(dt, expected):
    model, state, inputs = pair()
    result = api("XPBDSolver")(edge(compliance=0.01), iterations=5).step(model, state, inputs, dt)
    assert result.report.accepted
    assert result.report.max_constraint_error > 0.01
    assert result.report.max_residual < 1e-12
    np.testing.assert_allclose(result.state.position[:, 0], expected, atol=1e-14)


def test_coupled_compliant_chain_accumulates_multipliers_within_step():
    constraints = api("DistanceConstraints")(
        np.array([[0, 1], [1, 2]]), np.ones(2), np.full(2, 0.01)
    )
    model = ParticleModel(np.ones(3), np.zeros(3))
    state = ParticleState(
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]), np.zeros((3, 3))
    )
    solver = api("XPBDSolver")(constraints, iterations=100, tolerance=1e-12)
    result = solver.step(model, state, StepInputs.zeros(3), 0.1)
    assert result.report.accepted
    # Independent solution of [[3, -1], [-1, 3]] lambda = [-1, -1].
    np.testing.assert_allclose(result.state.position[:, 0], [0.5, 2.0, 3.5], atol=1e-12)
    assert result.report.max_constraint_error == pytest.approx(0.5, abs=1e-12)
    replay = solver.step(model, state, StepInputs.zeros(3), 0.1)
    np.testing.assert_array_equal(replay.state.position, result.state.position)


def test_fixed_anchor_stays_fixed_and_projection_changes_velocity():
    model = ParticleModel(np.array([0.0, 1.0]), np.array([0.0, -10.0, 0.0]))
    state = ParticleState(np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]]), np.zeros((2, 3)))
    result = api("XPBDSolver")(edge()).step(model, state, StepInputs.zeros(2), 0.1)
    assert result.report.accepted
    np.testing.assert_allclose(result.state.position, state.position, atol=1e-15)
    np.testing.assert_allclose(result.state.velocity, np.zeros((2, 3)), atol=1e-14)


def test_coincident_endpoints_get_deterministic_finite_separation():
    model = ParticleModel(np.ones(2), np.zeros(3))
    state = ParticleState(np.zeros((2, 3)), np.zeros((2, 3)))
    solver = api("XPBDSolver")(edge())
    result = solver.step(model, state, StepInputs.zeros(2), 0.1)
    assert result.report.accepted
    assert np.linalg.norm(result.state.position[1] - result.state.position[0]) == pytest.approx(1.0)
    np.testing.assert_allclose(result.state.position.sum(axis=0), np.zeros(3))
    replay = solver.step(model, state, StepInputs.zeros(2), 0.1)
    np.testing.assert_array_equal(result.state.position, replay.state.position)


def test_zero_rest_length_and_coincident_points_are_valid():
    model = ParticleModel(np.ones(2), np.zeros(3))
    state = ParticleState(np.zeros((2, 3)), np.zeros((2, 3)))
    result = api("XPBDSolver")(edge(rest=0.0)).step(model, state, StepInputs.zeros(2), 0.1)
    assert result.report.accepted
    assert result.report.max_residual == 0.0


def test_unsatisfiable_fixed_hard_edge_reports_failure():
    _, state, inputs = pair()
    result = api("XPBDSolver")(edge(), iterations=3).step(
        ParticleModel(np.zeros(2)), state, inputs, 0.1
    )
    assert not result.report.accepted
    assert result.report.iterations == 3
    assert result.report.max_residual == pytest.approx(1.0)
    assert result.report.reason
    np.testing.assert_array_equal(result.state.position, state.position)


def test_iteration_cap_does_not_report_unsolved_chain_as_success():
    constraints = api("DistanceConstraints")(np.array([[0, 1], [1, 2]]), np.ones(2), np.zeros(2))
    model = ParticleModel(np.array([0.0, 1.0, 1.0]), np.zeros(3))
    state = ParticleState(
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]), np.zeros((3, 3))
    )
    result = api("XPBDSolver")(constraints, iterations=1, tolerance=1e-10).step(
        model, state, StepInputs.zeros(3), 0.1
    )
    assert not result.report.accepted
    assert result.report.iterations == 1
    assert result.report.max_residual > 0.1
    assert result.report.reason


def test_plane_contact_stops_downward_motion_and_preserves_tangent_velocity():
    solver = api("XPBDSolver")(plane=api("Plane")(height=0.0))
    model = ParticleModel(np.ones(1), np.zeros(3))
    state = ParticleState(np.array([[0.0, 1.0, 0.0]]), np.array([[5.0, -20.0, 3.0]]))
    result = solver.step(model, state, StepInputs.zeros(1), 0.1)
    assert result.report.accepted
    np.testing.assert_allclose(result.state.position, [[0.5, 0.0, 0.3]], atol=1e-14)
    np.testing.assert_allclose(result.state.velocity, [[5.0, 0.0, 3.0]], atol=1e-14)


def test_plane_does_not_move_fixed_particles_and_reports_penetration():
    solver = api("XPBDSolver")(plane=api("Plane")(height=0.0), iterations=2)
    model = ParticleModel(np.zeros(1), np.zeros(3))
    state = ParticleState(np.array([[0.0, -1.0, 0.0]]), np.zeros((1, 3)))
    result = solver.step(model, state, StepInputs.zeros(1), 0.1)
    assert not result.report.accepted
    np.testing.assert_array_equal(result.state.position, state.position)
    assert result.report.max_constraint_error == pytest.approx(1.0)


def test_empty_system_is_supported():
    solver = api("XPBDSolver")(plane=api("Plane")())
    result = solver.step(
        ParticleModel(np.empty(0)),
        ParticleState(np.empty((0, 3)), np.empty((0, 3))),
        StepInputs.zeros(0),
        0.1,
    )
    assert result.report.accepted
    assert result.report.max_residual == 0.0


def test_constraint_parameters_are_owned_and_read_only():
    indices, rest, compliance = np.array([[0, 1]]), np.ones(1), np.zeros(1)
    constraints = api("DistanceConstraints")(indices, rest, compliance)
    indices[:] = 5
    rest[:] = 7
    compliance[:] = 9
    np.testing.assert_array_equal(constraints.indices, [[0, 1]])
    np.testing.assert_array_equal(constraints.rest_length, [1.0])
    np.testing.assert_array_equal(constraints.compliance, [0.0])
    for array in [constraints.indices, constraints.rest_length, constraints.compliance]:
        assert not array.flags.writeable


@pytest.mark.parametrize(
    "indices, rest, compliance",
    [
        ([[0, 0]], [1.0], [0.0]),
        ([[-1, 1]], [1.0], [0.0]),
        ([[0.0, 1.0]], [1.0], [0.0]),
        ([[0, 1, 2]], [1.0], [0.0]),
        ([[0, 1]], [-1.0], [0.0]),
        ([[0, 1]], [1.0], [-0.1]),
        ([[0, 1]], [np.nan], [0.0]),
        ([[0, 1]], [1.0], [np.inf]),
        ([[0, 1]], [1.0, 2.0], [0.0]),
    ],
)
def test_invalid_constraint_parameters_are_rejected(indices, rest, compliance):
    constraints = api("DistanceConstraints")
    with pytest.raises((TypeError, ValueError)):
        constraints(np.array(indices), np.array(rest), np.array(compliance))


@pytest.mark.parametrize(
    "options",
    [
        {"iterations": 0},
        {"iterations": -1},
        {"iterations": True},
        {"iterations": 2.5},
        {"tolerance": -1.0},
        {"tolerance": np.nan},
        {"tolerance": np.inf},
    ],
)
def test_invalid_solver_options_are_rejected(options):
    solver = api("XPBDSolver")
    with pytest.raises((TypeError, ValueError)):
        solver(**options)


def test_only_finite_y_up_plane_is_supported():
    plane = api("Plane")
    with pytest.raises(ValueError):
        plane(height=np.inf)
    with pytest.raises(ValueError):
        plane(normal=(1.0, 0.0, 0.0))


def test_out_of_bounds_constraint_is_rejected_before_step():
    solver = api("XPBDSolver")(edge())
    state = ParticleState(np.zeros((1, 3)), np.zeros((1, 3)))
    with pytest.raises(ValueError):
        solver.step(ParticleModel(np.ones(1)), state, StepInputs.zeros(1), 0.1)


def test_numerical_overflow_is_rejected_and_input_is_preserved():
    solver = api("XPBDSolver")(edge())
    state = ParticleState(np.array([[1e308, 0.0, 0.0], [-1e308, 0.0, 0.0]]), np.zeros((2, 3)))
    result = solver.step(ParticleModel(np.ones(2), np.zeros(3)), state, StepInputs.zeros(2), 0.1)
    assert not result.report.accepted
    assert result.report.reason
    assert np.isfinite(result.state.position).all()
    np.testing.assert_array_equal(result.state.position, state.position)


def test_constraint_and_plane_are_both_satisfied_after_alternating_projection():
    model = ParticleModel(np.array([0.0, 1.0]), np.zeros(3))
    state = ParticleState(np.array([[0.0, 1.0, 0.0], [2.0, -1.0, 0.0]]), np.zeros((2, 3)))
    result = api("XPBDSolver")(
        edge(rest=2.0), plane=api("Plane")(), iterations=100, tolerance=1e-9
    ).step(model, state, StepInputs.zeros(2), 0.1)
    assert result.report.accepted
    assert np.all(result.state.position[:, 1] >= 0.0)
    assert np.linalg.norm(result.state.position[1] - result.state.position[0]) == pytest.approx(
        2.0, abs=2e-9
    )
    assert result.state.velocity[1, 1] >= 0.0
    np.testing.assert_array_equal(result.state.position[0], state.position[0])


def test_empty_constraint_arrays_are_valid():
    constraints = api("DistanceConstraints")(
        np.empty((0, 2), dtype=np.int64), np.empty(0), np.empty(0)
    )
    model, state, inputs = pair()
    result = api("XPBDSolver")(constraints).step(model, state, inputs, 0.1)
    assert result.report.accepted
    np.testing.assert_array_equal(result.state.position, state.position)


def test_constraint_precision_must_match_state():
    solver = api("XPBDSolver")(edge(dtype=np.float32))
    model, state, inputs = pair()
    with pytest.raises(ValueError, match="dtype"):
        solver.step(model, state, inputs, 0.1)
