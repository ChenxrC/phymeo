"""Joint time integration, physical contact and whole-frame ownership/rollback."""

import importlib

import numpy as np
import pytest

from phymeo import (
    DistanceConstraints,
    ParticleModel,
    ParticleState,
    RigidBodyInputs,
    RigidBodyModel,
    RigidBodyState,
    RigidSemiImplicitSolver,
    SemiImplicitSolver,
    SolverFailure,
    SphereShapes,
    StepInputs,
)


def types():
    return importlib.import_module("phymeo")


def solver_type():
    return importlib.import_module("phymeo.solvers.coupled_xpbd").CoupledXPBDSolver


def problem(dtype=np.float64, np_count=1, nb_count=1):
    gravity = np.array([0, 0, 0], dtype=dtype)
    pm = ParticleModel(np.ones(np_count, dtype=dtype), gravity)
    bm = RigidBodyModel(
        np.ones(nb_count, dtype=dtype), np.ones((nb_count, 3), dtype=dtype), gravity
    )
    ps = ParticleState(np.full((np_count, 3), 3, dtype=dtype), np.zeros((np_count, 3), dtype=dtype))
    qs = np.tile(np.array([0, 0, 0, 1], dtype=dtype), (nb_count, 1))
    bs = RigidBodyState(
        np.zeros((nb_count, 3), dtype=dtype),
        qs,
        np.zeros((nb_count, 3), dtype=dtype),
        np.zeros((nb_count, 3), dtype=dtype),
    )
    return pm, bm, ps, bs, SphereShapes(np.ones(nb_count, dtype=dtype))


def arrays(state):
    return (
        state.particles.position,
        state.particles.velocity,
        state.rigid.position,
        state.rigid.orientation,
        state.rigid.velocity,
        state.rigid.angular_velocity,
    )


def test_mixed_state_and_inputs_own_copies_without_virtual_copy_calls():
    pm, bm, ps, bs, shapes = problem()

    class BorrowingParticles(ParticleState):
        def copy(self):
            raise AssertionError("must not use virtual copy")

    class BorrowingRigid(RigidBodyState):
        def copy(self):
            raise AssertionError("must not use virtual copy")

    ps = BorrowingParticles(ps.position, ps.velocity)
    bs = BorrowingRigid(bs.position, bs.orientation, bs.velocity, bs.angular_velocity)
    state = types().CoupledState(ps, bs)
    particle_input, rigid_input = StepInputs.zeros(1), RigidBodyInputs.zeros(1)
    inputs = types().CoupledInputs(particle_input, rigid_input)
    assert type(state.particles) is ParticleState and type(state.rigid) is RigidBodyState
    for owned, original in zip(
        arrays(state),
        (ps.position, ps.velocity, bs.position, bs.orientation, bs.velocity, bs.angular_velocity),
    ):
        assert not np.shares_memory(owned, original)
    assert not np.shares_memory(inputs.particles.force, particle_input.force)
    assert not np.shares_memory(inputs.rigid.force, rigid_input.force)
    assert not np.shares_memory(inputs.rigid.torque, rigid_input.torque)


@pytest.mark.parametrize("dtype,atol", [(np.float64, 1e-12), (np.float32, 2e-6)])
def test_contact_free_substeps_match_independent_predictors(dtype, atol):
    pm, bm, ps, bs, shapes = problem(dtype)
    ps.velocity[:] = [0.3, -0.4, 0.5]
    bs.angular_velocity[:] = [0, 0, 0.6]
    state = types().CoupledState(ps, bs)
    inputs = types().CoupledInputs(
        StepInputs(np.array([[0.2, 0.3, 0.4]], dtype=dtype)),
        RigidBodyInputs(
            np.array([[0.1, -0.2, 0.3]], dtype=dtype), np.array([[0, 0, 0.2]], dtype=dtype)
        ),
    )
    result = solver_type()(pm, bm, None, shapes, substeps=4, record_substeps=True).step(
        state, inputs, 0.2
    )
    p_ref, b_ref = ps, bs
    for _ in range(4):
        p_ref = SemiImplicitSolver().step(pm, p_ref, inputs.particles, dtype(0.05)).state
        b_ref = RigidSemiImplicitSolver().step(bm, b_ref, inputs.rigid, dtype(0.05)).state
    assert result.report.accepted
    assert result.report.completed_substeps == 4
    assert len(result.substep_states) == 4
    np.testing.assert_allclose(
        result.state.particles.velocity, [[0.34, -0.34, 0.58]], rtol=0, atol=atol
    )
    np.testing.assert_allclose(
        result.state.particles.position, [[3.065, 2.9275, 3.11]], rtol=0, atol=atol
    )
    for actual, wanted in zip(arrays(result.state), arrays(types().CoupledState(p_ref, b_ref))):
        np.testing.assert_allclose(actual, wanted, rtol=0, atol=atol)
    for actual, original in zip(arrays(result.state), arrays(state)):
        assert not np.shares_memory(actual, original)
    np.testing.assert_array_equal(state.particles.position, ps.position)


def test_unmodified_free_positions_do_not_erase_small_velocity_at_large_origin():
    pm, bm, ps, bs, shapes = problem()
    ps.position[:] = [1e16, 0, 0]
    ps.velocity[:] = [1, 0, 0]
    state = types().CoupledState(ps, bs)
    result = solver_type()(pm, bm, None, shapes, substeps=1).step(
        state, types().CoupledInputs.zeros(1, 1), 0.1
    )
    assert result.report.accepted
    np.testing.assert_array_equal(result.state.particles.velocity, [[1, 0, 0]])


def test_single_collision_has_opposite_impulse_and_no_double_application():
    pm, bm, ps, bs, shapes = problem()
    ps.position[:] = [1.125, 0, 0]
    ps.velocity[:] = [-1, 0, 0]
    bs.angular_velocity[:] = [0, 0, 1]
    state = types().CoupledState(ps, bs)
    inputs = types().CoupledInputs.zeros(1, 1)
    result = solver_type()(pm, bm, None, shapes, substeps=1, record_substeps=True).step(
        state, inputs, 0.25
    )
    assert result.report.accepted
    np.testing.assert_allclose(result.state.particles.position, [[0.9375, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(result.state.rigid.position, [[-0.0625, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(result.state.particles.velocity, [[-0.5, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(result.state.rigid.velocity, [[-0.5, 0, 0]], atol=1e-12)
    np.testing.assert_array_equal(result.state.rigid.angular_velocity, [[0, 0, 1]])
    expected_rotation = RigidSemiImplicitSolver().step(bm, bs, inputs.rigid, 0.25).state.orientation
    np.testing.assert_array_equal(result.state.rigid.orientation, expected_rotation)
    diag = result.report.substeps[0]
    np.testing.assert_allclose(diag.position_impulses.particle, [[0.25, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(diag.velocity_impulses.particle, [[0.25, 0, 0]], atol=1e-12)
    for ledger in (diag.position_impulses, diag.velocity_impulses):
        np.testing.assert_array_equal(ledger.rigid_linear, -ledger.particle)
        np.testing.assert_array_equal(ledger.rigid_angular, [[0, 0, 0]])
    assert result.report.max_residual <= 1e-7


def test_distance_and_contact_converge_together_not_in_separate_steps():
    pm, bm, ps, bs, shapes = problem(np_count=2)
    pm = ParticleModel(np.array([0.0, 1]), np.zeros(3))
    ps.position[:] = [[2, 0, 0], [0.5, 0, 0]]
    constraints = DistanceConstraints(np.array([[0, 1]]), np.array([2.0]), np.zeros(1))
    result = solver_type()(pm, bm, constraints, shapes, substeps=1, iterations=100).step(
        types().CoupledState(ps, bs), types().CoupledInputs.zeros(2, 1), 0.1
    )
    assert result.report.accepted
    assert result.report.iterations > 1
    np.testing.assert_array_equal(result.state.particles.position[0], [2, 0, 0])
    assert (
        abs(
            np.linalg.norm(result.state.particles.position[0] - result.state.particles.position[1])
            - 2
        )
        <= 2e-7
    )
    assert (
        np.linalg.norm(result.state.particles.position[1] - result.state.rigid.position[0])
        >= 1 - 1e-7
    )
    assert result.state.rigid.position[0, 0] < -0.99


def test_later_substep_budget_exhaustion_restores_the_whole_frame():
    pm, bm, ps, bs, shapes = problem(np_count=2)
    ps.position[:] = [[1.2, 0, 0], [-2, 0, 0]]
    ps.velocity[:] = [[0, 0, 0], [1.5, 0, 0]]
    bs.angular_velocity[:] = [0, 0, 0.5]
    state = types().CoupledState(ps, bs)
    result = solver_type()(
        pm, bm, None, shapes, substeps=2, iterations=1, record_substeps=True
    ).step(state, types().CoupledInputs.zeros(2, 1), 1.0)
    assert not result.report.accepted
    assert result.report.completed_substeps == 1
    assert "budget" in result.report.reason
    assert result.substep_states == ()
    for actual, original in zip(arrays(result.state), arrays(state)):
        np.testing.assert_array_equal(actual, original)
        assert not np.shares_memory(actual, original)


def test_later_prediction_overflow_restores_both_state_families():
    pm, bm, ps, bs, shapes = problem()
    ps.position[:] = 0
    bs.position[:] = [100, 0, 0]
    bs.angular_velocity[:] = [0, 0, 0.5]
    inputs = types().CoupledInputs(StepInputs(np.array([[1e308, 0, 0]])), RigidBodyInputs.zeros(1))
    state = types().CoupledState(ps, bs)
    result = solver_type()(pm, bm, None, shapes, substeps=2).step(state, inputs, 2)
    assert not result.report.accepted
    assert result.report.completed_substeps == 1
    for actual, original in zip(arrays(result.state), arrays(state)):
        np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(inputs.particles.force, [[1e308, 0, 0]])


def test_later_contact_failure_is_atomic_and_a_fresh_call_still_works(monkeypatch):
    pm, bm, ps, bs, shapes = problem()
    module = importlib.import_module("phymeo.solvers.coupled_xpbd")
    original = module.project_particle_spheres
    calls = 0

    def failing(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            args[0][:] = 42
            raise SolverFailure("injected contact failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "project_particle_spheres", failing)
    state = types().CoupledState(ps, bs)
    solver = solver_type()(pm, bm, None, shapes, substeps=2)
    result = solver.step(state, types().CoupledInputs.zeros(1, 1), 0.1)
    assert not result.report.accepted and result.report.completed_substeps == 1
    for actual, wanted in zip(arrays(result.state), arrays(state)):
        np.testing.assert_array_equal(actual, wanted)
    monkeypatch.setattr(module, "project_particle_spheres", original)
    assert solver.step(state, types().CoupledInputs.zeros(1, 1), 0.1).report.accepted


def test_velocity_nonconvergence_rejects_even_when_positions_have_converged():
    pm, bm, ps, bs, shapes = problem(np_count=2)
    ps.position[:] = [[1.1, 0, 0], [1.4, 0, 0]]
    ps.velocity[:] = [[-1, 0, 0], [-4, 0, 0]]
    state = types().CoupledState(ps, bs)
    result = solver_type()(pm, bm, None, shapes, substeps=1, iterations=1).step(
        state, types().CoupledInputs.zeros(2, 1), 0.1
    )
    assert not result.report.accepted
    assert "velocity" in result.report.reason and "budget" in result.report.reason
    assert result.report.contact_residual <= 1e-7
    assert result.report.velocity_residual > 1e-7
    for actual, original in zip(arrays(result.state), arrays(state)):
        np.testing.assert_array_equal(actual, original)


def test_recorded_substeps_are_independent_of_final_state_and_one_another():
    pm, bm, ps, bs, shapes = problem()
    state = types().CoupledState(ps, bs)
    result = solver_type()(pm, bm, None, shapes, record_substeps=True).step(
        state, types().CoupledInputs.zeros(1, 1), 0.1
    )
    histories = [state, result.state, *result.substep_states]
    for index, first in enumerate(histories):
        for second in histories[index + 1 :]:
            for left, right in zip(arrays(first), arrays(second)):
                assert not np.shares_memory(left, right)
    result.substep_states[0].particles.position[:] = 42
    np.testing.assert_array_equal(result.state.particles.position, [[3, 3, 3]])


def test_compliant_distance_stretch_is_not_mistaken_for_nonconvergence():
    pm, bm, ps, bs, shapes = problem(np_count=2, nb_count=0)
    pm = ParticleModel(np.array([0.0, 1]), np.zeros(3))
    ps.position[:] = [[0, 0, 0], [2, 0, 0]]
    constraint = DistanceConstraints(np.array([[0, 1]]), np.array([1.0]), np.array([0.01]))
    result = solver_type()(pm, bm, constraint, shapes, substeps=1).step(
        types().CoupledState(ps, bs), types().CoupledInputs.zeros(2, 0), 0.1
    )
    assert result.report.accepted
    np.testing.assert_allclose(result.state.particles.position[1], [1.5, 0, 0], atol=1e-12)
    assert result.report.distance_residual < 1e-12
    assert result.report.substeps[0].geometric_error == pytest.approx(0.5)


@pytest.mark.parametrize("np_count,nb_count", [(0, 0), (0, 2), (2, 0)])
def test_empty_state_families_are_supported(np_count, nb_count):
    pm, bm, ps, bs, shapes = problem(np_count=np_count, nb_count=nb_count)
    result = solver_type()(pm, bm, None, shapes).step(
        types().CoupledState(ps, bs), types().CoupledInputs.zeros(np_count, nb_count), 0.1
    )
    assert result.report.accepted and result.report.completed_substeps == 4
    assert result.state.particles.position.shape == (np_count, 3)
    assert result.state.rigid.position.shape == (nb_count, 3)


def test_fixed_objects_keep_pose_and_zero_both_velocities_under_load():
    pm, bm, ps, bs, shapes = problem()
    pm = ParticleModel(np.zeros(1), np.zeros(3))
    bm = RigidBodyModel(np.zeros(1), np.zeros((1, 3)), np.zeros(3))
    ps.velocity[:] = 1
    bs.velocity[:], bs.angular_velocity[:] = 1, 2
    state = types().CoupledState(ps, bs)
    inputs = types().CoupledInputs(
        StepInputs(np.full((1, 3), 1e308)),
        RigidBodyInputs(np.full((1, 3), 1e308), np.full((1, 3), 1e308)),
    )
    result = solver_type()(pm, bm, None, shapes).step(state, inputs, 1)
    assert result.report.accepted
    np.testing.assert_array_equal(result.state.particles.position, ps.position)
    np.testing.assert_array_equal(result.state.rigid.position, bs.position)
    np.testing.assert_array_equal(result.state.rigid.orientation, bs.orientation)
    for velocity in (
        result.state.particles.velocity,
        result.state.rigid.velocity,
        result.state.rigid.angular_velocity,
    ):
        np.testing.assert_array_equal(velocity, [[0, 0, 0]])


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("substeps", 0),
        ("substeps", True),
        ("iterations", 0),
        ("iterations", 2.5),
        ("tolerance", -1),
        ("tolerance", np.nan),
        ("reference_length", 0),
        ("reference_speed", 0),
        ("record_substeps", 1),
    ],
)
def test_invalid_configuration_is_rejected(keyword, value):
    pm, bm, ps, bs, shapes = problem()
    with pytest.raises((TypeError, ValueError)):
        solver_type()(pm, bm, None, shapes, **{keyword: value})


@pytest.mark.parametrize("kind", ["dtype", "count", "force", "mutated_nan", "wrong_state", "dt"])
def test_incompatible_or_mutated_inputs_are_rejected(kind):
    pm, bm, ps, bs, shapes = problem()
    state = types().CoupledState(ps, bs)
    inputs = types().CoupledInputs.zeros(1, 1)
    dt = 0.1
    if kind == "dtype":
        state.particles.position = state.particles.position.astype(np.float32)
    elif kind == "count":
        state.particles = ParticleState(np.zeros((2, 3)), np.zeros((2, 3)))
    elif kind == "force":
        inputs.rigid.force = np.zeros((2, 3))
    elif kind == "mutated_nan":
        state.rigid.angular_velocity[0, 0] = np.nan
    elif kind == "wrong_state":
        state = ps
    else:
        dt = 0
    with pytest.raises((TypeError, ValueError)):
        solver_type()(pm, bm, None, shapes).step(state, inputs, dt)
