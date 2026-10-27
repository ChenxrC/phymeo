"""Migration must preserve arithmetic, real phase execution, and independent physics."""

import importlib

import numpy as np
import pytest

from phymeo import (
    CoupledInputs,
    CoupledState,
    ParticleModel,
    ParticleState,
    RigidBodyInputs,
    RigidBodyModel,
    RigidBodyState,
    SolverFailure,
    SphereShapes,
    StepInputs,
)
from phymeo.builders.rigid_cloth import rigid_cloth_scene
from phymeo.core.scene import StageMetrics, fingerprint
from phymeo.solvers.colored_coupled import ColoredCoupledXPBDSolver
from phymeo.solvers.coupled_xpbd import CoupledXPBDSolver
from phymeo.validation.rigid_cloth import audit_substep


def api():
    return importlib.import_module("phymeo.solvers.coupled_program")


def arrays(state):
    return (
        state.particles.position,
        state.particles.velocity,
        state.rigid.position,
        state.rigid.orientation,
        state.rigid.velocity,
        state.rigid.angular_velocity,
    )


def pair(dtype=np.float64, origin=0):
    pm = ParticleModel(np.array([1.0], dtype), np.zeros(3, dtype))
    bm = RigidBodyModel(np.array([0.5], dtype), np.full((1, 3), 2, dtype), np.zeros(3, dtype))
    p = ParticleState(np.array([[origin, 1, 0]], dtype), np.array([[0, -1, 0]], dtype))
    b = RigidBodyState(
        np.array([[origin, 0, 0]], dtype),
        np.array([[0, 0, 0, 1]], dtype),
        np.zeros((1, 3), dtype),
        np.zeros((1, 3), dtype),
    )
    return (
        pm,
        bm,
        SphereShapes(np.ones(1, dtype)),
        CoupledState(p, b),
        CoupledInputs.zeros(1, 1, dtype=dtype),
    )


def compare_results(actual, expected):
    assert actual.report.accepted and expected.report.accepted
    assert actual.report.completed_substeps == expected.report.completed_substeps
    for a, b in zip(arrays(actual.state), arrays(expected.state), strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(actual.report.substeps, expected.report.substeps, strict=True):
        for field in (
            "iterations",
            "velocity_iterations",
            "distance_residual",
            "contact_residual",
            "velocity_residual",
            "geometric_error",
            "penetration",
        ):
            assert getattr(a, field) == getattr(b, field)
        for phase in ("position_impulses", "velocity_impulses"):
            for field in ("particle", "rigid_linear", "rigid_angular"):
                np.testing.assert_array_equal(
                    getattr(getattr(a, phase), field), getattr(getattr(b, phase), field)
                )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("colored", [False, True])
def test_twelve_frame_full_scene_preserves_every_state_and_load(dtype, colored):
    scene = rigid_cloth_scene(dtype)
    tolerance = 1e-7
    args = (scene.particle_model, scene.rigid_model, scene.constraints, scene.spheres)
    old = (ColoredCoupledXPBDSolver if colored else CoupledXPBDSolver)(*args, record_substeps=True)
    with api().make_coupled_runtime(
        *args, mode="numpy_colored" if colored else "numpy", record_substeps=True
    ) as new:
        previous = scene.initial_state
        for _ in range(12):
            actual = new.step(previous, scene.inputs, 1 / 120)
            compare_results(actual, old.step(previous, scene.inputs, 1 / 120))
            before = previous
            h = float(dtype(float(dtype(1 / 120)) / 4))
            for after, report in zip(actual.substep_states, actual.report.substeps, strict=True):
                audit_substep(scene, before, after, report, h, tolerance=tolerance, reaction=True)
                before = after
            previous = actual.state


def test_generic_schedule_executes_phases_without_legacy_whole_step(monkeypatch):
    pm, bm, shapes, state, inputs = pair()

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy whole frame must not execute")

    monkeypatch.setattr(CoupledXPBDSolver, "step", forbidden)
    monkeypatch.setattr(CoupledXPBDSolver, "_substep", forbidden)
    with api().make_coupled_runtime(pm, bm, None, shapes, substeps=1) as runner:
        result = runner.step(state, inputs, 0.1)
        assert runner.last_scene_result.report.accepted
        assert set(runner.last_scene_result.report.substeps[0].timings) >= {
            "predict",
            "position",
            "prepare_velocity",
            "velocity",
            "close",
        }
    assert result.report.accepted
    np.testing.assert_allclose(result.state.particles.velocity, [[0, -1 / 3, 0]], atol=1e-12)
    np.testing.assert_allclose(result.state.rigid.velocity, [[0, -1 / 3, 0]], atol=1e-12)
    np.testing.assert_allclose(
        result.state.particles.position - result.state.rigid.position, [[0, 1, 0]], atol=1e-12
    )


@pytest.mark.parametrize(
    "stage", ["predict", "position_sweep", "prepare_velocity", "velocity_sweep", "close"]
)
def test_last_substep_failure_discards_both_domains_and_can_retry(stage):
    pm, bm, shapes, state, inputs = pair()
    with api().make_coupled_runtime(
        pm, bm, None, shapes, substeps=2, record_substeps=True
    ) as runner:
        program = runner.program
        create = program.create_workspace
        count = 0
        closed = []

        def inject(*args):
            nonlocal count
            count += 1
            workspace = create(*args)
            close = workspace.close

            def cleanup():
                closed.append(True)
                close()
                if count == 2 and stage == "close":
                    raise SolverFailure("late close")

            workspace.close = cleanup
            if count == 2 and stage != "close":

                def fail():
                    raise SolverFailure("late phase failure")

                setattr(workspace, stage, fail)
            return workspace

        program.create_workspace = inject
        result = runner.step(state, inputs, 0.1)
        assert not result.report.accepted and result.substep_states == () and len(closed) == 2
        for a, b in zip(arrays(result.state), arrays(state), strict=True):
            np.testing.assert_array_equal(a, b)
        program.create_workspace = create
        assert runner.step(state, inputs, 0.1).report.accepted


@pytest.mark.parametrize("dtype,origin", [(np.float32, 1e8), (np.float64, 1e16)])
def test_migration_retains_large_origin_tangent_and_rotational_dynamics(dtype, origin):
    pm, bm, shapes, state, _ = pair(dtype, origin)
    state.particles.velocity[0, 0] = 1
    state.rigid.angular_velocity[0, 2] = 0.1
    inputs = CoupledInputs(
        StepInputs.zeros(1, dtype=dtype),
        RigidBodyInputs(np.zeros((1, 3), dtype), np.array([[0, 0, 2]], dtype)),
    )
    old = CoupledXPBDSolver(pm, bm, None, shapes, substeps=2, record_substeps=True)
    with api().make_coupled_runtime(pm, bm, None, shapes, substeps=2, record_substeps=True) as new:
        result = new.step(state, inputs, 0.1)
    compare_results(result, old.step(state, inputs, 0.1))
    assert result.state.particles.velocity[0, 0] == 1
    np.testing.assert_allclose(result.state.rigid.angular_velocity[0, 2], 0.2, atol=1e-7)


@pytest.mark.parametrize("phase", ["position", "velocity"])
def test_runtime_enforces_each_budget_instead_of_accepting_program_self_report(phase):
    pm, bm, shapes, state, inputs = pair()
    with api().make_coupled_runtime(pm, bm, None, shapes, iterations=2) as runner:
        create = runner.program.create_workspace

        def fail(*args):
            workspace = create(*args)
            setattr(workspace, phase + "_metrics", lambda: StageMetrics({"unconverged": 1.0}))
            return workspace

        runner.program.create_workspace = fail
        result = runner.step(state, inputs, 0.1)
        assert not result.report.accepted and phase in result.report.reason


def test_coloring_changes_execution_order_without_changing_physical_signature():
    scene = rigid_cloth_scene()
    args = (scene.particle_model, scene.rigid_model, scene.constraints, scene.spheres)
    a, b = (api().NumpyCoupledProgram(*args, colored=v) for v in (False, True))
    assert fingerprint(a.problem_signature()) == fingerprint(b.problem_signature())


def test_bound_geometry_change_rejects_before_work():
    pm, bm, shapes, state, inputs = pair()
    with api().make_coupled_runtime(pm, bm, None, shapes) as runner:
        object.__setattr__(shapes, "radii", np.array([2.0]))
        with pytest.raises(ValueError, match="physical problem"):
            runner.step(state, inputs, 0.1)


def test_numerically_unrepresentable_normalized_residual_rejects_frame():
    pm = ParticleModel(np.ones(2), np.zeros(3))
    bm = RigidBodyModel(np.ones(1), np.ones((1, 3)), np.zeros(3))
    state = CoupledState(
        ParticleState(np.array([[1.1, 0, 0], [1.4, 0, 0]]), np.array([[-1.0, 0, 0], [-4.0, 0, 0]])),
        RigidBodyState(
            np.zeros((1, 3)), np.array([[0.0, 0, 0, 1]]), np.zeros((1, 3)), np.zeros((1, 3))
        ),
    )
    with api().make_coupled_runtime(
        pm, bm, None, SphereShapes(np.ones(1)), substeps=1, iterations=1, reference_speed=1e-320
    ) as runner:
        result = runner.step(state, CoupledInputs.zeros(2, 1), 0.1)
    assert not result.report.accepted
    for a, b in zip(arrays(result.state), arrays(state), strict=True):
        np.testing.assert_array_equal(a, b)


def test_failed_call_clears_previous_scene_diagnostics():
    pm, bm, shapes, state, inputs = pair()
    with api().make_coupled_runtime(pm, bm, None, shapes) as runner:
        assert runner.step(state, inputs, 0.1).report.accepted

        def fail(*args):
            raise RuntimeError("construction failed")

        runner.program.create_workspace = fail
        with pytest.raises(RuntimeError, match="construction failed"):
            runner.step(state, inputs, 0.1)
        assert runner.last_scene_result is None


def ordered_constraint_case(dtype=np.float64):
    from phymeo import DistanceConstraints

    pm = ParticleModel(np.ones(4, dtype), np.zeros(3, dtype))
    bm = RigidBodyModel(np.empty(0, dtype), np.empty((0, 3), dtype), np.zeros(3, dtype))
    constraints = DistanceConstraints(
        np.array([[0, 1], [2, 3], [1, 2]]), np.ones(3, dtype), np.zeros(3, dtype)
    )
    state = CoupledState(
        ParticleState(
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype), np.zeros((4, 3), dtype)
        ),
        RigidBodyState(
            np.empty((0, 3), dtype),
            np.empty((0, 4), dtype),
            np.empty((0, 3), dtype),
            np.empty((0, 3), dtype),
        ),
    )
    return (
        pm,
        bm,
        constraints,
        SphereShapes(np.empty(0, dtype)),
        state,
        CoupledInputs.zeros(4, 0, dtype=dtype),
    )


@pytest.mark.parametrize("mode", ["numpy", "numpy_colored"])
def test_reordering_bound_constraints_is_rejected_before_workspace_creation(mode):
    pm, bm, constraints, spheres, state, inputs = ordered_constraint_case()
    with api().make_coupled_runtime(pm, bm, constraints, spheres, mode=mode) as runner:
        config = runner.program._config
        physical = fingerprint(runner.program.problem_signature())
        indices = config.constraints.indices
        indices.flags.writeable = True
        indices[[1, 2]] = indices[[2, 1]]
        indices.flags.writeable = False
        assert fingerprint(api().coupled_signature(config)) == physical

        def forbidden(*args):
            raise AssertionError("changed binding reached a workspace")

        runner.program.create_workspace = forbidden
        with pytest.raises(ValueError, match="configuration"):
            runner.step(state, inputs, 0.1)
    with api().make_coupled_runtime(pm, bm, config.constraints, spheres, mode=mode) as rebound:
        assert rebound.step(state, inputs, 0.1).report.accepted


@pytest.mark.parametrize("counts", [(0, 0), (0, 1), (1, 0)])
def test_empty_domains_preserve_shapes(counts):
    np_count, nb_count = counts
    pm = ParticleModel(np.ones(np_count), np.zeros(3))
    bm = RigidBodyModel(np.ones(nb_count), np.ones((nb_count, 3)), np.zeros(3))
    shapes = SphereShapes(np.ones(nb_count))
    state = CoupledState(
        ParticleState(np.zeros((np_count, 3)), np.zeros((np_count, 3))),
        RigidBodyState(
            np.zeros((nb_count, 3)),
            np.tile([0.0, 0.0, 0.0, 1.0], (nb_count, 1)),
            np.zeros((nb_count, 3)),
            np.zeros((nb_count, 3)),
        ),
    )
    with api().make_coupled_runtime(pm, bm, None, shapes) as runner:
        result = runner.step(state, CoupledInputs.zeros(*counts), 0.1)
    assert result.report.accepted
    for a, b in zip(arrays(state), arrays(result.state), strict=True):
        np.testing.assert_array_equal(a, b)
