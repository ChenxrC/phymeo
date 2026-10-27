"""Native cloth/solid phases, physical updates and whole-frame failure recovery."""

import importlib
from dataclasses import dataclass, replace

import numpy as np
import pytest
from test_tetrahedral_material import unit_tetra

from phymeo.builders.cloth import cloth_grid
from phymeo.core.capabilities import SolverFailure
from phymeo.core.scene import BackendContext, SceneInputs, SceneState, StepSettings
from phymeo.core.state import ParticleState, StepInputs
from phymeo.materials.stvk import elastic_forces
from phymeo.runtime.registry import StateRegistry
from phymeo.runtime.scene_stepper import SceneStepper
from phymeo.solvers.xpbd import XPBDSolver


def soft_api():
    return importlib.import_module("phymeo.solvers.tetrahedral")


def cloth_api():
    return importlib.import_module("phymeo.solvers.cloth")


def family():
    return importlib.import_module("phymeo.runtime.tetrahedral_family").tetrahedral_family()


def initial_tetra(model):
    return ParticleState(model.rest_position, np.zeros_like(model.rest_position))


def setup_program(kind):
    if kind == "soft":
        model = unit_tetra()
        initial = initial_tetra(model)
        program = soft_api().TetrahedralProgram(model)
        registry = StateRegistry()
        registry.register(family())
        domain = "soft"
    else:
        patch = cloth_grid(3, 3, pinned=[0, 2], origin=(0.0, 0.0, 0.0))
        model, initial = patch.model, patch.initial_state
        program = cloth_api().ClothProgram(model, patch.constraints)
        registry, domain = StateRegistry(), "cloth"
    inputs = StepInputs.zeros(model.particle_count)
    return model, initial, program, registry, domain, inputs


def test_tetrahedral_family_is_explicit_owned_and_checks_candidates():
    model, registry = unit_tetra(), StateRegistry()
    with pytest.raises(ValueError, match="unregistered"):
        registry.get("tetrahedra")
    registry.register(family())
    state, inputs, context = initial_tetra(model), StepInputs.zeros(4), BackendContext()
    owned, loads = family().snapshot(model, state, inputs, context)
    assert not np.shares_memory(owned.position, state.position)
    assert not np.shares_memory(loads.force, inputs.force)
    fixed = state.copy()
    fixed.position[0, 0] += 0.001
    with pytest.raises(ValueError, match="fixed"):
        family().candidate(model, state, fixed, context)
    invalid = state.copy()
    invalid.position[1, 0] = 1.5
    with pytest.raises(SolverFailure, match="domain"):
        family().candidate(model, state, invalid, context)

    @dataclass
    class HiddenState(ParticleState):
        material_history: np.ndarray

    with pytest.raises(TypeError, match="unregistered"):
        family().candidate(
            model, state, HiddenState(state.position, state.velocity, np.zeros(1)), context
        )


@pytest.mark.parametrize("fault", ["count", "float32", "force_count", "fixed_velocity", "domain"])
def test_tetrahedral_entry_rejects_incompatible_or_invalid_native_state(fault):
    model, state, inputs = unit_tetra(), initial_tetra(unit_tetra()), StepInputs.zeros(4)
    if fault == "count":
        state = ParticleState(np.zeros((5, 3)), np.zeros((5, 3)))
    elif fault == "float32":
        state = ParticleState(state.position.astype(np.float32), state.velocity.astype(np.float32))
    elif fault == "force_count":
        inputs = StepInputs.zeros(3)
    elif fault == "fixed_velocity":
        state.velocity[0] = 1
    else:
        state.position[1, 0] = -0.1
    with pytest.raises((TypeError, ValueError, SolverFailure)):
        family().snapshot(model, state, inputs, BackendContext())


def test_solid_rest_shape_external_force_has_analytic_semi_implicit_motion():
    model = unit_tetra(gravity=np.array([0.0, -2, 0]))
    state, inputs = initial_tetra(model), StepInputs.zeros(4)
    inputs.force[1] = [1, 2, 3]
    inputs.force[0] = [4, 5, 6]
    h = 0.01
    expected_acc = np.array([[0.0, 0, 0], [4.0, 6, 12], [0.0, -2, 0], [0.0, -2, 0]])
    with soft_api().make_tetrahedral_runtime(model) as runtime:
        result = runtime.step(SceneState({"soft": state}), SceneInputs({"soft": inputs}), h)
    assert result.report.accepted
    native = result.state.domains["soft"]
    np.testing.assert_allclose(native.velocity, h * expected_acc, atol=1e-12)
    np.testing.assert_allclose(native.position, state.position + h**2 * expected_acc, atol=1e-12)
    np.testing.assert_array_equal(native.position[0], state.position[0])
    diagnostic = result.report.substeps[0].diagnostics
    np.testing.assert_allclose(diagnostic["internal_force"], 0.0, atol=1e-12)
    np.testing.assert_allclose(diagnostic["support_force"][0], [-4, -4.5, -6], atol=1e-12)
    assert diagnostic["integration_residual"] <= 1e-12
    assert diagnostic["min_det_f"] > 0.2 and diagnostic["elastic_energy"] > 0
    assert "equilibrium_residual" not in diagnostic
    np.testing.assert_array_equal(state.position, model.rest_position)
    np.testing.assert_array_equal(inputs.force[1], [1, 2, 3])


def test_solid_initial_stretch_uses_full_internal_force_before_boundary_conditions():
    model = unit_tetra()
    state, inputs, h = initial_tetra(model), StepInputs.zeros(4), 0.001
    state.position *= 1.1
    # Unit element stress at uniform stretch, independent of the implemented force.
    strain, mu, lam = (1.1**2 - 1) / 2, 1000 / 2.6, 1000 * 0.3 / (1.3 * 0.4)
    traction = 1.1 * (2 * mu + 3 * lam) * strain / 6
    full = np.array([[1.0, 1, 1], [-1.0, 0, 0], [0.0, -1, 0], [0.0, 0, -1]]) * traction
    with soft_api().make_tetrahedral_runtime(model) as runtime:
        result = runtime.step(SceneState({"soft": state}), SceneInputs({"soft": inputs}), h)
    assert result.report.accepted
    np.testing.assert_allclose(
        result.state.domains["soft"].velocity, h * model.inverse_mass[:, None] * full, atol=1e-12
    )
    np.testing.assert_allclose(
        result.report.substeps[0].diagnostics["internal_force"], full, atol=1e-12
    )
    np.testing.assert_allclose(
        result.report.substeps[0].diagnostics["support_force"][0], -full[0], atol=1e-12
    )


def test_cloth_phases_match_old_xpbd_at_same_force_time_and_tolerance(monkeypatch):
    patch = cloth_grid(
        3, 3, spacing=0.05, pinned=[0, 2, 6, 8], structural_compliance=1e-4, shear_compliance=1e-4
    )
    settings = StepSettings(substeps=2, position_iterations=200, tolerance=1e-10)
    inputs = StepInputs.zeros(9)
    inputs.force[4] = [0.01, -0.02, 0.01]
    initial = patch.initial_state
    expected = initial.copy()
    old = XPBDSolver(patch.constraints, iterations=200, tolerance=1e-10)
    for _ in range(2):
        solved = old.step(patch.model, expected, inputs, 0.001)
        assert solved.report.accepted
        expected = solved.state
    # This old frame method must not be a hidden participant implementation.
    monkeypatch.setattr(XPBDSolver, "step", lambda *args: pytest.fail("nested old solver.step"))
    with cloth_api().make_cloth_runtime(
        patch.model, patch.constraints, settings=settings
    ) as runtime:
        result = runtime.step(SceneState({"cloth": initial}), SceneInputs({"cloth": inputs}), 0.002)
    assert result.report.accepted
    np.testing.assert_allclose(
        result.state.domains["cloth"].position, expected.position, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        result.state.domains["cloth"].velocity, expected.velocity, atol=1e-12, rtol=0
    )
    np.testing.assert_array_equal(
        result.state.domains["cloth"].position[[0, 2, 6, 8]], initial.position[[0, 2, 6, 8]]
    )


@pytest.mark.parametrize("kind", ["soft", "cloth"])
@pytest.mark.parametrize("fault", ["position", "velocity", "candidate", "close"])
def test_late_failure_recovers_all_nodal_fields_and_allows_reuse(kind, fault):
    model, initial, program, registry, domain, inputs = setup_program(kind)
    original, calls, closed = program.create_workspace, [0], []

    def create(*args):
        work = original(*args)
        calls[0] += 1
        index = calls[0]

        def close():
            closed.append(index)
            if index == 3 and fault == "close":
                raise SolverFailure("late resource failure")

        work.close = close
        if index == 3:

            def reject():
                raise SolverFailure("late stage failure")

            if fault == "position":
                work.position_sweep = reject
            elif fault == "velocity":
                work.velocity_sweep = reject
            elif fault == "candidate":
                work.result = lambda: SceneState(
                    {domain: ParticleState(np.zeros((1, 3)), np.zeros((1, 3)))}
                )
        return work

    program.create_workspace = create
    with SceneStepper(
        program.scene, program, StepSettings(substeps=3, record_substeps=True), registry=registry
    ) as runtime:
        result = runtime.step(SceneState({domain: initial}), SceneInputs({domain: inputs}), 0.003)
        assert not result.report.accepted and result.report.completed_substeps == 2
        assert result.substep_states == () and closed == [1, 2, 3]
        np.testing.assert_array_equal(result.state.domains[domain].position, initial.position)
        np.testing.assert_array_equal(result.state.domains[domain].velocity, initial.velocity)
        assert runtime.step(
            SceneState({domain: initial}), SceneInputs({domain: inputs}), 0.003
        ).report.accepted


def test_late_material_domain_failure_reverts_successful_first_substep():
    model = unit_tetra(young_modulus=1e-8)
    state, inputs = initial_tetra(model), StepInputs.zeros(4)
    state.velocity[1, 0] = 1
    with soft_api().make_tetrahedral_runtime(
        model, settings=StepSettings(substeps=2, record_substeps=True)
    ) as runtime:
        result = runtime.step(SceneState({"soft": state}), SceneInputs({"soft": inputs}), 0.6)
    assert not result.report.accepted and result.report.completed_substeps == 1
    assert "domain" in result.report.reason
    np.testing.assert_array_equal(result.state.domains["soft"].position, state.position)
    np.testing.assert_array_equal(result.state.domains["soft"].velocity, state.velocity)


@pytest.mark.parametrize("kind", ["soft", "cloth"])
def test_model_changes_and_foreign_bound_model_are_rejected(kind):
    model, initial, program, registry, domain, inputs = setup_program(kind)
    with SceneStepper(program.scene, program, registry=registry) as runtime:
        model.gravity.flags.writeable = True
        model.gravity[0] += 1
        with pytest.raises(ValueError, match="changed"):
            runtime.step(SceneState({domain: initial}), SceneInputs({domain: inputs}), 0.001)


def test_bound_cloth_constraint_order_cannot_change_after_runtime_creation():
    patch = cloth_grid(3, 3, pinned=[0, 2])
    with cloth_api().make_cloth_runtime(patch.model, patch.constraints) as runtime:
        patch.constraints.indices.flags.writeable = True
        patch.constraints.indices[[0, 1]] = patch.constraints.indices[[1, 0]]
        with pytest.raises(ValueError, match="changed"):
            runtime.step(
                SceneState({"cloth": patch.initial_state}),
                SceneInputs({"cloth": StepInputs.zeros(9)}),
                0.001,
            )


def test_soft_domain_anchors_use_valid_entry_positions_not_forced_rest_shape():
    model = unit_tetra()
    state = initial_tetra(model)
    state.position += [2.0, 3, 4]
    with soft_api().make_tetrahedral_runtime(model) as runtime:
        result = runtime.step(
            SceneState({"soft": state}), SceneInputs({"soft": StepInputs.zeros(4)}), 0.001
        )
    assert result.report.accepted
    np.testing.assert_array_equal(result.state.domains["soft"].position, state.position)


def test_solid_diagnostics_are_owned_and_cannot_mutate_later_steps():
    model, state = unit_tetra(), initial_tetra(unit_tetra())
    with soft_api().make_tetrahedral_runtime(model) as runtime:
        inputs = SceneInputs({"soft": StepInputs.zeros(4)})
        result = runtime.step(SceneState({"soft": state}), inputs, 0.001)
        with pytest.raises(ValueError):
            result.report.substeps[0].diagnostics["internal_force"][:] = 99
        assert runtime.step(result.state, inputs, 0.001).report.accepted


def test_solid_force_is_evaluated_at_substep_start_not_predicted_configuration():
    model, state = unit_tetra(), initial_tetra(unit_tetra())
    state.position[1, 0] = 1.05
    state.velocity[1, 0] = 0.4
    start_force = elastic_forces(model, state.position)
    with soft_api().make_tetrahedral_runtime(model) as runtime:
        result = runtime.step(
            SceneState({"soft": state}), SceneInputs({"soft": StepInputs.zeros(4)}), 0.001
        )
    assert result.report.accepted
    np.testing.assert_allclose(
        result.state.domains["soft"].velocity,
        state.velocity + 0.001 * model.inverse_mass[:, None] * start_force,
        atol=1e-12,
    )


def test_free_cloth_motion_keeps_velocity_when_large_position_rounds():
    from phymeo.core.model import ParticleModel

    model = ParticleModel(np.ones(1), np.zeros(3))
    state = ParticleState(np.full((1, 3), 1e16), np.ones((1, 3)))
    with cloth_api().make_cloth_runtime(model, None) as runtime:
        result = runtime.step(
            SceneState({"cloth": state}), SceneInputs({"cloth": StepInputs.zeros(1)}), 0.001
        )
    assert result.report.accepted
    np.testing.assert_array_equal(result.state.domains["cloth"].velocity, state.velocity)


def test_solid_fixed_loads_are_retained_without_advancing_fixed_nodes():
    model = replace(unit_tetra(), fixed=np.ones(4, bool))
    state, inputs = initial_tetra(model), StepInputs(np.ones((4, 3)))
    with soft_api().make_tetrahedral_runtime(model) as runtime:
        result = runtime.step(SceneState({"soft": state}), SceneInputs({"soft": inputs}), 0.001)
    assert result.report.accepted
    np.testing.assert_array_equal(result.state.domains["soft"].position, state.position)
    np.testing.assert_allclose(
        result.report.substeps[0].diagnostics["support_force"], -inputs.force, atol=1e-12
    )
