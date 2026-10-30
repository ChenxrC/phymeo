"""Whole-sweep and whole-outer-step rollback; failures never poison fresh retry."""

from dataclasses import replace

import numpy as np
import pytest

from phymeo.constraints.cloth_material import CompositeMaterialAuditSettings
from phymeo.core.capabilities import SolverFailure
from phymeo.core.scene import SceneInputs, SceneState, StepSettings, fingerprint
from phymeo.runtime.scene_stepper import SceneStepper
from phymeo.solvers.composite_cloth import CompositeClothProgram
from test_composite_cloth import BUDGETS, case, program_for, settings_for


def test_late_bending_failure_discards_distance_prefix_and_both_lambdas(monkeypatch):
    import phymeo.solvers.composite_cloth as production
    model, material, state, inputs = case("B")
    ws = program_for(model, material).workspace("cloth", state, inputs, 1/128, settings_for())
    original = production.project_bending
    def late_failure(position, *args):
        original(position, *args)
        raise SolverFailure("injected late hinge failure after private update")
    try:
        ws.predict()
        before = fingerprint((ws.candidate, ws.multipliers, tuple(ws.frames)))
        with monkeypatch.context() as patch:
            patch.setattr(production, "project_bending", late_failure)
            with pytest.raises(SolverFailure, match="late hinge"):
                ws.position_sweep()
        assert fingerprint((ws.candidate, ws.multipliers, tuple(ws.frames))) == before
        ws.position_sweep()
        retry = fingerprint((ws.candidate, ws.multipliers, tuple(ws.frames)))
        fresh = program_for(model, material).workspace("cloth", state, inputs, 1/128, settings_for())
        try:
            fresh.predict()
            fresh.position_sweep()
            assert fingerprint((fresh.candidate, fresh.multipliers, tuple(fresh.frames))) == retry
        finally:
            fresh.close()
    finally:
        ws.close()


@pytest.mark.parametrize("phase", ["position_sweep", "prepare_velocity", "result", "diagnostics", "close"])
def test_later_substep_failure_rolls_back_full_outer_state_history_and_retry(phase):
    model, material, initial, inputs = case()
    class FaultProgram(CompositeClothProgram):
        created = 0
        fault = True
        def create_workspace(self, *args):
            ws = super().create_workspace(*args)
            self.created += 1
            if self.fault and self.created == 2:
                original = getattr(ws, phase)
                def fail():
                    original()
                    raise SolverFailure("late second-substep "+phase)
                setattr(ws, phase, fail)
            return ws
    program = FaultProgram(model, material, budgets=BUDGETS)
    settings = StepSettings(substeps=2, position_iterations=512, velocity_iterations=1, tolerance=1e-8, record_substeps=True)
    scene_state, scene_inputs = SceneState({"cloth": initial}), SceneInputs({"cloth": inputs})
    before = fingerprint((scene_state, scene_inputs))
    with SceneStepper(program.scene, program, settings) as runtime:
        result = runtime.step(scene_state, scene_inputs, 1/32)
        assert not result.report.accepted
        assert result.report.substeps[0].accepted
        assert not result.report.substeps[-1].accepted
        assert fingerprint(result.state) == fingerprint(scene_state)
        assert not result.substep_states
        assert fingerprint((scene_state, scene_inputs)) == before
        program.fault = False
        retry = runtime.step(scene_state, scene_inputs, 1/32)
    clean = program_for(model, material)
    with SceneStepper(clean.scene, clean, settings) as runtime:
        fresh = runtime.step(scene_state, scene_inputs, 1/32)
    assert retry.report.accepted and fresh.report.accepted
    assert fingerprint(retry.state) == fingerprint(fresh.state)
    assert fingerprint(retry.substep_states) == fingerprint(fresh.substep_states)
    for a, b in zip(retry.report.substeps, fresh.report.substeps, strict=True):
        assert fingerprint(a.diagnostics) == fingerprint(b.diagnostics)


def test_independent_domain_unknown_rejects_before_private_publication():
    model, material, state, inputs = case()
    program = program_for(model, material, policy=CompositeMaterialAuditSettings(sqrt_bits=4, angle_terms=1))
    ws = program.workspace("cloth", state, inputs, 1/8, settings_for())
    try:
        ws.predict()
        before = fingerprint(ws.candidate)
        with pytest.raises(SolverFailure):
            ws.position_sweep()
        assert fingerprint(ws.candidate) == before
        assert not ws.frames
    finally:
        ws.close()


def test_hidden_mutation_after_complete_solve_cannot_be_reported_as_acceptance():
    from test_composite_cloth import settled_workspace
    ws = settled_workspace()
    try:
        ws.candidate.velocity[0, 0] += 1/16
        with pytest.raises(SolverFailure):
            ws.result()
        with pytest.raises(SolverFailure):
            ws.diagnostics()
    finally:
        ws.close()
