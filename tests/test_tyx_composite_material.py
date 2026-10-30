"""Actual native composite ownership and the unchanged frozen physical gates."""

import numpy as np
import pytest
from test_composite_cloth import BUDGETS, case, program_for, settings_for, settled_workspace
from test_composite_material_physics import equal_runs, run_window, standalone_replay

from phymeo.backends.tyx_composite_cloth import TyxCompositeClothParticipant
from phymeo.constraints.material_representation import NativeCompositeRepresentation, decode
from phymeo.core.capabilities import SolverFailure
from phymeo.core.scene import BackendContext, fingerprint

pytestmark = pytest.mark.tyx
REPRESENTATION = NativeCompositeRepresentation()


def test_actual_composite_complete_operation_receipts_and_storage(tyx_mode):
    ws = settled_workspace("B", mode=tyx_mode, representation=REPRESENTATION, h=1 / 128)
    try:
        data = ws.diagnostics()
        path = data["material_path"]
        assert path.prediction_launches == path.reconstruction_launches == 1
        assert data["execution"]["compute_dtype"] == "float32"
        assert path.backend == "tyx:" + tyx_mode
        for sweep in path.sweeps:
            for batch in sweep.blocks:
                assert batch.launches == batch.completed_launches == 1
                assert batch.rows.dtype == np.float32
                assert batch.displacement_in.dtype == batch.displacement_out.dtype == np.float32
                np.testing.assert_array_equal(batch.positions_in, decode(path.reference, batch.displacement_in)[0])
                np.testing.assert_array_equal(batch.positions_out, decode(path.reference, batch.displacement_out)[0])
                assert batch.reference_fingerprint == path.reference_fingerprint
        assert data["material_audit"]["physical_affine_domain"].status == "certified"
    finally:
        ws.close()


@pytest.mark.parametrize(
    "name,h",
    [("A", 1 / 64), ("A", 1 / 128), ("A", 1 / 256), ("B", 1 / 128), ("B", 1 / 256), ("B", 1 / 512)],
)
def test_actual_native_frozen_windows_two_rebuilds(tyx_mode, name, h):
    first, a = run_window(name, h, mode=tyx_mode, representation=REPRESENTATION)
    second, b = run_window(name, h, mode=tyx_mode, representation=REPRESENTATION)
    equal_runs(first, second)
    assert a == b


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mass_scale": 0.5},
        {"mass_scale": 2.0},
        {"perturb": 2**-12},
        {"perturb": -(2**-12)},
        {
            "rotation": np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]),
            "translation": np.array([3 / 8, -1 / 4, 1 / 8]),
        },
    ],
)
def test_actual_native_perturbation_windows_original_budgets(tyx_mode, kwargs):
    first, _ = run_window("B", 1 / 128, mode=tyx_mode, representation=REPRESENTATION, **kwargs)
    second, _ = run_window("B", 1 / 128, mode=tyx_mode, representation=REPRESENTATION, **kwargs)
    equal_runs(first, second)


def test_actual_native_external_replay_and_terminal_retry(tyx_mode):
    assert standalone_replay(tyx_mode, representation=REPRESENTATION) == standalone_replay(tyx_mode, representation=REPRESENTATION)


def test_late_native_launch_failure_discards_complete_sweep_and_retry_is_fresh(
    tyx_mode, monkeypatch
):
    model, material, initial, inputs = case("B")
    program = program_for(model, material, tyx_mode, representation=REPRESENTATION)
    ws = program.workspace("cloth", initial, inputs, 1 / 128, settings_for(tyx_mode))
    try:
        ws.predict()
        before = fingerprint((ws.candidate, ws.multipliers, tuple(ws.frames)))
        original = ws.engine.project_bending

        def late(*args, **kwargs):
            original(*args, **kwargs)
            raise SolverFailure("late actual bending launch failure")

        with monkeypatch.context() as patch:
            patch.setattr(ws.engine, "project_bending", late)
            with pytest.raises(SolverFailure, match="late actual"):
                ws.position_sweep()
        assert fingerprint((ws.candidate, ws.multipliers, tuple(ws.frames))) == before
        ws.position_sweep()
        fresh = program.workspace("cloth", initial, inputs, 1 / 128, settings_for(tyx_mode))
        try:
            fresh.predict()
            fresh.position_sweep()
            assert fingerprint((ws.candidate, ws.multipliers, tuple(ws.frames))) == fingerprint(
                (fresh.candidate, fresh.multipliers, tuple(fresh.frames))
            )
        finally:
            fresh.close()
    finally:
        ws.close()


def test_native_participant_rejects_unrecorded_position_load(tyx_mode):
    model, material, initial, inputs = case()
    factory = TyxCompositeClothParticipant(model, material, tyx_mode, budgets=BUDGETS, representation=REPRESENTATION)
    ws = factory.create_workspace(
        "cloth", initial, inputs, 1 / 64, settings_for(tyx_mode), BackendContext("tyx:" + tyx_mode)
    )
    try:
        ws.predict()
        with pytest.raises(SolverFailure, match="unrecorded"):
            ws.apply_position_load(np.array([0]), None, np.zeros((1, 3)))
    finally:
        ws.close()


@pytest.mark.parametrize("phase", ["prepare_velocity", "diagnostics", "close"])
def test_actual_native_late_substep_failure_preserves_outer_state_and_retry(tyx_mode, phase):
    from phymeo.backends.tyx_composite_cloth import TyxCompositeClothProgram
    from phymeo.core.scene import SceneInputs, SceneState, StepSettings
    from phymeo.runtime.scene_stepper import SceneStepper

    model, material, initial, inputs = case()

    class FaultProgram(TyxCompositeClothProgram):
        created = 0
        fault = True

        def create_workspace(self, *args):
            ws = super().create_workspace(*args)
            self.created += 1
            if self.fault and self.created == 2:
                original = getattr(ws, phase)

                def fail():
                    original()
                    raise SolverFailure("native second-substep " + phase)

                setattr(ws, phase, fail)
            return ws

    settings = StepSettings(
        substeps=2,
        position_iterations=512,
        velocity_iterations=1,
        tolerance=2e-5,
        record_substeps=True,
    )
    context = BackendContext("tyx:" + tyx_mode)
    state, forces = SceneState({"cloth": initial}), SceneInputs({"cloth": inputs})
    program = FaultProgram(model, material, tyx_mode, budgets=BUDGETS, representation=REPRESENTATION)
    with SceneStepper(program.scene, program, settings, context) as runtime:
        failed = runtime.step(state, forces, 1 / 32)
        assert not failed.report.accepted and failed.report.substeps[0].accepted
        assert fingerprint(failed.state) == fingerprint(state) and not failed.substep_states
        program.fault = False
        retry = runtime.step(state, forces, 1 / 32)
    fresh_program = program_for(model, material, tyx_mode, representation=REPRESENTATION)
    with SceneStepper(fresh_program.scene, fresh_program, settings, context) as runtime:
        fresh = runtime.step(state, forces, 1 / 32)
    assert fresh.report.accepted and retry.report.accepted
    assert fingerprint(fresh.state) == fingerprint(retry.state)
    assert fingerprint(fresh.substep_states) == fingerprint(retry.substep_states)


def test_actual_native_two_disconnected_copies_no_double_application(tyx_mode):
    one, _ = run_window("A", 1 / 64, mode=tyx_mode, representation=REPRESENTATION)
    two, _ = run_window("A", 1 / 64, mode=tyx_mode, representation=REPRESENTATION, copies=2)
    for a, b in zip(one, two, strict=True):
        x, y = a.state.domains["cloth"], b.state.domains["cloth"]
        np.testing.assert_allclose(y.position[:4], x.position, rtol=0, atol=2e-5)
        np.testing.assert_allclose(y.position[4:] - [4.0, 0.0, 0.0], x.position, rtol=0, atol=2e-5)
        np.testing.assert_allclose(y.velocity[4:], x.velocity, rtol=0, atol=2e-5)
