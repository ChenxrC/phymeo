"""Frozen physical windows, standalone powered replay, and independent controls."""

import gzip
import json
import os
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from fractions import Fraction as F
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from test_bending_bounds import independent_energy
from test_composite_cloth import BUDGETS, POLICY, case, program_for, settings_for

from phymeo.constraints.cloth_material import ClothMaterial
from phymeo.constraints.material_trace import CompositeExternalImpulses
from phymeo.core.capabilities import SolverFailure
from phymeo.core.model import ParticleModel
from phymeo.core.scene import BackendContext, SceneInputs, SceneState, fingerprint
from phymeo.core.state import ParticleState, StepInputs
from phymeo.coupling.composite_cloth import CompositeClothParticipant
from phymeo.runtime.load_replay import SceneLoadReplay
from phymeo.runtime.scene_stepper import SceneStepper
from phymeo.validation.composite_material_energy import (
    CompositeMaterialEnergyLedger,
    require_material_window,
)


def persist_evidence(label, data):
    """Save this invocation's complete observations without another physics run."""
    destination = os.environ.get("P65_B2_EVIDENCE")
    if not destination:
        return

    def encode(value):
        if isinstance(value, np.ndarray):
            return {"dtype": value.dtype.str, "shape": value.shape, "values": value.tolist()}
        if isinstance(value, np.generic):
            return value.item()
        if is_dataclass(value):
            return {
                "type": type(value).__module__ + "." + type(value).__qualname__,
                "fields": {f.name: encode(getattr(value, f.name)) for f in fields(value)},
            }
        if isinstance(value, Mapping):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(v) for v in value]
        if isinstance(value, np.dtype):
            return str(value)
        return value

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    with gzip.open(
        root / (label + "-" + uuid4().hex + ".json.gz"), "wt", encoding="utf-8"
    ) as stream:
        json.dump(encode(data), stream, sort_keys=True, allow_nan=False)


def run_window(name, h, *, mode=None, representation=None, **kwargs):
    model, material, initial, inputs = case(name, **kwargs)
    program = program_for(model, material, mode, representation=representation)
    scene_state, scene_inputs = SceneState({"cloth": initial}), SceneInputs({"cloth": inputs})
    context = BackendContext("numpy" if mode is None else "tyx:" + mode)
    records, physical, current = [], [], scene_state
    duration = F(1, 4) if name == "A" else F(1, 8)
    persist_evidence(
        "window-initial",
        dict(
            case=name,
            h=h,
            mode=mode,
            representation=representation,
            parameters=kwargs,
            model=model,
            material=material,
            initial=scene_state,
            inputs=scene_inputs,
        ),
    )
    steps = int(duration / F(h))
    assert steps * F(h) == duration
    with SceneStepper(program.scene, program, settings_for(mode), context) as runtime:
        for step_index in range(steps):
            result = runtime.step(current, scene_inputs, h)
            persist_evidence(
                "window-step",
                dict(
                    case=name,
                    h=h,
                    mode=mode,
                    representation=representation,
                    parameters=kwargs,
                    step=step_index,
                    result=result,
                ),
            )
            assert result.report.accepted, result.report.reason
            assert len(result.report.substeps) == 1
            diagnostic = result.report.substeps[0].diagnostics
            measure = diagnostic["physical_audit"]
            assert measure["external_transfer_work"] == 0
            assert diagnostic["material_audit"]["physical_affine_domain"].status == "certified"
            physical.append(measure)
            records.append(result)
            current = result.state
    total = require_material_window(physical, BUDGETS)
    assert total["positive_energy_sum"] <= 1e-3 and total["angular_l1_sum"] <= 1e-3
    persist_evidence(
        "window-total",
        dict(
            case=name,
            h=h,
            mode=mode,
            representation=representation,
            parameters=kwargs,
            total=total,
            steps=len(records),
        ),
    )
    return records, total


def equal_runs(first, second):
    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert fingerprint(a.state) == fingerprint(b.state)
        assert fingerprint(a.state.history) == fingerprint(b.state.history)
        # Exclude only typed wall-clock timings, never material/physical evidence.
        assert fingerprint(a.report.substeps[0].diagnostics) == fingerprint(
            b.report.substeps[0].diagnostics
        )


@pytest.mark.parametrize(
    "name,h",
    [("A", 1 / 64), ("A", 1 / 128), ("A", 1 / 256), ("B", 1 / 128), ("B", 1 / 256), ("B", 1 / 512)],
)
def test_frozen_release_windows_two_fresh_rebuilds(name, h):
    first, total1 = run_window(name, h)
    second, total2 = run_window(name, h)
    equal_runs(first, second)
    assert total1 == total2
    if name == "A":
        for record in first:
            state = record.state.domains["cloth"]
            np.testing.assert_allclose(
                state.position[2, [0, 2]], state.position[3, [0, 2]], atol=1e-8
            )
            assert abs(state.position[2, 1] + state.position[3, 1]) < 1e-8
            np.testing.assert_allclose(np.sum(state.velocity, axis=0), 0.0, atol=1e-8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"perturb": 2**-12},
        {"perturb": -(2**-12)},
        {"mass_scale": 0.5},
        {"mass_scale": 2.0},
        {
            "rotation": np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]),
            "translation": np.array([3 / 8, -1 / 4, 1 / 8]),
        },
    ],
)
def test_strip_perturbations_and_proper_transform_keep_original_window_budgets(kwargs):
    if "rotation" in kwargs:
        assert np.linalg.det(kwargs["rotation"]) == 1
    first, _ = run_window("B", 1 / 128, **kwargs)
    second, _ = run_window("B", 1 / 128, **kwargs)
    equal_runs(first, second)


def test_two_disconnected_copies_share_one_owner_without_double_application():
    one, _ = run_window("A", 1 / 64)
    two, _ = run_window("A", 1 / 64, copies=2)
    for a, b in zip(one, two, strict=True):
        x, y = a.state.domains["cloth"], b.state.domains["cloth"]
        np.testing.assert_allclose(y.position[:4], x.position, atol=1e-8, rtol=0)
        np.testing.assert_allclose(y.position[4:] - [4.0, 0.0, 0.0], x.position, atol=1e-8, rtol=0)
        np.testing.assert_allclose(y.velocity[4:], x.velocity, atol=1e-8, rtol=0)


def standalone_replay(mode=None, *, representation=None):
    model, material, initial, original = case("B")
    h, settings = 1 / 128, settings_for(mode)
    context = BackendContext("numpy" if mode is None else "tyx:" + mode)
    program = program_for(model, material, mode, representation=representation)
    scene_state, scene_inputs = SceneState({"cloth": initial}), SceneInputs({"cloth": original})
    held = np.zeros((6, 3))
    held[0, 0], held[5, 0] = 1 / 64, -1 / 64
    with SceneLoadReplay(
        program.scene,
        program,
        scene_state,
        scene_inputs,
        h,
        channels={"cloth": ("force",)},
        settings=settings,
        context=context,
    ) as replay:
        first = replay.evaluate({"cloth": {"force": held}})
        zero = replay.evaluate({})
        third = replay.evaluate({"cloth": {"force": held}})
        persist_evidence(
            "standalone-p-zero-p",
            dict(
                mode=mode,
                representation=representation,
                initial=scene_state,
                inputs=scene_inputs,
                held=held,
                trials=(first, zero, third),
            ),
        )
        assert all(t.result.report.accepted for t in (first, zero, third))
        equal_runs([first.result], [third.result])
        assert fingerprint(replay.snapshot()) == fingerprint(scene_state)
    if mode is None:
        factory = CompositeClothParticipant(model, material, budgets=BUDGETS)
    else:
        from phymeo.backends.tyx_composite_cloth import TyxCompositeClothParticipant

        factory = TyxCompositeClothParticipant(
            model, material, mode, budgets=BUDGETS, representation=representation
        )
    ws = factory.create_workspace(
        "cloth", initial, third.effective_inputs.domains["cloth"], h, settings, context
    )
    try:
        ws.predict()
        for _ in range(512):
            ws.position_sweep()
            if ws.position_metrics().max_residual <= settings.tolerance:
                break
        else:
            pytest.fail("standalone frozen position budget exhausted")
        ws.prepare_velocity()
        pre = ws.result()
        assert fingerprint(pre) == fingerprint(third.result.state.domains["cloth"])
        before_trace = fingerprint(ws.diagnostics()["material_path"])
        terminal = np.zeros((6, 3))
        terminal[2, 1] = 1 / 64
        ws.apply_velocity_impulses(np.array([2]), None, terminal[[2]])
        final = ws.result()
        assert fingerprint(ws.diagnostics()["material_path"]) == before_trace
        receipt = CompositeExternalImpulses.bind(held, terminal, h, original, pre, final)
        audit = CompositeMaterialEnergyLedger(
            model,
            material,
            initial,
            original,
            h,
            policy=POLICY,
            budgets=BUDGETS,
            tolerance=settings.tolerance,
        )
        result = audit.require(
            final,
            selected_pre=pre,
            external=receipt,
            held_requests=held,
            terminal_requests=terminal,
            effective_inputs=third.effective_inputs.domains["cloth"],
        )
        expected_terminal = float(
            terminal[2] @ pre.velocity[2] + model.inverse_mass[2] * (terminal[2] @ terminal[2]) / 2
        )
        assert result["terminal_transfer_work"] == pytest.approx(expected_terminal, abs=1e-15)
        for broken in (
            None,
            replace(receipt, selected_pre_fingerprint="stale"),
            replace(receipt, final_state_fingerprint="stale"),
        ):
            with pytest.raises(SolverFailure):
                audit.require(
                    final,
                    selected_pre=pre,
                    external=broken,
                    held_requests=held,
                    terminal_requests=terminal,
                    effective_inputs=third.effective_inputs.domains["cloth"],
                )
        with pytest.raises(SolverFailure, match="internal"):
            audit.require(
                final,
                selected_pre=pre,
                external=receipt,
                held_requests=held,
                terminal_requests=terminal,
                effective_inputs=third.effective_inputs.domains["cloth"],
                internal_boundary=True,
            )
        persist_evidence(
            "standalone-terminal",
            dict(
                mode=mode,
                representation=representation,
                selected_pre=pre,
                final=final,
                receipt=receipt,
                audit=result,
                diagnostics=ws.diagnostics(),
            ),
        )
        return fingerprint(final), before_trace, result
    finally:
        ws.close()


def test_standalone_p_zero_p_terminal_once_external_work_and_fresh_retry():
    assert standalone_replay() == standalone_replay()


def test_one_particle_independent_accounted_and_unaccounted_energy_control():
    model, material = ParticleModel(np.ones(1), np.zeros(3)), ClothMaterial()
    initial = ParticleState(np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0]]))
    original, h, j = StepInputs.zeros(1), 1 / 128, F(1, 64)
    held = np.array([[float(j), 0.0, 0.0]])
    final = ParticleState(
        np.array([[h * (1 + float(j)), 0.0, 0.0]]), np.array([[1 + float(j), 0.0, 0.0]])
    )
    audit = CompositeMaterialEnergyLedger(
        model, material, initial, original, h, policy=POLICY, budgets=BUDGETS, tolerance=1e-8
    )
    with pytest.raises(SolverFailure, match="energy"):
        audit.require(final)
    with pytest.raises(SolverFailure, match="receipt"):
        audit.require(final, held_requests=held)
    receipt = CompositeExternalImpulses.bind(held, np.zeros((1, 3)), h, original, final, final)
    result = audit.require(
        final, external=receipt, held_requests=held, effective_inputs=StepInputs(held / h)
    )
    assert result["raw_energy_defect"] == (float(j + j * j / 2),) * 2
    assert result["held_transfer_work"] == float(j * (1 + j))
    assert result["unexplained_defect"] == (float(-j * j / 2),) * 2


def test_window_positive_parts_and_component_absolute_values_forbid_cancellation():
    def report(energy, angular):
        return dict(
            unexplained_defect=(energy, energy), angular_defect=tuple((v, v) for v in angular)
        )

    with pytest.raises(SolverFailure, match="window"):
        require_material_window(
            [
                report(0.0006, (0.0, 0.0, 0.0)),
                report(-1.0, (0.0, 0.0, 0.0)),
                report(0.0006, (0.0, 0.0, 0.0)),
            ],
            BUDGETS,
        )
    with pytest.raises(SolverFailure, match="window"):
        require_material_window(
            [report(0.0, (0.0006, 0.0, 0.0)), report(0.0, (-0.0006, 0.0, 0.0))], BUDGETS
        )


def test_physical_frozen_force_moment_and_power_match_independent_energy_derivative():
    from phymeo.validation.composite_material_energy import material_power_bounds

    model, material, state, _ = case("A")
    state.position[2, 2] = 1 / 8
    state.position[3, 2] = -1 / 16
    state.velocity[:] = (
        np.array([[1.0, 2.0, -1.0], [-2.0, 1.0, 0.0], [0.0, -1.0, 2.0], [1.0, 0.0, -1.0]]) / 8
    )
    evidence = material_power_bounds(model, material, state, policy=POLICY)
    assert evidence["status"] == "bounded-smooth"
    for low, high in (*evidence["resultant"], *evidence["moment"]):
        assert low <= 0 <= high
    step = 2**-18
    independent_rate = (
        independent_energy(state.position + step * state.velocity)
        - independent_energy(state.position - step * state.velocity)
    ) / (2 * step)
    low, high = evidence["elastic_rate"]
    assert low - 2e-7 < independent_rate < high + 2e-7
