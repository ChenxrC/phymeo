"""Finite contact whole-frame failures and retry=fresh controls."""

import math
from dataclasses import replace

import numpy as np
import pytest
from test_finite_robot_contact import build, step
from test_terminal_impulse import ObservedProgram, ObservedWorkspace

import phymeo as ph
from phymeo.core.scene import fingerprint


class FaultProgram(ObservedProgram):
    def __init__(self, base, failure=None, fail_at=1):
        super().__init__(base)
        self.failure, self.fail_at, self.triggered = failure, fail_at, False

    def create_workspace(self, scene, state, *args):
        ws = FaultWorkspace(
            self, self.base.create_workspace(scene, state, *args), len(self.workspaces) + 1
        )
        ws.late = state.history["base/count"][0] >= self.fail_at
        ws.last_requests = False
        self.workspaces.append(ws)
        return ws


class FaultWorkspace(ObservedWorkspace):
    def apply_point_velocity_impulses(self, values):
        self.last_requests = bool(values)
        self.counts["terminal"] += 1
        self.native.apply_point_velocity_impulses(values)
        if self.late and values and not self.owner.triggered:
            if self.owner.failure == "terminal":
                self.owner.triggered = True
                raise ph.SolverFailure("injected late terminal")
            if self.owner.failure == "ledger":
                self.owner.triggered = True
                self.native.history["base/count"] += 1

    def diagnostics(self):
        d = self.native.diagnostics()
        if self.late and self.owner.failure == "path" and not self.owner.triggered:
            d["participants"]["nodes"]["material_path"] = dict(
                d["participants"]["nodes"]["material_path"], schema="corrupt"
            )
            self.owner.triggered = True
        return d

    def close(self):
        self.counts["close"] += 1
        self.native.close()
        if (
            self.late
            and self.last_requests
            and self.owner.failure == "close"
            and not self.owner.triggered
        ):
            self.owner.triggered = True
            raise ph.SolverFailure("injected late close")


@pytest.mark.parametrize("substeps", [2, 3])
@pytest.mark.parametrize("failure", [None, "path", "terminal", "ledger", "CCD", "close"])
def test_late_failure_and_retry_fresh(monkeypatch, failure, substeps):
    import phymeo.solvers.dynamic_robot_finite_contact as module

    original = module._FiniteWorkspace._require_ccd

    def certificate(self):
        owner = self.program.base
        if (
            self.initial.history["base/count"][0] >= substeps - 1
            and owner.failure == "CCD"
            and not owner.triggered
        ):
            owner.triggered = True
            raise ph.SolverFailure("injected late CCD unresolved")
        return original(self)

    monkeypatch.setattr(module._FiniteWorkspace, "_require_ccd", certificate)
    y = (2 * substeps - 1) / 8 + 0.125
    case = build(
        position=[[0.0, y, 0.0]], base_transform=lambda b: FaultProgram(b, failure, substeps - 1)
    )
    scene, state, inputs, program, settings, context = case
    settings = replace(settings, substeps=substeps)
    initial = fingerprint((state, inputs))
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        result = runtime.step(state, inputs, 0.25 * substeps)
        if failure:
            assert program.base.triggered
            assert not result.report.accepted and result.report.completed_substeps == substeps - 1
            assert fingerprint(result.state) == fingerprint(state)
            assert result.substep_states == ()
            program.base.failure = None
            result = runtime.step(state, inputs, 0.25 * substeps)
        assert result.report.accepted, result.report.reason
        assert result.state.history["base/count"][0] == substeps
        assert [s.history["base/count"][0] for s in result.substep_states] == list(
            range(1, substeps + 1)
        )
    assert fingerprint((state, inputs)) == initial
    assert all(w.counts["close"] == 1 for w in program.base.workspaces)
    fresh = build(
        position=[[0.0, y, 0.0]], base_transform=lambda b: FaultProgram(b, None, substeps - 1)
    )
    with ph.SceneStepper(
        fresh[0], fresh[3], replace(fresh[4], substeps=substeps), fresh[5], fresh[3].registry
    ) as runtime:
        expected = runtime.step(fresh[1], fresh[2], 0.25 * substeps)
    assert expected.report.accepted, expected.report.reason
    assert fingerprint(result.state) == fingerprint(expected.state)


def test_endpoint_safe_midpath_crossing_rejects():
    alpha = 0.25
    x = 1.5 * np.array([-math.sin(alpha), math.cos(alpha), 0.0])
    case = build(rotor=True, position=[x], velocity=[[0.0, 0.0, 0.0]], q=[alpha], omega=[math.pi])
    r = step(case, 1.0, False)
    assert "CCD violation" in r.report.reason
    assert fingerprint(r.state) == fingerprint(case[1])


def test_certificate_bound_to_selected_candidate(monkeypatch):
    import phymeo.solvers.dynamic_robot_finite_contact as module

    original = module.robot_patch_sweep

    def forged(*args, **kwargs):
        return replace(original(*args, **kwargs), input_fingerprint="wrong-path")

    monkeypatch.setattr(module, "robot_patch_sweep", forged)
    r = step(build(), accepted=False)
    assert "fingerprint" in r.report.reason


def test_best_cannot_mask_unknown_or_native_failure(monkeypatch):
    import phymeo.solvers.dynamic_robot_finite_contact as module

    original = module._FiniteWorkspace._evaluate
    armed = []

    def fail_after_best(self, terminal=False):
        if terminal and self.vbest is not None:
            armed.append(True)
            raise ph.SolverFailure("unknown/native after admitted best")
        return original(self, terminal)

    monkeypatch.setattr(module._FiniteWorkspace, "_evaluate", fail_after_best)
    case = build()
    case[3].relaxation = 0.5
    result = step(case, accepted=False)
    assert armed and "after admitted best" in result.report.reason


def test_half_constructed_resources_close(monkeypatch):
    case = build()
    base = case[3].base
    closed = []
    original = base.participants["robot"].create_workspace

    def robot(*args):
        ws = original(*args)
        close = ws.close

        def observed():
            closed.append(True)
            close()

        ws.close = observed
        return ws

    def fail(*args):
        raise ph.SolverFailure("injected node construction")

    monkeypatch.setattr(base.participants["robot"], "create_workspace", robot)
    monkeypatch.setattr(base.participants["nodes"], "create_workspace", fail)
    result = step(case, accepted=False)
    assert closed == [True] and "construction" in result.report.reason


@pytest.mark.parametrize("count", [2, 4])
def test_original_high_slip_refinement_energy_rejects_and_rolls_back(count):
    case = build(rotor=True, X=1.5)
    scene, state, inputs, p, settings, context = case
    with ph.SceneStepper(
        scene, p, replace(settings, substeps=count), context, p.registry
    ) as runtime:
        result = runtime.step(state, inputs, 1.0)
    assert not result.report.accepted and result.report.completed_substeps == 1
    assert "energy" in result.report.reason
    assert fingerprint(result.state) == fingerprint(state)


def test_rotor_trial_outside_supported_angle_rejects_transactionally():
    case = build(
        rotor=True, q=[3.9], omega=[1.0], position=[[0.0, 0.0, 2.0]], velocity=[[0.0, 0.0, 0.0]]
    )
    result = step(case, 1.0, False)
    assert "[-4,4]" in result.report.reason
    assert fingerprint(result.state) == fingerprint(case[1])


def test_actual_unknown_certificate_rejects_held_best():
    case = build(rotor=True)
    case[3].max_ccd_intervals = 1
    result = step(case, 1.0, False)
    assert "CCD unresolved" in result.report.reason
    assert fingerprint(result.state) == fingerprint(case[1])
