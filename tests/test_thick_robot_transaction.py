"""Whole-frame rollback, exact selected-pre identity and live resource controls."""

from dataclasses import replace
from functools import lru_cache

import pytest
from test_terminal_impulse import ObservedProgram, ObservedWorkspace
from test_thick_robot_contact import build, diag, step

import phymeo as ph
from phymeo.core.scene import fingerprint


class FaultProgram(ObservedProgram):
    def __init__(self, base, failure=None, fail_at=1):
        super().__init__(base)
        self.failure = failure
        self.fail_at = fail_at
        self.triggered = False

    def create_workspace(self, scene, state, *args):
        ws = FaultWorkspace(
            self, self.base.create_workspace(scene, state, *args), len(self.workspaces) + 1
        )
        ws.late = state.history["base/count"][0] >= self.fail_at
        self.workspaces.append(ws)
        return ws


class FaultWorkspace(ObservedWorkspace):
    def predict(self):
        super().predict()
        if self.late and self.owner.failure == "native" and not self.owner.triggered:
            self.owner.triggered = True
            raise ph.SolverFailure("injected late native")

    def apply_point_velocity_impulses(self, values):
        self.counts["terminal"] += 1
        self.native.apply_point_velocity_impulses(values)
        if self.late and self.owner.failure == "terminal" and not self.owner.triggered:
            self.owner.triggered = True
            raise ph.SolverFailure("injected late terminal")

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
        if self.late and self.owner.failure == "close" and not self.owner.triggered:
            self.owner.triggered = True
            raise ph.SolverFailure("injected late close")


@pytest.mark.parametrize("substeps", [2, 3])
@pytest.mark.parametrize("failure", ["native", "path", "terminal", "mapping", "CCD", "close"])
def test_late_failure_rollback_and_retry_fresh(monkeypatch, failure, substeps):
    from phymeo.solvers._thick_contact_workspace import ThickContactWorkspace as W

    original = W._evaluate
    ccd = W._require_ccd

    def evaluate(self, terminal=False):
        original(self, terminal)
        owner = self.program.base
        if (
            self.initial.history["base/count"][0] >= substeps - 1
            and owner.failure == "mapping"
            and not owner.triggered
        ):
            owner.triggered = True
            raise ph.SolverFailure("injected late independent mapping audit")

    def certificate(self):
        owner = self.program.base
        if (
            self.initial.history["base/count"][0] >= substeps - 1
            and owner.failure == "CCD"
            and not owner.triggered
        ):
            owner.triggered = True
            raise ph.SolverFailure("injected late whole-prism CCD unresolved")
        return ccd(self)

    monkeypatch.setattr(W, "_evaluate", evaluate)
    monkeypatch.setattr(W, "_require_ccd", certificate)
    scene, state, inputs, program, settings, context = build(
        base_transform=lambda b: FaultProgram(b, failure, substeps - 1)
    )
    settings = replace(settings, substeps=substeps)
    before = fingerprint((state, inputs))
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        rejected = runtime.step(state, inputs, 0.25 * substeps)
        assert program.base.triggered and not rejected.report.accepted
        assert rejected.report.completed_substeps == substeps - 1
        assert fingerprint(rejected.state) == fingerprint(state) and rejected.substep_states == ()
        program.base.failure = None
        retried = runtime.step(state, inputs, 0.25 * substeps)
        assert retried.report.accepted, retried.report.reason
    assert fingerprint((state, inputs)) == before
    assert all(w.counts["close"] == 1 for w in program.base.workspaces)
    expected = _fresh(substeps)
    assert fingerprint(retried.state) == fingerprint(expected)


@lru_cache(maxsize=2)
def _fresh(substeps):
    fresh = build(base_transform=lambda b: FaultProgram(b, None, substeps - 1))
    with ph.SceneStepper(
        fresh[0], fresh[3], replace(fresh[4], substeps=substeps), fresh[5], fresh[3].registry
    ) as runtime:
        expected = runtime.step(fresh[1], fresh[2], 0.25 * substeps)
    assert expected.report.accepted, expected.report.reason
    return expected.state


def test_best_does_not_hide_execution_or_unknown_failure(monkeypatch):
    import phymeo.solvers._thick_contact_workspace as module

    original = module.triangle_prism_patch_sweep

    def forged(*args, **kwargs):
        return replace(original(*args, **kwargs), input_fingerprint="wrong-path")

    monkeypatch.setattr(module, "triangle_prism_patch_sweep", forged)
    r = step(build(), accepted=False)
    assert "fingerprint" in r.report.reason


def test_terminal_pre_replay_matches_full_selected_held():
    r = step(build(base_transform=ObservedProgram))
    d = diag(r)
    pre = d["terminal_ledger"]["pre_impulse_state"]
    assert fingerprint(pre) == fingerprint(d["held_state"])
    assert pre.history["base/count"][0] == 1
    assert r.state.history["base/count"][0] == 1
    assert d["terminal_ledger"]["applications"] == 1


@pytest.mark.parametrize(
    "options",
    [
        {"max_anchors_per_pair": 2},
        {"max_manifold_rebuilds": 1, "max_anchors_per_pair": 3},
        {"max_manifold_rebuilds": 1},
    ],
)
def test_anchor_capacity_and_geometry_rebuild_budget_fail_closed(options):
    oblique = options == {"max_manifold_rebuilds": 1}
    r = step(
        build(
            scenario="A" if oblique else "C",
            mu=1.0 if oblique else 0.0,
            rotate="oblique" if oblique else False,
            program_options=options,
        ),
        accepted=False,
    )
    assert ("rebuild" if oblique else "anchor") in r.report.reason


def test_partial_construction_closes_workspaces(monkeypatch):
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
    r = step(case, accepted=False)
    assert closed == [True] and "construction" in r.report.reason


def test_all_actual_trials_count_once():
    case = build(base_transform=ObservedProgram)
    r = step(case)
    d = diag(r)
    observed = case[3].base
    assert len(observed.workspaces) == d["held_replays"] + d["terminal_replays"]
    assert len(observed.workspaces) > 2
    assert all(
        w.counts["predict"] == w.counts["terminal"] == w.counts["close"] == 1
        for w in observed.workspaces
    )


def test_terminal_only_replay_rechecks_complete_selected_pre_evidence():
    from phymeo.solvers._thick_contact_replay import require_selected_pre

    case = build()
    r = step(case)
    e = diag(r)["terminal_ledger"]
    require_selected_pre(e, e, case[2], case[2])
    altered = dict(e)
    altered["pre_impulse_diagnostics"] = {
        **e["pre_impulse_diagnostics"],
        "corrupted_path_metadata": True,
    }
    with pytest.raises(ph.SolverFailure, match="complete held pre"):
        require_selected_pre(altered, e, case[2], case[2])
