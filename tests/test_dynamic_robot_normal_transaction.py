"""Whole-frame rollback after real late native stage failures."""

from dataclasses import replace

import pytest
from test_dynamic_robot_normal_contact import custom
from test_terminal_impulse import ObservedProgram, ObservedWorkspace

import phymeo as ph
from phymeo.core.scene import fingerprint


class FaultProgram(ObservedProgram):
    def __init__(self, base, failure=None):
        super().__init__(base)
        self.failure = failure
        self.triggered = False
        self.current_terminal = False

    def create_workspace(self, scene, state, *args):
        ws = FaultWorkspace(
            self, self.base.create_workspace(scene, state, *args), len(self.workspaces) + 1
        )
        ws.late = bool(state.history["base/count"][0] >= 1)
        ws.last_requests = False
        self.workspaces.append(ws)
        return ws


class FaultWorkspace(ObservedWorkspace):
    def apply_point_velocity_impulses(self, values):
        self.last_requests = bool(values)
        self.owner.current_terminal = self.last_requests
        self.counts["terminal"] += 1
        self.native.apply_point_velocity_impulses(values)
        if self.late and self.last_requests and not self.owner.triggered:
            if self.owner.failure == "terminal":
                self.owner.triggered = True
                raise ph.SolverFailure("injected late terminal")
            if self.owner.failure == "history":
                self.owner.triggered = True
                self.native.history["base/count"] += 1

    def diagnostics(self):
        d = self.native.diagnostics()
        # Trigger only on the second-substep terminal replay, before impulse,
        # after held stage has been accepted; material trace remains native data.
        if self.late and not self.owner.triggered and self.owner.failure == "trace":
            path = d["participants"]["nodes"]["material_path"]
            path = dict(path)
            path["schema"] = "corrupt"
            d["participants"]["nodes"]["material_path"] = path
            self.owner.triggered = True
        return d

    def close(self):
        self.counts["close"] += 1
        self.native.close()
        if (
            self.late
            and self.last_requests
            and not self.owner.triggered
            and self.owner.failure == "close"
        ):
            self.owner.triggered = True
            raise ph.SolverFailure("injected late close")


@pytest.mark.parametrize("failure", [None, "terminal", "history", "trace", "close"])
def test_late_failure_full_rollback_and_retry_fresh(failure):
    case = custom(
        position=[[0.0, 3 / 8, 0.0]], base_transform=lambda base: FaultProgram(base, failure)
    )
    scene, state, inputs, program, settings, context = case
    settings = replace(settings, substeps=2)
    observed = program.base
    initial = fingerprint((state, inputs))
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        result = runtime.step(state, inputs, 0.5)
        if failure:
            assert observed.triggered, "injection must really execute"
            assert not result.report.accepted
            assert result.report.completed_substeps == 1
            assert fingerprint(result.state) == fingerprint(state)
            assert result.substep_states == ()
            observed.failure = None
            result = runtime.step(state, inputs, 0.5)
        assert result.report.accepted, result.report.reason
        assert result.state.history["base/count"][0] == 2
        assert [s.history["base/count"][0] for s in result.substep_states] == [1, 2]
        assert observed.close_calls == 0
    assert observed.close_calls == 1
    assert fingerprint((state, inputs)) == initial
    assert all(ws.counts["close"] == 1 for ws in observed.workspaces)
    fresh = custom(position=[[0.0, 3 / 8, 0.0]], base_transform=FaultProgram)
    with ph.SceneStepper(
        fresh[0], fresh[3], replace(fresh[4], substeps=2), fresh[5], fresh[3].registry
    ) as runtime:
        expected = runtime.step(fresh[1], fresh[2], 0.5)
    assert expected.report.accepted, expected.report.reason
    assert fingerprint(result.state) == fingerprint(expected.state)


def test_two_substeps_each_get_their_own_native_terminal_operation():
    case = custom(position=[[0.0, 3 / 8, 0.0]], base_transform=FaultProgram)
    with ph.SceneStepper(
        case[0], case[3], replace(case[4], substeps=2), case[5], case[3].registry
    ) as runtime:
        result = runtime.step(case[1], case[2], 0.5)
    assert result.report.accepted, result.report.reason
    assert len(result.report.substeps) == 2
    for sub in result.report.substeps:
        d = sub.diagnostics["dynamic_robot_normal_contact"]
        assert d["terminal_replays"] >= 1
        assert d["terminal_evidence"]["applications"] == 1
    assert all(
        ws.counts["predict"] == 1 and ws.counts["terminal"] == 1 for ws in case[3].base.workspaces
    )


@pytest.mark.parametrize("kind", [None, "terminal", "trace", "close"])
def test_best_terminal_candidate_never_masks_later_real_native_failure(monkeypatch, kind):
    import phymeo.solvers.dynamic_robot_normal_contact as module

    class ArmedProgram(ObservedProgram):
        armed = False
        triggered = False

        def create_workspace(self, *args):
            ws = ArmedWorkspace(self, self.base.create_workspace(*args), len(self.workspaces) + 1)
            self.workspaces.append(ws)
            return ws

    class ArmedWorkspace(ObservedWorkspace):
        def apply_point_velocity_impulses(self, values):
            super().apply_point_velocity_impulses(values)
            if self.owner.armed and kind == "terminal":
                self.owner.triggered = True
                raise ph.SolverFailure("injected terminal after admitted best")

        def diagnostics(self):
            d = self.native.diagnostics()
            if self.owner.armed and kind == "trace":
                self.owner.triggered = True
                path = dict(d["participants"]["nodes"]["material_path"])
                path["schema"] = "corrupt-after-best"
                d["participants"]["nodes"]["material_path"] = path
            return d

        def close(self):
            super().close()
            if self.owner.armed and kind == "close":
                self.owner.triggered = True
                raise ph.SolverFailure("injected close after admitted best")

    evaluate = module._DynamicNormalWorkspace._evaluate

    def arm_after_best(self, terminal=False):
        if terminal and self.vbest is not None:
            self.program.base.armed = True
        return evaluate(self, terminal)

    monkeypatch.setattr(module._DynamicNormalWorkspace, "_evaluate", arm_after_best)
    case = custom(base_transform=ArmedProgram)
    case[3].relaxation = 0.5
    with ph.SceneStepper(case[0], case[3], case[4], case[5], case[3].registry) as runtime:
        result = runtime.step(case[1], case[2], 0.25)
    assert case[3].base.armed, "an admissible best terminal candidate must precede fault"
    assert result.report.accepted == (kind is None), result.report.reason
    if kind:
        assert case[3].base.triggered
        assert fingerprint(result.state) == fingerprint(case[1])
    assert all(ws.counts["close"] == 1 for ws in case[3].base.workspaces)


def test_native_half_construction_failure_closes_already_built_participant(monkeypatch):
    case = custom()
    base = case[3].base
    closed = []
    create = base.participants["robot"].create_workspace

    def robot(*args):
        ws = create(*args)
        close = ws.close

        def observed_close():
            closed.append(True)
            close()

        ws.close = observed_close
        return ws

    def node(*args):
        raise ph.SolverFailure("injected node construction")

    monkeypatch.setattr(base.participants["robot"], "create_workspace", robot)
    monkeypatch.setattr(base.participants["nodes"], "create_workspace", node)
    with ph.SceneStepper(case[0], case[3], case[4], case[5], case[3].registry) as runtime:
        result = runtime.step(case[1], case[2], 0.25)
    assert not result.report.accepted and "construction" in result.report.reason
    assert closed == [True]
    assert fingerprint(result.state) == fingerprint(case[1])
