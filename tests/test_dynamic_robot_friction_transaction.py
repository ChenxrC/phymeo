"""Whole-frame rollback after real late native stage failures."""

from dataclasses import replace

import pytest
from test_dynamic_robot_friction_contact import build as custom
from test_terminal_impulse import ObservedProgram, ObservedWorkspace

import phymeo as ph
from phymeo.core.scene import fingerprint


class FaultProgram(ObservedProgram):
    def __init__(self, base, failure=None, fail_at=1):
        super().__init__(base)
        self.failure = failure
        self.triggered = False
        self.fail_at = fail_at
        self.current_terminal = False

    def create_workspace(self, scene, state, *args):
        ws = FaultWorkspace(
            self, self.base.create_workspace(scene, state, *args), len(self.workspaces) + 1
        )
        ws.late = bool(state.history["base/count"][0] >= self.fail_at)
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
@pytest.mark.parametrize("substeps", [2, 3])
def test_late_failure_full_rollback_and_retry_fresh(failure, substeps):
    case = custom(
        position=[[0.0, (2 * substeps - 1) / 8, 0.0]],
        base_transform=lambda base: FaultProgram(base, failure, substeps - 1),
    )
    scene, state, inputs, program, settings, context = case
    settings = replace(settings, substeps=substeps)
    observed = program.base
    initial = fingerprint((state, inputs))
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        result = runtime.step(state, inputs, 0.25 * substeps)
        if failure:
            assert observed.triggered, "injection must really execute"
            assert not result.report.accepted
            assert result.report.completed_substeps == substeps - 1
            assert fingerprint(result.state) == fingerprint(state)
            assert result.substep_states == ()
            observed.failure = None
            result = runtime.step(state, inputs, 0.25 * substeps)
        assert result.report.accepted, result.report.reason
        assert result.state.history["base/count"][0] == substeps
        assert [s.history["base/count"][0] for s in result.substep_states] == list(
            range(1, substeps + 1)
        )
        assert observed.close_calls == 0
    assert observed.close_calls == 1
    assert fingerprint((state, inputs)) == initial
    assert all(ws.counts["close"] == 1 for ws in observed.workspaces)
    fresh = custom(position=[[0.0, (2 * substeps - 1) / 8, 0.0]], base_transform=FaultProgram)
    with ph.SceneStepper(
        fresh[0], fresh[3], replace(fresh[4], substeps=substeps), fresh[5], fresh[3].registry
    ) as runtime:
        expected = runtime.step(fresh[1], fresh[2], 0.25 * substeps)
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
        d = sub.diagnostics["dynamic_robot_friction_contact"]
        assert d["terminal_replays"] >= 1
        assert d["terminal_evidence"]["applications"] == 1
    assert all(
        ws.counts["predict"] == 1 and ws.counts["terminal"] == 1 for ws in case[3].base.workspaces
    )


@pytest.mark.parametrize("kind", [None, "terminal", "trace", "close"])
def test_best_terminal_candidate_never_masks_later_real_native_failure(monkeypatch, kind):
    import phymeo.solvers._dynamic_friction_workspace as module

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

    evaluate = module._DynamicFrictionWorkspace._evaluate

    def arm_after_best(self, terminal=False):
        if terminal and self.vbest is not None:
            self.program.base.armed = True
        return evaluate(self, terminal)

    monkeypatch.setattr(module._DynamicFrictionWorkspace, "_evaluate", arm_after_best)
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


@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("domain", ["robot", "nodes"])
def test_terminal_replay_must_match_complete_held_pre(corrupt, domain, monkeypatch):
    import numpy as np

    from phymeo.solvers._dynamic_friction_workspace import _DynamicFrictionWorkspace as W

    class Program(ObservedProgram):
        perturb = False

        def create_workspace(self, *args):
            w = Workspace(self, self.base.create_workspace(*args), len(self.workspaces) + 1)
            self.workspaces.append(w)
            return w

    class Workspace(ObservedWorkspace):
        def result(self):
            state = self.native.result()
            if self.owner.perturb:
                if domain == "robot":
                    robot = state.domains[domain]
                    q = robot.q.copy()
                    q[0] = np.nextafter(q[0], np.inf)
                    changed = ph.RobotState(q, robot.v)
                else:
                    node = state.domains[domain]
                    x = node.position.copy()
                    x[0, 0] = np.nextafter(x[0, 0], np.inf)
                    changed = ph.ParticleState(x, node.velocity)
                return ph.SceneState({**state.domains, domain: changed}, state.history)
            return state

    original = W._evaluate

    def evaluate(self, terminal=False):
        self.program.base.perturb = terminal and corrupt
        return original(self, terminal)

    monkeypatch.setattr(W, "_evaluate", evaluate)
    c = custom(mu=0.5, chain=True, base_transform=Program)
    before = fingerprint((c[1], c[2]))
    with ph.SceneStepper(c[0], c[3], c[4], c[5], c[3].registry) as runtime:
        result = runtime.step(c[1], c[2], 0.25)
    assert result.report.accepted == (not corrupt), result.report.reason
    if corrupt:
        assert ("pre-state" if domain == "robot" else "unrecorded material") in result.report.reason
        assert fingerprint(result.state) == fingerprint(c[1])
    assert fingerprint((c[1], c[2])) == before


def test_capability_name_without_live_operation_is_not_accepted():
    c = custom(base_transform=lambda base: ObservedProgram(base, missing=True))
    with ph.SceneStepper(c[0], c[3], c[4], c[5], c[3].registry) as runtime:
        result = runtime.step(c[1], c[2], 0.25)
    assert not result.report.accepted and "callable" in result.report.reason
    assert all(w.counts["close"] == 1 for w in c[3].base.workspaces)


@pytest.mark.parametrize("substeps", [2, 3])
def test_late_physical_budget_rejection_rolls_back_and_retry_matches(substeps, monkeypatch):
    from phymeo.solvers._dynamic_friction_workspace import _DynamicFrictionWorkspace as W

    require = W._velocity_metrics
    injected = []
    enabled = [True]

    def limited(self):
        metrics = require(self)
        if self.initial.history["base/count"][0] == substeps - 1 and enabled[0]:
            injected.append(True)
            return ph.StageMetrics(
                {**metrics.residuals, "frozen_model_energy": 2 * self.settings.tolerance}
            )
        return metrics

    monkeypatch.setattr(W, "_velocity_metrics", limited)
    c = custom(position=[[0.0, (2 * substeps - 1) / 8, 0.0]], base_transform=FaultProgram)
    settings = replace(c[4], substeps=substeps)
    with ph.SceneStepper(c[0], c[3], settings, c[5], c[3].registry) as runtime:
        failed = runtime.step(c[1], c[2], 0.25 * substeps)
        assert injected and not failed.report.accepted
        assert failed.report.completed_substeps == substeps - 1
        assert fingerprint(failed.state) == fingerprint(c[1])
        enabled[0] = False
        retry = runtime.step(c[1], c[2], 0.25 * substeps)
        assert retry.report.accepted, retry.report.reason
    fresh = custom(position=[[0.0, (2 * substeps - 1) / 8, 0.0]], base_transform=FaultProgram)
    with ph.SceneStepper(
        fresh[0], fresh[3], replace(fresh[4], substeps=substeps), fresh[5], fresh[3].registry
    ) as runtime:
        expected = runtime.step(fresh[1], fresh[2], 0.25 * substeps)
    assert expected.report.accepted, expected.report.reason
    assert fingerprint(retry.state) == fingerprint(expected.state)


@pytest.mark.parametrize("fault", [None, "new_candidate_budget", "best_budget", "ledger_exception"])
def test_after_best_physical_budget_requires_audited_fallback(monkeypatch, fault):
    """A finite trial rejection may restore best; invalid/unknown best may not commit."""
    import numpy as np

    from phymeo.solvers._dynamic_friction_workspace import _DynamicFrictionWorkspace as W
    from phymeo.solvers._material_contact_balance import MaterialContactBalance

    enabled = [True]
    current = [None]
    observed = dict(workspace=None, candidate=None, best=None, injected=0, best_audits=[])
    velocity_metrics = W._velocity_metrics
    physical_metrics = MaterialContactBalance.require_metrics

    def ledger(self, *args, **kwargs):
        workspace = current[0]
        if (
            enabled[0]
            and fault == "ledger_exception"
            and workspace is observed["workspace"]
            and workspace.trial is observed["candidate"]
        ):
            observed["injected"] += 1
            raise ph.SolverFailure("injected physical ledger exception after admitted best")
        return physical_metrics(self, *args, **kwargs)

    def audited(workspace):
        # Locate the first new native trial after an admitted best by identity,
        # never by a hard-coded replay number, score or trajectory coefficient.
        if (
            enabled[0]
            and observed["candidate"] is None
            and workspace.vbest is not None
            and workspace.trial is not workspace.vbest[3]
        ):
            observed.update(workspace=workspace, candidate=workspace.trial, best=workspace.vbest[3])
        current[0] = workspace
        try:
            metrics = velocity_metrics(workspace)
        finally:
            current[0] = None
        if workspace is observed["workspace"] and workspace.trial is observed["best"]:
            observed["best_audits"].append((metrics, fingerprint(workspace.physical_report)))
        new_trial = workspace.trial is observed["candidate"]
        restored_best = workspace.trial is observed["best"]
        if (
            enabled[0]
            and workspace is observed["workspace"]
            and (
                (fault == "new_candidate_budget" and new_trial)
                or (fault == "best_budget" and (new_trial or restored_best))
            )
        ):
            observed["injected"] += 1
            return ph.StageMetrics(
                {**metrics.residuals, "frozen_model_energy": 2 * workspace.settings.tolerance},
                geometric_error=metrics.geometric_error,
                penetration=metrics.penetration,
            )
        return metrics

    monkeypatch.setattr(MaterialContactBalance, "require_metrics", ledger)
    monkeypatch.setattr(W, "_velocity_metrics", audited)

    def make_case():
        case = custom(
            mu=0.5,
            position=[[0.0, 1 / 8, 0.0]],
            velocity=[[1.0, -1.0, 0.0]],
            base_transform=ObservedProgram,
        )
        case[3].relaxation = 0.5
        return case

    case = make_case()
    scene, state, inputs, program, settings, context = case
    initial = fingerprint((state, inputs))
    rejected = fault in ("best_budget", "ledger_exception")
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        result = runtime.step(state, inputs, 0.25)
        assert observed["candidate"] is not None, "must reach a new native trial after best"
        assert observed["candidate"] is not observed["best"]
        assert result.report.accepted == (not rejected), result.report.reason
        assert fingerprint((state, inputs)) == initial
        if fault is None:
            assert observed["injected"] == 0
        elif fault == "new_candidate_budget":
            assert observed["injected"] == 1, "only the new candidate is polluted once"
            assert observed["workspace"].trial is observed["best"]
            assert fingerprint(result.state.domains) == fingerprint(
                observed["best"].result.state.domains
            )
            assert observed["best_audits"], "restored best must undergo physical re-audit"
            assert all(
                metrics.max_residual <= settings.tolerance for metrics, _ in observed["best_audits"]
            )
            diagnostic = result.report.substeps[0].diagnostics["dynamic_robot_friction_contact"]
            assert observed["best_audits"][-1][1] == fingerprint(diagnostic["physical_budgets"])
        else:
            assert observed["injected"] > 0
            if fault == "best_budget":
                assert observed["best_audits"], "must also reject the re-audited saved best"
            else:
                assert "physical ledger exception" in result.report.reason
            assert fingerprint(result.state) == fingerprint(state)
            assert result.report.completed_substeps == 0 and result.substep_states == ()
            enabled[0] = False
            result = runtime.step(state, inputs, 0.25)
            assert result.report.accepted, result.report.reason

        # Independent free-impact fractions and real final audits still hold
        # for normal completion, accepted best fallback and fault-free retry.
        np.testing.assert_allclose(
            result.state.domains["nodes"].position,
            [[1 / 4, -1 / 24, 0.0]],
            atol=settings.tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            result.state.domains["nodes"].velocity,
            [[2 / 3, -1 / 3, 0.0]],
            atol=settings.tolerance,
            rtol=0,
        )
        diagnostic = result.report.substeps[0].diagnostics["dynamic_robot_friction_contact"]
        physical = diagnostic["physical_budgets"]
        assert (
            physical["frozen_model_energy"]["positive_gain_upper_j"]
            <= program.budgets.max_energy_gain_j
        )
        assert (
            physical["material_angular_defect"]["l1_sum_nms"]
            <= program.budgets.max_material_angular_defect_nms
        )
        assert result.report.max_residual <= settings.tolerance
        assert result.state.history["base/count"][0] == 1
    assert fingerprint((state, inputs)) == initial
    assert all(workspace.counts["close"] == 1 for workspace in program.base.workspaces)

    if rejected:
        fresh = make_case()
        with ph.SceneStepper(fresh[0], fresh[3], fresh[4], fresh[5], fresh[3].registry) as runtime:
            expected = runtime.step(fresh[1], fresh[2], 0.25)
        assert expected.report.accepted, expected.report.reason
        assert fingerprint(result.state) == fingerprint(expected.state)
