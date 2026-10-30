"""Late failures in the finite-gripper assembly preserve whole-frame transactions."""

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest
from test_terminal_impulse import ObservedProgram, ObservedWorkspace

import phymeo as ph
from phymeo.core.scene import fingerprint
from phymeo.experiments.parallel_jaw_cloth import build_gripper_case
from phymeo.runtime.program import own_diagnostics
from phymeo.validation.thick_contact_mapping import ThickContactMappingLedger


class GripperFaultProgram(ObservedProgram):
    def __init__(self, base, failure, fail_at):
        super().__init__(base)
        self.failure, self.fail_at = failure, fail_at
        self.triggered = self.active_late = False

    def trip(self, kind):
        if self.active_late and self.failure == kind and not self.triggered:
            self.triggered = True
            raise ph.SolverFailure("injected finite-gripper " + kind)

    def create_workspace(self, scene, state, *args):
        self.active_late = state.history["base/count"][0] >= self.fail_at
        workspace = GripperFaultWorkspace(
            self, self.base.create_workspace(scene, state, *args), len(self.workspaces) + 1
        )
        self.workspaces.append(workspace)
        return workspace


class GripperFaultWorkspace(ObservedWorkspace):
    def predict(self):
        super().predict()
        self.owner.trip("native")

    def apply_point_velocity_impulses(self, requests):
        self.counts["terminal"] += 1
        self.native.apply_point_velocity_impulses(requests)
        self.owner.trip("terminal")

    def diagnostics(self):
        result = self.native.diagnostics()
        if self.owner.active_late and self.owner.failure == "path" and not self.owner.triggered:
            result["participants"]["nodes"]["material_path"] = dict(
                result["participants"]["nodes"]["material_path"], schema="corrupt"
            )
            self.owner.triggered = True
        return result

    def close(self):
        self.counts["close"] += 1
        self.native.close()
        self.owner.trip("close")


def fault_case(case, mode, failure, fail_at, substeps):
    original = build_gripper_case(case, mode=mode, h=1 / 8)
    old = original.program
    observed = GripperFaultProgram(old.base, failure, fail_at)
    program = ph.DynamicRobotThickContactProgram(
        original.scene,
        observed,
        old.surfaces,
        old.colliders,
        friction_by_collider=old.friction_by_collider,
        materials=old.materials,
        budgets=old.budgets,
        registry=old.registry,
        max_anchors_per_pair=old.max_anchors_per_pair,
        max_manifold_rebuilds=old.max_manifold_rebuilds,
        max_ccd_intervals=old.max_ccd_intervals,
        max_ccd_depth=old.max_ccd_depth,
    )
    # The unused outer binding has no workspace or resources. The new binding
    # owns the same base and closes it once; no physics state is changed.
    initial = own_diagnostics(
        ph.SceneState(
            original.initial.domains,
            {f.name: np.zeros(f.shape, f.dtype) for f in program.history_fields},
        )
    )
    return replace(
        original,
        program=program,
        initial=initial,
        settings=replace(original.settings, substeps=substeps),
    )


@lru_cache(maxsize=8)
def fresh_frame(case, mode, substeps):
    problem = fault_case(case, mode, None, substeps - 1, substeps)
    with ph.SceneStepper(
        problem.scene, problem.program, problem.settings, problem.context, problem.program.registry
    ) as runtime:
        result = runtime.step(problem.initial, problem.inputs, problem.h * substeps)
    assert result.report.accepted, result.report.reason
    assert all(w.counts["close"] == 1 for w in problem.program.base.workspaces)
    return own_diagnostics(result)


def require_failed_retry_fresh(monkeypatch, case, mode, failure, fail_at, substeps):
    problem = fault_case(case, mode, failure, fail_at, substeps)
    owner = problem.program.base
    before = fingerprint((problem.initial, problem.inputs))
    original_require = ThickContactMappingLedger.require

    def require(ledger, *args, **kwargs):
        value = original_require(ledger, *args, **kwargs)
        owner.trip("mapping")
        return value

    import phymeo.solvers._thick_contact_workspace as consumer

    original_sweep = consumer.triangle_prism_patch_sweep

    def sweep(*args, **kwargs):
        result = original_sweep(*args, **kwargs)
        if owner.active_late and owner.failure == "CCD" and not owner.triggered:
            owner.triggered = True
            return replace(result, input_fingerprint="forged-other-path")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(ThickContactMappingLedger, "require", require)
        patch.setattr(consumer, "triangle_prism_patch_sweep", sweep)
        with ph.SceneStepper(
            problem.scene,
            problem.program,
            problem.settings,
            problem.context,
            problem.program.registry,
        ) as runtime:
            rejected = runtime.step(problem.initial, problem.inputs, problem.h * substeps)
            assert owner.triggered
            assert not rejected.report.accepted
            assert rejected.report.completed_substeps == fail_at
            assert fingerprint(rejected.state) == fingerprint(problem.initial)
            assert rejected.substep_states == ()
            assert fingerprint((problem.initial, problem.inputs)) == before
            owner.failure = None
            retried = runtime.step(problem.initial, problem.inputs, problem.h * substeps)
            assert retried.report.accepted, retried.report.reason
    assert fingerprint((problem.initial, problem.inputs)) == before
    assert owner.close_calls == 1
    assert all(w.counts["close"] == 1 for w in owner.workspaces)
    fresh = fresh_frame(case, mode, substeps)
    assert fingerprint((retried.state, retried.substep_states)) == fingerprint(
        (fresh.state, fresh.substep_states)
    )
    for index, sub in enumerate(retried.report.substeps, 1):
        data = sub.diagnostics["dynamic_robot_thick_contact"]
        terminal = data["terminal_ledger"]
        assert terminal["applications"] == 1
        assert fingerprint(terminal["pre_impulse_state"]) == fingerprint(data["held_state"])
        assert retried.substep_states[index - 1].history["base/count"][0] == index
        if mode != "numpy":
            assert sub.diagnostics["execution"]["completed_launches"] > 0
            assert terminal["ledger"]["compute_precision"] == "float32"
    if case == "capture":
        impact = retried.report.substeps[2].diagnostics["dynamic_robot_thick_contact"]
        assert np.sum(impact["terminal_tangent"][:, 2]) > 0


@pytest.mark.parametrize("failure", ["native", "path", "terminal", "mapping", "CCD", "close"])
def test_late_hold_failure_and_whole_frame_retry(monkeypatch, failure):
    require_failed_retry_fresh(monkeypatch, "hold", "numpy", failure, 2, 3)


@pytest.mark.parametrize("failure", ["terminal", "close"])
def test_nonzero_capture_terminal_rollback_and_retry(monkeypatch, failure):
    require_failed_retry_fresh(monkeypatch, "capture", "numpy", failure, 2, 4)


def test_partial_gripper_construction_closes_created_participant(monkeypatch):
    problem = build_gripper_case("hold", mode="numpy", h=1 / 8)
    base = problem.program.base
    closed = []
    create = base.participants["robot"].create_workspace

    def observed(*args):
        workspace = create(*args)
        close = workspace.close

        def counted():
            closed.append(True)
            close()

        workspace.close = counted
        return workspace

    def fail(*args):
        raise ph.SolverFailure("injected finite-gripper node construction")

    monkeypatch.setattr(base.participants["robot"], "create_workspace", observed)
    monkeypatch.setattr(base.participants["nodes"], "create_workspace", fail)
    before = fingerprint((problem.initial, problem.inputs))
    with ph.SceneStepper(
        problem.scene, problem.program, problem.settings, problem.context, problem.program.registry
    ) as runtime:
        result = runtime.step(problem.initial, problem.inputs, problem.h)
    assert not result.report.accepted
    assert closed == [True]
    assert fingerprint((result.state, problem.inputs)) == before
