"""Late failures, exact selected-pre binding and atomic frame rollback."""

from dataclasses import replace

import numpy as np
import pytest
from test_cloth_self_physics import configured, fixture, save_evidence

from phymeo.core.capabilities import SolverFailure
from phymeo.core.scene import SceneState, fingerprint
from phymeo.runtime.program import SceneSubstepReport
from phymeo.solvers._cloth_self_contact_replay import observational
from phymeo.solvers._cloth_self_contact_workspace import ClothSelfContactWorkspace


@pytest.mark.parametrize("case,h", [("A", 1 / 64), ("B", 1 / 16)])
@pytest.mark.parametrize("failure", ["diagnostics", "close"])
def test_late_failure_preserves_original_and_retry_equals_fresh(
    case, h, failure, monkeypatch, tmp_path
):
    data = fixture(case)
    runtime, program, state, inputs = configured(data, h)
    original = fingerprint((state, inputs))
    operation = getattr(ClothSelfContactWorkspace, failure)

    def fail(workspace):
        operation(workspace)
        raise SolverFailure("injected late " + failure)

    monkeypatch.setattr(ClothSelfContactWorkspace, failure, fail)
    failed = runtime.step(state, inputs, h)
    save_evidence(
        tmp_path / "failed.json",
        result=failed,
        attempts=program.last_attempt_receipts,
        inputs=inputs,
        label=failure,
    )
    assert not failed.report.accepted
    assert fingerprint(failed.state) == fingerprint(state)
    assert fingerprint((state, inputs)) == original
    assert any(item["kind"] == "accepted-candidate" for item in program.last_attempt_receipts)
    monkeypatch.setattr(ClothSelfContactWorkspace, failure, operation)
    retry = runtime.step(state, inputs, h)
    save_evidence(
        tmp_path / "retry.json",
        result=retry,
        attempts=program.last_attempt_receipts,
        inputs=inputs,
        label="retry",
    )
    runtime.close()
    fresh_runtime, fresh_program, fresh_state, fresh_inputs = configured(fixture(case), h)
    try:
        fresh = fresh_runtime.step(fresh_state, fresh_inputs, h)
        save_evidence(
            tmp_path / "fresh.json",
            result=fresh,
            attempts=fresh_program.last_attempt_receipts,
            inputs=fresh_inputs,
            label="fresh",
        )
        assert retry.report.accepted and fresh.report.accepted
        assert fingerprint(retry.state) == fingerprint(fresh.state)
    finally:
        fresh_runtime.close()


def test_later_frame_substep_rolls_back_earlier_state_and_history(monkeypatch, tmp_path):
    runtime, program, state, inputs = configured(
        fixture("A", material=True), 1 / 64, outer_substeps=2
    )
    original = fingerprint((state, inputs))
    close, calls = ClothSelfContactWorkspace.close, []

    def late(workspace):
        close(workspace)
        calls.append(1)
        if len(calls) == 2:
            raise SolverFailure("late second-substep close")

    monkeypatch.setattr(ClothSelfContactWorkspace, "close", late)
    try:
        result = runtime.step(state, inputs, 1 / 32)
        save_evidence(
            tmp_path / "frame-failure.json",
            result=result,
            attempts=program.last_attempt_receipts,
            inputs=inputs,
            label="frame",
        )
        assert not result.report.accepted and result.report.completed_substeps == 1
        assert fingerprint(result.state) == fingerprint(state)
        assert fingerprint((state, inputs)) == original
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "key",
    [
        "pre_impulse_state",
        "pre_impulse_diagnostics",
        "pre_position_iterations",
        "pre_velocity_metrics",
    ],
)
def test_selected_pre_binding_rejects_each_forged_evidence_field(key, monkeypatch, tmp_path):
    import phymeo.solvers._cloth_self_contact_workspace as module

    original_check = module.require_selected_pre

    def forged(actual, selected):
        terminal = dict(actual["terminal"])
        if key == "pre_impulse_state":
            before = terminal[key]
            changed = before.domains["cloth"].copy()
            changed.position[0, 0] += 1 / 8
            terminal[key] = SceneState({"cloth": changed}, before.history)
        elif key == "pre_impulse_diagnostics":
            terminal[key] = {**terminal[key], "timings": {"must_not_ignore": 1.0}}
        elif key == "pre_position_iterations":
            terminal[key] += 1
        else:
            from phymeo.core.scene import StageMetrics

            terminal[key] = StageMetrics({"forged": 1.0})
        original_check({**actual, "terminal": terminal}, selected)

    monkeypatch.setattr(module, "require_selected_pre", forged)
    runtime, program, state, inputs = configured(fixture("B"), 1 / 16)
    try:
        result = runtime.step(state, inputs, 1 / 16)
        save_evidence(
            tmp_path / "selected-forgery.json",
            result=result,
            attempts=program.last_attempt_receipts,
            inputs=inputs,
            label=key,
        )
        assert not result.report.accepted
        assert "selected pre" in result.report.reason
        assert fingerprint(result.state) == fingerprint(state)
    finally:
        runtime.close()


def test_r5_normalizes_only_precise_report_timings():
    before = SceneSubstepReport(False, 0, 1.0, timings={"setup": 0.1})
    after = replace(before, timings={"setup": 0.2})
    assert fingerprint(observational(before)) == fingerprint(observational(after))
    assert fingerprint(observational({"timings": {"setup": 0.1}})) != fingerprint(
        observational({"timings": {"setup": 0.2}})
    )


@pytest.mark.parametrize("fault", ["missing_partition", "stale_path", "material_path"])
def test_unknown_geometry_and_material_evidence_cannot_publish(fault, monkeypatch, tmp_path):
    if fault == "material_path":
        import phymeo.solvers._cloth_self_contact_replay as module

        def reject(*args, **kwargs):
            raise SolverFailure("injected material path mismatch")

        monkeypatch.setattr(module.MaterialPathLedger, "require", reject)
    else:
        original = ClothSelfContactWorkspace._geometry

        def forged(workspace, state):
            result = original(workspace, state)
            if fault == "missing_partition":
                return ()
            items = [dict(item) for item in result]
            items[0]["path_a"] = items[0]["path_a"].copy()
            items[0]["path_a"][1, 0, 0] += 1 / 8
            return tuple(items)

        monkeypatch.setattr(ClothSelfContactWorkspace, "_geometry", forged)
    runtime, program, state, inputs = configured(fixture("A"), 1 / 64)
    try:
        result = runtime.step(state, inputs, 1 / 64)
        save_evidence(
            tmp_path / "proof-forgery.json",
            result=result,
            attempts=program.last_attempt_receipts,
            inputs=inputs,
            label=fault,
        )
        assert not result.report.accepted
        assert fingerprint(result.state) == fingerprint(state)
    finally:
        runtime.close()


def test_failed_attempt_receipts_are_owned_without_physical_history_mutation(monkeypatch, tmp_path):
    runtime, program, state, inputs = configured(fixture("A"), 1 / 64)
    original = fingerprint((state, inputs))
    try:
        result = runtime.step(state, inputs, 1 / 64)
        assert result.report.accepted
        observed = program.last_attempt_receipts
        with pytest.raises(ValueError):
            observed[0]["trial"].result.state.domains["cloth"].position[0, 0] = 2.0
        assert fingerprint((state, inputs)) == original
    finally:
        runtime.close()


@pytest.mark.parametrize("fault", ["thickness", "winding", "reported_velocity", "capacity_phase"])
def test_topology_and_actual_phase_receipts_are_independently_bound(fault, monkeypatch, tmp_path):
    if fault in ("thickness", "winding"):
        from phymeo.geometry.prism_prism_manifold import PrismPairIdentity
        from phymeo.geometry.prism_prism_sweep import prism_prism_sweep

        original = ClothSelfContactWorkspace._geometry

        def forged(workspace, state):
            items = [dict(item) for item in original(workspace, state)]
            item = items[0]
            if fault == "thickness":
                item["thickness"] = item["thickness"] / 2
            else:
                identity = item["identity"]
                item["identity"] = PrismPairIdentity(
                    identity.domain, identity.face_ids, identity.node_ids[:, ::-1].copy()
                )
                item["path_a"], item["path_b"] = (
                    item["path_a"][:, ::-1].copy(),
                    item["path_b"][:, ::-1].copy(),
                )
            item["certificate"] = prism_prism_sweep(
                item["path_a"],
                item["path_b"],
                identity=item["identity"],
                half_thickness_m=item["thickness"],
                h=workspace.h,
                penetration_budget_m=0.0,
                proof_limits=workspace.program.limits.geometry_limits,
            )
            return tuple(items)

        monkeypatch.setattr(ClothSelfContactWorkspace, "_geometry", forged)
    else:
        original = ClothSelfContactWorkspace._data

        def forged(workspace, *args, **kwargs):
            data = original(workspace, *args, **kwargs)
            if kwargs["phase"] == "terminal":
                if fault == "reported_velocity":
                    data["velocity"] = np.zeros_like(data["velocity"])
                else:
                    data["initial_closed"] = ~data["initial_closed"]
            return data

        monkeypatch.setattr(ClothSelfContactWorkspace, "_data", forged)
    case, h = ("B", 1 / 16) if fault in ("reported_velocity", "capacity_phase") else ("A", 1 / 64)
    runtime, program, state, inputs = configured(fixture(case), h)
    try:
        result = runtime.step(state, inputs, h)
        save_evidence(
            tmp_path / "binding-forgery.json",
            result=result,
            attempts=program.last_attempt_receipts,
            inputs=inputs,
            label=fault,
        )
        assert not result.report.accepted
        assert fingerprint(result.state) == fingerprint(state)
    finally:
        runtime.close()
