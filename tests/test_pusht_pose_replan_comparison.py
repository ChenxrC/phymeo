"""Readout arithmetic and publication guards; synthetic rows are not physics evidence."""

import math
from copy import deepcopy

import pytest

from experiments.area_pressure.pusht_pose_replan_comparison import (
    retained_state,
    summarize_motion,
    validate_counts,
)


def state(tick, x=0.5, yaw=0.5):
    q = [0.0, 0.0, x, 0.0, 0.125, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    return dict(
        local_tick=tick,
        time=tick * 0.001,
        q=q,
        v=[0.0] * 8,
        object_position_world_m=q[2:5],
        object_quaternion_wxyz=q[5:9],
    )


GOAL = dict(
    target_xy=[1.0, 0.0], target_yaw=0.2, position_tolerance_m=0.015, yaw_tolerance_rad=0.12
)


def accepted(before, after, phase, *, load=2.0, moment=0.3):
    return dict(
        input_state=before,
        output_state=after,
        phase=phase,
        selected=True,
        restored=False,
        controller=dict(stage="push" if phase == "regular" else phase),
        area=dict(normal_load_n=load, total_force_on_pusher_world_n=[-load, 0.0, 0.0]),
        feedback_accepted=dict(
            measurement=dict(
                group_force_world=[load, 0.0, 0.0], group_moment_com_world=[0.0, 0.0, moment]
            )
        ),
        goal_result=dict(success=False),
        prediction=dict(predicted_position_xy=[1.0, 0.0], predicted_yaw=0.2),
    )


def test_streaming_motion_counts_only_consumed_steps():
    initial, next_state = state(0), state(1)
    rows = iter([accepted(initial, next_state, "regular")])
    result = summarize_motion(initial, GOAL, rows)
    assert result["verified_steps"] == len(result["trajectory"]) == 1
    assert result["verified_final_state"] == next_state


def test_zero_step_rejection_reports_initial_pose_and_no_measured_load_or_goal():
    initial = state(0)
    result = summarize_motion(initial, GOAL, [])
    assert result["verified_steps"] == 0
    assert result["verified_final_state"] == initial
    assert result["initial"] == result["final"]
    assert result["peak_normal_load_n"] is None and result["normal_impulse_ns"] == 0.0
    assert result["goal_success_observed"] is False


def test_withdrawal_and_new_push_contributions_are_actual_not_geometric_predictions():
    a, b, c = state(0), state(1, 0.49, 0.52), state(2, 0.53, 0.42)
    result = summarize_motion(a, GOAL, [accepted(a, b, "withdraw"), accepted(b, c, "regular")])
    assert result["object_displacement_world_m"][0] == pytest.approx(0.03)
    assert result["phases"]["withdraw"]["position_error_change_m"] == pytest.approx(0.01)
    assert result["phases"]["regular"]["position_error_change_m"] == pytest.approx(-0.04)
    assert result["phases"]["withdraw"]["yaw_error_change_rad"] == pytest.approx(0.02)
    assert result["phases"]["regular"]["yaw_error_change_rad"] == pytest.approx(-0.10)
    assert result["final"]["position_error_m"] == pytest.approx(0.47)
    assert result["final"]["yaw_error_rad"] == pytest.approx(0.22)
    assert result["normal_impulse_ns"] == pytest.approx(0.004)
    assert result["object_moment_impulse_world_nms"][2] == pytest.approx(0.0006)
    assert result["goal_success_observed"] is False


def test_rejected_actual_trial_never_advances_retained_pose_or_contributes_load():
    before = state(0)
    trial = accepted(before, state(1, 0.9, 0.2), "withdraw", load=100.0)
    trial.update(selected=False, restored=True, rollback_verification=dict(passed=True))
    assert retained_state(trial, before) == before
    with pytest.raises(ValueError):
        summarize_motion(before, GOAL, [trial])
    trial["rollback_verification"]["passed"] = False
    with pytest.raises(ValueError):
        retained_state(trial, before)


def test_missing_state_continuity_is_not_a_valid_accepted_trajectory():
    before = state(0)
    trial = accepted(deepcopy(before), state(1), "regular")
    trial["input_state"]["q"][0] = 0.125
    with pytest.raises(ValueError):
        retained_state(trial, before)


@pytest.mark.parametrize(
    "change",
    (
        {"physical_attempts": 5},
        {"accepted_steps": 2},
        {"rejected_trials": 0},
        {"recorded_steps": 2},
    ),
)
def test_budget_and_trial_counters_cannot_hide_rejections_or_missing_callbacks(change):
    summary = dict(
        step_budget=4,
        physical_attempts=2,
        physical_trial_steps=2,
        accepted_steps=1,
        recorded_steps=1,
        rejected_trials=1,
        status="step_budget",
    )
    assert validate_counts(summary, candidate_count=2, selected_count=1, callback_count=1) is None
    summary.update(change)
    with pytest.raises(ValueError):
        validate_counts(summary, candidate_count=2, selected_count=1, callback_count=1)


def test_failed_physical_trial_restored_counts_separately_from_selected_prefix():
    summary = dict(
        step_budget=4,
        physical_attempts=2,
        physical_trial_steps=2,
        accepted_steps=1,
        recorded_steps=1,
        rejected_trials=0,
        status="partial_failure",
    )
    validate_counts(
        summary,
        candidate_count=1,
        selected_count=1,
        callback_count=1,
        failed_attempts=1,
        failed_physical=1,
    )
    with pytest.raises(ValueError):
        validate_counts(
            summary,
            candidate_count=1,
            selected_count=1,
            callback_count=1,
            failed_attempts=1,
            failed_physical=0,
        )


def test_callback_failure_preserves_an_explicit_unrecorded_accepted_tail():
    summary = dict(
        step_budget=4,
        physical_attempts=2,
        physical_trial_steps=2,
        accepted_steps=2,
        recorded_steps=1,
        rejected_trials=0,
        status="partial_failure",
    )
    validate_counts(summary, candidate_count=2, selected_count=2, callback_count=1)
    with pytest.raises(ValueError):
        validate_counts(summary, candidate_count=2, selected_count=2, callback_count=0)


def test_ordinary_replan_requires_each_non_support_pair_to_be_unloaded_and_separated():
    from experiments.area_pressure.pusht_pose_replan_comparison import verify_separation

    endpoint = state(1)
    pair = dict(
        key=dict(kind="area-material", identity=["tool", "object"]),
        normal_load_n=0.0,
        output_depth_m=0.0,
        output_contact=False,
        output_strictly_separated=True,
    )
    evidence = dict(
        output_tick=1,
        output_time=endpoint["time"],
        output_q=endpoint["q"],
        output_v=endpoint["v"],
        complete_pair_catalog=True,
        pairs=[pair],
    )
    verify_separation(evidence, endpoint, 1)
    pair["normal_load_n"] = 0.001
    with pytest.raises(ValueError):
        verify_separation(evidence, endpoint, 1)
    pair["normal_load_n"] = 0.0
    pair["output_strictly_separated"] = False
    with pytest.raises(ValueError):
        verify_separation(evidence, endpoint, 1)


def _split_case_files(tmp_path):
    import json

    from experiments.area_pressure import pusht_pose_replan_comparison as reader

    common = dict(
        schema="phymeo-pusht-pose-replan-case-v1",
        goal=GOAL,
        step_budget=4000,
        control_variant="load-compensation-half",
        selection_policy="joint-pose-descent",
        provenance=dict(
            source_report_sha256="source",
            source_checkpoint_sha256="capsule",
            scene_asset_sha256="material",
        ),
    )
    paths = {}
    for key, mode in (("A", "terminal-checkpoint"), ("B", "independent-pose-reset")):
        report = deepcopy(common)
        report.update(
            key=key,
            initialization_mode=mode,
            status="partial_failure",
            failure_stage="qualify_release",
            errors=["saved failure"],
        )
        report["provenance"]["initialization_mode"] = mode
        path = tmp_path / (key + ".json")
        path.write_text(json.dumps(report), encoding="utf-8")
        paths[key] = path
    return reader, paths


def test_split_case_reports_keep_independent_failures_and_hash_bindings(tmp_path, monkeypatch):
    reader, paths = _split_case_files(tmp_path)
    calls = []

    def audited(path):
        calls.append(path)
        report = reader._read(path)
        return dict(
            actual=dict(trajectory=[], peak_normal_load_n=None, verified_steps=0),
            termination={k: report[k] for k in ("status", "failure_stage", "errors")},
        )

    # Only the aggregation seam is replaced: real case validation remains the
    # existing compare_case; this fixture is not dynamic or source-audit evidence.
    monkeypatch.setattr(reader, "compare_case", audited)
    result = reader.compare_cases(paths["A"], paths["B"])
    assert calls == [paths["A"], paths["B"]]
    assert result["source"]["mode"] == "independent-case-reports"
    for key in ("A", "B"):
        assert result["source"]["cases"][key]["sha256"] == reader._digest(paths[key])
        assert result["cases"][key]["termination"]["status"] == "partial_failure"
        assert result["cases"][key]["termination"]["errors"] == ["saved failure"]
        assert result["cases"][key]["actual"]["peak_normal_load_n"] is None


@pytest.mark.parametrize(
    "change",
    ["source", "capsule", "material", "goal", "identity", "mode", "policy", "control", "budget"],
)
def test_split_case_reports_reject_incompatible_sources_or_initialization(
    tmp_path, monkeypatch, change
):
    import json

    reader, paths = _split_case_files(tmp_path)
    b = reader._read(paths["B"])
    if change in ("source", "capsule", "material"):
        field = dict(
            source="source_report_sha256",
            capsule="source_checkpoint_sha256",
            material="scene_asset_sha256",
        )[change]
        b["provenance"][field] = "different"
    elif change == "goal":
        b["goal"]["target_yaw"] += 0.1
    elif change == "identity":
        b["key"] = "A"
    elif change == "mode":
        b["initialization_mode"] = b["provenance"]["initialization_mode"] = "terminal-checkpoint"
    elif change == "policy":
        b["selection_policy"] = "pose-cost"
    elif change == "control":
        b["control_variant"] = "position"
    else:
        b["step_budget"] = 2000
    paths["B"].write_text(json.dumps(b), encoding="utf-8")
    monkeypatch.setattr(reader, "compare_case", lambda path: dict(actual=dict(trajectory=[])))
    with pytest.raises(ValueError):
        reader.compare_cases(paths["A"], paths["B"])


def test_split_case_cli_uses_both_reports_without_original_bundle(tmp_path, monkeypatch):
    import json
    import sys

    reader, paths = _split_case_files(tmp_path)
    output = tmp_path / "comparison.json"
    calls = []

    def combine(a, b):
        calls.append((a, b))
        return dict(status="test-only")

    monkeypatch.setattr(reader, "compare_cases", combine)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "comparison",
            "--case-a",
            str(paths["A"]),
            "--case-b",
            str(paths["B"]),
            "--output",
            str(output),
        ],
    )
    reader.main()
    assert calls == [(paths["A"], paths["B"])]
    assert json.loads(output.read_text("utf-8")) == dict(status="test-only")


@pytest.mark.parametrize(
    "inputs",
    [
        ["--case-a", "a.json"],
        ["--case-b", "b.json"],
        ["--input", "bundle.json", "--case-a", "a.json", "--case-b", "b.json"],
    ],
)
def test_split_case_cli_rejects_missing_or_mixed_sources(tmp_path, monkeypatch, inputs):
    import sys

    from experiments.area_pressure import pusht_pose_replan_comparison as reader

    monkeypatch.setattr(
        sys, "argv", ["comparison", *inputs, "--output", str(tmp_path / "out.json")]
    )
    with pytest.raises(SystemExit) as error:
        reader.main()
    assert error.value.code == 2


def test_original_bundle_cli_remains_supported(tmp_path, monkeypatch):
    import json
    import sys

    from experiments.area_pressure import pusht_pose_replan_comparison as reader

    source, output = tmp_path / "bundle.json", tmp_path / "audit.json"
    calls = []

    def compare_original(path):
        calls.append(path)
        return dict(status="original-cli-test-only")

    monkeypatch.setattr(reader, "compare", compare_original)
    monkeypatch.setattr(
        sys, "argv", ["comparison", "--input", str(source), "--output", str(output)]
    )
    reader.main()
    assert calls == [source]
    assert json.loads(output.read_text("utf-8")) == dict(status="original-cli-test-only")
