"""Saved-file reader cases, not simulated trajectories or CUDA evidence."""

import hashlib
import json
from copy import deepcopy

import pytest

from experiments.area_pressure.pusht_feedback_comparison import compare


def _state(tick, x=0.5):
    q = [0.0, 0.0, x, 0.0, 0.125, 1.0, 0.0, 0.0, 0.0]
    return dict(
        local_tick=tick,
        time=tick * 0.001,
        q=q,
        v=[0.0] * 8,
        object_position_world_m=q[2:5],
        object_quaternion_wxyz=q[5:9],
    )


def _saved_case():
    channels = dict(ctrl=[0.0, 0.0], qfrc_applied=[0.0] * 8, xfrc_applied=[[0.0] * 6])
    initial, endpoint = _state(0), _state(1)
    pd = dict(
        qpos_addresses=[0, 1],
        dof_addresses=[0, 1],
        actuator_ids=[0, 1],
        position=[0.0, 0.0],
        velocity=[0.0, 0.0],
        target=[0.0, 0.0],
        target_velocity=[0.0, 0.0],
        effort=[0.0, 0.0],
    )
    command = dict(tick=0, output_tick=1, h=0.001, **channels)
    rows = {}
    for lane, world in (("open", 5), ("feedback", 6)):
        controller = dict(
            input_tick=0,
            local_tick=0,
            input_time=0.0,
            measurement_q=initial["q"],
            measurement_v=initial["v"],
        )
        measurement = dict(world=world, input_tick=0, output_tick=1, time_s=0.0)
        if lane == "feedback":
            controller.update(
                target_q=[0.0, 0.0],
                target_v=[0.0, 0.0],
                pd=pd,
                generated_channels=channels,
                route={},
                stage="settle",
                transport=dict(
                    kind="original-env-fp32-atanh-cuda-tanh",
                    device="cuda:0",
                    operator="bias + scale * torch.tanh(raw)",
                    scale=20.0,
                    bias=0.0,
                    direct_pd_ctrl=[0.0, 0.0],
                    raw_action=[0.0, 0.0],
                    transported_ctrl=[0.0, 0.0],
                    direct_to_transported_delta=[0.0, 0.0],
                ),
            )
        rows[lane] = dict(
            input_state=initial,
            output_state=endpoint,
            submitted_control=channels,
            stored_control_matches_submitted=True,
            controller=controller,
            measurement=measurement,
            feedback_accepted=dict(measurement=measurement),
            driver_state_after=dict(tick=1),
            goal_frame=dict(
                phase="accepted-entry",
                world=world,
                tick=0,
                time_s=0.0,
                object_position=initial["object_position_world_m"],
                object_wxyz=initial["object_quaternion_wxyz"],
            ),
            goal_result=dict(
                success=False, history=dict(tick=0, time_s=0.0), metrics=[["position_error_m", 0.1]]
            ),
            area=dict(normal_load_n=2.0, zero_normal_load=False, samples=1, active_samples=1),
            native=dict(object_support_normal_n=1.5),
            geometry=dict(strict_disjoint_certified=False),
        )
    record = dict(
        schema="phymeo-pusht-public-feedback-step-v1",
        index=0,
        original_input_tick=0,
        original_output_tick=1,
        original_control=command,
        lanes=rows,
    )
    summary = dict(
        schema="phymeo-pusht-public-feedback-v1",
        status="complete",
        start_tick=0,
        requested_steps=1,
        accepted_pair_steps=1,
        recorded_pair_steps=1,
        worlds=dict(open=5, feedback=6),
        initial_states={lane: initial for lane in rows},
        final_states={lane: endpoint for lane in rows},
        controller_history_bootstrap=dict(
            start_tick=0, prefix_steps=0, future_archived_states_used=False
        ),
        initial_command_comparison=dict(
            same_input_q_fp32_bytes=True,
            same_input_v_fp32_bytes=True,
            ctrl_fp32_bytes_equal=True,
            original_ctrl=[0.0, 0.0],
            regenerated_ctrl=[0.0, 0.0],
            original_direct_effort=[0.0, 0.0],
            regenerated_direct_effort=[0.0, 0.0],
            original_raw_action=[0.0, 0.0],
            regenerated_raw_action=[0.0, 0.0],
            original_pd=pd,
            regenerated_pd=pd,
        ),
    )
    return deepcopy(summary), [deepcopy(record)]


def _write(tmp_path, summary, records):
    journal = tmp_path / "run.steps.jsonl"
    payload = "".join(json.dumps(row, allow_nan=False) + "\n" for row in records).encode()
    journal.write_bytes(payload)
    summary["journal"] = dict(
        file=journal.name, records=len(records), sha256=hashlib.sha256(payload).hexdigest()
    )
    path = tmp_path / "run.json"
    path.write_text(json.dumps(summary, allow_nan=False), encoding="utf-8")
    return path


def _callback_failure(summary):
    summary.update(
        status="partial_failure",
        requested_steps=2,
        accepted_pair_steps=2,
        unpublished_accepted_pairs=1,
        failure_stage="paired_receipt_and_controller",
        failed_original_input_tick=1,
        final_states={lane: _state(2, 0.75) for lane in ("open", "feedback")},
    )


def test_complete_recorded_pair_keeps_original_validation(tmp_path):
    summary, records = _saved_case()
    result = compare(_write(tmp_path, summary, records))
    assert result["verified_pair_steps"] == result["accepted_pair_steps"] == 1
    assert result["unrecorded_accepted_pair"] is None
    assert result["complete_requested_window"] is True


def test_post_accept_failure_preserves_verified_prefix_without_auditing_tail(tmp_path):
    summary, records = _saved_case()
    _callback_failure(summary)
    result = compare(_write(tmp_path, summary, records))
    assert result["accepted_pair_steps"] == 2 and result["verified_pair_steps"] == 1
    assert result["recorded_pair_steps"] == 1 and result["last_input_tick"] == 0
    assert result["complete_requested_window"] is False
    assert result["verification_scope"] == "recorded-prefix-only"
    tail = result["unrecorded_accepted_pair"]
    assert (tail["input_tick"], tail["output_tick"]) == (1, 2)
    assert tail["transition_verified"] is False
    assert tail["control_force_goal_evidence_verified"] is False
    assert tail["summary_endpoint_identity_checked"] is True
    assert tail["reported_output_states"] == summary["final_states"]
    assert result["verified_final_states"] == {lane: _state(1) for lane in ("open", "feedback")}
    for lane in ("open", "feedback"):
        assert result["lanes"][lane]["normal_impulse_ns"] == 0.002
        assert result["lanes"][lane]["object_displacement_world_m"] == [0.0, 0.0, 0.0]
        assert result["lanes"][lane]["last_goal_input_tick"] == 0


@pytest.mark.parametrize(
    "fault",
    (
        "status",
        "stage",
        "gap",
        "missing-tail-count",
        "failure-tick",
        "final-tick",
        "clock",
        "cross-lane-clock",
        "nonfinite",
        "coordinate-identity",
        "prefix",
        "requested",
    ),
)
def test_unrecorded_tail_requires_exact_failure_identity(tmp_path, fault):
    summary, records = _saved_case()
    _callback_failure(summary)
    if fault == "status":
        summary["status"] = "complete"
    elif fault == "stage":
        summary["failure_stage"] = "paired_scene_step"
    elif fault == "gap":
        summary["accepted_pair_steps"] = 3
        summary["requested_steps"] = 3
    elif fault == "missing-tail-count":
        del summary["unpublished_accepted_pairs"]
    elif fault == "failure-tick":
        summary["failed_original_input_tick"] = 2
    elif fault == "final-tick":
        summary["final_states"]["feedback"]["local_tick"] = 1
    elif fault == "clock":
        summary["final_states"]["feedback"]["time"] += 0.001
    elif fault == "cross-lane-clock":
        summary["final_states"]["feedback"]["time"] += 1e-7
    elif fault == "nonfinite":
        summary["final_states"]["feedback"]["q"][0] = "NaN"
    elif fault == "coordinate-identity":
        summary["final_states"]["feedback"]["q"][2] = 0.5
    elif fault == "prefix":
        records[0]["lanes"]["open"]["input_state"] = _state(0, 0.25)
    else:
        summary["requested_steps"] = 1
    with pytest.raises(ValueError):
        compare(_write(tmp_path, summary, records))


def test_equal_counts_still_reject_an_unmatched_summary_endpoint(tmp_path):
    summary, records = _saved_case()
    summary["final_states"]["feedback"] = _state(1, 0.75)
    with pytest.raises(ValueError, match="summary final states"):
        compare(_write(tmp_path, summary, records))


@pytest.mark.parametrize("reason", ("success", "no_progress", "segment_budget", "step_budget"))
def test_recorded_terminal_stop_is_not_a_failed_or_complete_requested_window(tmp_path, reason):
    summary, records = _saved_case()
    summary.update(
        status="controller_stopped",
        controller_status=reason,
        requested_steps=10,
        stopped_after_original_output_tick=1,
    )
    records[0]["lanes"]["feedback"]["driver_state_after"]["status"] = reason
    records[0]["lanes"]["feedback"]["goal_result"]["success"] = reason == "success"
    result = compare(_write(tmp_path, summary, records))
    assert result["verified_pair_steps"] == result["accepted_pair_steps"] == 1
    assert result["controller_status"] == reason
    assert result["stopped_after_original_output_tick"] == 1
    assert result["complete_requested_window"] is False
    assert result["unrecorded_accepted_pair"] is None
    assert result["lanes"]["feedback"]["goal_achieved_in_saved_window"] == (reason == "success")


@pytest.mark.parametrize("fault", ("reason", "stop-tick", "errors", "stage", "goal", "driver"))
def test_malformed_terminal_stop_does_not_hide_a_failure(tmp_path, fault):
    summary, records = _saved_case()
    summary.update(
        status="controller_stopped",
        controller_status="step_budget",
        requested_steps=10,
        stopped_after_original_output_tick=1,
    )
    records[0]["lanes"]["feedback"]["driver_state_after"]["status"] = "step_budget"
    if fault == "reason":
        summary["controller_status"] = "unsupported_tilt"
    elif fault == "stop-tick":
        summary["stopped_after_original_output_tick"] = 2
    elif fault == "errors":
        summary["errors"] = [{"type": "RuntimeError", "message": "cleanup failed"}]
    elif fault == "stage":
        summary["failure_stage"] = "scene_cleanup"
    elif fault == "goal":
        summary["controller_status"] = "success"
        records[0]["lanes"]["feedback"]["driver_state_after"]["status"] = "success"
    else:
        records[0]["lanes"]["feedback"]["driver_state_after"]["status"] = "continue"
    with pytest.raises(ValueError):
        compare(_write(tmp_path, summary, records))


def test_cleanup_failure_preserves_terminal_reason_without_claiming_normal_stop(tmp_path):
    summary, records = _saved_case()
    summary.update(
        status="partial_failure",
        controller_status="step_budget",
        requested_steps=10,
        failure_stage="scene_cleanup",
        unpublished_accepted_pairs=0,
        errors=[{"type": "RuntimeError", "message": "cleanup failed"}],
    )
    result = compare(_write(tmp_path, summary, records))
    assert result["source"]["producer_status"] == "partial_failure"
    assert result["controller_status"] == "step_budget"
    assert result["unrecorded_accepted_pair"] is None


def test_initial_empty_window_is_distinct_from_observed_unloading():
    from experiments.area_pressure.pusht_feedback_comparison import _load_record, _load_stats

    stats = _load_stats()

    def record(tick, load, samples, strict):
        _load_record(
            stats,
            dict(
                area=dict(
                    normal_load_n=load,
                    zero_normal_load=load == 0,
                    samples=samples,
                    active_samples=int(load > 0),
                ),
                native=dict(object_support_normal_n=1.5),
                geometry=dict(strict_disjoint_certified=strict),
            ),
            tick,
            0.001,
        )

    record(10, 0.0, 0, True)
    record(11, 0.0, 0, True)
    assert stats["first_empty_clip_input_tick"] == stats["first_strict_disjoint_input_tick"] == 10
    assert stats["first_empty_clip_after_load_input_tick"] is None
    assert stats["first_strict_disjoint_after_load_input_tick"] is None
    assert stats["final_unloaded_start_input_tick"] is None
    assert stats["final_empty_clip_start_input_tick"] == 10
    assert stats["final_empty_clip_after_load_start_input_tick"] is None
    record(12, 2.0, 1, False)
    record(13, 0.0, 1, False)  # Unloaded, but a positive-area clip remains.
    assert stats["final_unloaded_start_input_tick"] == 13
    assert stats["final_empty_clip_start_input_tick"] is None
    record(14, 0.0, 0, False)
    record(15, 0.0, 0, True)
    assert stats["first_empty_clip_after_load_input_tick"] == 14
    assert stats["first_strict_disjoint_after_load_input_tick"] == 15
    assert stats["final_empty_clip_after_load_start_input_tick"] == 14
    assert stats["final_strict_disjoint_after_load_start_input_tick"] == 15
    record(16, 1.0, 1, False)  # Recontact resets only the final suffix metrics.
    assert stats["final_unloaded_start_input_tick"] is None
    assert stats["final_empty_clip_after_load_start_input_tick"] is None
    assert stats["final_strict_disjoint_after_load_start_input_tick"] is None
    record(17, 0.0, 0, True)
    record(18, 0.0, 0, True)
    assert stats["first_empty_clip_after_load_input_tick"] == 14
    assert stats["final_unloaded_start_input_tick"] == 17
    assert stats["final_empty_clip_start_input_tick"] == 17
    assert stats["final_empty_clip_after_load_start_input_tick"] == 17
    assert stats["final_strict_disjoint_after_load_start_input_tick"] == 17


def test_first_pair_callback_failure_has_no_verified_control_force_or_goal(tmp_path):
    summary, _records = _saved_case()
    summary.update(
        status="partial_failure",
        accepted_pair_steps=1,
        recorded_pair_steps=0,
        failure_stage="paired_receipt_and_controller",
        failed_original_input_tick=0,
        unpublished_accepted_pairs=1,
    )
    result = compare(_write(tmp_path, summary, []))
    assert result["status"] == "no-recorded-pairs"
    assert result["accepted_pair_steps"] == 1 and result["verified_pair_steps"] == 0
    assert result["last_input_tick"] is None
    assert result["commands_bound_to_current_lane"] is False
    assert result["initial_command_comparison_verified"] is False
    assert result["unrecorded_accepted_pair"]["transition_verified"] is False
    for lane in ("open", "feedback"):
        assert result["lanes"][lane]["goal_achieved_in_saved_window"] is None
        assert result["lanes"][lane]["last_goal_input_tick"] is None
        assert result["lanes"][lane]["normal_load_change_n"] is None
