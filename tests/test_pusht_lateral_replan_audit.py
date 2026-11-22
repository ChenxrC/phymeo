"""Saved-report fixtures only: no rollout or physical validation is synthesized."""

from copy import deepcopy

import pytest


def terminal_source():
    from test_pusht_lateral_resume_audit import source_binding_records

    summary, parent, audited, metadata, seed = source_binding_records()
    parent.update(schema="phymeo-pusht-lateral-resume-v1", status="route_complete", world=6)
    parent["active_route"]["end_tick"] = summary["start_tick"]
    parent["controller_final"].update(partial_route=None, completed_routes=1)
    metadata["active_route"] = deepcopy(parent["active_route"])
    metadata["controller"] = deepcopy(parent["controller_final"])
    metadata["world"] = 6
    audited["schema"] = "phymeo-pusht-lateral-resume-audit-v1"
    summary.update(
        schema="phymeo-pusht-lateral-replan-v1",
        world=6,
        previous_route=deepcopy(parent["active_route"]),
    )
    summary.pop("active_route")
    summary["controller_initial"] = deepcopy(parent["controller_final"])
    return summary, parent, audited, metadata, seed


def test_replan_binds_completed_source_capsule_history_and_previous_actual_force():
    from experiments.area_pressure.pusht_lateral_replan_audit import bind_terminal_source

    bind_terminal_source(*terminal_source())


@pytest.mark.parametrize(
    "fault",
    ["clock", "state", "history", "force", "unfinished-route", "prior-route", "accepted-tail"],
)
def test_replan_rejects_disconnected_or_incomplete_terminal_source(fault):
    from experiments.area_pressure.pusht_lateral_replan_audit import bind_terminal_source

    records = terminal_source()
    summary, parent, _, metadata, _ = records
    if fault == "clock":
        summary["local_tick_origin"] = 0
    elif fault == "state":
        summary["initial_state"]["v"][0] = 0.125
    elif fault == "history":
        summary["goal_history_initial"]["tick"] -= 1
    elif fault == "force":
        summary["checkpoint_restore"]["previous_tool_force_world_n"][0] += 0.125
    elif fault == "unfinished-route":
        parent["active_route"]["end_tick"] += 1
        metadata["active_route"]["end_tick"] += 1
        summary["previous_route"]["end_tick"] += 1
    elif fault == "prior-route":
        summary["previous_route"]["sha256"] = "foreign-route"
    else:
        parent["unrecorded_accepted_step"] = {"output_state": deepcopy(parent["final_state"])}
    with pytest.raises(ValueError):
        bind_terminal_source(*records)


def pose(x, y, yaw):
    import math

    q = [0.0, 0.0, x, y, 0.015625, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    return dict(q=q, v=[0.0] * 8, object_position_world_m=q[2:5], object_quaternion_wxyz=q[5:9])


def test_planning_prediction_without_regular_steps_has_no_measured_improvement():
    from experiments.area_pressure.pusht_lateral_replan_audit import compare_prediction

    initial = pose(0.0, 0.0, 0.0)
    goal = dict(
        target_xy=[1.0, 0.0], target_yaw=0.2, position_tolerance_m=0.015, yaw_tolerance_rad=0.12
    )
    prediction = dict(
        position_error_before_m=1.0,
        position_error_after_m=0.9,
        yaw_error_before_rad=0.2,
        yaw_error_after_rad=0.1,
    )
    result = compare_prediction(prediction, initial, initial, goal, regular_steps=0)
    assert result["predicted_position_improvement_m"] == pytest.approx(0.1)
    assert result["measured_regular_steps"] == 0
    assert result["measured_position_improvement_m"] is None
    assert result["measured_absolute_yaw_improvement_rad"] is None
    assert result["actual_both_improved"] is None


def test_saved_prediction_cannot_replace_opposite_actual_pose_change():
    from experiments.area_pressure.pusht_lateral_replan_audit import compare_prediction

    goal = dict(
        target_xy=[1.0, 0.0], target_yaw=0.2, position_tolerance_m=0.015, yaw_tolerance_rad=0.12
    )
    prediction = dict(
        position_error_before_m=1.0,
        position_error_after_m=0.9,
        yaw_error_before_rad=0.2,
        yaw_error_after_rad=0.1,
    )
    result = compare_prediction(
        prediction, pose(0.0, 0.0, 0.0), pose(-0.1, 0.0, -0.05), goal, regular_steps=1
    )
    assert result["predicted_position_improvement_m"] == pytest.approx(0.1)
    assert result["measured_position_improvement_m"] == pytest.approx(-0.1)
    assert result["measured_absolute_yaw_improvement_rad"] == pytest.approx(-0.05)
    assert result["actual_both_improved"] is False
    assert "task_success" not in result


def rejected_release_events():
    from test_pusht_lateral_resume_audit import resumed_events

    _, summary = resumed_events(failed=True)
    for key in ("failure_stage", "errors", "failed_candidate"):
        summary.pop(key, None)
    summary.update(
        schema="phymeo-pusht-lateral-replan-v1",
        status="release_qualification_rejected",
        physical_attempts=0,
        physical_trial_steps=0,
        accepted_steps=0,
        recorded_steps=0,
        rejected_trials=0,
        planner_calls=0,
        release_planner_calls=1,
    )
    summary["entry_probe"] = dict(
        purpose="entry-separation-gate",
        selected=False,
        planner_requested=False,
        input_state=deepcopy(summary["initial_state"]),
        tick=summary["start_tick"],
        status="entry_separation_rejected",
        errors=[{"message": "real gate rejection"}],
        seed={"fixture": "ledger-only"},
    )
    summary["seed_evidence"] = deepcopy(summary["entry_probe"]["seed"])
    summary["release_reference"] = dict(
        file="rejected.json", sha256="route-sha", nodes=[], route_status="sampled_collision"
    )
    initial = deepcopy(summary)
    initial.update(status="running", release_planner_calls=0)

    def event(kind, payload):
        return dict(
            schema="phymeo-pusht-lateral-replan-event-v1",
            event=kind,
            trial_index=None,
            payload=deepcopy(payload),
        )

    sequence = [
        event("initialization", initial),
        event("planner_probe", summary["entry_probe"]),
        event(
            "release_reference",
            {"reference": summary["release_reference"], "seed": summary["seed_evidence"]},
        ),
        event("final", {k: v for k, v in summary.items() if k != "journal"}),
    ]
    return sequence, summary


def test_zero_physics_release_qualification_rejection_keeps_original_events():
    from experiments.area_pressure.pusht_lateral_replan_audit import iter_bound_events

    events, summary = rejected_release_events()
    assert list(iter_bound_events(events, summary)) == events
    assert summary["accepted_steps"] == summary["physical_attempts"] == 0


@pytest.mark.parametrize(
    "fault",
    [
        "false-completion",
        "invent-step",
        "probe-state",
        "duplicate-probe",
        "release-source",
        "after-final",
    ],
)
def test_rejected_release_cannot_be_relabelled_as_motion_or_a_different_source(fault):
    from experiments.area_pressure.pusht_lateral_replan_audit import iter_bound_events

    events, summary = rejected_release_events()
    if fault == "false-completion":
        summary["status"] = "route_complete"
        events[-1]["payload"]["status"] = "route_complete"
    elif fault == "invent-step":
        summary["physical_attempts"] = 1
        events[-1]["payload"]["physical_attempts"] = 1
    elif fault == "probe-state":
        events[1]["payload"]["input_state"]["time"] += 0.001
        summary["entry_probe"] = deepcopy(events[1]["payload"])
    elif fault == "duplicate-probe":
        events.insert(2, deepcopy(events[1]))
    elif fault == "release-source":
        events[2]["payload"]["reference"]["sha256"] = "foreign"
    else:
        events.append(deepcopy(events[1]))
    with pytest.raises(ValueError):
        list(iter_bound_events(events, summary))


def release_records():
    """Small caller-fed ledger evidence, explicitly not a dynamics fixture."""
    from dataclasses import replace

    from test_manipulation_release_monitor import AREA, TABLE, next_step, seed

    from experiments.area_pressure.pusht_lateral_replan_audit import _plain
    from phymeo.control.manipulation_release_monitor import ReleaseMonitor, ReleasePairKey

    support = ReleasePairKey("native-geometry", ("support", "object-box"), TABLE.body_ids)
    first = seed()
    first = replace(
        first, pairs=tuple(replace(p, key=support) if p.key == TABLE else p for p in first.pairs)
    )
    second = next_step(first)
    second = replace(
        second, pairs=tuple(replace(p, key=support) if p.key == TABLE else p for p in second.pairs)
    )
    monitor = ReleaseMonitor(support_keys=(support,), max_steps=320)
    monitor.start(first)

    def row(step):
        area = next(p for p in step.pairs if p.key == AREA)
        return dict(
            trial_index=0,
            trial_mode="smooth",
            selected=True,
            restored=False,
            rollback_verification=None,
            original_input_tick=step.input_tick,
            original_output_tick=step.output_tick,
            input_state=dict(q=list(step.input_q), v=list(step.input_v), time=step.input_time),
            output_state=dict(q=list(step.output_q), v=list(step.output_v), time=step.output_time),
            area=dict(
                samples=1,
                normal_load_n=area.normal_load_n,
                total_force_on_pusher_world_n=list(area.force_on_a_world),
                source_ids=[["cell", "face"]],
                sample_ids=[["triangle", 0]],
                raw_normal_force_n=[area.normal_load_n],
                raw_force_on_tool_world_n=[list(area.force_on_a_world)],
                raw_force_on_object_world_n=[list(area.force_on_b_world)],
            ),
            measurement=dict(group_force_world=list(area.force_on_b_world)),
            evidence=_plain(step),
            native=dict(
                detected_contacts=1,
                supports=[
                    dict(
                        geometry_names=list(p.key.identity),
                        normal_n=p.normal_load_n,
                        force_on_supported_world_n=list(p.force_on_b_world),
                    )
                    for p in step.pairs
                    if p.key == support
                ],
            ),
        )

    actual = row(second)
    actual["release_decision"] = _plain(monitor.assess(replace(second, publication="trial")))
    actual["committed_release_decision"] = _plain(monitor.observe(second))
    return _plain(first), row(first), actual, second, row


def test_release_reader_rechecks_trial_then_committed_per_pair_ledger():
    from experiments.area_pressure.pusht_lateral_replan_audit import ReleaseLedger

    seed, previous, actual, _, _ = release_records()
    ledger = ReleaseLedger(seed, previous, 7)
    ledger.candidate(actual)
    assert ledger.monitor.history == ()
    ledger.accepted(actual)
    assert len(ledger.monitor.history) == 1
    assert ledger.monitor.status == "continue"
    assert not ledger.monitor.history[0].regular_control_authorized


@pytest.mark.parametrize(
    "fault",
    [
        "deepening",
        "new-contact",
        "load-increase",
        "inward",
        "foreign-pair",
        "false-release",
        "force-phase",
    ],
)
def test_release_reader_rejects_forged_safe_or_complete_decisions(fault):
    from dataclasses import replace

    from test_manipulation_release_monitor import AREA, OTHER, changed

    from experiments.area_pressure.pusht_lateral_replan_audit import ReleaseLedger

    seed, previous, actual, step, make_row = release_records()
    if fault == "deepening":
        step = changed(step, AREA, output_depth_m=0.003)
    elif fault == "new-contact":
        step = changed(
            step, OTHER, output_contact=True, output_depth_m=0.0001, output_strictly_separated=False
        )
    elif fault == "load-increase":
        step = changed(
            step,
            AREA,
            normal_load_n=11.0,
            force_on_a_world=(0.0, -11.0, 0.0),
            force_on_b_world=(0.0, 11.0, 0.0),
        )
    elif fault == "inward":
        step = changed(step, AREA, input_normal_speed_m_s=-0.01)
    elif fault == "foreign-pair":
        step = replace(
            step,
            pairs=tuple(
                replace(p, key=replace(OTHER, identity=("foreign-link", "object-box")))
                if p.key == OTHER
                else p
                for p in step.pairs
            ),
        )
    elif fault == "force-phase":
        step = replace(step, force_phase="output-geometry")
    original_decision = deepcopy(actual["release_decision"])
    actual = make_row(step)
    actual["release_decision"] = original_decision
    if fault == "false-release":
        actual["release_decision"].update(status="released", release_complete=True)
    ledger = ReleaseLedger(seed, previous, 7)
    with pytest.raises(ValueError):
        ledger.candidate(actual)
    assert ledger.monitor.history == ()


def test_loaded_face_comes_from_actual_positive_point_load_and_bilateral_force():
    from experiments.area_pressure.pusht_lateral_replan_audit import point_loads

    _, _, actual, _, _ = release_records()
    area = actual["area"]
    result = point_loads(area)
    assert result["faces"]["face"]["normal_load_n"] == 9.0
    assert result["force_on_object_world_n"] == [0.0, 9.0, 0.0]
    area["raw_force_on_object_world_n"][0][1] = 8.0
    with pytest.raises(ValueError):
        point_loads(area)
