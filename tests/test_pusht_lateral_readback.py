"""Keep attempts and published motion distinct when reading lateral-route evidence."""

from copy import deepcopy

import pytest

from experiments.area_pressure.pusht_lateral_readback import accepted_pairs


def events():
    def event(kind, payload, index=None):
        return dict(
            schema="phymeo-pusht-lateral-route-event-v1",
            event=kind,
            trial_index=index,
            payload=payload,
        )

    measurement = {"group_force_world": [0, 1, 0]}
    return [
        event("initialization", {}),
        event("planner", {}),
        event("candidate", dict(selected=True, restored=False, measurement=measurement), 0),
        event(
            "accepted_callback", dict(feedback_accepted=dict(measurement=deepcopy(measurement))), 0
        ),
        event("final", {}),
    ]


def test_force_and_motion_must_come_from_one_accepted_callback():
    rows = list(accepted_pairs(events()))
    assert len(rows) == 1 and rows[0]["measurement"] == rows[0]["feedback_accepted"]["measurement"]


@pytest.mark.parametrize(
    "fault", ["wrench", "missing", "duplicate", "index", "unselected", "after-final"]
)
def test_readback_detects_misaligned_publications(fault):
    sequence = events()
    if fault == "wrench":
        sequence[3]["payload"]["feedback_accepted"]["measurement"]["group_force_world"][1] *= -1
    if fault == "missing":
        del sequence[3]
    if fault == "duplicate":
        sequence.insert(4, deepcopy(sequence[3]))
    if fault == "index":
        sequence[3]["trial_index"] += 1
    if fault == "unselected":
        sequence[2]["payload"].update(selected=False, restored=True)
    if fault == "after-final":
        sequence.append(deepcopy(sequence[2]))
    with pytest.raises(ValueError):
        list(accepted_pairs(sequence))


def bound_events(*, rejected=False, failed=False, callback_failed=False):
    """Synthetic writer-shaped records; no claim that these are physical evidence."""
    before = dict(
        local_tick=0,
        time=4.45,
        q=[0.0, 0.0, 0.5, 0.0, 0.015625, 1.0, 0.0, 0.0, 0.0],
        v=[0.0] * 8,
        object_position_world_m=[0.5, 0.0, 0.015625],
        object_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
    )
    after = deepcopy(before)
    after.update(local_tick=1, time=4.451)
    control = dict(
        tick=4450,
        output_tick=4451,
        ctrl=[0.0, 0.0],
        qfrc_applied=[0.0] * 8,
        xfrc_applied=[[0.0] * 6 for _ in range(5)],
    )
    plan = dict(
        tick=4450,
        source_q=before["q"],
        source_v=before["v"],
        route_file="route.json.gz",
        route_sha256="route-sha",
        status="sampled_reference_ready",
    )
    selected = not rejected and not failed
    summary = dict(
        schema="phymeo-pusht-lateral-route-v1",
        world=6,
        start_tick=4450,
        local_tick_origin=0,
        initial_state=deepcopy(before),
        final_state=deepcopy(after if selected else before),
        status="partial_failure"
        if failed or callback_failed
        else "initial_force_check_rejected"
        if rejected
        else "step_budget",
        goal={},
        side_coverage="all-lateral",
        material_admission="declared-finite-faces-v1",
        selection_policy="joint-pose-balanced",
        initialization_mode="new-material-episode-from-audited-qv",
        provenance={"material_identity": "new-material"},
        initialization_proof={"previous_receipt_inherited": False},
        ordinary_route=deepcopy(plan),
        step_budget=1,
        physical_attempts=1,
        physical_trial_steps=0 if failed else 1,
        accepted_steps=int(selected),
        recorded_steps=int(selected and not callback_failed),
        rejected_trials=int(rejected),
        journal={"file": "run.trials.jsonl"},
    )

    def event(kind, payload, index=None):
        return dict(
            schema="phymeo-pusht-lateral-route-event-v1",
            event=kind,
            trial_index=index,
            payload=deepcopy(payload),
        )

    initialization = deepcopy(summary)
    for key in ("ordinary_route",):
        initialization.pop(key)
    initialization.update(
        status="running",
        final_state=deepcopy(before),
        physical_attempts=0,
        physical_trial_steps=0,
        accepted_steps=0,
        recorded_steps=0,
        rejected_trials=0,
    )
    sequence = [event("initialization", initialization), event("planner", plan)]
    rollback = dict(passed=True, worlds=[6], mismatches=[])
    measurement = {"group_force_world": [0.0, 1.0, 0.0]}
    candidate = dict(
        index=0,
        trial_index=0,
        input_state=before,
        output_state=after,
        original_input_tick=4450,
        original_output_tick=4451,
        control=control,
        selected=selected,
        restored=rejected,
        measurement=measurement,
        rollback_verification=rollback if rejected else None,
        assessment={"safe": selected},
    )
    if not failed:
        sequence.append(event("candidate", candidate, 0))
        if selected and not callback_failed:
            sequence.append(
                event("accepted_callback", {"feedback_accepted": {"measurement": measurement}}, 0)
            )
    if failed or callback_failed:
        summary.update(
            failure_stage="accepted_controller_and_goal" if callback_failed else "scene_step",
            errors=[dict(type="RuntimeError", message="registered failure")],
        )
        if callback_failed:
            summary["unrecorded_accepted_step"] = dict(
                input_state=before, output_state=after, control=control
            )
        else:
            summary["failed_candidate"] = dict(
                physics_accepted_before_restore=False,
                restored=True,
                rollback_verification=rollback,
                input_states={"6": before},
                candidate_output_states=None,
                control=control,
            )
        failure = {
            key: summary.get(key)
            for key in ("failure_stage", "errors", "failed_candidate", "unrecorded_accepted_step")
        }
        sequence.append(event("failure", failure, 0))
    sequence.append(
        event("final", {key: value for key, value in summary.items() if key != "journal"})
    )
    return sequence, summary


def test_bound_events_keep_rejected_trial_and_callback_failure_out_of_motion():
    for kwargs, count in (
        ({}, 1),
        ({"rejected": True}, 0),
        ({"failed": True}, 0),
        ({"callback_failed": True}, 0),
    ):
        sequence, summary = bound_events(**kwargs)
        assert len(list(accepted_pairs(sequence, summary=summary))) == count


@pytest.mark.parametrize(
    "fault",
    [
        "initial-state",
        "initial-goal",
        "planner",
        "final-state",
        "final-status",
        "counter",
        "callback-overwrite",
        "missing-initial",
        "missing-planner",
    ],
)
def test_bound_events_reject_disconnected_initial_plan_and_final(fault):
    sequence, summary = bound_events()
    if fault == "initial-state":
        sequence[0]["payload"]["initial_state"]["time"] += 0.001
    elif fault == "initial-goal":
        sequence[0]["payload"]["goal"] = {"different": True}
    elif fault == "planner":
        sequence[1]["payload"]["route_sha256"] = "other-route"
    elif fault == "final-state":
        sequence[-1]["payload"]["final_state"]["time"] += 0.001
    elif fault == "final-status":
        sequence[-1]["payload"]["status"] = "route_complete"
    elif fault == "counter":
        summary["physical_attempts"] += 1
        sequence[-1]["payload"]["physical_attempts"] += 1
    elif fault == "missing-initial":
        del sequence[0]
    elif fault == "missing-planner":
        del sequence[1]
    else:
        sequence[3]["payload"]["output_state"] = deepcopy(sequence[2]["payload"]["input_state"])
    with pytest.raises(ValueError):
        list(accepted_pairs(sequence, summary=summary))


@pytest.mark.parametrize(
    "fault",
    [
        "input",
        "world",
        "mismatch",
        "failure-input",
        "failure-event",
        "failure-final",
        "failed-output",
    ],
)
def test_bound_events_reject_wrong_rollback_prestate_and_failure(fault):
    sequence, summary = bound_events(
        failed=fault.startswith("fail"), rejected=not fault.startswith("fail")
    )
    if fault in ("input", "world", "mismatch"):
        row = sequence[2]["payload"]
        if fault == "input":
            row["input_state"]["time"] += 0.001
        elif fault == "world":
            row["rollback_verification"]["worlds"] = [7]
        else:
            row["rollback_verification"]["mismatches"] = ["q"]
    else:
        packet = sequence[2]["payload"]
        if fault == "failure-input":
            packet["failed_candidate"]["input_states"]["6"]["time"] += 0.001
            summary["failed_candidate"] = deepcopy(packet["failed_candidate"])
        elif fault == "failure-event":
            packet["failure_stage"] = "unrelated-phase"
        elif fault == "failure-final":
            summary["final_state"]["time"] += 0.001
        else:
            packet["failed_candidate"]["candidate_output_states"] = {
                "6": deepcopy(summary["initial_state"])
            }
            summary["failed_candidate"] = deepcopy(packet["failed_candidate"])
        sequence[-1]["payload"] = {
            key: deepcopy(value) for key, value in summary.items() if key != "journal"
        }
    with pytest.raises(ValueError):
        list(accepted_pairs(sequence, summary=summary))


def initial_binding():
    from numpy import array, float32, linalg

    _, summary = bound_events()
    source = deepcopy(summary["initial_state"])
    source["local_tick"] = 2750
    source["q"][5:9] = [
        0.9984166026115417,
        -1.801822264724251e-7,
        1.3197508863527219e-8,
        -0.05625271424651146,
    ]
    source["object_quaternion_wxyz"] = source["q"][5:9]
    norm = float(linalg.norm(source["q"][5:9]))
    q = array(source["q"])
    q[5:9] /= norm
    q = q.astype(float32).tolist()
    summary["initial_state"] = dict(
        deepcopy(source), local_tick=0, q=q, object_quaternion_wxyz=q[5:9]
    )
    summary["initialization_proof"] = dict(
        previous_receipt_inherited=False,
        new_local_tick=0,
        source_local_tick=2750,
        initial_time=source["time"],
        source_q=deepcopy(source["q"]),
        published_q=deepcopy(q),
        initialization_policy="new-episode-public-free-quaternion-normalization",
        quaternion_address=5,
        source_quaternion_norm=norm,
        v_fp32_bytes_equal=True,
        time_float64_bytes_equal=True,
        nonrotation_q_fp32_bytes_equal=True,
        normalized_quaternion_fp32_bytes_equal=True,
        snapshot_q_lossless_fp32=True,
        snapshot_v_lossless_fp32=True,
    )
    return summary, source


def test_readback_initial_binding_allows_only_the_declared_public_normalization():
    from experiments.area_pressure.pusht_lateral_readback import _bind_initial

    summary, source = initial_binding()
    _bind_initial(summary, source)


@pytest.mark.parametrize("fault", ["time", "velocity", "translation", "proof-source", "receipt"])
def test_readback_initial_binding_rejects_coherent_report_changes_without_source(fault):
    from experiments.area_pressure.pusht_lateral_readback import _bind_initial

    summary, source = initial_binding()
    if fault == "time":
        summary["initial_state"]["time"] += 0.001
        summary["initialization_proof"]["initial_time"] = summary["initial_state"]["time"]
    elif fault == "velocity":
        summary["initial_state"]["v"][0] = 0.125
    elif fault == "translation":
        summary["initial_state"]["q"][2] = 0.625
        summary["initial_state"]["object_position_world_m"][0] = 0.625
        summary["initialization_proof"]["published_q"] = deepcopy(summary["initial_state"]["q"])
    elif fault == "proof-source":
        summary["initialization_proof"]["source_q"][0] = 0.125
    else:
        summary["initialization_proof"]["previous_receipt_inherited"] = True
    with pytest.raises(ValueError):
        _bind_initial(summary, source)


def test_bound_events_allow_pre_initialization_failure_without_claiming_physics():
    sequence, summary = bound_events(failed=True)
    for key in (
        "initial_state",
        "final_state",
        "ordinary_route",
        "failed_candidate",
        "initialization_proof",
    ):
        summary.pop(key)
    summary.update(physical_attempts=0, physical_trial_steps=0, failure_stage="construct_new_scene")
    failure = {
        key: summary.get(key)
        for key in ("failure_stage", "errors", "failed_candidate", "unrecorded_accepted_step")
    }
    sequence = [
        dict(
            schema="phymeo-pusht-lateral-route-event-v1",
            event="failure",
            trial_index=None,
            payload=failure,
        ),
        dict(
            schema="phymeo-pusht-lateral-route-event-v1",
            event="final",
            trial_index=None,
            payload={key: value for key, value in summary.items() if key != "journal"},
        ),
    ]
    assert list(accepted_pairs(sequence, summary=summary)) == []


def test_bound_events_preserve_accepted_tail_when_candidate_record_itself_failed():
    sequence, summary = bound_events(callback_failed=True)
    del sequence[2]
    summary["failure_stage"] = "candidate_record"
    sequence[-2]["payload"]["failure_stage"] = "candidate_record"
    sequence[-1]["payload"] = {
        key: deepcopy(value) for key, value in summary.items() if key != "journal"
    }
    assert list(accepted_pairs(sequence, summary=summary)) == []
    assert summary["accepted_steps"] == 1 and summary["recorded_steps"] == 0


def test_bound_events_check_restored_trial_that_physically_stepped_before_assessment_error():
    sequence, summary = bound_events(failed=True)
    after = deepcopy(summary["initial_state"])
    after.update(local_tick=1, time=4.451)
    summary["physical_trial_steps"] = 1
    summary["failed_candidate"].update(
        physics_accepted_before_restore=True, candidate_output_states={"6": after}
    )
    sequence[-2]["payload"]["failed_candidate"] = deepcopy(summary["failed_candidate"])
    sequence[-1]["payload"] = {
        key: deepcopy(value) for key, value in summary.items() if key != "journal"
    }
    assert list(accepted_pairs(sequence, summary=summary)) == []
    assert summary["final_state"] == summary["initial_state"]
