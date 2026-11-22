"""Independent saved-data audit checks; analytic fixtures are not rollout evidence."""

from copy import deepcopy

import numpy as np
import pytest


def material_case(offset=0.0):
    # Unit tetrahedron with p=x; the finite +x face cuts x=1/4.
    return dict(
        pressure_vertices=np.eye(4, 3, k=-1),
        cells=[(0, 1, 2, 3)],
        nodal_pressure=[0.0, 1.0, 0.0, 0.0],
        rigid_vertices=[(0.25, -1.0, -1.0), (0.25, 2.0, -1.0), (0.25, -1.0, 2.0)],
        faces=[(0, 1, 2)],
        pressure_transform=(np.eye(3), [offset, 0.0, 0.0]),
        rigid_transform=(np.eye(3), [offset, 0.0, 0.0]),
    )


def test_independent_exact_clip_integrates_known_finite_pressure_and_world_points():
    from experiments.area_pressure.pusht_lateral_resume_reference import reference_quadrature

    result = reference_quadrature(**material_case())
    assert sum(result["dA"]) == pytest.approx(9 / 32, abs=2e-15)
    assert np.dot(result["dA"], result["pressure"]) == pytest.approx(9 / 128, abs=2e-15)
    np.testing.assert_allclose(result["normals"], np.tile([1.0, 0.0, 0.0], (len(result["dA"]), 1)))
    np.testing.assert_allclose(result["gradient"], 1.0)
    np.testing.assert_allclose(
        np.average(result["points"], axis=0, weights=result["dA"]), [0.25] * 3
    )


def test_reference_never_forms_thin_world_edges_after_translation_rounding():
    from experiments.area_pressure.pusht_lateral_resume_reference import reference_quadrature

    near = reference_quadrature(**material_case())
    far = reference_quadrature(**material_case(2.0**54))
    # World x loses sub-unit offsets, while exact differences still retain area,
    # pressure and normal. No production intersection/publication helper is used.
    np.testing.assert_array_equal(far["dA"], near["dA"])
    np.testing.assert_array_equal(far["pressure"], near["pressure"])
    np.testing.assert_array_equal(far["normals"], near["normals"])
    assert np.all(far["points"][:, 0] == 2.0**54)


def retry_records():
    control = dict(
        tick=5836,
        output_tick=5837,
        ctrl=[0.25, -0.125],
        qfrc_applied=[0.0] * 8,
        xfrc_applied=[[0.0] * 6 for _ in range(5)],
    )
    parent = dict(
        schema="phymeo-pusht-lateral-route-v1",
        start_tick=4450,
        recorded_steps=1386,
        accepted_steps=1386,
        failed_candidate=dict(
            control=deepcopy(control), restored=True, physics_accepted_before_restore=False
        ),
    )
    summary = dict(start_tick=5836, first_command_binding={"matches_parent_rejected_command": True})
    return summary, parent, control


def test_retry_binds_all_three_original_channels_and_absolute_phase():
    from experiments.area_pressure.pusht_lateral_resume_audit import verify_retry_command

    summary, parent, command = retry_records()
    assert verify_retry_command(summary, parent, command)["verified"] is True


@pytest.mark.parametrize("fault", ["ctrl", "qfrc_applied", "xfrc_applied", "tick", "signed-zero"])
def test_retry_rejects_changed_channel_words_or_phase(fault):
    from experiments.area_pressure.pusht_lateral_resume_audit import verify_retry_command

    summary, parent, command = retry_records()
    if fault == "tick":
        command["tick"] += 1
    elif fault == "xfrc_applied":
        command[fault][0][0] = 0.125
    elif fault == "signed-zero":
        command["qfrc_applied"][0] = -0.0
    else:
        command[fault][0] += 0.125
    with pytest.raises(ValueError):
        verify_retry_command(summary, parent, command)


def first_packet_binding():
    import hashlib
    import json

    _, _, command = retry_records()
    channels = {k: command[k] for k in ("ctrl", "qfrc_applied", "xfrc_applied")}
    digest = hashlib.sha256(
        json.dumps(channels, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    packet = dict(
        schema="phymeo-pusht-lateral-resume-first-step-v1",
        input_state={"q": [1.0]},
        output_state={"q": [2.0]},
        control=command,
        measurement={"world": 6},
    )
    row = {k: deepcopy(packet[k]) for k in ("input_state", "output_state", "measurement")}
    row["control"] = dict(deepcopy(command), channels_sha256=digest)
    return packet, row


def test_first_packet_binds_real_writer_control_and_derived_journal_hash():
    from experiments.area_pressure.pusht_lateral_resume_reference import _bind_first_packet

    packet, row = first_packet_binding()
    _bind_first_packet(packet, row)


@pytest.mark.parametrize("fault", ["hash", "channel", "phase"])
def test_first_packet_rejects_wrong_hash_shared_channel_or_phase(fault):
    from experiments.area_pressure.pusht_lateral_resume_reference import _bind_first_packet

    packet, row = first_packet_binding()
    if fault == "hash":
        row["control"]["channels_sha256"] = "0" * 64
    elif fault == "channel":
        packet["control"]["ctrl"][0] += 0.125
    else:
        packet["control"]["tick"] += 1
    with pytest.raises(ValueError):
        _bind_first_packet(packet, row)


def source_binding_records():
    state = dict(
        local_tick=1386,
        time=5.836,
        q=[0.0, 0.0, 0.5, 0.0, 0.015625, 1.0, 0.0, 0.0, 0.0],
        v=[0.0] * 8,
        object_position_world_m=[0.5, 0.0, 0.015625],
        object_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
    )
    route = dict(
        start_tick=4450,
        end_tick=6700,
        sha256="same-route",
        nodes=[[0.0, "push", [0.0, 0.0]], [2.25, "push", [1.0, 1.0]]],
        completed_routes_before=0,
    )
    goal = {"target_xy": [0.55, 0.13], "target_yaw": 0.2}
    controller = dict(tick=5836, status="continue")
    history = dict(tick=1385, time_s=5.835)
    parent = dict(
        start_tick=4450,
        recorded_steps=1386,
        accepted_steps=1386,
        final_state=deepcopy(state),
        goal_tick_origin=0,
        goal=goal,
        controller_final=controller,
        goal_history_final=history,
        active_route=route,
        provenance={"scene_asset_sha256": "material"},
        control_variant="load-compensation-half",
    )
    parent_audit = dict(
        actual={"verified_steps": 1386, "verified_final_state": deepcopy(state)},
        physical_final_state=deepcopy(state),
    )
    metadata = dict(
        state=deepcopy(state),
        source_tick=5836,
        target=deepcopy(goal),
        controller=deepcopy(controller),
        goal_history=deepcopy(history),
        controller_at_physical_endpoint=True,
        physical_state_matches_report=True,
        active_route=deepcopy(route),
        asset_sha256="material",
        checkpoint_sha256="capsule",
    )
    seed = dict(
        input_state=dict(deepcopy(state), local_tick=1385, time=5.835),
        output_state=deepcopy(state),
        goal_result={"history": deepcopy(history)},
        original_input_tick=5835,
        original_output_tick=5836,
        area={"total_force_on_pusher_world_n": [0.125, -0.25, 0.0]},
    )
    summary = dict(
        initial_state=deepcopy(state),
        start_tick=5836,
        local_tick_origin=1386,
        source_reference_tick_origin=4450,
        goal_tick_origin=0,
        goal=deepcopy(goal),
        controller_initial=deepcopy(controller),
        goal_history_initial=deepcopy(history),
        active_route=deepcopy(route),
        control_variant="load-compensation-half",
        provenance={"scene_asset_sha256": "material"},
        checkpoint_restore=dict(
            checkpoint_sha256="capsule",
            restored_local_tick=1386,
            absolute_input_tick=5836,
            time=5.836,
            previous_receipt_bound=True,
            controller_callbacks_replayed=True,
            prior_physics_steps_replayed=0,
            previous_tool_force_world_n=[0.125, -0.25, 0.0],
        ),
    )
    return summary, parent, parent_audit, metadata, seed


def test_resume_source_keeps_nonzero_local_clock_and_immediate_preceding_force():
    from experiments.area_pressure.pusht_lateral_resume_audit import bind_source

    bind_source(*source_binding_records())


@pytest.mark.parametrize("fault", ["state", "clock", "load", "route", "history", "unrecorded"])
def test_resume_source_rejects_reset_or_foreign_load_route_history(fault):
    from experiments.area_pressure.pusht_lateral_resume_audit import bind_source

    values = source_binding_records()
    summary, parent = values[:2]
    if fault == "state":
        summary["initial_state"]["time"] += 0.001
    elif fault == "clock":
        summary["local_tick_origin"] = 0
    elif fault == "load":
        summary["checkpoint_restore"]["previous_tool_force_world_n"][0] += 0.125
    elif fault == "route":
        summary["active_route"]["sha256"] = "new-plan"
    elif fault == "history":
        summary["goal_history_initial"]["tick"] = 0
    else:
        parent["unrecorded_accepted_step"] = {"output_state": deepcopy(parent["final_state"])}
    with pytest.raises(ValueError):
        bind_source(*values)


def test_ordinary_second_segment_never_claims_a_nonexistent_rejected_command():
    from experiments.area_pressure.pusht_lateral_resume_audit import verify_retry_command

    summary, parent, command = retry_records()
    parent.pop("failed_candidate")
    summary["first_command_binding"]["matches_parent_rejected_command"] = None
    assert verify_retry_command(summary, parent, command)["verified"] is False
    summary["first_command_binding"]["matches_parent_rejected_command"] = True
    with pytest.raises(ValueError):
        verify_retry_command(summary, parent, command)


def resumed_events(*, failed=False, callback_failed=False):
    from test_pusht_lateral_readback import bound_events

    events, summary = bound_events(failed=failed, callback_failed=callback_failed)

    def move(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "local_tick":
                    value[key] += 1386
                elif key in (
                    "original_input_tick",
                    "original_output_tick",
                    "tick",
                    "output_tick",
                    "start_tick",
                ):
                    value[key] += 1386
                elif key == "time":
                    value[key] += 1.386
                else:
                    move(item)
        elif isinstance(value, list):
            for item in value:
                move(item)

    move(events)
    move(summary)
    summary.pop("ordinary_route")
    summary.update(
        schema="phymeo-pusht-lateral-resume-v1",
        source_reference_tick_origin=4450,
        local_tick_origin=1386,
        goal_tick_origin=0,
        publication_frame="rigid-origin",
    )
    events = [e for e in events if e["event"] != "planner"]
    initial = deepcopy(summary)
    initial.update(
        status="running",
        final_state=deepcopy(summary["initial_state"]),
        physical_attempts=0,
        physical_trial_steps=0,
        accepted_steps=0,
        recorded_steps=0,
        rejected_trials=0,
    )
    events[0]["payload"] = initial
    events[-1]["payload"] = {
        key: deepcopy(value) for key, value in summary.items() if key != "journal"
    }
    return events, summary


def test_resumed_event_stream_preserves_local_origin_and_failed_rollback_boundary():
    from experiments.area_pressure.pusht_lateral_resume_audit import iter_committed

    for failed in (False, True):
        events, summary = resumed_events(failed=failed)
        assert len(list(iter_committed(events, summary))) == int(not failed)


@pytest.mark.parametrize("fault", ["reset", "failed-input", "final", "budget", "replan"])
def test_resumed_stream_rejects_disconnected_state_and_extra_attempts(fault):
    from experiments.area_pressure.pusht_lateral_resume_audit import iter_committed

    events, summary = resumed_events(failed=fault == "failed-input")
    if fault == "reset":
        events[1]["payload"]["input_state"]["local_tick"] = 0
    elif fault == "failed-input":
        events[1]["payload"]["failed_candidate"]["input_states"]["6"]["time"] += 0.001
        summary["failed_candidate"] = deepcopy(events[1]["payload"]["failed_candidate"])
        events[-1]["payload"]["failed_candidate"] = deepcopy(summary["failed_candidate"])
    elif fault == "final":
        events[-1]["payload"]["final_state"]["time"] += 0.001
    elif fault == "budget":
        summary["physical_attempts"] += 1
        events[-1]["payload"]["physical_attempts"] += 1
    else:
        events.insert(
            1,
            dict(
                schema="phymeo-pusht-lateral-route-event-v1",
                event="planner",
                payload={},
                trial_index=None,
            ),
        )
    with pytest.raises(ValueError):
        list(iter_committed(events, summary))


def audit_failure_boundary(tmp_path, monkeypatch, *, callback_failed):
    """Report-only fixture: actual event/boundary checks, isolated source/physics.

    The independent physics stub below is deliberately not rollout evidence.
    Geometry/root correctness is covered separately; these tests check whether
    audit() publishes a claim only after its real candidate branch ran.
    """
    import hashlib
    import json
    from types import SimpleNamespace

    from experiments.area_pressure import pusht_lateral_resume_audit as reader
    from phymeo.training import manipulation_tasks

    sequence, summary = resumed_events(failed=not callback_failed, callback_failed=callback_failed)
    initial = summary["initial_state"]
    summary["goal"] = dict(
        target_xy=[0.55, 0.13], target_yaw=0.2, position_tolerance_m=0.01, yaw_tolerance_rad=0.01
    )
    summary["parent_physical_attempts"] = 1387
    summary["provenance"]["source_model_sha256"] = "unit-source-model"
    route = {"route": {"nodes": [[0.0, "push", [0.0, 0.0]], [2.25, "push", [1.0, 1.0]]]}}
    route_path = tmp_path / "route.json"
    route_path.write_text(json.dumps(route), encoding="utf-8")
    summary["active_route"] = dict(
        file=route_path.name,
        sha256=hashlib.sha256(route_path.read_bytes()).hexdigest(),
        nodes=route["route"]["nodes"],
    )
    summary["first_command_binding"] = {"matches_parent_rejected_command": None}
    parent = dict(recorded_steps=1386)
    parent_audit = dict(combined_verified_steps=1386)
    seed = {
        "area": {"raw_normal_force_n": [1.0], "total_force_on_pusher_world_n": [0.125, 0.0, 0.0]}
    }
    calls = []

    def independent_first(packet, row):
        calls.append(row["trial_index"])
        return dict(verified=True, reference="unit-test-only physics result")

    monkeypatch.setattr(reader, "_parent", lambda *_: (parent, parent_audit, seed))
    monkeypatch.setattr(reader, "_LoadCompensationAudit", lambda *_: object())
    monkeypatch.setattr(reader, "audit_first_step", independent_first)
    monkeypatch.setattr(
        manipulation_tasks,
        "manipulation_scenes",
        lambda: [SimpleNamespace(key="pusht-planar", source_sha256="unit-source-model")],
    )
    if callback_failed:
        packet = {
            "previous_receipt": dict(
                tick=initial["local_tick"],
                output_time=initial["time"],
                raw_contact_fields={
                    "area_friction": [{"normal_force": [1.0], "force_on_a": [[0.125, 0.0, 0.0]]}]
                },
            )
        }
        packet_path = tmp_path / "first.json"
        packet_path.write_text(json.dumps(packet), encoding="utf-8")
        summary["first_step_evidence"] = dict(
            file=packet_path.name,
            sha256=hashlib.sha256(packet_path.read_bytes()).hexdigest(),
            input_tick=summary["start_tick"],
            output_tick=summary["start_tick"] + 1,
        )
    # Refresh the writer-shaped event copies after supplying audit prerequisites.
    sequence[0]["payload"].update(
        goal=deepcopy(summary["goal"]),
        provenance=deepcopy(summary["provenance"]),
        active_route=deepcopy(summary["active_route"]),
        parent_physical_attempts=summary["parent_physical_attempts"],
    )
    sequence[-1]["payload"] = {
        key: deepcopy(value) for key, value in summary.items() if key != "journal"
    }
    journal_path = tmp_path / "run.trials.jsonl"
    journal_path.write_text(
        "".join(json.dumps(event) + "\n" for event in sequence), encoding="utf-8"
    )
    summary["journal"] = dict(
        file=journal_path.name,
        records=len(sequence),
        sha256=hashlib.sha256(journal_path.read_bytes()).hexdigest(),
    )
    summary_path = tmp_path / "run.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return reader.audit(summary_path), calls


def test_failure_before_first_candidate_does_not_claim_independent_physics(tmp_path, monkeypatch):
    result, calls = audit_failure_boundary(tmp_path, monkeypatch, callback_failed=False)
    assert calls == []
    assert result["status"] == "partial_failure"
    assert result["first_step_reference"] is None
    assert result["independent_first_step_physics"] is False
    assert result["scope"].startswith("no first-candidate independent physics verification; ")
    assert result["actual"]["verified_steps"] == 0
    assert result["physical_final_state"] == result["actual"]["verified_final_state"]
    assert result["unrecorded_accepted_step"] is None


def test_first_candidate_physics_survives_callback_failure_without_counting_committed_motion(
    tmp_path, monkeypatch
):
    result, calls = audit_failure_boundary(tmp_path, monkeypatch, callback_failed=True)
    assert calls == [0]
    assert result["status"] == "partial_failure"
    assert result["first_step_reference"]["verified"] is True
    assert result["independent_first_step_physics"] is True
    assert result["scope"].startswith(
        "first saved physical candidate independently verified in FP64; "
    )
    assert result["actual"]["verified_steps"] == 0
    assert result["actual"]["trajectory"] == []
    assert result["actual"]["normal_impulse_ns"] == 0.0
    assert result["unrecorded_accepted_step"]["output_state"] == result["physical_final_state"]
    assert (
        result["physical_final_state"]["local_tick"]
        == result["actual"]["verified_final_state"]["local_tick"] + 1
    )
