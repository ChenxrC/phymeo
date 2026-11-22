"""Portable saved-data bindings; fixtures do not claim simulated physical evidence."""

import gzip
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.area_pressure.pusht_lookahead_source import load_lookahead_source


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return dict(file=str(path), sha256=_sha(path))


def saved_source(tmp_path, *, fault=None):
    """Three fabricated accepted records exercise IO/identity, never dynamics."""

    def state(tick):
        q = [0.0, 0.0, 0.5 + (tick - 20) / 32, 0.125, 0.015625, 1.0, 0.0, 0.0, 0.0]
        return dict(
            local_tick=tick,
            time=1.0 + (tick - 20) * 0.001,
            q=q,
            v=[0.0] * 8,
            object_position_world_m=q[2:5],
            object_quaternion_wxyz=q[5:9],
        )

    states = [state(i) for i in range(20, 24)]
    goal = dict(
        target_xy=[0.75, 0.25], target_yaw=0.2, position_tolerance_m=0.015, yaw_tolerance_rad=0.12
    )
    model_sha, asset_sha = "a" * 64, "b" * 64
    dataset = dict(
        schema="original-push-boundary-endpoints-v1",
        rows=[
            dict(
                summary=dict(world=5, model_sha256="c" * 64),
                query=dict(q=states[0]["q"], identity="portable-only"),
            )
        ],
    )
    if fault == "candidate-world":
        dataset["rows"].append(deepcopy(dataset["rows"][0]))
    candidate = _write(tmp_path / "candidate.json", dataset)
    prior = _write(
        tmp_path / "prior.json",
        dict(
            worlds=dict(open=5, feedback=6),
            provenance=dict(
                candidate_source=candidate["file"],
                candidate_source_sha256=candidate["sha256"],
                source_sha256=model_sha,
            ),
        ),
    )
    history = dict(tick=19, time_s=0.999, binding=["portable", 6])
    controller = dict(
        tick=100,
        status="continue",
        partial_route=None,
        completed_routes=1,
        total_contact_impulse_ns=0.5,
        ik_calls=40,
        path_calls=16,
    )
    parent = _write(
        tmp_path / "parent.json",
        dict(
            schema="phymeo-pusht-lateral-resume-v1",
            status="route_complete",
            start_tick=99,
            accepted_steps=1,
            recorded_steps=1,
            final_state=states[0],
            goal=goal,
            controller_final=controller,
            goal_history_final=history,
            provenance=dict(scene_asset_sha256=asset_sha),
        ),
    )
    parent_audit = _write(
        tmp_path / "parent-audit.json",
        dict(
            source_sha256=parent["sha256"],
            physical_final_state=states[0],
            original_attempts_consumed=8,
        ),
    )
    metadata = _write(
        tmp_path / "metadata.json",
        dict(
            state=states[0],
            source_tick=100,
            controller=controller,
            goal_history=history,
            asset_sha256=asset_sha,
            target=goal,
        ),
    )
    prediction = dict(
        predicted_translation_xy=[0.01, 0.0],
        predicted_yaw_change=0.02,
        predicted_position_xy=[0.54125, 0.125],
        predicted_yaw=0.02,
        nominal_yaw_moment=0.01,
        selection_policy="joint-pose-balanced",
        selection_reason="joint_pose_balanced",
        position_error_before_m=0.25,
        position_error_after_m=0.24,
        yaw_error_before_rad=0.2,
        yaw_error_after_rad=0.18,
    )
    proposal = dict(index=5, distance_ordinal=2, input_binding="d" * 64, **prediction)
    route = dict(
        tick=101,
        q=states[1]["q"],
        v=states[1]["v"],
        cost=2.0,
        proposals=[proposal],
        route=dict(
            status="sampled_reference_ready",
            proposal=proposal,
            source_q=states[1]["q"],
            nodes=[[0.0, "approach", [0.0, 0.0]], [0.003, "push", [0.125, 0.25]]],
            initial_reference_projection=[0.0, 0.0],
            ik_calls=47,
            path_calls=21,
        ),
    )
    if fault == "route-q":
        route["q"] = deepcopy(route["q"])
        route["q"][0] = 0.125
    if fault == "projection":
        route["route"]["initial_reference_projection"][0] = 0.125
    route_path = tmp_path / "route.json.gz"
    with gzip.open(route_path, "wt", encoding="utf-8") as stream:
        json.dump(route, stream)
    release = _write(
        tmp_path / "release.json",
        dict(
            tick=100,
            q=states[0]["q"],
            v=states[0]["v"],
            route=dict(
                status="sampled_release_ready",
                ik_calls=41,
                path_calls=18,
                nodes=[[0.0, "withdraw", [0.0, 0.0]], [0.001, "withdraw", [0.0, 0.0]]],
            ),
        ),
    )
    plan = dict(
        tick=101,
        source_q=states[1]["q"],
        source_v=states[1]["v"],
        route_file=str(route_path),
        route_sha256=_sha(route_path),
        status="sampled_reference_ready",
        selection_policy="joint-pose-balanced",
    )
    summary = dict(
        schema="phymeo-pusht-lateral-replan-v1",
        status="step_budget",
        world=6,
        side_coverage="all-lateral",
        publication_frame="rigid-origin",
        control_variant="load-compensation-half",
        start_tick=100,
        local_tick_origin=20,
        source_reference_tick_origin=80,
        goal_tick_origin=0,
        initial_state=states[0],
        final_state=states[-1],
        goal=goal,
        controller_initial=controller,
        goal_history_initial=history,
        physical_attempts=3,
        accepted_steps=3,
        recorded_steps=3,
        step_budget=3,
        original_step_budget=11,
        parent_physical_attempts=8,
        release_reference=dict(
            file=release["file"], sha256=release["sha256"], nodes=read_release_nodes(release)
        ),
        ordinary_route=plan,
        reference_budget=dict(route_steps=3, available_steps=2, complete_route_within_budget=False),
        active_route=dict(
            start_tick=101,
            end_tick=104,
            nodes=route["route"]["nodes"],
            file=str(route_path),
            sha256=_sha(route_path),
        ),
        provenance=dict(
            prior_summary_path=prior["file"],
            prior_summary_sha256=prior["sha256"],
            source_model_sha256=model_sha,
            scene_asset_sha256=asset_sha,
            parent_report_path=parent["file"],
            parent_report_sha256=parent["sha256"],
            parent_audit_path=parent_audit["file"],
            parent_audit_sha256=parent_audit["sha256"],
            checkpoint_metadata_path=metadata["file"],
            checkpoint_metadata_sha256=metadata["sha256"],
        ),
    )
    events, trajectory = [], []

    def emit(kind, payload, index=None):
        events.append(
            dict(
                schema="phymeo-pusht-lateral-replan-event-v1",
                event=kind,
                trial_index=index,
                payload=deepcopy(payload),
            )
        )

    emit("initialization", summary)
    emit("release_reference", dict(reference=summary["release_reference"]))
    for i in range(3):
        if i == 1:
            emit("planner", plan)
        before, after = states[i], states[i + 1]
        phase = "withdraw" if i == 0 else "regular"
        measurement = dict(
            world=6,
            input_tick=100 + i,
            output_tick=101 + i,
            time_s=before["time"],
            phase="accepted-step-input",
            group_force_world=[0.0, 0.0, 0.0],
            area_samples=0,
            native_contacts=0,
        )
        row = dict(
            index=i,
            trial_index=i,
            phase=phase,
            original_input_tick=100 + i,
            original_output_tick=101 + i,
            input_state=before,
            output_state=after,
            selected=True,
            restored=False,
            measurement=measurement,
            material_identity=dict(compliant_body="tool", rigid_body="object"),
            area=dict(normal_load_n=0.0, total_force_on_pusher_world_n=[0.0, 0.0, 0.0]),
            controller=dict(
                stage="withdraw" if i == 0 else "push",
                measurement_q=before["q"],
                measurement_v=before["v"],
            ),
        )
        callback = dict(
            goal_result=dict(
                success=False,
                history=dict(
                    tick=before["local_tick"], time_s=before["time"], binding=history["binding"]
                ),
            ),
            feedback_accepted=dict(
                measurement=measurement,
                segment_impulse_ns=0.1,
                total_impulse_ns=0.6,
                completed_route=False,
            ),
            driver_state_after=dict(
                tick=101 + i,
                status="continue",
                partial_route=None if i == 0 else dict(start_tick=101),
                completed_routes=1,
                total_contact_impulse_ns=0.6,
                final_cost=2.0,
            ),
        )
        if fault == "goal-phase" and i == 0:
            callback["goal_result"]["history"]["tick"] += 1
        emit("candidate", row, i)
        emit("accepted_callback", callback, i)
        trajectory.append(
            dict(
                phase=phase,
                stage=row["controller"]["stage"],
                input_tick=100 + i,
                output_tick=101 + i,
                input_time_s=before["time"],
                output_time_s=after["time"],
                input_xy_m=before["q"][2:4],
                output_xy_m=after["q"][2:4],
                normal_load_n=0.0,
                input_position_error_m=0.25,
                output_position_error_m=0.24,
                input_yaw_error_rad=0.2,
                output_yaw_error_rad=0.18,
            )
        )
    summary["controller_final"] = dict(
        controller,
        tick=103,
        ik_calls=47,
        path_calls=21,
        routes=[
            dict(
                tick=101,
                file=route_path.name,
                sha256=_sha(route_path),
                status="sampled_reference_ready",
                ik_calls=47,
                path_calls=21,
            )
        ],
    )
    emit("final", summary)
    journal = tmp_path / "run.jsonl"
    journal.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    summary["journal"] = dict(file=journal.name, sha256=_sha(journal), records=len(events))
    report_path = tmp_path / "run.json"
    _write(report_path, summary)
    audit = dict(
        schema="phymeo-pusht-lateral-replan-audit-v1",
        source_sha256=_sha(report_path),
        journal_sha256=_sha(journal),
        status="step_budget",
        physical_final_state=states[-1],
        goal=goal,
        actual=dict(
            verified_steps=3,
            verified_final_state=states[-1],
            trajectory=trajectory,
            final=dict(position_error_m=0.24, yaw_error_rad=0.18),
            phases=dict(regular=dict(steps=2)),
        ),
        planning=dict(
            ordinary=dict(
                input_tick=101,
                status="sampled_reference_ready",
                route=route["route"],
                proposals=route["proposals"],
                heuristic_prediction=prediction,
            )
        ),
        prediction_vs_actual=dict(
            measured_regular_steps=2, prediction_available=True, actual_both_improved=True
        ),
        budget=dict(original=11, prior_attempts=8, allowance=3, used=3, remaining=0),
    )
    if fault == "audit-goal":
        audit["goal"] = dict(goal, target_yaw=0.3)
    audit_path = tmp_path / "audit.json"
    _write(audit_path, audit)
    return report_path, audit_path, states[1]


def read_release_nodes(source):
    return json.loads(Path(source["file"]).read_text(encoding="utf-8"))["route"]["nodes"]


def test_portable_binding_keeps_entry_before_future_baseline_and_before_plan_counters(tmp_path):
    report, audit, entry = saved_source(tmp_path)
    result = load_lookahead_source(report, audit)
    assert result["entry"]["state"] == entry
    assert result["entry"]["tick"] == 101
    assert result["entry"]["driver_candidate"]["world"] == 5
    assert result["entry"]["world"] == 6
    assert result["entry"]["planner_counts_before"] == dict(ik_calls=41, path_calls=18)
    assert result["baseline"]["planner_counts_after"] == dict(ik_calls=47, path_calls=21)
    assert result["baseline"]["selected_index"] == 5
    assert result["baseline"]["proposal_identities"] == [[5, 2, "d" * 64]]
    assert result["budget"]["remaining_steps"] == 2
    assert result["entry"]["state"] != result["baseline"]["final_state"]
    assert result["entry"]["controller_boundary"]["partial_route"] is None
    assert result["entry"]["segment_impulse_ns"] == 0.1
    assert result["entry"]["goal_history"]["tick"] == 20
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "fault", ["route-q", "projection", "goal-phase", "candidate-world", "audit-goal"]
)
def test_consistently_rehashed_but_mismatched_sources_are_rejected(tmp_path, fault):
    report, audit, _ = saved_source(tmp_path, fault=fault)
    with pytest.raises(ValueError):
        load_lookahead_source(report, audit)


@pytest.mark.parametrize("target", ["report", "journal", "route", "candidate"])
def test_changed_bound_bytes_are_rejected(tmp_path, target):
    report, audit, _ = saved_source(tmp_path)
    path = {
        "report": report,
        "journal": tmp_path / "run.jsonl",
        "route": tmp_path / "route.json.gz",
        "candidate": tmp_path / "candidate.json",
    }[target]
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError):
        load_lookahead_source(report, audit)


def test_explicit_foreign_route_is_rejected(tmp_path):
    report, audit, _ = saved_source(tmp_path)
    other = tmp_path / "other.json.gz"
    other.write_bytes(b"different source")
    with pytest.raises(ValueError):
        load_lookahead_source(report, audit, route_path=other)


def test_real_archive_regression_is_optional_without_private_artifacts():
    root = Path(__file__).resolve().parents[1]
    base = root / "artifacts/rr-loaded-replan-20261122-01"
    report, audit = base / "live.json", base / "live-audit.json"
    if not report.is_file() or not audit.is_file():
        pytest.skip("optional local saved episode is not part of an independent checkout")
    result = load_lookahead_source(report, audit)
    assert result["entry"]["tick"] == 6950
    assert result["entry"]["state"]["local_tick"] == 2500
    assert result["entry"]["state"]["time"] == 6.950000330107287
    assert result["budget"]["remaining_steps"] == 1499
    assert result["entry"]["planner_counts_before"] == dict(ik_calls=41, path_calls=18)
    assert result["baseline"]["planner_counts_after"] == dict(ik_calls=47, path_calls=21)
    assert result["baseline"]["selected_index"] == 5
    assert result["baseline"]["comparison"]["measured_regular_steps"] == 1499
