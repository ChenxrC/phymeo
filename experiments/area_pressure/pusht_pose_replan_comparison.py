"""Independent saved-evidence readouts for two differently initialized pose replans.

python -m experiments.area_pressure.pusht_pose_replan_comparison --input RUN.json --output AUDIT.json

Separate saved runs: --case-a A.case.json --case-b B.case.json --output AUDIT.json

These are separate rollouts, not a causal orientation-only A/B experiment. The
reader reuses accepted-state/receipt checks; planning predictions never replace
measured poses, forces or the original saved Goal result.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .pusht_active_comparison import (
    H,
    _array,
    _channels,
    _close,
    _constant,
    _digest,
    _integration,
    _load,
    _pose_errors,
    _read,
    _receipt,
    _require,
    _state,
)


def retained_state(candidate, previous):
    _require(candidate["input_state"] == previous, "candidate input is not the retained state")
    _require(
        type(candidate["selected"]) is bool and type(candidate["restored"]) is bool,
        "explicit candidate selection and restoration",
    )
    output = candidate["output_state"]
    _require(
        output["local_tick"] == previous["local_tick"] + 1 and output["time"] > previous["time"],
        "one actual candidate step",
    )
    if candidate["selected"]:
        _require(candidate["restored"] is False, "selected candidate was restored")
        return output
    _require(
        candidate["restored"] is True
        and candidate.get("rollback_verification", {}).get("passed") is True,
        "rejected candidate lacks verified rollback",
    )
    return previous


def validate_counts(
    summary,
    *,
    candidate_count,
    selected_count,
    callback_count,
    failed_attempts=0,
    failed_physical=0,
):
    for key in (
        "step_budget",
        "physical_attempts",
        "physical_trial_steps",
        "accepted_steps",
        "recorded_steps",
        "rejected_trials",
    ):
        _require(type(summary[key]) is int and summary[key] >= 0, "finite integral counter " + key)
    _require(
        summary["step_budget"] > 0 and summary["physical_attempts"] <= summary["step_budget"],
        "physical attempt budget",
    )
    _require(
        summary["physical_attempts"] == candidate_count + failed_attempts
        and summary["physical_trial_steps"] == candidate_count + failed_physical
        and summary["accepted_steps"] == selected_count
        and summary["recorded_steps"] == callback_count
        and summary["rejected_trials"] == candidate_count - selected_count
        and 0 <= selected_count - callback_count <= 1,
        "attempt/candidate/selection/callback counters",
    )


def summarize_motion(initial, goal, rows):
    """Aggregate only fully recorded selected steps; keep phase contributions separate."""
    prior, trajectory = initial, []
    phases, stages = {}, {}
    impulse, moment_impulse, force_impulse = 0.0, np.zeros(3), np.zeros(3)
    peak = None
    success = False
    for row in rows:
        _require(row["selected"] is True, "rejected trial must not enter accepted motion metrics")
        after = retained_state(row, prior)
        a, b = _pose_errors(prior, goal), _pose_errors(after, goal)
        displacement = (
            np.asarray(after["object_position_world_m"]) - prior["object_position_world_m"]
        )
        angle = math.atan2(
            math.sin(b["yaw_rad"] - a["yaw_rad"]), math.cos(b["yaw_rad"] - a["yaw_rad"])
        )
        measurement = row["feedback_accepted"]["measurement"]
        force = _array(measurement["group_force_world"], (3,), "actual object force")
        moment = _array(measurement["group_moment_com_world"], (3,), "actual object COM moment")
        load = float(row["area"]["normal_load_n"])
        _require(np.isfinite(load) and load >= 0, "actual normal load")
        impulse += H * load
        moment_impulse += H * moment
        force_impulse += H * force
        peak = load if peak is None else max(peak, load)
        success = success or row["goal_result"]["success"]
        for table, key in ((phases, row["phase"]), (stages, row["controller"]["stage"])):
            group = table.setdefault(
                key,
                dict(
                    steps=0,
                    object_displacement_world_m=[0.0, 0.0, 0.0],
                    yaw_change_rad=0.0,
                    position_error_change_m=0.0,
                    yaw_error_change_rad=0.0,
                    normal_impulse_ns=0.0,
                    object_moment_impulse_world_nms=[0.0, 0.0, 0.0],
                ),
            )
            group["steps"] += 1
            group["object_displacement_world_m"] = (
                np.asarray(group["object_displacement_world_m"]) + displacement
            ).tolist()
            group["yaw_change_rad"] += angle
            group["position_error_change_m"] += b["position_error_m"] - a["position_error_m"]
            group["yaw_error_change_rad"] += b["yaw_error_rad"] - a["yaw_error_rad"]
            group["normal_impulse_ns"] += H * load
            group["object_moment_impulse_world_nms"] = (
                np.asarray(group["object_moment_impulse_world_nms"]) + H * moment
            ).tolist()
        sample = dict(
            phase=row["phase"],
            stage=row["controller"]["stage"],
            input_tick=row.get("original_input_tick", prior["local_tick"]),
            output_tick=row.get("original_output_tick", after["local_tick"]),
            input_time_s=prior["time"],
            output_time_s=after["time"],
            input_xy_m=prior["object_position_world_m"][:2],
            output_xy_m=after["object_position_world_m"][:2],
            input_yaw_rad=a["yaw_rad"],
            output_yaw_rad=b["yaw_rad"],
            input_position_error_m=a["position_error_m"],
            output_position_error_m=b["position_error_m"],
            input_yaw_error_rad=a["yaw_error_rad"],
            output_yaw_error_rad=b["yaw_error_rad"],
            normal_load_n=load,
            tool_force_world_n=row["area"]["total_force_on_pusher_world_n"],
            object_force_world_n=force.tolist(),
            object_moment_com_world_nm=moment.tolist(),
            original_goal_success_on_input=row["goal_result"]["success"],
            geometry=row.get("geometry"),
        )
        direction = row.get("planned_push_direction_world")
        if direction is not None:
            direction = _array(direction, (3,), "selected reference push direction")
            sample["object_advance_along_reference_m"] = float(displacement @ direction)
        trajectory.append(sample)
        prior = after
    first, last = _pose_errors(initial, goal), _pose_errors(prior, goal)
    return dict(
        verified_steps=len(rows),
        initial=first,
        final=last,
        verified_final_state=prior,
        object_displacement_world_m=(
            np.asarray(prior["object_position_world_m"]) - initial["object_position_world_m"]
        ).tolist(),
        position_error_change_m=last["position_error_m"] - first["position_error_m"],
        yaw_error_change_rad=last["yaw_error_rad"] - first["yaw_error_rad"],
        phases=phases,
        stages=stages,
        normal_impulse_ns=float(impulse),
        peak_normal_load_n=peak,
        object_force_impulse_world_ns=force_impulse.tolist(),
        object_moment_impulse_world_nms=moment_impulse.tolist(),
        goal_success_observed=bool(success),
        final_pose_within_position_yaw_tolerances=bool(
            last["position_error_m"] <= goal["position_tolerance_m"]
            and last["yaw_error_rad"] <= goal["yaw_tolerance_rad"]
        ),
        trajectory=trajectory,
    )


def _bound(base, name, digest, label):
    path = (base / name).resolve()
    _require(_digest(path) == digest, label + " digest")
    return path, _read(path)


def _wrench(row, measurement, world):
    before = row["input_state"]
    _require(
        measurement["world"] == world
        and measurement["input_tick"] == row["original_input_tick"]
        and measurement["output_tick"] == row["original_output_tick"]
        and measurement["time_s"] == before["time"]
        and measurement["phase"] == "accepted-step-input",
        "actual wrench input geometry/tick phase",
    )
    a = _array(row["area"]["total_force_on_pusher_world_n"], (3,), "tool force")
    b = _array(measurement["group_force_world"], (3,), "object force")
    _array(measurement["group_moment_com_world"], (3,), "saved object COM moment")
    _close(a, -b, "actual bilateral force ownership", atol=1e-7, rtol=8 * np.finfo(np.float32).eps)


def _goal_row(row, goal, world):
    before, frame, result = row["input_state"], row["goal_frame"], row["goal_result"]
    _require(
        frame["world"] == world
        and frame["tick"] == row["original_input_tick"]
        and frame["time_s"] == before["time"]
        and frame["phase"] == "accepted-entry"
        and frame["object_position"] == before["object_position_world_m"]
        and frame["object_wxyz"] == before["object_quaternion_wxyz"]
        and result["history"]["tick"] == frame["tick"]
        and result["history"]["time_s"] == frame["time_s"]
        and type(result["success"]) is bool,
        "original Goal input/state phase",
    )
    errors, metrics = _pose_errors(before, goal), dict(result["metrics"])
    for key in ("position_error_m", "yaw_error_rad"):
        _close(metrics[key], errors[key], "independent " + key, atol=2e-6, rtol=2e-6)


def _planner(base, payload, retained, tick, *, selection_policy="joint-pose-descent"):
    path, saved = _bound(base, payload["route_file"], payload["route_sha256"], "planner route")
    _require(
        payload["tick"] == saved["tick"] == tick
        and payload["source_q"] == saved["q"] == retained["q"]
        and payload["source_v"] == saved["v"] == retained["v"]
        and payload["status"] == saved["route"]["status"],
        "planner uses retained actual state",
    )
    _require(
        selection_policy in ("joint-pose-descent", "joint-pose-balanced")
        and payload["selection_policy"] == selection_policy,
        "planner/case selection policy",
    )
    proposal = saved["route"].get("proposal")
    prediction = None
    if proposal is not None:
        _require(
            payload["selection_policy"] == proposal["selection_policy"] == selection_policy
            and proposal["selection_eligible"] is True,
            "selected pose prediction policy",
        )
        prediction = {
            key: proposal[key]
            for key in (
                "predicted_translation_xy",
                "predicted_yaw_change",
                "predicted_position_xy",
                "predicted_yaw",
                "nominal_yaw_moment",
                "selection_policy",
                "selection_reason",
                "position_error_before_m",
                "position_error_after_m",
                "yaw_error_before_rad",
                "yaw_error_after_rad",
            )
        }
    return dict(
        file=str(path),
        sha256=payload["route_sha256"],
        input_tick=tick,
        status=payload["status"],
        heuristic_prediction=prediction,
        interpretation="geometric/quasistatic prediction only, never a measured dynamic outcome",
    ), proposal


def verify_separation(evidence, retained, tick):
    """Check saved per-pair ordinary-planning admission, without running geometry."""
    _require(
        evidence is not None
        and evidence["output_tick"] == tick
        and evidence["output_time"] == retained["time"]
        and evidence["output_q"] == retained["q"]
        and evidence["output_v"] == retained["v"]
        and evidence["complete_pair_catalog"] is True,
        "ordinary plan separation endpoint",
    )
    pairs = [
        p
        for p in evidence["pairs"]
        if not (p["key"]["kind"] == "native-geometry" and "support" in p["key"]["identity"])
    ]
    _require(bool(pairs), "non-support separation catalogue")
    for pair in pairs:
        _require(
            pair["normal_load_n"] == 0
            and pair["output_depth_m"] == 0
            and pair["output_contact"] is False
            and pair["output_strictly_separated"] is True,
            "ordinary plan requires saved zero-load strict separation per pair",
        )


def compare_case(path):
    path = Path(path).resolve()
    summary, base = _read(path), path.parent
    _require(summary["schema"] == "phymeo-pusht-pose-replan-case-v1", "case schema")
    selection_policy = summary["selection_policy"]
    _require(
        selection_policy in ("joint-pose-descent", "joint-pose-balanced"), "case selection policy"
    )
    initial, goal = summary["initial_state"], summary["goal"]
    origin, start, world = summary["local_tick_origin"], summary["start_tick"], summary["world"]
    _state(initial, origin)
    p = summary["provenance"]
    source_path, source = _bound(
        base, p["source_report_path"], p["source_report_sha256"], "terminal source report"
    )
    _, audit = _bound(
        base, p["source_audit_path"], p["source_audit_sha256"], "terminal source audit"
    )
    _require(
        audit["status"] == "verified-saved-prefix"
        and audit["source"]["summary_sha256"] == _digest(source_path)
        and audit["verified_steps"] == source["recorded_steps"] == source["accepted_steps"]
        and audit["verified_final_state"] == source["final_state"],
        "complete audited terminal source",
    )
    metadata_path, metadata = _bound(
        base,
        p["checkpoint_metadata_path"],
        p["checkpoint_metadata_sha256"],
        "source capsule metadata",
    )
    capsule = metadata_path.parent / metadata["checkpoint_file"]
    _require(
        _digest(capsule)
        == metadata["checkpoint_sha256"]
        == p["source_checkpoint_sha256"]
        == source["final_checkpoint"]["sha256"]
        and metadata["state"] == source["final_state"]
        and metadata["asset_sha256"] == p["scene_asset_sha256"],
        "source capsule/state/material",
    )
    _require(goal == source["goal"], "unchanged original target and tolerances")
    inherited_row = None
    mode = summary["initialization_mode"]
    _require(mode == p["initialization_mode"], "initialization declaration")
    if mode == "terminal-checkpoint":
        _require(
            initial == source["final_state"] and start == metadata["source_tick"],
            "true terminal continuation state",
        )
        _require(
            summary["checkpoint_restore"]["checkpoint_sha256"] == metadata["checkpoint_sha256"]
            and summary["controller_initial"] == metadata["controller"],
            "actual restored capsule/controller",
        )
    elif mode == "independent-pose-reset":
        import mujoco as mj

        from phymeo.training.manipulation_tasks import manipulation_scenes

        scene = next(s for s in manipulation_scenes() if s.key == "pusht-planar")
        model = mj.MjModel.from_xml_string(scene.mjcf)
        declaration = p["reset_declaration"]
        qa = int(model.jnt_qposadr[model.joint("object-free").id])
        robot = [int(model.jnt_qposadr[model.joint(name).id]) for name in scene.robot_joint_names]
        q = np.asarray(initial["q"])
        _require(
            declaration["object_yaw_rad"] == 0.5
            and declaration["prior_receipt_inherited"] is False
            and declaration["robot"] == "original-model-qpos0"
            and declaration["velocity"] == "zero"
            and declaration["object_z"] == "original-model-qpos0"
            and np.array_equal(q, np.asarray(declaration["q"], np.float32))
            and np.array_equal(initial["v"], np.zeros(model.nv))
            and np.array_equal(initial["v"], declaration["v"])
            and initial["time"] == declaration["time"]
            and np.array_equal(q[robot], np.asarray(model.qpos0[robot], np.float32))
            and declaration["source_xy_world_m"]
            == source["final_state"]["object_position_world_m"][:2]
            and np.array_equal(q[qa : qa + 2], declaration["source_xy_world_m"])
            and q[qa + 2] == np.float32(model.qpos0[qa + 2])
            and q[qa + 4] == q[qa + 5] == 0.0,
            "independent public-qpos0/zero-velocity/pure-yaw reset",
        )
        _close(_pose_errors(initial, goal)["yaw_rad"], 0.5, "actual reset yaw", atol=2e-6)
    else:
        raise ValueError("unsupported pose initialization mode")
    journal = summary["journal"]
    journal_path = (base / journal["file"]).resolve()
    _require(_digest(journal_path) == journal["sha256"], "case journal digest")
    retained, pending, accepted_rows, rejected, plans = initial, None, [], [], []
    candidates = selected = records = failed_attempts = failed_physical = 0
    direction, latest_evidence, final_seen = None, None, False
    with journal_path.open("rt", encoding="utf-8-sig") as stream:
        for records, line in enumerate(stream, 1):
            event = json.loads(line, parse_constant=_constant)
            _require(
                not final_seen and event["schema"] == "phymeo-pusht-pose-replan-event-v1",
                "journal event schema/order",
            )
            kind, payload, index = event["event"], event["payload"], event.get("trial_index")
            if kind == "candidate":
                _require(
                    pending is None and index == candidates + failed_attempts == payload["index"],
                    "candidate ordering",
                )
                before, after = payload["input_state"], payload["output_state"]
                _state(before, origin + selected)
                _state(after, origin + selected + 1)
                _require(
                    payload["original_input_tick"] == start + selected
                    and payload["original_output_tick"] == start + selected + 1,
                    "candidate absolute clock",
                )
                _integration(before, after)
                _receipt(payload)
                command = payload.get("submitted_control", payload.get("control"))
                _require(
                    command["tick"] == start + selected
                    and command["output_tick"] == start + selected + 1,
                    "submitted candidate control clock",
                )
                _channels(command)
                c = payload["controller"]
                _require(
                    c["measurement_q"] == before["q"] and c["measurement_v"] == before["v"],
                    "controller actual input",
                )
                _require(
                    payload["phase"] in ("settle", "withdraw", "regular"), "actual execution phase"
                )
                measurement = payload["measurement"]
                _wrench(payload, measurement, world)
                load_view = dict(payload, feedback_accepted=dict(measurement=measurement))
                _load(load_view)
                retained = retained_state(payload, retained)
                candidates += 1
                if payload["selected"]:
                    selected += 1
                    latest_evidence = payload.get("evidence") or latest_evidence
                    pending = (index, payload)
                    if direction is not None and payload["phase"] == "regular":
                        payload["planned_push_direction_world"] = direction
                else:
                    rejected.append(
                        dict(
                            trial_index=index,
                            phase=payload["phase"],
                            input_state=before,
                            actual_candidate_output_state=after,
                            normal_load_n=payload["area"]["normal_load_n"],
                            measurement=measurement,
                            release_decision=payload.get("release_decision"),
                            restored=True,
                            rollback_verification=payload["rollback_verification"],
                        )
                    )
            elif kind == "accepted_callback":
                _require(
                    pending is not None and index == pending[0],
                    "callback belongs to selected candidate",
                )
                row = pending[1]
                _require(
                    payload["feedback_accepted"]["measurement"] == row["measurement"],
                    "same actual candidate/accepted wrench",
                )
                row.update(payload)
                _goal_row(row, goal, world)
                accepted_rows.append(row)
                pending = None
            elif kind == "planner":
                _require(pending is None, "planner runs after full accepted callback")
                verify_separation(latest_evidence, retained, start + selected)
                record, proposal = _planner(
                    base, payload, retained, start + selected, selection_policy=selection_policy
                )
                plans.append(record)
                direction = None if proposal is None else proposal["direction_world"]
            elif kind == "failure":
                failed = payload.get("failed_candidate")
                if failed is not None:
                    failed_attempts += 1
                    failed_physical += int(failed["physics_accepted_before_restore"])
                    _require(
                        failed["input_states"][str(world)] == retained,
                        "failed trial retained input",
                    )
                    _require(
                        failed["restored"] is True
                        and failed["rollback_verification"]["passed"] is True,
                        "failed trial must preserve retained state",
                    )
            elif kind == "release_reference":
                reference, seed = payload["reference"], payload["seed"]
                _, route = _bound(base, reference["file"], reference["sha256"], "release reference")
                _require(
                    route["q"] == retained["q"]
                    and route["v"] == retained["v"]
                    and route["tick"] == start + selected
                    and route["route"]["nodes"] == reference["nodes"]
                    and seed["output_q"] == retained["q"]
                    and seed["output_v"] == retained["v"]
                    and seed["output_tick"] == start + selected
                    and seed["output_time"] == retained["time"],
                    "release source retained boundary",
                )
                if selected == 0 and inherited_row is not None:
                    (whole,) = (
                        p
                        for p in seed["pairs"]
                        if p["key"]["kind"] == "area-material"
                        and p["key"]["identity"][-1] == "whole"
                    )
                    _require(
                        seed["input_q"] == inherited_row["input_state"]["q"]
                        and seed["input_v"] == inherited_row["input_state"]["v"]
                        and seed["input_time"] == inherited_row["input_state"]["time"]
                        and whole["normal_load_n"] == inherited_row["area"]["normal_load_n"]
                        and whole["force_on_a_world"]
                        == inherited_row["area"]["total_force_on_pusher_world_n"],
                        "release seed retains preceding accepted force geometry phase",
                    )
                latest_evidence = seed
            elif kind == "initialization":
                _require(
                    records == 1
                    and payload["initial_state"] == initial
                    and payload["controller_initial"] == summary["controller_initial"],
                    "initialization event binding",
                )
            elif kind == "final":
                _require(
                    payload == {k: v for k, v in summary.items() if k != "journal"},
                    "final event summary binding",
                )
                final_seen = True
            else:
                raise ValueError("unknown pose journal event " + str(kind))
    _require(final_seen and records == journal["records"], "complete case journal")
    validate_counts(
        summary,
        candidate_count=candidates,
        selected_count=selected,
        callback_count=len(accepted_rows),
        failed_attempts=failed_attempts,
        failed_physical=failed_physical,
    )
    _require(retained == summary["final_state"], "actual retained final endpoint")
    if pending is not None:
        _require(
            summary["status"] == "partial_failure", "unrecorded accepted callback requires failure"
        )
    metrics = summarize_motion(initial, goal, accepted_rows)
    return dict(
        schema="phymeo-pusht-pose-replan-case-audit-v1",
        status="verified-saved-prefix",
        source=dict(file=str(path), sha256=_digest(path), journal_sha256=journal["sha256"]),
        initialization_mode=mode,
        selection_policy=selection_policy,
        goal=goal,
        physical_budget=summary["step_budget"],
        physical_attempts=summary["physical_attempts"],
        physical_trial_steps=summary["physical_trial_steps"],
        accepted_steps=selected,
        recorded_steps=len(accepted_rows),
        rejected_trials=rejected,
        unrecorded_accepted_step=None
        if pending is None
        else dict(
            input_state=pending[1]["input_state"],
            reported_output_state=pending[1]["output_state"],
            transition_verified=False,
            control_force_goal_evidence_verified=False,
        ),
        reported_final_state=summary["final_state"],
        planning=plans,
        actual=metrics,
        termination=dict(
            status=summary["status"],
            failure_stage=summary.get("failure_stage"),
            errors=summary.get("errors"),
            failed_candidate=summary.get("failed_candidate"),
            checkpoint_save_error=summary.get("checkpoint_save_error"),
            release_status=summary.get("release_status"),
        ),
        wrench_scope="actual saved input-geometry force/COM-moment; ownership and phase checked, point moments not recomputed",
    )


def compare(path):
    path = Path(path).resolve()
    summary = _read(path)
    _require(
        summary["schema"] == "phymeo-pusht-pose-replan-v1" and set(summary["cases"]) == {"A", "B"},
        "two independent pose cases",
    )
    policies = {item["summary"]["selection_policy"] for item in summary["cases"].values()}
    _require(
        len(policies) == 1 and policies <= {"joint-pose-descent", "joint-pose-balanced"},
        "same fixed pose selection policy",
    )
    cases = {}
    for key, item in summary["cases"].items():
        case_path, saved = _bound(path.parent, item["file"], item["sha256"], "case " + key)
        _require(item["summary"] == saved, "inline case summary binding")
        cases[key] = compare_case(case_path)
        for row in cases[key]["actual"]["trajectory"]:
            row["case"] = key
    return _comparison(cases, dict(file=str(path), sha256=_digest(path)))


def _comparison(cases, source):
    return dict(
        schema="phymeo-pusht-pose-replan-comparison-v1",
        status="verified-saved-prefixes",
        source=source,
        cases=cases,
        interpretation="Separate initial conditions, not a causal orientation-only A/B experiment. Compare each case to its own input.",
        limits=[
            "A inherits terminal velocities, load and controller/Goal history; B resets robot, velocity, height and pure yaw.",
            "Only selected steps with complete callbacks contribute motion/load/Goal metrics; restored candidates are separate.",
            "Phase and stage contributions sum actual accepted endpoint increments, not planned path lengths.",
            "Planner values are heuristic predictions; measured output pose and saved Goal are reported independently.",
            "Zero recorded steps have no measured force peak, even if their inherited starting state was loaded.",
        ],
    )


def compare_cases(case_a, case_b):
    """Audit separately saved runs under one source and fixed A/B protocol.

    Each case still passes the full existing saved-evidence reader. A retry does
    not replace or hide the other case's retained prefix or failure diagnostics.
    """
    paths = {"A": Path(case_a).resolve(), "B": Path(case_b).resolve()}
    reports = {key: _read(path) for key, path in paths.items()}
    for key, mode in (("A", "terminal-checkpoint"), ("B", "independent-pose-reset")):
        report = reports[key]
        _require(
            report["schema"] == "phymeo-pusht-pose-replan-case-v1"
            and report["key"] == key
            and report["initialization_mode"] == mode
            and report["provenance"]["initialization_mode"] == mode,
            "fixed split-case initialization identity " + key,
        )
    a, b = reports["A"], reports["B"]
    common = {}
    for field in ("source_report_sha256", "source_checkpoint_sha256", "scene_asset_sha256"):
        _require(a["provenance"][field] == b["provenance"][field], "same split-case " + field)
        common[field] = a["provenance"][field]
    _require(a["goal"] == b["goal"], "same split-case original Goal")
    _require(
        a["selection_policy"] == b["selection_policy"]
        and a["selection_policy"] in ("joint-pose-descent", "joint-pose-balanced"),
        "same fixed pose selection policy",
    )
    _require(
        a["control_variant"] == b["control_variant"] == "load-compensation-half",
        "same split-case controller protocol",
    )
    _require(a["step_budget"] == b["step_budget"], "same split-case physical attempt budget")
    cases = {}
    for key, path in paths.items():
        cases[key] = compare_case(path)
        for row in cases[key]["actual"]["trajectory"]:
            row["case"] = key
    return _comparison(
        cases,
        dict(
            mode="independent-case-reports",
            common_source=common,
            cases={key: dict(file=str(path), sha256=_digest(path)) for key, path in paths.items()},
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--case-report", type=Path)
    parser.add_argument("--case-a", type=Path)
    parser.add_argument("--case-b", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.case_report is not None:
        if args.input is not None or args.case_a is not None or args.case_b is not None:
            parser.error("--case-report requires a single source")
        result = compare_case(args.case_report)
    elif args.input is not None:
        if args.case_a is not None or args.case_b is not None:
            parser.error("--input cannot be combined with --case-a or --case-b")
        result = compare(args.input)
    else:
        if args.case_a is None or args.case_b is None:
            parser.error("provide --case-report, --input, or both --case-a and --case-b")
        result = compare_cases(args.case_a, args.case_b)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
