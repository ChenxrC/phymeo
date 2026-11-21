"""Audit a saved active-route continuation from an audited terminal Scene capsule.

python -m experiments.area_pressure.pusht_resume_audit --input RESUMED.json --output AUDIT.json

Only the source adapter is new. Existing active-route reference, PD, independent
force-feedback Jacobian, receipt, integration, Goal and load checks are reused.
Checkpoint bytes are hash-bound, not restored by this saved-data reader.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .pusht_active_comparison import (
    H,
    _array,
    _bound_file,
    _close,
    _constant,
    _control,
    _digest,
    _final_checkpoint,
    _goal,
    _integration,
    _load,
    _LoadCompensationAudit,
    _pose_errors,
    _read,
    _receipt,
    _require,
    _state,
)

AUDIT_SCHEMAS = ("phymeo-pusht-active-comparison-v1", "phymeo-pusht-resume-audit-v1")


def bind_resume_source(summary, parent, parent_audit, metadata, seed):
    """Bind the true immediately preceding accepted load, never a failed trial."""
    _require(
        summary["schema"] == parent["schema"] == "phymeo-pusht-active-route-v1",
        "resume/parent summary schema",
    )
    variant = summary.get("control_variant", "position")
    _require(
        variant in ("position", "load-compensation-half")
        and parent.get("control_variant", "position") == variant
        and parent_audit["control"].get("variant", "position") == variant
        and metadata.get("control_variant", variant) == variant,
        "same resumed control variant",
    )
    count = parent["recorded_steps"]
    _require(
        type(count) is int
        and count > 0
        and parent["accepted_steps"]
        == parent_audit["verified_steps"]
        == parent_audit["accepted_steps"]
        == count
        and parent_audit["status"] == "verified-saved-prefix"
        and parent_audit["source_endpoint_bound"] is True
        and parent_audit.get("unrecorded_accepted_step") is None,
        "complete independently verified parent prefix, with no unrecorded accepted tail",
    )
    _require(
        parent_audit["control"].get("load_compensation_verified_steps", 0)
        == (count if variant == "load-compensation-half" else 0),
        "parent feedback audit count",
    )
    state, start = summary["initial_state"], summary["start_tick"]
    _require(
        state == parent["final_state"] == parent_audit["verified_final_state"] == metadata["state"]
        and summary["world"] == parent["world"] == metadata["world"]
        and summary["local_tick_origin"] == state["local_tick"] == metadata["local_tick"]
        and state["local_tick"] == parent["local_tick_origin"] + count
        and start == parent["start_tick"] + count == metadata["source_tick"]
        and start == parent["controller_final"]["tick"]
        and parent["controller_final"]["status"] == "continue",
        "resume exact accepted endpoint and controller clock",
    )
    _require(
        metadata["schema"] == "phymeo-pusht-active-terminal-checkpoint-v1"
        and metadata["q"] == state["q"]
        and metadata["v"] == state["v"]
        and metadata["time"] == state["time"]
        and metadata["controller_at_physical_endpoint"] is True
        and metadata["physical_state_matches_report"] is True
        and metadata["controller"] == parent["controller_final"]
        and metadata["goal_history"]
        == parent["goal_history_final"]
        == seed["goal_result"]["history"]
        and metadata["journal"] == parent["journal"]
        and metadata["accepted_steps"] == metadata["recorded_steps"] == count,
        "terminal capsule state/controller/Goal/publication metadata",
    )
    digest = metadata["checkpoint_sha256"]
    _require(
        digest == parent["final_checkpoint"]["sha256"] == parent_audit["final_checkpoint"]["sha256"]
        and parent_audit["final_checkpoint"]["capsule_bytes_verified"] is True,
        "independently audited terminal capsule identity",
    )
    for name in ("source_sha256", "scene_asset_sha256"):
        _require(
            summary["provenance"][name] == parent["provenance"][name],
            "resume physical source " + name,
        )
    _require(
        metadata["asset_sha256"] == summary["provenance"]["scene_asset_sha256"], "same material"
    )
    _require(summary["goal"] == parent["goal"] == metadata["target"], "unchanged goal")
    route = summary["active_route"]
    for name in (
        "start_tick",
        "end_tick",
        "duration_s",
        "route_index",
        "completed_routes_before",
        "sha256",
        "nodes",
    ):
        _require(
            route[name] == parent["active_route"][name] == metadata["active_route"][name],
            "same active route " + name,
        )
    _require(
        route["start_tick"] <= start < route["end_tick"]
        and summary["requested_steps"] == route["remaining_steps"] == route["end_tick"] - start,
        "only the remaining original route interval",
    )
    _close(
        route["elapsed_at_entry_s"], (start - route["start_tick"]) * H, "resumed reference phase"
    )
    _require(
        seed["original_input_tick"] == start - 1
        and seed["original_output_tick"] == start
        and seed["output_state"] == state
        and seed["input_state"]["local_tick"] + 1 == state["local_tick"],
        "immediately preceding committed source step",
    )
    measurement = seed["feedback_accepted"]["measurement"]
    _require(
        measurement["world"] == summary["world"]
        and measurement["input_tick"] == start - 1
        and measurement["output_tick"] == start
        and measurement["time_s"] == seed["input_state"]["time"]
        and measurement["phase"] == "accepted-step-input",
        "previous force belongs to previous input geometry",
    )
    force = _array(seed["area"]["total_force_on_pusher_world_n"], (3,), "preceding tool force")
    other = _array(measurement["group_force_world"], (3,), "preceding object force")
    _require(np.array_equal(force, -other), "preceding action/reaction force ownership")
    proof = summary["checkpoint_restore"]
    _require(
        proof["checkpoint_sha256"] == digest
        and proof["restored_local_tick"] == state["local_tick"]
        and proof["absolute_input_tick"] == start
        and proof["time"] == state["time"],
        "reported terminal restore identity",
    )
    chain_steps = parent_audit.get("combined_verified_steps", count)
    _require(type(chain_steps) is int and chain_steps >= count, "parent audited chain length")
    return dict(
        source_tick=start,
        control_variant=variant,
        parent_verified_steps=count,
        verified_chain_steps=chain_steps,
        previous_input_tick=start - 1,
        previous_force_on_tool_world_n=force.tolist(),
        checkpoint_sha256=digest,
        previous_force_phase="preceding-accepted-step-input-geometry",
    )


def verify_first_retry_command(summary, parent, command):
    """Check actual saved channels, not just the producer's matching assertion."""
    saved = parent.get("failed_candidate", {}).get("submitted_control")
    marker = summary.get("first_command_binding", {}).get("matches_parent_rejected_command")
    if saved is None:
        _require(marker is None, "retry match claimed without a parent rejected command")
        return dict(verified=False, reason="parent has no saved rejected command")
    start = summary["start_tick"]
    _require(
        marker is True and parent.get("failed_original_input_tick") == start,
        "first retry saved binding flag/source failure tick",
    )
    _require(
        type(command.get("tick")) is int
        and type(command.get("output_tick")) is int
        and command["tick"] == saved.get("tick") == start
        and command["output_tick"] == saved.get("output_tick") == start + 1,
        "first retry input/output tick",
    )
    channels = ("ctrl", "qfrc_applied", "xfrc_applied")
    for name in channels:
        expected = np.asarray(saved[name], dtype=np.float64)
        actual = np.asarray(command[name], dtype=np.float64)
        _require(
            actual.shape == expected.shape
            and np.isfinite(actual).all()
            and np.isfinite(expected).all()
            and np.array_equal(actual, expected)
            and actual.astype(np.float32).tobytes() == expected.astype(np.float32).tobytes(),
            "first retry differs from saved rejected command: " + name,
        )
    return dict(
        verified=True,
        input_tick=start,
        output_tick=start + 1,
        channels=list(channels),
        producer_match_flag_confirmed=True,
        fp32_words_equal=True,
        scope="saved command identity only; accepted transition verification is counted separately",
    )


def verify_resume_tail(summary, verified_endpoint):
    count, accepted = summary["recorded_steps"], summary["accepted_steps"]
    _require(
        type(count) is int
        and type(accepted) is int
        and 0 <= count <= accepted <= summary["requested_steps"]
        and accepted - count in (0, 1),
        "resumed accepted/recorded count",
    )
    if accepted == count:
        _require(summary["final_state"] == verified_endpoint, "retained resumed endpoint")
        return None
    _require(
        summary["status"] == "partial_failure"
        and summary["failure_stage"]
        in ("accepted_physical_record", "accepted_controller_and_goal", "journal_append"),
        "unrecorded physical acceptance requires a post-accept failure",
    )
    tail = summary["unrecorded_accepted_step"]
    tick = summary["start_tick"] + count
    _require(
        tail["input_state"] == verified_endpoint
        and tail["output_state"] == summary["final_state"]
        and tail["original_input_tick"] == tick
        and tail["original_output_tick"] == tick + 1,
        "unrecorded resumed step identity",
    )
    _state(tail["output_state"], summary["local_tick_origin"] + accepted)
    _require(tail["output_state"]["time"] > verified_endpoint["time"], "unrecorded output clock")
    return dict(
        input_tick=tick,
        output_tick=tick + 1,
        transition_verified=False,
        control_force_goal_evidence_verified=False,
        reported_output_state=tail["output_state"],
    )


def _source_files(path, summary):
    base, provenance = path.parent, summary["provenance"]
    parent_path, parent = _bound_file(
        base, provenance["parent_report_path"], provenance["parent_report_sha256"], "parent report"
    )
    audit_path, audit = _bound_file(
        base, provenance["parent_audit_path"], provenance["parent_audit_sha256"], "parent audit"
    )
    _require(
        audit["schema"] in AUDIT_SCHEMAS
        and audit["source"]["summary_sha256"] == _digest(parent_path)
        and audit["source"]["journal_sha256"] == parent["journal"]["sha256"],
        "independent parent audit source",
    )
    metadata_path, metadata = _bound_file(
        base,
        provenance["checkpoint_metadata_path"],
        provenance["checkpoint_metadata_sha256"],
        "resume capsule metadata",
    )
    _require(
        metadata_path
        == (parent_path.parent / parent["final_checkpoint"]["metadata_file"]).resolve()
        and provenance["checkpoint_metadata_sha256"]
        == parent["final_checkpoint"]["metadata_sha256"]
        and (metadata_path.parent / metadata["source_report"]).resolve() == parent_path,
        "parent terminal metadata/source report path",
    )
    _final_checkpoint(parent_path.parent, parent)
    _require(
        (metadata_path.parent / metadata["checkpoint_file"]).resolve()
        == (parent_path.parent / parent["final_checkpoint"]["file"]).resolve(),
        "terminal capsule path",
    )
    journal_path = (parent_path.parent / parent["journal"]["file"]).resolve()
    _require(_digest(journal_path) == parent["journal"]["sha256"], "parent journal bytes")
    seed, count = None, 0
    with journal_path.open("rt", encoding="utf-8-sig") as stream:
        for count, line in enumerate(stream, 1):
            seed = json.loads(line, parse_constant=_constant)
    _require(
        seed is not None and count == parent["recorded_steps"] == parent["journal"]["records"],
        "parent last recorded force source",
    )
    _receipt(seed)
    _load(seed)
    binding = bind_resume_source(summary, parent, audit, metadata, seed)
    _require(summary["controller_initial"] == metadata["controller"], "resumed controller history")
    return (
        parent,
        seed,
        binding,
        dict(
            parent_report_file=str(parent_path),
            parent_report_sha256=_digest(parent_path),
            parent_audit_file=str(audit_path),
            parent_audit_sha256=_digest(audit_path),
            checkpoint_metadata_file=str(metadata_path),
            checkpoint_metadata_sha256=_digest(metadata_path),
        ),
    )


def compare(path):
    path = Path(path).resolve()
    summary, base = _read(path), path.parent
    parent, seed, binding, sources = _source_files(path, summary)
    start, origin = summary["start_tick"], summary["local_tick_origin"]
    initial, route, goal = summary["initial_state"], summary["active_route"], summary["goal"]
    _state(initial, origin)
    _, route_file = _bound_file(base, route["file"], route["sha256"], "resumed active route")
    _require(
        route_file["tick"] == route["start_tick"]
        and route_file["route"]["nodes"] == route["nodes"]
        and route_file["route"]["status"] == "sampled_reference_ready",
        "unchanged admitted reference",
    )
    _close(route["duration_s"], route["nodes"][-1][0], "route duration")
    _close(route["duration_s"], (route["end_tick"] - route["start_tick"]) * H, "route end")
    # Adapt the proven accepted predecessor to the existing audit's seed shape.
    # This is not a newly invented release report or a claim of separation.
    variant = binding["control_variant"]
    seed_view = dict(seed, selected=True, restored=False)
    compensation = (
        _LoadCompensationAudit(summary, {"trials": [seed_view]}, route_file)
        if variant == "load-compensation-half"
        else None
    )
    count, accepted = summary["recorded_steps"], summary["accepted_steps"]
    _require(
        type(count) is int
        and type(accepted) is int
        and 0 <= count <= accepted
        and accepted - count in (0, 1)
        and accepted <= summary["requested_steps"],
        "resumed counts",
    )
    journal = summary["journal"]
    journal_path = (base / journal["file"]).resolve()
    _require(
        journal["records"] == count and _digest(journal_path) == journal["sha256"],
        "resumed journal digest/count",
    )
    prior, records, max_q, max_clock, peak, impulse, support_min = (
        initial,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        None,
    )
    first_loaded = peak_tick = None
    success_ticks, stages = [], []
    first_retry = None
    with journal_path.open("rt", encoding="utf-8-sig") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line, parse_constant=_constant)
            tick = start + index
            _require(
                index < count
                and row["schema"] == "phymeo-pusht-active-route-step-v1"
                and row["index"] == index
                and row["original_input_tick"] == tick
                and row["original_output_tick"] == tick + 1,
                "resumed journal order",
            )
            before, after = row["input_state"], row["output_state"]
            _state(before, origin + index)
            _state(after, origin + index + 1)
            _require(before == prior, "continuous resumed accepted states")
            qe, te = _integration(before, after)
            max_q, max_clock = max(max_q, qe), max(max_clock, te)
            if index == 0:
                first_retry = verify_first_retry_command(summary, parent, row["submitted_control"])
            stage = _control(row, summary, index, base, compensation)
            if not stages or stages[-1]["stage"] != stage:
                stages.append(dict(input_tick=tick, stage=stage))
            if _goal(row, summary, tick):
                success_ticks.append(tick)
            _receipt(row)
            load, support = _load(row)
            # Explicit bilateral equality at the seam and every resumed step.
            _require(
                np.array_equal(
                    np.asarray(row["area"]["total_force_on_pusher_world_n"]),
                    -np.asarray(row["feedback_accepted"]["measurement"]["group_force_world"]),
                ),
                "resumed action/reaction",
            )
            if compensation is not None:
                compensation.accepted(row)
            support_min = support if support_min is None else min(support_min, support)
            impulse += H * load
            if load > peak:
                peak, peak_tick = load, tick
            if load > 0 and first_loaded is None:
                first_loaded = tick
            prior, records = after, records + 1
    _require(records == count, "truncated resumed journal")
    tail = verify_resume_tail(summary, prior)
    if count == 0:
        packet = summary.get("failed_candidate", {}).get("submitted_control")
        if packet is None:
            packet = summary.get("unrecorded_accepted_step", {}).get("submitted_control")
        first_retry = (
            verify_first_retry_command(summary, parent, packet)
            if packet is not None
            else dict(verified=False, reason="no resumed command record")
        )
    status = summary["status"]
    if status == "route_complete":
        _require(
            count == accepted == summary["requested_steps"]
            and summary["controller_final"]["tick"] == route["end_tick"]
            and summary["controller_final"]["completed_routes"]
            == route["completed_routes_before"] + 1,
            "original route completion",
        )
        classification = (
            "route_complete_with_goal" if success_ticks else "route_complete_without_goal"
        )
    elif status == "controller_stopped":
        _require(
            summary["controller_status"] != "success" or bool(success_ticks), "saved Goal success"
        )
        classification = "controller_stopped"
    elif status == "partial_failure":
        classification = "post_accept_callback_failure" if tail else "physical_or_reference_failure"
    else:
        _require(status == "running", "resumed termination status")
        classification = "in_progress"
    return dict(
        schema="phymeo-pusht-resume-audit-v1",
        status="verified-saved-prefix",
        source_endpoint_bound=True,
        source_checkpoint_sha256=binding["checkpoint_sha256"],
        source=dict(
            summary_file=str(path),
            summary_sha256=_digest(path),
            journal_sha256=journal["sha256"],
            **sources,
        ),
        parent_binding=binding,
        parent_verified_steps=binding["parent_verified_steps"],
        verified_steps=count,
        accepted_steps=accepted,
        combined_verified_steps=binding["verified_chain_steps"] + count,
        verified_final_state=prior,
        unrecorded_accepted_step=tail,
        final_checkpoint=_final_checkpoint(base, summary),
        first_retry_command_binding=first_retry,
        control=dict(
            variant=variant,
            load_compensation_verified_steps=count if compensation is not None else 0,
            reference_phase_verified_steps=count,
            current_state_bound_steps=count,
            transport="saved CUDA output/publication binding; no substitute tanh",
        ),
        integration=dict(
            verified_steps=count, max_q_substitution_error=max_q, max_clock_step_error_s=max_clock
        ),
        termination=dict(
            classification=classification,
            producer_status=status,
            failure_stage=summary.get("failure_stage"),
            controller_status=summary.get("controller_status"),
            failed_original_input_tick=summary.get("failed_original_input_tick"),
            errors=summary.get("errors"),
        ),
        planning=dict(
            selected_route_index=route["route_index"],
            route_start_tick=route["start_tick"],
            route_end_tick=route["end_tick"],
            stages=stages,
        ),
        contact=dict(
            first_positive_input_tick=first_loaded,
            peak_normal_load_n=peak,
            peak_input_tick=peak_tick,
            normal_impulse_ns=impulse,
            minimum_object_support_n=support_min,
        ),
        object=dict(initial=_pose_errors(initial, goal), final=_pose_errors(prior, goal)),
        goal=dict(
            ever_success=bool(success_ticks),
            success_input_ticks=success_ticks,
            last_input_tick=start + count - 1 if count else None,
        ),
        limits=[
            "The terminal source and previous accepted load are bound to the independently audited parent prefix.",
            "Only recorded accepted transitions are verified; any unrecorded accepted tail is excluded.",
            "The previous force belongs to its preceding input geometry, not the resumed endpoint geometry.",
            "Capsule bytes are bound but are not restored or integrated by this reader.",
            "Saved integration/force/Goal checks reuse the original reader; endpoint errors do not reevaluate Goal.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.input)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
