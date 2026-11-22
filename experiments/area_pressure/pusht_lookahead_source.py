"""Bind a fair lookahead entry to one already audited ordinary-plan boundary.

This reader performs file/JSON/clock/identity checks only. It does not compile a
model, restore a driver, normalize q, rerun the audit, plan, or advance physics.
The old independent audit remains evidence, not a newly reproduced experiment.
Only ``entry`` may initialize a candidate. ``baseline`` contains future measured
outcomes for comparison and must never feed candidate control or initialization.

load_lookahead_source(report_path, audit_path, *, route_path=None) returns owned,
JSON-ready data. The full proposal archive and controller replay prefix are
referenced by their original files and hashes instead of duplicating a large
audit. The candidate query hash uses sorted compact ASCII JSON, not a model hash.
"""

import gzip
import hashlib
import json
import math
import struct
from copy import deepcopy
from pathlib import Path

REPORT = "phymeo-pusht-lateral-replan-v1"
AUDIT = "phymeo-pusht-lateral-replan-audit-v1"
EVENT = "phymeo-pusht-lateral-replan-event-v1"


def _require(ok, message):
    if not ok:
        raise ValueError("lookahead source: " + message)


def _constant(value):
    raise ValueError("lookahead source: nonfinite JSON constant " + value)


def _read(path):
    path = Path(path)
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf-8-sig") as stream:
        return json.load(stream, parse_constant=_constant)


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _hash(value):
    _require(
        type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        "SHA256 encoding",
    )
    return value


def _bound(base, name, sha, label):
    path = (base / name).resolve()
    _require(_digest(path) == _hash(sha), label + " SHA256")
    return path, _read(path)


def _source(path):
    return dict(path=str(path), sha256=_digest(path))


def _canonical_sha(value):
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _state(value):
    _require(
        type(value.get("local_tick")) is int
        and value["local_tick"] >= 0
        and _number(value["time"]),
        "saved state clock",
    )
    for key, width in (("q", 9), ("v", 8)):
        values = value[key]
        _require(
            type(values) is list and len(values) == width and all(_number(v) for v in values),
            "finite original RR " + key,
        )
        try:
            exact = all(struct.unpack("<f", struct.pack("<f", v))[0] == v for v in values)
        except (OverflowError, struct.error):
            exact = False
        _require(exact, "stored FP32 words: " + key)
    _require(
        value["q"][2:5] == value["object_position_world_m"]
        and value["q"][5:9] == value["object_quaternion_wxyz"],
        "saved pose aliases",
    )


def _counts(value):
    result = {key: value[key] for key in ("ik_calls", "path_calls")}
    _require(all(type(v) is int and v >= 0 for v in result.values()), "planner counters")
    return result


def _scan(journal, summary, audit):
    """Keep one predecessor and one first regular record; stream the whole chain."""
    trajectory = audit["actual"]["trajectory"]
    retained = summary["initial_state"]
    _state(retained)
    pending = previous = first_regular = entry_previous = planning_state = None
    count = records = attempts = attempts_before = callbacks_before = 0
    initialized = planned = ended = False
    sha = hashlib.sha256()
    origin = summary["source_reference_tick_origin"]
    _require(
        summary["start_tick"] == origin + retained["local_tick"]
        and summary["local_tick_origin"] == retained["local_tick"],
        "initial absolute/local clocks",
    )
    with journal.open("rb") as stream:
        for raw in stream:
            _require(raw.endswith(b"\n") and not ended, "complete ordered journal lines")
            sha.update(raw)
            records += 1
            event = json.loads(raw, parse_constant=_constant)
            _require(event["schema"] == EVENT, "journal event schema")
            kind, row, index = event["event"], event["payload"], event.get("trial_index")
            if kind == "initialization":
                _require(
                    not initialized
                    and records == 1
                    and all(
                        row.get(k) == summary.get(k)
                        for k in (
                            "initial_state",
                            "goal",
                            "provenance",
                            "controller_initial",
                            "goal_history_initial",
                        )
                    ),
                    "initialization source",
                )
                initialized = True
            elif kind == "planner":
                _require(
                    initialized
                    and not planned
                    and pending is None
                    and previous is not None
                    and row == summary["ordinary_route"]
                    and row["tick"] == origin + retained["local_tick"]
                    and row["source_q"] == retained["q"]
                    and row["source_v"] == retained["v"],
                    "ordinary planner accepted boundary",
                )
                planning_state, entry_previous = deepcopy(retained), deepcopy(previous)
                attempts_before, callbacks_before, planned = attempts, count, True
            elif kind == "candidate":
                _require(
                    initialized
                    and pending is None
                    and type(index) is int
                    and index == attempts
                    and row["trial_index"] == row["index"] == index
                    and row["input_state"] == retained,
                    "candidate index and retained q/v/time",
                )
                after = row["output_state"]
                _state(after)
                _require(
                    after["local_tick"] == retained["local_tick"] + 1
                    and after["time"] > retained["time"]
                    and row["original_input_tick"] == origin + retained["local_tick"]
                    and row["original_output_tick"] == row["original_input_tick"] + 1,
                    "candidate adjacent clocks",
                )
                _require(
                    row["phase"] == ("regular" if planned else "withdraw")
                    and type(row["selected"]) is bool
                    and row["restored"] is not row["selected"],
                    "candidate execution phase",
                )
                attempts += 1
                if row["selected"]:
                    pending = (index, row)
                else:
                    _require(
                        row.get("rollback_verification", {}).get("passed") is True,
                        "rejected candidate rollback",
                    )
            elif kind == "accepted_callback":
                _require(
                    pending is not None
                    and index == pending[0]
                    and not set(row).intersection(pending[1])
                    and count < len(trajectory),
                    "complete unique callback",
                )
                candidate = pending[1]
                before, after = candidate["input_state"], candidate["output_state"]
                measurement = row["feedback_accepted"]["measurement"]
                history = row["goal_result"]["history"]
                _require(
                    measurement == candidate["measurement"]
                    and measurement["world"] == summary["world"]
                    and measurement["phase"] == "accepted-step-input"
                    and measurement["input_tick"] == candidate["original_input_tick"]
                    and measurement["output_tick"] == candidate["original_output_tick"]
                    and measurement["time_s"] == before["time"],
                    "preceding actual force phase",
                )
                _require(
                    history["tick"] == before["local_tick"]
                    and history["time_s"] == before["time"]
                    and row["driver_state_after"]["tick"] == candidate["original_output_tick"],
                    "Goal/controller clock",
                )
                sample = trajectory[count]
                expected = dict(
                    phase=candidate["phase"],
                    stage=candidate["controller"]["stage"],
                    input_tick=candidate["original_input_tick"],
                    output_tick=candidate["original_output_tick"],
                    input_time_s=before["time"],
                    output_time_s=after["time"],
                    input_xy_m=before["q"][2:4],
                    output_xy_m=after["q"][2:4],
                    normal_load_n=candidate["area"]["normal_load_n"],
                )
                _require(
                    all(sample[k] == v for k, v in expected.items()),
                    "audit trajectory and journal sample",
                )
                previous = dict(candidate, **row)
                if planned and first_regular is None:
                    first_regular = deepcopy(previous)
                retained, pending, count = after, None, count + 1
            elif kind == "release_reference":
                _require(
                    not planned
                    and pending is None
                    and row["reference"] == summary["release_reference"],
                    "release source event",
                )
            elif kind == "planner_probe":
                _require(
                    not planned and pending is None and row == summary.get("entry_probe"),
                    "entry gate source event",
                )
            elif kind == "final":
                _require(
                    pending is None and row == {k: v for k, v in summary.items() if k != "journal"},
                    "final journal summary",
                )
                ended = True
            else:
                raise ValueError("lookahead source: unsupported incomplete source event " + kind)
    _require(
        ended
        and planned
        and entry_previous is not None
        and first_regular is not None
        and records == summary["journal"]["records"]
        and sha.hexdigest() == summary["journal"]["sha256"],
        "complete source stream/hash",
    )
    _require(
        count == summary["recorded_steps"] == audit["actual"]["verified_steps"] == len(trajectory)
        and attempts == summary["physical_attempts"]
        and retained == summary["final_state"] == audit["actual"]["verified_final_state"],
        "audited complete source endpoint/counts",
    )
    _require(
        first_regular["input_state"] == planning_state == entry_previous["output_state"]
        and entry_previous["phase"] == "withdraw"
        and entry_previous["driver_state_after"]["partial_route"] is None,
        "separated pre-plan entry, not a future baseline state",
    )
    return dict(
        state=planning_state,
        previous=entry_previous,
        first=first_regular,
        last=previous,
        attempts_before=attempts_before,
        callbacks_before=callbacks_before,
    )


def load_lookahead_source(report_path, audit_path, *, route_path=None):
    """Return source-bound entry, archived baseline and the original fair budget.

    No source values are normalized. Planner counts before planning come from
    the completed release request, never from the ordinary route's after-counts.
    Driver construction uses the original prior-summary candidate-world selector,
    which is distinct from the physical Scene world.
    """
    report_path, audit_path = Path(report_path).resolve(), Path(audit_path).resolve()
    summary, audit = _read(report_path), _read(audit_path)
    _require(
        summary["schema"] == REPORT
        and audit["schema"] == AUDIT
        and audit["source_sha256"] == _digest(report_path)
        and audit["journal_sha256"] == summary["journal"]["sha256"],
        "report/audit SHA256 binding",
    )
    _require(
        summary["world"] == 6
        and summary["side_coverage"] == "all-lateral"
        and summary["publication_frame"] == "rigid-origin"
        and summary["control_variant"] == "load-compensation-half"
        and summary["goal_tick_origin"] == 0,
        "declared original scene/clock/material",
    )
    _require(
        summary["status"] in ("step_budget", "route_complete")
        and audit["status"] == summary["status"]
        and summary["accepted_steps"] == summary["recorded_steps"]
        and not summary.get("unrecorded_accepted_step")
        and not audit.get("unrecorded_accepted_step")
        and audit["physical_final_state"] == summary["final_state"]
        and audit["goal"] == summary["goal"],
        "complete baseline, original Goal and physical endpoint",
    )
    base, p = report_path.parent, summary["provenance"]
    sources = dict(report=_source(report_path), audit=_source(audit_path))
    parents = {}
    for label in ("parent_report", "parent_audit", "checkpoint_metadata", "prior_summary"):
        path, value = _bound(base, p[label + "_path"], p[label + "_sha256"], label)
        sources[label], parents[label] = _source(path), value
    parent, parent_audit, meta = (
        parents[k] for k in ("parent_report", "parent_audit", "checkpoint_metadata")
    )
    _require(
        parent_audit["source_sha256"] == sources["parent_report"]["sha256"]
        and parent_audit["physical_final_state"]
        == parent["final_state"]
        == summary["initial_state"]
        == meta["state"]
        and parent["goal"] == summary["goal"] == meta["target"]
        and parent["controller_final"] == summary["controller_initial"] == meta["controller"]
        and parent["goal_history_final"] == summary["goal_history_initial"] == meta["goal_history"]
        and parent["status"] == "route_complete"
        and meta["source_tick"] == summary["start_tick"]
        and meta["asset_sha256"] == p["scene_asset_sha256"],
        "completed parent state/history/model binding",
    )
    prior = parents["prior_summary"]
    original = prior["provenance"]
    candidate_path, dataset = _bound(
        Path(sources["prior_summary"]["path"]).parent,
        original["candidate_source"],
        original["candidate_source_sha256"],
        "original candidate dataset",
    )
    candidate_world = prior["worlds"]["open"]
    _require(
        type(candidate_world) is int and original["source_sha256"] == p["source_model_sha256"],
        "driver candidate selector/source model",
    )
    candidates = [r for r in dataset["rows"] if r["summary"]["world"] == candidate_world]
    _require(
        dataset["schema"] == "original-push-boundary-endpoints-v1" and len(candidates) == 1,
        "unique original driver candidate world",
    )
    saved = candidates[0]
    model_sha = _hash(saved["summary"]["model_sha256"])
    _hash(p["source_model_sha256"])
    _hash(p["scene_asset_sha256"])
    query_sha = _canonical_sha(saved["query"])
    sources["candidate"] = _source(candidate_path)
    journal = (base / summary["journal"]["file"]).resolve()
    bound = _scan(journal, summary, audit)
    sources["journal"] = _source(journal)
    state, previous = bound["state"], bound["previous"]
    plan, active = summary["ordinary_route"], summary["active_route"]
    declared_route = (base / plan["route_file"]).resolve()
    chosen = declared_route if route_path is None else Path(route_path).resolve()
    _require(
        _digest(chosen) == _digest(declared_route) == plan["route_sha256"] == active["sha256"],
        "ordinary route SHA256",
    )
    route = _read(chosen)
    raw, proposed = route["route"], route["route"]["proposal"]
    _require(
        route["tick"] == plan["tick"] == active["start_tick"]
        and route["q"] == raw["source_q"] == plan["source_q"] == state["q"]
        and route["v"] == plan["source_v"] == state["v"],
        "ordinary route input q/v/clock",
    )
    _require(
        raw["status"] == plan["status"] == "sampled_reference_ready"
        and proposed in route["proposals"]
        and all(v == 0 for v in raw["initial_reference_projection"])
        and raw["nodes"] == active["nodes"],
        "actual selected route and unchanged entry projection",
    )
    ordinary = audit["planning"]["ordinary"]
    _require(
        ordinary["input_tick"] == plan["tick"]
        and ordinary["status"] == raw["status"]
        and ordinary["route"] == raw
        and ordinary["proposals"] == route["proposals"],
        "independent audit selected route archive",
    )
    prediction = ordinary["heuristic_prediction"]
    _require(
        all(proposed[k] == v for k, v in prediction.items()),
        "predicted fields belong to selected proposal",
    )
    identities = [
        [r["index"], r["distance_ordinal"], r["input_binding"]] for r in route["proposals"]
    ]
    _require(
        len({tuple(i) for i in identities}) == len(identities)
        and all(type(i[0]) is int and type(i[1]) is int and type(i[2]) is str for i in identities),
        "unique proposal identity catalogue",
    )
    release = summary["release_reference"]
    release_path, released = _bound(
        base, release["file"], release["sha256"], "release planner request"
    )
    _require(
        released["tick"] == summary["start_tick"]
        and released["q"] == summary["initial_state"]["q"]
        and released["v"] == summary["initial_state"]["v"]
        and released["route"]["status"] == "sampled_release_ready"
        and released["route"]["nodes"] == release["nodes"],
        "before-plan cumulative release counters source",
    )
    before_counts, after_counts = _counts(released["route"]), _counts(raw)
    _require(
        all(
            summary["controller_initial"][k] <= before_counts[k] <= after_counts[k]
            for k in before_counts
        ),
        "planner counts before and after ordinary planning",
    )
    sources["release_route"], sources["ordinary_route"] = _source(release_path), _source(chosen)
    remaining = summary["step_budget"] - bound["attempts_before"]
    prior_attempts = summary["parent_physical_attempts"]
    _require(
        all(
            type(summary[k]) is int and summary[k] >= 0
            for k in ("step_budget", "original_step_budget", "parent_physical_attempts")
        )
        and prior_attempts == parent_audit["original_attempts_consumed"]
        and 0
        < remaining
        <= summary["original_step_budget"] - prior_attempts - bound["attempts_before"]
        and remaining == summary["reference_budget"]["available_steps"],
        "unchanged fair remaining physical-attempt budget",
    )
    duration = raw["nodes"][-1][0]
    steps = round(duration / 0.001)
    _require(
        _number(duration)
        and duration > 0
        and abs(steps * 0.001 - duration) <= 1e-12
        and steps
        == summary["reference_budget"]["route_steps"]
        == active["end_tick"] - plan["tick"],
        "original reference duration",
    )
    regular = [r for r in audit["actual"]["trajectory"] if r["phase"] == "regular"]
    comparison = audit["prediction_vs_actual"]
    _require(
        len(regular) == comparison["measured_regular_steps"]
        and regular[0]["input_tick"] == plan["tick"]
        and bound["first"]["input_state"] == state
        and bound["last"]["phase"] == "regular",
        "baseline regular-only measured interval",
    )
    feedback, boundary = previous["feedback_accepted"], previous["driver_state_after"]
    _require(
        boundary["status"] == "continue"
        and boundary["tick"] == plan["tick"]
        and boundary["total_contact_impulse_ns"] == feedback["total_impulse_ns"]
        and all(
            _number(feedback[k]) and feedback[k] >= 0
            for k in ("total_impulse_ns", "segment_impulse_ns")
        ),
        "actual controller impulses at entry",
    )
    driver_candidate = dict(
        **sources["candidate"],
        world=candidate_world,
        query_sha256=query_sha,
        query_input_binding=saved["query"].get("input_binding"),
        model_sha256=model_sha,
    )
    entry = dict(
        world=summary["world"],
        tick=plan["tick"],
        state=state,
        goal=summary["goal"],
        source_reference_tick_origin=summary["source_reference_tick_origin"],
        goal_tick_origin=summary["goal_tick_origin"],
        source_model_sha256=p["source_model_sha256"],
        model_sha256=model_sha,
        query_sha256=query_sha,
        scene_asset_sha256=p["scene_asset_sha256"],
        side_coverage=summary["side_coverage"],
        publication_frame=summary["publication_frame"],
        control_variant=summary["control_variant"],
        material_identity=bound["first"].get("material_identity"),
        driver_candidate=driver_candidate,
        prior_summary=sources["prior_summary"],
        planner_counts_before=before_counts,
        controller_seed=summary["controller_initial"],
        controller_boundary=boundary,
        goal_history=previous["goal_result"]["history"],
        segment_impulse_ns=feedback["segment_impulse_ns"],
        total_impulse_ns=feedback["total_impulse_ns"],
        previous_measurement=previous["measurement"],
        previous_area=previous["area"],
        previous_input_state=previous["input_state"],
        history_source=dict(
            journal=sources["journal"],
            parent_report=sources["parent_report"],
            parent_audit=sources["parent_audit"],
            checkpoint_metadata=sources["checkpoint_metadata"],
            callbacks_before_plan=bound["callbacks_before"],
            through_trial_index=previous["trial_index"],
            phase="accepted callbacks only; no physics replay, no future regular state",
        ),
    )
    baseline = dict(
        selected_index=proposed["index"],
        proposal=proposed,
        proposal_identities=identities,
        route_path=str(chosen),
        route_sha256=_digest(chosen),
        nodes=raw["nodes"],
        planner_counts_after=after_counts,
        prediction=prediction,
        comparison=comparison,
        measured_regular_steps=len(regular),
        initial_metrics={
            k: regular[0]["input_" + k] for k in ("position_error_m", "yaw_error_rad")
        },
        final_metrics={
            k: regular[-1]["output_" + k] for k in ("position_error_m", "yaw_error_rad")
        },
        regular_metrics=audit["actual"]["phases"]["regular"],
        final_state=summary["final_state"],
        last_goal_result=bound["last"]["goal_result"],
        first_command_binding=summary.get("first_regular_command_binding"),
        interpretation="archived future baseline outcomes for comparison only; never candidate state or control",
    )
    return deepcopy(
        dict(
            schema="phymeo-pusht-lookahead-source-v1",
            sources=sources,
            entry=entry,
            baseline=baseline,
            budget=dict(
                original_steps=summary["original_step_budget"],
                parent_attempts=prior_attempts,
                attempts_before_plan=bound["attempts_before"],
                remaining_steps=remaining,
                baseline_reference_steps=steps,
            ),
            scope="original saved evidence bindings only; no new physics, planner call, controller restore or q normalization",
        )
    )
