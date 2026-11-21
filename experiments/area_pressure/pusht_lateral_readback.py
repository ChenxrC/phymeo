"""Read back actual lateral-material PushT motion without replaying physics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .pusht_active_comparison import (
    _channels,
    _constant,
    _digest,
    _integration,
    _load,
    _read,
    _receipt,
    _require,
    _state,
)
from .pusht_pose_replan_comparison import (
    _goal_row,
    _planner,
    _wrench,
    retained_state,
    summarize_motion,
    validate_counts,
)

EVENT_SCHEMA = "phymeo-pusht-lateral-route-event-v1"
FAILURE_KEYS = ("failure_stage", "errors", "failed_candidate", "unrecorded_accepted_step")


def _rollback(value, world):
    _require(
        isinstance(value, dict)
        and value.get("passed") is True
        and value.get("worlds") == [world]
        and value.get("mismatches") == [],
        "rollback must verify the declared world without mismatches",
    )


def _command_phase(command, before, start):
    _channels(command)
    _require(
        command["tick"] == start + before["local_tick"]
        and command["output_tick"] == command["tick"] + 1,
        "submitted command phase",
    )


def _one_step(before, after):
    _state(before, before["local_tick"])
    _state(after, before["local_tick"] + 1)
    _integration(before, after)


class _EventBinding:
    """Bind every event, including discarded candidates, to one retained state."""

    def __init__(self, summary):
        self.summary = summary
        self.retained = summary.get("initial_state")
        self.world, self.start = summary["world"], summary["start_tick"]
        self.candidates = self.selected = self.callbacks = 0
        self.failed_attempts = self.failed_physical = 0
        self.failure = False
        self.initialized = self.planned = False
        self.last = None

    def check(self, kind, payload, index, pending):
        summary = self.summary
        if kind == "initialization":
            self.initialized = True
            frozen = (
                "schema",
                "world",
                "start_tick",
                "local_tick_origin",
                "initial_state",
                "goal",
                "side_coverage",
                "material_admission",
                "selection_policy",
                "initialization_mode",
                "provenance",
                "initialization_proof",
                "controller_initial",
                "controller_bootstrap",
                "initial_separation",
                "material_geometry_provenance",
                "source_reference_tick_origin",
                "goal_tick_origin",
                "goal_history_initial",
                "episode_boundary",
            )
            _require(
                index is None
                and self.retained is not None
                and all(payload.get(key) == summary.get(key) for key in frozen)
                and payload["final_state"] == self.retained
                and payload["status"] == "running"
                and all(
                    payload[key] == 0
                    for key in (
                        "physical_attempts",
                        "physical_trial_steps",
                        "accepted_steps",
                        "recorded_steps",
                        "rejected_trials",
                    )
                ),
                "initialization event must bind the fresh source state and policy",
            )
        elif kind == "planner":
            self.planned = True
            _require(
                index is None
                and payload == summary.get("ordinary_route")
                and payload["tick"] == self.start
                and payload["source_q"] == self.retained["q"]
                and payload["source_v"] == self.retained["v"],
                "planner event must bind the declared route and retained state",
            )
        elif kind == "candidate":
            _require(
                index == self.candidates == payload["index"] == payload["trial_index"],
                "candidate event/payload index",
            )
            _require(payload["input_state"] == self.retained, "candidate rollback prestate")
            _one_step(payload["input_state"], payload["output_state"])
            _command_phase(
                payload.get("submitted_control", payload.get("control")), self.retained, self.start
            )
            _require(
                payload["original_input_tick"] == self.start + self.retained["local_tick"]
                and payload["original_output_tick"] == payload["original_input_tick"] + 1
                and type(payload["selected"]) is bool
                and type(payload["restored"]) is bool
                and payload["assessment"]["safe"] is payload["selected"],
                "candidate input phase and selection",
            )
            if not payload["selected"]:
                _rollback(payload["rollback_verification"], self.world)
            self.retained = retained_state(payload, self.retained)
            self.candidates += 1
            self.selected += int(payload["selected"])
            self.last = payload
        elif kind == "accepted_callback":
            _require(
                pending is not None and not set(payload).intersection(pending[1]),
                "callback cannot overwrite its physical candidate",
            )
            self.callbacks += 1
        elif kind == "failure":
            _require(
                not self.failure
                and summary["status"] == "partial_failure"
                and payload == {key: summary.get(key) for key in FAILURE_KEYS}
                and isinstance(payload["failure_stage"], str)
                and payload["failure_stage"]
                and isinstance(payload["errors"], list)
                and payload["errors"],
                "failure event must bind the saved failure stage and evidence",
            )
            self.failure = True
            failed, unpublished = payload["failed_candidate"], payload["unrecorded_accepted_step"]
            _require(
                not (failed is not None and unpublished is not None),
                "one failure publication boundary",
            )
            if failed is not None:
                _require(
                    pending is None
                    and index == self.candidates
                    and failed["input_states"] == {str(self.world): self.retained}
                    and type(failed["physics_accepted_before_restore"]) is bool
                    and failed["restored"] is True,
                    "failed attempt retained input and restoration",
                )
                _rollback(failed["rollback_verification"], self.world)
                _command_phase(failed["control"], self.retained, self.start)
                outputs = failed["candidate_output_states"]
                if failed["physics_accepted_before_restore"]:
                    _require(
                        isinstance(outputs, dict) and set(outputs) == {str(self.world)},
                        "failed physical candidate has an explicit output",
                    )
                    _one_step(self.retained, outputs[str(self.world)])
                else:
                    _require(outputs is None, "unaccepted physics cannot carry an accepted output")
                self.failed_attempts = 1
                self.failed_physical = int(failed["physics_accepted_before_restore"])
            elif unpublished is not None:
                if pending is not None:
                    row = pending[1]
                    _require(
                        index == pending[0]
                        and unpublished
                        == dict(
                            input_state=row["input_state"],
                            output_state=row["output_state"],
                            control=row.get("submitted_control", row.get("control")),
                        ),
                        "unrecorded accepted step must be the pending physical candidate",
                    )
                else:
                    # The real writer may fail while recording a selected step,
                    # before it can emit the candidate event. Do not invent a callback.
                    _require(
                        index == self.candidates and unpublished["input_state"] == self.retained,
                        "unrecorded accepted step input",
                    )
                    _one_step(self.retained, unpublished["output_state"])
                    _command_phase(unpublished["control"], self.retained, self.start)
                    self.retained = unpublished["output_state"]
                    self.selected += 1
                    self.failed_attempts = self.failed_physical = 1
            else:
                _require(pending is None, "missing unrecorded accepted candidate evidence")
        elif kind == "final":
            _require(
                index is None
                and payload == {key: value for key, value in summary.items() if key != "journal"},
                "final event/summary identity",
            )
            _require(self.retained == summary.get("final_state"), "physical retained final state")
            # validate_counts expects every selected step to have a candidate;
            # an explicitly saved accepted tail can fail before that event.
            unrecorded_before_event = int(self.selected > self.candidates)
            validate_counts(
                summary,
                candidate_count=self.candidates + unrecorded_before_event,
                selected_count=self.selected,
                callback_count=self.callbacks,
                failed_attempts=self.failed_attempts - unrecorded_before_event,
                failed_physical=self.failed_physical - unrecorded_before_event,
            )
            _require(self.planned == ("ordinary_route" in summary), "saved planner event presence")
            _require(
                self.initialized
                or (
                    self.failure
                    and summary["failure_stage"]
                    in (
                        "prepare_new_episode",
                        "construct_new_scene",
                        "verify_new_initial_separation",
                    )
                ),
                "initialization event presence",
            )
            status = summary["status"]
            _require((status == "partial_failure") == self.failure, "failure/status classification")
            _require(
                status
                in (
                    "partial_failure",
                    "route_complete",
                    "step_budget",
                    "controller_stopped",
                    "planning_rejected",
                    "initial_force_check_rejected",
                ),
                "terminal status",
            )
            if status == "planning_rejected":
                _require(
                    self.candidates == 0
                    and summary["ordinary_route"]["status"] != "sampled_reference_ready",
                    "planning rejection cannot contain physics",
                )
            if status == "initial_force_check_rejected":
                _require(
                    self.candidates == 1 and self.selected == 0, "initial load rejection boundary"
                )
            if status == "step_budget":
                _require(
                    summary["physical_attempts"] == summary["step_budget"],
                    "actual exhausted step budget",
                )
            if status == "route_complete":
                _require(
                    summary["active_route"]["end_tick"] == self.start + self.selected
                    and self.callbacks == self.selected,
                    "complete reference endpoint",
                )


def accepted_pairs(events, *, summary=None):
    """Yield complete accepted callbacks; also bind all other events when supplied a summary."""
    pending, phase, next_index = None, "start", 0
    binding = _EventBinding(summary) if summary is not None else None
    for event in events:
        kind, payload, index = event["event"], event["payload"], event.get("trial_index")
        _require(
            event["schema"] == EVENT_SCHEMA and phase != "final", "lateral journal schema/order"
        )
        if binding is not None:
            binding.check(kind, payload, index, pending)
        if kind == "initialization":
            _require(phase == "start", "one initial state")
            phase = "initialized"
        elif kind == "planner":
            _require(phase == "initialized", "one plan at episode entry")
            phase = "route"
        elif kind == "candidate":
            _require(
                phase == "route" and index == next_index and pending is None, "candidate sequence"
            )
            if payload["selected"]:
                _require(payload["restored"] is False, "selected candidate was rolled back")
                pending = (index, payload)
            else:
                _require(
                    payload["restored"] is True
                    and payload.get("rollback_verification", {}).get("passed"),
                    "unpublished candidate must carry rollback evidence",
                )
                next_index += 1
        elif kind == "accepted_callback":
            _require(pending is not None and index == pending[0], "unpaired accepted callback")
            row = pending[1]
            _require(
                not set(row).intersection(payload), "callback cannot overwrite physical fields"
            )
            _require(
                row["measurement"] == payload["feedback_accepted"]["measurement"],
                "accepted callback must bind the same candidate wrench",
            )
            yield {**row, **payload}
            pending = None
            next_index += 1
        elif kind == "failure":
            _require(
                phase != "failed" and (pending is None or binding is not None),
                "failure interrupts incomplete publication",
            )
            pending, phase = None, "failed"
        elif kind == "final":
            _require(
                pending is None and phase in ("route", "failed", "initialized"),
                "journal final boundary",
            )
            phase = "final"
        else:
            raise ValueError("unknown lateral route event " + kind)
    _require(phase == "final", "missing final event")


def _bind_initial(summary, source):
    """Independent reset binding: only the archived free quaternion may normalize."""
    _state(source, source["local_tick"])
    initial = summary["initial_state"]
    _state(initial, 0)
    q = np.asarray(source["q"], dtype=np.float64)
    norm = float(np.linalg.norm(q[5:9]))
    normalized = q.copy()
    normalized[5:9] /= norm
    _require(np.max(abs(normalized - q)) < 1e-7, "authorized quaternion normalization bound")
    expected_q = normalized.astype(np.float32).tolist()
    expected = dict(source, local_tick=0, q=expected_q, object_quaternion_wxyz=expected_q[5:9])
    _require(
        initial == expected
        and np.float64(initial["time"]).tobytes() == np.float64(source["time"]).tobytes(),
        "fresh initial state preserves original q/v/time except public quaternion normalization",
    )
    proof = summary["initialization_proof"]
    _require(
        proof["previous_receipt_inherited"] is False
        and proof["new_local_tick"] == 0
        and proof["source_local_tick"] == source["local_tick"]
        and proof["initial_time"] == source["time"]
        and proof["source_q"] == source["q"]
        and proof["published_q"] == initial["q"]
        and proof["initialization_policy"] == "new-episode-public-free-quaternion-normalization"
        and proof["quaternion_address"] == 5
        and proof["source_quaternion_norm"] == norm
        and all(
            proof[key] is True
            for key in (
                "v_fp32_bytes_equal",
                "time_float64_bytes_equal",
                "nonrotation_q_fp32_bytes_equal",
                "normalized_quaternion_fp32_bytes_equal",
                "snapshot_q_lossless_fp32",
                "snapshot_v_lossless_fp32",
            )
        ),
        "initialization proof/state binding",
    )


def _source_state(base, summary):
    provenance = summary["provenance"]
    sources = {}
    for kind in ("report", "audit", "route", "release"):
        path = (base / provenance[f"source_{kind}_path"]).resolve()
        _require(_digest(path) == provenance[f"source_{kind}_sha256"], "source " + kind + " bytes")
        sources[kind] = _read(path)
    old, audit, route = (sources[k] for k in ("report", "audit", "route"))
    if audit.get("schema") == "phymeo-pusht-pose-replan-comparison-v1":
        _require(audit["status"] == "verified-saved-prefixes", "source combined audit")
        audit = audit["cases"]["A"]
    _require(
        old["schema"] == "phymeo-pusht-pose-replan-case-v1"
        and old["key"] == "A"
        and audit["schema"] == "phymeo-pusht-pose-replan-case-audit-v1"
        and audit["status"] == "verified-saved-prefix"
        and audit["source"]["sha256"] == provenance["source_report_sha256"]
        and audit["source"]["journal_sha256"]
        == old["journal"]["sha256"]
        == provenance["source_journal_sha256"]
        and audit["goal"] == old["goal"] == summary["goal"]
        and old["ordinary_route"]["route_sha256"] == provenance["source_route_sha256"]
        and old["release_reference"]["sha256"] == provenance["source_release_sha256"]
        and route["tick"] == summary["start_tick"] == provenance["source_input_tick"],
        "audited source identity",
    )
    sha, pending, state, callbacks = hashlib.sha256(), None, None, 0
    journal_path = (base / provenance["source_journal_path"]).resolve()
    with journal_path.open("rb") as stream:
        for _ in range(old["journal"]["records"]):
            line = stream.readline()
            _require(line.endswith(b"\n"), "complete source journal prefix")
            sha.update(line)
            event = json.loads(line, parse_constant=_constant)
            payload = event["payload"]
            if event["event"] == "candidate" and payload["selected"]:
                _require(pending is None, "source selected candidate order")
                pending = (event["trial_index"], payload)
            elif event["event"] == "accepted_callback":
                _require(
                    pending is not None
                    and pending[0] == event["trial_index"]
                    and pending[1]["measurement"] == payload["feedback_accepted"]["measurement"],
                    "source callback binds candidate",
                )
                row = pending[1]
                if row["phase"] == "regular" and row["original_input_tick"] == route["tick"]:
                    _require(state is None, "unique source route input")
                    state = row["input_state"]
                pending, callbacks = None, callbacks + 1
    _require(
        sha.hexdigest() == provenance["source_journal_sha256"]
        and callbacks == old["recorded_steps"] == audit["recorded_steps"]
        and state is not None
        and state == provenance["source_input_state"]
        and state["q"] == route["q"] == old["ordinary_route"]["source_q"]
        and state["v"] == route["v"] == old["ordinary_route"]["source_v"],
        "actual original route-entry state",
    )
    return state


def readback(path):
    path = Path(path).resolve()
    summary = _read(path)
    _require(summary["schema"] == "phymeo-pusht-lateral-route-v1", "lateral run schema")
    journal = summary["journal"]
    journal_path = path.parent / journal["file"]
    _require(_digest(journal_path) == journal["sha256"], "lateral evidence bytes")
    start, world, provenance = summary["start_tick"], summary["world"], summary["provenance"]
    _require(
        summary["side_coverage"] == "all-lateral"
        and summary["material_admission"] == "declared-finite-faces-v1"
        and summary["initialization_mode"] == "new-material-episode-from-audited-qv"
        and summary["local_tick_origin"] == 0
        and provenance["source_capsule_restored"] is False
        and provenance["old_force_history_inherited"] is False
        and provenance["old_solver_cache_inherited"] is False,
        "fresh lateral material owner",
    )
    source = _source_state(path.parent, summary)
    initial, goal = summary.get("initial_state"), summary["goal"]
    if initial is not None:
        _bind_initial(summary, source)
    if "ordinary_route" in summary:
        _planner(
            path.parent,
            summary["ordinary_route"],
            initial,
            start,
            selection_policy=summary["selection_policy"],
            material_admission=summary["material_admission"],
            material_asset_sha256=provenance["scene_asset_sha256"],
            side_coverage="all-lateral",
        )
        route = _read((path.parent / summary["ordinary_route"]["route_file"]).resolve())["route"]
        if route["status"] == "sampled_reference_ready":
            active = summary["active_route"]
            _require(
                active["start_tick"] == start
                and active["nodes"] == route["nodes"]
                and active["sha256"] == summary["ordinary_route"]["route_sha256"]
                and Path(active["file"]).resolve()
                == (path.parent / summary["ordinary_route"]["route_file"]).resolve()
                and abs((active["end_tick"] - start) * 0.001 - route["nodes"][-1][0]) <= 1e-12
                and summary["actual_selected_contact"] == route["proposal"]["index"],
                "active route/selected proposal binding",
            )
    previous, count, records, loaded, first_loaded = initial, 0, 0, 0, None

    def verified_rows(stream):
        nonlocal previous, count, records, loaded, first_loaded

        def events():
            nonlocal records
            for line in stream:
                _require(line.endswith("\n"), "complete lateral event line")
                records += 1
                yield json.loads(line, parse_constant=_constant)

        for row in accepted_pairs(events(), summary=summary):
            before, after = row["input_state"], row["output_state"]
            _state(before, count)
            _state(after, count + 1)
            _require(
                row["original_input_tick"] == start + count
                and row["original_output_tick"] == start + count + 1,
                "lateral absolute command phase",
            )
            previous = retained_state(row, previous)
            _integration(before, after)
            _receipt(row)
            _channels(row.get("submitted_control", row.get("control")))
            _wrench(row, row["measurement"], world)
            _goal_row(row, goal, world, goal_tick=count)
            _load(row)
            _require(
                row["material_identity"] == provenance["material_identity"]
                and row["side_coverage"] == summary["side_coverage"]
                and row["stored_control_matches_submitted"] is True,
                "actual step material and command",
            )
            if row["area"]["normal_load_n"] > 0:
                loaded += 1
                if first_loaded is None:
                    first_loaded = start + count
            count += 1
            yield row

    with journal_path.open(encoding="utf-8-sig") as stream:
        if initial is None:
            _require(
                not list(verified_rows(stream)), "uninitialized episode contains no accepted motion"
            )
            actual = None
        else:
            actual = summarize_motion(initial, goal, verified_rows(stream))
    _require(
        records == journal["records"] and count == summary["recorded_steps"],
        "saved complete callback count",
    )
    if actual is not None:
        _require(actual["verified_final_state"] == previous, "complete callback motion endpoint")
    return dict(
        schema="phymeo-pusht-lateral-readback-v1",
        source_sha256=_digest(path),
        journal_sha256=journal["sha256"],
        status=summary["status"],
        first_loaded_tick=first_loaded,
        loaded_steps=loaded,
        actual=actual,
        physical_final_state=summary.get("final_state"),
        unrecorded_accepted_step=summary.get("unrecorded_accepted_step"),
        interpretation="motion metrics cover complete accepted callbacks; any accepted tail is reported separately",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = readback(args.input)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, allow_nan=False)
        stream.write("\n")
    print(
        result["status"],
        0 if result["actual"] is None else result["actual"]["verified_steps"],
        result["loaded_steps"],
    )


if __name__ == "__main__":
    main()
