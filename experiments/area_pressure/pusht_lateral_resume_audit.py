"""Independent first-step physics and streaming readback of saved lateral resumes.

python -m experiments.area_pressure.pusht_lateral_resume_audit --input RUN.json --output AUDIT.json
No Scene or CPU step is executed. First-step geometry is independently clipped
and integrated; subsequent samples bind saved receipts, control and Goal history.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .pusht_active_comparison import (
    _close,
    _constant,
    _control,
    _digest,
    _integration,
    _load,
    _LoadCompensationAudit,
    _read,
    _receipt,
    _require,
    _state,
)
from .pusht_lateral_readback import _EventBinding
from .pusht_lateral_readback import accepted_pairs as _root_pairs
from .pusht_lateral_resume_reference import audit_first_step
from .pusht_pose_replan_comparison import _goal_row, _wrench, summarize_motion, validate_counts

SCHEMA = "phymeo-pusht-lateral-resume-v1"
EVENT = "phymeo-pusht-lateral-route-event-v1"
AUDIT = "phymeo-pusht-lateral-resume-audit-v1"
CHANNELS = ("ctrl", "qfrc_applied", "xfrc_applied")


def verify_retry_command(summary, parent, command):
    """An ordinary successor has no rejected command to reproduce."""
    packet = parent.get("failed_candidate")
    marker = summary.get("first_command_binding", {}).get("matches_parent_rejected_command")
    if packet is None:
        _require(marker is None, "no retry claim without an actual parent rejected command")
        return dict(
            verified=False,
            reason="ordinary accepted continuation; control is audited against route/state/previous load",
        )
    saved = packet["control"]
    start = summary["start_tick"]
    _require(
        packet["restored"] is True
        and parent["accepted_steps"] == parent["recorded_steps"]
        and parent["start_tick"] + parent["accepted_steps"] == start
        and marker is True,
        "parent failed command belongs to the exact retained endpoint",
    )
    _require(
        command["tick"] == saved["tick"] == start
        and command["output_tick"] == saved["output_tick"] == start + 1,
        "retry absolute command phase",
    )
    words = {}
    for name in CHANNELS:
        actual, expected = np.asarray(command[name]), np.asarray(saved[name])
        _require(
            actual.shape == expected.shape
            and np.isfinite(actual).all()
            and np.isfinite(expected).all()
            and np.array_equal(actual, expected)
            and actual.astype(np.float32).tobytes() == expected.astype(np.float32).tobytes(),
            "retry unchanged FP32 command words: " + name,
        )
        words[name] = actual.astype(np.float32).view(np.uint32).tolist()
    binding = summary["first_command_binding"]
    if "channel_fp32_words" in binding:
        _require(
            binding["channel_fp32_words"] == words and binding["archived_command"] == saved,
            "saved retry word proof",
        )
    return dict(verified=True, input_tick=start, output_tick=start + 1, fp32_words_equal=True)


class _ResumeBinding(_EventBinding):
    def __init__(self, summary):
        super().__init__(summary)
        self.start = summary["source_reference_tick_origin"]

    def check(self, kind, payload, index, pending):
        summary = self.summary
        if kind == "initialization":
            _require(not self.initialized and index is None, "one resume initialization")
            keys = (
                "schema",
                "world",
                "side_coverage",
                "publication_frame",
                "control_variant",
                "start_tick",
                "local_tick_origin",
                "source_reference_tick_origin",
                "goal_tick_origin",
                "initial_state",
                "goal",
                "goal_history_initial",
                "controller_initial",
                "active_route",
                "step_budget",
                "original_step_budget",
                "parent_physical_attempts",
                "provenance",
                "checkpoint_restore",
            )
            _require(
                all(payload.get(k) == summary.get(k) for k in keys)
                and payload["status"] == "running"
                and payload["final_state"] == self.retained
                and all(
                    payload[k] == 0
                    for k in (
                        "physical_attempts",
                        "physical_trial_steps",
                        "accepted_steps",
                        "recorded_steps",
                        "rejected_trials",
                    )
                ),
                "resume initialization event/summary source binding",
            )
            self.initialized = True
        elif kind == "planner":
            raise ValueError("lateral resume audit: a continuation cannot replan")
        elif kind == "final":
            _require(
                index is None and payload == {k: v for k, v in summary.items() if k != "journal"},
                "resume final event/summary identity",
            )
            _require(
                self.retained == summary["final_state"] and self.initialized,
                "resume physical endpoint",
            )
            tail_before_event = int(self.selected > self.candidates)
            validate_counts(
                summary,
                candidate_count=self.candidates + tail_before_event,
                selected_count=self.selected,
                callback_count=self.callbacks,
                failed_attempts=self.failed_attempts - tail_before_event,
                failed_physical=self.failed_physical - tail_before_event,
            )
            status = summary["status"]
            _require(
                (status == "partial_failure") == self.failure,
                "resume failure/status classification",
            )
            _require(
                status
                in ("partial_failure", "route_complete", "controller_stopped", "step_budget"),
                "resume status",
            )
            if status == "route_complete":
                _require(
                    summary["start_tick"] + self.selected == summary["active_route"]["end_tick"]
                    and self.callbacks == self.selected,
                    "original route endpoint",
                )
            if status == "step_budget":
                _require(
                    summary["physical_attempts"] == summary["step_budget"],
                    "used declared bounded attempt budget",
                )
        else:
            super().check(kind, payload, index, pending)


def iter_committed(events, summary):
    binding = _ResumeBinding(summary)
    pending, phase, next_index = None, "start", 0
    for event in events:
        kind, payload, index = event["event"], event["payload"], event.get("trial_index")
        _require(event["schema"] == EVENT and phase != "final", "resume journal event schema/order")
        binding.check(kind, payload, index, pending)
        if kind == "initialization":
            _require(phase == "start", "one initialization")
            phase = "active"
        elif kind == "candidate":
            _require(
                phase == "active" and pending is None and index == next_index,
                "resume candidate sequence",
            )
            if payload["selected"]:
                pending = (index, payload)
            else:
                next_index += 1
        elif kind == "accepted_callback":
            _require(
                pending is not None
                and index == pending[0]
                and not set(payload).intersection(pending[1])
                and pending[1]["measurement"] == payload["feedback_accepted"]["measurement"],
                "resume callback binds physical wrench without overwriting candidate",
            )
            yield dict(pending[1], **payload)
            pending, next_index = None, next_index + 1
        elif kind == "failure":
            _require(phase != "failed", "one failure event")
            pending, phase = None, "failed"
        elif kind == "final":
            _require(phase in ("active", "failed") and pending is None, "resume final boundary")
            phase = "final"
        else:
            raise ValueError("lateral resume audit: unknown event " + kind)
    _require(phase == "final", "complete resumed journal")


def _events(path, summary):
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for line in stream:
            _require(line.endswith(b"\n"), "complete saved event line")
            digest.update(line)
            count += 1
            yield json.loads(line, parse_constant=_constant)
    _require(
        count == summary["journal"]["records"]
        and digest.hexdigest() == summary["journal"]["sha256"],
        "complete saved journal bytes",
    )


def _checkpoint(base, report):
    saved = report["final_checkpoint"]
    metadata_path = (base / saved["metadata_file"]).resolve()
    _require(_digest(metadata_path) == saved["metadata_sha256"], "terminal metadata bytes")
    metadata = _read(metadata_path)
    capsule = (metadata_path.parent / metadata["checkpoint_file"]).resolve()
    _require(
        capsule == (base / saved["file"]).resolve()
        and _digest(capsule) == metadata["checkpoint_sha256"] == saved["sha256"],
        "terminal capsule bytes",
    )
    state = report["final_state"]
    _require(
        metadata["schema"] == "phymeo-pusht-active-terminal-checkpoint-v1"
        and (metadata_path.parent / metadata["source_report"]).resolve()
        == (base / report["_filename"]).resolve()
        and metadata["state"] == state
        and metadata["q"] == state["q"]
        and metadata["v"] == state["v"]
        and metadata["time"] == state["time"]
        and metadata["local_tick"] == state["local_tick"]
        and metadata["source_tick"] == report["start_tick"] + report["accepted_steps"]
        and metadata["controller"] == report["controller_final"]
        and metadata["goal_history"] == report["goal_history_final"]
        and metadata["active_route"] == report["active_route"]
        and metadata["target"] == report["goal"]
        and metadata["asset_sha256"] == report["provenance"]["scene_asset_sha256"]
        and metadata["accepted_steps"] == report["accepted_steps"]
        and metadata["recorded_steps"] == report["recorded_steps"],
        "terminal physical/controller/Goal endpoint",
    )
    for key in (
        "source_tick",
        "local_tick",
        "time",
        "world",
        "controller_at_physical_endpoint",
        "physical_state_matches_report",
    ):
        _require(saved[key] == metadata[key], "checkpoint summary " + key)
    sha, count, tail = hashlib.sha256(), 0, []
    with (base / report["journal"]["file"]).open("rb") as stream:
        for count, line in enumerate(stream, 1):
            if count <= metadata["journal"]["records"]:
                sha.update(line)
            else:
                tail.append(json.loads(line, parse_constant=_constant)["event"])
    _require(
        metadata["journal"]["file"] == report["journal"]["file"]
        and sha.hexdigest() == metadata["journal"]["sha256"]
        and count == report["journal"]["records"]
        and tail in ([], ["final"], ["failure", "final"]),
        "checkpoint journal prefix and terminal-only suffix",
    )
    return metadata, dict(
        sha256=saved["sha256"],
        source_tick=metadata["source_tick"],
        capsule_bytes_verified=True,
        restored_by_reader=False,
        controller_at_physical_endpoint=metadata["controller_at_physical_endpoint"],
    )


def bind_source(summary, parent, parent_audit, metadata, seed):
    count = parent["recorded_steps"]
    state = summary["initial_state"]
    _require(
        type(count) is int
        and count > 0
        and parent["accepted_steps"] == count
        and parent_audit["actual"]["verified_steps"] == count
        and not parent.get("unrecorded_accepted_step")
        and not parent_audit.get("unrecorded_accepted_step"),
        "fully recorded independently audited parent prefix",
    )
    _require(
        state
        == parent["final_state"]
        == parent_audit["physical_final_state"]
        == parent_audit["actual"]["verified_final_state"]
        == metadata["state"]
        == seed["output_state"]
        and summary["start_tick"] == parent["start_tick"] + count == metadata["source_tick"]
        and summary["local_tick_origin"] == state["local_tick"]
        and summary["start_tick"] == summary["source_reference_tick_origin"] + state["local_tick"]
        and summary["goal_tick_origin"] == parent["goal_tick_origin"] == 0,
        "exact retained parent endpoint and two clocks",
    )
    _require(
        summary["goal"] == parent["goal"] == metadata["target"]
        and summary["controller_initial"] == parent["controller_final"] == metadata["controller"]
        and summary["goal_history_initial"]
        == parent["goal_history_final"]
        == metadata["goal_history"]
        == seed["goal_result"]["history"]
        and metadata["controller_at_physical_endpoint"] is True
        and metadata["physical_state_matches_report"] is True,
        "inherited controller and original Goal history",
    )
    for key in ("start_tick", "end_tick", "sha256", "nodes", "completed_routes_before"):
        _require(
            summary["active_route"][key]
            == parent["active_route"][key]
            == metadata["active_route"][key],
            "same original route " + key,
        )
    _require(
        summary["provenance"]["scene_asset_sha256"]
        == parent["provenance"]["scene_asset_sha256"]
        == metadata["asset_sha256"]
        and summary["control_variant"] == parent["control_variant"] == "load-compensation-half"
        and seed["original_output_tick"] == summary["start_tick"]
        and seed["original_input_tick"] + 1 == summary["start_tick"],
        "same material and immediate preceding load",
    )
    proof = summary["checkpoint_restore"]
    _require(
        proof["checkpoint_sha256"] == metadata["checkpoint_sha256"]
        and proof["restored_local_tick"] == state["local_tick"]
        and proof["absolute_input_tick"] == summary["start_tick"]
        and proof["time"] == state["time"]
        and proof["previous_receipt_bound"] is True
        and proof["controller_callbacks_replayed"] is True
        and proof["prior_physics_steps_replayed"] == 0
        and proof["previous_tool_force_world_n"] == seed["area"]["total_force_on_pusher_world_n"],
        "restored checkpoint and actual preceding load proof",
    )


def _parent(base, summary):
    provenance = summary["provenance"]
    bound = {}
    for key in ("parent_report", "parent_audit", "checkpoint_metadata"):
        path = (base / provenance[key + "_path"]).resolve()
        _require(_digest(path) == provenance[key + "_sha256"], "resume source " + key + " bytes")
        bound[key] = (path, _read(path))
    parent_path, parent = bound["parent_report"]
    audit = bound["parent_audit"][1]
    _require(
        parent["schema"] in ("phymeo-pusht-lateral-route-v1", SCHEMA)
        and audit["schema"] in ("phymeo-pusht-lateral-readback-v1", AUDIT)
        and audit["source_sha256"] == provenance["parent_report_sha256"]
        and audit["journal_sha256"] == parent["journal"]["sha256"],
        "parent independent audit identity",
    )
    metadata, _ = _checkpoint(parent_path.parent, dict(parent, _filename=parent_path.name))
    _require(metadata == bound["checkpoint_metadata"][1], "exact selected terminal metadata")
    events = _events(parent_path.parent / parent["journal"]["file"], parent)
    rows = (
        _root_pairs(events, summary=parent)
        if parent["schema"] != SCHEMA
        else iter_committed(events, parent)
    )
    seed, count = None, 0
    for seed in rows:
        count += 1
    _require(
        seed is not None and count == parent["recorded_steps"], "last complete parent callback"
    )
    bind_source(summary, parent, audit, metadata, seed)
    attempts, seen, current, path = 0, set(), parent, parent_path
    while True:
        _require(path not in seen and len(seen) < 32, "bounded acyclic parent budget chain")
        seen.add(path)
        attempts += current["physical_attempts"]
        if current["schema"] != SCHEMA:
            _require(
                current["schema"] == "phymeo-pusht-lateral-route-v1", "original lateral episode"
            )
            original_budget = current["step_budget"]
            break
        next_path = (path.parent / current["provenance"]["parent_report_path"]).resolve()
        _require(
            _digest(next_path) == current["provenance"]["parent_report_sha256"],
            "parent budget report bytes",
        )
        current, path = _read(next_path), next_path
    _require(
        summary["parent_physical_attempts"] == attempts
        and summary["original_step_budget"] == original_budget
        and 0
        < summary["step_budget"]
        <= min(
            original_budget - attempts, summary["active_route"]["end_tick"] - summary["start_tick"]
        ),
        "original route and global attempt budget",
    )
    return parent, audit, seed


def audit(path):
    path = Path(path).resolve()
    summary = _read(path)
    _require(
        summary["schema"] == SCHEMA
        and summary["side_coverage"] == "all-lateral"
        and summary["publication_frame"] == "rigid-origin",
        "resumed episode and publication schema",
    )
    _state(summary["initial_state"], summary["local_tick_origin"])
    parent, parent_audit, seed = _parent(path.parent, summary)
    route_path = (path.parent / summary["active_route"]["file"]).resolve()
    _require(_digest(route_path) == summary["active_route"]["sha256"], "unchanged route bytes")
    route = _read(route_path)
    _require(route["route"]["nodes"] == summary["active_route"]["nodes"], "active route nodes")
    compensation = _LoadCompensationAudit(summary, {"trials": [seed]}, route)
    _require(
        summary["provenance"]["source_model_sha256"]
        == next(
            s
            for s in __import__(
                "phymeo.training.manipulation_tasks", fromlist=["manipulation_scenes"]
            ).manipulation_scenes()
            if s.key == "pusht-planar"
        ).source_sha256,
        "original source model bytes",
    )
    previous = summary["initial_state"]
    count, first_physics, first_command = 0, None, None
    packet = None
    if summary.get("first_step_evidence") is not None:
        saved = summary["first_step_evidence"]
        evidence_path = (path.parent / saved["file"]).resolve()
        _require(
            _digest(evidence_path) == saved["sha256"]
            and saved["input_tick"] == summary["start_tick"]
            and saved["output_tick"] == summary["start_tick"] + 1,
            "first-step evidence bytes/phase",
        )
        packet = _read(evidence_path)
        previous_receipt = packet["previous_receipt"]
        _require(
            previous_receipt["tick"] == summary["initial_state"]["local_tick"]
            and previous_receipt["output_time"] == summary["initial_state"]["time"],
            "actual previous receipt output phase",
        )
        (previous_pair,) = previous_receipt["raw_contact_fields"]["area_friction"]
        _require(
            previous_pair["normal_force"] == seed["area"]["raw_normal_force_n"],
            "restored previous point loads",
        )
        _close(
            np.asarray(previous_pair["force_on_a"]).sum(axis=0),
            seed["area"]["total_force_on_pusher_world_n"],
            "actual preceding point-force sum",
            atol=1e-7,
            rtol=8 * np.finfo(np.float32).eps,
        )

    def checked_events():
        nonlocal first_physics
        for event in _events(path.parent / summary["journal"]["file"], summary):
            if event["event"] == "candidate" and event["trial_index"] == 0:
                _require(
                    packet is not None, "first actual step requires complete physical evidence"
                )
                first_physics = audit_first_step(packet, event["payload"])
            yield event

    def checked_rows():
        nonlocal previous, count, first_command
        for row in iter_committed(checked_events(), summary):
            _require(row["input_state"] == previous, "resumed complete-callback state chain")
            before, after = row["input_state"], row["output_state"]
            _state(before, summary["local_tick_origin"] + count)
            _state(after, summary["local_tick_origin"] + count + 1)
            _integration(before, after)
            _receipt(row)
            command = row.get("submitted_control", row.get("control"))
            view = dict(row, submitted_control=command)
            _control(view, summary, count, path.parent, compensation)
            # Same current q/v, independent MuJoCo RNE; never call the controller.
            mj, model, data = compensation.mj, compensation.model, compensation.data
            data.qpos[:], data.qvel[:] = before["q"], before["v"]
            mj.mj_kinematics(model, data)
            mj.mj_comPos(model, data)
            mj.mj_comVel(model, data)
            bias = np.zeros(model.nv)
            mj.mj_rne(model, data, 0, bias)
            _close(row["controller"]["pd"]["bias"], bias[:2], "independent current-state PD bias")
            if count == 0:
                first_command = verify_retry_command(summary, parent, command)
                _require(
                    packet["controller"] == row["controller"]
                    and packet["goal_frame"] == row["goal_frame"]
                    and packet["goal_result"] == row["goal_result"],
                    "first diagnostic/controller/Goal callback",
                )
            _wrench(row, row["measurement"], summary["world"])
            _goal_row(row, summary["goal"], summary["world"], goal_tick=before["local_tick"])
            _load(row)
            _require(
                row["side_coverage"] == "all-lateral"
                and row["material_identity"] == summary["provenance"]["material_identity"],
                "accepted material identity",
            )
            _require(
                row["driver_state_after"]["tick"] == row["original_output_tick"],
                "one feedback advance",
            )
            compensation.accepted(row)
            previous, count = after, count + 1
            yield row

    actual = summarize_motion(summary["initial_state"], summary["goal"], checked_rows())
    _require(count == summary["recorded_steps"], "complete resumed callback count")
    if count == 0 and summary.get("failed_candidate") is not None:
        first_command = verify_retry_command(
            summary, parent, summary["failed_candidate"]["control"]
        )
    if count:
        _require(
            first_physics is not None and first_physics["verified"],
            "first accepted physics independent verification",
        )
    checkpoint = None
    if summary.get("final_checkpoint") is not None:
        _, checkpoint = _checkpoint(path.parent, dict(summary, _filename=path.name))
    first_step_verified = first_physics is not None and first_physics.get("verified") is True
    physics_scope = (
        "first saved physical candidate independently verified in FP64; "
        if first_step_verified
        else "no first-candidate independent physics verification; "
    )
    return dict(
        schema=AUDIT,
        source_sha256=_digest(path),
        journal_sha256=summary["journal"]["sha256"],
        status=summary["status"],
        physical_final_state=summary["final_state"],
        actual=actual,
        unrecorded_accepted_step=summary.get("unrecorded_accepted_step"),
        first_command_binding=first_command,
        first_step_reference=first_physics,
        final_checkpoint=checkpoint,
        control=dict(variant="load-compensation-half", verified_steps=count),
        parent_verified_steps=parent["recorded_steps"],
        combined_verified_steps=parent_audit.get(
            "combined_verified_steps", parent["recorded_steps"]
        )
        + count,
        original_attempts_consumed=summary["parent_physical_attempts"]
        + summary["physical_attempts"],
        independent_first_step_physics=first_step_verified,
        scope=physics_scope
        + "actual includes only completed callbacks checked for saved receipt, integration, control, previous load and Goal phase; a physical candidate may be independently verified even when its callback failed, with the accepted tail reported separately",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, indent=2)
        stream.write("\n")
    print(result["status"], result["actual"]["verified_steps"], result["first_step_reference"])


if __name__ == "__main__":
    main()
