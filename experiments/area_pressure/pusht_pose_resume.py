"""Resume a saved independent-pose B rollout on its already selected route.

Controller/Goal reconstruction replays accepted callbacks only. The public
capsule restores physics; the existing active-route runner issues every new step.
"""

import argparse
import gzip
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from shutil import copyfile

import numpy as np

from .pusht_active_resume import _bad_constant, _digest, _read, _require


def iter_pose_events(bound):
    """Read the hash-bound event journal without retaining its full trajectory."""
    path = bound["journal_path"]
    _require(_digest(path) == bound["report"]["journal"]["sha256"], "pose journal bytes")
    with path.open("rt", encoding="utf-8-sig") as stream:
        for line in stream:
            event = json.loads(line, parse_constant=_bad_constant)
            _require(event["schema"] == "phymeo-pusht-pose-replan-event-v1", "pose event schema")
            yield event


def _last_accepted(bound):
    report = bound["report"]
    state, pending, last, count, records = report["initial_state"], None, None, 0, 0
    for event in iter_pose_events(bound):
        records += 1
        kind, payload, index = event["event"], event["payload"], event.get("trial_index")
        if kind == "candidate":
            _require(
                pending is None and payload["input_state"] == state, "pose candidate predecessor"
            )
            _require(
                payload["original_input_tick"] == report["start_tick"] + count
                and payload["original_output_tick"] == report["start_tick"] + count + 1
                and payload["output_state"]["local_tick"] == state["local_tick"] + 1,
                "pose candidate adjacent clock",
            )
            if payload["selected"]:
                _require(payload["restored"] is False, "selected pose candidate was restored")
                pending = (index, payload)
            else:
                _require(
                    payload["restored"] is True
                    and payload["rollback_verification"]["passed"] is True,
                    "unselected pose candidate rollback",
                )
        elif kind == "accepted_callback":
            _require(pending is not None and index == pending[0], "pose callback ownership")
            row = deepcopy(pending[1])
            row.update(deepcopy(payload))
            state, last, pending, count = row["output_state"], row, None, count + 1
    _require(
        records == report["journal"]["records"]
        and pending is None
        and count == report["recorded_steps"]
        and state == report["final_state"]
        and last is not None,
        "complete pose accepted journal endpoint",
    )
    return last


def _checkpoint_journal_binding(bound):
    """The capsule precedes diagnostic failure/final events in the pose writer."""
    report, metadata = bound["report"], bound["metadata"]
    prefix, full = metadata["journal"], report["journal"]
    count = prefix["records"]
    _require(
        prefix["file"] == full["file"] and type(count) is int and 0 < count <= full["records"],
        "pose checkpoint journal prefix extent",
    )
    prefix_hash, full_hash, tail, records = hashlib.sha256(), hashlib.sha256(), [], 0
    with bound["journal_path"].open("rb") as stream:
        for records, line in enumerate(stream, 1):
            full_hash.update(line)
            if records <= count:
                prefix_hash.update(line)
            else:
                _require(len(tail) < 2, "pose checkpoint journal extra events")
                tail.append(json.loads(line, parse_constant=_bad_constant))
    _require(
        records == full["records"]
        and prefix_hash.hexdigest() == prefix["sha256"]
        and full_hash.hexdigest() == full["sha256"],
        "pose checkpoint journal exact prefix/full bytes",
    )
    kinds = [event.get("event") for event in tail]
    _require(
        kinds in ([], ["final"], ["failure", "final"]),
        "pose checkpoint journal nonphysical terminal events only",
    )
    for event in tail:
        _require(
            event.get("schema") == "phymeo-pusht-pose-replan-event-v1",
            "pose checkpoint journal terminal schema",
        )
        if event["event"] == "failure":
            expected = {
                key: report.get(key)
                for key in (
                    "failure_stage",
                    "errors",
                    "failed_candidate",
                    "unrecorded_accepted_step",
                )
            }
            _require(
                report["status"] == "partial_failure" and event["payload"] == expected,
                "pose checkpoint journal failure/report identity",
            )
        else:
            _require(
                event["payload"]
                == {key: value for key, value in report.items() if key != "journal"}
                and event.get("trial_index") is None,
                "pose checkpoint journal final/report identity",
            )
    return dict(
        checkpoint_records=count,
        checkpoint_sha256=prefix["sha256"],
        complete_records=records,
        complete_sha256=full["sha256"],
        terminal_events=kinds,
    )


def load_pose_source(report_path, comparison_path, metadata_path):
    """Bind B's complete saved prefix, comparison, capsule, route and retry input.

    ``previous`` is the last actual selected candidate merged with its matching
    accepted_callback, copied in memory. No source report or journal is rewritten.
    """
    report_path, comparison_path, metadata_path = (
        Path(p).resolve() for p in (report_path, comparison_path, metadata_path)
    )
    report, comparison, meta = map(_read, (report_path, comparison_path, metadata_path))
    count = report["recorded_steps"]
    _require(
        report["schema"] == "phymeo-pusht-pose-replan-case-v1"
        and report["key"] == "B"
        and report["world"] == 6
        and report["initialization_mode"] == "independent-pose-reset"
        and report["start_tick"] == report["local_tick_origin"] == 0
        and report["control_variant"] == "load-compensation-half"
        and type(count) is int
        and count > 0
        and count == report["accepted_steps"]
        and report.get("unrecorded_accepted_step") is None,
        "complete independent-pose B source required",
    )
    audit = comparison["cases"]["B"]
    _require(
        comparison["schema"] == "phymeo-pusht-pose-replan-comparison-v1"
        and comparison["status"] == "verified-saved-prefixes"
        and audit["schema"] == "phymeo-pusht-pose-replan-case-audit-v1"
        and audit["status"] == "verified-saved-prefix"
        and Path(audit["source"]["file"]).resolve() == report_path
        and audit["source"]["sha256"] == _digest(report_path)
        and audit["source"]["journal_sha256"] == report["journal"]["sha256"]
        and audit["accepted_steps"] == audit["recorded_steps"] == count
        and audit["reported_final_state"] == report["final_state"]
        and audit.get("unrecorded_accepted_step") is None,
        "pose comparison does not bind this complete B endpoint",
    )
    saved, state, route = report["final_checkpoint"], report["final_state"], report["active_route"]
    _require(
        meta["schema"] == saved["schema"] == "phymeo-pusht-active-terminal-checkpoint-v1"
        and (report_path.parent / saved["metadata_file"]).resolve() == metadata_path
        and saved["metadata_sha256"] == _digest(metadata_path)
        and (metadata_path.parent / meta["source_report"]).resolve() == report_path,
        "pose checkpoint metadata source/schema/hash",
    )
    _require(
        meta["world"] == 6
        and meta["state"] == state
        and meta["q"] == state["q"]
        and meta["v"] == state["v"]
        and meta["local_tick"] == state["local_tick"] == count
        and meta["source_tick"] == count
        and meta["time"] == state["time"]
        and meta["controller_at_physical_endpoint"] is True
        and meta["physical_state_matches_report"] is True,
        "pose checkpoint exact physical endpoint/clock",
    )
    _require(
        meta["controller"] == report["controller_final"]
        and meta["controller"]["tick"] == count
        and meta["controller"]["status"] == "continue"
        and meta["controller"]["partial_route"] is not None
        and meta["controller"]["partial_route"]["start_tick"] == route["start_tick"]
        and route["start_tick"] <= count < route["end_tick"]
        and meta["active_route"] == route,
        "pose checkpoint active controller/selected route",
    )
    _require(
        meta["goal_history"] == report["goal_history_final"]
        and meta["goal_history"]["tick"] == count - 1
        and meta["target"] == report["goal"],
        "pose checkpoint accepted Goal history/target",
    )
    _require(
        meta["asset_sha256"] == report["provenance"]["scene_asset_sha256"]
        and meta["control_variant"] == report["control_variant"]
        and meta["accepted_steps"] == meta["recorded_steps"] == count,
        "pose checkpoint material/control/accepted counts",
    )
    for key in (
        "source_tick",
        "local_tick",
        "time",
        "world",
        "controller_at_physical_endpoint",
        "physical_state_matches_report",
    ):
        _require(saved[key] == meta[key], "pose checkpoint summary " + key)
    capsule_path = metadata_path.parent / meta["checkpoint_file"]
    _require(
        Path(meta["checkpoint_file"]).name == meta["checkpoint_file"]
        and capsule_path.resolve() == (report_path.parent / saved["file"]).resolve()
        and _digest(capsule_path) == meta["checkpoint_sha256"] == saved["sha256"],
        "pose capsule bytes",
    )
    route_path = (report_path.parent / route["file"]).resolve()
    _require(_digest(route_path) == route["sha256"], "unchanged selected pose route")
    failed = report.get("failed_candidate")
    retry = None if failed is None else failed["control"]
    if failed is not None:
        _require(
            failed["restored"] is True
            and failed["rollback_verification"]["passed"] is True
            and failed["input_states"]["6"] == state
            and retry["tick"] == count
            and retry["output_tick"] == count + 1,
            "failed pose input/retry command does not match capsule endpoint",
        )
        for name in ("ctrl", "qfrc_applied", "xfrc_applied"):
            _require(
                np.isfinite(np.asarray(retry[name], dtype=float)).all(), "finite retry " + name
            )
    bound = dict(
        report=report,
        metadata=meta,
        audit=audit,
        comparison=comparison,
        report_path=report_path,
        audit_path=comparison_path,
        metadata_path=metadata_path,
        capsule_path=capsule_path,
        route_path=route_path,
        journal_path=(report_path.parent / report["journal"]["file"]).resolve(),
        retry_command=retry,
    )
    bound["checkpoint_journal_binding"] = _checkpoint_journal_binding(bound)
    bound["previous"] = _last_accepted(bound)
    return bound


def prepare_pose_resume_entry(prior_path, source, comparison, metadata, route_output):
    """Reconstruct the original reset controller through saved callbacks, never physics."""
    import mujoco as mj

    from phymeo.backends.mujoco_runtime import MuJoCoInitialState
    from phymeo.training.manipulation_tasks import manipulation_scenes
    from scripts.run_manipulation_control import plain
    from scripts.run_manipulation_push import SOURCE_SHA
    from scripts.run_manipulation_push_feedback import PushFeedbackDriver, hold_plan

    from .pusht_active_entry import _accepted, _restore_active_route, _state, _words
    from .pusht_release_entry import ReleaseEntry

    bound = load_pose_source(source, comparison, metadata)
    report, meta = bound["report"], bound["metadata"]
    prior = _read(prior_path)
    provenance = report["provenance"]
    parent_path = Path(provenance["source_report_path"])
    _require(
        _digest(parent_path) == provenance["source_report_sha256"]
        and _read(parent_path)["provenance"]["prior_summary_sha256"] == _digest(prior_path),
        "original controller source summary",
    )
    candidate_path = Path(prior["provenance"]["candidate_source"])
    _require(
        _digest(candidate_path) == prior["provenance"]["candidate_source_sha256"] == SOURCE_SHA,
        "original controller candidate input",
    )
    scene = next(s for s in manipulation_scenes() if s.key == "pusht-planar")
    _require(scene.source_sha256 == prior["provenance"]["source_sha256"], "original model source")
    model = mj.MjModel.from_xml_string(scene.mjcf)
    saved = next(
        row
        for row in _read(candidate_path)["rows"]
        if row["summary"]["world"] == prior["worlds"]["open"]
    )
    driver = PushFeedbackDriver(scene, model, Path(route_output), saved)
    pd, _, _ = hold_plan(scene, model)
    declaration, initial_state = provenance["reset_declaration"], report["initial_state"]
    initial = MuJoCoInitialState(declaration["q"], declaration["v"], time=declaration["time"])
    _state(model, initial_state)
    _require(
        initial.time == initial_state["time"] == 0
        and initial_state["local_tick"] == 0
        and _words(initial.q.astype(np.float32), initial_state["q"])
        and _words(initial.v, initial_state["v"]),
        "original declared reset publication",
    )
    driver.last_q = np.asarray(initial_state["q"], dtype=float)
    driver.initial_cost = driver.last_cost = driver.cost(driver.last_q)
    driver.monitor.start(0, driver.initial_cost)
    _require(plain(driver.summary()) == report["controller_initial"], "cold original controller")
    target = dict(
        target_xy=plain(driver.target.position[:2]),
        target_yaw=driver.target.yaw,
        position_tolerance_m=driver.goal.position_tolerance_m,
        yaw_tolerance_rad=driver.goal.yaw_tolerance_rad,
    )
    _require(target == report["goal"], "unchanged original Goal")
    entry = ReleaseEntry(
        scene, model, driver, pd, None, deepcopy(initial_state), None, 0, initial, {}
    )
    state, history, pending, count, plans = initial_state, None, None, 0, 0
    for event in iter_pose_events(bound):
        kind, payload = event["event"], event["payload"]
        if kind == "planner":
            _require(
                pending is None
                and plans == 0
                and driver.active is None
                and payload["tick"] == driver.tick
                and _words(payload["source_q"], state["q"])
                and _words(payload["source_v"], state["v"])
                and payload["route_sha256"] == report["active_route"]["sha256"],
                "unique saved ordinary planner boundary",
            )
            record = deepcopy(report["controller_final"]["routes"][0])
            _require(
                record["sha256"] == payload["route_sha256"]
                and record["file"] == bound["route_path"].name,
                "saved route identity",
            )
            with gzip.open(bound["route_path"], "rt", encoding="utf-8") as stream:
                saved_route = json.load(stream)
            copyfile(bound["route_path"], driver.output / record["file"])
            _restore_active_route(driver, record, saved_route, state)
            plans += 1
        elif kind == "candidate" and payload["selected"]:
            _state(model, payload["input_state"])
            _state(model, payload["output_state"])
            _require(
                pending is None
                and payload["input_state"] == state
                and payload["original_input_tick"] == driver.tick
                and payload["phase"] in ("settle", "regular"),
                "saved accepted pose controller input",
            )
            pending = payload
        elif kind == "accepted_callback":
            _require(pending is not None, "accepted callback without selected physical row")
            history = _accepted(entry, pending, payload, history)
            state, pending, count = pending["output_state"], None, count + 1
    _require(
        plans == 1
        and pending is None
        and count == report["recorded_steps"]
        and state == meta["state"]
        and plain(history) == meta["goal_history"]
        and plain(driver.summary()) == meta["controller"]
        and driver.active is not None
        and plain(driver.active[1].nodes) == report["active_route"]["nodes"],
        "reconstructed pose controller/Goal/route terminal boundary",
    )
    entry = replace(
        entry,
        state=deepcopy(state),
        previous=deepcopy(bound["previous"]),
        history=history,
        tick=driver.tick,
    )
    entry.provenance = dict(
        scene_asset_sha256=meta["asset_sha256"],
        parent_pose_report_path=str(bound["report_path"]),
        parent_pose_report_sha256=_digest(bound["report_path"]),
        parent_pose_comparison_path=str(bound["audit_path"]),
        parent_pose_comparison_sha256=_digest(bound["audit_path"]),
        parent_pose_journal_path=str(bound["journal_path"]),
        parent_pose_journal_sha256=report["journal"]["sha256"],
        parent_pose_checkpoint_journal=deepcopy(bound["checkpoint_journal_binding"]),
        source_tick=entry.tick,
        source_local_tick=state["local_tick"],
        source_time=state["time"],
        source_controller="original reset controller plus saved accepted callbacks and selected route; no physics replay",
        initialization_mode="restored-independent-pose-checkpoint",
    )
    return entry, bound


def run(prior, source, comparison, metadata, output, *, cache_root=None):
    from .pusht_active_route import run as run_active

    output = Path(output).resolve()
    prepared = prepare_pose_resume_entry(
        prior, source, comparison, metadata, output.with_suffix(".routes")
    )
    return run_active(
        prior,
        None,
        output,
        checkpoint_metadata_path=metadata,
        cache_root=cache_root,
        control_variant="load-compensation-half",
        prepared_resume=prepared,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("summary", "source", "audit", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    result = run(
        args.summary,
        args.source,
        args.audit,
        args.checkpoint,
        args.output,
        cache_root=args.cache_root,
    )
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in (
                    "status",
                    "start_tick",
                    "accepted_steps",
                    "recorded_steps",
                    "failure_stage",
                )
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
