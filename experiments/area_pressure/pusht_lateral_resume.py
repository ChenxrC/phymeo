"""Continue a SHA-bound all-lateral capsule on its one existing route.

python -m experiments.area_pressure.pusht_lateral_resume --source live.json
  --audit live-readback.json --checkpoint live.final.scene.metadata.json --output resumed.json

Reconstructs saved controller/Goal callbacks without replaying physics. The first
new command must reproduce a saved rejected command when one exists. Every new
step is a public CUDA Scene transaction; no route or material is replaced.
"""

import argparse
import gzip
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from shutil import copyfile

import numpy as np

from scripts.run_manipulation_control import plain

from .pusht_active_resume import _bad_constant, _digest, _read
from .pusht_public_scene import _error_chain, _state, _write_progress

EVENT = "phymeo-pusht-lateral-route-event-v1"
SCHEMA = "phymeo-pusht-lateral-resume-v1"


def _require(ok, reason):
    if not ok:
        raise ValueError("lateral resume: " + reason)


def reference_tick(entry, snapshot):
    return entry.tick + snapshot.tick - entry.state["local_tick"]


def _words(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (
        a.shape == b.shape
        and np.isfinite(a).all()
        and np.isfinite(b).all()
        and np.array_equal(a, b)
        and a.astype(np.float32).tobytes() == b.astype(np.float32).tobytes()
    )


def bind_retry_command(control, saved, tick):
    _require(saved["tick"] == tick and saved["output_tick"] == tick + 1, "retry absolute tick")
    words = {}
    for name in ("ctrl", "qfrc_applied", "xfrc_applied"):
        actual, expected = np.asarray(getattr(control, name)), np.asarray(saved[name])
        _require(_words(actual, expected), "retry exact " + name)
        words[name] = actual.astype(np.float32).view(np.uint32).tolist()
    return dict(
        matches_parent_rejected_command=True,
        input_tick=tick,
        output_tick=tick + 1,
        channel_fp32_words=words,
        archived_command=deepcopy(saved),
    )


def validate_terminal_binding(report, meta, *, allow_completed=False):
    state = report["final_state"]
    _require(
        report["schema"] in ("phymeo-pusht-lateral-route-v1", SCHEMA)
        and report["world"] == meta["world"] == 6
        and report["side_coverage"] == "all-lateral"
        and report["control_variant"] == meta["control_variant"] == "load-compensation-half"
        and report["accepted_steps"] == report["recorded_steps"] > 0
        and not report.get("unrecorded_accepted_step"),
        "complete same-variant saved prefix",
    )
    _require(
        meta["schema"] == "phymeo-pusht-active-terminal-checkpoint-v1"
        and meta["state"] == state
        and _words(meta["q"], state["q"])
        and _words(meta["v"], state["v"])
        and meta["time"] == state["time"]
        and meta["local_tick"] == state["local_tick"]
        and meta["source_tick"] == report["start_tick"] + report["accepted_steps"]
        and state["local_tick"] == report["local_tick_origin"] + report["accepted_steps"]
        and meta["source_tick"] == report["source_reference_tick_origin"] + state["local_tick"]
        and report["goal_tick_origin"] == 0,
        "terminal exact state and dual clock",
    )
    _require(
        meta["controller"] == report["controller_final"]
        and meta["controller"]["tick"] == meta["source_tick"]
        and meta["controller"]["status"] == "continue"
        and meta["active_route"] == report["active_route"]
        and (
            (
                meta["controller"]["partial_route"] is not None
                and meta["active_route"]["start_tick"]
                <= meta["source_tick"]
                < meta["active_route"]["end_tick"]
            )
            or (
                allow_completed is True
                and report["status"] == "route_complete"
                and meta["controller"]["partial_route"] is None
                and meta["source_tick"] == meta["active_route"]["end_tick"]
                and meta["controller"]["completed_routes"]
                == meta["active_route"]["completed_routes_before"] + 1
            )
        )
        and meta["goal_history"] == report["goal_history_final"]
        and meta["goal_history"]["tick"] == state["local_tick"] - 1
        and meta["target"] == report["goal"],
        "controller route and local Goal endpoint",
    )
    _require(
        meta["asset_sha256"] == report["provenance"]["scene_asset_sha256"]
        and meta["controller_at_physical_endpoint"] is True
        and meta["physical_state_matches_report"] is True
        and meta["accepted_steps"] == meta["recorded_steps"] == report["accepted_steps"],
        "terminal asset and accepted counts",
    )


def accepted_pairs(events, initial, reference_origin):
    state, pending = deepcopy(initial), None
    for event in events:
        kind, row = event["event"], event["payload"]
        if kind == "candidate":
            _require(pending is None and row["input_state"] == state, "candidate predecessor")
            _require(
                row["original_input_tick"] == reference_origin + state["local_tick"]
                and row["original_output_tick"] == row["original_input_tick"] + 1
                and row["output_state"]["local_tick"] == state["local_tick"] + 1
                and row["output_state"]["time"] > state["time"],
                "candidate adjacent clocks",
            )
            if row["selected"]:
                _require(row["restored"] is False, "selected candidate cannot be restored")
                pending = (event["trial_index"], row)
            else:
                _require(
                    row["restored"] is True and row["rollback_verification"]["passed"] is True,
                    "rejected candidate rollback",
                )
        elif kind == "accepted_callback":
            _require(
                pending is not None and pending[0] == event["trial_index"],
                "callback candidate identity",
            )
            _require(row["goal_frame"]["tick"] == state["local_tick"], "local Goal callback clock")
            yield pending[1], row
            state, pending = pending[1]["output_state"], None
    _require(pending is None, "unrecorded accepted candidate")


def _events(bound):
    with bound["journal_path"].open("rt", encoding="utf-8-sig") as stream:
        for line in stream:
            event = json.loads(line, parse_constant=_bad_constant)
            _require(event["schema"] == EVENT, "journal event schema")
            yield event


def load_source(report_path, audit_path, metadata_path, *, allow_completed=False):
    report_path, audit_path, metadata_path = (
        Path(p).resolve() for p in (report_path, audit_path, metadata_path)
    )
    report, audit, meta = map(_read, (report_path, audit_path, metadata_path))
    validate_terminal_binding(report, meta, allow_completed=allow_completed)
    _require(
        audit["schema"]
        in ("phymeo-pusht-lateral-readback-v1", "phymeo-pusht-lateral-resume-audit-v1")
        and audit["source_sha256"] == _digest(report_path)
        and audit["journal_sha256"] == report["journal"]["sha256"]
        and audit["physical_final_state"] == report["final_state"]
        and audit["actual"]["verified_steps"] == report["recorded_steps"]
        and audit["actual"]["verified_final_state"] == report["final_state"]
        and not audit.get("unrecorded_accepted_step"),
        "independent saved-prefix audit",
    )
    saved = report["final_checkpoint"]
    _require(
        (report_path.parent / saved["metadata_file"]).resolve() == metadata_path
        and _digest(metadata_path) == saved["metadata_sha256"]
        and (metadata_path.parent / meta["source_report"]).resolve() == report_path,
        "metadata path/hash/source",
    )
    capsule = metadata_path.parent / meta["checkpoint_file"]
    _require(
        Path(meta["checkpoint_file"]).name == meta["checkpoint_file"]
        and capsule.resolve() == (report_path.parent / saved["file"]).resolve()
        and _digest(capsule) == meta["checkpoint_sha256"] == saved["sha256"],
        "capsule bytes",
    )
    for name in (
        "source_tick",
        "local_tick",
        "time",
        "world",
        "controller_at_physical_endpoint",
        "physical_state_matches_report",
    ):
        _require(saved[name] == meta[name], "capsule summary " + name)
    journal = report_path.parent / report["journal"]["file"]
    full, prefix = hashlib.sha256(), hashlib.sha256()
    tail = []
    with journal.open("rb") as stream:
        records = 0
        for records, line in enumerate(stream, 1):
            _require(line.endswith(b"\n"), "complete journal line")
            full.update(line)
            if records <= meta["journal"]["records"]:
                prefix.update(line)
            else:
                tail.append(json.loads(line, parse_constant=_bad_constant)["event"])
    _require(
        records == report["journal"]["records"]
        and full.hexdigest() == report["journal"]["sha256"]
        and meta["journal"]["file"] == report["journal"]["file"]
        and prefix.hexdigest() == meta["journal"]["sha256"]
        and tail in ([], ["final"], ["failure", "final"]),
        "checkpoint/full journal byte prefixes",
    )
    route = (report_path.parent / report["active_route"]["file"]).resolve()
    _require(_digest(route) == report["active_route"]["sha256"], "unchanged active route bytes")
    failed = report.get("failed_candidate")
    if failed is not None:
        _require(
            failed["restored"]
            and failed["rollback_verification"]["passed"]
            and failed["input_states"]["6"] == report["final_state"]
            and failed["control"]["tick"] == meta["source_tick"]
            and failed["control"]["output_tick"] == meta["source_tick"] + 1,
            "saved failed command endpoint",
        )
    return dict(
        report=report,
        audit=audit,
        metadata=meta,
        report_path=report_path,
        audit_path=audit_path,
        metadata_path=metadata_path,
        capsule_path=capsule,
        journal_path=journal,
        route_path=route,
        retry_command=None if failed is None else failed["control"],
    )


def _replay_callback(entry, row, callback, history):
    from phymeo.training.manipulation_tasks import SuccessFrame, evaluate_goal

    from .pusht_active_route import _controller_boundary
    from .pusht_public_feedback import PublicFeedbackMeasurement

    before, after = row["input_state"], row["output_state"]
    frame = SuccessFrame(**callback["goal_frame"])
    _require(
        frame.tick == before["local_tick"]
        and frame.world == 6
        and frame.time_s == before["time"]
        and frame.phase == "accepted-entry"
        and _words(frame.object_position, before["object_position_world_m"])
        and _words(frame.object_wxyz, before["object_quaternion_wxyz"]),
        "saved local Goal input",
    )
    goal = evaluate_goal(entry.scene, entry.driver.goal, frame, history)
    _require(
        plain(goal) == callback["goal_result"] and not goal.success, "recomputed saved Goal history"
    )
    measurement = PublicFeedbackMeasurement(**callback["feedback_accepted"]["measurement"])
    tick = row["original_input_tick"]
    _require(
        plain(measurement) == row["measurement"]
        and measurement.world == 6
        and measurement.input_tick == tick
        and measurement.output_tick == tick + 1
        and measurement.time_s == before["time"]
        and measurement.phase == "accepted-step-input"
        and measurement.native_contacts == 0
        and measurement.area_samples == row["area"]["samples"]
        and np.array_equal(
            measurement.group_force_world, -np.asarray(row["area"]["total_force_on_pusher_world_n"])
        ),
        "saved actual preceding force source",
    )
    result = entry.driver.accepted_measurement(
        tick + 1, np.asarray(after["q"]), measurement, goal, input_q=np.asarray(before["q"])
    )
    _require(
        plain(result) == callback["feedback_accepted"]
        and _controller_boundary(entry.driver) == callback["driver_state_after"]
        and not entry.driver.stopped,
        "replayed controller progress/impulses",
    )
    return goal.history


def prepare_entry(source, audit, metadata, route_output, *, allow_completed=False):
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState

    from .pusht_active_entry import _restore_active_route
    from .pusht_active_entry import _state as validate_state
    from .pusht_lateral_route import bootstrap_driver
    from .pusht_pose_replan import _fresh_entry

    latest = load_source(source, audit, metadata, allow_completed=allow_completed)
    chain, seen, current = [], set(), latest
    while True:
        _require(
            current["report_path"] not in seen and len(chain) < 32, "bounded noncyclic resume chain"
        )
        seen.add(current["report_path"])
        chain.append(current)
        report = current["report"]
        if report["schema"] == "phymeo-pusht-lateral-route-v1":
            break
        prov = report["provenance"]
        base = current["report_path"].parent
        paths = []
        for name in ("parent_report", "parent_audit", "checkpoint_metadata"):
            path = (base / prov[name + "_path"]).resolve()
            _require(_digest(path) == prov[name + "_sha256"], "parent chain " + name)
            paths.append(path)
        current = load_source(*paths)
    root = chain[-1]
    original = root["report"]
    prov = original["provenance"]
    prior = Path(prov["prior_summary_path"]).resolve()
    _require(_digest(prior) == prov["prior_summary_sha256"], "original controller source")
    entry = _fresh_entry(
        prior, {"metadata": {"state": original["initial_state"]}}, Path(route_output)
    )
    driver = entry.driver
    bootstrap_driver(
        driver,
        original["initial_state"],
        original["start_tick"],
        original["controller_bootstrap"]["source_budget_counts"],
    )
    _require(
        plain(driver.summary()) == original["controller_initial"],
        "fresh original controller boundary",
    )
    record = deepcopy(original["controller_final"]["routes"][0])
    _require(
        len(original["controller_final"]["routes"]) == 1
        and record["sha256"] == original["active_route"]["sha256"],
        "one original ordinary route",
    )
    with gzip.open(root["route_path"], "rt", encoding="utf-8") as stream:
        saved_route = json.load(stream)
    copyfile(root["route_path"], driver.output / record["file"])
    _restore_active_route(driver, record, saved_route, original["initial_state"])
    state, history, previous = original["initial_state"], None, None
    for item in reversed(chain):
        report, meta = item["report"], item["metadata"]
        _require(
            report["initial_state"] == state
            and report["start_tick"] == driver.tick
            and report["active_route"]["sha256"] == original["active_route"]["sha256"]
            and report["goal"] == original["goal"],
            "resume chain initial state/route/Goal",
        )
        count = 0
        for row, callback in accepted_pairs(
            _events(item), state, original["source_reference_tick_origin"]
        ):
            validate_state(entry.model, row["input_state"])
            validate_state(entry.model, row["output_state"])
            _require(
                row["phase"] == "regular" and row["original_input_tick"] == driver.tick,
                "same ordinary route callback",
            )
            history = _replay_callback(entry, row, callback, history)
            previous = dict(deepcopy(row), **deepcopy(callback))
            state = row["output_state"]
            count += 1
        _require(
            count == report["recorded_steps"]
            and state == meta["state"]
            and plain(history) == meta["goal_history"]
            and plain(driver.summary()) == meta["controller"],
            "complete restored HOST boundary",
        )
    entry.state = deepcopy(state)
    entry.history = history
    entry.previous = previous
    entry.tick = driver.tick
    entry.initial = MuJoCoInitialState(entry.model.qpos0, np.zeros(entry.model.nv), time=0.0)
    entry.provenance = dict(
        latest["report"]["provenance"],
        parent_report_path=str(latest["report_path"]),
        parent_report_sha256=_digest(latest["report_path"]),
        parent_audit_path=str(latest["audit_path"]),
        parent_audit_sha256=_digest(latest["audit_path"]),
        checkpoint_metadata_path=str(latest["metadata_path"]),
        checkpoint_metadata_sha256=_digest(latest["metadata_path"]),
        parent_journal_path=str(latest["journal_path"]),
        parent_journal_sha256=latest["report"]["journal"]["sha256"],
        source_model_sha256=entry.scene.source_sha256,
        source_reference_tick_origin=original["source_reference_tick_origin"],
        goal_tick_origin=0,
        history_restoration="saved accepted callbacks only; zero prior physics steps",
    )
    latest["root_report"] = original
    latest["parent_physical_attempts"] = sum(item["report"]["physical_attempts"] for item in chain)
    return entry, latest


def restore_entry(scene, batch, entry, bound):
    from phymeo.backends.mujoco_heterogeneous_area_scene import (
        MuJoCoTyxHeterogeneousAreaSceneCheckpoint,
    )

    from .pusht_active_checkpoint import require_saved_state

    _require(
        scene.asset_sha256 == bound["metadata"]["asset_sha256"], "unchanged all-lateral material"
    )
    scene.restore(
        MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(bound["capsule_path"].read_bytes()),
        expected_revision=batch.checkpoint_revision,
    )
    snapshot = batch.integration_snapshot()[6]
    require_saved_state(snapshot, entry.state)
    _require(
        _words(snapshot.q, entry.driver.last_q)
        and entry.driver.tick == bound["metadata"]["source_tick"],
        "restored driver exact physical endpoint",
    )
    receipt = snapshot.last_step
    _require(
        receipt is not None
        and receipt.tick == snapshot.tick
        and receipt.output_time == snapshot.time,
        "restored preceding accepted receipt",
    )
    (pair,) = receipt.raw_contact_fields["area_friction"]
    _require(
        _words(pair["normal_force"], entry.previous["area"]["raw_normal_force_n"])
        and _words(
            np.asarray(pair["force_on_a"]).sum(axis=0),
            entry.previous["area"]["total_force_on_pusher_world_n"],
        ),
        "restored previous area load source",
    )
    return dict(
        checkpoint_sha256=_digest(bound["capsule_path"]),
        restored_local_tick=snapshot.tick,
        absolute_input_tick=entry.tick,
        time=snapshot.time,
        previous_receipt_bound=True,
        previous_tool_force_world_n=np.asarray(pair["force_on_a"]).sum(axis=0).tolist(),
        controller_callbacks_replayed=True,
        prior_physics_steps_replayed=0,
    )


def _save_first(output, before, after, geometry, command, controller, measurement, frame, goal, qa):
    path = output.with_suffix(".first-step.json.gz")
    payload = dict(
        schema="phymeo-pusht-lateral-resume-first-step-v1",
        publication_frame="rigid-origin",
        input_state=_state(before, qa),
        output_state=_state(after, qa),
        control=command,
        geometry=geometry,
        receipt=after.last_step,
        previous_receipt=before.last_step,
        controller=controller,
        measurement=measurement,
        goal_frame=frame,
        goal_result=goal,
    )
    with gzip.open(path, "xt", encoding="utf-8") as stream:
        json.dump(plain(payload), stream, allow_nan=False, separators=(",", ":"))
    return dict(
        file=path.name,
        sha256=_digest(path),
        input_tick=command["tick"],
        output_tick=command["output_tick"],
    )


def run(source, audit, metadata, output, *, steps=None, cache_root=None):
    from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
    from phymeo.backends.mujoco_heterogeneous_area_scene import (
        MuJoCoTyxHeterogeneousAreaPressureScene,
    )

    from .pusht_active_route import (
        _controller_boundary,
        _first_binding,
        _save_final_checkpoint,
        active_command,
    )
    from .pusht_lateral_route import accumulate_measurement
    from .pusht_load_compensation import PushLoadCompensation
    from .pusht_loaded_release import _command_record, attempt_release_candidate
    from .pusht_public_feedback import accepted_measurement_and_goal
    from .pusht_public_scene import _step_record, build_scene_spec, build_side_asset

    output = Path(output).resolve()
    journal_path = output.with_suffix(".trials.jsonl")
    _require(not output.exists() and not journal_path.exists(), "fresh outputs")
    report = dict(
        schema=SCHEMA,
        status="running",
        world=6,
        side_coverage="all-lateral",
        publication_frame="rigid-origin",
        control_variant="load-compensation-half",
        physical_attempts=0,
        physical_trial_steps=0,
        accepted_steps=0,
        recorded_steps=0,
        rejected_trials=0,
        metrics={},
        positive_load_steps=0,
        peak_normal_load_n=None,
        journal=dict(file=journal_path.name, records=0, sha256=hashlib.sha256().hexdigest()),
    )
    phase, index, history = "prepare_resume", None, None
    digest = hashlib.sha256()
    with output.open("x+", encoding="utf-8") as stream, journal_path.open("xb") as journal:

        def emit(kind, payload, index=None):
            raw = (
                json.dumps(
                    dict(schema=EVENT, event=kind, trial_index=index, payload=plain(payload)),
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
            journal.write(raw)
            journal.flush()
            digest.update(raw)
            report["journal"].update(
                records=report["journal"]["records"] + 1, sha256=digest.hexdigest()
            )

        _write_progress(stream, report)
        try:
            entry, bound = prepare_entry(source, audit, metadata, output.with_suffix(".routes"))
            driver, model, history = entry.driver, entry.model, entry.history
            old = bound["report"]
            original = bound["root_report"]
            remaining = old["active_route"]["end_tick"] - entry.tick
            allowance = original["step_budget"] - bound["parent_physical_attempts"]
            limit = min(remaining, allowance) if steps is None else steps
            _require(
                type(limit) is int and 1 <= limit <= min(remaining, allowance),
                "original remaining route/attempt budget",
            )
            active = dict(
                old["active_route"],
                file=str((driver.output / driver.routes[-1]["file"]).resolve()),
                route_index=len(driver.routes),
            )
            report.update(
                start_tick=entry.tick,
                local_tick_origin=entry.state["local_tick"],
                source_reference_tick_origin=original["source_reference_tick_origin"],
                goal_tick_origin=0,
                provenance=entry.provenance,
                initial_state=deepcopy(entry.state),
                final_state=deepcopy(entry.state),
                goal=deepcopy(old["goal"]),
                goal_history_initial=plain(history),
                controller_initial=plain(driver.summary()),
                active_route=active,
                step_budget=limit,
                original_step_budget=original["step_budget"],
                parent_physical_attempts=bound["parent_physical_attempts"],
                selection_policy=original["selection_policy"],
                material_admission=original["material_admission"],
            )
            spec, asset = (
                build_scene_spec(entry.scene.mjcf, side_coverage="all-lateral"),
                build_side_asset(model, side_coverage="all-lateral"),
            )
            qa = int(model.jnt_qposadr[model.joint("object-free").id])
            phase = "construct_and_restore"
            with MuJoCoTyxHeterogeneousBatch(
                (spec,),
                (MuJoCoTyxWorldSpec(6, spec.key, entry.initial),),
                mode="cuda",
                cache_root=cache_root,
            ) as batch:
                scene = MuJoCoTyxHeterogeneousAreaPressureScene(
                    batch, {spec.key: (asset,)}, publication_frame="rigid-origin"
                )
                report["checkpoint_restore"] = restore_entry(scene, batch, entry, bound)
                compensate = PushLoadCompensation(entry, identity=asset.identity)
                emit("initialization", report)
                try:
                    while report["physical_attempts"] < limit:
                        before = batch.integration_snapshot()[6]
                        tick = reference_tick(entry, before)
                        phase = "active_reference"
                        control, controller = active_command(entry, before, compensate=compensate)
                        command = _command_record(control, tick)
                        if report["physical_attempts"] == 0:
                            report["first_command_binding"] = _first_binding(
                                entry, before, control, controller
                            )
                            if bound["retry_command"] is not None:
                                report["first_command_binding"].update(
                                    bind_retry_command(control, bound["retry_command"], tick)
                                )
                            else:
                                report["first_command_binding"][
                                    "matches_parent_rejected_command"
                                ] = None
                        index = report["physical_attempts"]
                        report["physical_attempts"] += 1
                        phase = "scene_step"
                        trial = attempt_release_candidate(
                            scene,
                            batch,
                            {6: control},
                            lambda *_: dict(safe=True, reason="public scene acceptance"),
                        )
                        report["physical_trial_steps"] += 1
                        report["accepted_steps"] += 1
                        after = trial["candidate_output"][6]
                        geometry = trial["candidate_geometry"][6][0]
                        report["final_state"] = _state(after, qa)
                        report["unrecorded_accepted_step"] = dict(
                            input_state=_state(before, qa),
                            output_state=_state(after, qa),
                            control=command,
                        )
                        phase = "candidate_record"
                        row = _step_record(
                            model, asset, before, after, geometry, command, qa, trial["wall_s"]
                        )
                        row.pop("stored_control_matches_original")
                        (pair,) = after.last_step.raw_contact_fields["area_friction"]
                        row["area"]["raw_normal_force_n"] = np.asarray(
                            pair["normal_force"]
                        ).tolist()
                        measurement, frame, goal = accepted_measurement_and_goal(
                            entry.scene,
                            model,
                            6,
                            tick,
                            before,
                            after,
                            history,
                            goal_tick=before.tick,
                        )
                        row.update(
                            index=index,
                            trial_index=index,
                            phase="regular",
                            controller=plain(controller),
                            side_coverage="all-lateral",
                            material_identity=plain(asset.identity),
                            stored_control_matches_submitted=True,
                            selected=True,
                            restored=False,
                            rollback_verification=None,
                            measurement=plain(measurement),
                            assessment=trial["assessment"],
                        )
                        if index == 0:
                            report["first_step_evidence"] = _save_first(
                                output,
                                before,
                                after,
                                geometry,
                                command,
                                controller,
                                measurement,
                                frame,
                                goal,
                                qa,
                            )
                        emit("candidate", row, index)
                        phase = "accepted_controller_and_goal"
                        metrics = accumulate_measurement(
                            report["metrics"],
                            row["input_state"],
                            row["output_state"],
                            plain(measurement),
                            tick,
                        )
                        feedback = driver.accepted_measurement(
                            tick + 1, after.q, measurement, goal, input_q=before.q
                        )
                        history = goal.history
                        callback = dict(
                            goal_frame=plain(frame),
                            goal_result=plain(goal),
                            feedback_accepted=plain(feedback),
                            driver_state_after=_controller_boundary(driver),
                            cumulative_actual=metrics,
                        )
                        emit("accepted_callback", callback, index)
                        report["recorded_steps"] += 1
                        report.pop("unrecorded_accepted_step", None)
                        report["metrics"] = metrics
                        entry.previous = dict(row, **callback)
                        fn = row["area"]["normal_load_n"]
                        report["peak_normal_load_n"] = (
                            fn
                            if report["peak_normal_load_n"] is None
                            else max(fn, report["peak_normal_load_n"])
                        )
                        report["positive_load_steps"] += int(fn > 0)
                        report.update(last_goal=plain(goal), goal_success=bool(goal.success))
                        if index == 0:
                            _write_progress(stream, report)
                        if driver.stopped:
                            report.update(
                                status="controller_stopped", controller_status=driver.reason
                            )
                            break
                        if driver.active is None:
                            _require(driver.tick == active["end_tick"], "original route end tick")
                            report["status"] = "route_complete"
                            break
                        if report["recorded_steps"] % 32 == 0:
                            _write_progress(stream, report)
                    else:
                        report["status"] = "step_budget"
                except Exception as error:
                    failed = getattr(error, "release_trial", None)
                    if failed is not None:
                        report["physical_trial_steps"] += int(failed["physics_accepted"])
                        report["failed_candidate"] = dict(
                            physics_accepted_before_restore=failed["physics_accepted"],
                            restored=failed["restored"],
                            rollback_verification=failed["rollback_verification"],
                            input_states={
                                str(k): _state(v, qa) for k, v in failed["input"].items()
                            },
                            candidate_output_states=None
                            if failed["output"] is None
                            else {str(k): _state(v, qa) for k, v in failed["output"].items()},
                            control=command,
                        )
                    raise
                finally:
                    report.update(
                        controller_final=plain(driver.summary()), goal_history_final=plain(history)
                    )
                    try:
                        report["final_checkpoint"] = _save_final_checkpoint(
                            scene, batch, entry, output, report, history
                        )
                    except Exception as error:
                        report["checkpoint_save_error"] = _error_chain(error)
        except Exception as error:
            report.update(status="partial_failure", failure_stage=phase, errors=_error_chain(error))
            emit(
                "failure",
                {
                    k: report.get(k)
                    for k in (
                        "failure_stage",
                        "errors",
                        "failed_candidate",
                        "unrecorded_accepted_step",
                    )
                },
                index,
            )
        emit("final", {k: v for k, v in report.items() if k != "journal"})
        _write_progress(stream, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "audit", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    result = run(
        args.source,
        args.audit,
        args.checkpoint,
        args.output,
        steps=args.steps,
        cache_root=args.cache_root,
    )
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ("status", "accepted_steps", "recorded_steps", "failure_stage", "errors")
            }
        )
    )
    raise SystemExit(
        0 if result["status"] in ("route_complete", "controller_stopped", "step_budget") else 1
    )


if __name__ == "__main__":
    main()
