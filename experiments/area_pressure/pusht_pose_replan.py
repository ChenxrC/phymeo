"""One bounded ordinary replan from a completed capsule or a registered new pose.

A continues the exact terminal owner state after guarded withdrawal. B is a new
reset episode: default robot q0, all velocities zero, terminal object XY, original
reset height and pure yaw +0.50 rad. The original Goal is unchanged. These cases
are separate rollouts, not a comparison that isolates only initial object yaw.
"""

import argparse
import gzip
import hashlib
import json
import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.run_manipulation_control import plain

H = 0.001
RESET_YAW = 0.50
SELECTION_POLICY = "joint-pose-descent"


def _require(ok, detail):
    if not ok:
        raise ValueError("pose replan: " + detail)


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reset_initial(model, terminal_state):
    """The preregistered new episode declaration; never a continuation injection."""
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState

    source = np.asarray(terminal_state["q"], dtype=float)
    _require(source.shape == (model.nq,) and np.isfinite(source).all(), "terminal reset XY source")
    qa = int(model.jnt_qposadr[model.joint("object-free").id])
    q = np.asarray(model.qpos0, dtype=float).copy()
    q[qa : qa + 2] = source[qa : qa + 2]
    q[qa + 3 : qa + 7] = (math.cos(RESET_YAW / 2), 0.0, 0.0, math.sin(RESET_YAW / 2))
    return MuJoCoInitialState(q, np.zeros(model.nv), time=0.0)


def require_separated(evidence, support_keys):
    """Require zero actual load and strict output separation per non-support pair."""
    supports = set(support_keys)
    _require(
        all(key.kind == "native-geometry" and "support" in key.identity for key in supports),
        "only the original native table support is exempt",
    )
    rows = tuple(row for row in evidence.pairs if row.key not in supports)
    _require(bool(rows), "complete non-support geometry/material catalogue required")
    for row in rows:
        _require(
            row.normal_load_n == 0.0
            and row.output_depth_m == 0.0
            and row.output_contact is False
            and row.output_strictly_separated is True,
            "ordinary replan requires strict zero-load separation for " + repr(row.key),
        )
    return dict(
        strictly_separated=True,
        non_support_pairs=len(rows),
        output_tick=getattr(evidence, "output_tick", None),
        output_time=getattr(evidence, "output_time", None),
        qualification="sampled output geometry and this actual input-geometry load; no swept-clearance claim",
    )


def release_seed_view(entry, *, source_report_sha256, source_journal_sha256):
    """Expose the bound saved callback to the legacy release-seed reader.

    This is a copied record view, not a new measurement: both saved force sides,
    the COM moment, and their accepted-step-input phase remain unchanged.
    """
    previous = entry.previous
    _require(isinstance(previous, dict), "accepted release predecessor required")
    _require(
        previous.get("schema") == "phymeo-pusht-active-route-step-v1",
        "release predecessor must be an actual active-route row",
    )
    measurement = previous.get("feedback_accepted", {}).get("measurement")
    _require(isinstance(measurement, dict), "accepted callback measurement required")
    before, after = previous["input_state"], previous["output_state"]
    input_tick, output_tick = previous["original_input_tick"], previous["original_output_tick"]
    _require(
        output_tick == input_tick + 1 == entry.tick
        and after["local_tick"] == before["local_tick"] + 1
        and plain(after) == plain(entry.state),
        "release predecessor endpoint and adjacent ticks",
    )
    for name in ("q", "v"):
        saved = np.asarray(after[name], dtype=np.float32)
        current = np.asarray(entry.state[name], dtype=np.float32)
        _require(
            np.isfinite(saved).all()
            and np.array_equal(saved.view(np.uint32), current.view(np.uint32)),
            "release predecessor endpoint words " + name,
        )
    area = previous["area"]
    _require(
        measurement.get("world") == 6
        and measurement.get("input_tick") == input_tick
        and measurement.get("output_tick") == output_tick
        and measurement.get("phase") == "accepted-step-input"
        and measurement.get("time_s") == before["time"]
        and np.isfinite((before["time"], after["time"])).all()
        and after["time"] > before["time"]
        and measurement.get("native_contacts") == 0
        and measurement.get("area_samples") == area["samples"],
        "release callback world, force phase, and actual sample identity",
    )
    force_a = np.asarray(area["total_force_on_pusher_world_n"], dtype=float)
    force_b = np.asarray(measurement["group_force_world"], dtype=float)
    moment_b = np.asarray(measurement["group_moment_com_world"], dtype=float)
    _require(
        all(
            value.shape == (3,) and np.isfinite(value).all()
            for value in (force_a, force_b, moment_b)
        )
        and np.array_equal(force_a, -force_b)
        and math.isfinite(area["normal_load_n"])
        and area["normal_load_n"] >= 0,
        "release seed actual bilateral forces and COM moment",
    )
    _require(
        "measurement" not in previous or plain(previous["measurement"]) == plain(measurement),
        "conflicting saved release measurement",
    )
    for digest in (source_report_sha256, source_journal_sha256):
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest),
            "release measurement source digest",
        )
    adapted = deepcopy(previous)
    adapted["measurement"] = deepcopy(measurement)
    provenance = dict(
        source_field="feedback_accepted.measurement",
        source_report_sha256=source_report_sha256,
        source_journal_sha256=source_journal_sha256,
        record_schema=previous["schema"],
        record_index=previous["index"],
        world=measurement["world"],
        input_tick=input_tick,
        output_tick=output_tick,
        input_time=before["time"],
        output_time=after["time"],
        force_phase=measurement["phase"],
        qualification="unchanged saved previous input-geometry load; not a new endpoint measurement",
    )
    return replace(entry, previous=adapted), provenance


def run_independent_cases(cases):
    """Always dispatch the other registered case after a case-local failure."""
    result = {}
    for key, execute in cases.items():
        try:
            result[key] = execute()
        except Exception as error:
            result[key] = dict(key=key, status="partial_failure", errors=[str(error)])
    return result


def _fresh_entry(prior_path, bound, route_output):
    import mujoco as mj

    from phymeo.training.manipulation_tasks import manipulation_scenes
    from scripts.run_manipulation_push import SOURCE_SHA
    from scripts.run_manipulation_push_feedback import PushFeedbackDriver, hold_plan

    from .pusht_release_entry import ReleaseEntry

    prior = json.loads(Path(prior_path).read_text("utf-8"))
    source = prior["provenance"]
    candidate_path = Path(source["candidate_source"])
    _require(
        _digest(candidate_path) == source["candidate_source_sha256"] == SOURCE_SHA,
        "original candidate source",
    )
    original = next(s for s in manipulation_scenes() if s.key == "pusht-planar")
    _require(original.source_sha256 == source["source_sha256"], "original reset scene")
    model = mj.MjModel.from_xml_string(original.mjcf)
    dataset = json.loads(candidate_path.read_text("utf-8"))
    saved = next(
        row for row in dataset["rows"] if row["summary"]["world"] == prior["worlds"]["open"]
    )
    driver = PushFeedbackDriver(original, model, route_output, saved)
    pd, _, _ = hold_plan(original, model)
    initial = reset_initial(model, bound["metadata"]["state"])
    return ReleaseEntry(original, model, driver, pd, None, {}, None, 0, initial, {})


def _plan_ordinary(entry, snapshot, tick):
    """Exactly one explicit policy selection, using the original path/IK gates."""
    from phymeo.control.manipulation_push_objective import propose_planar_pushes

    driver = entry.driver
    _require(
        driver.active is None
        and not driver.stopped
        and driver.tick == tick
        and np.array_equal(driver.last_q, snapshot.q)
        and not driver.planner.allow_loaded_withdrawal,
        "ordinary planner must start at the separated accepted controller boundary",
    )
    position, rotation, pusher = driver.pose(snapshot.q)
    proposals = propose_planar_pushes(
        driver.query,
        reference_position=driver.ref_position,
        reference_rotation=driver.ref_rotation,
        current_position=position,
        current_rotation=rotation,
        target_xy=driver.target.position[:2],
        target_yaw=driver.target.yaw,
        pusher_height_m=float(pusher[2]),
        position_tolerance_m=driver.goal.position_tolerance_m,
        yaw_tolerance_rad=driver.goal.yaw_tolerance_rad,
        selection_policy=SELECTION_POLICY,
    )
    route = driver.planner.plan(snapshot.q, proposals, previous_direction=None)
    path = driver.output / f"route-{len(driver.routes) + 1:02}.json.gz"
    with gzip.open(path, "xt", encoding="utf-8") as stream:
        json.dump(
            plain(
                dict(
                    tick=tick,
                    q=snapshot.q,
                    v=snapshot.v,
                    cost=driver.cost(snapshot.q),
                    proposals=proposals,
                    route=route,
                )
            ),
            stream,
            allow_nan=False,
        )
    record = dict(
        file=path.name,
        sha256=_digest(path),
        tick=tick,
        status=route.status,
        ik_calls=route.ik_calls,
        path_calls=route.path_calls,
    )
    driver.routes.append(record)
    if route.status == "sampled_reference_ready":
        driver.active = (tick, route)
        driver.previous_direction = None
    else:
        driver.stopped, driver.reason = True, route.status
    return route, dict(
        route_file=str(path.resolve()),
        route_sha256=record["sha256"],
        source_q=plain(snapshot.q),
        source_v=plain(snapshot.v),
        tick=tick,
        status=route.status,
        selection_policy=SELECTION_POLICY,
    )


def _run_case(
    key, output, prior, release, source, audit, metadata, bound, *, step_budget, cache_root
):
    from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
    from phymeo.backends.mujoco_heterogeneous_area_scene import (
        MuJoCoTyxHeterogeneousAreaPressureScene,
    )
    from phymeo.backends.mujoco_runtime import MuJoCoControl
    from phymeo.control.manipulation_release_monitor import ReleaseMonitor

    from .pusht_active_resume import prepare_resume_entry, restore_resume_entry
    from .pusht_active_route import (
        _controller_boundary,
        _first_binding,
        _save_final_checkpoint,
        active_command,
    )
    from .pusht_load_compensation import PushLoadCompensation
    from .pusht_loaded_release import (
        _command_record,
        _loaded_route,
        _release_control,
        attempt_release_candidate,
    )
    from .pusht_public_feedback import (
        _original_action_transport,
        accepted_measurement_and_goal,
        command_from_snapshot,
    )
    from .pusht_public_scene import (
        _error_chain,
        _state,
        _step_record,
        _write_progress,
        build_scene_spec,
        build_side_asset,
    )
    from .pusht_release_evidence import build_release_seed, build_release_step, release_support_keys

    output = Path(output).resolve()
    journal_path = output.with_suffix(".trials.jsonl")
    journal_hash = hashlib.sha256()
    mode_name = "terminal-checkpoint" if key == "A" else "independent-pose-reset"
    report = dict(
        schema="phymeo-pusht-pose-replan-case-v1",
        key=key,
        initialization_mode=mode_name,
        status="running",
        world=6,
        step_budget=step_budget,
        physical_attempts=0,
        physical_trial_steps=0,
        accepted_steps=0,
        recorded_steps=0,
        rejected_trials=0,
        planner_calls=0,
        regular_plan_ready=False,
        goal_success=False,
        journal=dict(file=journal_path.name, records=0, sha256=journal_hash.hexdigest()),
        control_variant="load-compensation-half",
        selection_policy=SELECTION_POLICY,
        force_phase="accepted-step-input-geometry",
        provenance=dict(
            source_report_path=str(source),
            source_report_sha256=_digest(source),
            source_audit_path=str(audit),
            source_audit_sha256=_digest(audit),
            checkpoint_metadata_path=str(metadata),
            checkpoint_metadata_sha256=_digest(metadata),
            source_checkpoint_sha256=bound["metadata"]["checkpoint_sha256"],
            scene_asset_sha256=bound["metadata"]["asset_sha256"],
            initialization_mode=mode_name,
        ),
    )
    phase, trial_index = "prepare_case", None
    history = None
    with output.open("x+", encoding="utf-8") as stream, journal_path.open("xb") as journal:

        def emit(event, payload, index=None):
            packet = dict(
                schema="phymeo-pusht-pose-replan-event-v1",
                event=event,
                trial_index=index,
                payload=plain(payload),
            )
            encoded = (json.dumps(packet, allow_nan=False, separators=(",", ":")) + "\n").encode()
            journal.write(encoded)
            journal.flush()
            journal_hash.update(encoded)
            report["journal"].update(
                records=report["journal"]["records"] + 1, sha256=journal_hash.hexdigest()
            )

        _write_progress(stream, report)
        try:
            if key == "A":
                entry, local_bound = prepare_resume_entry(
                    prior,
                    release,
                    source,
                    audit,
                    metadata,
                    output.with_suffix(".routes"),
                    control_variant="load-compensation-half",
                    allow_completed=True,
                )
                _require(entry.driver.active is None, "A requires the completed original route")
            else:
                entry = _fresh_entry(prior, bound, output.with_suffix(".routes"))
            model, driver, world = entry.model, entry.driver, 6
            spec, asset = build_scene_spec(entry.scene.mjcf), build_side_asset(model)
            qa = int(model.jnt_qposadr[model.joint("object-free").id])
            report["goal"] = dict(
                target_xy=plain(driver.target.position[:2]),
                target_yaw=driver.target.yaw,
                position_tolerance_m=driver.goal.position_tolerance_m,
                yaw_tolerance_rad=driver.goal.yaw_tolerance_rad,
            )
            _require(driver.target.yaw == 0.2, "registered original Goal yaw")
            phase = "construct_scene"
            with MuJoCoTyxHeterogeneousBatch(
                (spec,),
                (MuJoCoTyxWorldSpec(world, spec.key, entry.initial),),
                mode="cuda",
                cache_root=cache_root,
            ) as batch:
                scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (asset,)})
                _require(
                    scene.asset_sha256 == bound["metadata"]["asset_sha256"],
                    "unchanged material asset",
                )
                if key == "A":
                    phase = "restore_terminal"
                    report["checkpoint_restore"] = restore_resume_entry(
                        scene, batch, entry, local_bound
                    )
                initial = batch.integration_snapshot()[world]
                if key == "B":
                    _require(
                        initial.tick == 0 and initial.time == 0.0 and initial.last_step is None,
                        "new reset cannot inherit a receipt or clock",
                    )
                    entry.state = _state(initial, qa)
                    entry.provenance = dict(report["provenance"])
                    # Original cold driver initialization at its true reset state;
                    # tick zero only produces its original settle reference.
                    driver.reference(0, initial.q, initial.v)
                    _require(
                        driver.active is None and not driver.routes,
                        "reset initialization cannot plan",
                    )
                    report["provenance"]["reset_declaration"] = dict(
                        q=plain(entry.initial.q),
                        v=plain(entry.initial.v),
                        time=entry.initial.time,
                        object_yaw_rad=RESET_YAW,
                        source_xy_world_m=bound["metadata"]["state"]["q"][qa : qa + 2],
                        robot="original-model-qpos0",
                        velocity="zero",
                        object_z="original-model-qpos0",
                        prior_receipt_inherited=False,
                    )
                report.update(
                    initial_state=_state(initial, qa),
                    final_state=_state(initial, qa),
                    start_tick=entry.tick,
                    local_tick_origin=initial.tick,
                    controller_initial=plain(driver.summary()),
                    episode_scope="continued source Goal history"
                    if key == "A"
                    else "new owner and new Goal history; local episode id zero",
                )
                origin = entry.tick - entry.state["local_tick"]
                evidence_entry = replace(entry, tick=origin)
                history = entry.history
                compensate = PushLoadCompensation(entry)
                mode, release_start, release_route, monitor = "smooth", None, None, None
                last_evidence, supports, regular_started = None, (), False
                run_phase = "withdraw" if key == "A" else "settle"
                emit("initialization", report)
                try:
                    if key == "A":
                        phase = "qualify_release"
                        seed_entry, seed_provenance = release_seed_view(
                            entry,
                            source_report_sha256=_digest(source),
                            source_journal_sha256=local_bound["report"]["journal"]["sha256"],
                        )
                        report["seed_measurement_provenance"] = seed_provenance
                        seed = build_release_seed(seed_entry, asset)
                        supports = release_support_keys(seed)
                        monitor = ReleaseMonitor(support_keys=supports, max_steps=320)
                        monitor.start(seed)
                        release_route, proposals = _loaded_route(entry, initial)
                        release_start = initial.tick
                        release_path = output.with_suffix(".release-route.json")
                        with release_path.open("x", encoding="utf-8") as f:
                            json.dump(
                                plain(
                                    dict(
                                        tick=entry.tick,
                                        q=initial.q,
                                        v=initial.v,
                                        route=release_route,
                                        proposals=proposals,
                                    )
                                ),
                                f,
                                allow_nan=False,
                            )
                        report["release_reference"] = dict(
                            file=release_path.name,
                            sha256=_digest(release_path),
                            nodes=plain(release_route.nodes),
                        )
                        report["seed_evidence"] = plain(seed)
                        emit(
                            "release_reference",
                            dict(
                                reference=report["release_reference"],
                                seed=seed,
                                seed_measurement_provenance=seed_provenance,
                            ),
                        )
                    while report["physical_attempts"] < step_budget:
                        before = batch.integration_snapshot()[world]
                        tick = origin + before.tick
                        _require(
                            driver.tick == tick and np.array_equal(driver.last_q, before.q),
                            "causal accepted controller boundary",
                        )
                        if run_phase == "plan":
                            phase = "ordinary_replan"
                            report["separation_proof"] = require_separated(last_evidence, supports)
                            _require(
                                report["planner_calls"] == 0, "only one ordinary replan per case"
                            )
                            report["planner_calls"] += 1
                            route, plan_record = _plan_ordinary(entry, before, tick)
                            report["ordinary_route"] = plan_record
                            emit("planner", plan_record)
                            if route.status != "sampled_reference_ready":
                                report["status"] = "planning_rejected"
                                break
                            duration_steps = round(route.nodes[-1][0] / H)
                            _require(
                                duration_steps > 0
                                and abs(duration_steps * H - route.nodes[-1][0]) <= 1e-12,
                                "finite tick-aligned ordinary reference",
                            )
                            _require(
                                duration_steps <= step_budget - report["physical_attempts"],
                                "remaining budget must cover the whole admitted reference",
                            )
                            report.update(
                                regular_plan_ready=True,
                                active_route=dict(
                                    start_tick=tick,
                                    end_tick=tick + duration_steps,
                                    nodes=plain(route.nodes),
                                    file=plan_record["route_file"],
                                    sha256=plan_record["route_sha256"],
                                    completed_routes_before=driver.completed,
                                ),
                            )
                            run_phase = "regular"
                        if run_phase == "withdraw" and before.tick - release_start >= 320:
                            report["status"] = "release_step_budget"
                            break
                        phase = run_phase + "_reference"
                        modes = (
                            (mode, "constant_rate")
                            if run_phase == "withdraw"
                            and before.tick == release_start
                            and mode == "smooth"
                            else (mode,)
                        )
                        selected = False
                        for candidate_mode in modes:
                            if report["physical_attempts"] >= step_budget:
                                break
                            if run_phase == "withdraw":
                                control, controller = _release_control(
                                    entry,
                                    before,
                                    release_route.nodes,
                                    (before.tick - release_start) * H,
                                    candidate_mode,
                                )
                            elif run_phase == "settle":
                                control, controller = command_from_snapshot(
                                    driver, entry.pd, tick, before, start_tick=origin
                                )
                                actual, transport = _original_action_transport(control.ctrl)
                                control = MuJoCoControl(
                                    actual,
                                    qfrc_applied=control.qfrc_applied,
                                    xfrc_applied=control.xfrc_applied,
                                )
                                controller.update(
                                    transport=transport,
                                    generated_channels=_command_record(control, tick),
                                )
                            else:
                                control, controller = active_command(
                                    entry, before, compensate=compensate
                                )
                                if not regular_started:
                                    report["first_regular_command_binding"] = _first_binding(
                                        entry, before, control, controller
                                    )
                                    regular_started = True
                            command, captured = _command_record(control, tick), {}

                            def assess(inputs, outputs, geometry):
                                need_evidence = run_phase == "withdraw" or (
                                    run_phase == "settle" and outputs[world].tick == 200
                                )
                                if need_evidence:
                                    evidence = build_release_step(
                                        evidence_entry,
                                        asset,
                                        inputs[world],
                                        outputs[world],
                                        geometry[world][0],
                                        diagnostics=captured.setdefault("geometry_comparison", {}),
                                    )
                                    captured["evidence"] = evidence
                                if run_phase == "withdraw":
                                    decision = monitor.assess(
                                        replace(captured["evidence"], publication="trial")
                                    )
                                    return dict(
                                        safe=bool(decision.admissible),
                                        reason=decision.status,
                                        decision=decision,
                                    )
                                return dict(
                                    safe=True,
                                    reason="admitted_reference_or_original_settle",
                                    decision=None,
                                )

                            trial_index = report["physical_attempts"]
                            report["physical_attempts"] += 1
                            phase = "scene_step"
                            trial = attempt_release_candidate(
                                scene, batch, {world: control}, assess
                            )
                            report["physical_trial_steps"] += 1
                            after = trial["candidate_output"][world]
                            selected = trial["selected"]
                            if selected:
                                report["accepted_steps"] += 1
                                report["final_state"] = _state(after, qa)
                                report["unrecorded_accepted_step"] = dict(
                                    input_state=_state(before, qa),
                                    output_state=_state(after, qa),
                                    control=command,
                                )
                            else:
                                report["rejected_trials"] += 1
                            phase = "candidate_record"
                            row = _step_record(
                                model,
                                asset,
                                before,
                                after,
                                trial["candidate_geometry"][world][0],
                                command,
                                qa,
                                trial["wall_s"],
                            )
                            row.pop("stored_control_matches_original")
                            (pair,) = after.last_step.raw_contact_fields["area_friction"]
                            row["area"]["raw_normal_force_n"] = np.asarray(
                                pair["normal_force"]
                            ).tolist()
                            # Pure measurement/Goal evaluation for every real trial;
                            # only a selected trial publishes this history below.
                            measurement, frame, goal = accepted_measurement_and_goal(
                                entry.scene, model, world, tick, before, after, history
                            )
                            row.update(
                                index=trial_index,
                                trial_index=trial_index,
                                phase=run_phase,
                                trial_mode=candidate_mode if run_phase == "withdraw" else run_phase,
                                controller=plain(controller),
                                stored_control_matches_submitted=True,
                                selected=selected,
                                restored=trial["restored"],
                                rollback_verification=trial["rollback_verification"],
                                evidence=plain(captured.get("evidence")),
                                geometry_comparison=plain(captured.get("geometry_comparison")),
                                release_decision=plain(trial["assessment"]["decision"]),
                                measurement=plain(measurement),
                            )
                            emit("candidate", row, trial_index)
                            if not selected:
                                codes = {v.code for v in trial["assessment"]["decision"].violations}
                                if not codes or not codes <= {
                                    "inward_motion",
                                    "penetration_increased",
                                    "normal_load_increased",
                                }:
                                    break
                                continue
                            if run_phase == "withdraw":
                                mode = candidate_mode
                                decision = monitor.observe(captured["evidence"])
                                row["release_decision"] = plain(decision)
                                report["release_status"] = monitor.status
                            last_evidence = captured.get("evidence", last_evidence)
                            phase = "accepted_controller_and_goal"
                            feedback = driver.accepted_measurement(
                                tick + 1, after.q, measurement, goal, input_q=before.q
                            )
                            history = goal.history
                            callback = dict(
                                goal_frame=plain(frame),
                                goal_result=plain(goal),
                                feedback_accepted=plain(feedback),
                                driver_state_after=_controller_boundary(driver),
                                release_decision=row["release_decision"],
                            )
                            row.update(callback)
                            emit("accepted_callback", callback, trial_index)
                            report["recorded_steps"] += 1
                            report.pop("unrecorded_accepted_step", None)
                            report.update(last_goal=plain(goal), goal_success=bool(goal.success))
                            entry.previous = deepcopy(row)
                            if run_phase == "settle" and after.tick == 200:
                                supports = release_support_keys(last_evidence)
                                try:
                                    report["separation_proof"] = require_separated(
                                        last_evidence, supports
                                    )
                                except ValueError:
                                    phase = "qualify_release"
                                    monitor = ReleaseMonitor(support_keys=supports, max_steps=320)
                                    monitor.start(last_evidence)
                                    release_route, proposals = _loaded_route(entry, after)
                                    release_start, mode = after.tick, "smooth"
                                    release_path = output.with_suffix(".release-route.json")
                                    with release_path.open("x", encoding="utf-8") as f:
                                        json.dump(
                                            plain(
                                                dict(
                                                    tick=tick + 1,
                                                    q=after.q,
                                                    v=after.v,
                                                    route=release_route,
                                                    proposals=proposals,
                                                )
                                            ),
                                            f,
                                            allow_nan=False,
                                        )
                                    report["release_reference"] = dict(
                                        file=release_path.name,
                                        sha256=_digest(release_path),
                                        nodes=plain(release_route.nodes),
                                    )
                                    report["seed_evidence"] = plain(last_evidence)
                                    emit(
                                        "release_reference",
                                        dict(
                                            reference=report["release_reference"],
                                            seed=last_evidence,
                                        ),
                                    )
                                    run_phase = "withdraw"
                                else:
                                    run_phase = "plan"
                            elif (
                                run_phase == "withdraw"
                                and monitor.status == "released"
                                and (after.tick - release_start) * H >= release_route.nodes[-1][0]
                            ):
                                report["separation_proof"] = require_separated(
                                    last_evidence, supports
                                )
                                run_phase = "plan"
                            break
                        if not selected:
                            report["status"] = (
                                "release_candidate_rejected"
                                if run_phase == "withdraw"
                                else "step_budget"
                            )
                            break
                        if driver.stopped:
                            report.update(
                                status="controller_stopped", controller_status=driver.reason
                            )
                            break
                        if run_phase == "regular" and driver.active is None:
                            _require(
                                driver.tick == report["active_route"]["end_tick"],
                                "ordinary reference completion boundary",
                            )
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
                            control=plain(locals().get("command")),
                        )
                    raise
                finally:
                    report["controller_final"] = plain(driver.summary())
                    report["goal_history_final"] = plain(history)
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
                    key: report.get(key)
                    for key in (
                        "failure_stage",
                        "errors",
                        "failed_candidate",
                        "unrecorded_accepted_step",
                    )
                },
                trial_index,
            )
        emit("final", {key: value for key, value in report.items() if key != "journal"})
        _write_progress(stream, report)
    return report


def run(
    prior, release, source, audit, metadata, output, *, case="both", steps=4000, cache_root=None
):
    from .pusht_active_resume import load_resume_source

    _require(case in ("A", "B", "both"), "registered case selection")
    _require(type(steps) is int and 1 <= steps <= 4000, "finite 1..4000 physical-attempt budget")
    prior, release, source, audit, metadata, output = map(
        lambda p: Path(p).resolve(), (prior, release, source, audit, metadata, output)
    )
    bound = load_resume_source(
        source, audit, metadata, control_variant="load-compensation-half", allow_completed=True
    )
    _require(
        bound["metadata"]["controller"]["partial_route"] is None,
        "source must be the completed original route",
    )
    for path, name in ((prior, "prior_summary"), (release, "release_report")):
        _require(
            _digest(path) == bound["report"]["provenance"][name + "_sha256"],
            "bound original " + name,
        )
    selected = ("A", "B") if case == "both" else (case,)
    paths = {key: output.with_name(output.stem + "-" + key + ".json") for key in selected}
    _require(
        not output.exists() and all(not p.exists() for p in paths.values()),
        "new output files required",
    )
    reports = run_independent_cases(
        {
            key: (
                lambda key=key: _run_case(
                    key,
                    paths[key],
                    prior,
                    release,
                    source,
                    audit,
                    metadata,
                    bound,
                    step_budget=steps,
                    cache_root=cache_root,
                )
            )
            for key in selected
        }
    )
    cases = {}
    for key, report in reports.items():
        if not paths[key].exists():
            paths[key].write_text(json.dumps(report, allow_nan=False, indent=2), encoding="utf-8")
        cases[key] = dict(file=paths[key].name, sha256=_digest(paths[key]), summary=report)
    result = dict(
        schema="phymeo-pusht-pose-replan-v1",
        cases=cases,
        source=dict(
            summary_path=str(source),
            summary_sha256=_digest(source),
            audit_path=str(audit),
            audit_sha256=_digest(audit),
            checkpoint_metadata_path=str(metadata),
            checkpoint_metadata_sha256=_digest(metadata),
            checkpoint_sha256=bound["metadata"]["checkpoint_sha256"],
        ),
        qualification="A is continuous; B is an independently reset pose episode, not a yaw-only causal comparison",
    )
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("summary", "release", "source", "audit", "checkpoint", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--case", choices=("A", "B", "both"), default="both")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    result = run(
        args.summary,
        args.release,
        args.source,
        args.audit,
        args.checkpoint,
        args.output,
        case=args.case,
        steps=args.steps,
        cache_root=args.cache_root,
    )
    print(json.dumps({key: value["summary"]["status"] for key, value in result["cases"].items()}))
    raise SystemExit(
        0
        if all(
            value["summary"]["status"] in ("route_complete", "controller_stopped")
            for value in result["cases"].values()
        )
        else 1
    )


if __name__ == "__main__":
    main()
