"""One bounded all-lateral withdrawal/replan from a completed public capsule.

python -m experiments.area_pressure.pusht_lateral_replan --source suffix.json
  --audit suffix-audit.json --checkpoint suffix.final.scene.metadata.json --output NEW.json

Keeps original controller, Goal, material, cumulative budgets and stopping rules.
A long new reference may execute only its explicitly budgeted accepted prefix.
"""

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.run_manipulation_control import plain

from .pusht_active_resume import _digest
from .pusht_public_scene import H, _error_chain, _state, _write_progress

SCHEMA = "phymeo-pusht-lateral-replan-v1"
EVENT = "phymeo-pusht-lateral-replan-event-v1"


def _require(ok, reason):
    if not ok:
        raise ValueError("lateral replan: " + reason)


def remaining_budget(original, consumed, requested):
    _require(
        type(original) is int and type(consumed) is int and 0 <= consumed <= original,
        "original attempt accounting",
    )
    remaining = original - consumed
    if requested is None:
        return remaining
    _require(
        type(requested) is int and 1 <= requested <= remaining,
        "requested prefix cannot enlarge original budget",
    )
    return requested


def reference_budget(route, available):
    duration = float(route.nodes[-1][0])
    steps = round(duration / H)
    _require(
        steps > 0
        and abs(steps * H - duration) <= 1e-12
        and type(available) is int
        and available >= 0,
        "finite tick-aligned unchanged reference",
    )
    return dict(
        route_steps=steps,
        available_steps=available,
        complete_route_within_budget=steps <= available,
    )


def allow_constant_retry(mode, elapsed_steps, decision):
    codes = {item.code for item in decision.violations}
    return (
        mode == "smooth"
        and elapsed_steps == 0
        and bool(codes)
        and codes <= {"inward_motion", "penetration_increased", "normal_load_increased"}
    )


def point_loads(pair):
    return dict(
        source_ids=plain(pair["source_ids"]),
        sample_ids=plain(pair["sample_ids"]),
        raw_normal_force_n=plain(pair["normal_force"]),
        raw_force_on_tool_world_n=plain(pair["force_on_a"]),
        raw_force_on_object_world_n=plain(pair["force_on_b"]),
    )


def run(source, audit, metadata, output, *, steps=None, cache_root=None):
    from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
    from phymeo.backends.mujoco_heterogeneous_area_scene import (
        MuJoCoTyxHeterogeneousAreaPressureScene,
    )
    from phymeo.control.manipulation_release_monitor import ReleaseMonitor

    from .pusht_active_route import (
        _controller_boundary,
        _first_binding,
        _save_final_checkpoint,
        active_command,
    )
    from .pusht_lateral_resume import prepare_entry, reference_tick, restore_entry
    from .pusht_load_compensation import PushLoadCompensation
    from .pusht_loaded_release import (
        _command_record,
        _release_control,
        attempt_release_candidate,
        release_reference,
        request_loaded_route,
    )
    from .pusht_pose_replan import _plan_ordinary, freeze_declared_material, require_separated
    from .pusht_public_feedback import accepted_measurement_and_goal
    from .pusht_public_scene import _step_record, build_scene_spec, build_side_asset
    from .pusht_release_evidence import build_release_seed, build_release_step, release_support_keys

    output = Path(output).resolve()
    journal_path = output.with_suffix(".trials.jsonl")
    _require(not output.exists() and not journal_path.exists(), "fresh report and journal")
    report = dict(
        schema=SCHEMA,
        status="running",
        world=6,
        side_coverage="all-lateral",
        publication_frame="rigid-origin",
        control_variant="load-compensation-half",
        selection_policy="joint-pose-balanced",
        material_admission="declared-finite-faces-v1",
        physical_attempts=0,
        physical_trial_steps=0,
        accepted_steps=0,
        recorded_steps=0,
        rejected_trials=0,
        planner_calls=0,
        release_planner_calls=0,
        regular_plan_ready=False,
        goal_success=False,
        force_phase="accepted-step-input-geometry",
        goal_tick_origin=0,
        journal=dict(file=journal_path.name, records=0, sha256=hashlib.sha256().hexdigest()),
    )
    phase, index, history = "prepare_completed_entry", None, None
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
            entry, bound = prepare_entry(
                source, audit, metadata, output.with_suffix(".routes"), allow_completed=True
            )
            driver, model, history = entry.driver, entry.model, entry.history
            _require(
                driver.active is None and not driver.stopped,
                "completed running controller without active reference",
            )
            limit = remaining_budget(
                bound["root_report"]["step_budget"], bound["parent_physical_attempts"], steps
            )
            origin = entry.tick - entry.state["local_tick"]
            evidence_entry = replace(entry, tick=origin)
            report.update(
                start_tick=entry.tick,
                local_tick_origin=entry.state["local_tick"],
                source_reference_tick_origin=origin,
                provenance=entry.provenance,
                initial_state=deepcopy(entry.state),
                final_state=deepcopy(entry.state),
                controller_initial=plain(driver.summary()),
                goal_history_initial=plain(history),
                goal=deepcopy(bound["report"]["goal"]),
                previous_route=deepcopy(bound["report"]["active_route"]),
                step_budget=limit,
                original_step_budget=bound["root_report"]["step_budget"],
                parent_physical_attempts=bound["parent_physical_attempts"],
            )
            spec, asset = (
                build_scene_spec(entry.scene.mjcf, side_coverage="all-lateral"),
                build_side_asset(model, side_coverage="all-lateral"),
            )
            qa = int(model.jnt_qposadr[model.joint("object-free").id])
            with MuJoCoTyxHeterogeneousBatch(
                (spec,),
                (MuJoCoTyxWorldSpec(6, spec.key, entry.initial),),
                mode="cuda",
                cache_root=cache_root,
            ) as batch:
                scene = MuJoCoTyxHeterogeneousAreaPressureScene(
                    batch, {spec.key: (asset,)}, publication_frame="rigid-origin"
                )
                phase = "restore_completed_capsule"
                report["checkpoint_restore"] = restore_entry(scene, batch, entry, bound)
                initial = batch.integration_snapshot()[6]
                receipt = initial.last_step
                (pair,) = receipt.raw_contact_fields["area_friction"]
                report["inherited_point_loads"] = dict(
                    point_loads(pair),
                    identity=plain(pair["identity"]),
                    stamp=plain(pair["stamp"]),
                    input_q=plain(receipt.input_q),
                    input_v=plain(receipt.input_v),
                    input_time=receipt.input_time,
                    output_time=receipt.output_time,
                    receipt_local_tick=receipt.tick,
                    qualification="previous accepted input-geometry load; not a new measurement at the restored endpoint",
                )
                compensate = PushLoadCompensation(entry, identity=asset.identity)
                emit("initialization", report)
                mode, release_route, monitor, last_evidence, supports = (
                    "smooth",
                    None,
                    None,
                    None,
                    (),
                )
                release_start = None
                run_phase = "plan"
                regular_started = False
                try:
                    if limit == 0:
                        report["status"] = "step_budget"
                    else:
                        phase = "ordinary_entry_gate"
                        # The lateral row already carries the original accepted measurement.
                        # No legacy schema relabeling or fabricated endpoint force is needed.
                        seed = build_release_seed(entry, asset, publication_frame="rigid-origin")
                        supports = release_support_keys(seed)
                        last_evidence = seed
                        probe = dict(
                            purpose="entry-separation-gate",
                            selected=False,
                            input_state=_state(initial, qa),
                            tick=entry.tick,
                            seed=plain(seed),
                            planner_requested=False,
                        )
                        try:
                            report["separation_proof"] = require_separated(seed, supports)
                            probe.update(status="strictly_separated", errors=[])
                        except ValueError as error:
                            probe.update(
                                status="entry_separation_rejected", errors=_error_chain(error)
                            )
                            run_phase = "withdraw"
                        report["entry_probe"] = probe
                        emit("planner_probe", probe)
                        if run_phase == "withdraw":
                            phase = "qualify_release"
                            before_budget = dict(
                                ik_calls=driver.planner.ik_calls,
                                path_calls=driver.planner.path_calls,
                            )
                            report["release_planner_calls"] += 1
                            release_route, proposals = request_loaded_route(entry, initial)
                            release_path = output.with_suffix(".release-route.json")
                            saved = dict(
                                tick=entry.tick,
                                q=initial.q,
                                v=initial.v,
                                route=release_route,
                                proposals=proposals,
                                previous_direction=driver.previous_direction,
                                planning_budget_before=before_budget,
                                planning_budget_after=dict(
                                    ik_calls=driver.planner.ik_calls,
                                    path_calls=driver.planner.path_calls,
                                ),
                            )
                            with release_path.open("x", encoding="utf-8") as f:
                                json.dump(plain(saved), f, allow_nan=False)
                            report["release_reference"] = dict(
                                file=release_path.name,
                                sha256=_digest(release_path),
                                nodes=plain(release_route.nodes),
                                route_status=release_route.status,
                                max_steps=320,
                                support_keys=plain(supports),
                                qualification="original native sampled release request; dynamic unloading still unproven",
                            )
                            report["seed_evidence"] = plain(seed)
                            emit(
                                "release_reference",
                                dict(reference=report["release_reference"], seed=seed),
                            )
                            if (
                                release_route.status != "sampled_release_ready"
                                or release_route.proposal is not None
                            ):
                                report.update(
                                    status="release_qualification_rejected",
                                    release_rejection=dict(
                                        actual_status=release_route.status,
                                        required_status="sampled_release_ready",
                                        reason="original release-only qualification not returned; complete actual request retained",
                                        attempts=plain(release_route.attempts),
                                    ),
                                )
                            else:
                                release_reference(release_route.nodes, 0.0)
                                release_start = initial.tick
                                monitor = ReleaseMonitor(support_keys=supports, max_steps=320)
                                monitor.start(seed)
                        while report["status"] == "running" and report["physical_attempts"] < limit:
                            before = batch.integration_snapshot()[6]
                            tick = reference_tick(entry, before)
                            _require(
                                driver.tick == tick and np.array_equal(driver.last_q, before.q),
                                "accepted controller boundary",
                            )
                            if run_phase == "plan":
                                phase = "ordinary_replan"
                                report["separation_proof"] = require_separated(
                                    last_evidence, supports
                                )
                                _require(report["planner_calls"] == 0, "one ordinary route request")
                                material = freeze_declared_material(
                                    model,
                                    before.q,
                                    asset,
                                    model_key=spec.key,
                                    scene_asset_sha256=scene.asset_sha256,
                                )
                                report["planner_calls"] += 1
                                route, plan = _plan_ordinary(
                                    entry,
                                    before,
                                    tick,
                                    selection_policy="joint-pose-balanced",
                                    material_faces=material,
                                )
                                report["ordinary_route"] = plan
                                emit("planner", plan)
                                if route.status != "sampled_reference_ready":
                                    report["status"] = "planning_rejected"
                                    break
                                report["reference_budget"] = reference_budget(
                                    route, limit - report["physical_attempts"]
                                )
                                report.update(
                                    regular_plan_ready=True,
                                    active_route=dict(
                                        start_tick=tick,
                                        end_tick=tick + report["reference_budget"]["route_steps"],
                                        nodes=plain(route.nodes),
                                        file=plan["route_file"],
                                        sha256=plan["route_sha256"],
                                        completed_routes_before=driver.completed,
                                        route_index=len(driver.routes),
                                    ),
                                )
                                run_phase = "regular"
                            if run_phase == "withdraw" and before.tick - release_start >= 320:
                                report["status"] = "release_step_budget"
                                break
                            selected = False
                            modes = (
                                (mode, "constant_rate")
                                if run_phase == "withdraw"
                                and before.tick == release_start
                                and mode == "smooth"
                                else (mode,)
                            )
                            for candidate_mode in modes:
                                if report["physical_attempts"] >= limit:
                                    break
                                phase = run_phase + "_reference"
                                if run_phase == "withdraw":
                                    control, controller = _release_control(
                                        entry,
                                        before,
                                        release_route.nodes,
                                        (before.tick - release_start) * H,
                                        candidate_mode,
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
                                command = _command_record(control, tick)
                                captured = {}

                                def assess(inputs, outputs, geometry):
                                    if run_phase == "withdraw":
                                        evidence = build_release_step(
                                            evidence_entry,
                                            asset,
                                            inputs[6],
                                            outputs[6],
                                            geometry[6][0],
                                            diagnostics=captured.setdefault(
                                                "geometry_comparison", {}
                                            ),
                                            publication_frame="rigid-origin",
                                        )
                                        captured["evidence"] = evidence
                                        decision = monitor.assess(
                                            replace(evidence, publication="trial")
                                        )
                                        return dict(
                                            safe=bool(decision.admissible),
                                            reason=decision.status,
                                            decision=decision,
                                        )
                                    return dict(
                                        safe=True,
                                        reason="admitted ordinary reference",
                                        decision=None,
                                    )

                                index = report["physical_attempts"]
                                report["physical_attempts"] += 1
                                phase = "scene_step"
                                trial = attempt_release_candidate(
                                    scene, batch, {6: control}, assess
                                )
                                report["physical_trial_steps"] += 1
                                after = trial["candidate_output"][6]
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
                                    trial["candidate_geometry"][6][0],
                                    command,
                                    qa,
                                    trial["wall_s"],
                                )
                                row.pop("stored_control_matches_original")
                                (pair,) = after.last_step.raw_contact_fields["area_friction"]
                                row["area"].update(point_loads(pair))
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
                                    phase=run_phase,
                                    trial_mode=candidate_mode
                                    if run_phase == "withdraw"
                                    else "regular",
                                    controller=plain(controller),
                                    stored_control_matches_submitted=True,
                                    selected=selected,
                                    restored=trial["restored"],
                                    rollback_verification=trial["rollback_verification"],
                                    evidence=plain(captured.get("evidence")),
                                    geometry_comparison=plain(captured.get("geometry_comparison")),
                                    release_decision=plain(trial["assessment"]["decision"]),
                                    measurement=plain(measurement),
                                    side_coverage="all-lateral",
                                    material_identity=plain(asset.identity),
                                )
                                emit("candidate", row, index)
                                if not selected:
                                    if allow_constant_retry(
                                        candidate_mode,
                                        before.tick - release_start,
                                        trial["assessment"]["decision"],
                                    ):
                                        continue
                                    break
                                committed = None
                                if run_phase == "withdraw":
                                    mode = candidate_mode
                                    committed = monitor.observe(captured["evidence"])
                                    report["release_status"] = monitor.status
                                    last_evidence = captured["evidence"]
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
                                    committed_release_decision=plain(committed),
                                )
                                emit("accepted_callback", callback, index)
                                report["recorded_steps"] += 1
                                report.pop("unrecorded_accepted_step", None)
                                entry.previous = dict(row, **callback)
                                report.update(
                                    last_goal=plain(goal), goal_success=bool(goal.success)
                                )
                                if (
                                    run_phase == "withdraw"
                                    and monitor.status == "released"
                                    and (after.tick - release_start) * H
                                    >= release_route.nodes[-1][0]
                                ):
                                    report["separation_proof"] = require_separated(
                                        last_evidence, supports
                                    )
                                    run_phase = "plan"
                                break
                            if not selected:
                                report["status"] = (
                                    "step_budget"
                                    if report["physical_attempts"] >= limit
                                    else "release_candidate_rejected"
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
                                    "actual full route endpoint",
                                )
                                report["status"] = "route_complete"
                                break
                            if report["recorded_steps"] % 32 == 0:
                                _write_progress(stream, report)
                        if report["status"] == "running":
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
                            geometry_comparison=plain(captured.get("geometry_comparison")),
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
    report = run(
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
                k: report.get(k)
                for k in (
                    "status",
                    "accepted_steps",
                    "recorded_steps",
                    "rejected_trials",
                    "failure_stage",
                    "errors",
                )
            }
        )
    )
    raise SystemExit(
        0 if report["status"] in ("route_complete", "step_budget", "controller_stopped") else 1
    )


if __name__ == "__main__":
    main()
