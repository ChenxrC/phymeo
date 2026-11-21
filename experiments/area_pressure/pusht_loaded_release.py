"""Bounded, rollback-audited continuation from a committed PushT loaded state.

Only selected real public Scene steps update the restored controller. Rejected
physical trials remain in the report, including their actual receipt and endpoint.
"""

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.run_manipulation_control import plain, reference

H = 0.001


def release_reference(nodes, elapsed, *, mode="smooth"):
    """The original withdrawal, or its one fixed constant-rate fallback."""
    if (
        len(nodes) != 2
        or mode not in ("smooth", "constant_rate")
        or not np.isfinite(elapsed)
        or elapsed < 0
    ):
        raise ValueError("bounded two-node withdrawal reference required")
    (start, first_stage, q0), (duration, last_stage, q1) = nodes
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    if (
        start != 0
        or not np.isfinite(duration)
        or duration <= 0
        or first_stage != "withdraw"
        or last_stage != "withdraw"
        or q0.ndim != 1
        or not len(q0)
        or q1.shape != q0.shape
        or not np.isfinite(q0).all()
        or not np.isfinite(q1).all()
    ):
        raise ValueError("finite withdrawal endpoints and positive duration required")
    if mode == "smooth":
        return reference(nodes, elapsed)
    if elapsed >= duration:
        return q1.copy(), np.zeros_like(q1), "withdraw"
    return q0 + elapsed / duration * (q1 - q0), (q1 - q0) / duration, "withdraw"


def _rollback_readback(expected, actual):
    """All published numerical bytes, apart from the documented fresh stamp token."""
    from phymeo.geometry.area_pressure import AreaPressureStamp

    mismatches = []

    def compare(old, new, path):
        if len(mismatches) >= 32:
            return
        if isinstance(old, np.ndarray):
            if (
                not isinstance(new, np.ndarray)
                or old.dtype != new.dtype
                or old.shape != new.shape
                or old.tobytes() != new.tobytes()
            ):
                mismatches.append(path)
        elif is_dataclass(old):
            if type(new) is not type(old):
                mismatches.append(path)
                return
            for field in fields(old):
                if type(old) is AreaPressureStamp and field.name == "configuration_id":
                    continue
                compare(getattr(old, field.name), getattr(new, field.name), path + "." + field.name)
        elif isinstance(old, Mapping):
            if not isinstance(new, Mapping) or set(new) != set(old):
                mismatches.append(path + ".keys")
                return
            for key in old:
                compare(old[key], new[key], (path + "." if path else "") + str(key))
        elif isinstance(old, (tuple, list)):
            if type(new) is not type(old) or len(old) != len(new):
                mismatches.append(path)
                return
            for i, (a, b) in enumerate(zip(old, new, strict=True)):
                compare(a, b, path + "." + str(i))
        elif type(new) is not type(old) or old != new:
            mismatches.append(path)
        elif (
            isinstance(old, (float, np.generic))
            and np.asarray(old).tobytes() != np.asarray(new).tobytes()
        ):
            mismatches.append(path)

    compare(expected, actual, "")
    return dict(passed=not mismatches, worlds=list(expected), mismatches=mismatches)


def attempt_release_candidate(scene, batch, controls, assess):
    """One real step; rejected trials retain their receipt and verified rollback."""
    checkpoint = scene.checkpoint()
    before = batch.integration_snapshot()
    accepted = restored = restore_attempted = False
    verification = None
    after = geometry = wall = None

    def restore():
        nonlocal restore_attempted, restored, verification
        restore_attempted = True
        try:
            scene.restore(checkpoint, expected_revision=batch.checkpoint_revision)
            verification = _rollback_readback(before, batch.integration_snapshot())
        except Exception:
            verification = dict(
                passed=False, worlds=list(before), mismatches=["restore_or_readback_failed"]
            )
            raise
        restored = verification["passed"]
        if not restored:
            raise RuntimeError(
                "release rollback readback mismatch: " + ", ".join(verification["mismatches"])
            )

    try:
        batch.set_controls(controls)
        begin = perf_counter()
        result = scene.step(H, expected_revision=batch.checkpoint_revision)
        wall = perf_counter() - begin
        if dict(result) != {world: True for world in before}:
            raise RuntimeError("release candidate requires acceptance of every declared world")
        accepted = True
        after = batch.integration_snapshot()
        geometry = scene.last_geometry
        assessment = assess(before, after, geometry)
        if type(assessment.get("safe")) is not bool:
            raise ValueError("release assessment must explicitly select or reject")
        selected = assessment["safe"]
        if not selected:
            restore()
        return dict(
            physics_accepted=True,
            selected=selected,
            restored=restored,
            rollback_verification=verification,
            assessment=assessment,
            candidate_input=before,
            candidate_output=after,
            candidate_geometry=geometry,
            wall_s=wall,
        )
    except Exception as error:
        failure = error
        if not restore_attempted:
            try:
                restore()
            except Exception as rollback_error:
                failure = rollback_error
        failure.release_trial = dict(
            physics_accepted=accepted,
            restored=restored,
            rollback_verification=verification,
            input=before,
            output=after,
            geometry=geometry,
            wall_s=wall,
        )
        failure.add_note(
            "release trial physical acceptance="
            + str(accepted)
            + "; rollback readback verified="
            + str(restored)
        )
        if failure is not error:
            raise failure from error
        raise


def _controller_boundary(driver):
    summary = driver.summary()
    return plain(
        {
            key: summary[key]
            for key in (
                "status",
                "tick",
                "final_cost",
                "completed_routes",
                "total_contact_impulse_ns",
                "partial_route",
            )
        }
    )


def _loaded_route(entry, snapshot):
    from phymeo.control.manipulation_push_objective import propose_planar_pushes

    driver = entry.driver
    pos, rotation, pusher = driver.pose(snapshot.q)
    proposals = propose_planar_pushes(
        driver.query,
        reference_position=driver.ref_position,
        reference_rotation=driver.ref_rotation,
        current_position=pos,
        current_rotation=rotation,
        target_xy=driver.target.position[:2],
        target_yaw=driver.target.yaw,
        pusher_height_m=float(pusher[2]),
        position_tolerance_m=driver.goal.position_tolerance_m,
        yaw_tolerance_rad=driver.goal.yaw_tolerance_rad,
    )
    previous = driver.planner.allow_loaded_withdrawal
    try:
        driver.planner.allow_loaded_withdrawal = True
        route = driver.planner.plan(
            snapshot.q, proposals, previous_direction=driver.previous_direction
        )
    finally:
        driver.planner.allow_loaded_withdrawal = previous
    if route.status != "sampled_release_ready" or route.proposal is not None:
        raise ValueError(
            "loaded continuation requires an independently qualified release-only route"
        )
    release_reference(route.nodes, 0.0)
    return route, proposals


def _release_control(entry, snapshot, nodes, elapsed, mode):
    from phymeo.backends.mujoco_runtime import MuJoCoControl

    from .pusht_public_feedback import _original_action_transport

    qref, vref, stage = release_reference(nodes, elapsed, mode=mode)
    ctrl, pd = entry.pd.command(snapshot.q, snapshot.v, qref, vref)
    actual, transport = _original_action_transport(ctrl)
    control = MuJoCoControl(
        actual, qfrc_applied=np.zeros(entry.model.nv), xfrc_applied=np.zeros((entry.model.nbody, 6))
    )
    return control, dict(
        mode=mode,
        stage=stage,
        elapsed_s=elapsed,
        measurement_q=plain(snapshot.q),
        measurement_v=plain(snapshot.v),
        target_q=plain(qref),
        target_v=plain(vref),
        pd=plain(pd),
        transport=transport,
    )


def _command_record(control, tick):
    return dict(
        tick=tick,
        output_tick=tick + 1,
        ctrl=plain(control.ctrl),
        qfrc_applied=plain(control.qfrc_applied),
        xfrc_applied=plain(control.xfrc_applied),
    )


def run(summary_path, output, *, steps=320, regular_steps=64, cache_root=None):
    """Continue one true accepted state; retain every bounded physical trial.

    'regular_plan_ready' proves acceptance by the ordinary sampled planner, not
    task success. Only actual Goal results can establish success.
    """
    from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
    from phymeo.backends.mujoco_heterogeneous_area_scene import (
        MuJoCoTyxHeterogeneousAreaPressureScene,
    )
    from phymeo.backends.mujoco_runtime import MuJoCoControl
    from phymeo.control.manipulation_release_monitor import ReleaseMonitor

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
    from .pusht_release_entry import prepare_release_entry, require_release_seed
    from .pusht_release_evidence import build_release_seed, build_release_step, release_support_keys

    if (
        type(steps) is not int
        or not 250 <= steps <= 2048
        or type(regular_steps) is not int
        or not 1 <= regular_steps <= 2048
    ):
        raise ValueError("release steps must be 250..2048; regular steps 1..2048")
    output = Path(output).resolve()
    report = dict(
        schema="phymeo-pusht-loaded-release-v1",
        status="running",
        world=6,
        selected_steps=0,
        physical_trial_steps=0,
        rejected_trials=0,
        release_budget=steps,
        regular_budget=regular_steps,
        trials=[],
        evidence_scope="fresh public owner; true q/v/scalar clock; no prior physics replay",
        force_phase="accepted-step-input-geometry",
        external_channels="zero applied generalized/body force; original native PD + CUDA action",
        selection_policy="smooth first; one same-path/same-duration constant-rate fallback at entry; then locked",
        regular_plan_ready=False,
        goal_success=False,
    )
    phase = "prepare_entry"
    journal_path = output.with_suffix(".trials.jsonl")
    journal_hash = hashlib.sha256()
    report["journal"] = dict(
        file=journal_path.name,
        records=0,
        sha256=journal_hash.hexdigest(),
        schema="phymeo-pusht-release-event-v1",
    )
    report["geometry_definition"] = dict(
        depth="one CPU FK definition at both accepted input and output configurations",
        cuda_source="actual CUDA input surface and receipt source quadrature matched point by point",
        force="actual accepted CUDA receipt; never recomputed from CPU geometry",
        scope="discrete geometry samples, not continuous collision detection",
    )
    with output.open("x+", encoding="utf-8") as stream, journal_path.open("xb") as journal:

        def emit(kind, payload, trial_index=None):
            event = dict(
                schema="phymeo-pusht-release-event-v1",
                event=kind,
                trial_index=trial_index,
                payload=plain(payload),
            )
            encoded = (json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
            journal.write(encoded)
            journal.flush()
            journal_hash.update(encoded)
            report["journal"]["records"] += 1
            report["journal"]["sha256"] = journal_hash.hexdigest()

        _write_progress(stream, report)
        try:
            entry = prepare_release_entry(summary_path, output.with_suffix(".routes"))
            model, world = entry.model, 6
            spec, asset = build_scene_spec(entry.scene.mjcf), build_side_asset(model)
            report.update(
                provenance=entry.provenance,
                start_tick=entry.tick,
                prior_direction=plain(entry.driver.previous_direction),
                prior_controller=plain(entry.driver.summary()),
            )
            qa = int(model.jnt_qposadr[model.joint("object-free").id])
            declaration = MuJoCoTyxWorldSpec(world, spec.key, entry.initial)
            phase = "construct_scene"
            with MuJoCoTyxHeterogeneousBatch(
                (spec,), (declaration,), mode="cuda", cache_root=cache_root
            ) as batch:
                scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (asset,)})
                initial = batch.integration_snapshot()[world]
                require_release_seed(entry, initial)
                if scene.asset_sha256 != entry.provenance["scene_asset_sha256"]:
                    raise ValueError("release material differs from the continuous source")
                report["initial_state"] = report["final_state"] = _state(initial, qa)
                phase = "qualify_loaded_reference"
                route, proposals = _loaded_route(entry, initial)
                route_path = output.with_suffix(".release-route.json")
                with route_path.open("x", encoding="utf-8") as route_stream:
                    json.dump(
                        plain(
                            dict(
                                tick=entry.tick,
                                q=initial.q,
                                v=initial.v,
                                route=route,
                                proposals=proposals,
                            )
                        ),
                        route_stream,
                        allow_nan=False,
                    )
                report["release_reference"] = dict(
                    file=route_path.name,
                    sha256=hashlib.sha256(route_path.read_bytes()).hexdigest(),
                    nodes=plain(route.nodes),
                )
                seed = build_release_seed(entry, asset)
                monitor = ReleaseMonitor(support_keys=release_support_keys(seed), max_steps=steps)
                monitor.start(seed)
                report["seed_evidence"] = plain(seed)
                emit(
                    "initialization",
                    {
                        key: report[key]
                        for key in (
                            "provenance",
                            "start_tick",
                            "initial_state",
                            "release_reference",
                            "seed_evidence",
                            "geometry_definition",
                        )
                    },
                )
                _write_progress(stream, report)
                history, mode, released = entry.history, "smooth", False
                regular_count = 0
                for index in range(steps + regular_steps):
                    before = batch.integration_snapshot()[world]
                    tick = entry.tick + before.tick
                    if (
                        not np.array_equal(entry.driver.last_q, before.q)
                        or entry.driver.tick != tick
                    ):
                        raise ValueError(
                            "controller does not belong to the accepted physical boundary"
                        )
                    regular = report["regular_plan_ready"]
                    if regular and regular_count >= regular_steps:
                        report["status"] = "regular_step_budget"
                        break
                    if not regular and before.tick >= steps:
                        report["status"] = "release_step_budget"
                        break
                    phase = "regular_reference" if regular else "release_reference"
                    if regular:
                        control, controller = command_from_snapshot(
                            entry.driver, entry.pd, tick, before, start_tick=entry.tick
                        )
                        actual, transport = _original_action_transport(control.ctrl)
                        control = MuJoCoControl(
                            actual,
                            qfrc_applied=control.qfrc_applied,
                            xfrc_applied=control.xfrc_applied,
                        )
                        controller["transport"] = transport
                        controller["generated_channels"] = _command_record(control, tick)
                        controller["ctrl_transport"] = transport["kind"]
                        modes = ("regular",)
                    else:
                        modes = (
                            (mode, "constant_rate")
                            if before.tick == 0 and mode == "smooth"
                            else (mode,)
                        )
                    chosen = None
                    for candidate_mode in modes:
                        if not regular:
                            control, controller = _release_control(
                                entry, before, route.nodes, before.tick * H, candidate_mode
                            )
                        command = _command_record(control, tick)
                        captured = {}

                        def assess(inputs, outputs, geometry):
                            evidence = build_release_step(
                                entry,
                                asset,
                                inputs[world],
                                outputs[world],
                                geometry[world][0],
                                diagnostics=captured.setdefault("material_geometry", {}),
                            )
                            captured["evidence"] = evidence
                            if regular:
                                return dict(safe=True, reason="ordinary_reference", decision=None)
                            decision = monitor.assess(replace(evidence, publication="trial"))
                            return dict(
                                safe=bool(decision.admissible),
                                reason=decision.status,
                                decision=decision,
                            )

                        phase = "candidate_physics_and_policy"
                        trial = attempt_release_candidate(scene, batch, {world: control}, assess)
                        report["physical_trial_steps"] += 1
                        after = trial["candidate_output"][world]
                        if trial["selected"]:
                            report["selected_steps"] += 1
                            report["final_state"] = _state(after, qa)
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
                        row.update(
                            stored_control_matches_submitted=True,
                            controller=plain(controller),
                            phase="regular" if regular else "withdraw",
                            trial_mode=candidate_mode,
                            selected=trial["selected"],
                            restored=trial["restored"],
                            rollback_verification=trial["rollback_verification"],
                            evidence=plain(captured["evidence"]),
                            geometry_comparison=plain(captured["material_geometry"]),
                            release_decision=plain(trial["assessment"]["decision"]),
                        )
                        report["trials"].append(row)
                        trial_index = len(report["trials"]) - 1
                        emit("candidate", row, trial_index)
                        if not trial["selected"]:
                            report["rejected_trials"] += 1
                            # Only an actual inward/deepening/load-rise failure can
                            # unlock the fixed first-entry fallback.
                            decision = trial["assessment"]["decision"]
                            codes = {item.code for item in decision.violations}
                            allowed = {
                                "inward_motion",
                                "penetration_increased",
                                "normal_load_increased",
                            }
                            if not codes or not codes <= allowed:
                                break
                            continue
                        chosen = trial
                        if not regular:
                            mode = candidate_mode
                            decision = monitor.observe(captured["evidence"])
                            row["release_decision"] = plain(decision)
                            released = released or decision.release_complete
                        phase = "accepted_controller_and_goal"
                        measurement, goal_frame, goal = accepted_measurement_and_goal(
                            entry.scene, model, world, tick, before, after, history
                        )
                        history = goal.history
                        feedback = entry.driver.accepted_measurement(
                            tick + 1, after.q, measurement, goal, input_q=before.q
                        )
                        row.update(
                            goal_frame=plain(goal_frame),
                            goal_result=plain(goal),
                            feedback_accepted=plain(feedback),
                            driver_state_after=_controller_boundary(entry.driver),
                        )
                        emit(
                            "accepted_callback",
                            {
                                key: row[key]
                                for key in (
                                    "goal_frame",
                                    "goal_result",
                                    "feedback_accepted",
                                    "driver_state_after",
                                    "release_decision",
                                )
                            },
                            trial_index,
                        )
                        report["goal_success"] = bool(goal.success)
                        if regular:
                            regular_count += 1
                        elif (
                            not entry.driver.stopped
                            and released
                            and after.tick * H >= route.nodes[-1][0]
                        ):
                            # Verified unloaded endpoint, then the default planner
                            # must independently admit the next ordinary route.
                            phase = "ordinary_replan"
                            entry.driver.previous_direction = None
                            entry.driver.reference(entry.tick + after.tick, after.q, after.v)
                            if entry.driver.stopped or entry.driver.active is None:
                                raise ValueError("ordinary replan rejected: " + entry.driver.reason)
                            report.update(
                                regular_plan_ready=True,
                                ordinary_replan_tick=entry.tick + after.tick,
                                ordinary_route=plain(entry.driver.routes[-1]),
                            )
                            emit("ordinary_replan", report["ordinary_route"], trial_index)
                        if report["selected_steps"] % 32 == 0:
                            _write_progress(stream, report)
                        break
                    if chosen is None:
                        report["status"] = "release_candidate_rejected"
                        break
                    if entry.driver.stopped:
                        report["status"] = "controller_stopped"
                        report["controller_status"] = entry.driver.reason
                        break
                else:
                    report["status"] = (
                        "regular_step_budget"
                        if report["regular_plan_ready"]
                        else "release_step_budget"
                    )
                report["controller_final"] = plain(entry.driver.summary())
                report["release_monitor_status"] = monitor.status
                report["release_mode"] = mode
                phase = "scene_cleanup"
        except Exception as error:
            failed_trial = getattr(error, "release_trial", None)
            if failed_trial is not None:
                report["failed_trial"] = dict(
                    physics_accepted=failed_trial["physics_accepted"],
                    restored=failed_trial["restored"],
                    rollback_verification=failed_trial["rollback_verification"],
                    input_states={str(k): _state(v, qa) for k, v in failed_trial["input"].items()},
                    output_states=None
                    if failed_trial["output"] is None
                    else {str(k): _state(v, qa) for k, v in failed_trial["output"].items()},
                )
                report["physical_trial_steps"] += int(failed_trial["physics_accepted"])
                report["failed_trial"]["submitted_control"] = plain(locals().get("command"))
                report["failed_trial"]["controller"] = plain(locals().get("controller"))
                report["failed_trial"]["evidence"] = plain(
                    locals().get("captured", {}).get("evidence")
                )
                report["failed_trial"]["geometry_comparison"] = plain(
                    locals().get("captured", {}).get("material_geometry")
                )
                if failed_trial["physics_accepted"]:
                    try:
                        report["failed_trial"]["physical_record"] = _step_record(
                            model,
                            asset,
                            failed_trial["input"][world],
                            failed_trial["output"][world],
                            failed_trial["geometry"][world][0],
                            command,
                            qa,
                            failed_trial["wall_s"],
                        )
                    except Exception as record_error:
                        report["failed_trial"]["record_errors"] = _error_chain(record_error)
            if "entry" in locals():
                report["controller_final"] = plain(entry.driver.summary())
            report.update(status="partial_failure", failure_stage=phase, errors=_error_chain(error))
            emit(
                "failure",
                dict(
                    stage=phase,
                    errors=report["errors"],
                    failed_trial=report.get("failed_trial"),
                    selected_steps=report["selected_steps"],
                ),
            )
        emit(
            "final",
            {
                key: report.get(key)
                for key in (
                    "status",
                    "selected_steps",
                    "physical_trial_steps",
                    "rejected_trials",
                    "final_state",
                    "regular_plan_ready",
                    "goal_success",
                    "controller_final",
                )
            },
        )
        _write_progress(stream, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--regular-steps", type=int, default=64)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    report = run(
        args.summary,
        args.output,
        steps=args.steps,
        regular_steps=args.regular_steps,
        cache_root=args.cache_root,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "selected_steps",
                    "physical_trial_steps",
                    "regular_plan_ready",
                    "goal_success",
                )
            }
        )
    )
    raise SystemExit(0 if report["status"] in ("regular_step_budget", "controller_stopped") else 1)


if __name__ == "__main__":
    main()
