"""Continue the already active public PushT route, without planning another route.

python -m experiments.area_pressure.pusht_active_route --summary ORIGINAL.json
    --release RELEASE.json --checkpoint BOUND.metadata.json --output NEW.json [--cache-root CACHE]

The append-only journal has one line per fully recorded accepted step. A callback
failure can leave one committed endpoint beyond that prefix; counters and the
unrecorded tail keep this distinction explicit. No release geometry audit runs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from phymeo.backends.mujoco_runtime import MuJoCoControl
from scripts.run_manipulation_control import plain, reference

from .pusht_public_feedback import _original_action_transport, command_from_snapshot
from .pusht_public_scene import H


def _require(ok, reason):
    if not ok:
        raise ValueError("active route: " + reason)


def active_phase(entry, snapshot):
    driver = entry.driver
    tick = entry.tick + snapshot.tick - entry.state["local_tick"]
    _require(not driver.stopped and driver.active is not None, "no running active route")
    _require(
        driver.tick == tick and np.array_equal(driver.last_q, snapshot.q),
        "controller tick/q differ from accepted state",
    )
    start, route = driver.active
    _require(type(start) is int and start <= tick and len(route.nodes) >= 2, "active route start")
    duration = float(route.nodes[-1][0])
    _require(np.isfinite(duration) and duration > 0, "positive finite route duration")
    count = round(duration / H)
    _require(abs(duration - count * H) <= 1e-12, "route end is not on the physical tick grid")
    end = start + count
    _require(tick < end, "active route has ended; another reference would replan")
    return dict(
        absolute_tick=tick,
        route_start_tick=start,
        route_end_tick=end,
        route_duration_s=duration,
        elapsed_s=(tick - start) * H,
        remaining_steps=end - tick,
    )


def active_command(entry, snapshot):
    """Use the original active reference/PD and its actual CUDA action transport."""
    phase = active_phase(entry, snapshot)
    driver = entry.driver
    active = driver.active
    counts = (driver.planner.ik_calls, driver.planner.path_calls)
    routes = plain(driver.routes)
    control, diagnostic = command_from_snapshot(
        entry.driver,
        entry.pd,
        phase["absolute_tick"],
        snapshot,
        start_tick=entry.tick - entry.state["local_tick"],
    )
    _require(
        driver.active is active
        and plain(driver.routes) == routes
        and (driver.planner.ik_calls, driver.planner.path_calls) == counts,
        "reference changed the active route or planned another route",
    )
    _require(
        driver.tick == phase["absolute_tick"] and np.array_equal(driver.last_q, snapshot.q),
        "reference changed the accepted controller boundary",
    )
    route_info = diagnostic["route"]
    _require(
        route_info["route_start_tick"] == phase["route_start_tick"]
        and route_info["route_elapsed_s"] == phase["elapsed_s"],
        "reference phase differs",
    )
    actual, transport = _original_action_transport(control.ctrl)
    result = MuJoCoControl(
        actual, qfrc_applied=control.qfrc_applied, xfrc_applied=control.xfrc_applied
    )
    diagnostic.update(
        active_phase=phase,
        transport=transport,
        ctrl_transport=transport["kind"],
        generated_channels=dict(
            ctrl=plain(result.ctrl),
            qfrc_applied=plain(result.qfrc_applied),
            xfrc_applied=plain(result.xfrc_applied),
        ),
    )
    return result, diagnostic


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


def _first_binding(entry, snapshot, command, diagnostic):
    phase = diagnostic["active_phase"]
    qref, vref, stage = reference(entry.driver.active[1].nodes, phase["elapsed_s"])
    direct, _ = entry.pd.command(snapshot.q, snapshot.v, qref, vref)
    expected_direct = np.asarray(direct, np.float32)
    supplied_direct = np.asarray(diagnostic["transport"]["direct_pd_ctrl"], np.float32)
    _require(
        np.array_equal(qref, diagnostic["target_q"])
        and np.array_equal(vref, diagnostic["target_v"])
        and stage == diagnostic["stage"]
        and expected_direct.tobytes() == supplied_direct.tobytes(),
        "first command is not the original active reference and PD phase",
    )
    return dict(
        input_tick=phase["absolute_tick"],
        input_time=snapshot.time,
        measurement_q=plain(snapshot.q),
        measurement_v=plain(snapshot.v),
        route_start_tick=phase["route_start_tick"],
        route_elapsed_s=phase["elapsed_s"],
        target_q=plain(qref),
        target_v=plain(vref),
        stage=stage,
        independently_recomputed_target_q=plain(qref),
        independently_recomputed_target_v=plain(vref),
        independently_recomputed_direct_ctrl=plain(expected_direct),
        submitted_ctrl=plain(command.ctrl),
        transport=diagnostic["transport"],
        exact_matches=True,
        meaning="same saved active route at its true elapsed phase; no archived future command",
    )


def _save_final_checkpoint(scene, batch, entry, output, report, history):
    """Persist this live public owner's capsule and honest physical/controller metadata."""
    from .pusht_public_scene import _state

    snapshot = batch.integration_snapshot()[6]
    qa = int(entry.model.jnt_qposadr[entry.model.joint("object-free").id])
    state = _state(snapshot, qa)
    tick = entry.tick + snapshot.tick - entry.state["local_tick"]
    capsule = scene.checkpoint().to_bytes()
    path = output.with_suffix(".final.scene")
    metadata_path = output.with_suffix(".final.scene.metadata.json")
    digest = hashlib.sha256(capsule).hexdigest()
    with path.open("xb") as stream:
        stream.write(capsule)
        stream.flush()
    metadata = dict(
        schema="phymeo-pusht-active-terminal-checkpoint-v1",
        checkpoint_file=path.name,
        checkpoint_sha256=digest,
        source_report=output.name,
        source_tick=tick,
        local_tick=snapshot.tick,
        world=6,
        time=snapshot.time,
        state=state,
        q=plain(snapshot.q),
        v=plain(snapshot.v),
        asset_sha256=scene.asset_sha256,
        active_route=report.get("active_route"),
        target=report.get("goal"),
        controller=plain(entry.driver.summary()),
        goal_history=plain(history),
        provenance=report["provenance"],
        journal=dict(report["journal"]),
        accepted_steps=report["accepted_steps"],
        recorded_steps=report["recorded_steps"],
        controller_at_physical_endpoint=bool(
            entry.driver.tick == tick and np.array_equal(entry.driver.last_q, snapshot.q)
        ),
        physical_state_matches_report=state == report.get("final_state"),
        contract="public Scene capsule; terminal metadata is distinct from reconstructed release metadata",
    )
    encoded = (json.dumps(metadata, allow_nan=False, indent=2) + "\n").encode("utf-8")
    with metadata_path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
    return dict(
        schema=metadata["schema"],
        file=path.name,
        sha256=digest,
        metadata_file=metadata_path.name,
        metadata_sha256=hashlib.sha256(encoded).hexdigest(),
        source_tick=tick,
        local_tick=snapshot.tick,
        time=snapshot.time,
        world=6,
        controller_at_physical_endpoint=metadata["controller_at_physical_endpoint"],
        physical_state_matches_report=metadata["physical_state_matches_report"],
    )


def run(
    prior_summary_path, release_report_path, output, *, checkpoint_metadata_path, cache_root=None
):
    from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
    from phymeo.backends.mujoco_heterogeneous_area_scene import (
        MuJoCoTyxHeterogeneousAreaPressureScene,
    )

    from .pusht_active_checkpoint import restore_bound_entry
    from .pusht_active_entry import prepare_active_entry
    from .pusht_loaded_release import attempt_release_candidate
    from .pusht_public_feedback import accepted_measurement_and_goal
    from .pusht_public_scene import (
        _error_chain,
        _state,
        _step_record,
        _write_progress,
        build_scene_spec,
        build_side_asset,
    )

    prior_summary_path, release_report_path, output = map(
        lambda p: Path(p).resolve(), (prior_summary_path, release_report_path, output)
    )
    checkpoint_metadata_path = Path(checkpoint_metadata_path).resolve()
    journal_path = output.with_suffix(".steps.jsonl")
    journal_hash = hashlib.sha256()
    report = dict(
        schema="phymeo-pusht-active-route-v1",
        status="running",
        world=6,
        accepted_steps=0,
        recorded_steps=0,
        journal=dict(file=journal_path.name, records=0, sha256=journal_hash.hexdigest()),
        physics="public CUDA Scene; both bodies dynamic; no prior physics replay",
        force_phase="accepted-step-input-geometry",
        control="original active route, PD, FP32 atanh/CUDA tanh; zero external force channels",
        continuation="stop at this route end, original controller stop, or physical failure; never plan a third route",
    )
    phase = "prepare_entry"
    current_tick = None
    with output.open("x+", encoding="utf-8") as stream, journal_path.open("xb") as journal:
        _write_progress(stream, report)
        try:
            entry = prepare_active_entry(
                prior_summary_path, release_report_path, output.with_suffix(".routes")
            )
            world, model = 6, entry.model
            driver = entry.driver
            report.update(
                start_tick=entry.tick,
                local_tick_origin=entry.state["local_tick"],
                provenance=dict(entry.provenance),
                controller_initial=plain(driver.summary()),
                goal=dict(
                    target_xy=plain(driver.target.position[:2]),
                    target_yaw=driver.target.yaw,
                    position_tolerance_m=driver.goal.position_tolerance_m,
                    yaw_tolerance_rad=driver.goal.yaw_tolerance_rad,
                ),
            )
            report["provenance"].update(
                prior_summary_path=str(prior_summary_path),
                prior_summary_sha256=hashlib.sha256(prior_summary_path.read_bytes()).hexdigest(),
                release_report_path=str(release_report_path),
                release_report_sha256=hashlib.sha256(release_report_path.read_bytes()).hexdigest(),
                checkpoint_metadata_path=str(checkpoint_metadata_path),
                checkpoint_metadata_sha256=hashlib.sha256(
                    checkpoint_metadata_path.read_bytes()
                ).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            )
            spec, asset = build_scene_spec(entry.scene.mjcf), build_side_asset(model)
            qa = int(model.jnt_qposadr[model.joint("object-free").id])
            declaration = MuJoCoTyxWorldSpec(world, spec.key, entry.initial)
            phase = "construct_scene"
            with MuJoCoTyxHeterogeneousBatch(
                (spec,), (declaration,), mode="cuda", cache_root=cache_root
            ) as batch:
                scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (asset,)})
                phase = "restore_checkpoint"
                report["checkpoint_restore"] = plain(
                    restore_bound_entry(
                        scene, batch, entry, release_report_path, checkpoint_metadata_path
                    )
                )
                try:
                    prior = batch.integration_snapshot()[world]
                    _require(
                        scene.asset_sha256 == entry.provenance["scene_asset_sha256"],
                        "material asset binding",
                    )
                    initial_phase = active_phase(entry, prior)
                    original_active = driver.active
                    saved_route = driver.routes[-1]
                    route_path = (driver.output / saved_route["file"]).resolve()
                    _require(
                        hashlib.sha256(route_path.read_bytes()).hexdigest()
                        == saved_route["sha256"],
                        "active route file binding",
                    )
                    report.update(
                        initial_state=_state(prior, qa),
                        final_state=_state(prior, qa),
                        requested_steps=initial_phase["remaining_steps"],
                        active_route=dict(
                            start_tick=original_active[0],
                            end_tick=initial_phase["route_end_tick"],
                            duration_s=initial_phase["route_duration_s"],
                            elapsed_at_entry_s=initial_phase["elapsed_s"],
                            remaining_steps=initial_phase["remaining_steps"],
                            nodes=plain(original_active[1].nodes),
                            file=str(route_path),
                            sha256=saved_route["sha256"],
                            route_index=len(driver.routes),
                            completed_routes_before=driver.completed,
                        ),
                    )
                    history = entry.history
                    _write_progress(stream, report)
                    for index in range(report["requested_steps"]):
                        current_tick = entry.tick + prior.tick - entry.state["local_tick"]
                        phase = "active_reference"
                        _require(
                            driver.active is original_active,
                            "active route changed before its final step",
                        )
                        control, controller = active_command(entry, prior)
                        if index == 0:
                            report["first_command_binding"] = _first_binding(
                                entry, prior, control, controller
                            )
                            _write_progress(stream, report)
                        command = dict(
                            tick=current_tick,
                            output_tick=current_tick + 1,
                            ctrl=plain(control.ctrl),
                            qfrc_applied=plain(control.qfrc_applied),
                            xfrc_applied=plain(control.xfrc_applied),
                        )

                        def assess(inputs, outputs, geometry):
                            actual = inputs[world]
                            _require(
                                actual.tick == prior.tick
                                and actual.time == prior.time
                                and np.array_equal(actual.q, prior.q)
                                and np.array_equal(actual.v, prior.v),
                                "candidate does not continue the retained physical state",
                            )
                            return dict(safe=True, reason="existing_active_route")

                        phase = "scene_step"
                        candidate = attempt_release_candidate(
                            scene, batch, {world: control}, assess
                        )
                        before, after = (
                            candidate["candidate_input"][world],
                            candidate["candidate_output"][world],
                        )
                        report["accepted_steps"] += 1
                        report["final_state"] = _state(after, qa)
                        report["unrecorded_accepted_step"] = dict(
                            original_input_tick=current_tick,
                            original_output_tick=current_tick + 1,
                            input_state=_state(before, qa),
                            output_state=_state(after, qa),
                            submitted_control=command,
                            controller=plain(controller),
                        )
                        phase = "accepted_physical_record"
                        row = _step_record(
                            model,
                            asset,
                            before,
                            after,
                            candidate["candidate_geometry"][world][0],
                            command,
                            qa,
                            candidate["wall_s"],
                        )
                        row["submitted_control"] = row.pop("control")
                        row.pop("stored_control_matches_original")
                        row.update(
                            schema="phymeo-pusht-active-route-step-v1",
                            index=index,
                            stored_control_matches_submitted=True,
                            controller=plain(controller),
                        )
                        (pair,) = after.last_step.raw_contact_fields["area_friction"]
                        row["area"]["raw_normal_force_n"] = np.asarray(
                            pair["normal_force"]
                        ).tolist()
                        report["unrecorded_accepted_step"] = row
                        phase = "accepted_controller_and_goal"
                        measurement, goal_frame, goal = accepted_measurement_and_goal(
                            entry.scene, model, world, current_tick, before, after, history
                        )
                        feedback = driver.accepted_measurement(
                            current_tick + 1, after.q, measurement, goal, input_q=before.q
                        )
                        history = goal.history
                        row.update(
                            goal_frame=plain(goal_frame),
                            goal_result=plain(goal),
                            feedback_accepted=plain(feedback),
                            driver_state_after=_controller_boundary(driver),
                        )
                        phase = "journal_append"
                        encoded = (
                            json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"
                        ).encode("utf-8")
                        journal.write(encoded)
                        journal.flush()
                        journal_hash.update(encoded)
                        report["recorded_steps"] += 1
                        report["journal"].update(
                            records=report["recorded_steps"], sha256=journal_hash.hexdigest()
                        )
                        report.pop("unrecorded_accepted_step")
                        report.update(
                            last_goal=plain(goal),
                            last_goal_frame=plain(goal_frame),
                            goal_success=bool(goal.success),
                            last_original_output_tick=current_tick + 1,
                        )
                        prior = after
                        if driver.stopped:
                            report.update(
                                status="controller_stopped", controller_status=driver.reason
                            )
                            break
                        if driver.active is None:
                            _require(
                                current_tick + 1 == initial_phase["route_end_tick"]
                                and driver.completed
                                == report["active_route"]["completed_routes_before"] + 1,
                                "route ended before its original completion boundary",
                            )
                            report["status"] = "route_complete"
                            break
                        if report["recorded_steps"] % 32 == 0:
                            _write_progress(stream, report)
                    else:
                        raise RuntimeError("active route did not finish at its declared last step")
                    report["controller_final"] = plain(driver.summary())
                    report["goal_history_final"] = plain(history)
                    phase = "scene_cleanup"
                finally:
                    # This runs while the public owner is alive, including after
                    # a failed candidate has restored its pre-control checkpoint.
                    try:
                        report["final_checkpoint"] = _save_final_checkpoint(
                            scene,
                            batch,
                            entry,
                            output,
                            report,
                            locals().get("history", entry.history),
                        )
                    except Exception as save_error:
                        report["checkpoint_save_error"] = _error_chain(save_error)
        except Exception as error:
            failed = getattr(error, "release_trial", None)
            if failed is not None:
                report["failed_candidate"] = dict(
                    physics_accepted_before_restore=failed["physics_accepted"],
                    restored=failed["restored"],
                    rollback_verification=failed["rollback_verification"],
                    input_states={str(k): _state(v, qa) for k, v in failed["input"].items()},
                    candidate_output_states=None
                    if failed["output"] is None
                    else {str(k): _state(v, qa) for k, v in failed["output"].items()},
                    submitted_control=plain(locals().get("command")),
                )
            if "driver" in locals():
                report["controller_final"] = plain(driver.summary())
            if "history" in locals():
                report["goal_history_final"] = plain(history)
            report.update(
                status="partial_failure",
                failure_stage=phase,
                failed_original_input_tick=current_tick,
                errors=_error_chain(error),
            )
        _write_progress(stream, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    result = run(
        args.summary,
        args.release,
        args.output,
        checkpoint_metadata_path=args.checkpoint,
        cache_root=args.cache_root,
    )
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "accepted_steps",
                    "recorded_steps",
                    "failed_original_input_tick",
                    "goal_success",
                    "checkpoint_save_error",
                )
            },
            allow_nan=False,
        )
    )
    raise SystemExit(
        0
        if result["status"] in ("route_complete", "controller_stopped")
        and "checkpoint_save_error" not in result
        else 1
    )


if __name__ == "__main__":
    main()
