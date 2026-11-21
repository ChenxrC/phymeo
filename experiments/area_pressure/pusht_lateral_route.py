"""A fresh all-lateral material episode from an audited ordinary-route input.

python -m experiments.area_pressure.pusht_lateral_route --summary PRIOR.json
  --source OLD-A.json --audit OLD-A-AUDIT.json --output NEW.json [--steps 4000]

No old asset capsule, solver state, force receipt or Goal history is restored.
Only the saved free quaternion is publicly normalized for this new episode; all
other q, complete v and time remain bitwise bound. Absolute reference
clock labels are retained for comparison; the new physical owner starts at tick 0.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.run_manipulation_control import plain

from .pusht_public_scene import H, _digest, _error_chain, _state, _write_progress


def _require(ok, message):
    if not ok:
        raise ValueError("lateral route: " + message)


def _read(path):
    path = Path(path).resolve()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        return json.load(stream)


def bind_fresh_snapshot(snapshot, source, *, quaternion_address=None):
    """Bind a cold owner; normalization is opt-in for this new material episode."""
    original = np.asarray(source["q"], np.float32)
    expected = original.copy()
    normalization = {}
    if quaternion_address is not None:
        a = quaternion_address
        _require(
            type(a) is int and original.ndim == 1 and 0 <= a <= len(original) - 4,
            "free quaternion address",
        )
        _require(
            np.asarray(source["q"], np.float64).tobytes() == original.astype(np.float64).tobytes(),
            "source q must contain actual FP32 values",
        )
        rotation = original[a : a + 4].astype(np.float64)
        length = float(np.linalg.norm(rotation))
        _require(np.isfinite(length) and length > 0.0, "finite nonzero source quaternion")
        normalized = rotation / length
        _require(
            np.max(np.abs(normalized - rotation)) < 1e-7, "public initial quaternion repair bound"
        )
        expected[a : a + 4] = normalized.astype(np.float32)
        normalization = dict(
            initialization_policy="new-episode-public-free-quaternion-normalization",
            quaternion_address=a,
            source_quaternion_norm=length,
        )

    def exact_public_fp32(value, expected_value, name):
        container = np.asarray(value)
        _require(
            container.dtype in (np.dtype("float32"), np.dtype("float64")),
            name + " snapshot dtype must be public float32 or float64",
        )
        _require(container.shape == expected_value.shape, name + " snapshot shape")
        _require(np.isfinite(container).all(), name + " snapshot must be finite")
        packed = container.astype(np.float32)
        _require(
            packed.astype(container.dtype).tobytes() == container.tobytes(),
            name + " snapshot must contain lossless FP32 values",
        )
        _require(
            np.array_equal(container, expected_value.astype(container.dtype))
            and packed.tobytes() == expected_value.tobytes(),
            name + " snapshot differs from authorized FP32 bytes",
        )
        return container, packed

    actual, actual_fp32 = exact_public_fp32(snapshot.q, expected, "q")
    expected_v = np.asarray(source["v"], np.float32)
    _require(
        np.asarray(source["v"], np.float64).tobytes() == expected_v.astype(np.float64).tobytes(),
        "source v must contain actual FP32 values",
    )
    actual_v, _ = exact_public_fp32(snapshot.v, expected_v, "v")
    valid_time = (
        np.asarray(snapshot.time, np.float64).tobytes()
        == np.asarray(source["time"], np.float64).tobytes()
    )
    _require(valid_time, "time snapshot differs from source float64 bytes")
    _require(snapshot.tick == 0, "tick snapshot must be zero for the new owner")
    _require(snapshot.last_step is None, "last receipt must be absent for the new owner")
    if quaternion_address is not None:
        published = actual[a : a + 4].astype(np.float64)
        published /= np.linalg.norm(published)
        angle = float(
            4
            * np.arctan2(
                np.linalg.norm(published - normalized), np.linalg.norm(published + normalized)
            )
        )
        normalization.update(
            nonrotation_q_fp32_bytes_equal=True,
            normalized_quaternion_fp32_bytes_equal=True,
            quaternion_changed=actual_fp32[a : a + 4].tobytes() != original[a : a + 4].tobytes(),
            quaternion_angle_change_rad=angle,
            source_q=original.tolist(),
            published_q=actual.tolist(),
        )
    return dict(
        q_fp32_bytes_equal=actual_fp32.tobytes() == original.tobytes(),
        v_fp32_bytes_equal=True,
        snapshot_q_dtype=str(actual.dtype),
        snapshot_v_dtype=str(actual_v.dtype),
        snapshot_q_lossless_fp32=True,
        snapshot_v_lossless_fp32=True,
        time_float64_bytes_equal=valid_time,
        source_local_tick=source["local_tick"],
        new_local_tick=0,
        previous_receipt_inherited=False,
        initial_time=snapshot.time,
        **normalization,
    )


def bootstrap_driver(driver, source, tick, previous_budget):
    _require(
        driver.active is None
        and not driver.monitor.started
        and not driver.monitor.history
        and not driver.routes
        and driver.completed == 0
        and driver.total_impulse == 0
        and driver.segment_impulse == 0
        and driver.previous_direction is None
        and not driver.stopped
        and driver.tick == 0,
        "a fresh controller is required; old history cannot be cleared",
    )
    ik, paths = previous_budget
    _require(
        type(tick) is int
        and tick >= 200
        and type(ik) is int
        and type(paths) is int
        and 0 <= ik <= driver.planner.max_ik
        and 0 <= paths <= driver.planner.max_paths,
        "saved reference clock and original cumulative planning budgets",
    )
    driver.tick = tick
    driver.last_q = np.asarray(source["q"], dtype=np.float32).copy()
    driver.initial_cost = driver.last_cost = driver.cost(driver.last_q)
    driver.monitor.start(tick, driver.initial_cost)
    driver.planner.ik_calls, driver.planner.path_calls = ik, paths
    return dict(
        goal_history_inherited=False,
        source_force_history_inherited=False,
        controller_history_inherited=False,
        source_budget_counts=[ik, paths],
        reference_tick_origin=tick,
        initial_progress_cost=driver.initial_cost,
        qualification="new progress/Goal episode; only q/v/clock labels, target and budget accounting come from source",
    )


def initial_load_assessment(pair, geometry, identity):
    _require(
        pair["identity"] == geometry.surface.identity == identity,
        "first receipt must belong to the new material",
    )
    fn, area = np.asarray(pair["normal_force"]), np.asarray(pair["dA"])
    a, b = np.asarray(pair["force_on_a"]), np.asarray(pair["force_on_b"])
    empty = (
        not geometry.surface.triangles
        and fn.shape == area.shape == (0,)
        and a.shape == b.shape == (0, 3)
        and np.isfinite(a).all()
        and np.isfinite(b).all()
    )
    return dict(
        safe=bool(empty),
        reason="actual_new_material_zero_initial_load"
        if empty
        else "initial_new_material_load_not_zero",
        normal_load_n=float(fn.sum()),
        input_triangles=len(geometry.surface.triangles),
        force_history_source="first actual accepted trial; no archived previous force",
    )


def accumulate_measurement(totals, before, after, measurement, tick):
    _require(
        measurement["input_tick"] == tick
        and measurement["output_tick"] == tick + 1
        and measurement["time_s"] == before["time"]
        and measurement["phase"] == "accepted-step-input",
        "actual wrench must belong to this accepted input geometry",
    )
    dt = after["time"] - before["time"]
    force, moment = (
        np.asarray(measurement["group_force_world"], float),
        np.asarray(measurement["group_moment_com_world"], float),
    )
    _require(
        abs(dt - H) < 1e-7
        and force.shape == moment.shape == (3,)
        and np.isfinite(force).all()
        and np.isfinite(moment).all(),
        "finite actual wrench and step duration",
    )
    return dict(
        steps=totals.get("steps", 0) + 1,
        object_force_impulse_world_ns=(
            np.asarray(totals.get("object_force_impulse_world_ns", [0.0, 0.0, 0.0])) + force * dt
        ).tolist(),
        object_angular_impulse_z_nms=float(
            totals.get("object_angular_impulse_z_nms", 0.0) + moment[2] * dt
        ),
        phase="sum of actual object-side input-COM wrenches over complete accepted callbacks",
    )


def load_route_source(source_path, audit_path):
    source_path, audit_path = Path(source_path).resolve(), Path(audit_path).resolve()
    summary, audit = _read(source_path), _read(audit_path)
    if audit.get("schema") == "phymeo-pusht-pose-replan-comparison-v1":
        _require(audit["status"] == "verified-saved-prefixes", "verified source comparison")
        audit = audit["cases"]["A"]
    _require(
        summary["schema"] == "phymeo-pusht-pose-replan-case-v1"
        and summary["key"] == "A"
        and audit["status"] == "verified-saved-prefix"
        and audit["source"]["sha256"] == _digest(source_path)
        and audit["source"]["journal_sha256"] == summary["journal"]["sha256"]
        and audit["recorded_steps"] == summary["recorded_steps"]
        and audit["goal"] == summary["goal"],
        "source summary/audit binding",
    )
    route_path = (source_path.parent / summary["ordinary_route"]["route_file"]).resolve()
    _require(
        _digest(route_path) == summary["ordinary_route"]["route_sha256"],
        "saved ordinary route digest",
    )
    saved = _read(route_path)
    tick = saved["tick"]
    _require(
        tick == summary["active_route"]["start_tick"] == summary["ordinary_route"]["tick"],
        "ordinary route input clock",
    )
    state, pending, callbacks, sha = None, None, 0, hashlib.sha256()
    journal_path = source_path.parent / summary["journal"]["file"]
    with journal_path.open("rb") as stream:
        for _ in range(summary["journal"]["records"]):
            line = stream.readline()
            _require(line.endswith(b"\n"), "complete audited journal prefix")
            sha.update(line)
            event = json.loads(line)
            if event["event"] == "candidate" and event["payload"]["selected"]:
                pending = (event["trial_index"], event["payload"])
            elif event["event"] == "accepted_callback":
                _require(
                    pending is not None and pending[0] == event["trial_index"],
                    "source callback identity",
                )
                row = pending[1]
                callbacks += 1
                if row["phase"] == "regular" and row["original_input_tick"] == tick:
                    _require(state is None, "unique source ordinary entry")
                    state = row["input_state"]
                pending = None
    _require(
        sha.hexdigest() == summary["journal"]["sha256"] and callbacks == summary["recorded_steps"],
        "audited source journal",
    )
    _require(
        state is not None
        and state["q"] == saved["q"] == summary["ordinary_route"]["source_q"]
        and state["v"] == saved["v"] == summary["ordinary_route"]["source_v"],
        "actual accepted ordinary input q/v",
    )
    release_path = source_path.parent / summary["release_reference"]["file"]
    _require(
        _digest(release_path) == summary["release_reference"]["sha256"],
        "original pre-plan budget receipt",
    )
    release = _read(release_path)["route"]
    return dict(
        state=state,
        tick=tick,
        goal=summary["goal"],
        previous_budget=(release["ik_calls"], release["path_calls"]),
        old_asset_sha256=summary["provenance"]["scene_asset_sha256"],
        provenance=dict(
            source_report_path=str(source_path),
            source_report_sha256=_digest(source_path),
            source_audit_path=str(audit_path),
            source_audit_sha256=_digest(audit_path),
            source_journal_path=str(journal_path.resolve()),
            source_journal_sha256=sha.hexdigest(),
            source_route_path=str(route_path),
            source_route_sha256=_digest(route_path),
            source_release_path=str(release_path.resolve()),
            source_release_sha256=_digest(release_path),
            source_input_tick=tick,
            source_input_state=state,
            source_asset_sha256=summary["provenance"]["scene_asset_sha256"],
        ),
    )


def initial_separation(model, asset, state):
    from .pusht_release_material import _cpu_geometry, _data, _phase

    data = _data(model, state)
    bodies = (
        model.body(asset.identity.compliant_body).id,
        model.body(asset.identity.rigid_body).id,
    )
    geometry, transformed = _cpu_geometry(asset, data, bodies)
    whole, faces = _phase(model, asset, data, bodies, geometry, transformed=transformed)
    _require(
        whole["strict"] and not whole["contact"] and not geometry.surface.triangles,
        "new material requires strictly separated, empty finite entry geometry",
    )
    return dict(
        strictly_separated=True,
        cpu_triangles=0,
        static_pressure_load_n=0.0,
        actual_published_tool_force=None,
        faces=plain(faces),
        source_q=plain(state["q"]),
        source_v=plain(state["v"]),
        source_time=state["time"],
        qualification="all tetrahedron/finite-triangle pairs have outward-rounded CPU-FK SAT witnesses; first actual receipt zero-load is checked separately",
    )


def episode_measurement(entry, world, before, after, history):
    """Keep absolute force/control labels while starting a new local Goal history."""
    from .pusht_public_feedback import accepted_measurement_and_goal

    return accepted_measurement_and_goal(
        entry.scene,
        entry.model,
        world,
        entry.tick + before.tick,
        before,
        after,
        history,
        goal_tick=before.tick,
    )


def run(prior_path, source_path, audit_path, output, *, steps=4000, cache_root=None):
    from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
    from phymeo.backends.mujoco_heterogeneous_area_scene import (
        MuJoCoTyxHeterogeneousAreaPressureScene,
    )
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState

    from .pusht_active_route import (
        _controller_boundary,
        _first_binding,
        _save_final_checkpoint,
        active_command,
    )
    from .pusht_load_compensation import PushLoadCompensation
    from .pusht_loaded_release import _command_record, attempt_release_candidate
    from .pusht_pose_replan import _fresh_entry, _plan_ordinary, freeze_declared_material
    from .pusht_public_scene import (
        _step_record,
        build_scene_spec,
        build_side_asset,
        legal_initial_q,
        side_geometry_provenance,
    )

    _require(type(steps) is int and 1 <= steps <= 4000, "finite step budget 1..4000")
    output, prior_path = Path(output).resolve(), Path(prior_path).resolve()
    journal_path = output.with_suffix(".trials.jsonl")
    _require(not output.exists() and not journal_path.exists(), "new report/journal paths required")
    source = load_route_source(source_path, audit_path)
    report = dict(
        schema="phymeo-pusht-lateral-route-v1",
        status="running",
        world=6,
        initialization_mode="new-material-episode-from-audited-qv",
        side_coverage="all-lateral",
        selection_policy="joint-pose-balanced",
        material_admission="declared-finite-faces-v1",
        control_variant="load-compensation-half",
        goal=source["goal"],
        start_tick=source["tick"],
        local_tick_origin=0,
        source_reference_tick_origin=source["tick"],
        goal_tick_origin=0,
        goal_history_initial=None,
        step_budget=steps,
        physical_attempts=0,
        physical_trial_steps=0,
        accepted_steps=0,
        recorded_steps=0,
        rejected_trials=0,
        metrics={},
        peak_normal_load_n=None,
        positive_load_steps=0,
        provenance=dict(
            source["provenance"],
            prior_summary_path=str(prior_path),
            prior_summary_sha256=_digest(prior_path),
            source_reference_tick_origin=source["tick"],
            goal_tick_origin=0,
            source_capsule_restored=False,
            old_force_history_inherited=False,
            old_solver_cache_inherited=False,
        ),
        episode_boundary="New physical material and cold owner. Only the archived free quaternion is publicly normalized; all other q, complete v and time are byte-bound. Planning and feedback use the published initial state. Old force, Goal and controller histories are not continued.",
        journal=dict(file=journal_path.name, records=0, sha256=hashlib.sha256().hexdigest()),
    )
    phase, trial_index, history = "prepare_new_episode", None, None
    digest = hashlib.sha256()
    with output.open("x+", encoding="utf-8") as stream, journal_path.open("xb") as journal:

        def emit(kind, payload, index=None):
            event = dict(
                schema="phymeo-pusht-lateral-route-event-v1",
                event=kind,
                trial_index=index,
                payload=plain(payload),
            )
            encoded = (json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n").encode()
            journal.write(encoded)
            journal.flush()
            digest.update(encoded)
            report["journal"].update(
                records=report["journal"]["records"] + 1, sha256=digest.hexdigest()
            )

        _write_progress(stream, report)
        try:
            entry = _fresh_entry(
                prior_path, {"metadata": {"state": source["state"]}}, output.with_suffix(".routes")
            )
            declared_q = legal_initial_q(entry.model, source["state"]["q"])
            entry.initial = MuJoCoInitialState(
                declared_q, source["state"]["v"], time=source["state"]["time"]
            )
            entry.state = dict(source["state"], local_tick=0)
            entry.tick, entry.previous, entry.history = source["tick"], None, None
            entry.provenance = report["provenance"]
            model, driver, world = entry.model, entry.driver, 6
            actual_goal = dict(
                target_xy=plain(driver.target.position[:2]),
                target_yaw=driver.target.yaw,
                position_tolerance_m=driver.goal.position_tolerance_m,
                yaw_tolerance_rad=driver.goal.yaw_tolerance_rad,
            )
            _require(actual_goal == source["goal"], "unchanged original Goal target/tolerances")
            spec = build_scene_spec(entry.scene.mjcf, side_coverage="all-lateral")
            asset = build_side_asset(model, side_coverage="all-lateral")
            report["material_geometry_provenance"] = side_geometry_provenance(
                model, side_coverage="all-lateral"
            )
            qa = int(model.jnt_qposadr[model.joint("object-free").id])
            phase = "construct_new_scene"
            with MuJoCoTyxHeterogeneousBatch(
                (spec,),
                (MuJoCoTyxWorldSpec(world, spec.key, entry.initial),),
                mode="cuda",
                cache_root=cache_root,
            ) as batch:
                scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (asset,)})
                _require(
                    scene.asset_sha256 != source["old_asset_sha256"],
                    "new material must have a distinct asset digest",
                )
                report["provenance"].update(
                    scene_asset_sha256=scene.asset_sha256, material_identity=plain(asset.identity)
                )
                initial = batch.integration_snapshot()[world]
                report["initialization_proof"] = bind_fresh_snapshot(
                    initial, source["state"], quaternion_address=qa + 3
                )
                entry.state = _state(initial, qa)
                report["controller_bootstrap"] = bootstrap_driver(
                    driver, entry.state, source["tick"], source["previous_budget"]
                )
                report.update(
                    initial_state=_state(initial, qa),
                    final_state=_state(initial, qa),
                    controller_initial=plain(driver.summary()),
                )
                try:
                    phase = "verify_new_initial_separation"
                    report["initial_separation"] = initial_separation(
                        model, asset, report["initial_state"]
                    )
                    emit("initialization", report)
                    phase = "ordinary_plan"
                    material = freeze_declared_material(
                        model,
                        initial.q,
                        asset,
                        model_key=spec.key,
                        scene_asset_sha256=scene.asset_sha256,
                    )
                    route, plan = _plan_ordinary(
                        entry,
                        initial,
                        entry.tick,
                        selection_policy="joint-pose-balanced",
                        material_faces=material,
                    )
                    report["ordinary_route"] = plan
                    emit("planner", plan)
                    if route.status != "sampled_reference_ready":
                        report["status"] = "planning_rejected"
                    else:
                        report.update(
                            actual_selected_contact=route.proposal.index,
                            matches_prior_face4_prediction=route.proposal.index == 4,
                        )
                        duration = float(route.nodes[-1][0])
                        end = entry.tick + round(duration / H)
                        _require(
                            duration > 0 and abs(round(duration / H) * H - duration) <= 1e-12,
                            "tick-aligned route",
                        )
                        report["active_route"] = dict(
                            start_tick=entry.tick,
                            end_tick=end,
                            nodes=plain(route.nodes),
                            file=plan["route_file"],
                            sha256=plan["route_sha256"],
                            completed_routes_before=0,
                        )
                        compensate = PushLoadCompensation(entry, identity=asset.identity)
                        while report["physical_attempts"] < steps:
                            before = batch.integration_snapshot()[world]
                            tick = entry.tick + before.tick
                            _require(tick < end, "no second route or reference after end")
                            phase = "active_reference"
                            control, controller = active_command(
                                entry, before, compensate=None if before.tick == 0 else compensate
                            )
                            if before.tick == 0:
                                report["first_command_binding"] = _first_binding(
                                    entry, before, control, controller
                                )
                                controller["bootstrap_zero_compensation"] = dict(
                                    previous_receipt=None,
                                    applied_correction_nm=[0.0, 0.0],
                                    source="verified empty new material entry; actual first-step zero load still required",
                                )
                            command = _command_record(control, tick)

                            def assess(inputs, outputs, geometry):
                                after = outputs[world]
                                (pair,) = after.last_step.raw_contact_fields["area_friction"]
                                _require(
                                    pair["identity"] == asset.identity,
                                    "actual new material receipt identity",
                                )
                                if inputs[world].tick == 0:
                                    return initial_load_assessment(
                                        pair, geometry[world][0], asset.identity
                                    )
                                return dict(safe=True, reason="actual public scene acceptance")

                            trial_index = report["physical_attempts"]
                            report["physical_attempts"] += 1
                            phase = "scene_step"
                            trial = attempt_release_candidate(
                                scene, batch, {world: control}, assess
                            )
                            report["physical_trial_steps"] += 1
                            after = trial["candidate_output"][world]
                            if trial["selected"]:
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
                            measurement, frame, goal = episode_measurement(
                                entry, world, before, after, history
                            )
                            row.update(
                                index=trial_index,
                                trial_index=trial_index,
                                phase="regular",
                                controller=plain(controller),
                                side_coverage="all-lateral",
                                material_identity=plain(asset.identity),
                                stored_control_matches_submitted=True,
                                selected=trial["selected"],
                                restored=trial["restored"],
                                rollback_verification=trial["rollback_verification"],
                                measurement=plain(measurement),
                                assessment=plain(trial["assessment"]),
                            )
                            emit("candidate", row, trial_index)
                            if before.tick == 0:
                                report["first_actual_load_check"] = plain(trial["assessment"])
                            if not trial["selected"]:
                                report["status"] = "initial_force_check_rejected"
                                break
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
                            emit("accepted_callback", callback, trial_index)
                            report["recorded_steps"] += 1
                            report.pop("unrecorded_accepted_step", None)
                            report["metrics"] = metrics
                            fn = row["area"]["normal_load_n"]
                            report["peak_normal_load_n"] = (
                                fn
                                if report["peak_normal_load_n"] is None
                                else max(fn, report["peak_normal_load_n"])
                            )
                            report["positive_load_steps"] += int(fn > 0)
                            report.update(last_goal=plain(goal), goal_success=bool(goal.success))
                            entry.previous = dict(row, **callback)
                            if driver.stopped:
                                report.update(
                                    status="controller_stopped", controller_status=driver.reason
                                )
                                break
                            if driver.active is None:
                                _require(driver.tick == end, "ordinary route ended at wrong tick")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("summary", "source", "audit", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    result = run(
        args.summary,
        args.source,
        args.audit,
        args.output,
        steps=args.steps,
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
                    "peak_normal_load_n",
                    "goal_success",
                    "failure_stage",
                )
            }
        )
    )
    raise SystemExit(0 if result["status"] in ("route_complete", "controller_stopped") else 1)


if __name__ == "__main__":
    main()
