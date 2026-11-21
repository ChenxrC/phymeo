"""Saved-file validation fixtures only; these records are not CUDA/physics evidence."""

import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest

from experiments.area_pressure.pusht_active_comparison import compare


def _write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state(local_tick, time):
    q = [0.0, 0.0, 0.5, 0.0, 0.125, 1.0, 0.0, 0.0, 0.0]
    return dict(
        local_tick=local_tick,
        time=time,
        q=q,
        v=[0.0] * 8,
        object_position_world_m=q[2:5],
        object_quaternion_wxyz=q[5:9],
    )


def _case(tmp_path):
    start, route_start, duration = 2020, 2018, 0.006
    nodes = [[0.0, "approach", [0.0, 0.0]], [duration, "push", [0.003, -0.001]]]
    route_path = tmp_path / "route-02.json"
    route_sha = _write_json(
        route_path,
        dict(
            tick=route_start,
            q=_state(0, 0)["q"],
            v=[0.0] * 8,
            route=dict(
                status="sampled_reference_ready",
                nodes=nodes,
                proposal=dict(direction_world=[1.0, 0.0, 0.0]),
            ),
        ),
    )
    prior_path = tmp_path / "prior.json"
    prior_sha = _write_json(prior_path, {"schema": "phymeo-pusht-public-feedback-v1"})
    release_path = tmp_path / "release.json"
    release = dict(
        schema="phymeo-pusht-loaded-release-v1",
        status="regular_step_budget",
        world=6,
        regular_plan_ready=True,
        release_monitor_status="released",
        start_tick=1700,
        selected_steps=320,
        final_state=_state(320, 2.020),
        controller_final=dict(tick=start),
        provenance=dict(source_summary_sha256=prior_sha),
    )
    release_sha = _write_json(release_path, release)
    initial = _state(320, 2.020)
    capsule = tmp_path / "source.scene"
    capsule.write_bytes(b"opaque reader fixture; not a physics checkpoint")
    capsule_sha = hashlib.sha256(capsule.read_bytes()).hexdigest()
    checkpoint_path = tmp_path / "source.metadata.json"
    checkpoint_sha = _write_json(
        checkpoint_path,
        dict(
            schema="phymeo-pusht-reconstructed-scene-v1",
            world=6,
            release_sha256=release_sha,
            state=initial,
            source_tick=start,
            checked_steps=320,
            capsule=capsule.name,
            sha256=capsule_sha,
            size=capsule.stat().st_size,
            asset_sha256="a" * 64,
        ),
    )
    goal = dict(
        target_xy=[0.6, 0.0], target_yaw=0.0, position_tolerance_m=0.015, yaw_tolerance_rad=0.1
    )
    summary = dict(
        schema="phymeo-pusht-active-route-v1",
        status="route_complete",
        world=6,
        start_tick=start,
        requested_steps=4,
        accepted_steps=4,
        recorded_steps=4,
        initial_state=initial,
        final_state=_state(324, 2.024),
        local_tick_origin=320,
        goal=goal,
        checkpoint_restore=dict(
            checkpoint_sha256=capsule_sha,
            checked_prefix_steps=320,
            restored_local_tick=320,
            absolute_input_tick=start,
            time=2.020,
        ),
        provenance=dict(
            checkpoint_metadata_path=str(checkpoint_path),
            checkpoint_metadata_sha256=checkpoint_sha,
            scene_asset_sha256="a" * 64,
            prior_summary_path=str(prior_path),
            prior_summary_sha256=prior_sha,
            release_report_path=str(release_path),
            release_report_sha256=release_sha,
        ),
        active_route=dict(
            start_tick=route_start,
            end_tick=2024,
            duration_s=duration,
            elapsed_at_entry_s=0.002,
            remaining_steps=4,
            nodes=nodes,
            file=str(route_path),
            sha256=route_sha,
            route_index=2,
            completed_routes_before=1,
        ),
        controller_final=dict(tick=2024, completed_routes=2, status="continue"),
    )
    rows = []
    for index, load in enumerate((0.0, 1.0, 2.0, 0.0)):
        tick = start + index
        before, after = _state(320 + index, tick * 0.001), _state(321 + index, (tick + 1) * 0.001)
        elapsed = (tick - route_start) * 0.001
        u = elapsed / duration
        target = np.array(nodes[-1][2]) * (u * u * (3 - 2 * u))
        velocity = np.array(nodes[-1][2]) * (6 * u * (1 - u) / duration)
        effort = 25 * target + 3 * velocity
        direct = effort.astype(np.float32)
        raw = np.arctanh(direct / np.float32(20.0))
        channels = dict(
            ctrl=direct.astype(float).tolist(),
            qfrc_applied=[0.0] * 8,
            xfrc_applied=[[0.0] * 6 for _ in range(5)],
        )
        pd = dict(
            qpos_addresses=[0, 1],
            dof_addresses=[0, 1],
            actuator_ids=[0, 1],
            position=[0.0, 0.0],
            velocity=[0.0, 0.0],
            target=target.tolist(),
            target_velocity=velocity.tolist(),
            bias=[0.0, 0.0],
            unclipped_effort=effort.tolist(),
            effort=effort.tolist(),
            saturated=[False, False],
        )
        controller = dict(
            input_tick=tick,
            local_tick=320 + index,
            input_time=before["time"],
            measurement_q=before["q"],
            measurement_v=before["v"],
            target_q=target.tolist(),
            target_v=velocity.tolist(),
            pd=pd,
            goal=goal,
            stage="push",
            generated_channels=channels,
            route=dict(
                route=2,
                route_start_tick=route_start,
                route_elapsed_s=elapsed,
                route_file=str(route_path),
                route_sha256=route_sha,
            ),
            transport=dict(
                kind="original-env-fp32-atanh-cuda-tanh",
                device="cuda:0",
                operator="bias + scale * torch.tanh(raw)",
                scale=20.0,
                bias=0.0,
                direct_pd_ctrl=direct.tolist(),
                raw_action=raw.tolist(),
                transported_ctrl=channels["ctrl"],
                direct_to_transported_delta=[0.0, 0.0],
            ),
        )
        measurement = dict(
            world=6,
            input_tick=tick,
            output_tick=tick + 1,
            time_s=before["time"],
            group_force_world=[load, 0.0, 0.0],
        )
        rows.append(
            dict(
                schema="phymeo-pusht-active-route-step-v1",
                index=index,
                original_input_tick=tick,
                original_output_tick=tick + 1,
                input_state=before,
                output_state=after,
                submitted_control=dict(tick=tick, output_tick=tick + 1, **channels),
                stored_control_matches_submitted=True,
                controller=controller,
                goal_frame=dict(
                    phase="accepted-entry",
                    world=6,
                    tick=tick,
                    time_s=before["time"],
                    object_position=before["q"][2:5],
                    object_wxyz=before["q"][5:9],
                ),
                goal_result=dict(
                    success=False,
                    history=dict(tick=tick, time_s=before["time"]),
                    metrics=[["position_error_m", 0.1], ["yaw_error_rad", 0.0]],
                ),
                feedback_accepted=dict(measurement=measurement),
                driver_state_after=dict(tick=tick + 1, completed_routes=1 + (index == 3)),
                solver=dict(
                    schema="mujoco-tyx-area-friction-resident-native-v1",
                    meta=[0, 1, 2, 1],
                    normalized_residual=0.0,
                    native_solve_calls=2,
                    outer_iterations=1,
                    output_velocity=after["v"],
                ),
                area=dict(
                    normal_load_n=load,
                    raw_normal_force_n=[load] if load else [],
                    zero_normal_load=load == 0,
                    samples=int(load > 0),
                    active_samples=int(load > 0),
                ),
                native=dict(
                    object_support_normal_n=1.5,
                    supports=[dict(geometry_names=["support", "object-0"], normal_n=1.5)],
                ),
                geometry=dict(
                    strict_disjoint_certified=load == 0, no_positive_area_intersection=load == 0
                ),
            )
        )
    c = rows[0]["controller"]
    summary["first_command_binding"] = dict(
        measurement_q=c["measurement_q"],
        measurement_v=c["measurement_v"],
        input_tick=start,
        input_time=initial["time"],
        route_start_tick=route_start,
        route_elapsed_s=0.002,
        target_q=c["target_q"],
        target_v=c["target_v"],
        independently_recomputed_target_q=c["target_q"],
        independently_recomputed_target_v=c["target_v"],
        exact_matches=True,
    )
    return deepcopy(summary), deepcopy(rows)


def _write(tmp_path, summary, rows):
    journal = tmp_path / "active.steps.jsonl"
    payload = "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows).encode()
    journal.write_bytes(payload)
    summary["journal"] = dict(
        file=journal.name, records=len(rows), sha256=hashlib.sha256(payload).hexdigest()
    )
    path = tmp_path / "active.json"
    _write_json(path, summary)
    return path


def test_saved_active_route_distinguishes_reference_completion_contact_and_goal(tmp_path):
    summary, rows = _case(tmp_path)
    result = compare(_write(tmp_path, summary, rows))
    assert result["verified_steps"] == 4 and result["source_endpoint_bound"]
    assert result["termination"]["classification"] == "route_complete_without_goal"
    assert result["contact"]["first_positive_input_tick"] == 2021
    assert result["contact"]["peak_normal_load_n"] == 2.0
    assert result["contact"]["normal_impulse_ns"] == pytest.approx(0.003)
    assert result["contact"]["source_prior_release_reported"] is True
    assert result["control"]["reference_phase_verified_steps"] == 4
    assert result["integration"]["verified_steps"] == 4
    assert result["object"]["position_error_change_m"] == pytest.approx(0)
    assert result["object"]["motion_along_planned_push_m"] == pytest.approx(0)
    assert result["goal"]["last_input_tick"] == 2023 and not result["goal"]["ever_success"]


@pytest.mark.parametrize(
    "fault",
    [
        "tick",
        "controller-phase",
        "goal-phase",
        "route",
        "target",
        "false-load",
        "integration",
        "clock",
    ],
)
def test_saved_reader_rejects_discontinuous_misphased_or_falsely_loaded_records(tmp_path, fault):
    summary, rows = _case(tmp_path)
    if fault == "tick":
        rows[1]["original_input_tick"] += 1
    elif fault == "controller-phase":
        rows[1]["controller"]["input_time"] = rows[1]["output_state"]["time"]
    elif fault == "goal-phase":
        rows[1]["goal_frame"]["tick"] += 1
    elif fault == "route":
        rows[1]["controller"]["route"]["route"] = 3
    elif fault == "target":
        rows[1]["controller"]["target_q"][0] += 0.01
    elif fault == "false-load":
        rows[1]["area"]["raw_normal_force_n"] = [0.0]
    elif fault == "integration":
        endpoint = rows[-1]["output_state"]
        endpoint["q"][2] = float(np.float32(0.501))
        endpoint["object_position_world_m"] = endpoint["q"][2:5]
        summary["final_state"] = deepcopy(endpoint)
    else:
        rows[-1]["output_state"]["time"] += 0.0005
        summary["final_state"] = deepcopy(rows[-1]["output_state"])
    with pytest.raises(ValueError):
        compare(_write(tmp_path, summary, rows))


def test_source_file_and_first_command_flags_are_independently_bound(tmp_path):
    summary, rows = _case(tmp_path)
    summary["first_command_binding"]["target_q"] = [123.0, 0.0]
    summary["first_command_binding"]["exact_matches"] = True
    with pytest.raises(ValueError):
        compare(_write(tmp_path, summary, rows))
    summary, rows = _case(tmp_path)
    path = _write(tmp_path, summary, rows)
    (tmp_path / "release.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="source|digest"):
        compare(path)


def test_post_accept_callback_failure_reports_only_the_verified_recorded_prefix(tmp_path):
    summary, rows = _case(tmp_path)
    summary.update(
        status="partial_failure", recorded_steps=3, failure_stage="accepted_controller_and_goal"
    )
    summary["unrecorded_accepted_step"] = dict(
        input_state=rows[-1]["input_state"],
        output_state=rows[-1]["output_state"],
        submitted_control=rows[-1]["submitted_control"],
    )
    result = compare(_write(tmp_path, summary, rows[:-1]))
    assert result["verified_steps"] == 3 and result["accepted_steps"] == 4
    assert result["termination"]["classification"] == "post_accept_callback_failure"
    assert result["unrecorded_accepted_step"]["transition_verified"] is False
    assert result["goal"]["last_input_tick"] == 2022
    assert result["contact"]["normal_impulse_ns"] == pytest.approx(0.003)


def _variant_case(
    tmp_path, *, stage="push", zero_load=False, assisting=False, joint_position=(0.0, 0.0)
):
    """Synthetic saved-record fixture, with an analytic RR point Jacobian."""
    import math
    from pathlib import Path

    summary, rows = _case(tmp_path)
    summary["control_variant"] = "load-compensation-half"
    direction = np.array([0.6, 0.8, 0.0])
    # Keep the static saved-state fixture coherent at either joint posture.
    for state in (summary["initial_state"], summary["final_state"]):
        state["q"][:2] = joint_position
    summary["first_command_binding"]["measurement_q"][:2] = joint_position
    for row in rows:
        for state in (row["input_state"], row["output_state"]):
            state["q"][:2] = joint_position
        c = row["controller"]
        c["measurement_q"] = deepcopy(row["input_state"]["q"])
        pd = c["pd"]
        pd["position"] = list(joint_position)
        raw = (
            np.asarray(pd["bias"])
            + 25 * (np.asarray(pd["target"]) - joint_position)
            + 3 * np.asarray(pd["target_velocity"])
        )
        pd["unclipped_effort"] = raw.tolist()
        pd["effort"] = np.clip(raw, -19.9, 19.9).tolist()
        pd["saturated"] = (np.abs(raw) > 19.9).tolist()
    # The catalog RR has two 0.3 m links and fixed yaw offsets +0.6, -1.2.
    # This current-posture analytic value is independent of mj_jacGeom.
    a, b = 0.6 + joint_position[0], -0.6 + sum(joint_position)
    jacobian = 0.3 * np.array(
        [
            [-math.sin(a) - math.sin(b), -math.sin(b)],
            [math.cos(a) + math.cos(b), math.cos(b)],
            [0.0, 0.0],
        ]
    )
    route_path = Path(summary["active_route"]["file"])
    route = json.loads(route_path.read_text())
    route["route"]["proposal"]["direction_world"] = direction.tolist()
    for node in route["route"]["nodes"]:
        node[1] = stage
    route_sha = _write_json(route_path, route)
    summary["active_route"].update(nodes=deepcopy(route["route"]["nodes"]), sha256=route_sha)
    seed_force = np.zeros(3) if zero_load else (-1000 if not assisting else 1000) * direction
    seed = dict(
        selected=True,
        restored=False,
        original_input_tick=summary["start_tick"] - 1,
        original_output_tick=summary["start_tick"],
        input_state=_state(319, 2.019),
        output_state=deepcopy(summary["initial_state"]),
        area=dict(total_force_on_pusher_world_n=seed_force.tolist()),
    )
    seed["input_state"]["q"][:2] = joint_position
    release_path = Path(summary["provenance"]["release_report_path"])
    release = json.loads(release_path.read_text())
    # An unselected candidate after the seed must never become feedback input.
    rejected = deepcopy(seed)
    rejected.update(selected=False, restored=True)
    rejected["area"]["total_force_on_pusher_world_n"] = [1234.0, 5678.0, 0.0]
    release["final_state"] = deepcopy(summary["initial_state"])
    release["trials"] = [seed, rejected]
    release_sha = _write_json(release_path, release)
    summary["provenance"]["release_report_sha256"] = release_sha
    metadata_path = Path(summary["provenance"]["checkpoint_metadata_path"])
    metadata = json.loads(metadata_path.read_text())
    metadata["release_sha256"] = release_sha
    metadata["state"] = deepcopy(summary["initial_state"])
    summary["provenance"]["checkpoint_metadata_sha256"] = _write_json(metadata_path, metadata)
    previous_force = seed_force
    for row in rows:
        c, before, area = row["controller"], row["input_state"], row["area"]
        c["stage"], c["route"]["route_sha256"] = stage, route_sha
        force = (0 if zero_load else area["normal_load_n"]) * direction
        if not assisting:
            force = -force
        area["total_force_on_pusher_world_n"] = force.tolist()
        row["feedback_accepted"]["measurement"]["group_force_world"] = (-force).tolist()
        enabled = stage in ("entry", "push")
        resisting = max(0.0, -float(previous_force @ direction))
        correction = jacobian.T @ (0.5 * resisting * direction) if enabled else np.zeros(2)
        combined = np.asarray(c["pd"]["unclipped_effort"]) + correction
        c["load_compensation"] = dict(
            source_local_tick=before["local_tick"],
            source_output_time=before["time"],
            previous_force_on_tool_world_n=previous_force.tolist(),
            push_direction_world=direction.tolist(),
            jacobian_world=jacobian.tolist(),
            resisting_force_n=resisting,
            gain=0.5,
            enabled=enabled,
            correction_nm=correction.tolist(),
            combined_unclipped_nm=combined.tolist(),
            combined_effort_nm=np.clip(combined, -19.9, 19.9).tolist(),
        )
        _variant_transport(row)
        previous_force = force
    return summary, rows


def _variant_transport(row):
    """Bind fixture transport to the combined command, without invoking CUDA."""
    c = row["controller"]
    direct = np.asarray(c["load_compensation"]["combined_effort_nm"], dtype=np.float32)
    # Like the existing fixture, a zero delta is a saved transport witness only.
    c["transport"].update(
        direct_pd_ctrl=direct.tolist(),
        raw_action=np.arctanh(direct / np.float32(20)).tolist(),
        transported_ctrl=direct.tolist(),
        direct_to_transported_delta=[0.0, 0.0],
    )
    c["generated_channels"]["ctrl"] = direct.tolist()
    row["submitted_control"]["ctrl"] = direct.tolist()


@pytest.mark.parametrize("stage", ["entry", "push", "approach"])
def test_half_load_compensation_audits_previous_force_current_jacobian_and_transport(
    tmp_path, stage
):
    summary, rows = _variant_case(tmp_path, stage=stage)
    result = compare(_write(tmp_path, summary, rows))
    assert result["control"]["variant"] == "load-compensation-half"
    assert result["control"]["load_compensation_verified_steps"] == 4
    c = rows[0]["controller"]["load_compensation"]
    assert c["resisting_force_n"] == pytest.approx(1000)
    if stage in ("entry", "push"):
        assert c["combined_effort_nm"] == [19.9, 19.9]
        assert max(c["combined_unclipped_nm"]) > 19.9
    else:
        assert c["correction_nm"] == [0.0, 0.0]


def test_half_load_compensation_does_not_oppose_an_assisting_load(tmp_path):
    summary, rows = _variant_case(tmp_path, assisting=True)
    result = compare(_write(tmp_path, summary, rows))
    assert result["control"]["load_compensation_verified_steps"] == 4
    assert all(row["controller"]["load_compensation"]["resisting_force_n"] == 0 for row in rows)


@pytest.mark.parametrize(
    "fault",
    [
        "source-tick",
        "source-time",
        "previous-force",
        "direction",
        "nonunit-direction",
        "gain",
        "enabled",
        "jacobian",
        "resisting",
        "correction",
        "combined",
        "clipped",
    ],
)
def test_half_load_compensation_rejects_forged_zero_load_evidence(tmp_path, fault):
    # The old PD-only reader accepts these commands, so rejection cannot be a
    # side effect of an already-different direct command or a broken file hash.
    summary, rows = _variant_case(tmp_path, zero_load=True)
    c = rows[1]["controller"]["load_compensation"]
    if fault == "source-tick":
        c["source_local_tick"] -= 1
    elif fault == "source-time":
        c["source_output_time"] -= 0.001
    elif fault == "previous-force":
        c["previous_force_on_tool_world_n"] = [1.0, 0.0, 0.0]
    elif fault == "direction":
        c["push_direction_world"] = [-0.6, -0.8, 0.0]
    elif fault == "nonunit-direction":
        c["push_direction_world"] = [1.2, 1.6, 0.0]
    elif fault == "gain":
        c["gain"] = 1.0
    elif fault == "enabled":
        c["enabled"] = False
    elif fault == "jacobian":
        c["jacobian_world"][0][1] += 0.01
    elif fault == "resisting":
        c["resisting_force_n"] = 1.0
    elif fault == "correction":
        c["correction_nm"][0] = 1.0
    elif fault == "combined":
        c["combined_unclipped_nm"][0] += 1.0
    else:
        c["combined_effort_nm"][0] += 1.0
    with pytest.raises(ValueError, match="compensation"):
        compare(_write(tmp_path, summary, rows))


def test_half_load_compensation_rejects_a_self_consistent_wrong_jacobian(tmp_path):
    summary, rows = _variant_case(tmp_path)
    c = rows[2]["controller"]["load_compensation"]
    wrong = np.asarray(c["jacobian_world"])[:, ::-1]
    c["jacobian_world"] = wrong.tolist()
    correction = wrong.T @ (0.5 * c["resisting_force_n"] * np.asarray(c["push_direction_world"]))
    c["correction_nm"] = correction.tolist()
    combined = np.asarray(rows[2]["controller"]["pd"]["unclipped_effort"]) + correction
    c["combined_unclipped_nm"] = combined.tolist()
    c["combined_effort_nm"] = np.clip(combined, -19.9, 19.9).tolist()
    _variant_transport(rows[2])
    with pytest.raises(ValueError, match="compensation.*Jacobian"):
        compare(_write(tmp_path, summary, rows))


@pytest.mark.parametrize("fault", ["force", "endpoint", "selection"])
def test_half_load_compensation_binds_first_force_to_last_selected_release_step(tmp_path, fault):
    from pathlib import Path

    summary, rows = _variant_case(tmp_path, zero_load=True)
    release_path = Path(summary["provenance"]["release_report_path"])
    release = json.loads(release_path.read_text())
    seed = release["trials"][0]
    if fault == "force":
        seed["area"]["total_force_on_pusher_world_n"] = [1.0, 0.0, 0.0]
    elif fault == "endpoint":
        seed["output_state"]["local_tick"] -= 1
    else:
        seed["selected"] = False
    sha = _write_json(release_path, release)
    summary["provenance"]["release_report_sha256"] = sha
    metadata_path = Path(summary["provenance"]["checkpoint_metadata_path"])
    metadata = json.loads(metadata_path.read_text())
    metadata["release_sha256"] = sha
    summary["provenance"]["checkpoint_metadata_sha256"] = _write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="compensation"):
        compare(_write(tmp_path, summary, rows))


def test_old_position_variant_remains_explicit_and_does_not_require_mujoco_compensation(tmp_path):
    summary, rows = _case(tmp_path)
    summary["control_variant"] = "position"
    result = compare(_write(tmp_path, summary, rows))
    assert result["control"]["variant"] == "position"
    assert result["control"]["load_compensation_verified_steps"] == 0


def test_half_load_compensation_uses_nonzero_current_rr_joint_posture(tmp_path):
    summary, rows = _variant_case(tmp_path, joint_position=(0.125, 0.25))
    result = compare(_write(tmp_path, summary, rows))
    assert result["control"]["load_compensation_verified_steps"] == 4
    jacobian = np.asarray(rows[0]["controller"]["load_compensation"]["jacobian_world"])
    # q=(0,0) has exactly zero first-column x derivative; this posture does not.
    assert abs(jacobian[0, 0]) > 0.1
