"""Resume audit source tests use synthetic readout records, never simulated evidence."""

from copy import deepcopy

import pytest

from experiments.area_pressure.pusht_resume_audit import bind_resume_source, verify_resume_tail


def source_case():
    before = dict(
        local_tick=2018,
        time=3.718,
        q=[0.125, 0.25, 0.5, 0.0, 0.125, 1.0, 0.0, 0.0, 0.0],
        v=[0.0] * 8,
        object_position_world_m=[0.5, 0.0, 0.125],
        object_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
    )
    after = dict(before, local_tick=2019, time=3.719)
    route = dict(
        start_tick=1950,
        end_tick=4200,
        duration_s=2.25,
        route_index=2,
        completed_routes_before=1,
        sha256="b" * 64,
        nodes=[[0.0, "push", [0.0, 0.0]], [2.25, "push", [1.0, 1.0]]],
    )
    seed = dict(
        original_input_tick=3718,
        original_output_tick=3719,
        input_state=before,
        output_state=after,
        area=dict(total_force_on_pusher_world_n=[-0.1, -1.0, 0.2]),
        goal_result=dict(history=dict(tick=3718, time_s=3.718)),
        feedback_accepted=dict(
            measurement=dict(
                world=6,
                input_tick=3718,
                output_tick=3719,
                time_s=3.718,
                phase="accepted-step-input",
                group_force_world=[0.1, 1.0, -0.2],
            )
        ),
    )
    parent = dict(
        schema="phymeo-pusht-active-route-v1",
        control_variant="load-compensation-half",
        status="partial_failure",
        world=6,
        start_tick=2014,
        local_tick_origin=314,
        accepted_steps=1705,
        recorded_steps=1705,
        final_state=after,
        active_route=route,
        goal=dict(target_xy=[0.55, 0.13], target_yaw=0.2),
        controller_final=dict(tick=3719, status="continue"),
        goal_history_final=seed["goal_result"]["history"],
        final_checkpoint=dict(sha256="a" * 64),
        journal=dict(records=1705, sha256="j" * 64),
        provenance=dict(scene_asset_sha256="s" * 64, source_sha256="m" * 64),
    )
    audit = dict(
        status="verified-saved-prefix",
        source_endpoint_bound=True,
        verified_steps=1705,
        accepted_steps=1705,
        verified_final_state=after,
        control=dict(variant="load-compensation-half", load_compensation_verified_steps=1705),
        final_checkpoint=dict(sha256="a" * 64, capsule_bytes_verified=True),
    )
    metadata = dict(
        schema="phymeo-pusht-active-terminal-checkpoint-v1",
        world=6,
        checkpoint_sha256="a" * 64,
        source_tick=3719,
        local_tick=2019,
        time=3.719,
        state=after,
        q=after["q"],
        v=after["v"],
        asset_sha256="s" * 64,
        controller_at_physical_endpoint=True,
        physical_state_matches_report=True,
        controller=parent["controller_final"],
        goal_history=parent["goal_history_final"],
        journal=parent["journal"],
        accepted_steps=1705,
        recorded_steps=1705,
        active_route=route,
        target=parent["goal"],
    )
    summary = dict(
        schema="phymeo-pusht-active-route-v1",
        control_variant="load-compensation-half",
        world=6,
        start_tick=3719,
        local_tick_origin=2019,
        initial_state=after,
        requested_steps=481,
        active_route=dict(route, remaining_steps=481, elapsed_at_entry_s=1.769),
        goal=parent["goal"],
        provenance=parent["provenance"],
        checkpoint_restore=dict(
            checkpoint_sha256="a" * 64,
            restored_local_tick=2019,
            absolute_input_tick=3719,
            time=3.719,
        ),
    )
    return tuple(deepcopy(x) for x in (summary, parent, audit, metadata, seed))


def test_resume_seed_is_the_preceding_committed_force_and_exact_terminal_endpoint():
    result = bind_resume_source(*source_case())
    assert result["source_tick"] == 3719 and result["parent_verified_steps"] == 1705
    assert result["previous_input_tick"] == 3718
    assert result["previous_force_on_tool_world_n"] == [-0.1, -1.0, 0.2]
    assert result["checkpoint_sha256"] == "a" * 64


@pytest.mark.parametrize(
    "bad",
    ("state", "force_side", "future_force", "unaudited_tail", "route", "capsule", "goal_history"),
)
def test_resume_rejects_wrong_endpoint_force_phase_or_unverified_terminal_source(bad):
    summary, parent, audit, metadata, seed = source_case()
    if bad == "state":
        summary["initial_state"]["q"][0] += 0.001
    elif bad == "force_side":
        seed["feedback_accepted"]["measurement"]["group_force_world"][0] *= -1
    elif bad == "future_force":
        seed["original_input_tick"] = 3719
    elif bad == "unaudited_tail":
        parent["accepted_steps"] += 1
    elif bad == "route":
        summary["active_route"]["sha256"] = "c" * 64
    elif bad == "capsule":
        metadata["checkpoint_sha256"] = "c" * 64
    else:
        metadata["goal_history"]["tick"] -= 1
    with pytest.raises(ValueError):
        bind_resume_source(summary, parent, audit, metadata, seed)


def test_zero_step_physics_failure_and_post_accept_callback_tail_keep_different_evidence_limits():
    summary, *_ = source_case()
    summary.update(
        status="partial_failure",
        accepted_steps=0,
        recorded_steps=0,
        final_state=summary["initial_state"],
        failure_stage="scene_step",
    )
    assert verify_resume_tail(summary, summary["initial_state"]) is None
    before = deepcopy(summary["initial_state"])
    after = dict(deepcopy(before), local_tick=2020, time=3.720)
    summary.update(
        accepted_steps=1,
        final_state=after,
        failure_stage="accepted_controller_and_goal",
        unrecorded_accepted_step=dict(
            original_input_tick=3719,
            original_output_tick=3720,
            input_state=deepcopy(before),
            output_state=after,
        ),
    )
    tail = verify_resume_tail(summary, before)
    assert tail["input_tick"] == 3719 and tail["output_tick"] == 3720
    assert tail["transition_verified"] is False
    assert tail["control_force_goal_evidence_verified"] is False
    summary["unrecorded_accepted_step"]["input_state"]["time"] -= 0.001
    with pytest.raises(ValueError):
        verify_resume_tail(summary, before)


def retry_case():
    summary, parent, *_ = source_case()
    command = dict(
        tick=3719,
        output_tick=3720,
        ctrl=[0.25, -0.125],
        qfrc_applied=[0.0] * 8,
        xfrc_applied=[[0.0] * 6 for _ in range(5)],
    )
    parent.update(
        failure_stage="scene_step",
        failed_original_input_tick=3719,
        failed_candidate=dict(submitted_control=deepcopy(command), restored=True),
    )
    summary["first_command_binding"] = dict(matches_parent_rejected_command=True)
    return summary, parent, command


def test_reader_itself_binds_first_retry_three_channels_and_tick():
    from experiments.area_pressure.pusht_resume_audit import verify_first_retry_command

    result = verify_first_retry_command(*retry_case())
    assert result["verified"] is True and result["input_tick"] == 3719
    assert result["channels"] == ["ctrl", "qfrc_applied", "xfrc_applied"]


@pytest.mark.parametrize(
    "bad", ("ctrl", "qfrc_applied", "xfrc_applied", "tick", "output_tick", "flag", "signed_zero")
)
def test_true_producer_match_flag_cannot_hide_forged_first_retry(bad):
    from experiments.area_pressure.pusht_resume_audit import verify_first_retry_command

    summary, parent, command = retry_case()
    if bad == "ctrl":
        command["ctrl"][0] += 0.25
    elif bad == "qfrc_applied":
        command["qfrc_applied"][0] = 1.0
    elif bad == "xfrc_applied":
        command["xfrc_applied"][1][2] = 1.0
    elif bad in ("tick", "output_tick"):
        command[bad] += 1
    elif bad == "flag":
        summary["first_command_binding"]["matches_parent_rejected_command"] = False
    else:
        command["qfrc_applied"][0] = -0.0
    with pytest.raises(ValueError):
        verify_first_retry_command(summary, parent, command)


def test_position_resume_keeps_real_seed_force_without_claiming_feedback_compensation():
    summary, parent, audit, metadata, seed = source_case()
    summary["control_variant"] = parent["control_variant"] = "position"
    audit["control"] = dict(variant="position")
    result = bind_resume_source(summary, parent, audit, metadata, seed)
    assert result["control_variant"] == "position"
    assert result["previous_force_on_tool_world_n"] == [-0.1, -1.0, 0.2]
