from copy import deepcopy

import pytest


def boundary():
    state = {"local_tick": 9, "time": 0.009}
    controller = {"tick": 9, "status": "continue"}
    goal = {"target_yaw": 0.2}
    route = {
        "start_tick": 2,
        "end_tick": 12,
        "nodes": [[0.0], [0.010]],
        "sha256": "7" * 64,
        "completed_routes_before": 0,
    }
    meta = {
        "state": state,
        "world": 6,
        "source_tick": 9,
        "local_tick": 9,
        "controller": controller,
        "checkpoint_sha256": "a" * 64,
        "target": goal,
        "active_route": route,
        "asset_sha256": "b" * 64,
    }
    parent = {
        "world": 6,
        "key": "B",
        "control_variant": "load-compensation-half",
        "start_tick": 0,
        "recorded_steps": 9,
        "final_state": state,
        "controller_final": controller,
        "goal": goal,
        "active_route": route,
    }
    audited = {"actual": {"verified_steps": 9, "verified_final_state": state}, "recorded_steps": 9}
    proof = {
        "checkpoint_sha256": "a" * 64,
        "restored_local_tick": 9,
        "absolute_input_tick": 9,
        "time": 0.009,
    }
    summary = {
        "world": 6,
        "schema": "phymeo-pusht-active-route-v1",
        "start_tick": 9,
        "local_tick_origin": 9,
        "initial_state": state,
        "controller_initial": controller,
        "active_route": dict(route, remaining_steps=3),
        "goal": goal,
        "control_variant": "load-compensation-half",
        "requested_steps": 3,
        "provenance": {"scene_asset_sha256": "b" * 64},
        "checkpoint_restore": proof,
    }
    return deepcopy(summary), {"report": parent, "metadata": meta, "audit": audited}


@pytest.mark.parametrize(
    "bad", ["goal", "initial_state", "controller_initial", "route", "restore", "remaining", "audit"]
)
def test_pose_suffix_audit_requires_original_route_and_physical_boundary(bad):
    from experiments.area_pressure.pusht_resume_audit import bind_pose_boundary

    summary, source = boundary()
    assert bind_pose_boundary(summary, source)["verified_chain_steps"] == 9
    if bad == "route":
        summary["active_route"]["nodes"] = [[0.0], [0.011]]
    elif bad == "restore":
        summary["checkpoint_restore"]["checkpoint_sha256"] = "9" * 64
    elif bad == "remaining":
        summary["requested_steps"] -= 1
    elif bad == "audit":
        source["audit"]["actual"]["verified_steps"] -= 1
    else:
        summary[bad] = {}
    with pytest.raises(ValueError):
        bind_pose_boundary(summary, source)
