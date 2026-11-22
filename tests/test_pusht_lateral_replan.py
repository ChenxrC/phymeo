"""Completed-route continuation boundaries; no synthetic row is physical evidence."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.area_pressure.pusht_lateral_replan import (
    allow_constant_retry,
    reference_budget,
    remaining_budget,
)
from experiments.area_pressure.pusht_lateral_resume import validate_terminal_binding


def completed():
    state = dict(local_tick=2250, time=6.700000318232924, q=[0.1], v=[0.2])
    route = dict(start_tick=4450, end_tick=6700, completed_routes_before=0)
    controller = dict(tick=6700, status="continue", partial_route=None, completed_routes=1)
    report = dict(
        schema="phymeo-pusht-lateral-resume-v1",
        world=6,
        side_coverage="all-lateral",
        status="route_complete",
        control_variant="load-compensation-half",
        start_tick=5837,
        local_tick_origin=1387,
        source_reference_tick_origin=4450,
        goal_tick_origin=0,
        accepted_steps=863,
        recorded_steps=863,
        final_state=state,
        controller_final=controller,
        active_route=route,
        goal_history_final=dict(tick=2249),
        goal={"target_yaw": 0.2},
        provenance={"scene_asset_sha256": "asset"},
    )
    meta = dict(
        schema="phymeo-pusht-active-terminal-checkpoint-v1",
        world=6,
        state=deepcopy(state),
        q=state["q"],
        v=state["v"],
        time=state["time"],
        local_tick=2250,
        source_tick=6700,
        controller=deepcopy(controller),
        active_route=route,
        goal_history={"tick": 2249},
        target=report["goal"],
        asset_sha256="asset",
        control_variant="load-compensation-half",
        controller_at_physical_endpoint=True,
        physical_state_matches_report=True,
        accepted_steps=863,
        recorded_steps=863,
    )
    return report, meta


def test_completed_route_restore_is_explicit_and_does_not_revive_old_route():
    report, meta = completed()
    with pytest.raises(ValueError):
        validate_terminal_binding(report, meta)
    validate_terminal_binding(report, meta, allow_completed=True)
    assert meta["controller"]["partial_route"] is None
    assert meta["goal_history"]["tick"] == 2249


@pytest.mark.parametrize(
    "fault", ["end_tick", "completed_count", "goal_clock", "stopped", "unrecorded"]
)
def test_completed_restore_still_rejects_wrong_endpoint_or_history(fault):
    report, meta = completed()
    if fault == "end_tick":
        meta["source_tick"] -= 1
    elif fault == "completed_count":
        meta["controller"]["completed_routes"] = 0
    elif fault == "goal_clock":
        meta["goal_history"]["tick"] = 6699
    elif fault == "stopped":
        meta["controller"]["status"] = "no_progress"
    else:
        report["unrecorded_accepted_step"] = {"output_state": {}}
    with pytest.raises(ValueError):
        validate_terminal_binding(report, meta, allow_completed=True)


def test_new_phase_keeps_original_global_attempt_budget_including_failed_trials():
    assert remaining_budget(4000, 2251, None) == 1749
    assert remaining_budget(4000, 4000, None) == 0
    assert remaining_budget(4000, 2251, 1000) == 1000
    with pytest.raises(ValueError):
        remaining_budget(4000, 2251, 1750)
    with pytest.raises(ValueError):
        remaining_budget(4000, 4001, None)


def test_long_route_keeps_original_nodes_but_reports_only_budgeted_prefix():
    route = SimpleNamespace(nodes=((0.0, "approach", [0, 0]), (2.25, "push", [1, 1])))
    assert reference_budget(route, 1749) == dict(
        route_steps=2250, available_steps=1749, complete_route_within_budget=False
    )
    assert reference_budget(route, 2250)["complete_route_within_budget"] is True
    malformed = SimpleNamespace(nodes=((0.0, "approach", [0, 0]), (0.0005, "push", [1, 1])))
    with pytest.raises(ValueError):
        reference_budget(malformed, 1749)


def test_release_constant_rate_retry_only_at_first_same_entry_monotonic_rejection():
    decision = SimpleNamespace(violations=(SimpleNamespace(code="inward_motion"),))
    assert allow_constant_retry("smooth", 0, decision)
    assert not allow_constant_retry("smooth", 1, decision)
    assert not allow_constant_retry("constant_rate", 0, decision)
    assert not allow_constant_retry("smooth", 0, SimpleNamespace(violations=()))
    assert not allow_constant_retry(
        "smooth", 0, SimpleNamespace(violations=(SimpleNamespace(code="new_contact"),))
    )


def test_ordinary_candidate_survives_raw_withdrawal_request(monkeypatch):
    from experiments.area_pressure.pusht_loaded_release import _loaded_route, request_loaded_route
    from phymeo.control import manipulation_push_objective

    proposal = object()
    route = SimpleNamespace(status="ready", proposal=proposal)
    planner = SimpleNamespace(allow_loaded_withdrawal=False)

    def plan(q, proposals, *, previous_direction):
        assert planner.allow_loaded_withdrawal
        assert q == "saved joints"
        assert proposals == [proposal]
        assert previous_direction == (1, 0)
        return route

    planner.plan = plan
    driver = SimpleNamespace(
        query=object(),
        planner=planner,
        ref_position=(0, 0),
        ref_rotation=0,
        target=SimpleNamespace(position=(1, 2, 0), yaw=0.2),
        goal=SimpleNamespace(position_tolerance_m=0.015, yaw_tolerance_rad=0.12),
        previous_direction=(1, 0),
        pose=lambda q: ((0, 0, 0), 0, (0, 0, 0.1)),
    )
    monkeypatch.setattr(
        manipulation_push_objective, "propose_planar_pushes", lambda *a, **k: [proposal]
    )
    entry, snapshot = SimpleNamespace(driver=driver), SimpleNamespace(q="saved joints")
    assert request_loaded_route(entry, snapshot) == (route, [proposal])
    assert planner.allow_loaded_withdrawal is False
    with pytest.raises(ValueError, match="independently qualified release-only route"):
        _loaded_route(entry, snapshot)
    assert planner.allow_loaded_withdrawal is False
