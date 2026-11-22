"""Real objective/material proposals with a budgeted planner spy, not physics evidence."""

from types import SimpleNamespace

import numpy as np
from test_manipulation_push_objective import propose

from experiments.area_pressure.pusht_lookahead_candidates import evaluate_candidates
from phymeo.control.manipulation_push_material import NativePushMaterialFaces
from phymeo.control.manipulation_push_planner import NativePushRoute


class PlannerSpy:
    """Expose charged singleton calls and distinct native-style rejection details."""

    def __init__(
        self, model, *, ik_calls=0, max_ik=512, per_segment=64, path_calls=0, max_paths=128
    ):
        self.model = model
        self.ik_calls, self.path_calls = ik_calls, path_calls
        self.max_ik, self.max_ik_per_segment = max_ik, per_segment
        self.max_paths, self.max_candidates = max_paths, 12
        self.allow_loaded_withdrawal = False
        self.calls = []

    def plan(self, q, proposals, *, previous_direction, material_faces):
        assert len(proposals) == 1
        proposal = proposals[0]
        material_faces.validate_source(q)
        self.calls.append(
            dict(
                ordinal=proposal.distance_ordinal,
                ik_before=self.ik_calls,
                max_ik=self.max_ik,
                previous_direction=previous_direction,
            )
        )
        available = self.max_ik - self.ik_calls
        self.ik_calls += min(2, available)
        if available < 2:
            status, reason = "ik_budget", "ik_budget"
        else:
            self.path_calls += 1
            status = "no_qualified_route"
            reason = {0: "precontact_ik", 1: "approach_path", 2: "push_path"}[
                proposal.distance_ordinal
            ]
        diagnostic = dict(
            kind="candidate",
            index=proposal.index,
            distance_ordinal=proposal.distance_ordinal,
            reason=reason,
            paths=[("approach", dict(sampled_clear=False, reason="sampled_collision"))],
        )
        return NativePushRoute(
            status, None, q.copy(), (), (diagnostic,), self.ik_calls, self.path_calls, (0.0, 0.0)
        )


def inputs(**budgets):
    # Original producer outputs: no fabricated eligibility or cost/binding changes.
    proposals = tuple(
        p
        for p in propose(
            target_xy=(1.0, 0.0),
            target_yaw=0.0,
            push_distances_m=(0.005, 0.01, 0.02),
            selection_policy="pose-cost",
        )
        if p.index == 0
    )
    assert len(proposals) == 3 and all(p.predicted_improvement > 0 for p in proposals)
    q = np.zeros(7, dtype=np.float64)
    model = SimpleNamespace(nq=len(q))
    planner = PlannerSpy(model, **budgets)
    driver = SimpleNamespace(model=model, planner=planner)
    x = proposals[0].point_world[0]
    # Two finite triangles with the actual candidate's -X outward orientation.
    triangles = np.array(
        [[[x, -1, -1], [x, -1, 1], [x, 1, 1]], [[x, -1, -1], [x, 1, 1], [x, 1, -1]]], dtype=float
    )
    material = NativePushMaterialFaces(q, "a" * 64, ("left-a", "left-b"), triangles)
    return driver, q, proposals, material


def test_every_original_stroke_keeps_its_own_path_rejection_and_order():
    driver, q, proposals, material = inputs()
    direction = (1.0, 0.0, 0.0)
    result = evaluate_candidates(
        driver, q, proposals, material_faces=material, previous_direction=direction
    )
    assert result["complete"] and result["evaluated_count"] == 3 and result["reachable_count"] == 0
    assert [r["distance_ordinal"] for r in result["candidates"]] == [
        p.distance_ordinal for p in proposals
    ]
    expected = {0: "precontact_ik", 1: "approach_path", 2: "push_path"}
    for row in result["candidates"]:
        assert row["evaluated"] and not row["reachable"] and row["status"] == "no_qualified_route"
        assert row["route"].attempts[0]["reason"] == expected[row["distance_ordinal"]]
        assert row["route"].attempts[0]["paths"][0][1]["reason"] == "sampled_collision"
    assert all(call["previous_direction"] == direction for call in driver.planner.calls)
    assert driver.planner.ik_calls == 6 and driver.planner.path_calls == 3


def test_multiple_singletons_share_one_original_per_segment_ik_allowance():
    driver, q, proposals, material = inputs(ik_calls=10, per_segment=3)
    result = evaluate_candidates(driver, q, proposals, material_faces=material)
    assert [c["ik_before"] for c in driver.planner.calls] == [10, 12]
    assert [c["max_ik"] for c in driver.planner.calls] == [13, 13]
    assert driver.planner.ik_calls == 13 and driver.planner.max_ik == 512
    assert result["candidates"][1]["status"] == "ik_budget"
    last = result["candidates"][2]
    assert last["status"] == "not_evaluated" and last["reason"] == "ik_budget"
    assert last["route"] is None and not last["evaluated"] and not last["reachable"]
    assert not result["complete"]


def test_tighter_outer_search_envelope_is_never_widened_or_rolled_back():
    driver, q, proposals, material = inputs(ik_calls=10, max_ik=12, per_segment=64)
    result = evaluate_candidates(driver, q, proposals, material_faces=material)
    assert len(driver.planner.calls) == 1 and driver.planner.calls[0]["max_ik"] == 12
    assert driver.planner.ik_calls == driver.planner.max_ik == 12
    assert result["shared_ik_limit"] == 12 and not result["complete"]
    assert all(
        r["status"] == "not_evaluated" and r["reason"] == "ik_budget"
        for r in result["candidates"][1:]
    )


def test_exhausted_native_path_budget_is_explicitly_untested_not_unreachable():
    driver, q, proposals, material = inputs(ik_calls=7, path_calls=128, max_paths=128)
    result = evaluate_candidates(driver, q, proposals, material_faces=material)
    assert (
        driver.planner.calls == []
        and driver.planner.ik_calls == 7
        and driver.planner.path_calls == 128
    )
    assert result["evaluated_count"] == result["reachable_count"] == 0 and not result["complete"]
    assert all(
        r["status"] == "not_evaluated"
        and r["reason"] == "path_budget"
        and not r["evaluated"]
        and r["route"] is None
        for r in result["candidates"]
    )
