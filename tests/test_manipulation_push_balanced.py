"""Balanced proposal ranking and bounded native routing; no dynamic success claim."""

from dataclasses import asdict, replace

import pytest
from test_manipulation_push_objective import propose, query_fixture
from test_manipulation_push_planner import model, planner, proposal

POLICY = "joint-pose-balanced"


def weak_yaw_query():
    query = query_fixture()
    tiny = 1e-6
    original = query.candidates[0]
    near_center = replace(
        original,
        edge_fraction=(0.04 - tiny) / 0.08,
        point_world=(-0.03, -tiny, 0.0),
        contact_center_world=(-0.04, -tiny, 0.0),
        precontact_center_world=(-0.052, -tiny, 0.0),
        normal_wrench_com_world=(1.0, 0.0, 0.0, 0.0, 0.0, tiny),
    )
    return replace(query, candidates=(near_center, query.candidates[1]))


def options(**extra):
    return dict(
        target_xy=(0.1, 0.03),
        target_yaw=0.12,
        position_tolerance_m=0.001,
        yaw_tolerance_rad=0.01,
        **extra,
    )


def test_balanced_prioritizes_weakest_unmet_fraction_over_large_translation_score():
    query = weak_yaw_query()
    descent = propose(query, selection_policy="joint-pose-descent", **options())
    balanced = propose(query, selection_policy=POLICY, **options())
    assert descent[0].index == 0 and balanced[0].index == 1
    assert all(p.selection_eligible for p in balanced)
    assert balanced[0].predicted_improvement < descent[0].predicted_improvement
    for p in balanced:
        score = min(
            (p.position_error_before_m - p.position_error_after_m) / p.position_error_before_m,
            (abs(p.yaw_error_before_rad) - abs(p.yaw_error_after_rad))
            / abs(p.yaw_error_before_rad),
        )
        assert 0 < score <= 1
        assert p.sort_key == (0, -score, p.cost_after, p.index, p.distance_ordinal)
        assert p.selection_reason == "joint_pose_balanced"
    assert balanced[0].input_binding != descent[0].input_binding


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(
            target_xy=(0.0, 0.03),
            target_yaw=0.0,
            position_tolerance_m=0.001,
            yaw_tolerance_rad=0.08,
        ),
        dict(
            target_xy=(0.0, 0.0), target_yaw=0.12, position_tolerance_m=0.02, yaw_tolerance_rad=0.01
        ),
        dict(
            target_xy=(0.0, 0.03),
            target_yaw=0.0,
            position_tolerance_m=0.001,
            yaw_tolerance_rad=0.06,
        ),
        dict(
            target_xy=(0.03, 0.0),
            target_yaw=0.04,
            position_tolerance_m=0.001,
            yaw_tolerance_rad=0.03,
        ),
        dict(target_xy=(0.0, 0.0), target_yaw=0.0),
    ],
)
def test_balanced_keeps_exact_joint_admission_and_ignores_met_components_in_score(kwargs):
    old = {
        (p.index, p.distance_ordinal): p
        for p in propose(selection_policy="joint-pose-descent", **kwargs)
    }
    for p in propose(selection_policy=POLICY, **kwargs):
        previous = old[p.index, p.distance_ordinal]
        assert p.selection_eligible == previous.selection_eligible
        if not p.selection_eligible:
            assert p.selection_reason == previous.selection_reason
            continue
        fractions = []
        if p.position_error_before_m > kwargs.get("position_tolerance_m", 0.01):
            fractions.append(
                (p.position_error_before_m - p.position_error_after_m) / p.position_error_before_m
            )
        if abs(p.yaw_error_before_rad) > kwargs.get("yaw_tolerance_rad", 0.02):
            fractions.append(
                (abs(p.yaw_error_before_rad) - abs(p.yaw_error_after_rad))
                / abs(p.yaw_error_before_rad)
            )
        assert -p.sort_key[1] == min(fractions)


def test_balanced_has_complete_stable_length_order_and_distinct_old_bindings():
    before = [asdict(p) for p in propose()]
    records = propose(
        weak_yaw_query(), selection_policy=POLICY, **options(push_distances_m=(0.005, 0.01, 0.02))
    )
    assert len(records) == 6
    assert list(records) == sorted(records, key=lambda p: p.sort_key)
    assert [asdict(p) for p in propose(selection_policy="pose-cost")] == before
    assert len({p.input_binding for p in records}) == 1
    assert records == propose(
        weak_yaw_query(), selection_policy=POLICY, **options(push_distances_m=(0.005, 0.01, 0.02))
    )


def native_proposal(index, score, *, ordinal=0, pre=(0.4, 0, 1)):
    p = proposal(index, pre=pre)
    p.selection_policy = POLICY
    p.selection_eligible = True
    p.selection_reason = "joint_pose_balanced"
    p.distance_ordinal = ordinal
    p.sort_key = (0, -score, 1.0, index, ordinal)
    return p


def test_native_planner_orders_contact_groups_by_best_balance_and_keeps_real_path_gates():
    m = model()
    # Deliberately reversed input: the planner must group by the supplied rank,
    # not silently turn this into the historical input-order policy.
    result = planner(m).plan(m.qpos0, (native_proposal(0, 0.001), native_proposal(1, 0.1)))
    assert result.status == "sampled_reference_ready"
    assert result.proposal.index == 1
    assert result.ik_calls == result.path_calls == 3
    candidate = next(item for item in result.attempts if item["kind"] == "candidate")
    assert candidate["index"] == 1
    assert all(audit.sampled_clear for _, audit in candidate["paths"])


def test_balanced_infeasible_first_group_preserves_failure_then_tries_next_with_original_budget():
    m = model()
    first, second = native_proposal(0, 0.2, pre=(5.0, 0, 1)), native_proposal(1, 0.1)
    result = planner(m).plan(m.qpos0, (first, second))
    assert result.status == "sampled_reference_ready" and result.proposal.index == 1
    candidates = [a for a in result.attempts if a["kind"] == "candidate"]
    assert candidates[0]["reason"] == "precontact_ik"
    limited = planner(m, max_ik_per_segment=1).plan(m.qpos0, (first, second))
    assert limited.status == "ik_budget" and limited.proposal is None
    assert limited.ik_calls == 1


def test_balanced_cannot_mix_with_descent_or_legacy_proposals():
    m = model()
    for other in ("pose-cost", "joint-pose-descent"):
        a, b = native_proposal(0, 0.2), native_proposal(1, 0.1)
        b.selection_policy = other
        with pytest.raises(ValueError, match="selection policy"):
            planner(m).plan(m.qpos0, (a, b))


def test_balanced_candidate_cap_reports_untested_contact_groups():
    m = model()
    first, second = native_proposal(0, 0.2, pre=(5.0, 0, 1)), native_proposal(1, 0.1)
    result = planner(m, max_candidates=1).plan(m.qpos0, (first, second))
    assert result.status == "candidate_budget" and result.proposal is None
    assert result.attempts[-1]["unchecked_groups"] == (1,)
    assert result.ik_calls == 1 and result.path_calls == 0
