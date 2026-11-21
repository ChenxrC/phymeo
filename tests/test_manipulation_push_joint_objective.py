"""Joint pose-descent policy: analytic proposals, never a physical success claim."""

from dataclasses import asdict, replace
from math import cos, pi, sin

import numpy as np
import pytest
from test_manipulation_push_objective import HALF, propose, query_fixture
from test_manipulation_push_planner import model, planner

POLICY = "joint-pose-descent"


def selected(**kwargs):
    return propose(selection_policy=POLICY, **kwargs)


def test_default_and_explicit_legacy_policy_keep_scores_order_and_binding():
    default = propose()
    explicit = propose(selection_policy="pose-cost")
    assert default == explicit
    assert [asdict(p) for p in default] == [asdict(p) for p in explicit]
    changed = selected()
    assert default[0].input_binding != changed[0].input_binding
    assert len(default) == len(changed) == 2
    assert all(p.selection_policy == POLICY for p in changed)


def test_translation_gain_cannot_buy_wrong_signed_yaw_even_if_total_cost_drops():
    records = selected(
        target_xy=(0.03, 0.0),
        target_yaw=0.04,
        position_tolerance_m=0.001,
        yaw_tolerance_rad=0.03,
    )
    wrong = next(p for p in records if p.index == 0)
    assert wrong.predicted_improvement > 0
    assert wrong.nominal_yaw_moment < 0 < wrong.yaw_error_before_rad
    assert not wrong.selection_eligible
    assert wrong.selection_reason == "yaw_wrong_signed_moment"
    assert all(not p.selection_eligible for p in records)


def test_positive_moment_contact_strictly_improves_both_unmet_components():
    records = selected(
        target_xy=(0.02, 0.03),
        target_yaw=0.12,
        position_tolerance_m=0.001,
        yaw_tolerance_rad=0.01,
    )
    good = records[0]
    assert good.index == 1 and good.selection_eligible
    assert good.selection_reason == "joint_pose_descent"
    assert good.position_error_before_m == pytest.approx(np.hypot(0.02, 0.03))
    assert good.position_error_after_m == pytest.approx(np.hypot(0.02, 0.03 - 1 / 116))
    assert good.yaw_error_before_rad == pytest.approx(0.12)
    assert good.yaw_error_after_rad == pytest.approx(0.12 - 2 / 29)
    assert good.position_error_after_m < good.position_error_before_m
    assert abs(good.yaw_error_after_rad) < abs(good.yaw_error_before_rad)
    assert good.nominal_yaw_moment * good.yaw_error_before_rad > 0
    assert good.status == "PROPOSAL_ONLY"


@pytest.mark.parametrize(
    "tolerance,eligible,reason",
    (
        (0.08, True, "joint_pose_descent"),
        (0.06, False, "yaw_tolerance_lost"),
    ),
)
def test_already_qualified_yaw_must_stay_in_original_tolerance(tolerance, eligible, reason):
    record = next(
        p
        for p in selected(
            target_xy=(0.0, 0.03),
            target_yaw=0.0,
            position_tolerance_m=0.001,
            yaw_tolerance_rad=tolerance,
        )
        if p.index == 1
    )
    assert record.yaw_error_before_rad == 0
    assert abs(record.yaw_error_after_rad) > 0
    assert record.selection_eligible is eligible and record.selection_reason == reason


@pytest.mark.parametrize(
    "tolerance,eligible,reason",
    (
        (0.02, True, "joint_pose_descent"),
        (0.005, False, "position_tolerance_lost"),
    ),
)
def test_already_qualified_position_cannot_leave_its_original_tolerance(
    tolerance, eligible, reason
):
    record = next(
        p
        for p in selected(
            target_xy=(0.0, 0.0),
            target_yaw=0.12,
            position_tolerance_m=tolerance,
            yaw_tolerance_rad=0.01,
        )
        if p.index == 1
    )
    assert record.position_error_before_m == 0
    assert record.selection_eligible is eligible and record.selection_reason == reason


def test_correct_moment_sign_is_insufficient_when_the_finite_push_overshoots_yaw():
    records = selected(
        target_xy=(0.0, 0.03),
        target_yaw=0.02,
        position_tolerance_m=0.001,
        yaw_tolerance_rad=0.001,
        push_distances_m=(0.005, 0.01),
    )
    short, long = (
        next(p for p in records if p.index == 1 and p.distance_m == length)
        for length in (0.005, 0.01)
    )
    assert short.selection_eligible
    assert long.nominal_yaw_moment * long.yaw_error_before_rad > 0
    assert not long.selection_eligible and long.selection_reason == "yaw_not_improved"


def test_zero_moment_does_not_claim_progress_on_unmet_yaw():
    query = query_fixture()
    center = replace(
        query.candidates[0],
        edge_fraction=0.5,
        point_world=(-0.03, 0.0, 0.0),
        normal_wrench_com_world=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    record = propose(
        replace(query, candidates=(center,)),
        selection_policy=POLICY,
        target_xy=(0.03, 0.0),
        target_yaw=-0.1,
        position_tolerance_m=0.001,
        yaw_tolerance_rad=0.01,
    )[0]
    assert record.nominal_yaw_moment == 0
    assert not record.selection_eligible
    assert record.selection_reason == "yaw_wrong_signed_moment"


@pytest.mark.parametrize("angle", (pi / 2, pi, -0.73))
def test_policy_is_covariant_between_actual_object_orientations(angle):
    rotation = np.array(((cos(angle), -sin(angle), 0), (sin(angle), cos(angle), 0), (0, 0, 1)))
    shift = np.array((2.0, -3.0))
    target = shift + rotation[:2, :2] @ np.array((0.02, 0.03))
    options = dict(position_tolerance_m=0.001, yaw_tolerance_rad=0.01)
    base = selected(target_xy=(0.02, 0.03), target_yaw=0.12, **options)
    moved = selected(
        current_position=(*shift, 0.0),
        current_rotation=rotation,
        target_xy=target,
        target_yaw=angle + 0.12,
        **options,
    )
    assert [(p.index, p.selection_eligible, p.selection_reason) for p in base] == [
        (p.index, p.selection_eligible, p.selection_reason) for p in moved
    ]
    for left, right in zip(base, moved, strict=True):
        assert right.nominal_yaw_moment == pytest.approx(left.nominal_yaw_moment, abs=1e-14)
        assert right.yaw_error_before_rad == pytest.approx(left.yaw_error_before_rad, abs=1e-14)
        assert right.yaw_error_after_rad == pytest.approx(left.yaw_error_after_rad, abs=1e-14)
        assert right.position_error_after_m == pytest.approx(left.position_error_after_m, abs=1e-14)
        np.testing.assert_allclose(
            right.direction_world, rotation @ left.direction_world, atol=1e-14
        )


def test_target_wrap_uses_the_same_world_yaw_zero_as_current_pose():
    record = next(
        p
        for p in selected(
            current_rotation=HALF,
            target_xy=(0.0, -0.03),
            target_yaw=-pi + 0.12,
            position_tolerance_m=0.001,
            yaw_tolerance_rad=0.01,
        )
        if p.index == 1
    )
    assert record.selection_eligible
    assert record.yaw_error_before_rad == pytest.approx(0.12)
    assert record.yaw_error_after_rad == pytest.approx(0.12 - 2 / 29)


def test_goal_components_already_qualified_do_not_request_another_push():
    records = selected(target_xy=(0, 0), target_yaw=0)
    assert all(not p.selection_eligible for p in records)
    assert all(p.selection_reason == "already_within_pose_tolerances" for p in records)


def test_reference_unusable_candidate_keeps_its_rejection_and_all_length_records():
    query = query_fixture()
    blocked = replace(query.candidates[1], usable=False, reason="entry-occluded")
    records = propose(
        replace(query, candidates=(query.candidates[0], blocked)),
        selection_policy=POLICY,
        target_xy=(0.02, 0.03),
        target_yaw=0.12,
        position_tolerance_m=0.001,
        yaw_tolerance_rad=0.01,
        push_distances_m=(0.005, 0.01, 0.02),
    )
    assert len(records) == 6
    assert all(
        not p.selection_eligible and p.selection_reason == "reference_unusable"
        for p in records
        if p.index == 1
    )


@pytest.mark.parametrize("policy", (None, True, "joint", "", 1))
def test_unknown_or_non_string_policy_rejects(policy):
    with pytest.raises(ValueError, match="selection policy"):
        propose(selection_policy=policy)


def test_no_joint_candidate_is_an_explicit_refusal_without_spending_ik_or_path_budget(monkeypatch):
    import phymeo.control.manipulation_push_planner as module

    records = selected(
        target_xy=(0.03, 0.0), target_yaw=0.04, position_tolerance_m=0.001, yaw_tolerance_rad=0.03
    )
    assert all(not p.selection_eligible for p in records)

    def forbidden(*args, **kwargs):
        raise AssertionError("ineligible contact must not consume path or IK budget")

    monkeypatch.setattr(module, "solve_native_point_ik", forbidden)
    monkeypatch.setattr(module, "audit_native_joint_path", forbidden)
    m = model()
    owner = planner(m)
    result = owner.plan(m.qpos0, records)
    assert result.status == "no_jointly_improving_proposal"
    assert result.proposal is None and result.nodes == ()
    assert result.ik_calls == result.path_calls == 0
    assert result.attempts[0]["kind"] == "selection"
    assert result.attempts[0]["reason"] == "no_jointly_improving_proposal"
    assert len(result.attempts[0]["rejected_proposals"]) == len(records)
