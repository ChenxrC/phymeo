"""Two-push planning predictions, not a synthetic dynamics rollout."""

from math import cos, sin
from types import SimpleNamespace

import numpy as np
import pytest


def endpoint_case():
    import mujoco as mj
    from test_manipulation_push_planner import model

    m = model()
    q = m.qpos0.copy()
    q[5:9] = (cos(0.02), sin(0.02), 0.0, 0.0)
    body = int(m.body("object").id)

    def pose(values):
        data = mj.MjData(m)
        data.qpos[:] = values
        mj.mj_kinematics(m, data)
        return (
            data.xpos[body].copy(),
            data.xmat[body].reshape(3, 3).copy(),
            data.geom_xpos[m.geom("pusher").id].copy(),
        )

    driver = SimpleNamespace(model=m, qa=[0, 1], object_body=body, pose=pose)
    proposal = SimpleNamespace(
        status="PROPOSAL_ONLY",
        selection_policy="pose-cost",
        index=15,
        distance_ordinal=0,
        predicted_translation_xy=(0.01, 0.005),
        predicted_position_xy=(0.51, 0.005),
        predicted_yaw_change=0.05,
        predicted_yaw=0.05,
    )
    route = SimpleNamespace(
        status="sampled_reference_ready",
        source_q=q.copy(),
        proposal=proposal,
        initial_reference_projection=(0.0, 0.0),
        nodes=((0.0, "approach", q[:2].copy()), (1.5, "push", np.array([0.44, 0.02]))),
    )
    return driver, q, route


def test_predicted_endpoint_uses_real_first_robot_endpoint_and_preserves_free_height_and_tilt():
    from experiments.area_pressure.pusht_lookahead_pairs import predicted_endpoint

    driver, q, route = endpoint_case()
    before = q.tobytes()
    original = driver.model.qpos0.tobytes()
    result = predicted_endpoint(driver, q, route)
    predicted = np.asarray(result["q"])
    np.testing.assert_array_equal(predicted[:2], route.nodes[-1][2])
    np.testing.assert_allclose(predicted[2:4], (0.51, 0.005), rtol=0, atol=1e-14)
    assert predicted[4] == q[4]
    rotation = np.array(
        ((cos(0.05), -sin(0.05), 0.0), (sin(0.05), cos(0.05), 0.0), (0.0, 0.0, 1.0))
    )
    np.testing.assert_allclose(
        driver.pose(predicted)[1], rotation @ driver.pose(q)[1], rtol=0, atol=2e-14
    )
    assert result["kind"] == "synthetic-predicted-after-first"
    assert result["velocity_prediction"] is None
    assert result["reference_steps"] == 1500
    assert q.tobytes() == before and driver.model.qpos0.tobytes() == original


@pytest.mark.parametrize(
    "fault", ["projected", "source", "start", "foreign-prediction", "not-qualified"]
)
def test_predicted_endpoint_rejects_projected_or_disconnected_first_reference(fault):
    from experiments.area_pressure.pusht_lookahead_pairs import predicted_endpoint

    driver, q, route = endpoint_case()
    if fault == "projected":
        route.initial_reference_projection = (0.001, 0.0)
    elif fault == "source":
        route.source_q[0] += 0.001
    elif fault == "start":
        route.nodes[0][2][0] += 0.001
    elif fault == "foreign-prediction":
        route.proposal.predicted_position_xy = (0.7, 0.005)
    else:
        route.status = "withdraw_rejected"
    with pytest.raises(ValueError):
        predicted_endpoint(driver, q, route)


def predictions(second_yaw=0.05):
    first = SimpleNamespace(
        predicted_translation_xy=(0.01, 0.0),
        predicted_position_xy=(0.01, 0.0),
        predicted_yaw_change=-0.03,
        predicted_yaw=-0.03,
    )
    second = SimpleNamespace(
        predicted_translation_xy=(0.02, 0.0),
        predicted_position_xy=(0.03, 0.0),
        predicted_yaw_change=second_yaw + 0.03,
        predicted_yaw=second_yaw,
    )
    return first, second


def metrics(first, second, **extra):
    from experiments.area_pressure.pusht_lookahead_pairs import pair_metrics

    return pair_metrics(
        first,
        second,
        initial_position_xy=(0.0, 0.0),
        initial_yaw=0.0,
        target_xy=(1.0, 0.0),
        target_yaw=0.1,
        position_tolerance_m=0.015,
        yaw_tolerance_rad=0.12,
        **extra,
    )


def test_first_yaw_can_worsen_only_when_combination_strictly_improves_both_errors():
    result = metrics(
        *predictions(),
        baseline_prediction={"predicted_position_xy": (0.02, 0.0), "predicted_yaw": 0.03},
    )
    assert result["first_yaw_worsens"] is True
    assert result["net_both_improve"] is True
    assert result["position_improvement_m"] == pytest.approx(0.03)
    assert result["absolute_yaw_improvement_rad"] == pytest.approx(0.05)
    assert result["baseline_comparison"]["position_advantage_m"] == pytest.approx(0.01)
    assert result["baseline_comparison"]["absolute_yaw_advantage_rad"] == pytest.approx(0.02)
    assert result["prediction_only"] is True


def test_translation_gain_cannot_hide_combined_yaw_regression():
    result = metrics(*predictions(-0.02))
    assert result["position_improvement_m"] > 0
    assert result["absolute_yaw_improvement_rad"] < 0
    assert result["net_both_improve"] is False


def test_second_prediction_must_continue_first_predicted_object_pose():
    first, second = predictions()
    second.predicted_position_xy = (0.9, 0.0)
    with pytest.raises(ValueError):
        metrics(first, second)


def test_transfer_rejection_retains_original_withdraw_diagnostic_without_claiming_approach_pass():
    from experiments.area_pressure.pusht_lookahead_pairs import transfer_summary

    route = SimpleNamespace(
        status="withdraw_rejected",
        nodes=(),
        attempts=(
            dict(
                kind="withdraw",
                reason="withdraw_path",
                paths=(
                    (
                        "withdraw",
                        SimpleNamespace(
                            sampled_clear=False, reason="sampled_collision", samples=()
                        ),
                    ),
                ),
            ),
        ),
    )
    result = transfer_summary(route)
    assert result["withdrawal_checked"] is True
    assert result["withdrawal_sampled_clear"] is False
    assert result["approach_sampled_clear"] is False
    assert result["reason"] == "withdraw_path"
    assert result["geometry_feasible"] is False


def search_stubs(monkeypatch, *, first_charge=2, fail=False, second_yaw=0.05):
    from experiments.area_pressure import pusht_lookahead_candidates as candidates
    from experiments.area_pressure import pusht_lookahead_pairs as pairs

    planner = SimpleNamespace(
        ik_calls=100,
        path_calls=10,
        max_ik=512,
        max_paths=128,
        max_ik_per_segment=64,
        max_candidates=12,
        allow_loaded_withdrawal=False,
    )
    driver = SimpleNamespace(
        planner=planner,
        model=SimpleNamespace(nq=9),
        qa=[0, 1],
        target=SimpleNamespace(position=(1.0, 0.0, 0.0), yaw=0.1),
        goal=SimpleNamespace(position_tolerance_m=0.015, yaw_tolerance_rad=0.12),
        pose=lambda q: (np.asarray(q[2:5]), np.eye(3), np.zeros(3)),
    )
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0])
    first, second = predictions(second_yaw)
    first.index, first.distance_ordinal, first.direction_world = 15, 0, (1.0, 0.0, 0.0)
    second.index, second.distance_ordinal, second.direction_world = 7, 0, (0.0, 1.0, 0.0)
    first.input_binding, second.input_binding = "a" * 64, "b" * 64
    another = SimpleNamespace(**vars(first))
    another.index = 16
    calls = []

    def propose(_driver, values, **kwargs):
        return (first, another) if np.array_equal(values, q) else (second,)

    def evaluate(
        _driver, values, proposals, *, material_faces, indices=None, previous_direction=None
    ):
        calls.append((indices, previous_direction, planner.max_ik, values.copy()))
        if fail:
            planner.ik_calls += 1
            raise RuntimeError("registered evaluator fault")
        planner.ik_calls = min(
            planner.max_ik, planner.ik_calls + (first_charge if len(calls) == 1 else 1)
        )
        chosen = tuple(p for p in proposals if indices is None or p.index in indices)
        records = []
        for p in chosen:
            route = SimpleNamespace(
                status="sampled_reference_ready",
                proposal=p,
                source_q=values.copy(),
                initial_reference_projection=(0.0, 0.0),
                nodes=((0.0, "approach", [0.0, 0.0]), (1.5, "push", [0.0, 0.0])),
                attempts=(),
            )
            records.append(
                dict(
                    index=p.index,
                    distance_ordinal=0,
                    proposal=p,
                    route=route,
                    evaluated=True,
                    reachable=True,
                    status="sampled_reference_ready",
                )
            )
        return dict(candidates=records, complete=True, material_admission={"binding": "unit"})

    monkeypatch.setattr(candidates, "propose_candidates", propose)
    monkeypatch.setattr(candidates, "evaluate_candidates", evaluate)
    monkeypatch.setattr(candidates, "compact_evaluation", lambda value: pairs._plain(value))
    monkeypatch.setattr(
        pairs,
        "predicted_endpoint",
        lambda _d, _q, route: dict(
            kind="synthetic-predicted-after-first",
            q=[0.0, 0.0, 0.01, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0],
            reference_steps=1500,
            velocity_prediction=None,
        ),
    )
    monkeypatch.setattr(pairs, "transfer_summary", lambda route: dict(geometry_feasible=True))
    return pairs, driver, q, calls


def test_search_evaluates_all_requested_first_atoms_once_before_expansion_and_counts_full_duration(
    monkeypatch,
):
    pairs, driver, q, calls = search_stubs(monkeypatch)
    result = pairs.search_pairs(
        driver, q, lambda values: object(), remaining_steps=1499, first_indices=(15, 16)
    )
    assert calls[0][0] == (15, 16)
    assert len([call for call in calls if call[0] == (15, 16)]) == 1
    assert all(call[1] == (1.0, 0.0, 0.0) for call in calls[1:])
    assert all(call[2] == 164 for call in calls)
    assert driver.planner.max_ik == 512
    assert result["budget"]["new_ik_calls"] == 4
    assert result["selected_pair"] is None
    assert all(
        not record["within_remaining_steps"] and record["reference_steps"] == 3000
        for record in result["pairs"]
    )


def test_shared_64_ik_limit_does_not_refill_for_second_layer(monkeypatch):
    pairs, driver, q, calls = search_stubs(monkeypatch, first_charge=64)
    result = pairs.search_pairs(
        driver, q, lambda values: object(), remaining_steps=4000, first_indices=(15, 16)
    )
    assert len(calls) == 1
    assert result["budget"]["new_ik_calls"] == 64
    assert result["search_complete"] is False
    assert result["selected_pair"] is None
    assert all(branch["status"] == "not_evaluated" for branch in result["second_layers"])
    assert driver.planner.max_ik == 512


def test_search_restores_only_budget_limit_not_consumed_counters_on_error(monkeypatch):
    pairs, driver, q, _ = search_stubs(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="registered evaluator fault"):
        pairs.search_pairs(
            driver, q, lambda values: object(), remaining_steps=4000, first_indices=(15, 16)
        )
    assert driver.planner.max_ik == 512 and driver.planner.ik_calls == 101


def test_second_layer_net_ineligible_predictions_do_not_consume_native_ik(monkeypatch):
    pairs, driver, q, calls = search_stubs(monkeypatch, second_yaw=-0.02)
    result = pairs.search_pairs(
        driver, q, lambda values: object(), remaining_steps=4000, first_indices=(15, 16)
    )
    assert len(calls) == 1 and result["budget"]["new_ik_calls"] == 2
    assert result["search_complete"] is True
    assert result["pairs"] == []
    for branch in result["second_layers"]:
        assert branch["status"] == "no_net_jointly_improving_prediction"
        assert not branch["prediction_filter"][0]["requested_native_evaluation"]
        assert "combined_absolute_yaw_not_improved" in branch["prediction_filter"][0]["reason"]


def test_existing_global_ik_allowance_is_not_expanded_to_64_new_calls(monkeypatch):
    pairs, driver, q, calls = search_stubs(monkeypatch)
    driver.planner.max_ik = 101
    result = pairs.search_pairs(
        driver, q, lambda values: object(), remaining_steps=4000, first_indices=(15, 16)
    )
    assert len(calls) == 1 and calls[0][2] == 101
    assert result["budget"]["new_ik_calls"] == 1
    assert driver.planner.max_ik == 101 and driver.planner.ik_calls == 101
    assert not result["search_complete"]


def test_real_native_withdrawal_rejects_embedded_predicted_start_without_physics(monkeypatch):
    import mujoco as mj
    from test_manipulation_push_planner import model, planner, proposal

    from experiments.area_pressure.pusht_lookahead_pairs import transfer_summary

    def forbidden(*args, **kwargs):
        raise AssertionError("planning must not step physics")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mj, name, forbidden)
    m = model()
    predicted = m.qpos0.copy()
    predicted[0] = 0.5  # Sphere center lies inside the box in this planning-only case.
    native = planner(m)
    route = native.plan(predicted, (proposal(),), previous_direction=(1.0, 0.0, 0.0))
    assert route.status == "withdraw_rejected"
    evidence = transfer_summary(route)
    assert evidence["reason"] == "withdraw_path"
    assert evidence["withdrawal_checked"] and not evidence["withdrawal_sampled_clear"]
    assert not evidence["approach_sampled_clear"] and not evidence["geometry_feasible"]
    assert any(
        contact["distance_m"] < -0.001
        for contact in evidence["initial_predicted_endpoint_contacts"]
    )
    assert native.ik_calls > 0 and native.path_calls > 0
