"""Saved A/B analysis contracts; synthetic readout rows are not physics evidence."""

from copy import deepcopy

import pytest

from experiments.area_pressure.pusht_load_comparison import (
    bind_comparison,
    causal_prefix,
    stage_motion,
)


def sources():
    route = dict(
        start_tick=1950,
        end_tick=4200,
        duration_s=2.25,
        elapsed_at_entry_s=0.064,
        remaining_steps=2186,
        route_index=2,
        nodes=[[0.0, "push", [0.0, 0.0]]],
        sha256="b" * 64,
    )
    base = dict(
        start_tick=2014,
        requested_steps=4,
        recorded_steps=4,
        accepted_steps=4,
        status="route_complete",
        world=6,
        local_tick_origin=314,
        initial_state=dict(q=[0.0, 1.0], v=[0.0, 0.0], time=2.014, local_tick=314),
        goal=dict(target_xy=[0.55, 0.13], target_yaw=0.2),
        active_route=route,
        provenance=dict(source_sha256="m" * 64, scene_asset_sha256="s" * 64),
    )
    candidate = deepcopy(base)
    candidate["control_variant"] = "load-compensation-half"
    audit = dict(
        status="verified-saved-prefix",
        source_checkpoint_sha256="a" * 64,
        verified_steps=4,
        control=dict(variant="position"),
    )
    variant_audit = deepcopy(audit)
    variant_audit["control"]["variant"] = "load-compensation-half"
    return base, candidate, audit, variant_audit


def test_partial_failure_compares_only_the_independently_verified_common_prefix():
    base, candidate, audit, variant_audit = sources()
    candidate.update(status="partial_failure", recorded_steps=2, accepted_steps=3)
    variant_audit["verified_steps"] = 2
    bound = bind_comparison(base, candidate, audit, variant_audit)
    assert bound["common_steps"] == 2 and not bound["full_recorded_window"]
    assert bound["excluded_baseline_steps"] == 2
    assert bound["excluded_candidate_steps"] == 0
    assert bound["candidate_unrecorded_accepted_steps"] == 1


@pytest.mark.parametrize(
    "wrong", ("checkpoint", "initial", "route", "goal", "phase", "variant", "audit_count")
)
def test_different_sources_or_unaudited_windows_are_not_an_ab_experiment(wrong):
    base, candidate, audit, variant_audit = sources()
    if wrong == "checkpoint":
        variant_audit["source_checkpoint_sha256"] = "c" * 64
    elif wrong == "initial":
        candidate["initial_state"]["q"][0] = 0.001
    elif wrong == "route":
        candidate["active_route"]["sha256"] = "c" * 64
    elif wrong == "goal":
        candidate["goal"]["target_yaw"] = 0.3
    elif wrong == "phase":
        candidate["active_route"]["elapsed_at_entry_s"] = 0.0
    elif wrong == "variant":
        candidate["control_variant"] = "position"
    else:
        variant_audit["verified_steps"] = 3
    with pytest.raises(ValueError):
        bind_comparison(base, candidate, audit, variant_audit)


def trace():
    return [dict(tick=10 + i, q=[0.0], v=[0.0], ctrl=[0.0], correction_nm=[0.0]) for i in range(3)]


def test_first_control_change_has_identical_entry_state_and_precedes_state_change():
    base, candidate = trace(), trace()
    candidate[1]["correction_nm"] = [0.25]
    candidate[1]["ctrl"] = [0.25]
    candidate[2].update(q=[0.001], v=[0.1], ctrl=[0.2], correction_nm=[0.25])
    result = causal_prefix(base, candidate)
    assert result["first_nonzero_compensation_tick"] == 11
    assert result["first_command_difference_tick"] == 11
    assert result["first_input_state_difference_tick"] == 12
    assert result["identical_entry_and_command_steps"] == 1


@pytest.mark.parametrize(
    "wrong", ("state_before_control", "control_without_compensation", "unaligned_tick")
)
def test_causal_prefix_rejects_future_state_or_unexplained_command_difference(wrong):
    base, candidate = trace(), trace()
    if wrong == "state_before_control":
        candidate[1].update(q=[0.01], ctrl=[0.25], correction_nm=[0.25])
    elif wrong == "control_without_compensation":
        candidate[1]["ctrl"] = [0.25]
    else:
        candidate[1]["tick"] += 1
    with pytest.raises(ValueError):
        causal_prefix(base, candidate)


def test_push_advances_use_same_stage_input_states_not_whole_run_or_reference_as_object_motion():
    rows = [
        dict(
            tick=10,
            stage="approach",
            pusher_along_m=-5.0,
            reference_along_m=-3.0,
            object_along_m=-7.0,
        ),
        dict(tick=11, stage="push", pusher_along_m=0.1, reference_along_m=0.2, object_along_m=0.3),
        dict(
            tick=12, stage="push", pusher_along_m=0.11, reference_along_m=0.24, object_along_m=0.305
        ),
    ]
    metrics = stage_motion(rows, "push")
    assert metrics["input_tick_start"] == 11 and metrics["input_tick_end"] == 12
    assert metrics["samples"] == 2
    assert metrics["pusher_advance_m"] == pytest.approx(0.01)
    assert metrics["reference_advance_m"] == pytest.approx(0.04)
    assert metrics["object_advance_m"] == pytest.approx(0.005)
