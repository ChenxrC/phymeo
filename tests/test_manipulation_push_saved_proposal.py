"""Portable saved-route compatibility: only seven explicit policy defaults may be absent."""

import gzip
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from test_manipulation_push_objective import propose

DEFAULTS = {
    "selection_policy": "pose-cost",
    "selection_eligible": True,
    "selection_reason": "legacy_pose_cost",
    "position_error_before_m": None,
    "position_error_after_m": None,
    "yaw_error_before_rad": None,
    "yaw_error_after_rad": None,
}


def payload():
    value = json.loads(json.dumps(asdict(propose()[0])))
    return {key: item for key, item in value.items() if key not in DEFAULTS}


def records(proposal):
    state = dict(q=[0.0, 0.25], v=[0.1, 0.2], time=1.0)
    route = dict(
        status="sampled_reference_ready",
        proposal=proposal,
        source_q=state["q"],
        nodes=[[0.0, "approach", [0.0, 0.25]], [0.5, "push", [0.1, 0.3]]],
        attempts=[],
        ik_calls=4,
        path_calls=2,
        initial_reference_projection=[0.0, 0.0],
    )
    saved = dict(tick=1000, q=state["q"], v=state["v"], cost=2.0, proposals=[proposal], route=route)
    record = dict(
        file="route-01.json.gz",
        sha256="caller-verifies",
        tick=1000,
        status="sampled_reference_ready",
        ik_calls=4,
        path_calls=2,
    )
    driver = SimpleNamespace(
        tick=1000,
        last_q=np.asarray(state["q"]),
        active=None,
        previous_direction=(1.0, 0.0, 0.0),
        planner=SimpleNamespace(),
        routes=[],
        segment_impulse=0.03,
        total_impulse=0.7,
        cost=lambda q: 2.0,
    )
    return driver, record, saved, state


def test_default_generated_proposals_use_exact_legacy_extension_defaults():
    for proposal in propose():
        values = asdict(proposal)
        assert {key: values[key] for key in DEFAULTS} == DEFAULTS


@pytest.mark.parametrize("missing", (tuple(DEFAULTS), *((key,) for key in DEFAULTS)))
def test_complete_saved_proposal_only_fills_explicit_defaults_without_mutating_source(missing):
    from phymeo.control.manipulation_push_objective import complete_saved_push_proposal

    value = payload() | DEFAULTS
    for key in missing:
        del value[key]
    before = json.dumps(value, sort_keys=True).encode()
    complete = complete_saved_push_proposal(value)
    assert complete == payload() | DEFAULTS
    assert json.dumps(value, sort_keys=True).encode() == before
    assert complete is not value


@pytest.mark.parametrize("change", ("old-field-missing", "unknown-field", "partial-new-policy"))
def test_compatibility_does_not_guess_other_schema_changes(change):
    from phymeo.control.manipulation_push_objective import complete_saved_push_proposal

    value = payload()
    if change == "old-field-missing":
        del value["predicted_yaw"]
    elif change == "unknown-field":
        value["mystery"] = 1
    else:
        value["selection_policy"] = "joint-pose-descent"
    with pytest.raises(ValueError):
        complete_saved_push_proposal(value)


def test_complete_nondefault_policy_fields_are_preserved_verbatim():
    from phymeo.control.manipulation_push_objective import complete_saved_push_proposal

    record = propose(
        selection_policy="joint-pose-descent",
        target_xy=(0.02, 0.03),
        target_yaw=0.12,
        position_tolerance_m=0.001,
        yaw_tolerance_rad=0.01,
    )[0]
    value = json.loads(json.dumps(asdict(record)))
    assert value["selection_eligible"] is True
    assert complete_saved_push_proposal(value) == value
    assert value["selection_reason"] == "joint_pose_descent"


def test_old_active_route_installs_without_rewriting_saved_payload_or_impulse():
    from experiments.area_pressure.pusht_active_entry import _restore_active_route
    from experiments.area_pressure.pusht_public_feedback import plain

    driver, record, saved, state = records(payload())
    before = deepcopy(saved)
    _restore_active_route(driver, record, saved, state)
    expected = deepcopy(saved["route"])
    expected["proposal"].update(DEFAULTS)
    assert plain(driver.active[1]) == expected
    assert saved == before
    assert (driver.segment_impulse, driver.total_impulse) == (0.03, 0.7)


def test_old_prefix_replay_binds_original_gzip_and_recomputed_default_proposals(tmp_path):
    from experiments.area_pressure.pusht_public_feedback import _PrefixPlanner, plain

    proposals = propose()
    _, record, saved, state = records(payload())
    saved["proposals"] = [
        {k: v for k, v in json.loads(json.dumps(asdict(p))).items() if k not in DEFAULTS}
        for p in proposals
    ]
    path = tmp_path / record["file"]
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(saved, stream)
    original = path.read_bytes()
    record["sha256"] = hashlib.sha256(original).hexdigest()
    live = SimpleNamespace(ik_calls=0, path_calls=0)
    prefix = _PrefixPlanner(live, tmp_path.resolve(), [record])
    prefix.tick = 1000
    route = prefix.plan(np.asarray(state["q"]), proposals)
    assert plain(route.proposal) == payload() | DEFAULTS
    assert path.read_bytes() == original
    assert prefix.used[0]["sha256"] == hashlib.sha256(original).hexdigest()


@pytest.mark.parametrize(
    "change", ("old-field-missing", "unknown-field", "old-value-changed", "new-value-changed")
)
def test_prefix_rejects_every_nondefault_change_even_with_a_matching_gzip_digest(tmp_path, change):
    from experiments.area_pressure.pusht_public_feedback import _PrefixPlanner

    proposals = propose()
    _, record, saved, state = records(payload())
    saved["proposals"] = [
        {k: v for k, v in json.loads(json.dumps(asdict(p))).items() if k not in DEFAULTS}
        for p in proposals
    ]
    first = saved["proposals"][0]
    if change == "old-field-missing":
        del first["predicted_yaw"]
    elif change == "unknown-field":
        first["mystery"] = 1
    elif change == "old-value-changed":
        first["predicted_yaw"] += 0.01
    else:
        first.update(DEFAULTS)
        first["selection_eligible"] = False
    path = tmp_path / record["file"]
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(saved, stream)
    record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    prefix = _PrefixPlanner(SimpleNamespace(ik_calls=0, path_calls=0), tmp_path.resolve(), [record])
    prefix.tick = 1000
    with pytest.raises(ValueError):
        prefix.plan(np.asarray(state["q"]), proposals)
