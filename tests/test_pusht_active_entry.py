"""Restore the saved active ordinary route without executing any prior physics."""

import gzip
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from shutil import copyfile, copytree
from types import SimpleNamespace

import mujoco as mj
import numpy as np
import pytest

from experiments.area_pressure.pusht_active_entry import prepare_active_entry
from phymeo.control.manipulation_push_planner import NativePushRoutePlanner
from scripts.run_manipulation_control import plain, reference
from scripts.run_manipulation_push_feedback import PushFeedbackDriver

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "artifacts/rr-public-feedback-20261121-01/continuous-damped.json"
RELEASE = ROOT / "artifacts/rr-loaded-withdrawal-20261121-01/release-b.json"


@pytest.fixture
def source():
    if not PRIOR.is_file() or not RELEASE.is_file():
        pytest.skip("requires the independently saved continuous-damped/release-b run")
    prior = json.loads(PRIOR.read_text("utf-8"))
    report = json.loads(RELEASE.read_text("utf-8"))
    dependencies = [
        PRIOR.with_name(PRIOR.stem + "-comparison.json"),
        PRIOR.parent / prior["journal"]["file"],
        Path(prior["provenance"]["candidate_source"]),
        RELEASE.parent / report["journal"]["file"],
        RELEASE.parent / report["release_reference"]["file"],
    ]
    dependencies.extend(
        PRIOR.with_suffix(".routes") / r["file"] for r in prior["controller_final"]["routes"]
    )
    dependencies.extend(
        RELEASE.with_suffix(".routes") / r["file"] for r in report["controller_final"]["routes"]
    )
    if not all(path.is_file() for path in dependencies):
        pytest.skip("the local saved run is incomplete; portable restoration test still runs")
    return report


def _forbidden(*args, **kwargs):
    raise AssertionError("active-entry restore must not simulate, plan or issue a reference")


def test_saved_route02_resumes_at_64ms_with_both_impulses_and_exact_state(
    source, tmp_path, monkeypatch
):
    final = source["controller_final"]
    with monkeypatch.context() as guard:
        for name in ("mj_forward", "mj_inverse", "mj_step", "mj_step1", "mj_step2"):
            guard.setattr(mj, name, _forbidden)
        guard.setattr(NativePushRoutePlanner, "plan", _forbidden)
        guard.setattr(PushFeedbackDriver, "reference", _forbidden)
        entry = prepare_active_entry(PRIOR, RELEASE, tmp_path / "resumed.routes")
    assert entry.tick == entry.driver.tick == 2014
    assert entry.state == source["final_state"]
    assert entry.state["time"] == 2.014000095659867
    assert entry.history.tick == 2013
    assert plain(entry.driver.summary()) == final
    start, route = entry.driver.active
    assert start == 1950 and route.nodes[-1][0] == 2.25
    assert final["partial_route"]["elapsed_s"] == 0.064
    assert entry.driver.segment_impulse == 0.014326137206355822
    assert entry.driver.total_impulse == 0.5389230065050594
    assert entry.driver.segment_impulse > 0  # Withdrawal impulse is not reset by replan.
    assert entry.driver.previous_direction is None
    # The published 2014 quaternion has no equivalent unit declaration in its
    # FP32 cells. initial creates the original 1700 host; physical continuation
    # must separately restore a public Scene checkpoint and verify its readback.
    for name in ("q", "v"):
        assert (
            np.asarray(getattr(entry.initial, name), np.float32).tobytes()
            == np.asarray(source["initial_state"][name], np.float32).tobytes()
        )
    assert entry.initial.time == source["initial_state"]["time"]
    assert entry.provenance["bootstrap_tick"] == source["start_tick"]
    assert entry.provenance["bootstrap_time"] == source["initial_state"]["time"]
    assert entry.provenance["source_local_tick"] == 314
    assert entry.provenance["continuation"] == "public-scene-checkpoint"
    assert entry.provenance["checkpoint_required"] is True
    record = source["ordinary_route"]
    copied = entry.driver.output / record["file"]
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == record["sha256"]
    with gzip.open(copied, "rt", encoding="utf-8") as stream:
        saved_route = json.load(stream)
    from phymeo.control.manipulation_push_planner import complete_saved_push_route

    assert plain(route) == complete_saved_push_route(saved_route["route"])
    assert entry.provenance["active_route_sha256"] == record["sha256"]
    assert entry.provenance["controller_replayed_callbacks"] == 314
    assert "fresh" in entry.provenance["owner_scope"]
    # Pure interpolation from the retained route is now available. A replan
    # remains forbidden, so resetting active/start or using a hold cannot pass.
    monkeypatch.setattr(NativePushRoutePlanner, "plan", _forbidden)
    qref, vref, stage, detail = entry.driver.reference(
        entry.tick, np.asarray(entry.state["q"]), np.asarray(entry.state["v"])
    )
    expected = reference(route.nodes, 0.064)
    np.testing.assert_array_equal(qref, expected[0])
    np.testing.assert_array_equal(vref, expected[1])
    assert stage == expected[2] == "approach"
    assert detail["route_start_tick"] == 1950 and detail["route_elapsed_s"] == 0.064


def _copy_release(source, target):
    path = target / RELEASE.name
    copyfile(RELEASE, path)
    copyfile(RELEASE.parent / source["journal"]["file"], target / source["journal"]["file"])
    copyfile(RELEASE.with_suffix(".release-route.json"), path.with_suffix(".release-route.json"))
    copytree(RELEASE.with_suffix(".routes"), path.with_suffix(".routes"))
    return path


@pytest.mark.parametrize("changed", ["history", "segment_impulse", "total_impulse"])
def test_rehashed_callback_corruption_cannot_replace_independently_replayed_history(
    source, tmp_path, changed
):
    path = _copy_release(source, tmp_path)
    report = deepcopy(source)
    journal = path.parent / report["journal"]["file"]
    events = [json.loads(line) for line in journal.read_text("utf-8").splitlines()]
    callback = next(e for e in reversed(events) if e["event"] == "accepted_callback")
    payload = callback["payload"]
    summary_row = report["trials"][callback["trial_index"]]
    if changed == "history":
        payload["goal_result"]["history"]["held_s"] += 0.01
        summary_row["goal_result"] = deepcopy(payload["goal_result"])
    else:
        field = changed + "_ns"
        payload["feedback_accepted"][field] += 0.01
        summary_row["feedback_accepted"] = deepcopy(payload["feedback_accepted"])
    encoded = "".join(
        json.dumps(e, allow_nan=False, separators=(",", ":")) + "\n" for e in events
    ).encode()
    journal.write_bytes(encoded)
    report["journal"]["sha256"] = hashlib.sha256(encoded).hexdigest()
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    with pytest.raises(ValueError):
        prepare_active_entry(PRIOR, path, tmp_path / "invalid.routes")


def test_active_route_source_hash_and_terminal_state_are_required(source, tmp_path):
    path = _copy_release(source, tmp_path)
    route = path.with_suffix(".routes") / source["ordinary_route"]["file"]
    with route.open("ab") as stream:
        stream.write(b"unbound-route-byte")
    with pytest.raises(ValueError):
        prepare_active_entry(PRIOR, path, tmp_path / "invalid.routes")


def test_summary_endpoint_cannot_override_last_accepted_journal_state(source, tmp_path):
    path = _copy_release(source, tmp_path)
    source["final_state"]["v"][0] = 0.0
    path.write_text(json.dumps(source, allow_nan=False), encoding="utf-8")
    with pytest.raises(ValueError):
        prepare_active_entry(PRIOR, path, tmp_path / "invalid.routes")


def test_portable_active_route_restore_binds_source_and_keeps_accumulated_impulse():
    """Small saved-record parser/installation test; no archive or path claim."""
    from experiments.area_pressure.pusht_active_entry import _restore_active_route

    proposal = dict(
        status="PROPOSAL_ONLY",
        input_binding="portable-binding",
        index=0,
        distance_ordinal=0,
        distance_m=0.01,
        reference_usable=True,
        point_world=[0.0, 0.0, 0.0],
        outward_normal_world=[0.0, -1.0, 0.0],
        object_com_world=[0.0, 0.0, 0.0],
        contact_center_world=[0.0, 0.0, 0.0],
        precontact_center_world=[0.0, -0.01, 0.0],
        push_center_world=[0.0, 0.01, 0.0],
        direction_world=[0.0, 1.0, 0.0],
        nominal_yaw_moment=0.0,
        mobility_radius_m=0.1,
        predicted_translation_xy=[0.0, 0.01],
        predicted_yaw_change=0.0,
        predicted_position_xy=[0.0, 0.01],
        predicted_yaw=0.0,
        cost_before=2.0,
        cost_after=1.0,
        predicted_improvement=1.0,
        sort_key=[0, -1.0, 0, 0],
    )
    state = dict(q=[0.0, 0.25], v=[0.1, 0.2], time=1.0)
    raw = dict(
        status="sampled_reference_ready",
        proposal=proposal,
        source_q=state["q"],
        nodes=[[0.0, "approach", [0.0, 0.25]], [0.5, "push", [0.1, 0.3]]],
        attempts=[],
        ik_calls=4,
        path_calls=2,
        initial_reference_projection=[0.0, 0.0],
    )
    saved = dict(tick=1000, q=state["q"], v=state["v"], cost=2.0, proposals=[proposal], route=raw)
    record = dict(
        file="route-02.json.gz",
        sha256="portable-parser-test",
        tick=1000,
        status="sampled_reference_ready",
        ik_calls=4,
        path_calls=2,
    )

    def driver():
        return SimpleNamespace(
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

    current = driver()
    _restore_active_route(current, record, saved, state)
    assert current.active[0] == 1000
    from phymeo.control.manipulation_push_planner import complete_saved_push_route

    assert plain(current.active[1]) == complete_saved_push_route(raw)
    assert current.segment_impulse == 0.03 and current.total_impulse == 0.7
    assert current.previous_direction is None
    assert current.planner.ik_calls == 4 and current.planner.path_calls == 2
    for field in ("q", "v", "tick"):
        stale = deepcopy(saved)
        if field == "tick":
            stale[field] += 1
        else:
            stale[field][0] += 1.0
        with pytest.raises(ValueError):
            _restore_active_route(driver(), record, stale, state)
