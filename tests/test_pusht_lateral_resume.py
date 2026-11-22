"""Portable continuation-boundary tests; synthetic rows are not physics evidence."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.area_pressure.pusht_lateral_resume import (
    accepted_pairs,
    bind_retry_command,
    reference_tick,
    validate_terminal_binding,
)


def state(tick):
    return dict(local_tick=tick, time=4.45 + tick * 0.001, q=[0.1, -0.2], v=[0.03, 0.04])


def terminal():
    endpoint = state(1386)
    route = dict(start_tick=4450, end_tick=6700)
    controller = dict(tick=5836, status="continue", partial_route=dict(start_tick=4450))
    report = dict(
        schema="phymeo-pusht-lateral-route-v1",
        world=6,
        side_coverage="all-lateral",
        control_variant="load-compensation-half",
        start_tick=4450,
        local_tick_origin=0,
        source_reference_tick_origin=4450,
        goal_tick_origin=0,
        accepted_steps=1386,
        recorded_steps=1386,
        final_state=endpoint,
        controller_final=controller,
        goal_history_final=dict(tick=1385),
        active_route=route,
        goal=dict(target_yaw=0.2),
        provenance=dict(scene_asset_sha256="asset"),
    )
    meta = dict(
        schema="phymeo-pusht-active-terminal-checkpoint-v1",
        world=6,
        state=deepcopy(endpoint),
        q=endpoint["q"],
        v=endpoint["v"],
        time=endpoint["time"],
        local_tick=1386,
        source_tick=5836,
        controller=deepcopy(controller),
        goal_history=dict(tick=1385),
        active_route=route,
        target=report["goal"],
        asset_sha256="asset",
        control_variant="load-compensation-half",
        accepted_steps=1386,
        recorded_steps=1386,
        controller_at_physical_endpoint=True,
        physical_state_matches_report=True,
    )
    return report, meta


def command(tick=5836):
    return dict(
        tick=tick,
        output_tick=tick + 1,
        ctrl=[0.125, -0.25],
        qfrc_applied=[0.0, -0.0],
        xfrc_applied=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    )


def test_terminal_preserves_absolute_reference_and_local_goal_clock():
    report, meta = terminal()
    validate_terminal_binding(report, meta)
    entry = SimpleNamespace(tick=5836, state=state(1386))
    assert reference_tick(entry, SimpleNamespace(tick=1386)) == 5836
    assert reference_tick(entry, SimpleNamespace(tick=2249)) == 6699
    assert report["active_route"]["end_tick"] - meta["source_tick"] == 864


@pytest.mark.parametrize("fault", ["goal_absolute", "q", "v", "time", "tick", "tail", "variant"])
def test_terminal_rejects_inexact_or_unrecorded_boundary(fault):
    report, meta = terminal()
    if fault == "goal_absolute":
        meta["goal_history"]["tick"] = 5835
    elif fault in ("q", "v"):
        meta[fault] = list(meta[fault])
        meta[fault][0] += 0.001
    elif fault == "time":
        meta["time"] += 0.001
    elif fault == "tick":
        meta["source_tick"] += 1
    elif fault == "tail":
        report["unrecorded_accepted_step"] = {"output_state": state(1387)}
    else:
        meta["control_variant"] = "position"
    with pytest.raises(ValueError):
        validate_terminal_binding(report, meta)


def test_retry_command_binds_all_channels_and_signed_zero_words():
    saved = command()
    actual = SimpleNamespace(
        **{k: np.asarray(saved[k], np.float64) for k in ("ctrl", "qfrc_applied", "xfrc_applied")}
    )
    proof = bind_retry_command(actual, saved, 5836)
    assert proof["matches_parent_rejected_command"] is True
    assert set(proof["channel_fp32_words"]) == {"ctrl", "qfrc_applied", "xfrc_applied"}


@pytest.mark.parametrize(
    "fault", ["ctrl", "qfrc_applied", "xfrc_applied", "tick", "zero_sign", "hidden_ulp"]
)
def test_retry_does_not_ignore_external_channels_or_fp64_rounding(fault):
    saved = command()
    actual = SimpleNamespace(
        **{k: np.asarray(saved[k], np.float64) for k in ("ctrl", "qfrc_applied", "xfrc_applied")}
    )
    if fault == "tick":
        saved["tick"] += 1
    elif fault == "zero_sign":
        actual.qfrc_applied[1] = 0.0
    elif fault == "hidden_ulp":
        actual.ctrl[0] = np.nextafter(actual.ctrl[0], np.inf)
    else:
        getattr(actual, fault).flat[0] += 0.001
    with pytest.raises(ValueError):
        bind_retry_command(actual, saved, 5836)


def events():
    row = dict(
        input_state=state(1386),
        output_state=state(1387),
        original_input_tick=5836,
        original_output_tick=5837,
        selected=True,
        restored=False,
    )
    return [
        dict(event="candidate", trial_index=0, payload=row),
        dict(event="accepted_callback", trial_index=0, payload={"goal_frame": {"tick": 1386}}),
    ]


def test_callback_replay_requires_selected_adjacent_pair_and_retains_local_goal():
    pairs = list(accepted_pairs(events(), state(1386), 4450))
    assert len(pairs) == 1 and pairs[0][1]["goal_frame"]["tick"] == 1386


@pytest.mark.parametrize("fault", ["callback_id", "gap", "state", "missing_callback", "restored"])
def test_replay_rejects_mismatched_or_unrecorded_candidate(fault):
    rows = events()
    if fault == "callback_id":
        rows[1]["trial_index"] = 1
    elif fault == "gap":
        rows[0]["payload"]["original_input_tick"] += 1
    elif fault == "state":
        rows[0]["payload"]["input_state"]["q"][0] += 0.01
    elif fault == "missing_callback":
        rows.pop()
    else:
        rows[0]["payload"]["restored"] = True
    with pytest.raises(ValueError):
        list(accepted_pairs(rows, state(1386), 4450))
