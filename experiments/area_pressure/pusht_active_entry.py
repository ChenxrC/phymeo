"""Restore a saved active ordinary Push-T route after a loaded-release prefix.

Only saved Goal/controller callbacks are replayed. No dynamics, PD command,
CUDA transport, reference issuance or planning is executed during restoration.
The returned initial declaration only creates the original release-entry host.
Physical continuation requires a separately generated public Scene checkpoint
and exact final-state readback; this controller adapter restores no physics cache.
"""

import gzip
import json
from copy import deepcopy
from pathlib import Path
from shutil import copyfile

import numpy as np

from phymeo.control.manipulation_push_objective import NativePlanarPushProposal
from phymeo.control.manipulation_push_planner import NativePushRoute, complete_saved_push_route
from phymeo.training.manipulation_tasks import H, SuccessFrame, evaluate_goal
from scripts.run_manipulation_control import plain, reference
from scripts.run_manipulation_push import tuple_values

from .pusht_entry_rows import _digest
from .pusht_public_feedback import PublicFeedbackMeasurement
from .pusht_release_entry import (
    ReleaseEntry,
    _bound_file,
    prepare_release_entry,
)


def _require(ok, reason):
    if not ok:
        raise ValueError("active entry: " + reason)


def _bad_constant(value):
    raise ValueError("active entry: nonfinite JSON constant " + value)


def _json(path):
    return json.loads(path.read_text("utf-8"), parse_constant=_bad_constant)


def _words(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (
        a.shape == b.shape
        and np.array_equal(a, b)
        and a.astype(np.float32).tobytes() == b.astype(np.float32).tobytes()
    )


def _same_state(a, b):
    return a["time"] == b["time"] and _words(a["q"], b["q"]) and _words(a["v"], b["v"])


def _state(model, state):
    for field, length in (("q", model.nq), ("v", model.nv)):
        values = np.asarray(state[field])
        _require(
            values.shape == (length,)
            and values.dtype.kind in "fiu"
            and np.isfinite(values).all()
            and _words(values, values.astype(np.float32)),
            "finite published FP32 " + field,
        )
    _require(
        type(state["local_tick"]) is int
        and state["local_tick"] >= 0
        and np.isfinite(state["time"])
        and state["time"] >= 0,
        "saved tick/clock",
    )
    qa = int(model.jnt_qposadr[model.joint("object-free").id])
    _require(
        _words(state["object_position_world_m"], state["q"][qa : qa + 3])
        and _words(state["object_quaternion_wxyz"], state["q"][qa + 3 : qa + 7]),
        "saved object observation/state binding",
    )


def _restore_active_route(driver, record, saved, state):
    """Install an already bound route at its saved creation boundary, without planning.

    The caller verifies the gzip digest. Keeping segment_impulse unchanged is
    essential: the ordinary replan did not erase earlier withdrawal impulse.
    """
    _require(
        driver.active is None
        and driver.tick == record["tick"] == saved["tick"]
        and _words(driver.last_q, state["q"])
        and _words(saved["q"], state["q"])
        and _words(saved["v"], state["v"]),
        "active route source tick/q/v differs from its accepted boundary",
    )
    raw = saved["route"]
    _require(
        record["status"] == raw["status"] == "sampled_reference_ready"
        and raw["proposal"] is not None
        and raw["proposal"] in saved["proposals"]
        and _words(raw["source_q"], state["q"])
        and saved["cost"] == driver.cost(np.asarray(state["q"])),
        "ordinary route source/proposal/cost binding",
    )
    _require(
        raw["ik_calls"] == record["ik_calls"] and raw["path_calls"] == record["path_calls"],
        "route planner budget binding",
    )
    nodes = raw["nodes"]
    _require(
        2 <= len(nodes) <= 32
        and nodes[0][0] == 0
        and all(np.isfinite(row[0]) and row[0] >= 0 for row in nodes)
        and all(b[0] > a[0] for a, b in zip(nodes, nodes[1:])),
        "bounded route timeline",
    )
    width = len(nodes[0][2])
    _require(
        width > 0
        and all(
            len(n[2]) == width
            and np.isfinite(n[2]).all()
            and n[1] in ("withdraw", "approach", "entry", "push")
            for n in nodes
        ),
        "finite saved route nodes",
    )
    complete = complete_saved_push_route(raw)
    proposal = NativePlanarPushProposal(
        **{k: tuple_values(v) for k, v in complete["proposal"].items()}
    )
    route = NativePushRoute(
        raw["status"],
        proposal,
        np.asarray(raw["source_q"], dtype=float).copy(),
        tuple((t, phase, np.asarray(q, dtype=float).copy()) for t, phase, q in nodes),
        tuple(deepcopy(raw["attempts"])),
        raw["ik_calls"],
        raw["path_calls"],
        tuple(raw["initial_reference_projection"]),
    )
    _require(plain(route) == complete, "lossless saved route reconstruction")
    driver.previous_direction = None
    driver.active = (record["tick"], route)
    driver.routes.append(deepcopy(record))
    driver.planner.ik_calls = record["ik_calls"]
    driver.planner.path_calls = record["path_calls"]


def _candidate(entry, row, state, start_tick):
    before, after = row["input_state"], row["output_state"]
    _state(entry.model, before)
    _state(entry.model, after)
    tick = start_tick + before["local_tick"]
    _require(
        before == state
        and entry.driver.tick == tick
        and row["original_input_tick"] == tick
        and row["original_output_tick"] == tick + 1
        and after["local_tick"] == before["local_tick"] + 1
        and after["time"] > before["time"],
        "continuous candidate state/clock/tick",
    )
    controller, command = row["controller"], row["control"]
    _require(
        _words(controller["measurement_q"], before["q"])
        and _words(controller["measurement_v"], before["v"])
        and command["tick"] == tick
        and command["output_tick"] == tick + 1
        and row["stored_control_matches_submitted"] is True,
        "candidate controller measurement/control binding",
    )
    transport = controller["transport"]
    _require(
        transport["kind"] == "original-env-fp32-atanh-cuda-tanh"
        and _words(transport["transported_ctrl"], command["ctrl"]),
        "actual archived CUDA control transport binding",
    )
    if row["phase"] == "regular":
        _require(
            entry.driver.active is not None and row["trial_mode"] == "regular",
            "regular input lacks its installed active route",
        )
        start, route = entry.driver.active
        elapsed = (tick - start) * H
        q, v, phase = reference(route.nodes, elapsed)
        meta = controller["route"]
        _require(
            controller["input_tick"] == tick
            and controller["input_time"] == before["time"]
            and meta["route_start_tick"] == start
            and meta["route_elapsed_s"] == elapsed
            and meta["route"] == len(entry.driver.routes)
            and meta["route_sha256"] == entry.driver.routes[-1]["sha256"]
            and Path(meta["route_file"]).name == entry.driver.routes[-1]["file"]
            and np.array_equal(q, controller["target_q"])
            and np.array_equal(v, controller["target_v"])
            and phase == controller["stage"],
            "saved active reference differs from original route interpolation",
        )
    else:
        _require(
            row["phase"] == "withdraw" and entry.driver.active is None,
            "withdrawal/ordinary phase ordering",
        )


def _accepted(entry, row, callback, history):
    before, after = row["input_state"], row["output_state"]
    tick = row["original_input_tick"]
    frame = SuccessFrame(**callback["goal_frame"])
    _require(
        frame.world == 6
        and frame.tick == tick
        and frame.time_s == before["time"]
        and frame.phase == "accepted-entry"
        and _words(frame.object_position, before["object_position_world_m"])
        and _words(frame.object_wxyz, before["object_quaternion_wxyz"]),
        "Goal input phase/state binding",
    )
    goal = evaluate_goal(entry.scene, entry.driver.goal, frame, history)
    _require(
        plain(goal) == callback["goal_result"] and not goal.success,
        "independently replayed Goal history/result",
    )
    feedback = callback["feedback_accepted"]
    measurement = PublicFeedbackMeasurement(**feedback["measurement"])
    _require(
        measurement.world == 6
        and measurement.input_tick == tick
        and measurement.output_tick == tick + 1
        and measurement.time_s == before["time"]
        and measurement.phase == "accepted-step-input"
        and measurement.native_contacts == 0
        and measurement.area_samples == row["area"]["samples"]
        and np.array_equal(
            measurement.group_force_world, -np.asarray(row["area"]["total_force_on_pusher_world_n"])
        ),
        "accepted actual material-force source binding",
    )
    replayed = entry.driver.accepted_measurement(
        tick + 1, np.asarray(after["q"]), measurement, goal, input_q=np.asarray(before["q"])
    )
    _require(plain(replayed) == feedback, "replayed controller progress/segment/total impulse")
    summary = plain(entry.driver.summary())
    boundary = {
        k: summary[k]
        for k in (
            "status",
            "tick",
            "final_cost",
            "completed_routes",
            "total_contact_impulse_ns",
            "partial_route",
        )
    }
    _require(
        boundary == callback["driver_state_after"] and not entry.driver.stopped,
        "accepted controller boundary differs from saved state",
    )
    return goal.history


def prepare_active_entry(prior_summary_path, release_report_path, route_output_dir):
    """Return an active-route ReleaseEntry from bound summary/journal/route sources.

    ``initial`` retains the earlier legal release bootstrap declaration, not the
    final published quaternion. The caller must restore a public Scene checkpoint
    before pairing this active controller with the final saved physical state.
    A successful finite release report must end while its ordinary route is
    active. The complete selected prefix is audited against the independent
    JSONL callbacks and summary. Rejected trial endpoints never advance history.
    """
    prior_path, path = Path(prior_summary_path).resolve(), Path(release_report_path).resolve()
    report = _json(path)
    count = len(report["trials"])
    _require(
        report["schema"] == "phymeo-pusht-loaded-release-v1"
        and report["status"] == "regular_step_budget"
        and report["world"] == 6
        and report["regular_plan_ready"] is True
        and report["goal_success"] is False
        and 1 <= count <= 4097
        and report["physical_trial_steps"] == count
        and report["selected_steps"] + report["rejected_trials"] == count,
        "completed bounded release/ordinary prefix required",
    )
    _require(
        report["provenance"]["source_summary_sha256"] == _digest(prior_path),
        "prior physical source summary",
    )
    journal = _bound_file(path.parent, report["journal"]["file"], report["journal"]["sha256"])
    route_root = path.with_suffix(".routes").resolve()
    ordinary = report["ordinary_route"]
    route_path = _bound_file(route_root, ordinary["file"], ordinary["sha256"])
    _require(
        route_path.name == ordinary["file"] and ordinary["tick"] == report["ordinary_replan_tick"],
        "ordinary route file/tick binding",
    )
    with gzip.open(route_path, "rt", encoding="utf-8") as stream:
        saved_route = json.load(stream, parse_constant=_bad_constant)
    release_reference = report["release_reference"]
    withdrawal_path = _bound_file(
        path.parent, release_reference["file"], release_reference["sha256"]
    )
    withdrawal = _json(withdrawal_path)
    _require(
        withdrawal["tick"] == report["start_tick"]
        and withdrawal["route"]["status"] == "sampled_release_ready"
        and withdrawal["route"]["nodes"] == release_reference["nodes"],
        "original loaded-withdrawal source",
    )
    entry = prepare_release_entry(prior_path, route_output_dir)
    _require(
        entry.provenance == report["provenance"]
        and plain(entry.driver.summary()) == report["prior_controller"]
        and entry.tick == report["start_tick"]
        and _same_state(entry.state, report["initial_state"])
        and _words(withdrawal["q"], entry.state["q"])
        and _words(withdrawal["v"], entry.state["v"]),
        "release initialization/source history",
    )
    expected_routes = report["prior_controller"]["routes"] + [ordinary]
    _require(
        report["controller_final"]["routes"] == expected_routes
        and report["controller_final"]["partial_route"] is not None,
        "one saved active ordinary route after the prior prefix",
    )
    state, history = report["initial_state"], entry.history
    _state(entry.model, state)
    _require(state["local_tick"] == 0, "fresh source owner local tick")
    pending = last = None
    candidates = accepted = rejected = records = replans = 0
    initialized = ended = False
    with journal.open("rb") as stream:
        for line in stream:
            records += 1
            event = json.loads(line, parse_constant=_bad_constant)
            _require(
                not ended and event["schema"] == "phymeo-pusht-release-event-v1",
                "journal event schema/order",
            )
            kind, payload, index = event["event"], event["payload"], event["trial_index"]
            if kind == "initialization":
                _require(
                    records == 1
                    and index is None
                    and not initialized
                    and payload == {k: report[k] for k in payload},
                    "journal initialization",
                )
                initialized = True
            elif kind == "candidate":
                _require(
                    initialized
                    and pending is None
                    and type(index) is int
                    and index == candidates < count,
                    "candidate/callback ordering",
                )
                _candidate(entry, payload, state, report["start_tick"])
                candidates += 1
                _require(
                    type(payload["selected"]) is bool and type(payload["restored"]) is bool,
                    "explicit candidate selection",
                )
                if payload["selected"]:
                    _require(not payload["restored"], "selected candidate was rolled back")
                    pending = (index, payload)
                else:
                    _require(
                        payload["restored"]
                        and payload["rollback_verification"]["passed"]
                        and payload == report["trials"][index],
                        "rejected trial binding/rollback",
                    )
                    rejected += 1
            elif kind == "accepted_callback":
                _require(pending is not None and index == pending[0], "accepted callback identity")
                row = pending[1]
                history = _accepted(entry, row, payload, history)
                row.update(payload)
                _require(
                    row == report["trials"][index], "independent journal/summary trial binding"
                )
                state, last = row["output_state"], row
                pending = None
                accepted += 1
            elif kind == "ordinary_replan":
                _require(
                    pending is None
                    and last is not None
                    and not replans
                    and index == candidates - 1
                    and last["phase"] == "withdraw"
                    and payload == ordinary,
                    "ordinary replan event/source boundary",
                )
                _restore_active_route(entry.driver, ordinary, saved_route, state)
                copied = entry.driver.output / route_path.name
                _require(not copied.exists(), "active route output already exists")
                copyfile(route_path, copied)
                _require(_digest(copied) == ordinary["sha256"], "copied original route bytes")
                replans += 1
            elif kind == "final":
                _require(
                    pending is None
                    and index is None
                    and payload
                    == {
                        k: report.get(k)
                        for k in (
                            "status",
                            "selected_steps",
                            "physical_trial_steps",
                            "rejected_trials",
                            "final_state",
                            "regular_plan_ready",
                            "goal_success",
                            "controller_final",
                        )
                    },
                    "independent journal final summary",
                )
                ended = True
            else:
                raise ValueError("active entry: unsupported/failing journal event " + str(kind))
    _require(
        initialized
        and ended
        and replans == 1
        and last is not None
        and records == report["journal"]["records"]
        and candidates == count
        and accepted == report["selected_steps"]
        and rejected == report["rejected_trials"]
        and state == report["final_state"]
        and last["phase"] == "regular",
        "complete selected journal prefix and exact saved endpoint",
    )
    driver = entry.driver
    summary = plain(driver.summary())
    _require(
        summary == report["controller_final"]
        and driver.active is not None
        and driver.reason == "continue"
        and history.tick == driver.tick - 1
        and driver.tick == report["start_tick"] + state["local_tick"],
        "full restored active controller/Goal boundary",
    )
    partial = summary["partial_route"]
    _require(0 < partial["elapsed_s"] < partial["duration_s"], "route must remain active")
    provenance = dict(
        entry.provenance,
        prior_summary_sha256=_digest(prior_path),
        source_summary_sha256=_digest(path),
        release_report_sha256=_digest(path),
        release_journal_sha256=report["journal"]["sha256"],
        journal_sha256=report["journal"]["sha256"],
        active_route_sha256=ordinary["sha256"],
        active_route_start_tick=partial["start_tick"],
        active_route_elapsed_s=partial["elapsed_s"],
        controller_replayed_callbacks=accepted,
        source_tick=driver.tick,
        source_local_tick=state["local_tick"],
        source_time=state["time"],
        source_controller="saved accepted callbacks and original active route; no physics/replan replay",
        bootstrap_tick=entry.tick,
        bootstrap_time=entry.initial.time,
        continuation="public-scene-checkpoint",
        checkpoint_required=True,
        initializer="original release bootstrap declaration for a compatible checkpoint restore host",
        owner_scope="fresh local owner; bootstrap is not continuation state; public Scene checkpoint and exact readback required",
    )
    return ReleaseEntry(
        entry.scene,
        entry.model,
        driver,
        entry.pd,
        history,
        deepcopy(state),
        deepcopy(last),
        driver.tick,
        entry.initial,
        provenance,
    )
