"""Read-only continuity/provenance audit of paired public PushT feedback runs.

No model, controller, native solver or physics is executed. Goal results are
reported from the original evaluator's saved accepted-entry frames; this is
not a replacement physical receipt or independent Goal implementation.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

LANES = ("open", "feedback")
CHANNELS = ("ctrl", "qfrc_applied", "xfrc_applied")


def _require(condition, detail):
    if not condition:
        raise ValueError("PushT feedback comparison: " + detail)


def _constant(value):
    raise ValueError("nonfinite JSON number: " + value)


def _read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        return json.load(stream, parse_constant=_constant)


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _array(value, shape, detail):
    result = np.asarray(value, dtype=float)
    _require(result.shape == shape and np.isfinite(result).all(), detail)
    return result


def _same(actual, expected, detail):
    _require(actual == expected, detail)


def _state(state, tick):
    _require(type(state["local_tick"]) is int and state["local_tick"] == tick, "local state tick")
    q, v = _array(state["q"], (9,), "state q"), _array(state["v"], (8,), "state v")
    _require(np.isfinite(state["time"]) and state["time"] >= 0, "state time")
    _require(
        np.array_equal(q, q.astype(np.float32)) and np.array_equal(v, v.astype(np.float32)),
        "state must preserve actual FP32 coordinates",
    )
    _same(state["object_position_world_m"], state["q"][2:5], "object translation/state binding")
    _same(state["object_quaternion_wxyz"], state["q"][5:9], "object orientation/state binding")
    _require(abs(np.linalg.norm(q[5:9]) - 1) <= 2e-6, "state quaternion")
    return q, v


def _channels(command):
    _array(command["ctrl"], (2,), "two motor commands")
    _array(command["qfrc_applied"], (8,), "external generalized load")
    body = np.asarray(command["xfrc_applied"], dtype=float)
    _require(
        body.ndim == 2 and body.shape[1] == 6 and np.isfinite(body).all(), "external body load"
    )
    return {name: command[name] for name in CHANNELS}


def _goal(row, world, tick, state):
    frame, result = row["goal_frame"], row["goal_result"]
    _require(
        frame["phase"] == "accepted-entry"
        and frame["world"] == world
        and frame["tick"] == tick
        and frame["time_s"] == state["time"],
        "original Goal accepted-entry phase",
    )
    _same(frame["object_position"], state["object_position_world_m"], "Goal entry object position")
    _same(frame["object_wxyz"], state["object_quaternion_wxyz"], "Goal entry object quaternion")
    _require(type(result["success"]) is bool, "literal original Goal success")
    history = result["history"]
    _require(
        history["tick"] == tick and history["time_s"] == state["time"],
        "original Goal history phase",
    )
    metrics = dict(result["metrics"])
    _require(all(np.isfinite(value) for value in metrics.values()), "finite original Goal metrics")
    return result["success"], metrics


def _load_stats():
    return dict(
        first_positive_input_tick=None,
        last_positive_input_tick=None,
        first_zero_after_load_input_tick=None,
        peak_normal_load_n=0.0,
        peak_input_tick=None,
        initial_normal_load_n=None,
        final_normal_load_n=None,
        normal_impulse_ns=0.0,
        minimum_object_support_n=None,
        final_zero_load_steps=0,
        first_empty_clip_input_tick=None,
        first_strict_disjoint_input_tick=None,
        first_empty_clip_after_load_input_tick=None,
        first_strict_disjoint_after_load_input_tick=None,
        final_unloaded_start_input_tick=None,
        final_empty_clip_start_input_tick=None,
        final_empty_clip_after_load_start_input_tick=None,
        final_strict_disjoint_after_load_start_input_tick=None,
        goal_success_input_ticks=[],
        final_goal_metrics=None,
    )


def _load_record(stats, row, tick, h):
    area = row["area"]
    load = float(area["normal_load_n"])
    _require(
        np.isfinite(load)
        and load >= 0
        and type(area["zero_normal_load"]) is bool
        and area["zero_normal_load"] == (load == 0),
        "area normal load",
    )
    _require(
        type(area["samples"]) is int and 0 <= area["active_samples"] <= area["samples"],
        "area samples",
    )
    if stats["initial_normal_load_n"] is None:
        stats["initial_normal_load_n"] = load
    stats["final_normal_load_n"] = load
    stats["normal_impulse_ns"] += h * load
    if load > 0:
        if stats["first_positive_input_tick"] is None:
            stats["first_positive_input_tick"] = tick
        stats["last_positive_input_tick"] = tick
        stats["final_zero_load_steps"] = 0
    else:
        stats["final_zero_load_steps"] += 1
        if (
            stats["first_positive_input_tick"] is not None
            and stats["first_zero_after_load_input_tick"] is None
        ):
            stats["first_zero_after_load_input_tick"] = tick
    if load > stats["peak_normal_load_n"]:
        stats["peak_normal_load_n"], stats["peak_input_tick"] = load, tick
    support = float(row["native"]["object_support_normal_n"])
    _require(np.isfinite(support) and support >= 0, "native object support")
    if stats["minimum_object_support_n"] is None or support < stats["minimum_object_support_n"]:
        stats["minimum_object_support_n"] = support
    if area["samples"] == 0 and stats["first_empty_clip_input_tick"] is None:
        stats["first_empty_clip_input_tick"] = tick
    if (
        row["geometry"]["strict_disjoint_certified"]
        and stats["first_strict_disjoint_input_tick"] is None
    ):
        stats["first_strict_disjoint_input_tick"] = tick

    # Event metrics distinguish an initially empty window from unloading after
    # a load actually observed in this recorded prefix. No inference uses the
    # potentially accepted but unrecorded tail.
    had_load = stats["first_positive_input_tick"] is not None
    if load > 0:
        stats["final_unloaded_start_input_tick"] = None
    elif had_load and stats["final_unloaded_start_input_tick"] is None:
        stats["final_unloaded_start_input_tick"] = tick
    empty = area["samples"] == 0
    if empty:
        if stats["final_empty_clip_start_input_tick"] is None:
            stats["final_empty_clip_start_input_tick"] = tick
        if had_load:
            if stats["first_empty_clip_after_load_input_tick"] is None:
                stats["first_empty_clip_after_load_input_tick"] = tick
            if stats["final_empty_clip_after_load_start_input_tick"] is None:
                stats["final_empty_clip_after_load_start_input_tick"] = tick
    else:
        stats["final_empty_clip_start_input_tick"] = None
        stats["final_empty_clip_after_load_start_input_tick"] = None
    strict = row["geometry"]["strict_disjoint_certified"]
    if strict and had_load:
        if stats["first_strict_disjoint_after_load_input_tick"] is None:
            stats["first_strict_disjoint_after_load_input_tick"] = tick
        if stats["final_strict_disjoint_after_load_start_input_tick"] is None:
            stats["final_strict_disjoint_after_load_start_input_tick"] = tick
    else:
        stats["final_strict_disjoint_after_load_start_input_tick"] = None


def _accepted_tail(summary, prior):
    """Check summary endpoint identity, never infer an unwritten transition."""
    count, accepted = summary["recorded_pair_steps"], summary["accepted_pair_steps"]
    gap = accepted - count
    _require(gap in (0, 1), "unrecorded accepted pair count")
    _same(set(summary["final_states"]), set(LANES), "summary final lane identities")
    if summary["status"] == "controller_stopped":
        _require(
            gap == 0
            and count >= 1
            and summary.get("controller_status")
            in ("success", "no_progress", "segment_budget", "step_budget")
            and type(summary.get("stopped_after_original_output_tick")) is int
            and summary["stopped_after_original_output_tick"] == summary["start_tick"] + count
            and not any(key in summary for key in ("failure_stage", "errors")),
            "normal controller stop identity",
        )
    elif summary["status"] == "complete":
        _require(gap == 0 and count == summary["requested_steps"], "complete window counts")
    else:
        _require(summary["status"] == "partial_failure", "producer termination status")
    if gap == 0:
        _require(summary.get("unpublished_accepted_pairs", 0) == 0, "unexpected unpublished pair")
        _same(prior, summary["final_states"], "summary final states")
        return None
    _require(
        summary["status"] == "partial_failure"
        and summary.get("failure_stage") == "paired_receipt_and_controller"
        and type(summary.get("unpublished_accepted_pairs")) is int
        and summary["unpublished_accepted_pairs"] == 1
        and type(summary.get("failed_original_input_tick")) is int
        and summary["failed_original_input_tick"] == summary["start_tick"] + count,
        "post-accept callback failure identity",
    )
    for lane in LANES:
        endpoint = summary["final_states"][lane]
        _state(endpoint, accepted)
        _require(
            abs(endpoint["time"] - prior[lane]["time"] - 0.001) <= 2e-6,
            "unrecorded accepted endpoint clock",
        )
    _same(
        summary["final_states"]["open"]["time"],
        summary["final_states"]["feedback"]["time"],
        "unrecorded paired endpoint clock identity",
    )
    return dict(
        input_tick=summary["start_tick"] + count,
        output_tick=summary["start_tick"] + accepted,
        reported_output_states=summary["final_states"],
        summary_endpoint_identity_checked=True,
        transition_verified=False,
        control_force_goal_evidence_verified=False,
    )


def compare(path):
    """Validate saved paired provenance and return JSON-ready aggregate evidence."""
    path = Path(path).resolve()
    summary = _read(path)
    _require(summary["schema"] == "phymeo-pusht-public-feedback-v1", "summary schema")
    start, count = summary["start_tick"], summary["recorded_pair_steps"]
    _require(
        type(start) is int
        and start >= 0
        and type(count) is int
        and count >= 0
        and type(summary["accepted_pair_steps"]) is int
        and type(summary["requested_steps"]) is int
        and summary["requested_steps"] >= 1
        and (count > 0 or summary["accepted_pair_steps"] == 1)
        and count <= summary["accepted_pair_steps"] <= summary["requested_steps"],
        "paired step counts",
    )
    _same(set(summary["worlds"]), set(LANES), "lane identities")
    _require(len(set(summary["worlds"].values())) == 2, "distinct physical lane owners")
    prior = {lane: summary["initial_states"][lane] for lane in LANES}
    for state in prior.values():
        _state(state, 0)
    _same(prior["open"], prior["feedback"], "two lanes must share their one initial state")
    bootstrap = summary["controller_history_bootstrap"]
    _require(
        bootstrap["start_tick"] == start
        and bootstrap["prefix_steps"] == start
        and bootstrap["future_archived_states_used"] is False,
        "controller bootstrap must end at the selected input",
    )
    journal = summary["journal"]
    journal_path = (path.parent / journal["file"]).resolve()
    _require(
        journal["records"] == count and _digest(journal_path) == journal["sha256"],
        "journal digest/count",
    )
    stats = {lane: _load_stats() for lane in LANES}
    differences = {
        name: dict(max_abs=0.0, input_tick=None, nonzero_steps=0)
        for name in ("motor_ctrl", "output_q", "output_v", "object_position", "area_normal_load")
    }
    route_cache, routes, stages = {}, [], []
    previous_route, previous_stage = None, None
    records = 0
    with journal_path.open("rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line, parse_constant=_constant)
            tick = start + index
            _require(
                index < count
                and row["schema"] == "phymeo-pusht-public-feedback-step-v1"
                and row["index"] == index
                and row["original_input_tick"] == tick
                and row["original_output_tick"] == tick + 1,
                "journal order/phase",
            )
            original = row["original_control"]
            _require(
                original["tick"] == tick and original["output_tick"] == tick + 1,
                "original control phase",
            )
            h = float(original["h"])
            _require(h == 0.001, "original timestep")
            original_channels = _channels(original)
            _same(set(row["lanes"]), set(LANES), "complete paired row")
            for lane in LANES:
                item = row["lanes"][lane]
                before, after = item["input_state"], item["output_state"]
                _state(before, index)
                _state(after, index + 1)
                _same(before, prior[lane], lane + " does not continue its own accepted endpoint")
                _require(abs(after["time"] - before["time"] - h) <= 2e-6, "one-step clock")
                controller, command = item["controller"], item["submitted_control"]
                _require(controller["input_tick"] == tick, "control input tick")
                _same(
                    controller["measurement_q"],
                    before["q"],
                    "control q comes from this lane's current state",
                )
                _same(
                    controller["measurement_v"],
                    before["v"],
                    "control v comes from this lane's current state",
                )
                channels = _channels(command)
                _require(
                    item["stored_control_matches_submitted"] is True, "accepted control publication"
                )
                for name in CHANNELS if lane == "open" else CHANNELS[1:]:
                    _same(channels[name], original_channels[name], lane + " original " + name)
                if lane == "feedback":
                    _require(
                        controller["local_tick"] == index
                        and controller["input_time"] == before["time"],
                        "feedback input phase",
                    )
                    _same(
                        _channels(controller["generated_channels"]),
                        channels,
                        "feedback generation/submission binding",
                    )
                    pd = controller["pd"]
                    qa, va, actors = pd["qpos_addresses"], pd["dof_addresses"], pd["actuator_ids"]
                    _same(qa, [0, 1], "RR controlled q addresses")
                    _same(va, [0, 1], "RR controlled v addresses")
                    _same(actors, [0, 1], "RR motor identities")
                    _same(pd["position"], before["q"][:2], "PD current joint position")
                    _same(pd["velocity"], before["v"][:2], "PD current joint velocity")
                    _same(pd["target"], controller["target_q"], "PD reference position")
                    _same(pd["target_velocity"], controller["target_v"], "PD reference velocity")
                    transport = controller["transport"]
                    _require(
                        transport["kind"] == "original-env-fp32-atanh-cuda-tanh"
                        and transport["device"].startswith("cuda")
                        and transport["operator"] == "bias + scale * torch.tanh(raw)"
                        and transport["scale"] == 20.0
                        and transport["bias"] == 0.0,
                        "original Env CUDA transport provenance",
                    )
                    direct = _array(transport["direct_pd_ctrl"], (2,), "direct PD ctrl")
                    _same(
                        np.asarray(pd["effort"], dtype=np.float32).astype(float).tolist(),
                        direct.tolist(),
                        "PD effort before transport",
                    )
                    raw_action = _array(transport["raw_action"], (2,), "raw FP32 action")
                    _require(np.all(abs(direct) < 20.0), "finite original action encoding")
                    encoded = np.arctanh(direct.astype(np.float32) / np.float32(20.0))
                    _require(
                        np.array_equal(raw_action, encoded), "original HOST FP32 atanh encoding"
                    )
                    _same(
                        transport["transported_ctrl"],
                        channels["ctrl"],
                        "actual CUDA transport/publication binding",
                    )
                    _same(
                        (np.asarray(channels["ctrl"]) - direct).tolist(),
                        transport["direct_to_transported_delta"],
                        "direct/transported difference",
                    )
                    # CUDA tanh is not reimplemented with NumPy: audit the saved
                    # actual output through generated/submitted/published data.
                    if index == 0:
                        first = summary["initial_command_comparison"]
                        for flag in (
                            "same_input_q_fp32_bytes",
                            "same_input_v_fp32_bytes",
                            "ctrl_fp32_bytes_equal",
                        ):
                            _require(type(first[flag]) is bool, "initial command comparison flags")
                        _same(
                            first["original_ctrl"],
                            original_channels["ctrl"],
                            "initial original ctrl",
                        )
                        _same(
                            first["regenerated_ctrl"], channels["ctrl"], "initial regenerated ctrl"
                        )
                        _same(
                            first["regenerated_direct_effort"],
                            pd["effort"],
                            "initial regenerated PD effort",
                        )
                        _same(first["regenerated_pd"], pd, "initial regenerated PD diagnostic")
                        _same(
                            first["original_pd"]["effort"],
                            first["original_direct_effort"],
                            "initial original PD diagnostic",
                        )
                        original_direct = _array(
                            first["original_direct_effort"], (2,), "original direct effort"
                        ).astype(np.float32)
                        _require(
                            np.all(abs(original_direct) < 20.0), "original bounded action encoding"
                        )
                        _same(
                            np.arctanh(original_direct / np.float32(20.0)).astype(float).tolist(),
                            first["original_raw_action"],
                            "original archived raw action encoding",
                        )
                        _same(
                            first["regenerated_raw_action"],
                            transport["raw_action"],
                            "initial regenerated raw action",
                        )
                        equal = (
                            np.asarray(first["original_ctrl"], dtype=np.float32).tobytes()
                            == np.asarray(first["regenerated_ctrl"], dtype=np.float32).tobytes()
                        )
                        _require(
                            first["ctrl_fp32_bytes_equal"] == equal,
                            "initial actual ctrl bit comparison",
                        )
                        _require(
                            not (
                                first["same_input_q_fp32_bytes"]
                                and first["same_input_v_fp32_bytes"]
                            )
                            or equal,
                            "same-source initial command differs before feedback divergence",
                        )
                    route = controller["route"]
                    if "route_start_tick" in route:
                        created = route["route_start_tick"]
                        _require(type(created) is int and created <= tick, "future route creation")
                        identity = (created, route["route_file"], route["route_sha256"])
                        if identity not in route_cache:
                            source = (path.parent / route["route_file"]).resolve()
                            _require(_digest(source) == route["route_sha256"], "route file digest")
                            route_cache[identity] = _read(source)
                        plan = route_cache[identity]
                        _require(plan["tick"] == created, "route creation source tick")
                        if identity != previous_route:
                            _require(
                                previous_route is None or created == tick,
                                "new route must be created at the current endpoint",
                            )
                            if created >= start:
                                _require(created == tick, "unrecorded in-window route creation")
                                _same(
                                    plan["q"],
                                    before["q"],
                                    "replanning q must be this lane's accepted endpoint",
                                )
                                _same(
                                    plan["v"],
                                    before["v"],
                                    "replanning v must be this lane's accepted endpoint",
                                )
                            else:
                                origins = [
                                    r
                                    for r in bootstrap["original_routes_used"]
                                    if r["tick"] == created
                                ]
                                _require(
                                    len(origins) == 1, "pre-window route absent from bootstrap"
                                )
                                origin = origins[0]
                                original_path = (path.parent / origin["file"]).resolve()
                                _require(
                                    _digest(original_path) == origin["sha256"],
                                    "original prefix route digest",
                                )
                                original_plan = _read(original_path)
                                # Re-serialization may change gzip timestamp bytes;
                                # bind the actual route and its source state instead.
                                for field in ("tick", "q", "v", "route"):
                                    _same(
                                        plan[field],
                                        original_plan[field],
                                        "historical route payload " + field,
                                    )
                            routes.append(
                                dict(
                                    created_input_tick=created,
                                    first_used_input_tick=tick,
                                    file=identity[1],
                                    sha256=identity[2],
                                    bootstrap=created < start,
                                )
                            )
                            previous_route = identity
                    measurement = item["measurement"]
                    _require(
                        measurement["world"] == summary["worlds"][lane]
                        and measurement["input_tick"] == tick
                        and measurement["output_tick"] == tick + 1
                        and measurement["time_s"] == before["time"],
                        "feedback accepted force source phase",
                    )
                    _same(
                        item["feedback_accepted"]["measurement"],
                        measurement,
                        "controller accepted callback uses this step's saved force",
                    )
                    _require(
                        item["driver_state_after"]["tick"] == tick + 1,
                        "controller advances once after accepted step",
                    )
                    stage = controller["stage"]
                    if stage != previous_stage:
                        stages.append(dict(input_tick=tick, stage=stage))
                        previous_stage = stage
                success, metrics = _goal(item, summary["worlds"][lane], tick, before)
                if success:
                    stats[lane]["goal_success_input_ticks"].append(tick)
                stats[lane]["final_goal_metrics"] = metrics
                _load_record(stats[lane], item, tick, h)
                prior[lane] = after
            left, right = (row["lanes"][lane] for lane in LANES)
            for name, a, b in (
                (
                    "motor_ctrl",
                    left["submitted_control"]["ctrl"],
                    right["submitted_control"]["ctrl"],
                ),
                ("output_q", left["output_state"]["q"], right["output_state"]["q"]),
                ("output_v", left["output_state"]["v"], right["output_state"]["v"]),
                (
                    "object_position",
                    left["output_state"]["object_position_world_m"],
                    right["output_state"]["object_position_world_m"],
                ),
                ("area_normal_load", left["area"]["normal_load_n"], right["area"]["normal_load_n"]),
            ):
                delta = float(np.max(abs(np.asarray(a) - np.asarray(b)), initial=0.0))
                value = differences[name]
                value["nonzero_steps"] += int(delta > 0)
                if delta > value["max_abs"]:
                    value.update(max_abs=delta, input_tick=tick)
            records += 1
    _require(records == count, "journal truncated")
    tail = _accepted_tail(summary, prior)
    if summary["status"] == "controller_stopped":
        last_feedback = row["lanes"]["feedback"]
        _same(
            summary["controller_status"],
            last_feedback["driver_state_after"]["status"],
            "terminal reason must match the last recorded controller",
        )
        _require(
            last_feedback["goal_result"]["success"] == (summary["controller_status"] == "success"),
            "terminal success must match the last original Goal result",
        )
    for lane in LANES:
        result = stats[lane]
        result["goal_achieved_in_saved_window"] = (
            bool(result["goal_success_input_ticks"]) if count else None
        )
        result["last_goal_input_tick"] = start + count - 1 if count else None
        result["normal_load_change_n"] = (
            result["final_normal_load_n"] - result["initial_normal_load_n"] if count else None
        )
        result["object_displacement_world_m"] = (
            np.asarray(prior[lane]["object_position_world_m"])
            - np.asarray(summary["initial_states"][lane]["object_position_world_m"])
        ).tolist()
    return dict(
        schema="phymeo-pusht-feedback-comparison-v1",
        status="verified-saved-prefix" if count else "no-recorded-pairs",
        source=dict(
            summary_file=str(path),
            summary_sha256=_digest(path),
            journal_sha256=journal["sha256"],
            producer_status=summary["status"],
        ),
        start_input_tick=start,
        last_input_tick=start + count - 1 if count else None,
        accepted_pair_steps=summary["accepted_pair_steps"],
        recorded_pair_steps=count,
        verified_pair_steps=count,
        verification_scope="recorded-prefix-only",
        verified_final_states=prior,
        unrecorded_accepted_pair=tail,
        controller_status=summary.get("controller_status"),
        stopped_after_original_output_tick=summary.get("stopped_after_original_output_tick"),
        requested_steps=summary["requested_steps"],
        complete_requested_window=count == summary["requested_steps"],
        same_initial_states=True,
        same_external_channels=True,
        each_lane_state_continuous=True,
        initial_command_comparison=summary.get("initial_command_comparison"),
        initial_command_comparison_verified=bool(count),
        commands_bound_to_current_lane=bool(count),
        feedback_routes=routes,
        feedback_stage_transitions=stages,
        lanes=stats,
        differences=differences,
        limits=[
            "All control, force, Goal and motion aggregates cover recorded pairs only; an unrecorded accepted tail is not audited.",
            "Saved numerical differences are observations, not significance thresholds.",
            "Goal results are the recorded original evaluator outputs; no independent Goal or physics replay.",
            "CUDA action transport output is bound to its saved publication, not recomputed with NumPy tanh.",
            "Zero area force, empty clipping and certified sampled separation remain distinct.",
            "Bootstrap history is checked through its declared boundary and route files, not re-executed.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="paired run summary JSON")
    parser.add_argument("--output", type=Path, required=True, help="new comparison JSON")
    args = parser.parse_args()
    result = compare(args.input)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
