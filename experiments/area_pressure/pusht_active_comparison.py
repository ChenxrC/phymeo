"""Read-only audit of saved public PushT active-route frames; no physics replay.

The recorded receipt witnesses and files are cross-bound, not recreated. Goal
booleans remain the original evaluator's saved results. Independent endpoint
errors and sampled Euler substitution do not constitute a new Goal/solver audit.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

H = 0.001
EPS = np.finfo(np.float32).eps
CHANNELS = ("ctrl", "qfrc_applied", "xfrc_applied")


def _require(ok, detail):
    if not ok:
        raise ValueError("PushT active comparison: " + detail)


def _constant(value):
    raise ValueError("PushT active comparison: nonfinite JSON " + value)


def _read(path):
    def nonfinite(value):
        raise ValueError("PushT active comparison: nonfinite JSON " + value)

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        return json.load(stream, parse_constant=nonfinite)


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _bound_file(base, name, sha, detail):
    path = (base / name).resolve()
    _require(_digest(path) == sha, detail + " source digest")
    return path, _read(path)


def _array(value, shape, detail):
    a = np.asarray(value, dtype=float)
    _require(a.shape == shape and np.isfinite(a).all(), detail)
    return a


def _close(actual, expected, detail, *, atol=1e-12, rtol=1e-12):
    _require(np.allclose(actual, expected, atol=atol, rtol=rtol), detail)


def _state(value, tick):
    _require(type(value["local_tick"]) is int and value["local_tick"] == tick, "state local tick")
    q, v = _array(value["q"], (9,), "state q"), _array(value["v"], (8,), "state v")
    _require(np.isfinite(value["time"]) and value["time"] >= 0, "state time")
    _require(
        np.array_equal(q, q.astype(np.float32)) and np.array_equal(v, v.astype(np.float32)),
        "published FP32 state values",
    )
    _require(
        value["object_position_world_m"] == value["q"][2:5]
        and value["object_quaternion_wxyz"] == value["q"][5:9],
        "object/state binding",
    )
    _require(abs(np.linalg.norm(q[5:9]) - 1) <= 2e-6, "state quaternion")
    return q, v


def _channels(command):
    _array(command["ctrl"], (2,), "two RR motor channels")
    _require(
        np.array_equal(command["qfrc_applied"], np.zeros(8)), "zero external generalized force"
    )
    _require(np.array_equal(command["xfrc_applied"], np.zeros((5, 6))), "zero external body force")
    return {key: command[key] for key in CHANNELS}


def _reference(nodes, elapsed):
    _require(len(nodes) >= 2 and nodes[0][0] == 0, "route starts at zero")
    for i, node in enumerate(nodes):
        _array(node[2], (2,), "route joint target")
        _require(np.isfinite(node[0]) and type(node[1]) is str, "route node time/stage")
        if i:
            _require(node[0] > nodes[i - 1][0], "ordered route nodes")
    for left, right in zip(nodes, nodes[1:]):
        if elapsed <= right[0]:
            u = min(1.0, max(0.0, (elapsed - left[0]) / (right[0] - left[0])))
            delta = np.asarray(right[2]) - left[2]
            return (
                np.asarray(left[2]) + u * u * (3 - 2 * u) * delta,
                6 * u * (1 - u) / (right[0] - left[0]) * delta,
                right[1],
            )
    return np.asarray(nodes[-1][2]), np.zeros(2), nodes[-1][1]


def _integration(before, after):
    q, v = np.asarray(before["q"]), np.asarray(after["v"])
    actual = np.asarray(after["q"])
    expected = q.copy()
    expected[:5] += H * v[:5]
    # Same RR scalar/FREE Euler contract as the accepted receipt validator,
    # evaluated independently in FP64 with its existing 16*eps32 bounds.
    w, xyz = q[5] / np.linalg.norm(q[5:9]), q[6:9] / np.linalg.norm(q[5:9])
    omega = v[5:8]
    speed = float(np.linalg.norm(omega))
    angle = 0.5 * H * speed
    dw = math.cos(angle)
    dxyz = omega * (math.sin(angle) / speed if speed else 0.0)
    rotation = np.r_[w * dw - xyz @ dxyz, w * dxyz + dw * xyz + np.cross(xyz, dxyz)]
    expected[5:9] = rotation / np.linalg.norm(rotation)
    bound = np.r_[
        16 * EPS * (abs(q[:5]) + H * abs(v[:5])) + np.finfo(np.float32).tiny, np.full(4, 16 * EPS)
    ]
    delta = abs(actual - expected)
    _require(np.all(delta <= bound), "saved scalar/FREE Euler integration")
    clock_error = abs(after["time"] - before["time"] - H)
    clock_bound = 16 * EPS * H + 64 * np.finfo(float).eps * max(1.0, abs(before["time"]))
    _require(clock_error <= clock_bound, "one-step compensated clock")
    return float(delta.max()), float(clock_error)


def _pose_errors(state, goal):
    q = np.asarray(state["q"], dtype=float)
    w, x, y, z = q[5:9] / np.linalg.norm(q[5:9])
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    signed = math.atan2(math.sin(goal["target_yaw"] - yaw), math.cos(goal["target_yaw"] - yaw))
    return dict(
        position_error_m=float(np.linalg.norm(np.asarray(goal["target_xy"]) - q[2:4])),
        yaw_rad=yaw,
        signed_yaw_error_rad=signed,
        yaw_error_rad=abs(signed),
    )


def _first(binding, controller, state, tick, route, target, velocity):
    _require(
        binding["input_tick"] == tick
        and binding["input_time"] == state["time"]
        and binding["measurement_q"] == state["q"]
        and binding["measurement_v"] == state["v"],
        "first command source state/phase",
    )
    _require(binding["route_start_tick"] == route["start_tick"], "first route start")
    _close(binding["route_elapsed_s"], (tick - route["start_tick"]) * H, "first route phase")
    for name, expected in (
        ("target_q", target),
        ("target_v", velocity),
        ("independently_recomputed_target_q", target),
        ("independently_recomputed_target_v", velocity),
    ):
        _close(binding[name], expected, "first command " + name)
    _require(binding["exact_matches"] is True, "producer first command binding")
    if "independently_recomputed_direct_ctrl" in binding:
        _require(
            binding["independently_recomputed_direct_ctrl"]
            == controller["transport"]["direct_pd_ctrl"],
            "first recomputed direct ctrl",
        )
    if "submitted_ctrl" in binding:
        _require(
            binding["submitted_ctrl"] == controller["generated_channels"]["ctrl"],
            "first submitted ctrl",
        )


class _LoadCompensationAudit:
    """One-step force feedback with independent current-state RR kinematics."""

    def __init__(self, summary, source, route_file):
        import mujoco as mj

        from phymeo.training.manipulation_tasks import manipulation_scenes

        trials = source.get("trials", [])
        _require(type(trials) is list, "compensation release trial list")
        selected = [row for row in trials if row.get("selected") is True]
        _require(bool(selected), "compensation source has no selected release step")
        seed = selected[-1]
        initial = summary["initial_state"]
        _require(
            seed.get("restored") is False
            and seed["original_input_tick"] == summary["start_tick"] - 1
            and seed["original_output_tick"] == summary["start_tick"]
            and seed["input_state"]["local_tick"] == initial["local_tick"] - 1
            and seed["output_state"] == initial,
            "compensation last selected release endpoint",
        )
        self.previous_state = initial
        self.previous_force = _array(
            seed["area"]["total_force_on_pusher_world_n"], (3,), "compensation source tool force"
        )
        self.direction = _array(
            route_file["route"]["proposal"]["direction_world"],
            (3,),
            "compensation route push direction",
        )
        _require(
            abs(np.linalg.norm(self.direction) - 1.0) <= 1e-12 and abs(self.direction[2]) <= 1e-12,
            "compensation unit horizontal route direction",
        )
        scene = next(scene for scene in manipulation_scenes() if scene.key == "pusht-planar")
        self.mj, self.model = mj, mj.MjModel.from_xml_string(scene.mjcf)
        self.data = mj.MjData(self.model)
        self.dofs = [
            int(self.model.jnt_dofadr[self.model.joint(name).id])
            for name in scene.robot_joint_names
        ]
        self.geom = int(self.model.geom("pusher").id)
        _require(
            tuple(scene.robot_kinds) == ("R", "R")
            and self.model.nq == 9
            and self.model.nv == 8
            and self.dofs == [0, 1],
            "compensation original RR model identities",
        )

    def effort(self, row, raw):
        state, controller = row["input_state"], row["controller"]
        c = controller.get("load_compensation")
        _require(type(c) is dict, "compensation per-step evidence missing")
        _require(
            type(c["source_local_tick"]) is int
            and c["source_local_tick"] == state["local_tick"]
            and c["source_output_time"] == state["time"]
            and state == self.previous_state,
            "compensation previous accepted output phase",
        )
        force = _array(c["previous_force_on_tool_world_n"], (3,), "compensation saved tool force")
        _require(
            np.array_equal(force, self.previous_force), "compensation previous actual tool force"
        )
        direction = _array(c["push_direction_world"], (3,), "compensation saved direction")
        _close(direction, self.direction, "compensation selected route direction")
        _require(c["gain"] == 0.5, "compensation half gain")
        enabled = controller["stage"] in ("entry", "push")
        _require(
            type(c["enabled"]) is bool and c["enabled"] == enabled, "compensation stage enable"
        )
        resisting = max(0.0, -float(self.previous_force @ self.direction))
        _close(c["resisting_force_n"], resisting, "compensation resisting force")
        # Only kinematics/computation of a point Jacobian: no CPU forward solve
        # or step, and no production feedback helper is reused by this reader.
        self.data.qpos[:] = state["q"]
        self.mj.mj_kinematics(self.model, self.data)
        self.mj.mj_comPos(self.model, self.data)
        jacp = np.zeros((3, self.model.nv))
        self.mj.mj_jacGeom(self.model, self.data, jacp, None, self.geom)
        jacobian = jacp[:, self.dofs]
        _close(
            _array(c["jacobian_world"], (3, 2), "compensation Jacobian dimensions"),
            jacobian,
            "compensation current pusher Jacobian",
        )
        correction = jacobian.T @ (0.5 * resisting * self.direction) if enabled else np.zeros(2)
        _close(
            _array(c["correction_nm"], (2,), "compensation correction dimensions"),
            correction,
            "compensation force/Jacobian torque mapping",
        )
        combined = raw + correction
        _close(
            _array(c["combined_unclipped_nm"], (2,), "compensation combined dimensions"),
            combined,
            "compensation adds to unclipped PD",
        )
        effort = _array(c["combined_effort_nm"], (2,), "compensation effort dimensions")
        _close(effort, np.clip(combined, -19.9, 19.9), "compensation combined effort clipping")
        return effort

    def accepted(self, row):
        # Called after the row's receipt/load checks, so rejected or future
        # candidate forces cannot be consumed as feedback for the next command.
        self.previous_force = _array(
            row["area"]["total_force_on_pusher_world_n"], (3,), "compensation accepted tool force"
        )
        self.previous_state = row["output_state"]


def _control(row, summary, index, base, compensation=None):
    tick, route = summary["start_tick"] + index, summary["active_route"]
    state, c = row["input_state"], row["controller"]
    _require(
        c["input_tick"] == tick
        and c["local_tick"] == summary["initial_state"]["local_tick"] + index
        and c["input_time"] == state["time"]
        and c["measurement_q"] == state["q"]
        and c["measurement_v"] == state["v"],
        "controller current accepted input phase",
    )
    info = c["route"]
    _require(
        info["route"] == route["route_index"]
        and info["route_start_tick"] == route["start_tick"]
        and info["route_sha256"] == route["sha256"]
        and (base / info["route_file"]).resolve() == (base / route["file"]).resolve(),
        "selected route identity changed",
    )
    elapsed = (tick - route["start_tick"]) * H
    _close(info["route_elapsed_s"], elapsed, "selected route elapsed phase")
    target, velocity, stage = _reference(route["nodes"], elapsed)
    _close(c["target_q"], target, "independent route position reference")
    _close(c["target_v"], velocity, "independent route velocity reference")
    _require(c["stage"] == stage and c["goal"] == summary["goal"], "reference stage/unchanged goal")
    command = row["submitted_control"]
    _require(
        command["tick"] == tick
        and command["output_tick"] == tick + 1
        and row["stored_control_matches_submitted"] is True,
        "accepted control publication",
    )
    channels = _channels(command)
    _require(_channels(c["generated_channels"]) == channels, "generated/submitted control binding")
    pd = c["pd"]
    for name in ("qpos_addresses", "dof_addresses", "actuator_ids"):
        _require(pd[name] == [0, 1], "original RR PD identities")
    _require(
        pd["position"] == state["q"][:2] and pd["velocity"] == state["v"][:2],
        "PD current position/velocity",
    )
    _close(pd["target"], target, "PD target")
    _close(pd["target_velocity"], velocity, "PD target velocity")
    bias = _array(pd["bias"], (2,), "saved PD bias")
    raw = bias + 25 * (target - state["q"][:2]) + 3 * (velocity - state["v"][:2])
    _close(pd["unclipped_effort"], raw, "native PD arithmetic")
    _close(pd["effort"], np.clip(raw, -19.9, 19.9), "native PD effort clipping")
    transport = c["transport"]
    _require(
        transport["kind"] == "original-env-fp32-atanh-cuda-tanh"
        and transport["device"].startswith("cuda")
        and transport["scale"] == 20.0
        and transport["bias"] == 0.0
        and transport["operator"] == "bias + scale * torch.tanh(raw)",
        "CUDA action transport",
    )
    if compensation is None:
        _require("load_compensation" not in c, "compensation evidence requires explicit variant")
        effort = pd["effort"]
    else:
        effort = compensation.effort(row, raw)
    # The transport's historical direct_pd_ctrl name contains the actual
    # combined effort for this variant; pd.effort remains the original PD value.
    direct = np.asarray(effort, np.float32)
    _require(np.array_equal(transport["direct_pd_ctrl"], direct), "PD/direct FP32 effort")
    _require(
        np.array_equal(transport["raw_action"], np.arctanh(direct / np.float32(20))),
        "HOST FP32 action encoding",
    )
    _require(transport["transported_ctrl"] == channels["ctrl"], "CUDA decoded/submitted ctrl")
    _close(
        transport["direct_to_transported_delta"],
        np.asarray(channels["ctrl"]) - direct,
        "transport difference",
        atol=0,
        rtol=0,
    )
    if index == 0:
        _first(summary["first_command_binding"], c, state, tick, route, target, velocity)
    return stage


def _goal(row, summary, tick):
    state, frame, result = row["input_state"], row["goal_frame"], row["goal_result"]
    _require(
        frame["phase"] == "accepted-entry"
        and frame["world"] == summary["world"]
        and frame["tick"] == tick
        and frame["time_s"] == state["time"]
        and frame["object_position"] == state["q"][2:5]
        and frame["object_wxyz"] == state["q"][5:9],
        "Goal accepted-entry phase",
    )
    _require(
        type(result["success"]) is bool
        and result["history"]["tick"] == tick
        and result["history"]["time_s"] == state["time"],
        "Goal history phase",
    )
    errors, metrics = _pose_errors(state, summary["goal"]), dict(result["metrics"])
    _require(all(np.isfinite(x) for x in metrics.values()), "finite saved Goal metrics")
    for name in ("position_error_m", "yaw_error_rad"):
        _close(metrics[name], errors[name], "Goal pose metric " + name, atol=2e-6, rtol=2e-6)
    measurement = row["feedback_accepted"]["measurement"]
    _require(
        measurement["world"] == summary["world"]
        and measurement["input_tick"] == tick
        and measurement["output_tick"] == tick + 1
        and measurement["time_s"] == state["time"],
        "accepted force callback phase",
    )
    expected_completed = summary["active_route"]["completed_routes_before"] + int(
        tick + 1 == summary["active_route"]["end_tick"]
    )
    _require(
        row["driver_state_after"]["tick"] == tick + 1
        and row["driver_state_after"]["completed_routes"] == expected_completed,
        "controller advances once without completing another route",
    )
    return result["success"]


def _receipt(row):
    solver = row["solver"]
    meta = _array(solver["meta"], (4,), "solver meta")
    _require(
        solver["schema"] == "mujoco-tyx-area-friction-resident-native-v1"
        and meta[0] == 0
        and meta[3] == 1
        and meta[2] == meta[1] + 1
        and 1 <= meta[2] <= 25
        and meta[2] == int(meta[2]),
        "converged resident receipt",
    )
    _require(
        solver["native_solve_calls"] == meta[2]
        and solver["outer_iterations"] == meta[1]
        and np.isfinite(solver["normalized_residual"])
        and 0 <= solver["normalized_residual"] <= 2e-5,
        "accepted resident solver gate",
    )
    _require(
        solver["output_velocity"] == row["output_state"]["v"], "solver/output velocity binding"
    )


def _checkpoint_source(base, summary, source, origin):
    provenance = summary["provenance"]
    path, metadata = _bound_file(
        base,
        provenance["checkpoint_metadata_path"],
        provenance["checkpoint_metadata_sha256"],
        "checkpoint metadata",
    )
    _require(
        metadata["schema"] == "phymeo-pusht-reconstructed-scene-v1"
        and metadata["world"] == summary["world"]
        and metadata["release_sha256"] == provenance["release_report_sha256"]
        and metadata["state"] == source["final_state"]
        and metadata["source_tick"] == summary["start_tick"]
        and metadata["asset_sha256"] == provenance["scene_asset_sha256"],
        "checkpoint source metadata",
    )
    leaf = Path(metadata["capsule"])
    _require(
        leaf.name == metadata["capsule"] and not leaf.is_absolute(), "checkpoint sibling capsule"
    )
    capsule = path.parent / leaf
    _require(
        capsule.stat().st_size == metadata["size"] and _digest(capsule) == metadata["sha256"],
        "checkpoint capsule bytes",
    )
    proof = summary["checkpoint_restore"]
    _require(
        proof["checkpoint_sha256"] == metadata["sha256"]
        and proof["checked_prefix_steps"] == metadata["checked_steps"]
        and proof["restored_local_tick"] == origin
        and proof["absolute_input_tick"] == summary["start_tick"]
        and proof["time"] == summary["initial_state"]["time"],
        "reported checkpoint restore binding",
    )
    return metadata["sha256"]


def _final_checkpoint(base, summary):
    saved = summary.get("final_checkpoint")
    if saved is None:
        return None
    metadata_path, metadata = _bound_file(
        base, saved["metadata_file"], saved["metadata_sha256"], "terminal checkpoint metadata"
    )
    _require(
        metadata["schema"] == saved["schema"] == "phymeo-pusht-active-terminal-checkpoint-v1",
        "terminal checkpoint schema",
    )
    capsule = (base / saved["file"]).resolve()
    _require(
        capsule == (metadata_path.parent / metadata["checkpoint_file"]).resolve()
        and _digest(capsule) == saved["sha256"] == metadata["checkpoint_sha256"],
        "terminal checkpoint bytes",
    )
    for key in (
        "source_tick",
        "local_tick",
        "time",
        "world",
        "controller_at_physical_endpoint",
        "physical_state_matches_report",
    ):
        _require(saved[key] == metadata[key], "terminal checkpoint summary " + key)
    _require(
        metadata["state"] == summary["final_state"]
        and metadata["q"] == metadata["state"]["q"]
        and metadata["v"] == metadata["state"]["v"]
        and metadata["source_tick"] == summary["start_tick"] + summary["accepted_steps"]
        and metadata["journal"] == summary["journal"]
        and metadata["accepted_steps"] == summary["accepted_steps"]
        and metadata["recorded_steps"] == summary["recorded_steps"],
        "terminal checkpoint endpoint/journal",
    )
    return dict(
        sha256=saved["sha256"],
        source_tick=saved["source_tick"],
        capsule_bytes_verified=True,
        restored_by_reader=False,
        controller_at_physical_endpoint=saved["controller_at_physical_endpoint"],
    )


def _load(row):
    area = row["area"]
    values = np.asarray(area["raw_normal_force_n"], dtype=float)
    _require(
        values.ndim == 1 and np.isfinite(values).all() and np.all(values >= 0),
        "actual nonnegative per-sample normal loads",
    )
    _require(
        type(area["samples"]) is int
        and area["samples"] == len(values)
        and type(area["active_samples"]) is int
        and area["active_samples"] == np.count_nonzero(values > 0),
        "actual load sample counts",
    )
    total = math.fsum(values.tolist())
    _require(
        np.isfinite(area["normal_load_n"]) and area["normal_load_n"] >= 0,
        "reported nonnegative normal load",
    )
    _close(
        area["normal_load_n"], total, "raw force/summarized normal load", atol=1e-12, rtol=8 * EPS
    )
    _require(
        type(area["zero_normal_load"]) is bool and area["zero_normal_load"] == (total == 0),
        "actual zero normal load",
    )
    if row["geometry"]["strict_disjoint_certified"]:
        _require(len(values) == 0, "certified disjoint geometry carries samples")
    native = row["native"]
    support = math.fsum(
        float(item["normal_n"])
        for item in native["supports"]
        if any(name in ("object-0", "object-1") for name in item["geometry_names"])
    )
    _require(np.isfinite(support) and support >= 0, "native object support")
    _close(native["object_support_normal_n"], support, "native support decomposition", rtol=8 * EPS)
    if "total_force_on_pusher_world_n" in area:
        _close(
            row["feedback_accepted"]["measurement"]["group_force_world"],
            -np.asarray(area["total_force_on_pusher_world_n"]),
            "actual object/tool force sides",
            atol=1e-7,
            rtol=8 * EPS,
        )
    return total, support


def compare(path):
    """Cross-check one saved active route and return a compact JSON-ready report."""
    path = Path(path).resolve()
    summary, base = _read(path), path.parent
    _require(summary["schema"] == "phymeo-pusht-active-route-v1", "summary schema")
    variant = summary.get("control_variant", "position")
    _require(variant in ("position", "load-compensation-half"), "supported control variant")
    count, accepted = summary["recorded_steps"], summary["accepted_steps"]
    _require(
        type(count) is int
        and type(accepted) is int
        and 0 <= count <= accepted
        and accepted - count in (0, 1),
        "accepted/recorded counts",
    )
    provenance = summary["provenance"]
    _, source = _bound_file(
        base, provenance["release_report_path"], provenance["release_report_sha256"], "release"
    )
    _bound_file(base, provenance["prior_summary_path"], provenance["prior_summary_sha256"], "prior")
    _require(
        source["schema"] == "phymeo-pusht-loaded-release-v1"
        and source["provenance"]["source_summary_sha256"] == provenance["prior_summary_sha256"],
        "release/prior source chain",
    )
    start, route = summary["start_tick"], summary["active_route"]
    _require(
        type(start) is int
        and start >= 0
        and source["controller_final"]["tick"] == start
        and source["start_tick"] + source["selected_steps"] == start,
        "source absolute tick",
    )
    initial = summary["initial_state"]
    origin = initial["local_tick"]
    _state(initial, origin)
    _require(
        summary["local_tick_origin"] == origin
        and initial == source["final_state"]
        and summary["world"] == source["world"],
        "complete checkpoint source endpoint",
    )
    checkpoint_sha = _checkpoint_source(base, summary, source, origin)
    _, route_file = _bound_file(base, route["file"], route["sha256"], "active route")
    _require(
        route_file["tick"] == route["start_tick"]
        and route_file["route"]["nodes"] == route["nodes"]
        and route_file["route"]["status"] == "sampled_reference_ready",
        "saved ordinary route",
    )
    _require(
        route["start_tick"] <= start < route["end_tick"]
        and route["end_tick"] - start == summary["requested_steps"] == route["remaining_steps"]
        and accepted <= summary["requested_steps"],
        "active route remaining interval",
    )
    _close(route["duration_s"], route["nodes"][-1][0], "route duration")
    _close(route["duration_s"], (route["end_tick"] - route["start_tick"]) * H, "route end tick")
    _close(route["elapsed_at_entry_s"], (start - route["start_tick"]) * H, "source route phase")
    compensation = (
        _LoadCompensationAudit(summary, source, route_file)
        if variant == "load-compensation-half"
        else None
    )
    goal = summary["goal"]
    _array(goal["target_xy"], (2,), "goal target")
    _require(
        np.isfinite(goal["target_yaw"])
        and goal["position_tolerance_m"] > 0
        and goal["yaw_tolerance_rad"] > 0,
        "goal yaw/tolerances",
    )
    journal = summary["journal"]
    journal_path = (base / journal["file"]).resolve()
    _require(
        journal["records"] == count and _digest(journal_path) == journal["sha256"],
        "journal digest/count",
    )
    prior, records, peak, impulse, episodes, stages = initial, 0, 0.0, 0.0, [], []
    peak_tick, success_ticks, min_support = None, [], None
    max_q_error = max_clock_error = 0.0
    last_loaded = False
    with journal_path.open("rt", encoding="utf-8-sig") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line, parse_constant=_constant)
            tick = start + index
            _require(
                index < count
                and row["schema"] == "phymeo-pusht-active-route-step-v1"
                and row["index"] == index
                and row["original_input_tick"] == tick
                and row["original_output_tick"] == tick + 1,
                "journal tick order",
            )
            before, after = row["input_state"], row["output_state"]
            _state(before, origin + index)
            _state(after, origin + index + 1)
            _require(before == prior, "continuous accepted q/v/time chain")
            qe, te = _integration(before, after)
            max_q_error, max_clock_error = max(max_q_error, qe), max(max_clock_error, te)
            stage = _control(row, summary, index, base, compensation)
            if not stages or stages[-1]["stage"] != stage:
                stages.append(dict(input_tick=tick, stage=stage))
            if _goal(row, summary, tick):
                success_ticks.append(tick)
            _receipt(row)
            load, support = _load(row)
            if compensation is not None:
                compensation.accepted(row)
            min_support = support if min_support is None else min(min_support, support)
            impulse += H * load
            if load > peak:
                peak, peak_tick = load, tick
            if load > 0:
                if not last_loaded:
                    episodes.append(
                        dict(
                            start_input_tick=tick,
                            end_input_tick=tick,
                            peak_normal_load_n=load,
                            normal_impulse_ns=0.0,
                        )
                    )
                episodes[-1]["end_input_tick"] = tick
                episodes[-1]["normal_impulse_ns"] += H * load
                episodes[-1]["peak_normal_load_n"] = max(episodes[-1]["peak_normal_load_n"], load)
            last_loaded, prior, records = load > 0, after, records + 1
    _require(records == count, "truncated journal")
    tail = None
    if accepted > count:
        _require(
            summary["status"] == "partial_failure", "unrecorded accepted tail requires failure"
        )
        saved = summary["unrecorded_accepted_step"]
        _require(saved["input_state"] == prior, "unrecorded tail input identity")
        _state(saved["output_state"], origin + accepted)
        _require(
            saved["output_state"] == summary["final_state"], "unrecorded tail endpoint identity"
        )
        tail = dict(
            input_tick=start + count,
            output_tick=start + accepted,
            transition_verified=False,
            control_force_goal_evidence_verified=False,
            reported_output_state=saved["output_state"],
        )
    else:
        _require(prior == summary["final_state"], "summary final accepted endpoint")
    status = summary["status"]
    _require(
        status in ("running", "route_complete", "controller_stopped", "partial_failure"),
        "producer termination status",
    )
    if status == "route_complete":
        _require(
            summary["controller_final"]["tick"] == start + accepted
            and accepted == count == summary["requested_steps"]
            and summary["controller_final"]["completed_routes"]
            == route["completed_routes_before"] + 1,
            "actual selected route completion boundary",
        )
        classification = (
            "route_complete_with_goal" if success_ticks else "route_complete_without_goal"
        )
    elif status == "controller_stopped":
        reason = summary["controller_status"]
        _require(type(reason) is str and reason, "controller stop reason")
        _require(reason != "success" or bool(success_ticks), "controller success lacks saved Goal")
        classification = "goal_success" if reason == "success" else "controller_stopped"
    elif status == "partial_failure":
        classification = "post_accept_callback_failure" if tail else "physical_or_reference_failure"
    else:
        classification = "in_progress"
    first_errors, last_errors = _pose_errors(initial, goal), _pose_errors(prior, goal)
    displacement = np.asarray(prior["q"])[2:5] - np.asarray(initial["q"])[2:5]
    direction = _array(
        route_file["route"]["proposal"]["direction_world"], (3,), "planned push direction"
    )
    _require(np.linalg.norm(direction[:2]) > 0, "nonzero planar push direction")
    return dict(
        schema="phymeo-pusht-active-comparison-v1",
        status="verified-saved-prefix",
        source_endpoint_bound=True,
        source_checkpoint_sha256=checkpoint_sha,
        final_checkpoint=_final_checkpoint(base, summary),
        verified_steps=count,
        accepted_steps=accepted,
        source=dict(
            summary_file=str(path), summary_sha256=_digest(path), journal_sha256=journal["sha256"]
        ),
        termination=dict(
            classification=classification,
            producer_status=status,
            failure_stage=summary.get("failure_stage"),
            controller_status=summary.get("controller_status"),
        ),
        unrecorded_accepted_step=tail,
        verified_final_state=prior,
        planning=dict(
            selected_route_index=route["route_index"],
            route_start_tick=route["start_tick"],
            route_end_tick=route["end_tick"],
            saved_route_admitted=True,
            stages=stages,
        ),
        control=dict(
            variant=variant,
            load_compensation_verified_steps=count if compensation is not None else 0,
            reference_phase_verified_steps=count,
            current_state_bound_steps=count,
            transport="saved CUDA output/publication binding; no NumPy tanh substitute",
        ),
        integration=dict(
            verified_steps=count,
            max_q_substitution_error=max_q_error,
            max_clock_step_error_s=max_clock_error,
            bound="existing 16*FP32-epsilon scalar/FREE receipt bound",
        ),
        contact=dict(
            source_prior_release_reported=source.get("release_monitor_status") == "released",
            renewed_contact_observed=bool(episodes),
            first_positive_input_tick=episodes[0]["start_input_tick"] if episodes else None,
            peak_normal_load_n=peak,
            peak_input_tick=peak_tick,
            normal_impulse_ns=impulse,
            minimum_object_support_n=min_support,
            episodes=episodes,
        ),
        object=dict(
            initial=first_errors,
            final=last_errors,
            displacement_world_m=displacement.tolist(),
            position_error_change_m=last_errors["position_error_m"]
            - first_errors["position_error_m"],
            yaw_error_change_rad=last_errors["yaw_error_rad"] - first_errors["yaw_error_rad"],
            motion_along_planned_push_m=float(
                displacement[:2] @ direction[:2] / np.linalg.norm(direction[:2])
            ),
        ),
        goal=dict(
            ever_success=bool(success_ticks),
            success_input_ticks=success_ticks,
            last_input_tick=start + count - 1 if count else None,
        ),
        limits=[
            "Recorded-prefix evidence only; unrecorded accepted tails are excluded from force and Goal metrics.",
            "Raw saved normal forces are checked against counts and sums, not recomputed by physics.",
            "Checkpoint bytes and restore metadata are bound; this reader does not restore or replay them.",
            "Reference completion, renewed contact and original Goal success are separate outcomes.",
            "Prior release is source-reported; sampled separation is not continuous-time clearance.",
            "Endpoint pose errors do not evaluate the original Goal at the final output state.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.input)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
