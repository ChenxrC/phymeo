"""Paired public CUDA Scene replay and original PushT goal/PD feedback.

python -m experiments.area_pressure.pusht_public_feedback --trajectory ORIGINAL_DIR
    --source ORIGINAL_CANDIDATES.json --output NEW.json --start 1699 --steps 69

The exclusive NEW.steps.jsonl journal contains only complete paired frames.
Original controller/Goal history before start is replayed without physics. During
the new window each world advances only by Scene.step; feedback never receives
future archived states or routes. CPU queries/PD do not advance physical state.
"""

import argparse
import gzip
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import mujoco as mj
import numpy as np

from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
from phymeo.control.manipulation_push_objective import (
    NativePlanarPushProposal,
    complete_saved_push_proposal,
)
from phymeo.control.manipulation_push_planner import NativePushRoute, complete_saved_push_route
from phymeo.control.manipulation_wrench import (
    ObjectContactWrenchMeasurement,
    ObjectContactWrenchRecord,
)
from phymeo.training.manipulation_tasks import SuccessFrame, evaluate_goal
from scripts.run_manipulation_control import plain
from scripts.run_manipulation_push import SOURCE_SHA, tuple_values
from scripts.run_manipulation_push_feedback import PushFeedbackDriver, hold_plan

from .pusht_entry_rows import (
    _archive_context,
    _array,
    _digest,
    _read_json,
    _reject_constant,
    iter_controls,
)
from .pusht_public_scene import (
    CHANNELS,
    IDENTITY,
    MU,
    SLIP,
    H,
    _error_chain,
    _payload_hash,
    _state,
    _step_record,
    _write_progress,
    build_scene_spec,
    build_side_asset,
    control_from_record,
    legal_initial_q,
)
from .pusht_rejected_candidate import capture_rejected_candidate

MAX_STEPS = 2048
WORLDS = {"open": 5, "feedback": 6}


def _require(ok, message):
    if not ok:
        raise ValueError("public feedback: " + message)


def _accepted_controller_stop(*, stopped, reason, monitor_status, accepted_steps):
    """Classify an already recorded step without treating planner errors as stops."""
    _require(
        type(stopped) is bool and type(accepted_steps) is int and accepted_steps > 0,
        "terminal classification requires an accepted paired step",
    )
    if not stopped:
        _require(
            reason == monitor_status == "continue",
            "running controller and progress status disagree",
        )
        return None
    _require(
        reason in ("success", "no_progress", "segment_budget", "step_budget")
        and monitor_status == reason,
        "accepted stop is not an original progress-monitor termination",
    )
    return reason


def command_from_snapshot(driver, pd, absolute_tick, snapshot, *, start_tick=0):
    """Use this controller's accepted q/v; return direct native FP32 motor effort.

    The CLI supplies the same original external-force channels to both worlds.
    This helper's external channels start at zero and never hide archived state.
    """
    _require(
        type(absolute_tick) is int and absolute_tick == start_tick + snapshot.tick,
        "absolute/local tick mismatch",
    )
    _require(driver.tick == absolute_tick, "controller is not at this accepted tick")
    first = absolute_tick == 0 and not driver.monitor.started
    _require(
        first or np.array_equal(driver.last_q, snapshot.q),
        "controller belongs to another accepted q",
    )
    q = _array(snapshot.q, (driver.model.nq,), "controller q")
    v = _array(snapshot.v, (driver.model.nv,), "controller v")
    target, target_v, stage, route = driver.reference(absolute_tick, q, v)
    _require(not driver.stopped, "original feedback stopped: " + driver.reason)
    _require(np.array_equal(driver.last_q, q), "reference changed accepted q binding")
    ctrl, diagnostic = pd.command(q, v, target, target_v)
    ctrl = np.asarray(ctrl, np.float32).astype(np.float64)
    channels = dict(
        ctrl=ctrl,
        qfrc_applied=np.zeros(driver.model.nv),
        xfrc_applied=np.zeros((driver.model.nbody, 6)),
    )
    record = dict(
        mode="current-accepted-state-feedback",
        input_tick=absolute_tick,
        local_tick=snapshot.tick,
        input_time=snapshot.time,
        measurement_q=q.tolist(),
        measurement_v=v.tolist(),
        target_q=np.asarray(target).tolist(),
        target_v=np.asarray(target_v).tolist(),
        stage=stage,
        route=plain(route),
        pd=plain(diagnostic),
        generated_channels=plain(channels),
        ctrl_transport="direct FP32 native ctrl; original Env atanh/tanh transport is not replayed",
        goal=dict(
            target_xy=list(driver.target.position[:2]),
            target_yaw=driver.target.yaw,
            position_tolerance_m=driver.goal.position_tolerance_m,
            yaw_tolerance_rad=driver.goal.yaw_tolerance_rad,
        ),
    )
    if "route" in route:
        saved = driver.routes[int(route["route"]) - 1]
        record["route"].update(
            route_file=str(driver.output / saved["file"]), route_sha256=saved["sha256"]
        )
    return MuJoCoControl(**channels), record


def _original_action_transport(ctrl):
    """Repeat the original script's encoder and Env's CUDA decoder in order."""
    import torch

    direct = np.asarray(ctrl, dtype=np.float32)
    _require(
        direct.shape == (2,) and np.isfinite(direct).all() and np.all(abs(direct) < 20),
        "bounded original motor action",
    )
    raw = np.arctanh(direct / np.float32(20.0))
    with torch.no_grad():
        action = torch.tensor(raw, dtype=torch.float32, device="cuda:0")
        bias, scale = torch.zeros_like(action), torch.full_like(action, 20.0)
        values = bias + scale * torch.tanh(action)
        actual = values.detach().cpu().numpy().copy()
    _require(np.isfinite(actual).all(), "finite CUDA decoded control")
    return actual.astype(np.float64), dict(
        kind="original-env-fp32-atanh-cuda-tanh",
        direct_pd_ctrl=direct.tolist(),
        raw_action=raw.tolist(),
        transported_ctrl=actual.tolist(),
        direct_to_transported_delta=(actual.astype(float) - direct.astype(float)).tolist(),
        device="cuda:0",
        operator="bias + scale * torch.tanh(raw)",
        scale=20.0,
        bias=0.0,
        torch_version=str(torch.__version__),
    )


def _same_fp32_bytes(left, right):
    a, b = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


class _PrefixPlanner:
    """Replay only previously generated, source-bound routes; then discard this adapter."""

    def __init__(self, live, root, records):
        self.live, self.root, self.records = live, root, records
        self.index, self.tick, self.used = 0, 0, []

    def __getattr__(self, key):
        return getattr(self.live, key)

    def plan(self, q, proposals, *, previous_direction=None):
        _require(self.index < len(self.records), "missing historical route")
        record = self.records[self.index]
        _require(
            record["tick"] == self.tick, "historical route was not generated at this prefix tick"
        )
        path = (self.root / record["file"]).resolve()
        _require(
            path.is_relative_to(self.root) and _digest(path) == record["sha256"],
            "historical route digest/path",
        )
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            saved = json.load(stream, parse_constant=_reject_constant)
        raw = saved["route"]
        _require(
            saved["tick"] == self.tick
            and np.array_equal(saved["q"], q)
            and np.array_equal(raw["source_q"], q),
            "historical route source state",
        )
        _require(
            plain(proposals) == [complete_saved_push_proposal(p) for p in saved["proposals"]],
            "historical current-pose proposals differ",
        )
        complete = complete_saved_push_route(raw)
        proposal = (
            None
            if complete["proposal"] is None
            else NativePlanarPushProposal(
                **{k: tuple_values(v) for k, v in complete["proposal"].items()}
            )
        )
        route = NativePushRoute(
            raw["status"],
            proposal,
            np.asarray(raw["source_q"], dtype=np.float64),
            tuple(
                (t, stage, np.asarray(node, dtype=np.float64)) for t, stage, node in raw["nodes"]
            ),
            tuple(raw["attempts"]),
            int(raw["ik_calls"]),
            int(raw["path_calls"]),
            tuple(raw["initial_reference_projection"]),
        )
        _require(plain(route) == complete, "lossless historical route reconstruction")
        _require(
            route.ik_calls == record["ik_calls"] and route.path_calls == record["path_calls"],
            "historical planner budget",
        )
        self.live.ik_calls, self.live.path_calls = route.ik_calls, route.path_calls
        self.used.append(dict(tick=self.tick, file=str(path), sha256=record["sha256"]))
        self.index += 1
        return route


def _archived_measurement(raw):
    values = dict(raw)
    values["contacts"] = tuple(ObjectContactWrenchRecord(**r) for r in values["contacts"])
    return ObjectContactWrenchMeasurement(**values)


def _bootstrap(trajectory, source, output_routes, start):
    _, scene, model, manifest, world, trace, trace_sha = _archive_context(trajectory)
    _require(
        manifest["reference_feedback"] is True and 0 <= start < manifest["recorded_steps"],
        "original feedback interval",
    )
    _require(_digest(source) == SOURCE_SHA, "original candidate dataset SHA")
    dataset = _read_json(source)
    _require(dataset["schema"] == "original-push-boundary-endpoints-v1", "candidate schema")
    saved = next(row for row in dataset["rows"] if row["summary"]["world"] == world["world"])
    driver = PushFeedbackDriver(scene, model, output_routes, saved)
    pd, _, _ = hold_plan(scene, model)
    summary_path = trajectory / "reference-feedback-summary.json"
    summary = _read_json(summary_path)[str(world["world"])]
    records = [r for r in summary["routes"] if r["tick"] < start]
    planner = _PrefixPlanner(
        driver.planner, (trajectory / (scene.key + "-feedback")).resolve(), records
    )
    driver.planner = planner
    history, previous, selected = None, None, None
    key = str(world["world"])
    with gzip.open(trace, "rb") as stream:
        for tick, line in enumerate(stream):
            if tick > start:
                break
            row = json.loads(line, parse_constant=_reject_constant)
            _require(
                row["step"] == tick + 1 and row["accepted"] == {key: True},
                "contiguous accepted original prefix",
            )
            entry = row["entry"]
            q = _array(entry["entry_q"][0][: model.nq], (model.nq,), "prefix q")
            v = _array(entry["entry_v"][0][: model.nv], (model.nv,), "prefix v")
            clock = _array(entry["entry_clock"][0], (4,), "prefix clock")
            _require(entry["entry_meta"][0][:3] == [1, tick, tick + 1], "prefix input identity")
            if previous is not None:
                _require(
                    np.array_equal(q, previous[0])
                    and np.array_equal(v, previous[1])
                    and float(clock[0]) - float(clock[1]) == previous[2],
                    "prefix q/v/time continuity",
                )
            if tick == start:
                selected = dict(
                    q=q,
                    v=v,
                    time=float(clock[0]) - float(clock[1]),
                    trace_line_sha256=hashlib.sha256(line).hexdigest(),
                    original_controller=row["controller"][key],
                    original_raw_action=row["raw_action"][0][: model.nu],
                )
                break  # Never inspect this row's future q/v/feedback or a later route.
            planner.tick = tick
            desired, velocity, _, info = driver.reference(tick, q, v)
            _require(not driver.stopped, "historical driver unexpectedly stopped")
            old_pd = row["controller"][key]
            _require(
                np.array_equal(old_pd["target"], desired)
                and np.array_equal(old_pd["target_velocity"], velocity),
                "historical reference/phase",
            )
            old_feedback = row["reference_feedback"][key]
            _require(
                all(old_feedback[k] == plain(value) for k, value in info.items()),
                "historical route identity",
            )
            frame = SuccessFrame(**row["success_frames"][key])
            result = evaluate_goal(scene, driver.goal, frame, history)
            _require(plain(result) == row["results"][key], "historical Goal/history replay")
            history = result.history
            endpoint = _array(row["q"][0][: model.nq], (model.nq,), "prefix output q")
            end_v = _array(row["v"][0][: model.nv], (model.nv,), "prefix output v")
            measurement = _archived_measurement(old_feedback["accepted"]["measurement"])
            accepted = driver.accepted_measurement(
                tick + 1, endpoint, measurement, result, input_q=q
            )
            _require(
                plain(accepted) == old_feedback["accepted"], "historical controller progress replay"
            )
            previous = endpoint, end_v, float(clock[2]) - float(clock[3])
    _require(
        selected is not None and planner.index == len(records),
        "complete prefix/route reconstruction",
    )
    driver.planner = planner.live
    bootstrap = dict(
        prefix_steps=start,
        start_tick=start,
        history_mode="archive-prefix-controller-and-goal-replay",
        future_archived_states_used=False,
        original_routes_used=planner.used,
        controller_state=plain(driver.summary()),
        goal_history=plain(history),
    )
    provenance = dict(
        source_sha256=scene.source_sha256,
        original_trace_sha256=trace_sha,
        original_manifest_sha256=_digest(trajectory / "manifest.json"),
        original_feedback_summary_sha256=_digest(summary_path),
        candidate_source_sha256=SOURCE_SHA,
        initial_trace_line_sha256=selected["trace_line_sha256"],
    )
    return scene, model, driver, pd, history, selected, bootstrap, provenance


@dataclass(frozen=True)
class PublicFeedbackMeasurement:
    world: int
    input_tick: int
    output_tick: int
    time_s: float
    group_force_world: tuple
    group_moment_com_world: tuple
    area_samples: int
    native_contacts: int
    phase: str = "accepted-step-input"
    force_source: str = "actual area sample forces on object plus retained native pusher contacts"


def accepted_measurement_and_goal(
    scene, model, world, absolute_tick, before, after, history, *, goal_tick=None
):
    """Bind actual mixed forces and input-state Goal fields to one accepted step."""
    receipt = after.last_step
    _require(
        receipt is not None
        and receipt.world == world
        and receipt.tick == after.tick
        and after.tick == before.tick + 1,
        "accepted receipt world/tick",
    )
    _require(
        np.array_equal(receipt.input_q, before.q)
        and np.array_equal(receipt.input_v, before.v)
        and receipt.input_time == before.time
        and receipt.output_time == after.time,
        "accepted receipt state/clock",
    )
    dyn = receipt.dynamics
    _require(
        dyn is not None
        and np.array_equal(dyn.input_q, before.q)
        and np.array_equal(dyn.input_v, before.v),
        "same-entry dynamics",
    )
    body, tool = model.body("object").id, model.body("tool").id
    support, pusher = model.geom("support").id, model.geom("pusher").id
    force, moment, support_z = np.zeros(3), np.zeros(3), 0.0
    samples = 0
    for pair in receipt.raw_contact_fields["area_friction"]:
        _require(tuple(pair["body_ids"]) == (tool, body), "area object-side ownership")
        forces, points = np.asarray(pair["force_on_b"]), np.asarray(pair["points"])
        force += forces.sum(axis=0)
        moment += np.cross(points - dyn.body_com[body], forces).sum(axis=0)
        samples += len(forces)
    native_group = 0
    for contact in receipt.contacts:
        g1, g2 = map(int, contact.geometry_ids)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        if b1 == body and b2 != body:
            other, applied = g2, contact.force_on_geometry1_world
        elif b2 == body and b1 != body:
            other, applied = g1, contact.force_on_geometry2_world
        else:
            continue
        if other == support:
            support_z += float(applied[2])
        if other == pusher:
            # The configured whole body pair is replaced; any retained row here
            # would be duplicate native/area ownership, so fail rather than sum it.
            native_group += 1
    _require(native_group == 0, "replaced tool/object native contact survived")
    measurement = PublicFeedbackMeasurement(
        world,
        absolute_tick,
        absolute_tick + 1,
        before.time,
        tuple(force),
        tuple(moment),
        samples,
        native_group,
    )
    qa = int(model.jnt_qposadr[model.joint("object-free").id])
    jac = np.asarray(dyn.jacobian_com[body])
    linear_com, angular = jac[:3] @ before.v, jac[3:] @ before.v
    linear = linear_com - np.cross(angular, dyn.body_com[body] - dyn.body_position[body])
    if goal_tick is not None:
        _require(
            type(goal_tick) is int and goal_tick == before.tick,
            "Goal tick must equal the episode input tick",
        )
    frame = SuccessFrame(
        world,
        0,
        absolute_tick if goal_tick is None else goal_tick,
        before.time,
        tuple(dyn.body_position[body]),
        tuple(before.q[qa + 3 : qa + 7]),
        tuple(linear),
        tuple(angular),
        (0.0, 0.0),
        max(0.0, support_z),
        object_key=scene.object_key,
    )
    result = evaluate_goal(scene, scene.goals[scene.default_goal], frame, history)
    return measurement, frame, result


def _check_pair(states, prior):
    _require(set(states) == set(WORLDS.values()), "two physical worlds required")
    _require(
        states[5].tick == states[6].tick and states[5].time == states[6].time,
        "paired worlds are not synchronous",
    )
    if prior is not None:
        for world in WORLDS.values():
            a, b = states[world], prior[world]
            _require(
                a.tick == b.tick
                and a.time == b.time
                and np.array_equal(a.q, b.q)
                and np.array_equal(a.v, b.v),
                "control publication changed accepted q/v",
            )


def run_window(trajectory, source, output, *, start=1699, steps=69, cache_root=None):
    _require(
        type(start) is int
        and 0 <= start < 12000
        and type(steps) is int
        and 1 <= steps <= MAX_STEPS
        and start + steps <= 12000,
        "bounded start/steps",
    )
    trajectory, source, output = (
        Path(trajectory).resolve(),
        Path(source).resolve(),
        Path(output).resolve(),
    )
    journal_path, routes_path = output.with_suffix(".steps.jsonl"), output.with_suffix(".routes")
    _require(
        journal_path != output and not journal_path.exists() and not routes_path.exists(),
        "fresh journal/routes required",
    )
    report = dict(
        schema="phymeo-pusht-public-feedback-v1",
        status="running",
        controller_status=None,
        start_tick=start,
        requested_steps=steps,
        accepted_pair_steps=0,
        recorded_pair_steps=0,
        worlds=WORLDS,
        journal=dict(file=journal_path.name, records=0),
        interpretation=dict(
            physics="two original RR worlds; actual public mixed Scene.step; no state feedback injection",
            controls="open: original three channels; feedback: current-state original goal/PD ctrl and the same original external-force channels",
            controller="object feedback at route boundaries; joint PD at every accepted input; HOST reference planning",
            ctrl_transport="feedback repeats original FP32 encoder and CUDA tanh decoder; direct and transported effort are recorded separately",
            goal="unchanged original Goal on actual accepted input; prefix history preserved and world routing rebased",
            physical_calibration_validated=False,
            geometric_accuracy_validated=False,
        ),
    )
    stage, current_tick, pair, driver = "bootstrap", start, None, None
    with (
        output.open("x+", encoding="utf-8") as summary,
        journal_path.open("x", encoding="utf-8") as journal,
    ):
        _write_progress(summary, report)
        try:
            original, model, driver, pd, goal_history, entry, bootstrap, provenance = _bootstrap(
                trajectory, source, routes_path, start
            )
            commands = list(iter_controls(trajectory, start, start + steps))
            _require(
                len(commands) == steps
                and commands[0]["trace_line_sha256"] == entry["trace_line_sha256"],
                "window controls bind initial entry",
            )
            q = legal_initial_q(model, entry["q"])
            spec, asset = build_scene_spec(original.mjcf), build_side_asset(model)
            qa = int(model.jnt_qposadr[model.joint("object-free").id])
            histories = {
                world: None
                if goal_history is None
                else replace(goal_history, binding=(*goal_history.binding[:-2], world, 0))
                for world in WORLDS.values()
            }
            provenance.update(
                original_trajectory_directory=str(trajectory),
                candidate_source=str(source),
                script_sha256=_digest(Path(__file__)),
                original_control_window_sha256=_payload_hash(
                    [[c["tick"], c["trace_line_sha256"]] for c in commands]
                ),
                controller_sha256=_digest(
                    Path(__file__).parents[2] / "scripts/run_manipulation_push_feedback.py"
                ),
                mujoco_version=mj.__version__,
                numpy_version=np.__version__,
            )
            report.update(
                provenance=provenance,
                controller_history_bootstrap=bootstrap,
                initialization=dict(
                    archived_q=entry["q"].tolist(),
                    declared_q=q.tolist(),
                    archived_v=entry["v"].tolist(),
                    normalized_once=True,
                    original_time=entry["time"],
                ),
                material=dict(
                    identity=plain(IDENTITY),
                    center_pressure_pa=18000.0,
                    mu=MU,
                    slip_speed_m_s=SLIP,
                    pressure_tetrahedra=len(asset.pressure_local.cells),
                    rigid_faces=len(asset.rigid_local.faces),
                ),
            )
            declarations = tuple(
                MuJoCoTyxWorldSpec(
                    w, spec.key, MuJoCoInitialState(q, entry["v"], time=entry["time"])
                )
                for w in WORLDS.values()
            )
            stage = "scene_construction"
            with MuJoCoTyxHeterogeneousBatch(
                (spec,), declarations, mode="cuda", cache_root=cache_root
            ) as batch:
                scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (asset,)})
                pair = batch.integration_snapshot()
                _check_pair(pair, None)
                _require(
                    np.array_equal(pair[5].q, pair[6].q) and np.array_equal(pair[5].v, pair[6].v),
                    "identical paired initial q/v",
                )
                _require(
                    np.array_equal(pair[6].q, q.astype(np.float32))
                    and np.array_equal(pair[6].v, entry["v"].astype(np.float32)),
                    "declared initial quantization",
                )
                if start:
                    _require(
                        np.array_equal(driver.last_q, entry["q"]), "prefix controller endpoint"
                    )
                    driver.last_q = pair[6].q.copy()
                    driver.last_cost = driver.cost(pair[6].q)
                bootstrap["initial_q_rebind_max_abs"] = float(np.max(abs(pair[6].q - entry["q"])))
                report["initial_states"] = {lane: _state(pair[w], qa) for lane, w in WORLDS.items()}
                report["final_states"] = report["initial_states"]
                provenance["scene_asset_sha256"] = scene.asset_sha256
                provenance["compiled_bindings"] = {
                    lane: dict(
                        primary_sha256=batch.layout[w].primary_sha256,
                        model_sha256=batch.layout[w].model_sha256,
                        archive_sha256=batch.layout[w].archive_sha256,
                    )
                    for lane, w in WORLDS.items()
                }
                _require(
                    all(
                        batch.layout[w].primary_sha256 == original.source_sha256
                        for w in WORLDS.values()
                    ),
                    "original compiled source",
                )
                _write_progress(summary, report)
                for index, command in enumerate(commands):
                    current_tick, stage = command["tick"], "feedback_reference"
                    _check_pair(pair, None)
                    before = pair
                    feedback, diagnostic = command_from_snapshot(
                        driver, pd, current_tick, before[6], start_tick=start
                    )
                    transported, transport = _original_action_transport(feedback.ctrl)
                    diagnostic["transport"] = transport
                    diagnostic["ctrl_transport"] = transport["kind"]
                    submitted = {
                        5: control_from_record(command),
                        6: MuJoCoControl(
                            ctrl=transported,
                            qfrc_applied=command["qfrc_applied"],
                            xfrc_applied=command["xfrc_applied"],
                        ),
                    }
                    diagnostic["generated_channels"] = {
                        k: np.asarray(getattr(submitted[6], k)).tolist() for k in CHANNELS
                    }
                    diagnostic["external_channels_source"] = (
                        "same original held qfrc_applied/xfrc_applied as open lane"
                    )
                    if index == 0:
                        comparison = dict(
                            same_input_q_fp32_bytes=_same_fp32_bytes(before[6].q, entry["q"]),
                            same_input_v_fp32_bytes=_same_fp32_bytes(before[6].v, entry["v"]),
                            original_ctrl=command["ctrl"].tolist(),
                            regenerated_ctrl=transported.tolist(),
                            ctrl_fp32_bytes_equal=_same_fp32_bytes(command["ctrl"], transported),
                            original_direct_effort=entry["original_controller"]["effort"],
                            regenerated_direct_effort=diagnostic["pd"]["effort"],
                            original_raw_action=entry["original_raw_action"],
                            regenerated_raw_action=transport["raw_action"],
                            original_pd=entry["original_controller"],
                            regenerated_pd=diagnostic["pd"],
                        )
                        report["initial_command_comparison"] = comparison
                        _require(
                            not (
                                comparison["same_input_q_fp32_bytes"]
                                and comparison["same_input_v_fp32_bytes"]
                            )
                            or comparison["ctrl_fp32_bytes_equal"],
                            "same-state initial control differs after original CUDA action transport",
                        )
                    stage = "publish_controls"
                    batch.set_controls(submitted)
                    _check_pair(batch.integration_snapshot(), before)
                    stage = "paired_scene_step"
                    begun = perf_counter()
                    try:
                        accepted = dict(scene.step(H, expected_revision=batch.checkpoint_revision))
                    except Exception as error:
                        # Capture scratch before closing the owner. The simulation
                        # error and the last published physical frame stay authoritative.
                        try:
                            report["rejected_attempt"] = dict(
                                input_tick=current_tick,
                                submitted={
                                    lane: {
                                        key: plain(getattr(submitted[world], key))
                                        for key in CHANNELS
                                    }
                                    for lane, world in WORLDS.items()
                                },
                                candidate=capture_rejected_candidate(batch, error),
                            )
                        except Exception as diagnostic_error:
                            report["candidate_capture_error"] = _error_chain(diagnostic_error)
                        raise
                    wall = perf_counter() - begun
                    _require(
                        accepted == {5: True, 6: True}, "paired Scene did not accept both worlds"
                    )
                    pair = batch.integration_snapshot()
                    _check_pair(pair, None)
                    report["accepted_pair_steps"] += 1
                    report["final_states"] = {
                        lane: _state(pair[w], qa) for lane, w in WORLDS.items()
                    }
                    stage = "paired_receipt_and_controller"
                    lanes, pending_histories = {}, {}
                    for lane, world in WORLDS.items():
                        actual = dict(
                            tick=current_tick,
                            output_tick=current_tick + 1,
                            **{k: np.asarray(getattr(submitted[world], k)) for k in CHANNELS},
                        )
                        row = _step_record(
                            model,
                            asset,
                            before[world],
                            pair[world],
                            scene.last_geometry[world][0],
                            actual,
                            qa,
                            wall,
                        )
                        row["submitted_control"] = row.pop("control")
                        row.pop("stored_control_matches_original")
                        row["stored_control_matches_submitted"] = True
                        row["controller"] = (
                            diagnostic
                            if lane == "feedback"
                            else dict(
                                mode="original-held",
                                input_tick=current_tick,
                                local_tick=before[world].tick,
                                input_time=before[world].time,
                                measurement_q=before[world].q.tolist(),
                                measurement_v=before[world].v.tolist(),
                            )
                        )
                        measurement, frame, result = accepted_measurement_and_goal(
                            original,
                            model,
                            world,
                            current_tick,
                            before[world],
                            pair[world],
                            histories[world],
                        )
                        row.update(
                            measurement=plain(measurement),
                            goal_frame=plain(frame),
                            goal_result=plain(result),
                        )
                        pending_histories[world] = result.history
                        if lane == "feedback":
                            row["feedback_accepted"] = plain(
                                driver.accepted_measurement(
                                    current_tick + 1,
                                    pair[world].q,
                                    measurement,
                                    result,
                                    input_q=pair[world].last_step.input_q,
                                )
                            )
                            row["driver_state_after"] = plain(driver.summary())
                        lanes[lane] = row
                    packet = dict(
                        schema="phymeo-pusht-public-feedback-step-v1",
                        index=index,
                        original_input_tick=current_tick,
                        original_output_tick=current_tick + 1,
                        original_control=plain(command),
                        lanes=lanes,
                    )
                    payload = json.dumps(packet, allow_nan=False, separators=(",", ":")) + "\n"
                    journal.write(payload)
                    journal.flush()
                    histories = pending_histories
                    report["recorded_pair_steps"] += 1
                    report["journal"]["records"] += 1
                    report["controller_final"] = plain(driver.summary())
                    stop = _accepted_controller_stop(
                        stopped=driver.stopped,
                        reason=driver.reason,
                        monitor_status=driver.monitor.status,
                        accepted_steps=report["recorded_pair_steps"],
                    )
                    report["controller_status"] = driver.reason
                    if stop is not None:
                        report.update(
                            status="controller_stopped",
                            stopped_after_original_output_tick=current_tick + 1,
                        )
                    _write_progress(summary, report)
                    if stop is not None:
                        break
                stage = "scene_cleanup"
            if report["status"] == "running":
                report["status"] = "complete"
        except Exception as error:
            if driver is not None:
                report["controller_status"] = driver.reason
            report.update(
                status="partial_failure",
                failure_stage=stage,
                failed_original_input_tick=current_tick,
                errors=_error_chain(error),
            )
            # A callback failure can follow a real accepted step. Keep that fact
            # and its paired states in the summary, never pretend it rolled back.
            if pair is not None:
                report["unpublished_accepted_pairs"] = (
                    report["accepted_pair_steps"] - report["recorded_pair_steps"]
                )
        journal.flush()
        report["journal"]["sha256"] = _digest(journal_path)
        _write_progress(summary, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=1699)
    parser.add_argument("--steps", type=int, default=69)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    report = run_window(
        args.trajectory,
        args.source,
        args.output,
        start=args.start,
        steps=args.steps,
        cache_root=args.cache_root,
    )
    if report["status"] not in ("complete", "controller_stopped"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
