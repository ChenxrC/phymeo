"""Branch only from archived accepted handoff frames; all new motion comes from the stepper."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from phymeo.control.grasp_transfer import bounded_approach
from phymeo.control.kinodynamic_approach import ApproachBudget
from phymeo.core.robot import RobotState
from phymeo.core.scene import SceneInputs, SceneState, StepSettings
from phymeo.core.state import ParticleState
from phymeo.experiments.compliant_cloth import Case, record_step, trace_start
from phymeo.experiments.compliant_handoff import (
    _observations,
    _shared_support,
    _travel,
    build_handoff,
)
from phymeo.experiments.handoff_grippers import BASE_Z, CLOSED
from phymeo.experiments.handoff_reception import CooperativeReception
from phymeo.experiments.receiver_approach import (
    ReceiverControl,
    plan_receiver_approach,
    structural_clearance,
)


def read(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        return json.load(stream)


def restore(frame):
    parts = {
        "cloth": ParticleState(
            np.asarray(frame["cloth_position_m"]), np.asarray(frame["cloth_velocity_m_s"])
        )
    }
    parts.update(
        {
            name: RobotState(np.asarray(row["q"]), np.asarray(row["v"]))
            for name, row in frame["robots"].items()
        }
    )
    return SceneState(parts)


def accepted_delta(first, second):
    values = {}
    for name, part in first.domains.items():
        other = second.domains[name]
        for field in ("position", "velocity") if isinstance(part, ParticleState) else ("q", "v"):
            values[name + "." + field] = float(
                np.max(np.abs(getattr(part, field) - getattr(other, field)))
            )
    return values


def coordinator(case):
    a0, b0 = [case.grippers[n].position(case.initial.domains[n]) for n in ("A", "B")]
    span = np.ptp(case.initial.domains["cloth"].position, axis=0)
    transfer = np.array([a0[0] + 0.025, BASE_Z + np.linalg.norm(span) + 0.04])
    pad = next(p for p in case.program.patches if p.geometry.key == "B_left")
    rho = case.program.cloth_links.rho[0]
    weight = float(np.sum(1 / case.program.cloth.inverse_mass) * 9.81)
    controls = CooperativeReception(
        case.receiver_nodes,
        case.dt,
        weight,
        rho + pad.radius_m,
        a0,
        b0,
        transfer,
        layer_width=pad.width_m,
        axial_clearance=rho + pad.width_m,
    )
    return controls, a0, transfer, weight


def command_at(c, case, frame, state, home, transfer, lift):
    measured = dict(frame)
    measured["actual_tool_velocity_xz_m_s"] = {
        n: g.velocity(state.domains[n]).tolist() for n, g in case.grippers.items()
    }
    elapsed = frame["t"] - lift
    return c.command("lift_A", measured, _travel(home, transfer, elapsed, 0.55), elapsed)


def frame_advance(case, state, motion, apertures, budget):
    inputs = dict(case.inputs.domains)
    a, b = case.grippers["A"], case.grippers["B"]
    for name, g in case.grippers.items():
        closed = name == "A" or motion.closed
        apertures[name] += float(
            np.clip((CLOSED if closed else 0.0) - apertures[name], -0.09 * case.dt, 0.09 * case.dt)
        )
    inputs["A"], a_joint = a.command(
        state.domains["A"],
        motion.donor,
        apertures["A"],
        case.dt,
        tool_velocity_m_s=motion.donor_rate,
    )
    robot = state.domains["B"]
    obs = motion.observation or {}
    if obs.get("action") not in ("stage", "await_aperture", "approach", "track"):
        drive, joints = b.command(
            robot, motion.receiver, apertures["B"], case.dt, tool_velocity_m_s=motion.receiver_rate
        )
        return (
            SceneInputs({**inputs, "B": drive}),
            a_joint,
            ReceiverControl(drive, joints, motion.receiver, motion.receiver_rate),
            None,
        )
    target = obs.get("exterior_point_m", obs.get("material_point_m"))
    if target is None:
        goal, rate = bounded_approach(
            b.position(robot),
            b.velocity(robot),
            b.position(robot),
            np.zeros(2),
            case.dt,
            peak_speed=budget.peak_speed,
            peak_acceleration=budget.peak_acceleration,
        )
        drive, joints = b.command(robot, goal, apertures["B"], case.dt, tool_velocity_m_s=rate)
        choice = ReceiverControl(drive, joints, goal, rate)
        return (
            SceneInputs({**inputs, "B": drive}),
            a_joint,
            choice,
            dict(admissible=False, reason="no_material_direction"),
        )
    start_aperture = apertures["B"]
    final_aperture = CLOSED if motion.closed else 0.0

    def finger_command(index):
        reach = 0.09 * case.dt * index
        return start_aperture + float(np.clip(final_aperture - start_aperture, -reach, reach))

    plan = plan_receiver_approach(
        case.program,
        b,
        "B",
        state,
        SceneInputs(inputs),
        np.asarray(target)[[0, 2]],
        np.asarray(obs["material_velocity_m_s"])[[0, 2]],
        finger_command,
        budget,
    )
    best = plan.selected
    choice = best.control
    if choice is None:
        p, v = b.position(robot), b.velocity(robot)
        goal, rate = bounded_approach(
            p,
            v,
            p,
            np.zeros(2),
            case.dt,
            peak_speed=budget.peak_speed,
            peak_acceleration=budget.peak_acceleration,
        )
        drive, joints = b.command(robot, goal, apertures["B"], case.dt, tool_velocity_m_s=rate)
        choice = ReceiverControl(drive, joints, goal, rate)

    def trial(row):
        return dict(
            name=row.name,
            clear_steps=row.clear_steps,
            brake_steps=row.brake_steps,
            progress_m=row.progress,
            minimum_margin_m=row.minimum_margin,
            reason=row.reason,
            first_tool_m=None if row.first_position is None else row.first_position.tolist(),
            first_tool_rate_m_s=None if row.first_velocity is None else row.first_velocity.tolist(),
            effort=None if row.control is None else row.control.inputs.effort.tolist(),
        )

    forecast = dict(
        admissible=plan.admissible,
        selected=best.name,
        selected_reason=best.reason,
        first_tool_forecast_m=None if best.first_position is None else best.first_position.tolist(),
        candidates=[trial(row) for row in plan.candidates],
    )
    return SceneInputs({**inputs, "B": choice.inputs}), a_joint, choice, forecast


def actual_layers(samples, width, h):
    active = 0
    first = None
    peak = 0.0
    impulse = np.zeros(3)
    gap = None
    for sample in samples:
        groups = [r for r in sample["cloth_links"] if r["robot"] == "B"]
        if groups:
            active += 1
            if first is None:
                first = dict(t=sample["t"], pairs=[[r["link"], r["face"]] for r in groups])
        for row in groups:
            margin = row["gap_m"] - width
            gap = margin if gap is None else min(gap, margin)
            for site in row["sites"]:
                impulse += h * np.asarray(site["force_on_cloth_n"])
        peak = max(peak, sum(row["normal_force_n"] for row in groups))
    return dict(
        layer_substeps=active,
        first=first,
        min_sampled_active_gap_m=gap,
        peak_normal_n=peak,
        impulse_on_cloth_ns=impulse.tolist(),
    )


def physical_path(case, entry, result, t):
    """Check every accepted affine backed-pad drift using its real q and cloth nodes."""
    contacts = case.program.cloth_links
    grip = case.grippers["B"]
    previous = entry
    h = case.dt / case.substeps
    rows = []
    first = None
    minimum, pair = structural_clearance(contacts, "B", entry)
    peak_speed = float(np.linalg.norm(grip.velocity(entry.domains["B"])))
    for k, state in enumerate(result.substep_states):
        blocker = contacts.approach_blocker("B", previous, state)
        if first is None and blocker is not None:
            first = dict(interval_s=[t + k * h, t + (k + 1) * h], pair=blocker)
        gap, current_pair = structural_clearance(contacts, "B", state)
        if gap < minimum:
            minimum, pair = gap, current_pair
        previous_robot, robot = previous.domains["B"], state.domains["B"]
        # The backed solve holds robot.v throughout this affine q drift.
        midpoint = RobotState((previous_robot.q + robot.q) / 2, robot.v)
        speed = grip.velocity(robot)
        peak_speed = max(
            peak_speed, float(np.linalg.norm(speed)), float(np.linalg.norm(grip.velocity(midpoint)))
        )
        rows.append(
            dict(
                t=t + (k + 1) * h,
                q=robot.q.tolist(),
                v=robot.v.tolist(),
                tip_m=grip.position(robot).tolist(),
                tool_rate_m_s=speed.tolist(),
                end_gap_m=gap,
                pair=current_pair,
                drift_blocker=blocker,
            )
        )
        previous = state
    assert len(rows) == case.substeps
    return dict(
        minimum_sampled_gap_m=minimum,
        minimum_pair=pair,
        first_blocked_drift=first,
        peak_sampled_tool_speed_m_s=peak_speed,
        actual_drifts=len(rows),
    ), rows


def run(args):
    folder = args.archive
    landmarks = json.loads((folder / "first-structural-options.json").read_text(encoding="utf8"))
    block = next(row for row in landmarks if row["variant"] == args.variant)
    barrier = next(
        row
        for row in block["probes"]
        if row["recorded_fraction"] == 0 and row["recorded_endpoint_blocker"]
    )
    window = read(folder / f"window-{args.variant}.json.gz")
    capture = read(folder / f"capture-{args.variant}.json.gz")
    reference_samples = capture["robot_samples"]
    frames = window["frames"] + capture["frames"][1:]
    tilt = capture["metrics"]["source_initial_tilt_deg"]
    del window, capture
    # The earlier unmodified trajectory is needed solely to restore observer history before
    # the saved aperture window. No future frame is visible to the planner or material forecast.
    prefix = read(folder.parent / "with-exterior-self-v1" / f"full-{args.variant}.json.gz")
    lifted = next(e["t"] for e in prefix["events"] if e["phase"] == "lift_A")
    prefix_frames = [
        f for f in prefix["frames"] if lifted - 1e-12 <= f["t"] < frames[0]["t"] - 1e-12
    ]
    del prefix
    case = build_handoff(
        robot="articulated-palm",
        fabric="soft",
        initial_tilt_deg=tilt,
        finger_pad_radius_m=0.002,
        finger_backing_clearance_m=0.000125,
        self_contact=True,
    )
    dt = case.dt
    h = dt / case.substeps
    width = case.program.cloth_links.law.width_m
    c, home, transfer, weight = coordinator(case)
    budget = ApproachBudget(
        dt,
        c.peak_speed,
        c.peak_acceleration,
        args.horizon,
        int(np.ceil(c.peak_speed / (c.peak_acceleration * dt))) + 10,
        min(0.05 * c.peak_speed, width / (8 * dt)),
    )
    hit = next(i for i, f in enumerate(frames) if abs(f["t"] - barrier["input_s"]) < 1e-10)
    continuing = read(args.resume) if args.resume is not None else None
    if continuing is not None:
        if args.compare_first or args.verify_restore:
            raise ValueError(
                "resume the accepted trajectory without revalidating an unrelated old drive"
            )
        if (
            continuing["metrics"]["variant"] != args.variant
            or continuing["substeps"] != case.substeps
        ):
            raise ValueError("resumed input belongs to another physical case")
        if continuing["metrics"]["motion_budget"] != vars(budget) or continuing["metrics"][
            "status"
        ] not in (
            "window_complete",
            "receiver_bilateral",
            "unresolved_structural_drift",
            "layer_endpoint_before_force_sample",
        ):
            raise ValueError(
                "resume needs the same bounded controller and a completed physical window"
            )
        branch_frames = continuing["frames"]
        branch_start = next(
            i for i, f in enumerate(frames) if abs(f["t"] - branch_frames[0]["t"]) < 1e-10
        )
        if (
            max(accepted_delta(restore(frames[branch_start]), restore(branch_frames[0])).values())
            != 0
        ):
            raise ValueError("resumed trajectory lacks its accepted original-state anchor")
        index = next(
            i for i, f in enumerate(frames) if abs(f["t"] - branch_frames[-1]["t"]) < 1e-10
        )
        history = prefix_frames + frames[:branch_start] + branch_frames[:-1]
        entry = branch_frames[-1]
    else:
        index = hit - args.lead_frames
        if index < 0:
            raise ValueError("entry precedes the saved aperture trajectory")
        history = prefix_frames + frames[:index]
        entry = frames[index]
    if args.frames < 1 or index + args.frames >= len(frames):
        raise ValueError("requested physical interval is outside the archive")
    observed_outputs = (
        {}
        if continuing is None
        else {f["t"]: g for f, g in zip(branch_frames[:-1], branch_frames[1:], strict=True)}
    )
    for f in history:
        old_motion = command_at(c, case, f, restore(f), home, transfer, lifted)
        if f["t"] in observed_outputs:
            recorded = observed_outputs[f["t"]]
            assert old_motion.observation["action"] == recorded["receiver_observation"]["action"]
            np.testing.assert_array_equal(old_motion.donor, recorded["requested_tool_xz_m"]["A"])
            for field in ("material_point_m", "material_velocity_m_s", "exterior_point_m"):
                if field in old_motion.observation:
                    np.testing.assert_allclose(
                        old_motion.observation[field],
                        recorded["receiver_observation"][field],
                        atol=1e-12,
                        rtol=0,
                    )
        c.lost(f["measurements"])
    state = restore(entry)
    aperture = {n: entry["joint_target"][n][g.fingers[0]] for n, g in case.grippers.items()}
    if continuing is not None:
        trace = continuing
        trace["events"].append(dict(phase="continued accepted branch", t=entry["t"]))
        trace["coordination_events"] = c.events
        rows = trace["metrics"]["rows"].copy()
        old_replay = trace["metrics"]["old_drive_replay_max_abs"]
    else:
        trace = trace_start(
            Case(case.program, state, case.inputs, dt, case.substeps), "kinodynamic-" + args.variant
        )
        trace["frames"][0] = dict(entry)
        trace["receiver_nodes"] = case.receiver_nodes.tolist()
        trace["initial_tilt_deg"] = tilt
        trace["robot_layout"] = "articulated-palm"
        trace["fabric"] = case.fabric.record(case.grid)
        trace["events"] = [dict(phase="accepted measured entry", t=entry["t"])]
        trace["coordination_events"] = c.events
        trace["receiver_substep_motion"] = []
        rows = []
        old_replay = None
    prior_rows = len(rows)
    settings = StepSettings(
        substeps=case.substeps, position_iterations=1, velocity_iterations=1, record_substeps=True
    )
    status = "window_complete"
    with case.program.stepper(settings) as stepper:
        for j in range(args.frames):
            original = frames[index + j + 1]
            frame = trace["frames"][-1]
            t = frame["t"]
            motion = command_at(c, case, frame, state, home, transfer, lifted)
            inputs, aj, choice, forecast = frame_advance(case, state, motion, aperture, budget)
            if j == 0 and continuing is None:
                np.testing.assert_array_equal(
                    inputs.domains["A"].effort, original["actuator_effort"]["A"]
                )
                obs = original["receiver_observation"]
                assert motion.observation["action"] == obs["action"]
                if "material_point_m" in obs:
                    np.testing.assert_allclose(
                        motion.observation["material_point_m"],
                        obs["material_point_m"],
                        atol=1e-13,
                        rtol=0,
                    )
            if args.compare_first is not None and j == 0:
                if args.frames != 1:
                    raise ValueError("single-step drive comparison requires one frame")
                previous = read(args.compare_first)
                for key in (
                    "dt_s",
                    "substeps",
                    "robot_models",
                    "robot_contact",
                    "cloth_link_material",
                    "cloth_self_material",
                    "patches",
                ):
                    assert json.loads(json.dumps(trace[key])) == previous[key], key
                assert max(accepted_delta(state, restore(previous["frames"][0])).values()) == 0
                old_row = previous["frames"][1]
                if all(
                    np.array_equal(inputs.domains[n].effort, old_row["actuator_effort"][n])
                    for n in "AB"
                ):
                    comparison = dict(
                        status="identical_previously_executed_drive",
                        source=str(args.compare_first),
                        entry_s=t,
                        variant=args.variant,
                        source_tilt_deg=tilt,
                        forecast=forecast,
                        effort={n: inputs.domains[n].effort.tolist() for n in "AB"},
                        original_solver_substeps=case.substeps,
                    )
                    with args.output.with_suffix(".reuse.json").open("x", encoding="utf8") as out:
                        json.dump(comparison, out, indent=2, allow_nan=False)
                    print(
                        json.dumps(
                            {k: v for k, v in comparison.items() if k not in ("forecast", "effort")}
                        ),
                        flush=True,
                    )
                    return
            original_path = None
            if args.verify_restore and j in (0, args.lead_frames):
                recorded_entry = restore(frames[index + j])
                data = dict(case.inputs.domains)
                for n in "AB":
                    drive = case.inputs.domains[n].__class__.zeros(len(recorded_entry.domains[n].q))
                    drive.effort[:] = original["actuator_effort"][n]
                    data[n] = drive
                control_check = stepper.step(recorded_entry, SceneInputs(data), dt)
                assert control_check.report.accepted, control_check.report.reason
                replay = accepted_delta(control_check.state, restore(original))
                assert max(replay.values()) < 1e-10, replay
                if j == 0:
                    old_replay = replay
                original_path, _ = physical_path(case, recorded_entry, control_check, t)
                original_path["replay_max_abs"] = replay
            s0, pair0 = structural_clearance(case.program.cloth_links, "B", state)
            result = stepper.step(state, inputs, dt)
            if not record_step(case, trace, result, t):
                status = trace["failure"]
                break
            path, samples = physical_path(case, state, result, t)
            trace["receiver_substep_motion"].extend(samples)
            state = result.state
            diag = result.report.substeps[-1].diagnostics
            m = _observations(state, diag["patches"], weight, diag["robot_contact"])
            actual = case.grippers["B"].position(state.domains["B"])
            velocity = case.grippers["B"].velocity(state.domains["B"])
            end, pair = structural_clearance(case.program.cloth_links, "B", state)
            old_gap, old_pair = structural_clearance(
                case.program.cloth_links, "B", restore(original)
            )
            body = actual_layers(trace["robot_samples"][-case.substeps :], width, h)
            ref_body = actual_layers(
                [s for s in reference_samples if t - 1e-9 <= s["t"] < t + dt - 1e-9], width, h
            )
            ff = trace["frames"][-1]
            ff.update(
                phase="measured receiver approach",
                actuator_effort={n: inputs.domains[n].effort.tolist() for n in "AB"},
                requested_tool_xz_m={"A": motion.donor.tolist(), "B": choice.goal.tolist()},
                joint_target={"A": aj.tolist(), "B": choice.joint_target.tolist()},
                measurements=m,
                actual_tool_xz_m={
                    n: g.position(state.domains[n]).tolist() for n, g in case.grippers.items()
                },
                actual_tool_velocity_xz_m_s={
                    n: g.velocity(state.domains[n]).tolist() for n, g in case.grippers.items()
                },
                receiver_observation=motion.observation,
                approach_forecast=forecast,
                issued_receiver_reference_rate_m_s=None
                if choice.rate is None
                else choice.rate.tolist(),
            )
            pred = None if forecast is None else forecast.get("first_tool_forecast_m")
            row = dict(
                entry_s=t,
                end_s=ff["t"],
                entry_clearance_m=s0,
                entry_pair=pair0,
                end_clearance_m=end,
                end_pair=pair,
                original_end_clearance_m=old_gap,
                original_end_pair=old_pair,
                actual_tool_m=actual.tolist(),
                actual_tool_rate_m_s=velocity.tolist(),
                original_tool_m=original["actual_tool_xz_m"]["B"],
                entry_tool_m=frame["actual_tool_xz_m"]["B"],
                effort=inputs.domains["B"].effort.tolist(),
                original_effort=original["actuator_effort"]["B"],
                prediction=forecast,
                first_pose_forecast_error_m=None
                if pred is None
                else float(np.linalg.norm(actual - pred)),
                physical_path=path,
                original_physical_path=original_path,
                structure=body,
                original_structure=ref_body,
                measurements=m,
                original_measurements=original["measurements"],
                shared_support=_shared_support(m, weight),
                accepted_substeps=len(result.report.substeps),
            )
            rows.append(row)
            print(
                json.dumps(
                    dict(
                        t=t,
                        selected=None if forecast is None else forecast.get("selected"),
                        end_gap=end,
                        old_gap=old_gap,
                        first=body["first"],
                        shared=row["shared_support"],
                    )
                ),
                flush=True,
            )
            if args.stop_on_layer and (body["layer_substeps"] or path["first_blocked_drift"]):
                status = (
                    "first_structural_layer"
                    if body["layer_substeps"]
                    else (
                        "layer_endpoint_before_force_sample"
                        if end <= 0
                        else "unresolved_structural_drift"
                    )
                )
                break
            if args.stop_on_grip and m["B_both_normal_n"] > 0.20:
                status = (
                    "shared_support"
                    if len(rows) >= 2 and all(r["shared_support"] for r in rows[-2:])
                    else "receiver_bilateral"
                )
                break
            if args.stop_on_grip and c.lost(m):
                status = "both_grips_lost"
                break
    trace["completed"] = len(rows) - prior_rows == args.frames
    summary = dict(
        variant=args.variant,
        source_tilt_deg=tilt,
        reference_block_s=barrier["input_s"],
        lead_frames=continuing["metrics"]["lead_frames"] if continuing else args.lead_frames,
        entry_s=trace["frames"][0]["t"],
        resumed_s=entry["t"] if continuing else None,
        continued_from=None if args.resume is None else str(args.resume),
        new_physical_substeps=(len(rows) - prior_rows) * case.substeps,
        observer_frames_reproduced=len(observed_outputs),
        status=status,
        dt_s=dt,
        substeps=case.substeps,
        motion_budget=vars(budget),
        old_drive_replay_max_abs=old_replay,
        rows=rows,
        force_sample="physical substep entry",
        forecast="midpoint articulated force model, affine material, no backing impulse",
    )
    trace["metrics"] = summary
    with gzip.open(args.output, "xt", encoding="utf8", compresslevel=3) as out:
        json.dump(trace, out, separators=(",", ":"), allow_nan=False)
    with args.output.with_suffix(".summary.json").open("x", encoding="utf8") as out:
        json.dump(summary, out, indent=2, allow_nan=False)
    if len(rows) == prior_rows or status not in (
        "window_complete",
        "first_structural_layer",
        "unresolved_structural_drift",
        "layer_endpoint_before_force_sample",
        "receiver_bilateral",
        "shared_support",
        "both_grips_lost",
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--variant", choices=("vertical", "tilted"), required=True)
    parser.add_argument("--lead-frames", type=int, default=0)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--verify-restore", action="store_true")
    parser.add_argument("--compare-first", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--stop-on-grip", action="store_true")
    parser.add_argument("--stop-on-layer", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
