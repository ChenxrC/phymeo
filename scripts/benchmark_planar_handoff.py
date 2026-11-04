"""Resume the real soft handoff from the first force-qualified donor presentation.

The recorded prefix is used only to reconstruct the controller and original states.
Every frame after the measured branch point runs the original force-limited robots and cloth.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

import phymeo as ph
from phymeo.core.scene import fingerprint
from phymeo.experiments.compliant_cloth import Case, record_step, trace_start
from phymeo.experiments.compliant_handoff import (
    _observations,
    _shared_support,
    _travel,
    build_handoff,
)
from phymeo.experiments.handoff_boundary import MountedBoundaryAccess
from phymeo.experiments.handoff_grippers import BASE_Z, CLOSED
from phymeo.experiments.handoff_reception import CooperativeReception


def read(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        return json.load(stream)


def physical(frame):
    cloth = ph.ParticleState(
        np.array(frame["cloth_position_m"]), np.array(frame["cloth_velocity_m_s"])
    )
    return ph.SceneState(
        {
            "cloth": cloth,
            **{
                n: ph.RobotState(np.array(r["q"]), np.array(r["v"]))
                for n, r in frame["robots"].items()
            },
        }
    )


def run(original_path, output, *, frames=None, stop_time=None, resume=None, boundary=False):
    summary = output.with_suffix("").with_suffix(".summary.json")
    if output.exists() or summary.exists():
        raise FileExistsError("choose a fresh output and summary path for this physical window")
    original = read(original_path)
    source = original["frames"]
    pad = next(p for p in original["patches"] if p["name"] == "B_left")
    case = build_handoff(
        robot=original["robot_layout"],
        fabric=original["fabric"]["name"],
        grid=original["fabric"]["grid"],
        initial_tilt_deg=original["initial_tilt_deg"],
        finger_pad_radius_m=pad["radius_m"],
        finger_backing_clearance_m=pad["backing_clearance_m"],
        self_contact=True,
    )
    dt = case.dt
    assert (
        dt == original["dt_s"]
        and case.substeps == original["substeps"]
        and len(original["cloth_self_material"]["layers"]) == 276
    )
    a0, b0 = (case.grippers[n].position(case.initial.domains[n]) for n in ("A", "B"))
    span = np.ptp(case.initial.domains["cloth"].position, axis=0)
    transfer = np.array([a0[0] + 0.025, BASE_Z + np.linalg.norm(span) + 0.04])
    weight = float(np.sum(1 / case.program.cloth.inverse_mass) * 9.81)
    rho = original["fabric"]["half_thickness_m"]
    coord = CooperativeReception(
        case.receiver_nodes,
        dt,
        weight,
        rho + pad["radius_m"],
        a0,
        b0,
        transfer,
        layer_width=pad["width_m"],
        axial_clearance=rho + pad["width_m"],
        paired_presentation=True,
        boundary_access=MountedBoundaryAccess(case) if boundary else None,
    )
    start = next(ev["t"] for ev in original["events"] if ev["phase"] == "lift_A")
    begin = round(start / dt)
    aperture = {n: source[begin]["joint_target"][n][g.fingers[0]] for n, g in case.grippers.items()}

    def drive(frame, state, t):
        frame["actual_tool_velocity_xz_m_s"] = {
            n: g.velocity(state.domains[n]).tolist() for n, g in case.grippers.items()
        }
        motion = coord.command("lift_A", frame, _travel(a0, transfer, t - start, 0.55), t - start)
        inputs = {"cloth": case.inputs.domains["cloth"]}
        targets = {}
        for n, g in case.grippers.items():
            want = CLOSED if n == "A" or motion.closed else 0.0
            aperture[n] += float(np.clip(want - aperture[n], -0.09 * dt, 0.09 * dt))
            if (
                n == "B"
                and motion.observation
                and motion.observation["action"] in ("stage", "await_aperture", "approach", "track")
            ):
                inputs[n], targets[n] = coord.drive_approach(
                    n, g, state, motion, aperture[n], case.program.cloth_links
                )
            else:
                goal, rate = (
                    (motion.donor, motion.donor_rate)
                    if n == "A"
                    else (motion.receiver, motion.receiver_rate)
                )
                inputs[n], targets[n] = g.command(
                    state.domains[n],
                    goal,
                    aperture[n],
                    dt,
                    **({"tool_velocity_m_s": rate} if rate is not None else {}),
                )
        return motion, inputs, targets

    def matched(next_frame, command):
        m, u, target = command
        for n, goal in (("A", m.donor), ("B", m.receiver)):
            for measured, expected in (
                (u[n].effort, next_frame["actuator_effort"][n]),
                (target[n], next_frame["joint_target"][n]),
                (goal, next_frame["requested_tool_xz_m"][n]),
            ):
                np.testing.assert_array_equal(measured, expected)

    entry = None
    for i in range(begin, len(source) - 1):
        restored = physical(source[i])
        command = drive(source[i], restored, i * dt)
        if coord.presentation.goal is not None or (boundary and coord.boundary.report is not None):
            entry = i
            break
        matched(source[i + 1], command)
    if entry is None:
        raise ValueError("source has no measured released-from-solo presentation entry")
    state = physical(source[entry])
    local = Case(case.program, state, case.inputs, dt, case.substeps)
    mode = "boundary-presentation" if boundary else "paired-presentation"
    trace = trace_start(local, mode + "-" + str(original["initial_tilt_deg"]))
    trace["frames"][0] = dict(source[entry])
    signature = fingerprint(case.program.problem_signature())
    sha = hashlib.sha256(Path(original_path).read_bytes()).hexdigest()
    trace["origin"] = dict(
        sha256=sha,
        initial_tilt_deg=original["initial_tilt_deg"],
        program=signature,
        entry_state=fingerprint(state),
        prefix_input_frames=entry - begin,
        entry_s=entry * dt,
        grid=original["fabric"]["grid"],
        substeps=case.substeps,
        force_limits={n: g.effort_limits.tolist() for n, g in case.grippers.items()},
    )

    if boundary:
        trace["origin"]["receiver_control"] = mode

    def qualified(frame):
        return coord.holder is not None and _shared_support(frame["measurements"], weight)

    count = 0
    i = entry
    if resume:
        saved = read(resume)
        if trace["origin"] != saved["origin"] or saved["metrics"]["stop"] != "bounded window":
            raise ValueError(
                "resume does not match the original source, model and qualified controller prefix"
            )
        trace = saved
        for fr in trace["frames"][1:]:
            matched(fr, command)
            state = physical(fr)
            i += 1
            if coord.lost(fr["measurements"]):
                raise ValueError("cannot resume after loss of both grippers")
            count = count + 1 if qualified(fr) else 0
            command = drive(fr, state, i * dt)
    end = i + frames if frames is not None else round(stop_time / dt)
    if end <= i:
        raise ValueError("new window must extend the accepted prefix")
    stop = "bounded window"
    trace["phases"] = ["lift_A"]
    trace["events"] = [{"phase": "original measured donor solo support", "t": entry * dt}]
    with case.program.stepper(
        ph.StepSettings(substeps=case.substeps, position_iterations=1, velocity_iterations=1)
    ) as runtime:
        for tick in range(i, end):
            if tick > i:
                try:
                    command = drive(trace["frames"][-1], state, tick * dt)
                except ph.SolverFailure as error:
                    stop = "physical actuator command rejected: " + str(error)
                    trace["failure"] = stop
                    break
            motion, inputs, targets = command
            result = runtime.step(state, ph.SceneInputs(inputs), dt)
            if not record_step(local, trace, result, tick * dt):
                stop = trace.get("failure", "physics rejected")
                break
            state = result.state
            diagnostics = result.report.substeps[-1].diagnostics
            fr = trace["frames"][-1]
            fr.update(
                phase="lift_A",
                measurements=_observations(
                    state, diagnostics["patches"], weight, diagnostics["robot_contact"]
                ),
                actuator_effort={n: inputs[n].effort.tolist() for n in ("A", "B")},
                joint_target={n: q.tolist() for n, q in targets.items()},
                requested_tool_xz_m={"A": motion.donor.tolist(), "B": motion.receiver.tolist()},
                actual_tool_xz_m={
                    n: g.position(state.domains[n]).tolist() for n, g in case.grippers.items()
                },
                receiver_observation=motion.observation,
            )
            count = count + 1 if qualified(fr) else 0
            if count >= 2:
                stop = "original shared-load gate on two consecutive frames"
                break
            if coord.lost(fr["measurements"]):
                stop = "both grips lost"
                break
    data = trace["frames"][1:]
    sub = trace["robot_samples"]
    assert len(sub) == len(data) * case.substeps == len(trace["self_samples"])
    metrics = dict(
        stop=stop,
        entry_s=entry * dt,
        end_s=trace["frames"][-1]["t"],
        frames=len(data),
        substeps=len(sub),
        entry_state=trace["origin"]["entry_state"],
        end_state=fingerprint(state),
        prefix_inputs=entry - begin,
        source_sha256=sha,
        shared_gate=count >= 2,
        actual_shared_frames=[f["t"] for f in data if _shared_support(f["measurements"], weight)],
        blocked=[
            (f["t"] - dt, f["receiver_observation"]["blocked_approach_candidates"][-1].get("pair"))
            for f in data
            if f["receiver_observation"]
            and f["receiver_observation"].get("predicted_structural_clear") is False
        ],
        first_B_bilateral_s=next(
            (f["t"] for f in data if f["measurements"]["B_both_normal_n"] > 0.20), None
        ),
        qualified_A_frames=sum(f["measurements"]["A_both_normal_n"] > 0.20 for f in data),
        peak_B_structure_n=max((f["measurements"]["B_link_normal_n"] for f in data), default=0),
        peak_B_pad_n=max((f["measurements"]["B_normal_n"] for f in data), default=0),
        actual_A_xz_m=trace["frames"][-1]["actual_tool_xz_m"]["A"],
        actual_B_xz_m=trace["frames"][-1]["actual_tool_xz_m"]["B"],
        peak_motor={
            n: np.max(abs(np.asarray([f["actuator_effort"][n] for f in data])), axis=0).tolist()
            for n in ("A", "B")
        }
        if data
        else {},
        registered_pairs=sorted({f["registered_pairs"] for f in trace["self_samples"]}),
    )
    if boundary:
        windows = {}
        for f in data:
            obs = f["receiver_observation"] or {}
            selection = obs.get("boundary_selection")
            if selection is not None:
                windows[selection["decision_time_s"]] = selection
        patch_count = len(trace["patches"])
        weakest = 0.0
        peak_structure = {}
        bilateral = []
        for k, body in enumerate(sub):
            g = {s["patch"]: s for s in trace["samples"][k * patch_count : (k + 1) * patch_count]}
            lower = min(g[n]["normal_n"] for n in ("B_left", "B_right"))
            weakest = max(weakest, lower)
            if lower > 0.20:
                bilateral.append(body["t"])
            forces = {}
            for c in body["cloth_links"]:
                if c["robot"] == "B":
                    forces[c["link"]] = forces.get(c["link"], 0.0) + c["normal_force_n"]
            for link, value in forces.items():
                peak_structure[link] = max(peak_structure.get(link, 0.0), value)
        metrics.update(
            boundary_decisions=list(windows.values()),
            substep_B_weaker_pad_peak_n=weakest,
            actual_B_bilateral_substeps=len(bilateral),
            first_B_bilateral_substep_s=bilateral[0] if bilateral else None,
            substep_B_structural_normal_peaks_n=peak_structure,
        )
    trace["metrics"] = metrics
    trace["coordination_events"] = coord.events
    trace["completed"] = "failure" not in trace
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "xt", encoding="utf8", compresslevel=3) as stream:
        json.dump(trace, stream, allow_nan=False, separators=(",", ":"))
    summary.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps(metrics, allow_nan=False))
    return 0 if trace["completed"] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--frames", type=int)
    g.add_argument("--stop-time", type=float)
    ap.add_argument("--resume", type=Path)
    ap.add_argument(
        "--boundary",
        action="store_true",
        help="choose and latch a structurally checked topological material boundary",
    )
    args = ap.parse_args()
    return run(
        args.source,
        args.output,
        frames=args.frames,
        stop_time=args.stop_time,
        resume=args.resume,
        boundary=args.boundary,
    )


if __name__ == "__main__":
    raise SystemExit(main())
