"""Measure how a force-limited donor moves material while replaying identical receiver drives."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from benchmark_receiver_approach import accepted_delta, physical_path, read, restore

from phymeo.control.material_grasp import MaterialEdgeTarget
from phymeo.core.robot import RobotInputs
from phymeo.core.scene import SceneInputs, StepSettings
from phymeo.experiments.compliant_cloth import Case, record_step, trace_start
from phymeo.experiments.compliant_handoff import _observations, _shared_support, build_handoff
from phymeo.experiments.donor_material_authority import (
    DonorPulse,
    pad_coordinates,
    qualify_response,
)


def definitions(trace, case):
    frame = trace["frames"][0]
    obs = trace["frames"][1]["receiver_observation"]
    if "guidance_nodes" in obs:
        address = MaterialEdgeTarget(tuple(obs["guidance_nodes"]), obs["guidance_fraction"])
    elif "nodes" in obs:
        address = MaterialEdgeTarget(tuple(obs["nodes"]), obs["fraction"])
    else:
        raise ValueError("the accepted entry must carry an observed receiver material address")
    p = address.evaluate(frame["cloth_position_m"])
    np.testing.assert_allclose(p, obs["material_point_m"], rtol=0, atol=1e-13)
    group = next(
        x
        for row in trace["robot_samples"]
        for x in row["cloth_links"]
        if x["robot"] == "B" and x["normal_force_n"] > 0
    )
    site = next(s for s in group["sites"] if s["normal_force_n"] > 0)
    reserve = case.program.cloth_links.rho[0] + next(
        p.radius_m for p in case.program.patches if p.geometry.key == "B_left"
    )
    jaws = frame["patch_vertices_m"]
    win = pad_coordinates(jaws["B_left"], jaws["B_right"], p, reserve)
    entrance = win["center"] - p
    entrance -= win["axis"] * (entrance @ win["axis"])
    if np.linalg.norm(entrance) == 0:
        raise ValueError("no entry direction in the observed pad plane")
    entrance /= np.linalg.norm(entrance)
    budget = trace["metrics"]["motion_budget"]
    pulse = DonorPulse.toward(
        frame["actual_tool_xz_m"]["B"],
        p[[0, 2]],
        duration=6 * case.dt,
        peak_speed=budget["peak_speed"],
        peak_acceleration=budget["peak_acceleration"],
    )
    hazard = dict(
        link=group["link"],
        face=group["face"],
        normal=group["normal"],
        weights=site["face_weights"],
        axis_parameter=site["axis_parameter"],
    )
    return address, hazard, entrance, reserve, pulse


def drive(case, state, source_next, pulse, time, sign):
    gripper = case.grippers["A"]
    offset, rate = pulse.sample(time, sign)
    park = np.asarray(source_next["requested_tool_xz_m"]["A"])
    aperture = source_next["joint_target"]["A"][gripper.fingers[0]]
    a, aj = gripper.command(
        state.domains["A"],
        park + offset,
        aperture,
        case.dt,
        tool_velocity_m_s=rate if sign else None,
    )
    b = RobotInputs.zeros(case.grippers["B"].model.dof_count)
    b.effort[:] = source_next["actuator_effort"]["B"]
    for arm, command in (("A", a), ("B", b)):
        assert np.all(np.abs(command.effort) <= case.grippers[arm].effort_limits)
    return SceneInputs({**case.inputs.domains, "A": a, "B": b}), aj, offset, rate


def measure(frame, ss, pads, physical, address, hazard, clearance, weight, peak_a, peak_speed):
    x = np.asarray(frame["cloth_position_m"])
    v = np.asarray(frame["cloth_velocity_m_s"])
    point = address.evaluate(x)
    face = sum(w * x[i] for i, w in hazard["weights"])
    ends = np.asarray(frame["body_capsules_m"][hazard["link"]])
    u = hazard["axis_parameter"]
    axis = (1 - u) * ends[0] + u * ends[1]
    normal = np.asarray(hazard["normal"])
    left, right = [frame["patch_vertices_m"][key] for key in ("B_left", "B_right")]
    win = pad_coordinates(left, right, point, clearance)
    summaries = []
    for k in range(0, len(pads), len(pads) // len(ss)):
        group = pads[k : k + len(pads) // len(ss)]
        values = {r["patch"]: r for r in group}
        assert len({r["t"] for r in group}) == 1
        a, b = values["A_left"], values["A_right"]
        bl, br = values["B_left"], values["B_right"]
        summaries.append(
            (
                min(a["normal_n"], b["normal_n"]),
                a["force_n"][2] + b["force_n"][2],
                min(bl["normal_n"], br["normal_n"]),
                bl["normal_n"] + br["normal_n"],
            )
        )
    sampled = np.asarray(summaries)
    contact = next(
        (
            (s["t"], g["link"], g["face"])
            for s in ss
            for g in s["cloth_links"]
            if g["robot"] == "B" and g["normal_force_n"] > 0
        ),
        None,
    )
    peak = max(sum(g["normal_force_n"] for g in s["cloth_links"] if g["robot"] == "B") for s in ss)
    return dict(
        end_s=frame["t"],
        material_m=point.tolist(),
        material_velocity_m_s=address.evaluate(v).tolist(),
        jaw_center_m=win["center"].tolist(),
        jaw_axis=win["axis"].tolist(),
        donor_m=frame["actual_tool_xz_m"]["A"],
        side_margin_m=win["side_margin_m"],
        axial_margin_m=win["axial_margin_m"],
        hazard_face_point_m=face.tolist(),
        hazard_axis_point_m=axis.tolist(),
        hazard_relative_projection_m=float(normal @ (face - axis)),
        min_A_weaker_pad_n=float(sampled[:, 0].min()),
        mean_A_up_n=float(sampled[:, 1].mean()),
        peak_B_weaker_pad_n=float(sampled[:, 2].max()),
        peak_B_pad_n=float(sampled[:, 3].max()),
        first_structure_force_s=None if contact is None else contact[0],
        first_structure_pair=None if contact is None else contact[1:],
        peak_B_structure_n=peak,
        min_structure_m=physical["minimum_sampled_gap_m"],
        first_blocked=physical["first_blocked_drift"],
        peak_A_speed_m_s=peak_a,
        within_motion_budget=peak_a <= peak_speed * (1 + 1e-12)
        and physical["peak_sampled_tool_speed_m_s"] <= peak_speed * (1 + 1e-12),
        peak_joint_violation=max(s["joint_violation"] for s in ss),
        shared=_shared_support(frame["measurements"], weight),
    )


def study(args):
    reference = read(args.source)
    frames = reference["frames"]
    initial = restore(frames[0])
    tilt = reference["initial_tilt_deg"]
    case = build_handoff(
        robot=reference["robot_layout"],
        fabric=reference["fabric"]["name"],
        grid=reference["fabric"]["grid"],
        initial_tilt_deg=tilt,
        finger_pad_radius_m=0.002,
        finger_backing_clearance_m=0.000125,
        self_contact=True,
    )
    assert case.substeps == reference["substeps"] and case.dt == reference["dt_s"]
    address, hazard, entrance, reserve, pulse = definitions(reference, case)
    width = case.program.cloth_links.law.width_m
    weight = float(np.sum(1 / case.program.cloth.inverse_mass) * 9.81)
    budget = reference["metrics"]["motion_budget"]
    n = case.substeps
    b_schedule = np.array([f["actuator_effort"]["B"] for f in frames[1:]], dtype="<f8")
    checksum = hashlib.sha256(b_schedule.tobytes()).hexdigest()
    metadata = dict(
        source=str(args.source),
        tilt=tilt,
        sign=args.sign,
        entry_s=frames[0]["t"],
        dt=case.dt,
        substeps=n,
        width=width,
        weight=weight,
        pulse_duration_s=pulse.duration,
        pulse_displacement_m=pulse.distance,
        pulse_axis_xz=pulse.direction,
        material_nodes=address.nodes,
        material_fraction=address.fraction,
        historical_hazard=hazard,
        entrance_direction=entrance.tolist(),
        B_schedule_sha256=checksum,
    )
    active = args.sign != 0
    trace = (
        trace_start(
            Case(case.program, initial, case.inputs, case.dt, n), "donor-material-authority"
        )
        if active
        else None
    )
    if active:
        trace.update(
            frames=[dict(frames[0])],
            receiver_substep_motion=[],
            events=[dict(phase="accepted input", t=frames[0]["t"])],
        )
    state = initial
    rows = []
    repeats = []
    actual_count = 0
    status = "complete"
    settings = StepSettings(
        substeps=n, position_iterations=1, velocity_iterations=1, record_substeps=True
    )
    with case.program.stepper(settings) as stepper:
        for i, source_next in enumerate(frames[1:]):
            t = frames[i]["t"]
            if active or i in (0, len(frames) - 2):
                if not active:
                    state = restore(frames[i])
                inputs, aj, offset, rate = drive(
                    case, state, source_next, pulse, (i + 1) * case.dt, args.sign
                )
                np.testing.assert_array_equal(inputs.domains["B"].effort, b_schedule[i])
                if not active:
                    np.testing.assert_array_equal(
                        inputs.domains["A"].effort, source_next["actuator_effort"]["A"]
                    )
                result = stepper.step(state, inputs, case.dt)
                if not result.report.accepted:
                    status = result.report.reason
                    break
                actual_count += n
                if not active:
                    diff = accepted_delta(result.state, restore(source_next))
                    assert max(diff.values()) < 1e-11, diff
                    repeats.append(dict(entry_s=t, component_max_abs=diff))
            if active:
                assert record_step(case, trace, result, t)
                physical, samples = physical_path(case, state, result, t)
                trace["receiver_substep_motion"].extend(samples)
                peak_a = max(
                    np.linalg.norm(case.grippers["A"].velocity(s.domains["A"]))
                    for s in result.substep_states
                )
                state = result.state
                diag = result.report.substeps[-1].diagnostics
                f = trace["frames"][-1]
                f.update(
                    actual_tool_xz_m={
                        k: g.position(state.domains[k]).tolist() for k, g in case.grippers.items()
                    },
                    measurements=_observations(
                        state, diag["patches"], weight, diag["robot_contact"]
                    ),
                    actuator_effort={k: inputs.domains[k].effort.tolist() for k in "AB"},
                    A_joint_target=aj.tolist(),
                    A_reference_offset_m=offset.tolist(),
                    A_reference_rate_m_s=rate.tolist(),
                )
                source = trace
            else:
                physical = reference["metrics"]["rows"][i]["physical_path"]
                f = source_next
                source = reference
                # The neutral path uses the accepted archive. A's two endpoint rates are
                # reported as samples; its force channels still cover all original substeps.
                peak_a = max(
                    np.linalg.norm(case.grippers["A"].velocity(restore(ff).domains["A"]))
                    for ff in (frames[i], f)
                )
            pad_count = len(source["patches"])
            sr = source["robot_samples"][i * n : (i + 1) * n]
            pr = source["samples"][i * n * pad_count : (i + 1) * n * pad_count]
            assert len(sr) == n and len(pr) == n * pad_count
            row = measure(
                f,
                sr,
                pr,
                physical,
                address,
                hazard,
                reserve,
                weight,
                float(peak_a),
                budget["peak_speed"],
            )
            rows.append(row)
            print(
                json.dumps(
                    dict(
                        sign=args.sign,
                        t=f["t"],
                        min_structure_m=row["min_structure_m"],
                        first_force_s=row["first_structure_force_s"],
                        A_weak=row["min_A_weaker_pad_n"],
                        A_up=row["mean_A_up_n"],
                    )
                ),
                flush=True,
            )
    report = dict(
        **metadata,
        status=status,
        physical_steps_computed=actual_count,
        neutral_repeat=repeats,
        rows=rows,
    )
    if active:
        trace["metrics"] = report
        with gzip.open(
            args.out.with_suffix(".json.gz"), "xt", encoding="utf8", compresslevel=3
        ) as stream:
            json.dump(trace, stream, separators=(",", ":"), allow_nan=False)
    return report


def compare(args):
    reports = {r["sign"]: r for r in (json.loads(p.read_text("utf8")) for p in args.compare)}
    zero = reports[0]
    for key in (
        "source",
        "entry_s",
        "dt",
        "substeps",
        "width",
        "weight",
        "material_nodes",
        "material_fraction",
        "historical_hazard",
        "entrance_direction",
        "B_schedule_sha256",
    ):
        assert reports[-1][key] == zero[key] == reports[1][key], key
    assert all(r["status"] == "complete" for r in reports.values())
    base = next(i for i, r in enumerate(zero["rows"]) if r["first_structure_force_s"] is not None)
    response = qualify_response(
        zero["rows"],
        reports[1]["rows"],
        reports[-1]["rows"],
        zero["historical_hazard"]["normal"],
        zero["entrance_direction"],
        width=zero["width"],
        weight=zero["weight"],
        dt=zero["dt"],
        baseline_contact_index=base,
    )
    for sign in (1, -1):
        for b, r, delta in zip(
            zero["rows"], reports[sign]["rows"], response["response"][sign], strict=True
        ):
            delta.update(
                donor_delta_m=np.subtract(r["donor_m"], b["donor_m"]).tolist(),
                actual_side_margin_change_m=r["side_margin_m"] - b["side_margin_m"],
                actual_axial_margin_change_m=r["axial_margin_m"] - b["axial_margin_m"],
                hazard_face_minus_axis_change_m=r["hazard_relative_projection_m"]
                - b["hazard_relative_projection_m"],
                A_weaker_pad_n=r["min_A_weaker_pad_n"],
                A_mean_up_n=r["mean_A_up_n"],
            )
    return dict(
        source=zero["source"],
        initial_tilt_deg=zero["tilt"],
        B_schedule_sha256=zero["B_schedule_sha256"],
        baseline_contact_index=base,
        neutral_repeat=zero["neutral_repeat"],
        **response,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--sign", type=int, choices=(-1, 0, 1))
    p.add_argument("--compare", type=Path, nargs=3)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    observation = compare(args) if args.compare else study(args)
    with args.out.open("x", encoding="utf8") as stream:
        json.dump(observation, stream, indent=2, allow_nan=False)
    if observation.get("status", "complete") != "complete":
        raise SystemExit(1)
    if args.compare:
        print(json.dumps({k: v for k, v in observation.items() if k != "response"}))
