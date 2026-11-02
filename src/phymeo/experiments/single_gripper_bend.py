"""Actual load and bending of a free sheet with one force-driven articulated gripper."""

import numpy as np

from ..constraints.bending import bending_angle_gradient
from ..core.capabilities import SolverFailure
from ..core.scene import SceneInputs, StepSettings
from .compliant_cloth import record_step, trace_start
from .compliant_handoff import _travel, build_gripper_sheet
from .handoff_grippers import BASE_Z, CLOSED


def build_single_gripper(
    *,
    grip_friction=0.9,
    initial_tilt_deg=2.0,
    finger_pad_radius_m=0.002,
    finger_backing_clearance_m=None,
):
    return build_gripper_sheet(
        ("A",),
        grip_friction=grip_friction,
        robot="articulated",
        fabric="soft",
        grid=None,
        initial_tilt_deg=initial_tilt_deg,
        finger_pad_radius_m=finger_pad_radius_m,
        finger_backing_clearance_m=finger_backing_clearance_m,
    )


def bend_measurements(program, x):
    bend = program.material.bending
    angles = [
        abs(bending_angle_gradient(x[ids])[0] - rest)
        for ids, rest in zip(bend.indices, bend.rest_angle, strict=True)
    ]
    singular = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
    return float(np.rad2deg(max(angles))), float(singular[-1] / np.sqrt(len(x)))


def loaded_bend(o, weight):
    return (
        o["bottom_m"] > 0.019
        and o["table_up_n"] < 0.02 * weight
        and o["jaw_min_normal_n"] > 0.20
        and o["jaw_up_n"] > 0.25 * weight
        and o["rest_hinge_change_deg"] > 30
        and o["plane_rms_m"] > 0.003
    )


def run_single_gripper(
    *,
    grip_friction=0.9,
    initial_tilt_deg=2.0,
    finger_pad_radius_m=0.002,
    finger_backing_clearance_m=None,
    progress=None,
):
    """Pinch at the initial pose, lift with bounded drives and measure gravity-driven bending."""
    case = build_single_gripper(
        grip_friction=grip_friction,
        initial_tilt_deg=initial_tilt_deg,
        finger_pad_radius_m=finger_pad_radius_m,
        finger_backing_clearance_m=finger_backing_clearance_m,
    )
    state = case.initial
    robot = case.grippers["A"]
    home = robot.position(state.domains["A"])
    size = np.ptp(state.domains["cloth"].position, axis=0)
    target = np.array([home[0] + 0.025, BASE_Z + np.linalg.norm(size) + 0.04])
    weight = float(np.sum(1 / case.program.cloth.inverse_mass) * 9.81)
    trace = trace_start(case, "single-gripper-loaded-bend")
    trace.update(
        fabric=case.fabric.record(case.grid),
        initial_tilt_deg=initial_tilt_deg,
        grip_friction=grip_friction,
    )
    phase = "grasp_A"
    entered = 0.0
    aperture = 0.0
    stable = lost = 0
    trace["events"].append(dict(phase=phase, t=0.0))
    with case.program.stepper(
        StepSettings(substeps=case.substeps, position_iterations=1, velocity_iterations=1)
    ) as runtime:
        for i in range(150):
            t = i * case.dt
            request = home if phase == "grasp_A" else _travel(home, target, t - entered, 0.55)
            aperture += float(np.clip(CLOSED - aperture, -0.09 * case.dt, 0.09 * case.dt))
            try:
                cmd, joint_target = robot.command(state.domains["A"], request, aperture, case.dt)
            except SolverFailure as exc:
                trace.update(completed=False, failure=str(exc), failure_phase=phase, time_s=t)
                break
            inputs = SceneInputs({"cloth": case.inputs.domains["cloth"], "A": cmd})
            out = runtime.step(state, inputs, case.dt)
            if not record_step(case, trace, out, t):
                trace["failure_phase"] = phase
                break
            state = out.state
            t = (i + 1) * case.dt
            x = state.domains["cloth"].position
            angle, rms = bend_measurements(case.program, x)
            rows = out.report.substeps[-1].diagnostics["patches"]
            jaws = rows["A_left"], rows["A_right"]
            actual = robot.position(state.domains["A"])
            o = dict(
                bottom_m=float(x[:, 2].min()),
                table_up_n=float(rows["table"]["force_n"][:, 2].sum()),
                jaw_min_normal_n=float(min(row["normal_force_n"].sum() for row in jaws)),
                jaw_up_n=float(sum(row["force_n"][:, 2].sum() for row in jaws)),
                rest_hinge_change_deg=angle,
                plane_rms_m=rms,
                tool_lift_m=float(actual[1] - home[1]),
                weight_n=weight,
            )
            trace["frames"][-1].update(
                phase=phase,
                measurements=o,
                requested_tool_xz_m={"A": request.tolist()},
                actual_tool_xz_m={"A": actual.tolist()},
                actuator_effort={"A": cmd.effort.tolist()},
                joint_target={"A": joint_target.tolist()},
            )
            if progress and (i + 1) % 10 == 0:
                progress(phase, t, o)
            if phase == "grasp_A":
                stable = stable + 1 if o["jaw_min_normal_n"] > 0.20 else 0
                if stable >= 2:
                    phase = "lift_A"
                    entered = t
                    stable = 0
                    trace["events"].append(dict(phase=phase, t=t, measurements=o))
            else:
                stable = stable + 1 if o["tool_lift_m"] > 0.03 and loaded_bend(o, weight) else 0
                if stable >= 3:
                    trace["events"].append(dict(phase="loaded_bend", t=t, measurements=o))
                    trace.update(completed=True, time_s=t)
                    break
                slipped = (
                    o["tool_lift_m"] > 0.04
                    and o["jaw_min_normal_n"] < 0.05
                    and o["bottom_m"] < 0.005
                    and o["table_up_n"] > 0.7 * weight
                )
                lost = lost + 1 if slipped else 0
                if lost >= 3:
                    trace.update(
                        completed=False,
                        time_s=t,
                        failure_phase=phase,
                        failure="gripper lost the sheet to the table",
                    )
                    break
        else:
            trace.update(
                completed=False,
                time_s=150 * case.dt,
                failure_phase=phase,
                failure="single-gripper load and bend gate not reached",
            )
    return trace
