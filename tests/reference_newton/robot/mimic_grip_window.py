"""Close, lift under gravity, lower and release with one mechanical actuator and two passive faces."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from mimic_oracle import grip_command, kkt

from phymeo.builders.mimic_urdf import load_mimic_urdf
from phymeo.core.capabilities import SolverFailure
from phymeo.core.rigid import RigidBodyState
from phymeo.core.robot import RobotState
from phymeo.runtime.robot_payload_scene import PadFace, RobotPayloadScene

SOURCE = Path(__file__).with_name("mimic_jaws.urdf")
H = 0.001
CENTER = np.array([0.25, -0.125, 0.25])
WEIGHT = 0.1 * 9.81


def run(cap):
    asset = load_mimic_urdf(SOURCE)
    cols = {c.name: c for c in asset.tree.collisions}
    pads = [
        PadFace(
            cols[side + "_rubber"].shape.key, 0, sign, tuple(product((-0.005, 0.005), repeat=2))
        )
        for side, sign in (("left", 1), ("right", -1))
    ]
    body = RigidBodyState(
        CENTER[None], np.array([[0.0, 0.0, 0.0, 1.0]]), np.zeros((1, 3)), np.zeros((1, 3))
    )
    scene = RobotPayloadScene.from_urdf(
        SOURCE,
        loader=load_mimic_urdf,
        robot_state=RobotState(np.zeros(2), np.zeros(2)),
        payload_state=body,
        payload_geometry={"kind": "box", "full_size_m": [0.04, 0.04, 0.04]},
        mass_kg=0.1,
        table_center_m=(0.25, -0.125, 0.22),
        table_half_extents_m=(0.12, 0.12, 0.01),
        dt_s=H,
        kp=[2700, 6000],
        kd=[95, 35],
        pads=pads,
    )
    trace = []
    held = []
    end = []
    errors = {
        k: 0.0
        for k in (
            "acceleration",
            "constraint_force",
            "relative_q",
            "relative_v",
            "constraint_power",
            "full_contact",
            "payload_impulse",
        )
    }
    peak_constraint = 0.0
    peak_bilateral = 0.0
    other_hits = 0
    fail = None
    last_bilateral = first_unloaded = None
    for k in range(1720):
        before = scene.observe()
        qcmd, vcmd = grip_command(k * H)
        try:
            r = scene.step(qcmd, target_velocity=vcmd, effort_limit=[40, cap])
        except (ValueError, RuntimeError, SolverFailure) as exc:
            geometry = scene.geometry()
            payload_high = float(geometry.poses[geometry.payload_key].start[:, 2].max())
            pad_low = min(geometry.poses[c.shape.key].start[:, 2].min() for c in cols.values())
            # For these independent translational jaws the motor sees 0.2+0.3 kg.
            # This bound uses the greatest available braking force after physical separation.
            brake = cap / (0.2 + 0.3)
            min_next = before.q[1] + H * (before.v[1] - H * brake)
            fail = dict(
                step=k,
                time_s=before.time_s,
                reason=str(exc),
                atomic=scene.observe() is before,
                accepted_q=before.q.tolist(),
                accepted_v=before.v.tolist(),
                accepted_full_q=before.full_q.tolist(),
                previous_drive_N=before.drive_effort.tolist(),
                table_N=before.payload_table_wrench[2],
                payload_position_m=before.payload_position_m.tolist(),
                pad_normal_N=before.pad_normal_loads_N.tolist(),
                pad_bottom_above_payload_top_m=float(pad_low - payload_high),
                remaining_to_stop_m=asset.upper[1] - before.q[1],
                braking_distance_m=before.v[1] ** 2 / (2 * brake),
                smallest_next_q_m=float(min_next),
                upper_m=asset.upper[1],
            )
            break
        loads = r.mimic_loads
        hands = r.pad_forces_on_link_N.reshape(2, 4, 3).sum(axis=1)
        contact = np.array([hands[:, 2].sum(), hands[0, 0], hands[1, 0]])
        motor = np.r_[r.drive_effort, 0.0]
        acc, react = kkt(
            np.diag([1.5, 0.2, 0.3]), np.array([1.5 * 9.81, 0, 0]), contact, motor, [0, 1.0, 1.0]
        )
        errors["acceleration"] = max(
            errors["acceleration"], np.max(abs((r.full_v - before.full_v) / H - acc))
        )
        errors["constraint_force"] = max(
            errors["constraint_force"], np.max(abs(loads.constraint_effort_full - react))
        )
        errors["full_contact"] = max(
            errors["full_contact"], np.max(abs(loads.contact_effort_full - contact))
        )
        errors["relative_q"] = max(errors["relative_q"], abs(r.full_q[2] + r.full_q[1] - 0.004))
        errors["relative_v"] = max(errors["relative_v"], abs(r.full_v[2] + r.full_v[1]))
        errors["constraint_power"] = max(errors["constraint_power"], abs(loads.constraint_power_W))
        applied = r.payload_table_wrench[:3] - hands.sum(axis=0) - np.array([0, 0, WEIGHT])
        errors["payload_impulse"] = max(
            errors["payload_impulse"],
            np.max(abs(0.1 * (r.payload_velocity_m_s - before.payload_velocity_m_s) - H * applied)),
        )
        peak_constraint = max(peak_constraint, abs(loads.multiplier_force))
        bilateral = r.pad_normal_loads_N.reshape(2, 4).sum(axis=1)
        peak_bilateral = max(peak_bilateral, min(bilateral))
        if min(bilateral) > 0:
            last_bilateral = r.load_time_s
            first_unloaded = None
        elif last_bilateral is not None and first_unloaded is None:
            first_unloaded = r.load_time_s
        other_hits += len(r.structural_contacts) + len(r.link_table_contacts)
        # Read the lowest of all eight material corners, including any rigid tilt.
        pose = scene.geometry() if r.step_number % 10 == 0 else None
        if pose:
            clearance = float(pose.poses[pose.payload_key].start[:, 2].min() - 0.23)
            trace.append(
                dict(
                    time_s=r.time_s,
                    load_time_s=r.load_time_s,
                    q=r.q.tolist(),
                    v=r.v.tolist(),
                    full_q=r.full_q.tolist(),
                    full_v=r.full_v.tolist(),
                    target_q=r.target_q.tolist(),
                    drive_full=loads.actuator_effort_full.tolist(),
                    constraint_full=loads.constraint_effort_full.tolist(),
                    contact_full=loads.contact_effort_full.tolist(),
                    pad_points_m=r.pad_points_m.tolist(),
                    pad_forces_N=r.pad_forces_on_link_N.tolist(),
                    bilateral_N=bilateral.tolist(),
                    payload_xyz=r.payload_position_m.tolist(),
                    payload_xyzw=r.payload_xyzw.tolist(),
                    payload_velocity_m_s=r.payload_velocity_m_s.tolist(),
                    clearance_m=clearance,
                    table_wrench=r.payload_table_wrench.tolist(),
                )
            )
            if 0.8 <= r.time_s <= 0.94:
                held.append((clearance, min(bilateral), r.payload_table_wrench[2]))
            if r.time_s >= 1.64:
                end.append((clearance, max(bilateral), r.payload_table_wrench[2]))
    return dict(
        motor_cap_N=cap,
        h_s=H,
        accepted_steps=scene.observe().step_number,
        first_failure=fail,
        last_bilateral_load_time_s=last_bilateral,
        first_unloaded_after_bilateral_s=first_unloaded,
        full_window_completed=scene.observe().step_number == 1720,
        other_contacts=other_hits,
        peak_passive_reaction_N=float(peak_constraint),
        peak_bilateral_normal_N=float(peak_bilateral),
        min_held_clearance_m=float(min(row[0] for row in held)) if held else None,
        max_held_clearance_m=float(max(row[0] for row in held)) if held else None,
        min_held_bilateral_N=float(min(row[1] for row in held)) if held else None,
        max_held_table_N=float(max(row[2] for row in held)) if held else None,
        min_held_table_N=float(min(row[2] for row in held)) if held else None,
        final_max_pad_N=float(max(row[1] for row in end)) if end else None,
        final_mean_table_N=float(np.mean([row[2] for row in end])) if end else None,
        max_errors=dict((k, float(v)) for k, v in errors.items()),
        trace=trace,
    )


def audit(r, strong):
    assert r["accepted_steps"] == 1720 and r["first_failure"] is None and r["other_contacts"] == 0
    assert max(r["max_errors"].values()) < 1e-8
    assert r["peak_bilateral_normal_N"] > 0.05
    assert r["final_max_pad_N"] < 0.001 and abs(r["final_mean_table_N"] - WEIGHT) < 0.025
    if strong:
        assert r["min_held_clearance_m"] > 0.035 and r["min_held_bilateral_N"] > 1.0
        assert r["max_held_table_N"] < 0.001 and r["peak_passive_reaction_N"] > 0.5
    else:
        assert r["max_held_clearance_m"] < 0.002 and r["min_held_table_N"] > 0.65


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("strong", "weak"), required=True)
    parser.add_argument(
        "--expect",
        choices=("full-window", "loss-and-limit"),
        default="full-window",
        help="the documented low-force limit response is a separate acceptance target",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    strong = args.case == "strong"
    if strong and args.expect == "loss-and-limit":
        parser.error("loss-and-limit is a low-force diagnostic")
    result = run(8.0 if strong else 0.35)
    result["requested_check"] = args.expect
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}))
    if args.expect == "full-window":
        audit(result, strong)
    else:
        failure = result["first_failure"]
        assert failure and failure["atomic"] and not result["full_window_completed"]
        assert (
            result["last_bilateral_load_time_s"]
            < result["first_unloaded_after_bilateral_s"]
            < failure["time_s"]
        )
        assert failure["pad_bottom_above_payload_top_m"] > 0 and max(failure["pad_normal_N"]) == 0
        assert abs(failure["table_N"] - WEIGHT) < 0.025 and result["peak_bilateral_normal_N"] > 0
        assert failure["braking_distance_m"] > failure["remaining_to_stop_m"] > 0
        assert failure["smallest_next_q_m"] > failure["upper_m"]
        assert max(result["max_errors"].values()) < 1e-8 and result["other_contacts"] == 0


if __name__ == "__main__":
    main()
