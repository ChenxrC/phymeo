"""Run a classic Panda session. The operator can change the place target while the body is moving."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from phymeo.builders.panda import load_panda
from phymeo.core.capabilities import SolverFailure
from phymeo.core.rigid import RigidBodyState
from phymeo.core.robot import RobotState
from phymeo.dynamics.robot_dynamics import robot_dynamics_terms
from phymeo.dynamics.rotation import matrix_quaternion, quaternion_matrix
from phymeo.experiments.panda_cube import _command_at, _waypoints
from phymeo.runtime.robot_payload_scene import PadFace, RobotPayloadScene
from phymeo.solvers.sampled_payload import SampledPayload


def make_session(kind="box"):
    """Return the public scene and the operator's trajectory data for the selected load."""
    if kind not in ("box", "sphere"):
        raise ValueError("choose box or sphere")
    asset = load_panda()
    q0 = asset.configuration(np.pi * np.array([0, -0.25, 0, -0.75, 0, 0.5, 0.25]), 0.076)
    arm = RobotState(q0, np.zeros(9))
    feet = [asset.authored_link_pose("panda_" + side + "pad", q0)[0] for side in ("left", "right")]
    mid = np.mean(feet, axis=0)
    y = (feet[0] - feet[1]) / np.linalg.norm(feet[0] - feet[1])
    z = np.array([0.0, 0.0, 1.0])
    R = np.column_stack((y, np.cross(z, y), z))
    mass, closed, grip = (0.10, 0.029, 0.012) if kind == "box" else (0.04096, 0.024, 0.0)
    descriptor = (
        {"kind": "box", "full_size_m": [0.05, 0.05, 0.05]}
        if kind == "box"
        else {"kind": "sphere", "radius_m": 0.020}
    )
    payload = SampledPayload(payload_geometry=descriptor)
    table_top = mid[2] - 0.042 - 0.012 - 0.025
    center = np.array([*mid[:2], table_top + payload.half[2]])
    body = RigidBodyState(
        center[None], matrix_quaternion(R)[None], np.zeros((1, 3)), np.zeros((1, 3))
    )
    colliders = {c.name: c.shape.key for c in asset.collisions}
    pads = tuple(
        PadFace(
            colliders["panda_" + side + "finger_pad"],
            1,
            orientation,
            tuple(product((-0.005, 0.005), repeat=2)),
        )
        for side, orientation in (("left", -1), ("right", 1))
    )
    # Preserve the established task-space arm gains, distributing the diagonal part
    # to the public PD drive. The remainder is recomputed at the observed joint state.
    diag = np.diag(robot_dynamics_terms(asset.model, arm)[3]).copy()
    kp, kd = 900 * diag, 60 * diag
    kp[-2:], kd[-2:] = 620.0, 5.0
    table_center = np.r_[mid[:2] + [0.0375, 0.0], table_top - 0.008]
    table_half = (0.102, 0.055, 0.008)
    scene = RobotPayloadScene(
        asset,
        robot_state=arm,
        payload_state=body,
        payload_geometry=descriptor,
        mass_kg=mass,
        table_center_m=table_center,
        table_half_extents_m=table_half,
        dt_s=1 / 1600,
        kp=kp,
        kd=kd,
        pads=pads,
    )
    return scene, dict(
        asset=asset,
        q0=q0,
        diagonal=diag,
        mass=mass,
        top=table_top,
        initial=center,
        table_center=table_center,
        table_half=table_half,
        payload=payload,
        descent=mid[2] - (center[2] + grip),
        closed=closed,
    )


def _goal(info, travel):
    return _waypoints(
        info["asset"], info["q0"], info["descent"], 0.065, travel, 0.076, info["closed"]
    )


def _amended_segment(t, begin, end, q, v, goal):
    # Connect the live command (position and slope) to the revised stopped endpoint.
    h = end - begin
    u = (t - begin) / h
    s = u**3 * (10 + u * (-15 + 6 * u))
    ds = 30 * u**2 * (1 - u) ** 2 / h
    b = u * (1 - u) ** 3 * (1 + 3 * u)
    db = (1 - u) ** 2 * (1 + 2 * u - 15 * u * u)
    return q + s * (goal - q) + h * b * v, ds * (goal - q) + db * v


def run_session(scene, info, choose_target, *, on_event=None):
    """The operator sees the actual in-flight receipt before selecting a new horizontal place offset.

    on_event(label, scene) gives access to the committed state and exact camera geometry.
    choose_target(reading) returns a destination in metres from the original object centre.
    The run records an unmet lift or rejected physical step separately from placement.
    """
    nominal = 0.075
    times, knots = _goal(info, nominal)
    final_goal = nominal
    switch = 1.05
    dt = scene.dt_s
    trace, events, transfer, final = [], {}, [], []
    joins = None
    first_failure = None
    first_structure = None
    first_table_link = None
    changed = False
    payload = info["payload"]

    def bottom(read):
        return float(
            payload.low_z(read.payload_position_m, quaternion_matrix(read.payload_xyzw[None])[0])
            - info["top"]
        )

    def emit(name, read):
        bilateral = (
            read.pad_normal_loads_N.reshape(2, 4).sum(axis=1) if read.step_number else np.zeros(2)
        )
        events[name] = dict(
            time_s=read.time_s,
            load_time_s=read.load_time_s,
            q=read.q.tolist(),
            payload_position_m=read.payload_position_m.tolist(),
            payload_xyzw=read.payload_xyzw.tolist(),
            clearance_m=bottom(read),
            bilateral_normal_N=bilateral.tolist(),
            table_force_N=float(read.payload_table_wrench[2]),
        )
        events[name].update(
            v=read.v.tolist(),
            payload_velocity_m_s=read.payload_velocity_m_s.tolist(),
            payload_omega_rad_s=read.payload_omega_rad_s.tolist(),
            command=read.target_q.tolist(),
            target_velocity=read.target_velocity.tolist(),
            effort_limit=read.effort_limit.tolist(),
            actuator_effort=read.drive_effort.tolist(),
            contact_effort=read.contact_effort.tolist(),
            pad_points_m=read.pad_points_m.tolist(),
            forces_on_pad_N=read.pad_forces_on_link_N.tolist(),
            payload_table_wrench=read.payload_table_wrench.tolist(),
            fixed_table_wrench=read.fixed_table_wrench.tolist(),
            structural_contacts=[dict(row) for row in read.structural_contacts],
            link_table_contacts=[dict(row) for row in read.link_table_contacts],
        )
        snapshot = scene.geometry()
        events[name]["payload_key"] = (
            snapshot.payload_key.world,
            snapshot.payload_key.body,
            snapshot.payload_key.shape,
        )
        events[name]["table_key"] = (
            snapshot.table_key.world,
            snapshot.table_key.body,
            snapshot.table_key.shape,
        )
        events[name]["static_motions"] = [
            dict(
                key=[shape.key.world, shape.key.body, shape.key.shape],
                name=snapshot.names[shape.key],
                kind=shape.kind,
                vertex_ids=shape.vertex_ids,
                radius_m=shape.radius,
                group=shape.group,
                mask=shape.mask,
                start_m=snapshot.poses[shape.key].start.tolist(),
                body_orientation_xyzw=snapshot.poses[shape.key].body_orientation_xyzw,
            )
            for shape in snapshot.model.shapes
        ]
        if on_event is not None:
            on_event(name, scene)

    emit("initial", scene.observe())
    for step in range(round(times[-1] / dt)):
        t = step * dt
        sample = scene.observe()
        if step == round(switch / dt):
            emit("moving_before_change", sample)
            if (
                bottom(sample) > 0.035
                and np.all(sample.pad_normal_loads_N.reshape(2, 4).sum(axis=1) > 0)
                and sample.payload_table_wrench[2] < 0.001
            ):
                destination = float(choose_target(sample))
                if not np.isfinite(destination) or not 0.045 <= destination <= 0.085:
                    raise ValueError("choose a place offset within the declared 45–85 mm workspace")
                joins = _command_at(times, knots, t)
                times, knots = _goal(info, destination)
                final_goal = destination
                changed = destination != nominal
                events["moving_before_change"]["old_destination_m"] = nominal
                events["moving_before_change"]["new_destination_m"] = destination
            else:
                first_failure = dict(
                    time_s=t,
                    reason="payload did not reach the supported in-flight change condition",
                )
        if joins is not None and switch <= t < times[4]:
            qcmd, vcmd = _amended_segment(t, switch, times[4], *joins, knots[4])
        else:
            qcmd, vcmd = _command_at(times, knots, t)
        arm = RobotState(sample.q, sample.v)
        _, _, _, M, bias = robot_dynamics_terms(info["asset"].model, arm)
        acceleration = 900 * (qcmd - arm.q) + 60 * (vcmd - arm.v)
        ff = bias + (M - np.diag(info["diagonal"])) @ acceleration
        ff[-2:] = bias[-2:]
        try:
            read = scene.step(qcmd, target_velocity=vcmd, feedforward=ff)
        except (ValueError, RuntimeError, SolverFailure) as exc:
            first_failure = dict(time_s=t, reason=str(exc))
            break
        if read.structural_contacts and first_structure is None:
            row = read.structural_contacts[0]
            first_structure = dict(time_s=t, shape=row["shape"], force_N=row["force_on_payload_N"])
        if read.link_table_contacts and first_table_link is None:
            row = read.link_table_contacts[0]
            first_table_link = dict(time_s=t, shape=row["shape"], force_N=row["force_on_table_N"])
        normals = read.pad_normal_loads_N.reshape(2, 4).sum(axis=1)
        if 1.26 <= read.time_s <= 1.35:
            transfer.append(
                (
                    bottom(read),
                    read.payload_position_m[0] - info["initial"][0],
                    min(normals),
                    read.payload_table_wrench[2],
                )
            )
        if read.time_s >= 2.15:
            final.append(
                (
                    read.payload_table_wrench[2],
                    np.linalg.norm(read.pad_forces_on_link_N, axis=1).sum(),
                )
            )
        if read.step_number % 16 == 0:
            trace.append(
                dict(
                    time_s=read.time_s,
                    load_time_s=read.load_time_s,
                    q=read.q.tolist(),
                    v=read.v.tolist(),
                    target_q=qcmd.tolist(),
                    effort=read.drive_effort.tolist(),
                    payload_position_m=read.payload_position_m.tolist(),
                    payload_xyzw=read.payload_xyzw.tolist(),
                    payload_velocity_m_s=read.payload_velocity_m_s.tolist(),
                    payload_omega_rad_s=read.payload_omega_rad_s.tolist(),
                    normal_N=normals.tolist(),
                    table_N=read.payload_table_wrench[2],
                )
            )
        if read.step_number in (
            round(0.55 / dt),
            round(0.92 / dt),
            round(1.35 / dt),
            round(1.95 / dt),
        ):
            emit(
                {
                    round(0.55 / dt): "close",
                    round(0.92 / dt): "lift",
                    round(1.35 / dt): "transfer",
                    round(1.95 / dt): "open",
                }[read.step_number],
                read,
            )
    last = scene.observe()
    emit("end", last)
    error = float(np.linalg.norm(last.payload_position_m - info["initial"] - [final_goal, 0, 0]))
    weight = info["mass"] * 9.81
    carried = bool(
        transfer
        and min(row[0] for row in transfer) > 0.035
        and transfer[-1][1] > 0.045
        and min(row[2] for row in transfer) > 0
        and max(row[3] for row in transfer) < 0.001
    )
    settled = bool(
        final
        and error < 0.012
        and max(row[1] for row in final) < 0.001
        and abs(np.mean([row[0] for row in final]) - weight) < 0.025
    )
    return dict(
        accepted_steps=last.step_number,
        time_s=last.time_s,
        dt_s=scene.dt_s,
        payload_geometry=payload.descriptor(),
        mass_kg=info["mass"],
        joint_names=info["asset"].joint_names,
        gravity_m_s2=info["asset"].model.gravity.tolist(),
        table_center_m=info["table_center"].tolist(),
        table_half_extents_m=info["table_half"],
        table_xyzw=[0, 0, 0, 1],
        coordinates="world metres, XYZW body-to-world, rad or m per joint, force N and COM torque N m",
        asset_collisions=[
            dict(
                key=[c.shape.key.world, c.shape.key.body, c.shape.key.shape],
                name=c.name,
                owner_link=c.link_name,
                local_vertices_m=c.local_vertices,
                local_xyz_m=c.origin_xyz,
                local_xyzw=c.origin_xyzw,
            )
            for c in info["asset"].collisions
        ],
        nominal_destination_m=nominal,
        selected_destination_m=final_goal,
        changed_in_flight=changed,
        carried=carried,
        placed=settled,
        qualified=bool(
            carried
            and settled
            and changed
            and first_failure is None
            and first_structure is None
            and first_table_link is None
        ),
        target_error_m=error,
        mean_final_table_N=float(np.mean([row[0] for row in final])) if final else None,
        max_final_pad_N=float(max(row[1] for row in final)) if final else None,
        first_failure=first_failure,
        first_robot_payload_structure=first_structure,
        first_robot_table_contact=first_table_link,
        events=events,
        trace=trace,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=("box", "sphere"), default="box")
    parser.add_argument("--place-offset-m", type=float, default=0.055)
    parser.add_argument(
        "--interactive", action="store_true", help="read a new place offset after the actual lift"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene, info = make_session(args.shape)

    def choose_target(reading):
        if args.interactive:
            print(
                "Accepted at",
                reading.time_s,
                "s with actual finger forces",
                reading.pad_normal_loads_N.reshape(2, 4).sum(axis=1),
                "N",
            )
            return float(input("New place offset along world X (0.045 to 0.085 m): "))
        return args.place_offset_m

    result = run_session(scene, info, choose_target)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("events", "trace", "asset_collisions")}
        )
    )
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
