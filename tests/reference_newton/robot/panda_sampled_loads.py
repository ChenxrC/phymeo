"""Run Featherstone on its own Panda state using Phymeo's finite material-point contact law."""

import argparse
import hashlib
import json
from importlib.metadata import version
from itertools import product
from pathlib import Path

import newton
import numpy as np
from panda_loaded_box import make_model, named

from phymeo.builders.panda import load_panda
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotState
from phymeo.dynamics.robot_dynamics import robot_dynamics_terms
from phymeo.dynamics.robot_kinematics import _point_jacobians_at
from phymeo.dynamics.rotation import quaternion_matrix
from phymeo.experiments.panda_cube import (
    _command_at,
    _finger_samples,
    _PassiveGeometry,
    _servo_effort,
    _world_points,
)
from phymeo.solvers.sampled_box_contact import box_surface_samples, sample_loads


def box_corners(extents):
    return np.array(list(product(*[(-h, h) for h in extents])))


def tabletop_load(vertices, center, rotation, twist, table_center, table_half, law):
    arms = vertices @ rotation.T
    world = center + arms
    if np.any(np.abs(world[:, :2] - table_center[:2]) > table_half[:2]):
        raise ValueError("payload corner left the finite tabletop footprint")
    velocity = twist[:3] + np.cross(twist[3:], arms)
    normal = np.tile([0.0, 0.0, 1.0], (len(world), 1))
    loads, forces = sample_loads(
        table_center[2] + table_half[2] - world[:, 2], normal, velocity, law
    )
    return forces, loads, arms


def finger_material(model, pads):
    """Use the imported pad boxes, including their body-local orientation and actual extent."""
    body = model.shape_body.numpy()
    local = model.shape_transform.numpy().astype(float)
    shape_rot = quaternion_matrix(local[pads, 3:])
    half = model.shape_scale.numpy()[pads]
    owners, material = [], []
    for index, (pad, sign) in enumerate(zip(pads, (-1, 1), strict=True)):
        if np.any(half[index, [0, 2]] <= 0.005):
            raise ValueError("the four material samples do not fit the imported fingertip")
        patch = np.array(
            [[x, sign * half[index, 1], z] for x, z in product((-0.005, 0.005), repeat=2)]
        )
        owners.extend([body[pad]] * 4)
        material.extend(local[pad, :3] + patch @ shape_rot[index].T)
    return np.array(owners), np.array(material)


def sampled_wrenches(
    body, twist, com_local, owners, material, payload, ext, corners, cn, half, laws
):
    rotations = quaternion_matrix(body[:, 3:])
    com = body[:, :3] + np.einsum("nij,nj->ni", rotations, com_local)
    points = body[owners, :3] + np.einsum("nij,nj->ni", rotations[owners], material)
    pad_arms = points - com[owners]
    speeds = twist[owners, :3] + np.cross(twist[owners, 3:], pad_arms)
    box_rot, vel = rotations[payload], twist[payload]
    contact = box_surface_samples(
        points, speeds, com[payload], box_rot, vel[:3], vel[3:], ext, laws[0]
    )
    table_f, table_n, corner_arms = tabletop_load(
        corners, com[payload], box_rot, vel, cn, half, laws[1]
    )

    w = np.zeros_like(twist)
    np.add.at(w[:, :3], owners, contact.forces)
    np.add.at(w[:, 3:], owners, np.cross(pad_arms, contact.forces))
    w[payload, :3] = contact.body_force
    w[payload, 3:] = contact.body_torque
    pad_f_balance = np.linalg.norm(w[:, :3].sum(axis=0))
    # Include each body's COM lever before summing moments about the payload COM.
    pad_t_balance = np.linalg.norm((w[:, 3:] + np.cross(com - com[payload], w[:, :3])).sum(axis=0))
    w[payload, :3] += table_f.sum(axis=0)
    w[payload, 3:] += np.cross(corner_arms, table_f).sum(axis=0)
    object_point_v = vel[:3] + np.cross(vel[3:], points - com[payload])
    contact_power = np.sum(contact.forces * (speeds - object_point_v))
    table_power = np.sum(table_f * (vel[:3] + np.cross(vel[3:], corner_arms)))
    power_error = abs(np.sum(w * twist) - contact_power - table_power)
    return w, points, speeds, contact, table_n, (pad_f_balance, pad_t_balance, power_error)


def run(urdf, inputs, device):
    spec = json.loads(inputs.read_text(encoding="utf8"))
    assert spec["schema"] == "panda-box-control-v1" and "trace" not in spec
    assert hashlib.sha256(urdf.read_bytes()).hexdigest() == spec["urdf_sha256"]
    assert version("newton") == "1.7.0"
    asset = load_panda(urdf)
    model, state, iq, iv, pads, _, payload, _, _ = make_model(urdf, spec, device)
    joints = [named(model.joint_label, n) for n in spec["joint_names"]]
    native_child = model.joint_child.numpy()[joints]
    np.testing.assert_allclose(
        model.body_mass.numpy()[native_child], asset.model.mass, rtol=2e-6, atol=1e-7
    )
    next_state, control = model.state(), model.control()
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    assert str(state.body_f.device) == str(state.joint_q.device) == device
    owners, material = finger_material(model, pads)
    com_local = model.body_com.numpy().astype(float)
    # The box has its origin at its COM in the shared model.
    np.testing.assert_array_equal(com_local[payload], np.zeros(3))
    ext = np.array(spec["full_size_m"]) / 2
    corners = box_corners(ext)
    cn, half = np.array(spec["table_center_m"]), np.array(spec["table_full_size_m"]) / 2
    laws = [RobotContactMaterial(*spec[k]) for k in ("pad_material", "table_material")]
    frame_ids, local_points = _finger_samples(asset)
    passive = _PassiveGeometry(asset)
    dt, n = spec["dt_s"], spec["steps"]
    gravity, initial = np.array(spec["gravity_m_s2"]), np.array(spec["body_position_m"])
    times, knots = np.array(spec["knot_times_s"]), np.array(spec["knot_q"])
    eff, lo, hi = np.array(asset.effort_limit), np.array(asset.lower), np.array(asset.upper)
    tau = np.zeros_like(control.joint_f.numpy())
    verify = set(np.linspace(0, n - 1, 17, dtype=int))
    pad_balance = np.zeros(3)
    pos_error = vel_error = 0.0
    momentum_error = max_limits = max_depth = 0.0
    other_margin = table_margin = np.inf
    rows, stop = [], None
    for step in range(n):
        t = step * dt
        q, v = state.joint_q.numpy()[iq].astype(float), state.joint_qd.numpy()[iv].astype(float)
        body, twist = state.body_q.numpy().astype(float), state.body_qd.numpy().astype(float)
        arm = RobotState(q, v)
        geometry, _, _, mass, bias = robot_dynamics_terms(asset.model, arm)
        qcmd, vcmd = _command_at(times, knots, t)
        tau[iv] = _servo_effort(arm, qcmd, vcmd, mass, bias, eff)
        try:
            w, points, speeds, pad, table_n, audit = sampled_wrenches(
                body, twist, com_local, owners, material, payload, ext, corners, cn, half, laws
            )
        except ValueError as exc:
            stop = dict(step=step, t=t, reason=str(exc), q=q.tolist(), box=body[payload].tolist())
            break
        pad_balance = np.maximum(pad_balance, audit)
        if step in verify:
            # Read-only comparison at Newton's own instantaneous q/v; neither feeds contact forces.
            jac, _ = _point_jacobians_at(asset.model, geometry, frame_ids, local_points)
            pos_error = max(
                pos_error,
                float(np.max(np.abs(_world_points(geometry, frame_ids, local_points) - points))),
            )
            vel_error = max(vel_error, float(np.max(np.abs(jac @ v - speeds))))
        gap, table_gap, _, _ = passive.clearances(
            geometry,
            body[payload, :3],
            quaternion_matrix(body[payload, None, 3:])[0],
            ext,
            cn,
            half,
        )
        other_margin, table_margin = min(other_margin, gap), min(table_margin, table_gap)
        state.clear_forces()
        state.body_f.assign(w.astype(np.float32))
        control.joint_f.assign(tau)
        # Sole load routes: the servo uses joint_f, all paired surface forces use body_f.
        # No collide() and no contacts buffer are supplied to this native integration step.
        solver.step(state, next_state, control, None, dt)
        state, next_state = next_state, state
        result = state.body_q.numpy()[payload].astype(float)
        speed = state.body_qd.numpy()[payload].astype(float)
        actual_q, actual_v = (
            state.joint_q.numpy()[iq].astype(float),
            state.joint_qd.numpy()[iv].astype(float),
        )
        if not np.isfinite(np.r_[result, speed, actual_q, actual_v]).all():
            raise ArithmeticError(f"nonfinite native state after physical step {step + 1}")
        max_limits = max(max_limits, float(np.max(lo - actual_q)), float(np.max(actual_q - hi)))
        max_depth = max(max_depth, float(np.max(pad.depths)))
        impulse_error = (
            spec["mass_kg"] * (speed[:3] - twist[payload, :3] - dt * gravity) - dt * w[payload, :3]
        )
        momentum_error = max(momentum_error, float(np.linalg.norm(impulse_error)))
        clearance = (
            result[2]
            + (corners @ quaternion_matrix(result[None, 3:])[0].T)[:, 2].min()
            - (cn[2] + half[2])
        )
        rows.append(
            dict(
                t=(step + 1) * dt,
                q=actual_q.tolist(),
                v=actual_v.tolist(),
                cube=result[:3].tolist(),
                xyzw=result[3:].tolist(),
                velocity=speed[:3].tolist(),
                omega=speed[3:].tolist(),
                pad_normal_N=pad.normal_loads.reshape(2, 4).sum(axis=1).tolist(),
                table_N=float(table_n.sum()),
                pad_total_N=float(np.linalg.norm(pad.forces, axis=1).sum()),
                clearance_m=float(clearance),
                applied_payload_wrench=w[payload].tolist(),
                drive=tau[iv].astype(float).tolist(),
            )
        )
    weight = spec["mass_kg"] * abs(gravity[2])
    carry = next((r for r in rows if abs(r["t"] - 1.35) < dt / 2), None)
    during = [r for r in rows if 0.95 <= r["t"] <= 1.35]
    release = [r for r in rows if r["t"] > times[-1] - 0.05]
    clear = None if carry is None else carry["clearance_m"]
    travel = None if carry is None else carry["cube"][0] - initial[0]
    bilateral = min((min(r["pad_normal_N"]) for r in during), default=None)
    table_peak = max((r["table_N"] for r in during), default=None)
    placement = (
        float(np.linalg.norm(np.array(rows[-1]["cube"]) - initial - [spec["travel_m"], 0, 0]))
        if rows
        else None
    )
    final_table = float(np.mean([r["table_N"] for r in release])) if release else None
    final_pad = max((r["pad_total_N"] for r in release), default=None)
    carried = bool(
        carry is not None
        and clear > 0.035
        and travel > 0.045
        and bilateral > 0
        and table_peak < 0.001
    )
    placed = bool(
        release and placement < 0.012 and final_pad < 0.001 and abs(final_table - weight) < 0.025
    )
    return dict(
        engine="newton-1.7.0",
        solver="SolverFeatherstone",
        device=device,
        steps=len(rows),
        simulated_seconds=len(rows) * dt,
        cube_full_size_m=spec["full_size_m"],
        cube_mass_kg=spec["mass_kg"],
        closed_width_m=float(np.min(knots[:, -2:].sum(axis=1))),
        finger_friction=spec["pad_material"][2],
        table_center_m=cn.tolist(),
        table_full_size_m=spec["table_full_size_m"],
        stop=stop,
        inputs_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
        urdf_sha256=spec["urdf_sha256"],
        contact="shared HOST material laws on Newton pose/COM twist; paired body_f, solver contacts=None",
        carry_clearance_m=clear,
        carry_displacement_m=travel,
        min_carry_bilateral_normal_N=bilateral,
        peak_carry_table_N=table_peak,
        final_target_error_m=placement,
        final_table_force_N=final_table,
        release_pad_force_N=final_pad,
        peak_bilateral_normal_N=max((min(r["pad_normal_N"]) for r in rows), default=None),
        max_pad_penetration_m=max_depth,
        other_robot_cube_axis_clearance_m=other_margin,
        robot_table_axis_clearance_m=table_margin,
        max_joint_limit_violation=max_limits,
        carried_during_transfer=carried,
        released_on_target=placed,
        success=bool(
            len(rows) == n
            and carried
            and placed
            and other_margin > 0
            and table_margin > 0
            and max_limits < 1e-4
        ),
        max_payload_impulse_error_Ns=momentum_error,
        max_pad_pair_force_error_N=pad_balance[0],
        max_pad_pair_torque_error_Nm=pad_balance[1],
        max_power_error_W=pad_balance[2],
        max_site_position_error_m=pos_error,
        max_site_velocity_error_m_s=vel_error,
        trace=rows,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf", required=True, type=Path)
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--expect", choices=("carry", "slip"), required=True)
    args = p.parse_args()
    result = run(args.urdf, args.inputs, args.device)
    with args.out.open("x", encoding="utf8") as out:
        json.dump(result, out, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}, allow_nan=False))
    assert result["stop"] is None and result["steps"] == 3600
    assert (
        max(
            result[k]
            for k in (
                "max_pad_pair_force_error_N",
                "max_pad_pair_torque_error_Nm",
                "max_power_error_W",
            )
        )
        < 1e-10
    )
    assert result["max_payload_impulse_error_Ns"] < 2e-5
    assert (
        result["max_site_position_error_m"] < 2e-5 and result["max_site_velocity_error_m_s"] < 2e-4
    )
    assert result["success"] == (args.expect == "carry")
    if args.expect == "slip":
        assert result["carry_clearance_m"] < 0.005 and abs(result["carry_displacement_m"]) < 0.015
