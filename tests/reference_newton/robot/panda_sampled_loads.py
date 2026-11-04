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
from phymeo.experiments.panda_payload import SampledPayload


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


def sampled_wrenches(body, twist, com_local, owners, material, payload, shape, cn, half, laws):
    rotations = quaternion_matrix(body[:, 3:])
    com = body[:, :3] + np.einsum("nij,nj->ni", rotations, com_local)
    points = body[owners, :3] + np.einsum("nij,nj->ni", rotations[owners], material)
    pad_arms = points - com[owners]
    speeds = twist[owners, :3] + np.cross(twist[owners, 3:], pad_arms)
    box_rot, vel = rotations[payload], twist[payload]
    contact = shape.pad_loads(points, speeds, com[payload], box_rot, vel[:3], vel[3:], laws[0])
    table_f, table_n, corner_arms = shape.table_loads(
        com[payload], box_rot, vel[:3], vel[3:], cn, half, laws[1]
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
    return (
        w,
        points,
        speeds,
        contact,
        table_n,
        table_f,
        com[payload] + corner_arms,
        (pad_f_balance, pad_t_balance, power_error),
    )


def run(urdf, inputs, device):
    spec = json.loads(inputs.read_text(encoding="utf8"))
    assert spec["schema"] == "panda-box-control-v1" and "trace" not in spec
    assert hashlib.sha256(urdf.read_bytes()).hexdigest() == spec["urdf_sha256"]
    assert version("newton") == "1.7.0"
    asset = load_panda(urdf)
    shape = SampledPayload.from_manifest(spec)
    model, state, iq, iv, pads, payload_shape, payload, table_id, _ = make_model(urdf, spec, device)
    shape_type = int(model.shape_type.numpy()[payload_shape])
    shape_scale = model.shape_scale.numpy()[payload_shape].astype(float)
    expected_type = newton.GeoType.BOX if shape.radius is None else newton.GeoType.SPHERE
    expected_scale = shape.half if shape.radius is None else [shape.radius, 0.0, 0.0]
    assert shape_type == int(expected_type)
    np.testing.assert_allclose(shape_scale, expected_scale, atol=1e-9, rtol=1e-7)
    np.testing.assert_allclose(
        model.body_inertia.numpy()[payload], np.diag(shape.inertia(spec["mass_kg"])), atol=2e-9
    )
    joints = [named(model.joint_label, n) for n in spec["joint_names"]]
    native_child = model.joint_child.numpy()[joints]
    np.testing.assert_allclose(
        model.body_mass.numpy()[native_child], asset.model.mass, rtol=2e-6, atol=1e-7
    )
    initial_native_pose = state.body_q.numpy()[payload].astype(float)
    initial_native_arm = state.joint_q.numpy()[iq].astype(float)
    next_state, control = model.state(), model.control()
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    assert str(state.body_f.device) == str(state.joint_q.device) == device
    owners, material = finger_material(model, pads)
    com_local = model.body_com.numpy().astype(float)
    # Both uniform payloads have their origin at their COM.
    np.testing.assert_array_equal(com_local[payload], np.zeros(3))
    cn, half = np.array(spec["table_center_m"]), np.array(spec["table_full_size_m"]) / 2
    laws = [RobotContactMaterial(*spec[k]) for k in ("pad_material", "table_material")]
    frame_ids, local_points = _finger_samples(asset)
    passive = _PassiveGeometry(asset)
    table_links = None
    table_law_spec = spec.get("link_table_material")
    if table_law_spec is not None:
        assert (
            model.shape_body.numpy()[table_id] == -1
            and model.shape_type.numpy()[table_id] == newton.GeoType.BOX
        )
        native_table = model.shape_transform.numpy()[table_id].astype(float)
        table_links = passive.table_registry(
            asset,
            center_m=native_table[:3],
            half_extents_m=model.shape_scale.numpy()[table_id],
            orientation_xyzw=native_table[3:],
            material=RobotContactMaterial(*table_law_spec),
            authored_moving_frames=True,
        )
    law_spec = spec.get("structural_material")
    structure_law = None if law_spec is None else RobotContactMaterial(*law_spec)
    structure = (
        None
        if structure_law is None
        else passive.registry(asset, shape, authored_moving_frames=True)
    )
    body_shapes = None if structure is None else passive.registry(asset, shape)
    fixed_ids = [
        i for i, s in enumerate(model.body_label) if s.rsplit("/", 1)[-1] == asset.root_link
    ]
    fixed_id = fixed_ids[0] if len(fixed_ids) == 1 else None
    first_contact = first_table_contact = None
    structural_pose_error = 0.0
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
    rows, stop, first_unresolved = [], None, None
    for step in range(n):
        t = step * dt
        q, v = state.joint_q.numpy()[iq].astype(float), state.joint_qd.numpy()[iv].astype(float)
        body, twist = state.body_q.numpy().astype(float), state.body_qd.numpy().astype(float)
        arm = RobotState(q, v)
        geometry, linear, angular, mass, bias = robot_dynamics_terms(asset.model, arm)
        qcmd, vcmd = _command_at(times, knots, t)
        tau[iv] = _servo_effort(arm, qcmd, vcmd, mass, bias, eff)
        try:
            w, points, speeds, pad, table_n, table_f, table_points, audit = sampled_wrenches(
                body, twist, com_local, owners, material, payload, shape, cn, half, laws
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
        gap, table_gap, object_part, table_part = passive.clearances(
            geometry,
            body[payload, :3],
            quaternion_matrix(body[payload, None, 3:])[0],
            shape.half,
            cn,
            half,
            sphere_radius=shape.radius,
        )
        structural_rows = []
        structural_joint = np.zeros_like(v)
        if structure is not None or table_links is not None:
            Rs = quaternion_matrix(body[:, 3:])
            body_com = body[:, :3] + np.einsum("nij,nj->ni", Rs, com_local)
            positions, rotations = body[native_child, :3], Rs[native_child]
        if structure is not None:
            if step in verify:
                for phy, nat in zip(body_shapes.solids, structure.solids, strict=True):
                    i = nat.owner_index
                    if i >= 0:
                        cooked = geometry.position[i] + phy.vertices @ geometry.rotation[i].T
                        observed = positions[i] + nat.vertices @ rotations[i].T
                        structural_pose_error = max(
                            structural_pose_error, float(np.max(np.abs(cooked - observed)))
                        )
            fixed_reference = asset.root_xyz if fixed_id is None else body_com[fixed_id]
            structural = structure.evaluate(
                positions,
                rotations,
                body_com[native_child],
                twist[native_child],
                body_com[payload],
                shape.half if shape.radius is None else shape.radius,
                twist[payload, :3],
                twist[payload, 3:],
                structure_law,
                fixed_reference_m=fixed_reference,
                **({"payload_rotation": Rs[payload]} if shape.radius is None else {}),
            )
            structural_rows = structural.contacts
            w[native_child] += structural.link_wrenches
            w[payload] += structural.payload_wrench
            if fixed_id is not None:
                for contact in structural_rows:
                    if contact["owner_index"] < 0:
                        w[fixed_id] += np.r_[
                            contact["force_on_link_N"], contact["torque_on_link_Nm"]
                        ]
            # Only body_f drives native structural response. This readback uses the
            # shared point-J identity with the small native-to-reference COM lever.
            structural_joint = structural.joint_load(linear, angular) + np.einsum(
                "nki,nk->i",
                angular,
                np.cross(
                    body_com[native_child] - geometry.com_position, structural.link_wrenches[:, :3]
                ),
            )
            if first_contact is None and structural_rows:
                first_contact = dict(
                    step_index=step, t_before_s=t, shape=structural_rows[0]["shape"]
                )
        table_rows = []
        table_joint = np.zeros_like(v)
        robot_on_table = np.zeros(6)
        if table_links is not None:
            table_load = table_links.evaluate(
                positions, rotations, body_com[native_child], twist[native_child]
            )
            w[native_child] += table_load.link_wrenches
            table_rows = table_load.contacts
            table_joint = table_load.joint_load(linear, angular) + np.einsum(
                "nki,nk->i",
                angular,
                np.cross(
                    body_com[native_child] - geometry.com_position, table_load.link_wrenches[:, :3]
                ),
            )
            robot_on_table = table_load.table_wrench
            if table_rows and first_table_contact is None:
                first_table_contact = dict(
                    step_index=step, t_before_s=t, shape=table_rows[0]["shape"]
                )
        if first_unresolved is None:
            unchecked_table, unchecked_table_part = table_gap, table_part
            if table_links is not None and table_gap <= 0:
                unchecked_table, unchecked_table_part = passive.unhandled_table(geometry, cn, half)
            unchecked = gap if structure is None else np.inf
            if structure is not None and shape.radius is None and gap <= 0:
                unchecked, _, unchecked_part, _ = passive.clearances(
                    geometry, body[payload, :3], Rs[payload], shape.half, cn, half, omit_round=True
                )
            else:
                unchecked_part = object_part
            first_unresolved = passive.unresolved(
                step, t, unchecked, unchecked_table, unchecked_part, unchecked_table_part
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
        clearance = shape.low_z(result[:3], quaternion_matrix(result[None, 3:])[0]) - (
            cn[2] + half[2]
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
                pad_points_before_m=points.tolist(),
                pad_forces_N=pad.forces.tolist(),
                table_points_before_m=table_points.tolist(),
                table_forces_N=table_f.tolist(),
                pad_total_N=float(np.linalg.norm(pad.forces, axis=1).sum()),
                clearance_m=float(clearance),
                applied_payload_wrench=w[payload].tolist(),
                structural_contacts=structural_rows,
                link_table_contacts=table_rows,
                robot_on_table_wrench=robot_on_table.tolist(),
                table_joint_load=table_joint.tolist(),
                structural_joint_load=structural_joint.tolist(),
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
        payload_geometry=shape.descriptor(),
        native_payload_shape=dict(
            id=int(payload_shape), kind=expected_type.name, scale=shape_scale.tolist()
        ),
        cube_mass_kg=spec["mass_kg"],
        closed_width_m=float(np.min(knots[:, -2:].sum(axis=1))),
        finger_friction=spec["pad_material"][2],
        table_center_m=cn.tolist(),
        table_full_size_m=spec["table_full_size_m"],
        stop=stop,
        inputs_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
        urdf_sha256=spec["urdf_sha256"],
        initial_arm_q=initial_native_arm.tolist(),
        first_structural_contact=first_contact,
        first_link_table_contact=first_table_contact,
        max_structural_frame_error_m=structural_pose_error,
        initial_payload_position_m=initial_native_pose[:3].tolist(),
        initial_payload_xyzw=initial_native_pose[3:].tolist(),
        placement_target_m=(initial + [spec["travel_m"], 0.0, 0.0]).tolist(),
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
        first_unresolved_structure=first_unresolved,
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
    p.add_argument("--expect", choices=("carry", "slip", "structural-contact"), required=True)
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
    if args.expect == "structural-contact":
        assert (
            result["first_structural_contact"] is not None
            and result["max_structural_frame_error_m"] < 2e-5
        )
        assert result["first_unresolved_structure"] is None
    else:
        assert result["success"] == (args.expect == "carry")
    if args.expect == "slip":
        assert result["carry_clearance_m"] < 0.005 and abs(result["carry_displacement_m"]) < 0.015
