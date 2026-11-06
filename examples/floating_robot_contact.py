"""One floating URDF root and moving child on a finite table: hold, then lift.

Run from an installed Phymeo environment. Only RobotCommand and explicit COM
loads change between phases. Every state is the result of the previous step.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import phymeo as ph

ASSET = """<robot name="floating_vertical_slide">
<link name="base"><inertial><mass value="2"/>
<inertia ixx=".25" iyy=".5" izz=".75" ixy="0" ixz="0" iyz="0"/></inertial>
<collision name="root_sphere"><geometry><sphere radius=".0625"/></geometry></collision></link>
<link name="tool"><inertial><mass value="1"/>
<inertia ixx=".25" iyy=".25" izz=".25" ixy="0" ixz="0" iyz="0"/></inertial></link>
<joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/>
<axis xyz="0 0 1"/><limit lower="-.1" upper=".1" effort="128" velocity="10"/></joint>
</robot>
"""


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _key(key):
    return [key.world, key.body, key.shape]


def _geometry(snapshot):
    return dict(
        time_s=snapshot.time_s,
        units="m, body-to-world xyzw",
        shapes=[
            dict(
                key=_key(shape.key),
                name=snapshot.names[shape.key],
                kind=shape.kind,
                radius_m=shape.radius,
                vertex_ids=shape.vertex_ids,
                world_points_m=snapshot.poses[shape.key].start,
            )
            for shape in snapshot.model.shapes
        ],
    )


def run_window(*, reverse_order=False):
    """Return physical receipts and independently checked 32+8-step acceptance."""
    initial = ph.FloatingRobotState(
        np.array([0, 0, 0.0625 - 0.0015, 0, 0, 0, 1, 0], dtype=float),
        np.zeros(7),
    )
    builder = ph.ContactSceneBuilder(gravity=(0, 0, -1))
    with TemporaryDirectory(prefix="phymeo-floating-") as directory:
        asset_path = Path(directory) / "vertical.urdf"
        asset_path.write_text(ASSET, encoding="utf-8")
        for owner in (20, 10) if reverse_order else (10, 20):
            if owner == 10:
                builder.add_floating_urdf(
                    owner,
                    asset_path,
                    robot_state=initial,
                    kp=[0],
                    kd=[0],
                    effort_limit=[128],
                )
            else:
                builder.add_box(owner, (0.5, 0.5, 0.03125), mass=None, position=(0, 0, -0.03125))
        builder.pair(10, 20, ph.RobotContactMaterial(2000, 0, 0))
    frames = []
    errors = dict(
        impulse_ns=0.0,
        drift_m=0.0,
        kinetic_j=0.0,
        support_force_n=0.0,
        support_speed_m_s=0.0,
        material_m=0.0,
        table_reaction_n=0.0,
    )
    ended = False
    with builder.build(backend="host") as scene:
        for index in range(40):
            hold = index < 32
            dt, effort = (1 / 8192, 1.0) if hold else (1 / 128, 0.0)
            external = ph.FloatingRobotInputs(
                np.zeros(1),
                np.zeros((2, 3)) if hold else np.array([[0.0, 0.0, 16.0], [0.0, 0.0, 8.0]]),
                np.zeros((2, 3)),
            )
            command = ph.FloatingRobotCommand(ph.RobotCommand([0], feedforward=[effort]), external)
            entry = scene.observe().robots[0, 10]
            result = scene.step(dt, actions={10: command})
            reading, geometry = scene.observe_scene()
            row = reading.robots[0, 10]
            if reading.time_s != result.time_s or reading.step_number != result.step_number:
                raise RuntimeError("observation and geometry do not describe this accepted step")
            np.testing.assert_array_equal(row.entry_q, entry.q)
            np.testing.assert_array_equal(row.entry_v, entry.v)
            normal = float(sum(pair.force_n for pair in reading.loads.contacts))
            old_speed = np.array([entry.v[2], entry.v[2] + entry.v[6]])
            speed = np.array([row.v[2], row.v[2] + row.v[6]])
            forces = np.array(
                [normal + external.force[0, 2] - 2 - effort, external.force[1, 2] - 1 + effort]
            )
            masses = np.array([2, 1])
            errors["impulse_ns"] = max(
                errors["impulse_ns"], float(np.max(abs(masses * (speed - old_speed) - dt * forces)))
            )
            drift = np.r_[
                row.q[:3] - entry.q[:3] - dt * row.v[:3], row.q[7:] - entry.q[7:] - dt * row.v[6:]
            ]
            errors["drift_m"] = max(errors["drift_m"], float(np.max(abs(drift))))
            kinetic = float((masses * (speed**2 - old_speed**2)).sum() / 2)
            discrete_work = float(
                dt * (forces @ old_speed) + dt**2 * (forces**2 / masses).sum() / 2
            )
            errors["kinetic_j"] = max(errors["kinetic_j"], abs(kinetic - discrete_work))
            errors["table_reaction_n"] = max(
                errors["table_reaction_n"],
                float(abs(reading.loads.inputs.rigid.force[0, 2] + normal)),
            )
            if hold:
                errors["support_force_n"] = max(errors["support_force_n"], abs(normal - 3))
                errors["support_speed_m_s"] = max(
                    errors["support_speed_m_s"], float(np.max(abs(row.v)))
                )
            entry_geometry = ph.floating_kinematics(
                scene.assets[0, 10].model, ph.FloatingRobotState(row.entry_q, row.entry_v)
            )
            events = scene.contact_events
            ended |= any(event.kind == "end" for event in events.events)
            material_rows = []
            for event in events.active:
                for component in event.components:
                    for site, material in zip(
                        component.load.sites, component.material, strict=True
                    ):
                        for owner, address, point in (
                            (event.first_owner, material.first, site.material.first_point),
                            (event.second_owner, material.second, site.material.second_point),
                        ):
                            if owner.body == 10:
                                actual = (
                                    entry_geometry.position[owner.link_index]
                                    + entry_geometry.rotation[owner.link_index]
                                    @ address.local_point_m
                                )
                                errors["material_m"] = max(
                                    errors["material_m"], float(np.max(abs(actual - point)))
                                )
                                material_rows.append(
                                    dict(
                                        owner=owner.body,
                                        link=owner.link_name,
                                        frame=owner.local_frame,
                                        local_point_m=address.local_point_m,
                                        surface_m=point,
                                    )
                                )
            frames.append(
                dict(
                    step=reading.step_number,
                    phase="support" if hold else "lift",
                    dt_s=dt,
                    load_time_s=reading.load_time_s,
                    time_s=reading.time_s,
                    entry_q=row.entry_q,
                    entry_v=row.entry_v,
                    q=row.q,
                    v=row.v,
                    command=dict(target_q=[0], feedforward_n=[effort]),
                    external_force_n=row.external.force,
                    external_torque_nm=row.external.torque,
                    contact_effort_full=row.contact_effort_full,
                    applied_effort_full=row.applied_effort_full,
                    link_com_wrenches_world=row.link_wrenches_world,
                    acceleration_full=row.acceleration_full,
                    normal_force_n=normal,
                    table_reaction_n=reading.loads.inputs.rigid.force[0],
                    table_position_m=reading.rigid.position[0],
                    material=material_rows,
                    event_kinds=[event.kind for event in events.events],
                    geometry=_geometry(geometry),
                )
            )
        checks = dict(
            continuous_40_steps=len(frames) == 40 and reading.step_number == 40,
            weight_supported_32_steps=errors["support_force_n"] <= 1e-10
            and errors["support_speed_m_s"] <= 1e-12,
            true_external_lift=row.q[2] > 0.0635 and row.v[2] > 0,
            joint_remains_independent=row.q[7] != 0 and row.v[6] != 0,
            unloaded_last_four=all(frame["normal_force_n"] == 0 for frame in frames[-4:]) and ended,
            impulse_balance=errors["impulse_ns"] <= 1e-12,
            single_drift=errors["drift_m"] <= 1e-12,
            discrete_kinetic_balance=errors["kinetic_j"] <= 1e-12,
            reciprocal_support=errors["table_reaction_n"] <= 1e-10
            and not np.any(reading.rigid.velocity),
            true_material_frames=errors["material_m"] <= 1e-12,
            common_clock=reading.time_s == 32 / 8192 + 8 / 128
            and geometry.time_s == reading.time_s,
        )
        return _json_value(
            dict(
                case="floating-root-finite-table",
                backend="HOST FP64",
                reverse_order=reverse_order,
                asset_source_xml=ASSET,
                initial_q=initial.q,
                initial_v=initial.v,
                units=dict(
                    q="m and body-to-world XYZW and m",
                    v="world COM m/s, world rad/s, joint m/s",
                    force="N",
                    torque="N m",
                    inertia="kg m^2",
                ),
                limits="instantaneous compliant contact; no CCD, floating mimic, folded fixed joints or pads",
                checks=checks,
                errors=errors,
                accepted_steps=len(frames),
                frames=frames,
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reverse-order", action="store_true")
    args = parser.parse_args()
    result = run_window(reverse_order=args.reverse_order)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "frames"}, allow_nan=False
        )
    )
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
