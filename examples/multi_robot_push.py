"""Two local URDF mechanisms share a sphere, hold a preload, then withdraw.

This is a finite 40-step physical control example, not a pick-and-place claim.
All materials and inertial data are declared below; no state is overwritten.
"""

import argparse
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from phymeo.builders.mimic_urdf import load_mimic_urdf
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.capabilities import SolverFailure
from phymeo.core.robot import RobotState
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_scene import RobotCommand

RADIUS_M = 1 / 16
HOLD_DT_S = 1 / 8192
WITHDRAW_DT_S = 1 / 256


def _asset(path, *, mimic):
    links = (("driver", 1), ("follower", 1), ("counter", 1)) if mimic else (("pusher", 2),)
    tool = "follower" if mimic else "pusher"
    parts = ['<robot name="shared_sphere_fixture"><link name="base"/>']
    for name, mass in links:
        parts.append(
            f'<link name="{name}"><inertial><mass value="{mass}"/>'
            '<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>'
            "</inertial>"
        )
        if name == tool:
            parts.append(
                f'<collision name="round_tool"><geometry><sphere radius="{RADIUS_M}"/>'
                "</geometry></collision>"
            )
        parts.append("</link>")
    joints = (
        (
            ("drive", "driver", "2 0 0.25", None),
            ("contact_slide", "follower", "0.121 0 0.25", (1, 0.003)),
            ("counter_slide", "counter", "3 0 0.25", (-1, -0.002)),
        )
        if mimic
        else (("slide", "pusher", "-0.124 0 0.25", None),)
    )
    for name, child, position, relation in joints:
        parts.append(
            f'<joint name="{name}" type="prismatic"><parent link="base"/>'
            f'<child link="{child}"/><origin xyz="{position}"/><axis xyz="1 0 0"/>'
            '<limit lower="-0.1" upper="0.1" effort="128" velocity="10"/>'
        )
        if relation is not None:
            parts.append(
                f'<mimic joint="drive" multiplier="{relation[0]}" offset="{relation[1]}"/>'
            )
        parts.append("</joint>")
    parts.append("</robot>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path


def make_scene(asset_directory, *, reverse_registration=False):
    """Use the public builder; mechanical ownership remains two separate robots."""
    directory = Path(asset_directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        10: _asset(directory / "ordinary.urdf", mimic=False),
        20: _asset(directory / "transmission.urdf", mimic=True),
    }
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    for robot in (20, 10) if reverse_registration else (10, 20):
        options = {"loader": load_mimic_urdf} if robot == 20 else {}
        builder.add_urdf(
            robot,
            paths[robot],
            robot_state=RobotState(np.zeros(1), np.zeros(1)),
            kp=[0],
            kd=[0],
            **options,
        )
    builder.add_sphere(30, RADIUS_M, mass=1, position=(0, 0, 0.25))
    material = RobotContactMaterial(2000.0, 0.0, 0.0)
    builder.pair(10, 20, None)
    builder.pair(10, 30, material)
    builder.pair(20, 30, material)
    return builder.build(backend="host")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _geometry(scene, label):
    reading, geometry = scene.observe_scene()
    return {
        "label": label,
        "step_number": reading.step_number,
        "time_s": geometry.time_s,
        "shapes": [
            {
                "key": _json_value(shape.key),
                "kind": shape.kind,
                "vertex_ids": list(shape.vertex_ids),
                "radius_m": shape.radius,
                "world_points_m": geometry.poses[shape.key].start.tolist(),
                "name": geometry.names[shape.key],
            }
            for shape in geometry.model.shapes
        ],
    }


def run_window(asset_directory, *, reverse_registration=False):
    trace, geometries = [], []
    maxima = dict(
        ordinary_impulse_ns=0.0,
        full_axis_impulse_ns=0.0,
        payload_impulse_ns=0.0,
        constraint_power_w=0.0,
        discrete_kinetic_error_j=0.0,
        hold_force_error_n=0.0,
    )
    execution_failure = None
    with make_scene(asset_directory, reverse_registration=reverse_registration) as scene:
        initial = _json_value(scene.observe())
        geometries.append(_geometry(scene, "initial"))
        row = scene.rigid_rows[0, 30]
        for index in range(40):
            holding = index < 32
            dt = HOLD_DT_S if holding else WITHDRAW_DT_S
            actions = {
                10: RobotCommand([0], feedforward=[2 if holding else -128]),
                20: RobotCommand([0], feedforward=[-2 if holding else 128]),
            }
            old = scene.observe()
            try:
                out = scene.step(dt, actions=actions)
            except (ValueError, RuntimeError, SolverFailure) as error:
                execution_failure = {
                    "attempt": index + 1,
                    "time_s": old.time_s,
                    "type": type(error).__name__,
                    "reason": str(error),
                }
                break
            left, right = out.robots[0, 10], out.robots[0, 20]
            forces = right.mimic_loads
            full_force = (
                forces.actuator_effort_full
                + forces.contact_effort_full
                + forces.constraint_effort_full
            )
            v0 = old.robots[0, 20].full_v
            kinetic_error = abs(
                0.5 * (right.full_v @ right.full_v - v0 @ v0)
                - dt * (full_force @ v0)
                - 0.5 * dt * dt * (full_force @ full_force)
            )
            errors = dict(
                ordinary_impulse_ns=float(
                    np.max(
                        np.abs(
                            2 * (left.v - old.robots[0, 10].v)
                            - dt * (left.drive_effort + left.contact_effort)
                        )
                    )
                ),
                full_axis_impulse_ns=float(np.max(np.abs(right.full_v - v0 - dt * full_force))),
                payload_impulse_ns=float(
                    np.max(
                        np.abs(
                            out.rigid.velocity[row]
                            - old.rigid.velocity[row]
                            - dt * out.loads.inputs.rigid.force[row]
                        )
                    )
                ),
                constraint_power_w=abs(forces.constraint_power_W),
                discrete_kinetic_error_j=float(kinetic_error),
                hold_force_error_n=(
                    float(max(abs(left.contact_effort[0] + 2), abs(right.contact_effort[0] - 2)))
                    if holding
                    else 0.0
                ),
            )
            for name, value in errors.items():
                maxima[name] = max(maxima[name], value)
            trace.append(
                {
                    "phase": "hold" if holding else "withdraw",
                    "dt_s": dt,
                    "frame": _json_value(out),
                    "contact_count": len(out.loads.contacts),
                    "errors": errors,
                }
            )
            if index in (31, 39):
                geometries.append(_geometry(scene, "hold_end" if holding else "withdraw_end"))
        final = scene.observe()
        checks = dict(
            completed_40_steps=len(trace) == 40 and execution_failure is None,
            real_32_step_bilateral_hold=(
                len(trace) >= 32
                and all(row["contact_count"] == 2 for row in trace[:32])
                and maxima["hold_force_error_n"] < 2e-10
            ),
            ordinary_impulse=maxima["ordinary_impulse_ns"] < 2e-12,
            full_axis_impulse=maxima["full_axis_impulse_ns"] < 2e-12,
            payload_impulse=maxima["payload_impulse_ns"] < 2e-12,
            passive_constraint_power=maxima["constraint_power_w"] < 2e-11,
            discrete_kinetic_update=maxima["discrete_kinetic_error_j"] < 2e-11,
            actual_withdrawal=(
                final.robots[0, 10].q[0] < -0.001 and final.robots[0, 20].q[0] > 0.001
            ),
            last_four_steps_separated=(
                len(trace) == 40 and all(row["contact_count"] == 0 for row in trace[-4:])
            ),
        )
    directory = Path(asset_directory)
    return {
        "capability": "two fixed-root URDFs share a dynamically integrated free sphere",
        "backend": "HOST",
        "units": {"length": "m", "time": "s", "force": "N", "energy": "J"},
        "inputs": {
            "ordinary_mass_kg": 2,
            "mimic_link_masses_kg": [1, 1, 1],
            "mimic_projection": [1, 1, -1],
            "mimic_offsets_m": [0, 0.003, -0.002],
            "payload_mass_kg": 1,
            "sphere_radius_m": RADIUS_M,
            "gravity_m_s2": [0, 0, 0],
            "stiffness_n_m": 2000,
            "damping_ns_m": 0,
            "friction": 0,
            "hold_steps": 32,
            "hold_dt_s": HOLD_DT_S,
            "withdraw_steps": 8,
            "withdraw_dt_s": WITHDRAW_DT_S,
            "motor_cap_n": 128,
            "robot_pair": "explicitly excluded",
            "reverse_registration": reverse_registration,
        },
        "asset_sources": {
            name: (directory / name).read_text(encoding="utf-8")
            for name in ("ordinary.urdf", "transmission.urdf")
        },
        "initial": initial,
        "checks": checks,
        "maxima": maxima,
        "execution_failure": execution_failure,
        "trace": trace,
        "geometry": geometries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reverse-registration", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {args.output}")
    with TemporaryDirectory(prefix="phymeo-multi-robot-") as directory:
        result = run_window(directory, reverse_registration=args.reverse_registration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "checks": result["checks"],
                "maxima": result["maxima"],
                "execution_failure": result["execution_failure"],
                "output": str(args.output),
            }
        )
    )
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
