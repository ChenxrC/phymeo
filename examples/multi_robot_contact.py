"""Two real URDF sliders maintain direct solid contact, then physically withdraw.

The finite 40-step window is a preload/withdrawal example, not a grasping claim.
Both geometries, inertias, commands and physical substeps are explicit; no free
or dummy rigid body is added and no accepted state is overwritten.
"""

import argparse
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.capabilities import SolverFailure
from phymeo.core.robot import RobotState
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_scene import RobotCommand

RADIUS_M = 1 / 16
HOLD_DT_S = 1 / 8192
WITHDRAW_DT_S = 1 / 256
FORCE_TOLERANCE_N = 1e-10
MECHANICS_TOLERANCE = 1e-12


def _asset(path, center_x):
    source = f'''<robot name="direct_slider"><link name="base"/>
<link name="pusher"><inertial><mass value="2"/>
<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
</inertial><collision name="round_tool"><geometry><sphere radius="{RADIUS_M}"/>
</geometry></collision></link>
<joint name="slide" type="prismatic"><parent link="base"/><child link="pusher"/>
<origin xyz="{center_x} 0 0"/><axis xyz="1 0 0"/>
<limit lower="-0.1" upper="0.1" effort="128" velocity="10"/>
</joint></robot>
'''
    path.write_text(source, encoding="utf-8")
    return path


def make_scene(asset_directory, *, reverse_registration=False):
    """Compile the public pair; each registered robot remains its actual owner."""
    directory = Path(asset_directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        10: _asset(directory / "left.urdf", 0),
        20: _asset(directory / "right.urdf", 0.124),
    }
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    for owner in (20, 10) if reverse_registration else (10, 20):
        builder.add_urdf(
            owner, paths[owner], robot_state=RobotState(np.zeros(1), np.zeros(1)),
            kp=[0], kd=[0], effort_limit=[128],
        )
    builder.pair(10, 20, RobotContactMaterial(2000.0, 0.0, 0.0))
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
                "name": geometry.names[shape.key],
                "kind": shape.kind,
                "vertex_ids": list(shape.vertex_ids),
                "radius_m": shape.radius,
                "world_points_m": geometry.poses[shape.key].start.tolist(),
            }
            for shape in geometry.model.shapes
        ],
    }


def run_window(asset_directory, *, reverse_registration=False):
    trace, geometries = [], []
    maxima = dict(
        impulse_error_ns=0.0,
        drift_error_m=0.0,
        discrete_kinetic_error_j=0.0,
        hold_force_error_n=0.0,
        net_contact_force_n=0.0,
        total_momentum_kg_m_s=0.0,
        virtual_power_error_w=0.0,
    )
    execution_failure, had_end = None, False
    with make_scene(asset_directory, reverse_registration=reverse_registration) as scene:
        initial = _json_value(scene.observe())
        geometries.append(_geometry(scene, "initial"))
        real_owners = len(scene.assets) == 2 and scene.state.rigid.body_count == 0
        for index in range(40):
            holding = index < 32
            dt = HOLD_DT_S if holding else WITHDRAW_DT_S
            actions = {
                10: RobotCommand([0], feedforward=[2 if holding else -8]),
                20: RobotCommand([0], feedforward=[-2 if holding else 8]),
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
            for owner in (10, 20):
                row, prior = out.robots[0, owner], old.robots[0, owner]
                force = row.drive_effort + row.contact_effort
                impulse = 2 * (row.v - prior.v) - dt * force
                drift = row.q - prior.q - dt * row.v
                delta_kinetic = float(row.v @ row.v - prior.v @ prior.v)
                discrete_work = float(dt * (force @ prior.v) + dt * dt * (force @ force) / 4)
                maxima["impulse_error_ns"] = max(maxima["impulse_error_ns"], float(abs(impulse[0])))
                maxima["drift_error_m"] = max(maxima["drift_error_m"], float(abs(drift[0])))
                maxima["discrete_kinetic_error_j"] = max(
                    maxima["discrete_kinetic_error_j"], abs(delta_kinetic - discrete_work),
                )
                if holding:
                    maxima["hold_force_error_n"] = max(
                        maxima["hold_force_error_n"], abs(float(row.contact_effort[0]) - (-2 if owner == 10 else 2)),
                    )
            left, right = out.robots[0, 10], out.robots[0, 20]
            maxima["net_contact_force_n"] = max(
                maxima["net_contact_force_n"], abs(float(left.contact_effort[0] + right.contact_effort[0])),
            )
            maxima["total_momentum_kg_m_s"] = max(
                maxima["total_momentum_kg_m_s"], abs(float(2 * (left.v[0] + right.v[0]))),
            )
            generalized_power = sum(float(row.contact_effort_full @ row.entry_v) for row in out.robots.values())
            site_power = sum(site.power_w for pair in out.loads.contacts for site in pair.sites)
            maxima["virtual_power_error_w"] = max(
                maxima["virtual_power_error_w"], abs(generalized_power - site_power),
            )
            events = scene.contact_events
            had_end |= any(event.kind == "end" for event in events.events)
            trace.append({
                "phase": "hold" if holding else "withdraw",
                "dt_s": dt,
                "contact_count": len(out.loads.contacts),
                "observation": _json_value(out),
                "contact_events": _json_value(events),
            })
            if index == 31:
                geometries.append(_geometry(scene, "held_32_steps"))
        final = scene.observe()
        geometries.append(_geometry(scene, "final"))
        checks = dict(
            two_real_robots_without_dummy=real_owners,
            completed_40_steps=(len(trace) == 40 and execution_failure is None),
            bilateral_32_step_hold=(
                len(trace) >= 32
                and all(row["contact_count"] == 1 for row in trace[:32])
                and maxima["hold_force_error_n"] <= FORCE_TOLERANCE_N
            ),
            impulse=(maxima["impulse_error_ns"] <= MECHANICS_TOLERANCE),
            single_drift=(maxima["drift_error_m"] <= MECHANICS_TOLERANCE),
            discrete_kinetic_update=(maxima["discrete_kinetic_error_j"] <= MECHANICS_TOLERANCE),
            reciprocal_force=(maxima["net_contact_force_n"] <= FORCE_TOLERANCE_N),
            linear_momentum=(maxima["total_momentum_kg_m_s"] <= MECHANICS_TOLERANCE),
            point_virtual_power=(maxima["virtual_power_error_w"] <= MECHANICS_TOLERANCE),
            actual_withdrawal=(
                final.robots[0, 10].q[0] < -0.001 and final.robots[0, 20].q[0] > 0.001
                and final.robots[0, 10].v[0] < 0 and final.robots[0, 20].v[0] > 0
            ),
            last_four_steps_separated=(
                len(trace) == 40 and all(row["contact_count"] == 0 for row in trace[-4:])
                and not scene.contact_events.active and had_end
            ),
        )
    directory = Path(asset_directory)
    return _json_value({
        "capability": "direct solid contact between two real fixed-root URDF trees",
        "backend": "HOST",
        "units": {"length": "m", "time": "s", "mass": "kg", "force": "N", "torque": "N m", "energy": "J"},
        "inputs": {
            "link_masses_kg": [2, 2],
            "com_principal_inertia_kg_m2": [0.01, 0.01, 0.01],
            "sphere_radius_m": RADIUS_M,
            "center_distance_m": 0.124,
            "initial_q_m": [0, 0],
            "initial_v_m_s": [0, 0],
            "gravity_m_s2": [0, 0, 0],
            "stiffness_n_per_m": 2000,
            "damping_n_s_per_m": 0,
            "friction": 0,
            "motor_cap_n": 128,
            "hold_steps": 32,
            "hold_dt_s": HOLD_DT_S,
            "hold_feedforward_n": [2, -2],
            "withdraw_steps": 8,
            "withdraw_dt_s": WITHDRAW_DT_S,
            "withdraw_feedforward_n": [-8, 8],
            "reverse_registration": reverse_registration,
        },
        "tolerances": {
            "force_n": FORCE_TOLERANCE_N,
            "impulse_ns": MECHANICS_TOLERANCE,
            "drift_m": MECHANICS_TOLERANCE,
            "discrete_kinetic_j": MECHANICS_TOLERANCE,
            "momentum_kg_m_s": MECHANICS_TOLERANCE,
            "virtual_power_w": MECHANICS_TOLERANCE,
        },
        "asset_sources": {
            name: (directory / name).read_text(encoding="utf-8") for name in ("left.urdf", "right.urdf")
        },
        "initial": initial,
        "final": final,
        "checks": checks,
        "maxima": maxima,
        "execution_failure": execution_failure,
        "trace": trace,
        "geometry": geometries,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reverse-registration", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {args.output}")
    with TemporaryDirectory(prefix="phymeo-direct-robots-") as directory:
        result = _json_value(run_window(directory, reverse_registration=args.reverse_registration))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "checks": result["checks"], "maxima": result["maxima"],
        "execution_failure": result["execution_failure"], "output": str(args.output),
    }, allow_nan=False))
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
