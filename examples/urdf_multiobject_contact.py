"""A public URDF scene keeps two free bodies live while the operator releases a preload.

This A1 example starts with the selected object already between the fingers.
It measures holding and release; it is not a demonstration of grasp acquisition.
Run the generic jaw fixture with --asset mimic --urdf /path/to/mimic_jaws.urdf.
"""

import argparse
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from itertools import combinations, product
from pathlib import Path

import numpy as np

from phymeo.builders.mimic_urdf import load_mimic_urdf
from phymeo.builders.panda import PANDA_URDF, load_panda
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.capabilities import SolverFailure
from phymeo.core.robot import RobotState
from phymeo.dynamics.rotation import quaternion_matrix
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_payload_scene import PadFace
from phymeo.runtime.robot_scene import RobotCommand

PAD = RobotContactMaterial(1200, 1.2, 0.65, 0.04)
SUPPORT = RobotContactMaterial(2200, 3, 0.45, 0.04)
STRUCTURE = RobotContactMaterial(6000, 16, 0.35, 0.04)


def make_case(kind, urdf=None):
    b = ContactSceneBuilder()
    if kind == "panda":
        asset = load_panda()
        arm = np.pi * np.array([0, -0.25, 0, -0.75, 0, 0.5, 0.25])
        q = asset.configuration(arm, 0.036)
        centers = [asset.authored_link_pose("panda_" + s + "pad", q)[0] for s in ("left", "right")]
        primary = np.mean(centers, axis=0)
        pads = [
            PadFace.named(
                "panda_" + side + "finger",
                "panda_" + side + "finger_pad",
                1,
                sign,
                tuple(product((-0.005, 0.005), repeat=2)),
            )
            for side, sign in (("left", -1), ("right", 1))
        ]
        b.add_urdf(
            10,
            PANDA_URDF,
            loader=load_panda,
            robot_state=RobotState(q, np.zeros(9)),
            kp=[500] * 7 + [4500] * 2,
            kd=[30] * 7 + [18] * 2,
            pads=pads,
        )
        b.add_sphere(20, 0.020, mass=0.04096, position=primary)
        b.add_box(30, (0.015, 0.012, 0.014), mass=0.06, position=primary + [0.06, 0, -0.026])
        top = primary[2] - 0.04
        b.add_box(40, (0.09, 0.06, 0.02), mass=None, position=primary - [0, 0, 0.06])
        goal = asset.configuration(arm, 0.060)
        dt, release_steps, masses, primary_radius, half = 1 / 1600, 256, (0.04096, 0.06), 0.02, None
        pad_names = [
            ("panda_" + s + "finger", "panda_" + s + "finger_pad") for s in ("left", "right")
        ]
    elif kind == "mimic":
        if urdf is None:
            raise ValueError("mimic example needs the explicit local counter-moving jaw URDF")
        primary = np.array([0.25, -0.125, 0.25])
        q = np.array([0.0, 0.014])
        pads = [
            PadFace.named(
                side, side + "_rubber", 0, sign, tuple(product((-0.005, 0.005), repeat=2))
            )
            for side, sign in (("left", 1), ("right", -1))
        ]
        b.add_urdf(
            10,
            urdf,
            loader=load_mimic_urdf,
            robot_state=RobotState(q, np.zeros(2)),
            kp=[2700, 6000],
            kd=[95, 35],
            effort_limit=[40, 8],
            pads=pads,
        )
        b.add_box(20, (0.02, 0.02, 0.02), mass=0.1, position=primary)
        b.add_sphere(30, 0.015, mass=0.03, position=(0.33, -0.125, 0.235))
        top = 0.22
        b.add_box(40, (0.12, 0.12, 0.01), mass=None, position=(0.25, -0.125, 0.21))
        goal = np.zeros(2)
        dt, release_steps, masses, primary_radius, half = (
            0.001,
            160,
            (0.1, 0.03),
            None,
            np.full(3, 0.02),
        )
        pad_names = [(side, side + "_rubber") for side in ("left", "right")]
    else:
        raise ValueError("unknown example asset")
    for a, c in combinations((10, 20, 30, 40), 2):
        law = PAD if (a, c) == (10, 20) else SUPPORT if c == 40 and a != 10 else STRUCTURE
        b.pair(a, c, law)
    scene = b.build()
    return scene, dict(
        kind=kind,
        q=q,
        goal=goal,
        dt=dt,
        hold_steps=32,
        release_steps=release_steps,
        masses=masses,
        initial_position=primary,
        top=top,
        primary_radius=primary_radius,
        primary_half=half,
        pads=[scene.shape_key(10, *name) for name in pad_names],
    )


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _event(scene, label):
    reading, geometry = scene.observe_scene()
    return dict(
        label=label,
        reading=_plain(reading),
        geometry=dict(
            time_s=geometry.time_s,
            units="m",
            shapes=[
                dict(
                    key=_plain(s.key),
                    kind=s.kind,
                    radius_m=s.radius,
                    vertex_ids=s.vertex_ids,
                    triangles=s.triangles,
                    names=geometry.names[s.key],
                    points_m=geometry.poses[s.key].start.tolist(),
                )
                for s in geometry.model.shapes
            ],
        ),
    )


def run_case(scene, info):
    dt, hold, release = info["dt"], info["hold_steps"], info["release_steps"]
    ids = [scene.rigid_rows[0, body] for body in (20, 30)]
    primary_key = scene.body_keys[0, 20][0]
    secondary_key = scene.body_keys[0, 30][0]
    table_key = scene.body_keys[0, 40][0]
    trace, events, bilateral, clearances = [], [_event(scene, "preloaded")], [], []
    max_impulse_error = 0.0
    support_primary = support_secondary = False
    error = None
    for index in range(hold + release):
        before = scene.observe()
        u = np.clip((index - hold) / release, 0, 1)
        blend = u**3 * (10 + u * (-15 + 6 * u))
        slope = 30 * u**2 * (1 - u) ** 2 / (release * dt)
        q = info["q"] + blend * (info["goal"] - info["q"])
        v = slope * (info["goal"] - info["q"])
        try:
            after = scene.step(dt, actions={10: RobotCommand(q, target_velocity=v)})
        except (ValueError, RuntimeError, SolverFailure) as exc:
            error = dict(index=index, accepted_time_s=before.time_s, reason=str(exc))
            break
        normals = np.zeros(2)
        for contact in after.contacts:
            pair = {contact["first"], contact["second"]}
            if contact["channel"] == "robot-pad" and primary_key in pair:
                for k, key in enumerate(info["pads"]):
                    if key in pair:
                        normals[k] += contact["normal_load_N"]
            support_primary |= pair == {primary_key, table_key} and contact["normal_load_N"] > 0
            support_secondary |= pair == {secondary_key, table_key} and contact["normal_load_N"] > 0
        bilateral.append(normals)
        R = quaternion_matrix(after.rigid.orientation[ids[0] : ids[0] + 1])[0]
        extent = info["primary_radius"]
        if extent is None:
            extent = float(np.abs(R[2]) @ info["primary_half"])
        clearance = float(after.rigid.position[ids[0], 2] - extent - info["top"])
        clearances.append(clearance)
        for row, mass in zip(ids, info["masses"], strict=True):
            impulse = mass * (after.rigid.velocity[row] - before.rigid.velocity[row])
            force = after.loads.inputs.rigid.force[row] + mass * np.array([0, 0, -9.81])
            max_impulse_error = max(max_impulse_error, float(np.max(abs(impulse - dt * force))))
        trace.append(
            dict(reading=_plain(after), bilateral_N=normals.tolist(), clearance_m=clearance)
        )
        if index + 1 == hold:
            events.append(_event(scene, "holding"))
    events.append(_event(scene, "last-accepted"))
    last = scene.observe()
    normals = np.asarray(bilateral).reshape(-1, 2)
    initial_clearance = 0.02 if info["kind"] == "panda" else 0.01
    checks = dict(
        complete_window=error is None and last.step_number == hold + release,
        bilateral_hold=len(normals) >= hold and bool(np.all(normals[:hold] > 0.5)),
        held_clearance=len(clearances) >= hold
        and min(clearances[:hold]) > initial_clearance - 0.005,
        actual_release=len(normals) >= 16 and bool(np.all(normals[-16:] == 0)),
        payload_moved_down=float(info["initial_position"][2] - last.rigid.position[ids[0], 2])
        >= 0.005,
        payload_received_by_finite_table=bool(support_primary),
        second_free_body_has_real_support=bool(support_secondary),
        physical_impulse=max_impulse_error < 3e-13,
    )
    return dict(
        schema="robot-multiobject-a1-v1",
        kind=info["kind"],
        units="SI",
        initialization="preloaded between real pads; no acquisition claim",
        dt_s=dt,
        hold_steps=hold,
        release_steps=release,
        checks=checks,
        failure=error,
        max_linear_impulse_error_Ns=max_impulse_error,
        final_position_m=last.rigid.position[ids].tolist(),
        final_velocity_m_s=last.rigid.velocity[ids].tolist(),
        final_table_wrench_N_Nm=np.r_[
            last.loads.inputs.rigid.force[scene.rigid_rows[0, 40]],
            last.loads.inputs.rigid.torque[scene.rigid_rows[0, 40]],
        ].tolist(),
        trace=trace,
        events=events,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", choices=("panda", "mimic"), required=True)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scene, info = make_case(args.asset, args.urdf)
    with scene:
        result = run_case(scene, info)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"trace", "events"}}, indent=2))
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
