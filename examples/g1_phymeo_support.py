"""Reproduce free-root G1 standing or the official 29-joint dance policy on HOST.

Examples (assets are separately downloaded research inputs, never bundled)::

    python examples/g1_phymeo_support.py --assets PATH --mode stand \
        --duration 2 --dt .002 --output stand.jsonl
    python examples/g1_phymeo_support.py --assets PATH --mode policy \
        --duration 20 --dt .002 --output policy.jsonl

The default XML is the physical primitive model with armature .01, viscous
damping .05, zero dry friction, and the official motor limits. No masses, shapes,
gains, initial reference velocities, or published states are fitted or repaired.
Standing places the actual lowest foot capsule on z=0 once before integration.
Policy mode uses the motion's untouched root origin/q/v and official 154-vector;
its explicit clipped torque is sent through the public bounded command.

Every attempt is fsynced before step(), then its accepted state and load-time
foot forces are fsynced afterward. A rejection preserves both the entry state
and command. JSONL and its .summary.json sibling must not already exist.
Engineering criteria are declared below; a short trace is inconclusive, not a
standing/dance success. Floor contact is discrete compliance with velocity
friction; self-contact, CCD and persistent material friction are not claimed.
This is a HOST diagnostic, not a training benchmark.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import yaml

import phymeo as ph
from phymeo.builders.floating_mjcf import load_floating_mjcf
from phymeo.dynamics.rotation import quaternion_matrix
from phymeo.experiments.humanoid_support import read_foot_support

ROBOT, FLOOR = 1, 2
GRAVITY = 9.81
EXPECTED_OBSERVATIONS = (
    ("motion_command", 58),
    ("motion_anchor_ori_b", 6),
    ("base_ang_vel", 3),
    ("joint_pos_rel", 29),
    ("joint_vel_rel", 29),
    ("last_action", 29),
)
# Declared engineering gates, not universal bounds or reference-engine equality.
CRITERIA = {
    "minimum_assessed_duration_s": 1.0,
    "settling_time_s": 0.25,
    "minimum_root_height_m": 0.35,
    "maximum_fall_tilt_rad": 1.0,
    "stand_maximum_tilt_rad": 0.35,
    "stand_maximum_root_xy_drift_m": 0.10,
    "stand_maximum_foot_center_xy_drift_m": 0.08,
    "stand_maximum_joint_rmse_rad": 0.35,
    "policy_maximum_reference_xy_error_m": 1.0,
    "policy_maximum_joint_rmse_rad": 0.50,
    "stand_mean_vertical_load_ratio": (0.70, 1.30),
    "policy_mean_vertical_load_ratio": (0.50, 1.50),
    "maximum_foot_force_bodyweight_ratio": 8.0,
    "stand_minimum_both_feet_loaded_fraction": 0.90,
    "positive_foot_load_threshold_n": 1.0,
    "reaction_balance_relative_tolerance": 1e-8,
    "effort_cap_absolute_tolerance_nm": 1e-9,
    "stand_maximum_nonfoot_ground_force_n": 1.0,
}


def json_value(value):
    """Preserve diagnostic nonfinite words without emitting invalid JSON."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    return value


class Journal:
    def __init__(self, path):
        self.path = path.resolve()
        self.summary_path = self.path.with_suffix(".summary.json")
        if self.path == self.summary_path:
            raise ValueError("--output must differ from its .summary.json sibling")
        if self.path.exists() or self.summary_path.exists():
            raise FileExistsError("refusing to replace an existing trace or summary")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="utf-8")
        self.context = {}

    def append(self, event, **payload):
        self.stream.write(
            json.dumps(json_value({"event": event, **payload}), allow_nan=False) + "\n"
        )
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def finish(self, summary):
        self.append("summary", **summary)
        with self.summary_path.open("x", encoding="utf-8") as stream:
            json.dump(json_value(summary), stream, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())

    def close(self):
        self.stream.close()


def source_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def vector(value, length, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return array


def rotation(wxyz):
    quat = vector(wxyz, 4, "wxyz quaternion")[[1, 2, 3, 0]]
    norm = np.linalg.norm(quat)
    if norm == 0:
        raise ValueError("zero reference quaternion")
    return quaternion_matrix((quat / norm)[None])[0]


def axis_rotation(axis, angle):
    quat = np.zeros(4)
    quat[0] = np.cos(angle / 2)
    quat[axis + 1] = np.sin(angle / 2)
    return rotation(quat)


def torso(root_wxyz, joints):
    return (
        rotation(root_wxyz)
        @ axis_rotation(2, joints[12])
        @ axis_rotation(0, joints[13])
        @ axis_rotation(1, joints[14])
    )


def yaw(matrix):
    return axis_rotation(2, np.arctan2(matrix[1, 0], matrix[0, 0]))


def protocol(config, asset, scene_path):
    if config.get("joint_ids_map") != list(range(29)):
        raise ValueError("this official G1 protocol requires joint_ids_map=0..28")
    entries = ET.parse(scene_path).getroot().find("actuator")
    if entries is None or any(entry.tag != "motor" for entry in entries):
        raise ValueError("scene must declare the 29 official direct motors")
    names = tuple(entry.get("joint") for entry in entries)
    if len(names) != 29 or len(set(names)) != 29 or set(names) != set(asset.joint_names):
        raise ValueError("motor names must cover every physical joint exactly once")
    expected = tuple(
        side + "_" + part + "_joint"
        for side in ("left", "right")
        for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ) + ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint") + tuple(
        side + "_" + part + "_joint"
        for side in ("left", "right")
        for part in (
            "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
            "wrist_roll", "wrist_pitch", "wrist_yaw",
        )
    )
    if names != expected:
        raise ValueError("motor order differs from the official 29-joint policy protocol")
    physical_from_policy = np.array([names.index(name) for name in asset.joint_names])
    policy_from_physical = np.array([asset.joint_names.index(name) for name in names])
    observations = config["observations"]
    if tuple(observations) != tuple(name for name, _ in EXPECTED_OBSERVATIONS):
        raise ValueError("official observation insertion order has changed")
    for name, size in EXPECTED_OBSERVATIONS:
        entry = observations[name]
        if (
            entry.get("clip") is not None
            or entry.get("history_length") != 1
            or not np.array_equal(vector(entry["scale"], size, name), np.ones(size))
        ):
            raise ValueError(f"unsupported observation normalization for {name}")
    spec = config["actions"]["JointPositionAction"]
    if spec.get("clip") is not None or spec.get("joint_ids") is not None:
        raise ValueError("official policy actions must use all joints without clipping")
    if spec.get("joint_names") != [".*"]:
        raise ValueError("unsupported policy joint selection")
    default = vector(config["default_joint_pos"], 29, "default_joint_pos")
    kp = vector(config["stiffness"], 29, "stiffness")
    kd = vector(config["damping"], 29, "damping")
    scale = vector(spec["scale"], 29, "action scale")
    offset = vector(spec["offset"], 29, "action offset")
    if np.any(kp < 0) or np.any(kd < 0):
        raise ValueError("official drive gains must be nonnegative")
    return names, physical_from_policy, policy_from_physical, default, kp, kd, scale, offset


def foot_colliders(asset):
    feet = {}
    for side in ("left", "right"):
        selected = tuple(
            collider
            for collider in asset.collisions
            if collider.link_name == side + "_ankle_roll_link"
            and collider.name.startswith(side + "_foot")
        )
        if len(selected) != 7 or any(row.shape.kind != "capsule" for row in selected):
            raise ValueError("expected seven actual physical G1 foot capsules per side")
        feet[side] = selected
    return feet


def foot_geometry(asset, q, colliders):
    motion = asset.collision_motion(q)
    result = {}
    for name, rows in colliders.items():
        points = np.vstack([motion[row.shape.key].start for row in rows])
        sole = min(
            float(np.min(motion[row.shape.key].start[:, 2]) - row.shape.radius)
            for row in rows
        )
        result[name] = {"capsule_endpoint_center_m": points.mean(axis=0), "sole_height_m": sole}
    return result


def initial_state(mode, asset, default, physical_from_policy, colliders, motion):
    if mode == "policy":
        wxyz = vector(motion["body_quat_w"][0, 0], 4, "initial root quaternion")
        quat = wxyz[[1, 2, 3, 0]] / np.linalg.norm(wxyz)
        q = np.r_[
            motion["body_pos_w"][0, 0], quat, motion["joint_pos"][0, physical_from_policy]
        ].astype(float)
        omega = vector(motion["body_ang_vel_w"][0, 0], 3, "initial world angular velocity")
        # Public v[:3] is root COM velocity, while motion data uses body origin.
        center_offset = rotation(wxyz) @ asset.model.root_com_m
        v = np.r_[
            motion["body_lin_vel_w"][0, 0] + np.cross(omega, center_offset),
            omega,
            motion["joint_vel"][0, physical_from_policy],
        ].astype(float)
        placement = {"method": "original_motion_q_v", "root_z_shift_m": 0.0}
    else:
        q = np.r_[np.zeros(3), [0.0, 0.0, 0.0, 1.0], default[physical_from_policy]]
        v = np.zeros(35)
        raw = foot_geometry(asset, q, colliders)
        shift = -min(row["sole_height_m"] for row in raw.values())
        q[2] += shift
        placement = {
            "method": "once_only_lowest_real_foot_capsule_to_z_zero",
            "unplaced_foot_geometry": raw,
            "root_z_shift_m": shift,
        }
    return ph.FloatingRobotState(q, v), placement


def foot_contacts(reading, feet, ground):
    selected = {key for rows in feet.values() for key in rows}
    ground = set(ground)
    return tuple(
        pair
        for pair in reading.loads.contacts
        if (pair.first in selected and pair.second in ground)
        or (pair.second in selected and pair.first in ground)
    )


def state_record(reading):
    return {
        "q": reading.q,
        "v": reading.v,
        "entry_q": reading.entry_q,
        "entry_v": reading.entry_v,
        "drive_effort_nm": reading.drive_effort,
        "contact_effort_full": reading.contact_effort_full,
        "applied_effort_full": reading.applied_effort_full,
        "acceleration_full": reading.acceleration_full,
        "link_contact_wrenches_world": reading.link_wrenches_world,
    }


def evaluate(mode, stats, completed, failure):
    duration = stats["accepted_duration_s"]
    assessed = duration >= CRITERIA["minimum_assessed_duration_s"]
    count = stats["settled_samples"]
    mean_load = stats["settled_vertical_load_sum"] / count if count else None
    both = stats["both_loaded_samples"] / count if count else None
    ratios = CRITERIA[mode + "_mean_vertical_load_ratio"]
    gates = {
        "requested_duration_completed": completed,
        "no_failure_or_fall": failure is None,
        "long_enough_to_assess": assessed,
        "actual_effort_caps": stats["effort_caps_observed"],
        "paired_foot_reaction_balance": stats["reaction_balance_observed"],
        "bounded_peak_foot_force": (
            stats["maximum_force_bodyweight_ratio"]
            <= CRITERIA["maximum_foot_force_bodyweight_ratio"]
        ),
        "mean_vertical_support_load": (
            mean_load is not None and ratios[0] <= mean_load <= ratios[1]
        ),
        "joint_tracking": (
            stats["maximum_joint_rmse_rad"]
            <= CRITERIA[mode + "_maximum_joint_rmse_rad"]
        ),
    }
    if mode == "stand":
        gates.update(
            upright=stats["maximum_root_tilt_rad"] <= CRITERIA["stand_maximum_tilt_rad"],
            bounded_root_drift=(
                stats["maximum_root_xy_drift_m"] <= CRITERIA["stand_maximum_root_xy_drift_m"]
            ),
            bounded_foot_center_drift=(
                stats["maximum_foot_center_xy_drift_m"]
                <= CRITERIA["stand_maximum_foot_center_xy_drift_m"]
            ),
            both_feet_carry_load=(
                both is not None
                and both >= CRITERIA["stand_minimum_both_feet_loaded_fraction"]
            ),
            no_nonfoot_ground_support=(
                stats["maximum_nonfoot_ground_force_n"]
                <= CRITERIA["stand_maximum_nonfoot_ground_force_n"]
            ),
        )
    else:
        gates["bounded_reference_drift"] = (
            stats["maximum_reference_xy_error_m"]
            <= CRITERIA["policy_maximum_reference_xy_error_m"]
        )
    return {
        "success": all(gates.values()),
        "gates": gates,
        "settled_mean_vertical_load_bodyweights": mean_load,
        "settled_both_feet_loaded_fraction": both,
        "assessment": "engineering_window" if assessed else "inconclusive_short_trace",
    }


def run(args, journal):
    assets = args.assets.resolve()
    scene_path = Path(args.scene)
    if not scene_path.is_absolute():
        scene_path = assets / scene_path
    config_path = assets / "deploy.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    asset = load_floating_mjcf(scene_path)
    names, p_from_o, o_from_p, default, kp, kd, scale, offset = protocol(
        config, asset, scene_path
    )
    policy_dt = float(config["step_dt"])
    if not math.isclose(policy_dt, 0.02, rel_tol=0, abs_tol=1e-12):
        raise ValueError("official G1 policy must run at 50 Hz")
    decimation = round(policy_dt / args.dt)
    steps = round(args.duration / args.dt)
    if (
        decimation < 1
        or steps < 1
        or not math.isclose(decimation * args.dt, policy_dt, rel_tol=0, abs_tol=1e-12)
        or not math.isclose(steps * args.dt, args.duration, rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError("duration must contain whole physics steps; dt must divide .02 s")
    cap = vector(asset.effort_limit, 29, "physical effort limits")
    if np.any(cap <= 0):
        raise ValueError("all official motor caps must be positive")
    if (
        np.any(default[p_from_o] < np.asarray(asset.lower))
        or np.any(default[p_from_o] > np.asarray(asset.upper))
    ):
        raise ValueError("official default target violates the physical joint limits")
    colliders = foot_colliders(asset)
    records = {"scene": source_record(scene_path), "deploy": source_record(config_path)}
    motion = session = None
    if args.mode == "policy":
        import onnxruntime as ort

        motion_path, policy_path = assets / "dance1_subject2.npz", assets / "policy.onnx"
        records.update(motion=source_record(motion_path), policy=source_record(policy_path))
        if (assets / "policy.onnx.data").exists():
            records["policy_external_weights"] = source_record(assets / "policy.onnx.data")
        with np.load(motion_path, allow_pickle=False) as data:
            motion = {name: data[name].copy() for name in data.files}
        for name in ("joint_pos", "joint_vel"):
            if motion[name].ndim != 2 or motion[name].shape[1] != 29:
                raise ValueError("motion requires every original 29-joint reference vector")
        if len(motion["joint_pos"]) < math.ceil(args.duration / policy_dt):
            raise ValueError("requested duration exceeds the unchanged reference motion")
        session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        if (
            session.get_inputs()[0].name != "obs"
            or session.get_outputs()[0].name != "actions"
        ):
            raise ValueError("official ONNX obs/actions names have changed")
    state, placement = initial_state(
        args.mode, asset, default, p_from_o, colliders, motion
    )
    initial_feet = foot_geometry(asset, state.q, colliders)
    weight = GRAVITY * (
        1 / float(asset.model.root.inverse_mass[0]) + float(asset.model.children.mass.sum())
    )
    initial_root = state.q[:3].copy()
    alignment = None
    if args.mode == "policy":
        alignment = yaw(torso(state.q[[6, 3, 4, 5]], state.q[7:][o_from_p])) @ yaw(
            rotation(motion["body_quat_w"][0, 0])
        ).T
    journal.append(
        "initialization",
        sources=records,
        physical_joint_names=asset.joint_names,
        policy_joint_names=names,
        physical_from_policy=p_from_o,
        kp_policy_order=kp,
        kd_policy_order=kd,
        default_joint_pos_policy_order=default,
        effort_caps_physical_order_nm=cap,
        joint_armature=asset.model.joint_armature,
        joint_damping=asset.model.joint_damping,
        initial_q=state.q,
        initial_v=state.v,
        placement=placement,
        initial_foot_geometry=initial_feet,
        bodyweight_n=weight,
        policy_alignment=alignment,
        control_hz=50,
        physics_dt_s=args.dt,
        normal_contact={"stiffness_n_m": 50000, "damping_ns_m": 100, "friction": 1},
        scene_method="free_root_HOST_compliant_discrete_contact_no_material_registry",
    )
    builder = ph.ContactSceneBuilder(gravity=(0, 0, -GRAVITY))
    builder.add_floating_urdf(
        ROBOT,
        scene_path,
        loader=load_floating_mjcf,
        robot_state=state,
        kp=kp[p_from_o] if args.mode == "stand" else np.zeros(29),
        kd=kd[p_from_o] if args.mode == "stand" else np.zeros(29),
        effort_limit=cap,
    )
    builder.add_box(FLOOR, (10, 10, 0.1), mass=None, position=(0, 0, -0.1))
    builder.pair(
        ROBOT, FLOOR, ph.RobotContactMaterial(stiffness=50000, damping=100, friction=1)
    )
    stats = {
        "accepted_steps": 0,
        "accepted_duration_s": 0.0,
        "minimum_root_height_m": float(state.q[2]),
        "maximum_root_tilt_rad": 0.0,
        "maximum_root_xy_drift_m": 0.0,
        "maximum_foot_center_xy_drift_m": 0.0,
        "maximum_reference_xy_error_m": 0.0,
        "maximum_joint_rmse_rad": 0.0,
        "maximum_force_bodyweight_ratio": 0.0,
        "maximum_nonfoot_ground_force_n": 0.0,
        "settled_samples": 0,
        "settled_vertical_load_sum": 0.0,
        "both_loaded_samples": 0,
        "effort_caps_observed": True,
        "reaction_balance_observed": True,
    }
    journal.context["statistics"] = stats
    action, raw_target, policy_frame = np.zeros(29), default.copy(), None
    observation = None
    with builder.build(backend="host") as scene:
        feet = {
            name: tuple(scene.shape_key(ROBOT, row.link_name, row.name) for row in rows)
            for name, rows in colliders.items()
        }
        ground = scene.body_keys[0, FLOOR]
        journal.append("registered_shapes", feet=feet, ground=ground)
        for tick in range(steps):
            entry = scene.observe().robots[0, ROBOT]
            journal.context["last_entry"] = {
                "tick": tick, "entry_time_s": tick * args.dt, "q": entry.q, "v": entry.v
            }
            if args.mode == "policy" and tick % decimation == 0:
                # Exact 50 Hz tick index equals round(time * 50) in the reference.
                policy_frame = tick // decimation
                reference_q = motion["joint_pos"][policy_frame]
                relative = (
                    torso(entry.q[[6, 3, 4, 5]], entry.q[7:][o_from_p]).T
                    @ alignment
                    @ torso(motion["body_quat_w"][policy_frame, 0], reference_q)
                )
                root_rotation = rotation(entry.q[[6, 3, 4, 5]])
                observation = np.concatenate(
                    (
                        reference_q,
                        motion["joint_vel"][policy_frame],
                        relative[:, :2].reshape(-1),
                        root_rotation.T @ entry.v[3:6],
                        entry.q[7:][o_from_p] - default,
                        entry.v[6:][o_from_p],
                        action,
                    )
                ).astype(np.float32)[None]
                if observation.shape != (1, 154) or not np.isfinite(observation).all():
                    raise ValueError("invalid complete official policy observation")
                journal.append(
                    "policy_request",
                    tick=tick,
                    policy_frame=policy_frame,
                    observation=observation,
                    previous_action=action,
                )
                action = session.run(["actions"], {"obs": observation})[0][0]
                vector(action, 29, "raw policy action")
                raw_target = offset + scale * action
            target = raw_target[p_from_o].copy()
            if args.mode == "policy":
                target = np.clip(target, asset.lower, asset.upper)
                torque = (
                    kp * (raw_target - entry.q[7:][o_from_p])
                    - kd * entry.v[6:][o_from_p]
                )[p_from_o]
                feedforward = np.clip(torque, -cap, cap)
            else:
                torque, feedforward = None, np.zeros(29)
            # In policy mode zero registered gains make the bounded packet target
            # inactive. Clipping only this packet does not alter raw PD or q/v.
            packet = ph.FloatingRobotCommand(
                ph.RobotCommand(target, feedforward=feedforward)
            )
            attempted = {
                "tick": tick,
                "entry_time_s": tick * args.dt,
                "dt_s": args.dt,
                "entry_q": entry.q,
                "entry_v": entry.v,
                "policy_frame": policy_frame,
                "policy_observation": observation,
                "raw_policy_action": action,
                "raw_target_policy_order": raw_target,
                "raw_pd_torque_physical_order_nm": torque,
                "explicit_torque_physical_order_nm": feedforward,
                "effort_caps_physical_order_nm": cap,
                "public_command": packet,
            }
            journal.context["last_attempt"] = attempted
            journal.append("attempt", **attempted)
            scene.step(args.dt, actions={ROBOT: packet})
            accepted = scene.observe()
            robot = accepted.robots[0, ROBOT]
            support = read_foot_support(
                accepted,
                feet,
                ground,
                floor_rigid_rows=(scene.rigid_rows[0, FLOOR],),
                robot_address=(0, ROBOT),
                reference_root_position_m=initial_root,
                positive_load_threshold_n=CRITERIA["positive_foot_load_threshold_n"],
            )
            geometry = foot_geometry(asset, robot.q, colliders)
            root_height, tilt = float(robot.q[2]), float(support.root_tilt_rad)
            root_drift = float(np.linalg.norm(robot.q[:2] - initial_root[:2]))
            foot_drift = max(
                float(
                    np.linalg.norm(
                        geometry[name]["capsule_endpoint_center_m"][:2]
                        - initial_feet[name]["capsule_endpoint_center_m"][:2]
                    )
                )
                for name in feet
            )
            expected_q = default
            reference_xy_error = 0.0
            if args.mode == "policy":
                reference_tick = min(
                    int(round(accepted.time_s / policy_dt)), len(motion["joint_pos"]) - 1
                )
                expected_q = motion["joint_pos"][reference_tick]
                ref_delta = (
                    motion["body_pos_w"][reference_tick, 0]
                    - motion["body_pos_w"][0, 0]
                )
                expected_root = initial_root + alignment @ ref_delta
                reference_xy_error = float(np.linalg.norm(robot.q[:2] - expected_root[:2]))
            rmse = float(np.sqrt(np.mean((robot.q[7:][o_from_p] - expected_q) ** 2)))
            force = np.asarray(support.total_force_n)
            reaction = np.asarray(support.selected_ground_force_n)
            floor_force = np.asarray(support.published_floor_force_n)
            cap_ok = bool(
                np.all(
                    np.abs(robot.drive_effort)
                    <= cap + CRITERIA["effort_cap_absolute_tolerance_nm"]
                )
            )
            balance_ok = bool(
                np.linalg.norm(force + reaction)
                <= CRITERIA["reaction_balance_relative_tolerance"] * max(1.0, weight)
            )
            stats["accepted_steps"] += 1
            stats["accepted_duration_s"] = float(accepted.time_s)
            stats["minimum_root_height_m"] = min(stats["minimum_root_height_m"], root_height)
            for key, value in (
                ("maximum_root_tilt_rad", tilt),
                ("maximum_root_xy_drift_m", root_drift),
                ("maximum_foot_center_xy_drift_m", foot_drift),
                ("maximum_reference_xy_error_m", reference_xy_error),
                ("maximum_joint_rmse_rad", rmse),
                ("maximum_force_bodyweight_ratio", float(np.linalg.norm(force) / weight)),
                ("maximum_nonfoot_ground_force_n", float(np.linalg.norm(floor_force - reaction))),
            ):
                stats[key] = max(stats[key], value)
            stats["effort_caps_observed"] &= cap_ok
            stats["reaction_balance_observed"] &= balance_ok
            if support.sample_time_s >= CRITERIA["settling_time_s"]:
                stats["settled_samples"] += 1
                stats["settled_vertical_load_sum"] += float(force[2] / weight)
                stats["both_loaded_samples"] += int(len(support.loaded_feet) == 2)
            journal.context["last_accepted"] = {
                "tick": tick, "time_s": accepted.time_s, "q": robot.q, "v": robot.v
            }
            journal.append(
                "accepted",
                tick=tick,
                time_s=accepted.time_s,
                load_time_s=accepted.load_time_s,
                state=state_record(robot),
                support=support,
                foot_contacts=foot_contacts(accepted, feet, ground),
                foot_geometry=geometry,
                root_height_m=root_height,
                root_tilt_rad=tilt,
                root_xy_drift_m=root_drift,
                joint_rmse_rad=rmse,
                reference_xy_error_m=reference_xy_error,
                effort_caps_observed=cap_ok,
                reaction_balance_observed=balance_ok,
            )
            if (
                root_height < CRITERIA["minimum_root_height_m"]
                or tilt > CRITERIA["maximum_fall_tilt_rad"]
                or not cap_ok
                or not balance_ok
            ):
                raise RuntimeError("accepted state failed no-fall, effort or reaction gate")
    assessment = evaluate(args.mode, stats, True, None)
    assessment_failure = None
    if not assessment["success"]:
        assessment_failure = {
            "type": "EngineeringAssessment",
            "message": "completed trace did not satisfy every declared engineering criterion",
            "failed_criteria": tuple(
                name for name, passed in assessment["gates"].items() if not passed
            ),
            "assessment": assessment["assessment"],
        }
        journal.append("first_failure", **assessment_failure)
    return {
        "schema": "g1-phymeo-support-summary-v1",
        "mode": args.mode,
        "backend": "host",
        "requested_duration_s": args.duration,
        "criteria": CRITERIA,
        "statistics": stats,
        "first_failure": assessment_failure,
        **assessment,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--scene", default="g1_primitive_actuated.xml")
    parser.add_argument("--mode", choices=("stand", "policy"), default="stand")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not math.isfinite(args.duration)
        or args.duration <= 0
        or not math.isfinite(args.dt)
        or args.dt <= 0
    ):
        parser.error("--duration and --dt must be positive finite seconds")
    journal = Journal(args.output)
    summary = None
    try:
        journal.append(
            "request",
            schema="g1-phymeo-support-trace-v1",
            arguments=vars(args),
            criteria=CRITERIA,
            python=sys.version,
            phymeo_file=ph.__file__,
            declared_status="runtime_attempt_not_prevalidated",
        )
        summary = run(args, journal)
    except BaseException as error:
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "context": journal.context,
        }
        journal.append("first_failure", **failure)
        stats = journal.context.get("statistics")
        summary = {
            "schema": "g1-phymeo-support-summary-v1",
            "mode": args.mode,
            "backend": "host",
            "requested_duration_s": args.duration,
            "criteria": CRITERIA,
            "statistics": stats,
            "first_failure": failure,
            "success": False,
        }
        if stats is not None:
            summary.update(evaluate(args.mode, stats, False, failure))
    finally:
        try:
            if summary is not None:
                journal.finish(summary)
        finally:
            journal.close()
    print(json.dumps(json_value(summary), allow_nan=False))
    return 0 if summary is not None and summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
