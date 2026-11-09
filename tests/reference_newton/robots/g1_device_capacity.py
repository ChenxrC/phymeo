"""Two real G1 free-flight steps: resident TYX versus prepared HOST dynamics.

AUTHORING_ONLY / UNEXECUTED at delivery. Run in the parent's sole serial queue.
This is device/HOST implementation agreement, not an independent dynamics oracle,
contact/standing acceptance, a policy rollout or a performance benchmark.

Example (choose a new output path; existing output is never overwritten):
  python -B tests/reference_newton/robots/g1_device_capacity.py --assets <assets>
    --mode interpret --output <new.json> --cache-root <cache>
"""

import argparse
import hashlib
import json
import math
import platform
import sys
import traceback
from dataclasses import fields, is_dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

import phymeo as ph
from phymeo.builders.floating_mjcf import load_floating_mjcf
from phymeo.dynamics.floating_robot import FloatingRobotState, floating_kinematics
from phymeo.dynamics.rotation import quaternion_matrix
from phymeo.solvers.floating_tree import (
    FloatingRobotInputs,
    floating_dynamics_terms,
    floating_kinetic_energy,
    step_floating_robot,
)

H = 0.002
FRAMES = (0, 200)
WORLD = 41
DYNAMICS_RTOL, DYNAMICS_ATOL = 5e-4, 2e-5
STATE_ATOL = 2e-5
MAX_OUTPUT_BYTES = 64 * 1024 * 1024


def encode(value):
    """Retain array shape/dtype/raw bytes even in a rejected nonfinite attempt."""
    if isinstance(value, np.ndarray):
        return {
            "shape": value.shape,
            "dtype": value.dtype.str,
            "bytes_hex": value.tobytes(order="C").hex(),
            "values": encode(value.tolist()),
        }
    if isinstance(value, np.generic):
        return encode(value.item())
    if is_dataclass(value):
        return {field.name: encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_ieee_value": repr(value)}
    return value


def file_fact(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def comparison(actual, expected, *, rtol=DYNAMICS_RTOL, atol=DYNAMICS_ATOL):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return {"passed": False, "actual_shape": actual.shape, "expected_shape": expected.shape}
    difference = np.abs(actual - expected)
    allowance = atol + rtol * np.abs(expected)
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    return {
        "passed": bool(finite and np.all(difference <= allowance)),
        "shape": actual.shape,
        "rtol": rtol,
        "atol": atol,
        "max_abs_error": float(np.max(difference)),
        "max_error_to_allowance": float(np.max(difference / allowance)),
    }


def state_from_motion(asset, motion, frame, policy_to_asset):
    """Official origin/world-omega convention -> public root COM/world omega."""
    wxyz = np.asarray(motion["body_quat_w"][frame, 0], dtype=np.float64).copy()
    if wxyz.shape != (4,) or not np.isfinite(wxyz).all() or np.linalg.norm(wxyz) == 0:
        raise ValueError("invalid official root quaternion")
    wxyz /= np.linalg.norm(wxyz)
    xyzw = wxyz[[1, 2, 3, 0]]
    rotation = quaternion_matrix(xyzw[None])[0]
    omega = np.asarray(motion["body_ang_vel_w"][frame, 0], dtype=np.float64)
    origin_velocity = np.asarray(motion["body_lin_vel_w"][frame, 0], dtype=np.float64)
    offset = rotation @ asset.model.root_com_m
    # Same common translation as the already executed fixed-pose HOST/native check.
    translation = np.array([0.0, 0.0, 2.0])
    q = np.r_[
        np.asarray(motion["body_pos_w"][frame, 0], dtype=np.float64) + translation,
        xyzw,
        np.asarray(motion["joint_pos"][frame, policy_to_asset], dtype=np.float64),
    ]
    v = np.r_[
        origin_velocity + np.cross(omega, offset),
        omega,
        np.asarray(motion["joint_vel"][frame, policy_to_asset], dtype=np.float64),
    ]
    provenance = {
        "official_root_position": motion["body_pos_w"][frame, 0],
        "official_root_quaternion_wxyz": motion["body_quat_w"][frame, 0],
        "official_root_origin_velocity_world": origin_velocity,
        "official_root_angular_velocity_world": omega,
        "root_com_offset_world": offset,
        "common_world_translation_m": translation,
        "official_joint_position": motion["joint_pos"][frame],
        "official_joint_velocity": motion["joint_vel"][frame],
    }
    return FloatingRobotState(q, v), provenance


def run_pose(asset, state, provenance, effort, frame, mode, cache_root, row):
    inputs = FloatingRobotInputs(effort, np.zeros((30, 3)), np.zeros((30, 3)))
    row.update(
        stage="host_preparation",
        source_state=state,
        source_motion=provenance,
        inputs=inputs,
        physical_model=asset.model,
    )
    terms = floating_dynamics_terms(asset.model, state, inputs)
    host = step_floating_robot(asset.model, state, inputs, H)
    row.update(
        host_entry_terms=terms,
        host_step=host,
        host_entry_fk=floating_kinematics(asset.model, state),
        host_entry_kinetic_energy_j=floating_kinetic_energy(asset.model, state),
    )
    if not host.report.accepted:
        raise AssertionError(f"frame {frame}: HOST step rejected: {host.report.reason}")
    host_fk = floating_kinematics(asset.model, host.state)
    host_energy = floating_kinetic_energy(asset.model, host.state)
    row.update(host_accepted_fk=host_fk, host_accepted_kinetic_energy_j=host_energy)
    tree = ph.FloatingRobotTree(
        asset.model,
        state,
        kp=np.zeros(29),
        kd=np.zeros(29),
        effort_limit=np.asarray(asset.effort_limit, dtype=np.float64),
    )
    command = tree.inputs(np.column_stack((np.zeros(29), np.zeros(29), effort)))
    row.update(command=command, declared_effort_limit=tree.effort_limit, stage="device_create")
    batch = None
    try:
        batch = ph.TyxWorldBatch(
            [(WORLD, tree)], mode, cache_root=cache_root / f"frame-{frame:04d}"
        )
        with batch:
            entry = batch.snapshot()[WORLD]
            row.update(
                device_entry=entry,
                device_entry_fk=batch.tree_kinematics()[WORLD],
                device_entry_summary=batch.summary()[WORLD],
                stage="device_step",
            )
            batch.set_controls({WORLD: command})
            flags = batch.step(H)
            row.update(device_acceptance=flags, physical_steps=int(flags == {WORLD: True}))
            row.update(
                device_accepted=batch.snapshot()[WORLD],
                device_diagnostic=batch.tree_diagnostics()[WORLD],
                device_accepted_fk=batch.tree_kinematics()[WORLD],
                device_accepted_summary=batch.summary()[WORLD],
                device_stats=batch.stats(),
            )
            if flags != {WORLD: True}:
                raise AssertionError(f"frame {frame}: resident physical step rejected")
            diagnostic = row["device_diagnostic"]
            geometry = row["device_accepted_fk"]
            accepted = row["device_accepted"]
            if diagnostic is None:
                raise AssertionError(f"frame {frame}: accepted step has no physical diagnostic")
            if (
                geometry.joint_count,
                geometry.body_count,
                geometry.velocity_count,
                geometry.configuration_count,
            ) != (29, 30, 35, 36):
                raise AssertionError("G1 physical N/B/D/Q do not match 29/30/35/36")
            if entry.tick != 0 or accepted.tick != 1:
                raise AssertionError(
                    "each source frame must independently reset and advance one step"
                )
            checks = {
                "physical_mass": comparison(diagnostic.mass, terms.mass_matrix),
                "physical_bias": comparison(diagnostic.bias, terms.bias),
                "continuous_acceleration": comparison(diagnostic.acceleration, terms.acceleration),
                "entry_q": comparison(entry.position, state.q, rtol=0, atol=STATE_ATOL),
                "entry_v": comparison(entry.velocity, state.v, rtol=0, atol=STATE_ATOL),
                "accepted_q": comparison(accepted.position, host.state.q, rtol=0, atol=STATE_ATOL),
                "accepted_v": comparison(accepted.velocity, host.state.v, rtol=0, atol=STATE_ATOL),
                "accepted_body_origins": comparison(
                    geometry.link_position_m, host_fk.position, rtol=0, atol=STATE_ATOL
                ),
                "accepted_body_rotations": comparison(
                    geometry.link_rotation, host_fk.rotation, rtol=0, atol=STATE_ATOL
                ),
                "accepted_body_com": comparison(
                    geometry.com_position_m, host_fk.com_position, rtol=0, atol=STATE_ATOL
                ),
                "accepted_body_com_velocity": comparison(
                    geometry.linear_velocity_m_s, host_fk.com_velocity
                ),
                "accepted_body_angular_velocity": comparison(
                    geometry.angular_velocity_rad_s, host_fk.angular_velocity
                ),
                "kinetic_energy": comparison(
                    np.array([row["device_accepted_summary"]["kinetic_energy_j"]]),
                    np.array([host_energy]),
                ),
                "joint_drive": comparison(diagnostic.drive, effort, rtol=0, atol=STATE_ATOL),
            }
            row.update(stage="agreement_checks", checks=checks)
            for name, result in checks.items():
                if not result["passed"]:
                    raise AssertionError(f"frame {frame}: {name} agreement failed: {result}")
            if diagnostic.equation_residual > 2e-5:
                raise AssertionError(f"frame {frame}: original step equation residual gate failed")
            row.update(stage="passed", physical_steps=1, agreement_passed=True)
    finally:
        if batch is not None:
            batch.close()
            row["closed_device_stats"] = batch.stats()


def run(args, report):
    assets = args.assets.resolve(strict=True)
    path, motion_path = assets / "g1_primitive_actuated.xml", assets / "dance1_subject2.npz"
    report["sources"] = {
        "mjcf": file_fact(path),
        "motion": file_fact(motion_path),
        "script": file_fact(__file__),
    }
    asset = load_floating_mjcf(path)
    if asset.model.body_count != 30 or len(asset.joint_names) != 29:
        raise ValueError("expected the actual 29-joint G1 primitive asset")
    if asset.model.children.joint_types != ("revolute",) * 29:
        raise ValueError("this probe requires all 29 physical G1 revolute joints")
    if not np.any(asset.model.joint_armature) or not np.any(asset.model.joint_damping):
        raise ValueError("the supplied G1 must retain actual armature and damping")
    motors = ET.parse(path).getroot().findall("actuator/motor")
    policy_names = tuple(motor.get("joint") for motor in motors)
    if (
        len(policy_names) != 29
        or len(set(policy_names)) != 29
        or set(policy_names) != set(asset.joint_names)
    ):
        raise ValueError("actuator-order official motion mapping is incomplete or ambiguous")
    policy_to_asset = np.array(
        [policy_names.index(name) for name in asset.joint_names], dtype=np.int64
    )
    policy_effort = 0.2 * np.sin(np.arange(29) + 1)
    effort = policy_effort[policy_to_asset]
    if np.any(abs(effort) >= asset.effort_limit):
        raise ValueError("fixed declared effort must lie strictly within original caps")
    report.update(
        imported_joint_names=asset.joint_names,
        official_motor_joint_order=policy_names,
        policy_index_for_asset_joint=policy_to_asset,
        official_order_effort=policy_effort,
        physical_joint_effort=effort,
        asset=asset,
    )
    with np.load(motion_path, allow_pickle=False) as motion:
        for frame in FRAMES:
            row = {"frame": frame, "mode": args.mode, "stage": "prepare"}
            report["poses"].append(row)
            state, provenance = state_from_motion(asset, motion, frame, policy_to_asset)
            run_pose(asset, state, provenance, effort, frame, args.mode, args.cache_root, row)
    report.update(status="passed", physical_steps=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema": "g1-h1-device-capacity-v1",
        "status": "running",
        "poses": [],
        "comparison_kind": "device_host_implementation_agreement_not_independent_oracle",
        "backend": args.mode,
        "dt_s": H,
        "declared_frames": FRAMES,
        "python": sys.version,
        "platform": platform.platform(),
        "tolerances": {
            "dynamics_rtol": DYNAMICS_RTOL,
            "dynamics_atol": DYNAMICS_ATOL,
            "state_absolute": STATE_ATOL,
        },
        "scope": "two independent free-flight single steps; no contacts or policy rollout",
    }
    failed = False
    # Reserve the output before physics. Re-running cannot erase a first failure.
    with args.output.open("x", encoding="utf-8") as stream:
        try:
            run(args, report)
        except Exception as exc:
            failed = True
            report.update(
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        report["physical_steps"] = sum(row.get("physical_steps", 0) for row in report["poses"])
        payload = json.dumps(encode(report), ensure_ascii=False, allow_nan=False, indent=2)
        if len(payload.encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise OSError("full G1 device agreement record exceeds the fixed 64 MiB output bound")
        stream.write(payload)
        stream.flush()
    print(
        json.dumps(
            {
                "status": report["status"],
                "mode": args.mode,
                "output": str(args.output),
                "poses_recorded": len(report["poses"]),
                "error": report.get("error"),
            }
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
