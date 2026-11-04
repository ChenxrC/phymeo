"""Reconstruct a spherical payload's support forces from saved material points and forces."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np


def analyze(inputs, run):
    radius = inputs["payload_geometry"]["radius_m"]
    if run["payload_geometry"] != {"kind": "sphere", "radius_m": radius}:
        raise ValueError("expected a matching spherical payload")
    mu = inputs["pad_material"][2]
    mg = -inputs["mass_kg"] * inputs["gravity_m_s2"][2]
    dt = inputs["dt_s"]
    times = inputs["knot_times_s"]
    if run["steps"] != inputs["steps"] or len(run["trace"]) != run["steps"]:
        raise ValueError("expected a dense physical trajectory")
    if run["cube_mass_kg"] != inputs["mass_kg"] or run["finger_friction"] != mu:
        raise ValueError("payload mass or pad friction differs from the input")
    center = np.asarray(
        run.get(
            "initial_payload_position_m",
            run.get("initial_cube_center_m", inputs["body_position_m"]),
        )
    )
    events = {}
    maxima = np.zeros(6)
    carry_samples = []
    for i, row in enumerate(run["trace"], 1):
        time = row["t"]
        if abs(time - i * dt) > 1e-10:
            raise ValueError("missing or reordered step")
        arms = np.asarray(row["pad_points_before_m"]) - center
        forces = np.asarray(row["pad_forces_N"])
        table_arms = np.asarray(row["table_points_before_m"]) - center
        table = np.asarray(row["table_forces_N"])
        if (
            arms.shape != (8, 3)
            or forces.shape != (8, 3)
            or table_arms.shape != (1, 3)
            or table.shape != (1, 3)
        ):
            raise ValueError("expected eight finger samples and one finite-table point")
        lengths = np.linalg.norm(arms, axis=1)
        n = arms / lengths[:, None]
        normal = np.einsum("ij,ij->i", forces, n)
        tangent = forces - normal[:, None] * n
        load_z = -normal * n[:, 2]
        cap_z = mu * normal * np.sqrt(np.maximum(0, 1 - n[:, 2] ** 2))
        vertical = -forces[:, 2].sum()
        wrench = np.r_[
            table.sum(0) - forces.sum(0),
            np.cross(table_arms, table).sum(0) - np.cross(arms, forces).sum(0),
        ]
        structure_z = 0.0
        for contact in row.get("structural_contacts", ()):
            structural_force = np.asarray(contact["force_on_payload_N"])
            lever = np.asarray(contact["point_world_m"]) - center
            wrench[:3] += structural_force
            wrench[3:] += np.cross(lever, structural_force)
            structure_z += structural_force[2]
        declared = np.asarray(row["applied_payload_wrench"])
        actual_support = np.asarray(row["pad_normal_N"])
        error = np.array(
            [
                np.linalg.norm(wrench[:3] - declared[:3]),
                np.linalg.norm(wrench[3:] - declared[3:]),
                np.max(np.abs(normal.reshape(2, 4).sum(1) - actual_support)),
                np.max(np.maximum(0, np.linalg.norm(tangent, axis=1) - mu * normal)),
                abs(np.linalg.norm(table_arms[0]) - radius),
                max(0.0, float(-normal.min())),
            ]
        )
        if not np.isfinite(error).all():
            raise ValueError("invalid contact measurements")
        maxima = np.maximum(maxima, error)
        record = dict(
            t_s=time,
            pad_normal_z_N=float(load_z.sum()),
            pad_tangent_z_N=float(-tangent[:, 2].sum()),
            pad_total_z_N=float(vertical),
            pad_upward_limit_N=float((load_z + cap_z).sum()),
            table_z_N=float(table[0, 2]),
            structure_z_N=float(structure_z),
            bilateral_N=float(min(actual_support)),
        )
        if min(actual_support) > 0 and "first_bilateral" not in events:
            events["first_bilateral"] = record
        if abs(time - times[2]) < dt / 2:
            events["lift_command_start"] = record
        if times[2] <= time <= times[4]:
            carry_samples.append(record)
        if abs(time - times[4]) < dt / 2:
            events["transfer_end"] = record
        center = np.asarray(row["cube"])
    if not carry_samples:
        raise ValueError("no lift/transfer samples")
    events["largest_downward_radial_force_in_lift"] = min(
        carry_samples, key=lambda v: v["pad_normal_z_N"]
    )
    return dict(
        weight_N=mg,
        steps=run["steps"],
        events=events,
        samples_in_lift_and_transfer=len(carry_samples),
        table_unloaded_and_finger_vertical_at_least_weight_samples=sum(
            v["table_z_N"] < 0.001 and v["pad_total_z_N"] >= mg for v in carry_samples
        ),
        maximum_errors=dict(
            zip(
                (
                    "force_N",
                    "torque_Nm",
                    "normal_load_N",
                    "coulomb_excess_N",
                    "table_radius_m",
                    "tensile_N",
                ),
                map(float, maxima),
                strict=True,
            )
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "trace", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    raw, trajectory = args.inputs.read_bytes(), args.trace.read_bytes()
    spec, saved = json.loads(raw), json.loads(trajectory)
    digest = sha256(raw).hexdigest()
    if "inputs_sha256" in saved and saved["inputs_sha256"] != digest:
        raise ValueError("trace/input mismatch")
    result = analyze(spec, saved)
    result.update(inputs_sha256=digest, trace_sha256=sha256(trajectory).hexdigest())
    with args.out.open("x", encoding="utf8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps(result["events"]["lift_command_start"], allow_nan=False))
    if max(result["maximum_errors"].values()) > 1e-8:
        raise ValueError(
            f"saved contact points do not explain applied wrench or material capacity: {result['maximum_errors']}"
        )


if __name__ == "__main__":
    main()
