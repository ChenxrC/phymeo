"""Read paired sphere/link forces and the first changed step from two dense physical traces."""

import argparse
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from payload_angular_balance import vector

from phymeo.builders.panda import load_panda
from phymeo.core.robot import RobotState
from phymeo.dynamics.robot_dynamics import robot_dynamics_terms
from phymeo.dynamics.robot_kinematics import forward_kinematics


def generalized_reaction(model, kin, owner, point, force):
    """Take the component of the moment or force on each physical ancestor axis."""
    tau = np.zeros(model.dof_count)
    if not -1 <= owner < model.dof_count:
        raise ValueError("invalid structure owner")
    while owner >= 0:
        moment = np.cross(point - kin.position[owner], force)
        tau[owner] = kin.axis[owner] @ (moment if model.joint_types[owner] == "revolute" else force)
        owner = int(model.parents[owner])
    return tau


def pair_errors(c, center, radius, mu):
    p, fs, fr, n, v = (
        vector(c[key], 3)
        for key in (
            "point_world_m",
            "force_on_payload_N",
            "force_on_link_N",
            "normal_world",
            "relative_velocity_m_s",
        )
    )
    a, b = (vector(c[key], 3) for key in ("link_surface_m", "sphere_surface_m"))
    link_com = vector(c["link_com_world_m"], 3)
    normal = float(fs @ n)
    tangential = fs - normal * n
    norm = np.linalg.norm
    return dict(
        normal_unit=abs(norm(n) - 1),
        common_point_m=norm(p - 0.5 * (a + b)),
        surface_gap_m=norm(b - a - c["gap_m"] * n),
        sphere_radius_m=abs(norm(b - center) - radius),
        pair_force_N=norm(fs + fr),
        ball_moment_Nm=norm(np.cross(p - center, fs) - vector(c["torque_on_payload_Nm"], 3)),
        link_moment_Nm=norm(np.cross(p - link_com, fr) - vector(c["torque_on_link_Nm"], 3)),
        normal_load_N=abs(normal - c["normal_load_N"]),
        coulomb_excess_N=max(0.0, norm(tangential) - mu * normal),
        pair_power_W=abs(c["pair_power_W"] - fs @ v),
        positive_friction_power_W=max(0.0, tangential @ v),
    )


def analyze(spec, saved, previous, model):
    trace, old = saved["trace"], previous["trace"]
    if len(trace) != len(old) or len(trace) != spec["steps"]:
        raise ValueError("expected two complete equal-length traces")
    dt = spec["dt_s"]
    mu = spec["structural_material"][2]
    radius = spec["payload_geometry"]["radius_m"]
    limits = defaultdict(float)
    seen = defaultdict(int)
    first = peak = None
    prefix = True
    impulse = np.zeros(3)
    for i, (r, ref) in enumerate(zip(trace, old, strict=True)):
        if r["t"] != ref["t"] or abs(r["t"] - (i + 1) * dt) > 1e-10:
            raise ValueError("misaligned trace times")
        contacts = r["structural_contacts"]
        prior = (
            trace[i - 1]
            if i
            else dict(
                cube=saved["initial_payload_position_m"],
                q=saved["initial_arm_q"],
                v=spec["initial_v"],
            )
        )
        if not contacts:
            if first is None:
                prefix &= all(
                    r[k] == ref[k] for k in ("cube", "xyzw", "velocity", "omega", "q", "v")
                )
            if np.any(r["structural_joint_load"]):
                raise ValueError("joint load without structural contacts")
            continue
        center, q = np.asarray(prior["cube"]), np.asarray(prior["q"])
        kin = forward_kinematics(model, q)
        tau, force = np.zeros(model.dof_count), np.zeros(3)
        for c in contacts:
            errors = pair_errors(c, center, radius, mu)
            for k, val in errors.items():
                if not np.isfinite(val):
                    raise ValueError(f"nonfinite {k}")
                limits[k] = max(limits[k], float(val))
            point = np.asarray(c["point_world_m"])
            fs = np.asarray(c["force_on_payload_N"])
            tau += generalized_reaction(
                model, kin, c["owner_index"], point, np.asarray(c["force_on_link_N"])
            )
            force += fs
            seen[c["shape"]] += 1
            amount = np.linalg.norm(fs)
            if peak is None or amount > peak["force_N"]:
                peak = dict(
                    t_entry_s=i * dt, shape=c["shape"], force_N=float(amount), gap_m=c["gap_m"]
                )
        limits["joint_projection"] = max(
            limits["joint_projection"],
            float(np.max(np.abs(tau - vector(r["structural_joint_load"], model.dof_count)))),
        )
        impulse += dt * force
        if first is None and np.any(force):
            _, _, _, inertia, _ = robot_dynamics_terms(model, RobotState(q, np.asarray(prior["v"])))
            measured_ball = spec["mass_kg"] * (np.asarray(r["velocity"]) - ref["velocity"])
            measured_arm = inertia @ (np.asarray(r["v"]) - ref["v"])
            first = dict(
                index=i,
                t_entry_s=i * dt,
                t_exit_s=r["t"],
                prefix_identical=bool(prefix),
                shape_ids=[c["shape"] for c in contacts],
                ball_J_predicted_Ns=(dt * force).tolist(),
                ball_J_measured_difference_Ns=measured_ball.tolist(),
                joint_J_predicted=(dt * tau).tolist(),
                joint_J_measured_difference=measured_arm.tolist(),
            )
    if first is None:
        raise ValueError("recording has no structural impulse")
    return dict(
        steps=len(trace),
        contacts_by_shape=dict(seen),
        first=first,
        peak=peak,
        structure_impulse_Ns=impulse.tolist(),
        maximum_errors=dict(limits),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "trace", "baseline", "baseline-inputs", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    raw = {k: getattr(args, k).read_bytes() for k in ("inputs", "trace", "baseline")}
    spec, saved, old = (json.loads(raw[k]) for k in ("inputs", "trace", "baseline"))
    raw["baseline_inputs"] = args.baseline_inputs.read_bytes()
    baseline_spec = json.loads(raw["baseline_inputs"])
    if {k: v for k, v in spec.items() if k != "structural_material"} != {
        k: v for k, v in baseline_spec.items() if k != "structural_material"
    }:
        raise ValueError("inputs differ beyond the structural contact law")
    hashes = {k: sha256(v).hexdigest() for k, v in raw.items()}
    if old.get("inputs_sha256", hashes["baseline_inputs"]) != hashes["baseline_inputs"]:
        raise ValueError("baseline input binding mismatch")
    if saved.get("inputs_sha256", hashes["inputs"]) != hashes["inputs"]:
        raise ValueError("input binding mismatch")
    result = analyze(spec, saved, old, load_panda().model)
    result["source_hashes"] = hashes
    with args.out.open("x", encoding="utf8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps(result, allow_nan=False))
    if max(result["maximum_errors"].values()) > 1e-5:
        raise ValueError("saved structural force pair, geometry or projection did not agree")


if __name__ == "__main__":
    main()
