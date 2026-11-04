"""Observe an uncollapsed Newton URDF and step its independently collapsed twin on the chosen device."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from fixed_tool_oracle import CASES, FORCE, MOTOR, H, reference

ASSET = Path(__file__).with_name("fixed_tool_probe.urdf")


def rotation(q):
    x, y, z, w = np.asarray(q, dtype=float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def build(device, folded):
    import newton

    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    b.default_shape_cfg.density = 0
    b.add_urdf(
        str(ASSET),
        floating=False,
        collapse_fixed_joints=folded,
        ignore_inertial_definitions=False,
        enable_self_collisions=False,
        joint_damping_mode="passive",
        up_axis=newton.Axis.Z,
    )
    labels = [str(n).rsplit("/", 1)[-1] for n in b.joint_label]
    active = [labels.index(n) for n in ("shoulder", "extension")]
    iq = [b.joint_q_start[j] for j in active]
    iv = [b.joint_qd_start[j] for j in active]
    for j in iv:
        for key in (
            "joint_armature",
            "joint_damping",
            "joint_target_ke",
            "joint_target_kd",
            "joint_limit_ke",
            "joint_limit_kd",
        ):
            getattr(b, key)[j] = 0.0
    model = b.finalize(device=device)
    bodies = {str(label).rsplit("/", 1)[-1]: i for i, label in enumerate(b.body_label)}
    native = model.joint_child.numpy()[active]
    tip_body = native[-1] if folded else bodies["probe"]
    (tip,) = np.flatnonzero(
        (model.shape_type.numpy() == int(newton.GeoType.SPHERE))
        & (model.shape_body.numpy() == tip_body)
    )
    return model, iq, iv, bodies, native, int(tip)


def evaluate(setup, q, v, folded):
    import newton

    model, iq, iv, bodies, native, tip_shape = setup
    s = model.state()
    coordinates = s.joint_q.numpy()
    rates = s.joint_qd.numpy()
    coordinates[iq] = q
    rates[iv] = v
    s.joint_q.assign(coordinates)
    s.joint_qd.assign(rates)
    newton.eval_fk(model, s.joint_q, s.joint_qd, s)
    poses = s.body_q.numpy().astype(float)
    twists = s.body_qd.numpy().astype(float)
    local_com = model.body_com.numpy().astype(float)
    local_I = model.body_inertia.numpy().astype(float)
    masses = model.body_mass.numpy().astype(float)
    R = np.array([rotation(row[3:]) for row in poses])
    c = poses[:, :3] + np.einsum("nij,nj->ni", R, local_com)
    owner = int(model.shape_body.numpy()[tip_shape])
    offset = model.shape_transform.numpy()[tip_shape, :3].astype(float)
    point = poses[owner, :3] + R[owner] @ offset
    velocity = twists[owner, :3] + np.cross(twists[owner, 3:], point - c[owner])
    # Native rows are in joint order; translate the native COM Jacobian to the
    # actual sphere centre before comparison with the independent point oracle.
    joint_start = int(model.articulation_start.numpy()[0])
    (owner_joint,) = np.flatnonzero(model.joint_child.numpy() == owner)
    start = 6 * (int(owner_joint) - joint_start)
    cols = np.asarray(iv) - int(model.joint_qd_start.numpy()[joint_start])
    jac = newton.eval_jacobian(model, s)
    com_J = jac.numpy()[0, start : start + 6, :][:, cols].astype(float)
    tip_J = com_J[:3] + np.cross(com_J[3:].T, point - c[owner]).T
    native_mass = newton.eval_mass_matrix(model, s, J=jac).numpy()[0]
    row = dict(
        q=q.tolist(),
        v=v.tolist(),
        body_count=model.body_count,
        shape_id=tip_shape,
        owner_id=owner,
        shape_radius_m=float(model.shape_scale.numpy()[tip_shape, 0]),
        tip_m=point.tolist(),
        tip_velocity_m_s=velocity.tolist(),
        omega_rad_s=twists[owner, 3:].tolist(),
        tip_J=tip_J.tolist(),
        M=native_mass[np.ix_(cols, cols)].astype(float).tolist(),
    )
    which = list(bodies) if not folded else [n for n, i in bodies.items() if i in native]
    row["frames"] = {
        n: dict(p=poses[bodies[n], :3].tolist(), R=R[bodies[n]].tolist()) for n in which
    }
    row["groups"] = []
    group_members = (
        [[int(i)] for i in native]
        if folded
        else [
            [bodies[n] for n in names]
            for names in (("upper", "elbow"), ("slide", "collet", "probe"))
        ]
    )
    momentum = np.zeros(6)
    kinetic = 0.0
    for ids in group_members:
        group_mass = sum(masses[i] for i in ids)
        group_com = sum(masses[i] * c[i] for i in ids) / group_mass
        group_inertia = np.zeros((3, 3))
        for i in ids:
            tensor = R[i] @ local_I[i] @ R[i].T
            arm = c[i] - group_com
            group_inertia += tensor + masses[i] * ((arm @ arm) * np.eye(3) - np.outer(arm, arm))
            linear = masses[i] * twists[i, :3]
            spin = tensor @ twists[i, 3:]
            kinetic += (twists[i, :3] @ linear + twists[i, 3:] @ spin) / 2
            momentum += np.r_[linear, np.cross(c[i], linear) + spin]
        row["groups"].append(
            dict(
                mass=float(group_mass),
                com_m=group_com.tolist(),
                inertia_world=group_inertia.tolist(),
            )
        )
    row["energy_J"] = float(kinetic)
    row["momentum"] = momentum.tolist()
    s.clear_forces()
    external = np.zeros_like(s.body_f.numpy())
    external[owner] = np.r_[FORCE, np.cross(point - c[owner], FORCE)]
    s.body_f.assign(external)
    control = model.control()
    held = np.zeros_like(control.joint_f.numpy())
    held[iv] = MOTOR
    control.joint_f.assign(held)
    nxt = model.state()
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    solver.step(s, nxt, control, None, H)
    row["force_wrench_world"] = external[owner].tolist()
    row["v_before"] = s.joint_qd.numpy()[iv].astype(float).tolist()
    row["v_after"] = nxt.joint_qd.numpy()[iv].astype(float).tolist()
    row["q_after"] = nxt.joint_q.numpy()[iq].astype(float).tolist()
    row["acceleration"] = ((np.asarray(row["v_after"]) - row["v_before"]) / H).tolist()
    return row


def audit(rows, *, folded):
    metrics = {}

    def matching(key, actual, expected, tol):
        error = float(np.max(abs(np.asarray(actual) - expected)))
        entry = metrics.setdefault(key, dict(limit=tol, max_error=0.0, passes=True))
        entry["max_error"] = max(error, entry["max_error"])
        entry["passes"] = bool(entry["passes"] and np.isfinite(error) and error <= tol)

    for row in rows:
        q, v = np.array(row["q"]), np.array(row["v"])
        ref = reference(q, v)
        for name, (p, R) in ref["frames"].items():
            if folded and name not in row["frames"]:
                continue
            measured = row["frames"][name]
            matching("authored_position_m", measured["p"], p, 5e-6)
            matching("authored_orientation", measured["R"], R, 5e-6)
        matching("tip_m", row["tip_m"], ref["tip"], 5e-6)
        matching("tip_velocity_m_s", row["tip_velocity_m_s"], ref["tip_velocity"], 3e-6)
        matching("native_point_J", row["tip_J"], ref["jacobian"], 5e-6)
        matching("native_M", row["M"], ref["M"], 5e-6)
        matching("omega_rad_s", row["omega_rad_s"], ref["omega"], 3e-6)
        matching("radius_m", row["shape_radius_m"], 0.025, 1e-7)
        for group, (m, c, tensor) in zip(row["groups"], ref["combined"], strict=True):
            matching("mass_kg", group["mass"], m, 2e-6)
            matching("com_m", group["com_m"], c, 5e-6)
            matching("inertia_kg_m2", group["inertia_world"], tensor, 2e-6)
        matching("momentum", row["momentum"], ref["momentum"], 7e-6)
        matching("energy_J", row["energy_J"], ref["energy"], 3e-6)
        matching("acceleration", row["acceleration"], ref["acceleration"], 1.5e-3)
        matching("position_post", row["q_after"], q + H * (v + H * ref["acceleration"]), 4e-7)
        row["independent_acceleration"] = ref["acceleration"].tolist()
        row["independent_M"] = ref["M"].tolist()
    return dict(qualified=all(x["passes"] for x in metrics.values()), metrics=metrics)


def main():
    import newton
    import warp as wp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--require",
        choices=("all", "unmerged"),
        default="all",
        help="which table must pass for process success; full per-mode qualification is always written",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    wp.init()
    result = dict(
        device=str(wp.get_device(args.device)),
        newton_version=newton.__version__,
        urdf_sha256=sha256(ASSET.read_bytes()).hexdigest(),
        h_s=H,
        applied_tip_force_N=FORCE.tolist(),
        motor_effort=MOTOR.tolist(),
        unmerged=[],
        merged=[],
    )
    for folded, key in ((False, "unmerged"), (True, "merged")):
        setup = build(args.device, folded)
        for values in CASES:
            result[key].append(evaluate(setup, np.array(values[:2]), np.array(values[2:]), folded))
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    result["required_profile"] = args.require
    result["qualification"] = {
        key: audit(result[key], folded=key == "merged") for key in ("unmerged", "merged")
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    for mode, table in result["qualification"].items():
        print(
            mode,
            "qualified",
            table["qualified"],
            "failed",
            {k: v for k, v in table["metrics"].items() if not v["passes"]},
        )
    required = ("unmerged", "merged") if args.require == "all" else ("unmerged",)
    return int(not all(result["qualification"][mode]["qualified"] for mode in required))


if __name__ == "__main__":
    raise SystemExit(main())
