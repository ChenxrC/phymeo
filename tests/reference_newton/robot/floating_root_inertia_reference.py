"""Newton 1.7 full root COM inertia and uncollapsed fixed tools, F3 reference.

No Phymeo imports. Four fixed cases preserve all native arrays and advance
independently for 40 steps. FP32 engineering gates are declared before execution.
"""
import argparse
import hashlib
import json
import math
import traceback
from importlib.metadata import version
from pathlib import Path

import numpy as np

H = 1 / 1024
CENTER = np.array([1 / 8, -1 / 16, 1 / 32])
BODY_I = np.array([[3 / 8, -1 / 8, 0], [-1 / 8, 3 / 8, 0], [0, 0, 1 / 2]])
OFFSET = np.array([1 / 4, 1 / 8, 0])
GATES = dict(tensor=8e-6, position=8e-6, rotation=8e-6,
             velocity=8e-6, acceleration=3e-5, energy=8e-6)
SOURCE_NAMES = ("_src/sim/builder.py", "_src/sim/model.py",
                "_src/sim/articulation.py", "_src/sim/state.py",
                "_src/solvers/featherstone/solver_featherstone.py",
                "_src/solvers/featherstone/kernels.py")
MODEL_NAMES = ("body_mass", "body_com", "body_inertia", "body_inv_inertia",
               "joint_type", "joint_parent", "joint_child", "joint_X_p",
               "joint_X_c", "joint_q_start", "joint_qd_start", "joint_dof_dim",
               "joint_armature", "joint_target_ke", "joint_target_kd",
               "joint_damping", "joint_limit_ke", "joint_limit_kd", "gravity")


def plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [plain(v) for v in x]
    return x


def skew(x):
    a, b, c = x
    return np.array([[0, -c, b], [c, 0, -a], [-b, a, 0.0]])


def rotation(q):
    k = skew(q[:3])
    return np.eye(3) + 2 * q[3] * k + 2 * k @ k


def exponential(v):
    angle = np.linalg.norm(v)
    if angle == 0:
        return np.eye(3)
    k = skew(v / angle)
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def gate(row, name, actual, expected, tolerance):
    a, b = np.asarray(actual), np.asarray(expected)
    error = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")
    row.setdefault("checks", {})[name] = dict(
        passed=bool(np.isfinite(error) and error <= tolerance),
        maximum_error=error if np.isfinite(error) else None, tolerance=tolerance)


def arrays(owner, names):
    return {name: getattr(owner, name).numpy().copy() for name in names}


def record(state):
    return arrays(state, ("joint_q", "joint_qd", "body_q", "body_qd", "body_f"))


def authored(fixed):
    if fixed:
        parts = [(2.0, np.zeros(3), np.zeros(3), np.diag([.2, .3, .4])),
                 (1.0, OFFSET, np.zeros(3), np.eye(3) * .1)]
        center = OFFSET / 3
        inertia = np.diag([.3, .4, .5]) + (2 / 3) * (
            OFFSET @ OFFSET * np.eye(3) - np.outer(OFFSET, OFFSET))
        return parts, 3.0, center, inertia
    return [(2.0, np.zeros(3), CENTER, BODY_I)], 2.0, CENTER, BODY_I


def aggregate(raw, parts):
    centers, velocities, energies = [], [], []
    angular = raw["body_qd"][0, 3:].astype(float)
    total = sum(item[0] for item in parts)
    for i, (mass, _, com, tensor) in enumerate(parts):
        pose = raw["body_q"][i].astype(float)
        rot = rotation(pose[3:])
        twist = raw["body_qd"][i].astype(float)
        centers.append(pose[:3] + rot @ com)
        velocities.append(twist[:3])
        energies.append(.5 * mass * twist[:3] @ twist[:3] +
                        .5 * twist[3:] @ (rot @ tensor @ rot.T) @ twist[3:])
    center = sum(part[0] * c for part, c in zip(parts, centers, strict=True)) / total
    velocity = sum(part[0] * v for part, v in zip(parts, velocities, strict=True)) / total
    return center, velocity, angular, sum(energies)


def run_case(newton, wp, device, output, fixed, cycled):
    name = ("fixed_parts" if fixed else "full_root") + ("_cycle" if cycled else "_base")
    row = dict(name=name, checks={}, trace=[], failure=None)
    path = output / (name + ".json")

    def save():
        path.write_text(json.dumps(plain(row), indent=2, allow_nan=False), encoding="utf-8")

    try:
        parts, mass, local_center, tensor = authored(fixed)
        quat = np.array([.5, .5, .5, .5] if cycled else [0., 0., 0., 1.])
        rot0 = rotation(quat)
        shift = np.array([.5, -.25, .75]) if cycled else np.zeros(3)
        gravity = rot0 @ [0., 0., -.5]
        force = rot0 @ [1 / 4, -1 / 8, 1 / 16]
        torque = rot0 @ [1 / 8, 1 / 4, -1 / 16]
        builder = newton.ModelBuilder(gravity=tuple(gravity), up_axis=newton.Axis.Z)
        body_ids = []
        for i, (m, _, c, ii) in enumerate(parts):
            body_ids.append(builder.add_link(
                label="root" if i == 0 else "fixed_tool", mass=m, com=tuple(c),
                inertia=ii, lock_inertia=True))
        joints = [builder.add_joint_free(body_ids[0])]
        if fixed:
            joints.append(builder.add_joint_fixed(
                body_ids[0], body_ids[1], parent_xform=wp.transform(tuple(OFFSET), wp.quat_identity())))
        builder.add_articulation(joints)
        for attr in ("joint_armature", "joint_target_ke", "joint_target_kd",
                     "joint_damping", "joint_limit_ke", "joint_limit_kd"):
            values = getattr(builder, attr)
            for i in range(len(values)):
                values[i] = 0.
        model = builder.finalize(device=device)
        row.update(authored_parts=parts, aggregate_mass=mass, aggregate_com_body=local_center,
                   aggregate_inertia_body=tensor, h_s=H, count=40, gates=GATES,
                   initial_origin=shift, initial_xyzw=quat, gravity=gravity,
                   force_world_at_aggregate_com=force, torque_world_about_aggregate_com=torque,
                   model=arrays(model, MODEL_NAMES), body_labels=model.body_label,
                   joint_labels=model.joint_label, actual_device=str(model.device))
        assert model.body_count == len(parts) and model.joint_dof_count == 6
        gate(row, "authored_body_mass", row["model"]["body_mass"],
             np.array([p[0] for p in parts]), GATES["tensor"])
        gate(row, "authored_full_body_inertia", row["model"]["body_inertia"],
             np.array([p[3] for p in parts]), GATES["tensor"])
        gate(row, "authored_body_com", row["model"]["body_com"],
             np.array([p[2] for p in parts]), GATES["position"])
        state, candidate, control = model.state(), model.state(), model.control()
        control.joint_f.zero_()
        state.joint_q.assign(np.r_[shift, quat].astype(np.float32))
        state.joint_qd.zero_()
        state.clear_forces()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        jac = newton.eval_jacobian(model, state)
        native_mass = newton.eval_mass_matrix(model, state, J=jac).numpy().copy()
        basis = np.eye(6)
        basis[:3, 3:] = skew(rot0 @ (local_center - parts[0][2]))
        expected_mass = np.zeros((6, 6))
        expected_mass[:3, :3] = mass * np.eye(3)
        expected_mass[3:, 3:] = rot0 @ tensor @ rot0.T
        row.update(native_jacobian=jac.numpy().copy(), native_mass=native_mass,
                   native_from_aggregate_basis=basis, static_before=record(state))
        gate(row, "full_mass_in_aggregate_com_basis",
             basis.T @ native_mass.reshape(6, 6) @ basis, expected_mass, GATES["tensor"])
        # Distinct rest-torque experiment; subtract known gravity, never reset a live trajectory.
        wrench = np.zeros((len(parts), 6), dtype=np.float32)
        wrench[0, 3:] = rot0 @ [1., 0., 0.]
        state.body_f.assign(wrench)
        solver = newton.solvers.SolverFeatherstone(
            model, angular_damping=0., update_mass_matrix_interval=1)
        solver.step(state, candidate, control, None, H)
        rest = record(candidate)
        row.update(static_after=rest, static_wrench=wrench.copy(),
                   static_native_qdd=solver.joint_qdd.numpy().copy())
        expected_alpha = rot0 @ np.linalg.solve(tensor, [1., 0., 0.])
        gate(row, "rest_torque_angular_acceleration", rest["body_qd"][0, 3:] / H,
             expected_alpha, GATES["acceleration"])
        gate(row, "rest_gravity_com_velocity", aggregate(rest, parts)[1],
             H * gravity, GATES["velocity"])
        # Separate nonzero-velocity experiment, independently advanced by Newton.
        velocity = rot0 @ [1 / 8, -1 / 4, 1 / 16]
        omega = rot0 @ [1 / 4, -1 / 8, 3 / 8]
        center = shift + rot0 @ local_center
        orient = rot0.copy()
        state, candidate = model.state(), model.state()
        state.joint_q.assign(np.r_[shift, quat].astype(np.float32))
        root_velocity = velocity + np.cross(omega, rot0 @ (parts[0][2] - local_center))
        state.joint_qd.assign(np.r_[root_velocity, omega].astype(np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        row["window_initial"] = record(state)
        row["oracle_initial"] = dict(com=center.copy(), velocity=velocity.copy(), omega=omega.copy())
        for step in range(1, 41):
            state.clear_forces()
            before = record(state)
            actual_center = aggregate(before, parts)[0]
            root_pose = before["body_q"][0].astype(float)
            root_com = root_pose[:3] + rotation(root_pose[3:]) @ parts[0][2]
            wrench[:] = 0
            wrench[0, :3] = force
            wrench[0, 3:] = torque + np.cross(actual_center - root_com, force)
            state.body_f.assign(wrench)
            inertia = orient @ tensor @ orient.T
            acceleration = np.linalg.solve(inertia, torque - np.cross(omega, inertia @ omega))
            velocity = velocity + H * (force / mass + gravity)
            omega = omega + H * acceleration
            center = center + H * velocity
            orient = exponential(H * omega) @ orient
            solver.step(state, candidate, control, None, H)
            actual = record(candidate)
            ac, av, aw, energy = aggregate(actual, parts)
            item = dict(step=step, time_s=step * H, before=before, actual=actual,
                        input_body_f=wrench.copy(), native_qdd=solver.joint_qdd.numpy().copy(),
                        native_H=solver.H.numpy().copy(), control_joint_f=control.joint_f.numpy().copy(),
                        oracle_com=center.copy(), oracle_v=velocity.copy(), oracle_w=omega.copy(),
                        oracle_rotation=orient.copy(), actual_com=ac, actual_energy=energy)
            row["trace"].append(item)
            gate(item, "com_position", ac, center, GATES["position"])
            gate(item, "com_velocity", av, velocity, GATES["velocity"])
            gate(item, "angular_velocity", aw, omega, GATES["velocity"])
            gate(item, "root_rotation", rotation(actual["body_q"][0, 3:]), orient, GATES["rotation"])
            gate(item, "kinetic_energy", energy,
                 .5 * mass * velocity @ velocity + .5 * omega @ (orient @ tensor @ orient.T) @ omega,
                 GATES["energy"])
            for i, (_, offset, _, _) in enumerate(parts):
                gate(item, f"body_{i}_origin", actual["body_q"][i, :3],
                     center + orient @ (offset - local_center), GATES["position"])
            save()
            state, candidate = candidate, state
    except Exception:
        row["failure"] = traceback.format_exc()
    row["passed"] = (row["failure"] is None and len(row["trace"]) == 40 and
                     all(c["passed"] for section in [row, *row["trace"]]
                         for c in section.get("checks", {}).values()))
    save()
    return dict(name=name, passed=row["passed"], path=str(path.resolve()),
                steps=len(row["trace"]), failure=row["failure"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda:0"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import newton
    import warp as wp
    versions = {name: version(name) for name in ("newton", "warp-lang", "numpy")}
    if versions["newton"] != "1.7.0" or versions["warp-lang"] != "1.17.0":
        raise RuntimeError(f"pinned versions required: {versions}")
    if args.device == "cuda:0" and not wp.is_cuda_available():
        raise RuntimeError("CUDA unavailable; no fallback")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = {}
    src = Path(newton.__file__).resolve().parent
    for name in SOURCE_NAMES:
        data = (src / name).read_bytes()
        target = args.output_dir / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        sources[name] = dict(path=str(src / name), sha256=hashlib.sha256(data).hexdigest())
    driver = Path(__file__).resolve()
    data = driver.read_bytes()
    (args.output_dir / "sources" / driver.name).write_bytes(data)
    sources["driver"] = dict(path=str(driver), sha256=hashlib.sha256(data).hexdigest())
    summary = dict(versions=versions, device=args.device, sources=sources, results=[])
    for fixed in (False, True):
        for cycled in (False, True):
            result = run_case(newton, wp, args.device, args.output_dir, fixed, cycled)
            summary["results"].append(result)
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(result), flush=True)
    return 0 if all(r["passed"] for r in summary["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
