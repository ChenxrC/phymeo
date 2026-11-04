"""Native Newton Featherstone with and without the same URDF mimic, from identical q/v and wrenches."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from mimic_oracle import H, physical


def rotation(q):
    x, y, z, w = np.asarray(q, dtype=float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def observe(case, device, engaged):
    import newton

    path = Path(__file__).with_name(f"mimic_{case}.urdf")
    ref = physical(case)
    labels = ("lift", "screw", "idler") if case == "jaws" else ("shaft", "gear")
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    b.default_shape_cfg.density = 0
    b.add_urdf(
        str(path),
        floating=False,
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        ignore_inertial_definitions=False,
        joint_damping_mode="passive",
        up_axis=newton.Axis.Z,
    )
    names = [str(n).rsplit("/", 1)[-1] for n in b.joint_label]
    joints = [names.index(n) for n in labels]
    f, d = joints[-1], joints[-2]
    parsed = int(b.joint_mimic_joint[f])
    coeff = list(b.joint_mimic_coeffs[f])
    if parsed != d:
        raise AssertionError("native importer did not register the authored mimic")
    if not engaged:
        b.set_joint_mimic(f, None)
    iq = [b.joint_q_start[j] for j in joints]
    iv = [b.joint_qd_start[j] for j in joints]
    for i in iv:
        for name in (
            "joint_armature",
            "joint_damping",
            "joint_target_ke",
            "joint_target_kd",
            "joint_limit_ke",
            "joint_limit_kd",
        ):
            getattr(b, name)[i] = 0.0
    model = b.finalize(device=device)
    state, nxt, control = model.state(), model.state(), model.control()
    q = state.joint_q.numpy()
    v = state.joint_qd.numpy()
    q[iq] = ref["q"]
    v[iv] = ref["v"]
    state.joint_q.assign(q)
    state.joint_qd.assign(v)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    child = int(model.joint_child.numpy()[f])
    pose = state.body_q.numpy()[child].astype(float)
    R = rotation(pose[3:])
    com = pose[:3] + R @ model.body_com.numpy()[child].astype(float)
    local = [-0.003, 0.006, -0.004] if case == "jaws" else [0.22, 0.06, -0.03]
    point = pose[:3] + R @ local
    twist = state.body_qd.numpy()[child].astype(float)
    point_v = twist[:3] + np.cross(twist[3:], point - com)
    first = int(model.articulation_start.numpy()[0])
    cols = np.asarray(iv) - int(model.joint_qd_start.numpy()[first])
    jac = newton.eval_jacobian(model, state)
    com_J = jac.numpy()[0, 6 * (f - first) : 6 * (f - first + 1), :][:, cols].astype(float)
    J = com_J[:3] + np.cross(com_J[3:].T, point - com).T
    mass = newton.eval_mass_matrix(model, state, J=jac).numpy()[0][np.ix_(cols, cols)].astype(float)
    state.clear_forces()
    body_force = state.body_f.numpy() * 0
    body_force[child] = np.r_[ref["force"], np.cross(point - com, ref["force"])]
    state.body_f.assign(body_force)
    motor = control.joint_f.numpy() * 0
    motor[iv] = ref["motor"]
    control.joint_f.assign(motor)
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    solver.step(state, nxt, control, None, H)
    pre_v = state.joint_qd.numpy()[iv].astype(float)
    post_v = nxt.joint_qd.numpy()[iv].astype(float)
    post_q = nxt.joint_q.numpy()[iq].astype(float)
    a = (post_v - pre_v) / H
    react = mass @ a + ref["bias"] - J.T @ ref["force"] - motor[iv]
    return dict(
        case=case,
        engaged=engaged,
        dof_count=model.joint_dof_count,
        authored_leader=names[parsed],
        actual_model_leader=int(model.joint_mimic_joint.numpy()[f]),
        coeff=coeff,
        source_sha256=sha256(path.read_bytes()).hexdigest(),
        point_m=point.tolist(),
        point_velocity_m_s=point_v.tolist(),
        q_pre=state.joint_q.numpy()[iq].tolist(),
        v_pre=pre_v.tolist(),
        q_post=post_q.tolist(),
        v_post=post_v.tolist(),
        native_M=mass.tolist(),
        native_J=J.tolist(),
        reduced_M=(ref["T"].T @ mass @ ref["T"]).tolist(),
        applied_body_wrench=body_force[child].tolist(),
        applied_motor_full=motor[iv].tolist(),
        acceleration=a.tolist(),
        reconstructed_constraint_effort_full=react.tolist(),
    )


def audit(rows):
    maxima = {}

    def same(name, actual, expected, tol):
        error = float(np.max(abs(np.asarray(actual) - expected)))
        maxima[name] = max(error, maxima.get(name, 0.0))
        np.testing.assert_allclose(actual, expected, atol=tol, rtol=0, err_msg=name)

    for row in rows:
        ref = physical(row["case"])
        same("native mass", row["native_M"], ref["mass"], 2e-6)
        same("native point J", row["native_J"], ref["J"], 2e-6)
        same("reduced mass", row["reduced_M"], ref["T"].T @ ref["mass"] @ ref["T"], 3e-6)
        same("point", row["point_m"], ref["point"], 2e-6)
        same("velocity", row["point_velocity_m_s"], ref["J"] @ ref["v"], 2e-6)
        same(
            "offset and scale",
            row["coeff"],
            (0.004, -1.0) if row["case"] == "jaws" else (-0.125, 1.5),
            2e-7,
        )
        a = (
            ref["accel"]
            if row["engaged"]
            else np.linalg.solve(ref["mass"], ref["motor"] + ref["external"] - ref["bias"])
        )
        same("real step acceleration", row["acceleration"], a, 5e-4)
        same("real end q", row["q_post"], ref["q"] + H * (ref["v"] + H * a), 2e-7)
        expected = ref["reaction"] if row["engaged"] else np.zeros_like(a)
        same("derived full reaction", row["reconstructed_constraint_effort_full"], expected, 3e-4)
        if row["engaged"]:
            assert abs(ref["C"] @ row["v_post"]) < 3e-6
        else:
            assert abs(ref["C"] @ row["acceleration"]) > 0.2
    return maxima


def main():
    import newton
    import warp as wp

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    wp.init()
    output = dict(
        device=str(wp.get_device(args.device)),
        newton_version=newton.__version__,
        dt_s=H,
        rows=[
            observe(case, args.device, enabled)
            for case in ("jaws", "elbow")
            for enabled in (True, False)
        ],
    )
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    output["max_error"] = audit(output["rows"])
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(output["max_error"]))


if __name__ == "__main__":
    main()
