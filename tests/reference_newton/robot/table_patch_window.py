"""Two real joints press, twist and lift an offset box shoe on a clipped finite top face."""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np

from phymeo.builders.urdf import load_urdf
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotInputs, RobotState
from phymeo.core.scene import SceneInputs, SceneState
from phymeo.dynamics.rotation import quaternion_matrix
from phymeo.solvers.fixed_table_contact import FixedTableContacts
from phymeo.solvers.robot import make_robot_runtime
from phymeo.solvers.sphere_link_contact import LinkSolid

ASSET = Path(__file__).with_name("table_patch_press.urdf")
CENTER = np.array([0.0, 0.0, -0.03125])
HALF = np.array([0.125, 0.125, 0.03125])
Q0 = np.array([0.0, np.arctan2(0.6, 0.8)])
H = 1 / 1024
STEPS, HOLD, SPIN = 96, 24, 48
LAW = RobotContactMaterial(4096.0, 8.0, 0.35, 0.0625)


def motor(k):
    return np.array([0.0 if k < SPIN else 8.0, 0.03 if HOLD <= k < SPIN else 0.0])


def row(k, pre, v0, q, v, load, Q):
    return dict(
        t=(k + 1) * H,
        pre_q=pre.tolist(),
        pre_v=v0.tolist(),
        q=q.tolist(),
        v=v.tolist(),
        motor=motor(k).tolist(),
        robot_on_table_wrench=load.table_wrench.tolist(),
        joint_load=Q.tolist(),
        contacts=load.contacts,
    )


def run_phy():
    asset = load_urdf(ASSET, gravity=(0, 0, -2))
    law = FixedTableContacts.from_asset(asset, center_m=CENTER, half_extents_m=HALF, material=LAW)
    state = SceneState({"robot": RobotState(Q0.copy(), np.zeros(2))})
    rows = []
    with make_robot_runtime(asset.model) as rt:
        for k in range(STEPS):
            pre = state.domains["robot"]
            load, Q = law.evaluate_robot(asset.model, pre)
            step = rt.step(
                state,
                SceneInputs(
                    {"robot": RobotInputs(motor(k) + Q, np.zeros((2, 3)), np.zeros((2, 3)))}
                ),
                H,
            )
            assert step.report.accepted
            state = step.state
            end = state.domains["robot"]
            rows.append(row(k, pre.q, pre.v, end.q, end.v, load, Q))
    return rows, None


def native_patch(newton, wp, model, state, shoe, floor, body, control, index):
    # Use the installed 1.7 native narrow phase and its own solver step in a distinct output
    # state. Separately recompute the original kernel into a scratch wrench buffer.
    from panda_native_contact_audit import kernel_wrench

    pipe = newton.CollisionPipeline(
        model,
        broad_phase="explicit",
        shape_pairs_filtered=wp.array(
            [(min(shoe, floor), max(shoe, floor))], dtype=wp.vec2i, device="cpu"
        ),
        reduce_contacts=False,
        rigid_contact_max=64,
    )
    stream = pipe.contacts()
    pipe.collide(state, stream)
    n = int(stream.rigid_contact_count.numpy()[0])
    if not 0 < n < 64:
        raise ValueError(f"native contact count {n}")
    zeros = []
    for typ in (float, wp.vec3, wp.vec3):
        zeros.append(wp.empty(64, dtype=typ, device="cpu"))
    newton.eval_rigid_contact_kinematics(
        model,
        state,
        stream,
        out_distance=zeros[0],
        out_point0_world=zeros[1],
        out_point1_world=zeros[2],
    )
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    after = model.state()
    state.clear_forces()
    solver.step(state, after, control, stream, H)
    wrench = kernel_wrench(model, state, solver, stream, body, model.shape_material_mu)
    gaps, p0, p1 = [a.numpy()[:n] for a in zeros]
    a, b = stream.rigid_contact_shape0.numpy()[:n], stream.rigid_contact_shape1.numpy()[:n]
    norm = stream.rigid_contact_normal.numpy()[:n]
    records = []
    for i in range(n):
        assert {int(a[i]), int(b[i])} == {shoe, floor}
        floor_first = a[i] == floor
        records.append(
            dict(
                gap_m=float(gaps[i]),
                normal_table_to_shoe=(norm[i] if floor_first else -norm[i]).tolist(),
                shoe_point_m=(p1[i] if floor_first else p0[i]).tolist(),
                table_point_m=(p0[i] if floor_first else p1[i]).tolist(),
                kernel_wrench=kernel_wrench(
                    model, state, solver, stream, body, model.shape_material_mu, i
                ).tolist(),
            )
        )
    net_from_dp = (
        2 * (float(after.joint_qd.numpy()[index]) - float(state.joint_qd.numpy()[index])) / H + 4.0
    )
    np.testing.assert_allclose(
        wrench, np.array([r["kernel_wrench"] for r in records]).sum(axis=0), atol=2e-5
    )
    np.testing.assert_allclose(wrench[2], net_from_dp, atol=2e-4)
    return dict(
        count=n,
        rows=records,
        total_shoe_wrench=wrench.tolist(),
        vertical_force_from_momentum_N=net_from_dp,
        native_first_joint_velocities=after.joint_qd.numpy().tolist(),
        law="native Featherstone per-row materials",
    )


def run_newton():
    import newton
    import warp as wp

    builder = newton.ModelBuilder(gravity=(0, 0, -2), up_axis=newton.Axis.Z)
    cfg = builder.default_shape_cfg
    cfg.density = 0
    cfg.ke = 4096.0
    cfg.kd = 8.0
    cfg.mu = 0.35
    cfg.kf = 1000.0
    cfg.gap = 0
    cfg.margin = 0
    builder.rigid_gap = 0
    builder.add_urdf(
        str(ASSET),
        floating=False,
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
        joint_damping_mode="passive",
        up_axis=newton.Axis.Z,
    )
    names = [x.rsplit("/", 1)[-1] for x in builder.joint_label]
    joints = [names.index(x) for x in ("lift", "twist")]
    iq = [builder.joint_q_start[x] for x in joints]
    iv = [builder.joint_qd_start[x] for x in joints]
    for dof in iv:
        for key in (
            "joint_armature",
            "joint_damping",
            "joint_target_ke",
            "joint_target_kd",
            "joint_limit_ke",
            "joint_limit_kd",
        ):
            getattr(builder, key)[dof] = 0.0
    table = builder.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(*CENTER), wp.quat_identity()),
        hx=HALF[0],
        hy=HALF[1],
        hz=HALF[2],
        cfg=cfg,
        label="fixed_table",
    )
    model = builder.finalize(device="cpu")
    state, after, control = model.state(), model.state(), model.control()
    body = model.joint_child.numpy()[joints]
    types = model.shape_type.numpy()
    owners = model.shape_body.numpy()
    (shoe,) = np.flatnonzero((types == newton.GeoType.BOX) & (owners == body[1]))
    assert types[table] == newton.GeoType.BOX and owners[table] == -1
    shape_pose = model.shape_transform.numpy().astype(float)
    scales = model.shape_scale.numpy().astype(float)
    vertices = (
        shape_pose[shoe, :3]
        + (np.array(list(product((-1, 1), repeat=3))) * scales[shoe])
        @ quaternion_matrix(shape_pose[shoe, None, 3:])[0].T
    )
    solid = LinkSolid("sole", "shoe", 1, "box", vertices, 0.0)
    law = FixedTableContacts(
        [solid],
        center_m=shape_pose[table, :3],
        half_extents_m=scales[table],
        orientation_xyzw=shape_pose[table, 3:],
        material=LAW,
    )
    np.testing.assert_allclose(model.body_mass.numpy()[body], [0.5, 1.5])
    np.testing.assert_allclose(
        np.diag(model.body_inertia.numpy()[body[1]]), [0.00244140625, 0.00830078125, 0.009765625]
    )
    qa = state.joint_q.numpy()
    qa[iq] = Q0
    state.joint_q.assign(qa)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    native = native_patch(newton, wp, model, state, int(shoe), table, int(body[1]), control, iv[0])
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    rows = []
    local_com = model.body_com.numpy()[body].astype(float)
    for k in range(STEPS):
        q = state.joint_q.numpy()[iq].astype(float)
        v = state.joint_qd.numpy()[iv].astype(float)
        pose = state.body_q.numpy()[body].astype(float)
        R = quaternion_matrix(pose[:, 3:])
        com = pose[:, :3] + np.einsum("nij,nj->ni", R, local_com)
        load = law.evaluate(pose[:, :3], R, com, state.body_qd.numpy()[body].astype(float))
        Q = np.array([load.link_wrenches[:, 2].sum(), load.link_wrenches[1, 5]])
        state.clear_forces()
        w = np.zeros_like(state.body_f.numpy())
        w[body] = load.link_wrenches
        state.body_f.assign(w)
        cmd = np.zeros_like(control.joint_f.numpy())
        cmd[iv] = motor(k)
        control.joint_f.assign(cmd)
        solver.step(state, after, control, None, H)
        state, after = after, state
        rows.append(row(k, q, v, state.joint_q.numpy()[iq], state.joint_qd.numpy()[iv], load, Q))
    return rows, native


def check(rows):
    for r in rows[:HOLD]:
        np.testing.assert_allclose(r["joint_load"], [4.0, 0], atol=3e-4)
        np.testing.assert_allclose(r["q"], Q0, atol=3e-6)
        assert r["contacts"] and all(p["patch_dimension"] == 2 for p in r["contacts"])
    assert rows[HOLD]["v"][1] > 1e-4 and rows[SPIN - 1]["robot_on_table_wrench"][5] > 0
    assert rows[SPIN]["v"][0] > 0 and not rows[-1]["contacts"]
    first = rows[0]["contacts"]
    power = max(
        abs(np.dot(r["joint_load"], r["pre_v"]) - sum(p["link_power_W"] for p in r["contacts"]))
        for r in rows
    )
    assert power < 1e-6 and all(p["table_power_W"] == 0 for r in rows for p in r["contacts"])
    assert sum(w["weight"] for w in first) > 0.999999
    return dict(
        steps=len(rows),
        urdf_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(),
        first_area_m2=first[0]["patch_area_m2"],
        first_points=len(first),
        first_table_wrench=rows[0]["robot_on_table_wrench"],
        prelift_table_moment_z=rows[SPIN - 1]["robot_on_table_wrench"][5],
        last_loaded_s=max(r["t"] for r in rows if r["contacts"]),
        max_power_error_W=power,
        final_q=rows[-1]["q"],
        final_velocity=rows[-1]["v"],
    )


def impulse_audit(rows):
    linear = angular = pair_moment = 0.0
    for r in rows:
        combined = np.zeros(3)
        for c in r["contacts"]:
            f = np.array(c["force_on_link_N"])
            other = np.array(c["force_on_table_N"])
            np.testing.assert_allclose(f + other, 0, atol=1e-12)
            combined += np.array(c["torque_on_link_Nm"]) + np.cross(c["link_com_world_m"], f)
            combined += np.array(c["torque_on_table_Nm"]) + np.cross(
                c["table_center_world_m"], other
            )
        pair_moment = max(pair_moment, float(np.linalg.norm(combined)))
        # The prismatic carriage carries both masses. The hinge rotates the shoe about its COM.
        dv = np.array(r["v"]) - r["pre_v"]
        p = 2 * dv[0] - H * (r["motor"][0] - 4 - r["robot_on_table_wrench"][2])
        L = 0.009765625 * dv[1] - H * (
            r["motor"][1] + sum(c["torque_on_link_Nm"][2] for c in r["contacts"])
        )
        linear = max(linear, abs(p))
        angular = max(angular, abs(L))
    assert linear < 4e-8 and angular < 2e-10 and pair_moment < 1e-12
    return dict(
        max_slide_impulse_error_Ns=linear,
        max_hinge_impulse_error_Nms=angular,
        max_pair_world_moment_error_Nm=pair_moment,
        held_table_impulse_Ns=(
            H * np.array([r["robot_on_table_wrench"][:3] for r in rows[:HOLD]]).sum(axis=0)
        ).tolist(),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--engine", choices=("phymeo", "newton"))
    mode.add_argument(
        "--saved", type=Path, help="audit the recorded joint impulses without stepping"
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.saved:
        data = json.loads(args.saved.read_text(encoding="utf8"))
        result = impulse_audit(data["trace"])
        with args.out.open("x", encoding="utf8") as stream:
            json.dump(result, stream, allow_nan=False)
        print(json.dumps(result, allow_nan=False))
    else:
        rows, observation = run_phy() if args.engine == "phymeo" else run_newton()
        with args.out.open("x", encoding="utf8") as f:
            json.dump(dict(trace=rows, native_initial=observation), f, allow_nan=False)
        print(json.dumps(dict(matched=check(rows), native_initial=observation), allow_nan=False))
