"""Hold and unload an independently imported R-P arm on a tilted finite table."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from phymeo.builders.urdf import load_urdf
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotInputs, RobotState
from phymeo.core.scene import SceneInputs, SceneState
from phymeo.dynamics.rotation import matrix_quaternion, quaternion_matrix
from phymeo.solvers.fixed_table_contact import FixedTableContacts
from phymeo.solvers.robot import make_robot_runtime
from phymeo.solvers.sphere_link_contact import LinkSolid

ASSET = Path(__file__).with_name("arm.urdf")
N = np.array([0.36, 0.48, 0.8])
U = np.array([0.8, -0.6, 0.0])
V = np.cross(N, U)
ROT = np.column_stack([U, V, N])
SPHERE = np.array([0.8125, -0.375, 0.15625])
HALF = np.array([0.125, 0.125, 1 / 32])
CENTER = SPHERE - N * (1 / 32 - 1 / 1024 + HALF[2]) - U / 16
LAW = RobotContactMaterial(2048.0, 4.0, 0.2, 0.04)
H = 1 / 1024
STEPS, HOLD = 160, 32
MOTOR = -np.array([0.45, 0.96])


def measure(k, q, v, p, state, load, effort):
    force = load.table_wrench
    return dict(
        t=(k + 1) * H,
        q=q.tolist(),
        v=v.tolist(),
        tool_position_m=p.tolist(),
        pre_q=state[0].tolist(),
        pre_v=state[1].tolist(),
        motor=effort.tolist(),
        table_joint_load=state[2].tolist(),
        robot_on_table_wrench=force.tolist(),
        link_table_contacts=load.contacts,
    )


def phy():
    asset = load_urdf(ASSET)
    rows = []
    scene = SceneState({"robot": RobotState(np.zeros(2), np.zeros(2))})
    contacts = FixedTableContacts.from_asset(
        asset,
        center_m=CENTER,
        half_extents_m=HALF,
        orientation_xyzw=matrix_quaternion(ROT),
        material=LAW,
    )
    from phymeo.dynamics.robot_kinematics import forward_kinematics

    with make_robot_runtime(asset.model) as rt:
        for k in range(STEPS):
            before = scene.domains["robot"]
            load, Q = contacts.evaluate_robot(asset.model, before)
            motor = MOTOR if k < HOLD else np.zeros(2)
            actual = rt.step(
                scene,
                SceneInputs({"robot": RobotInputs(motor + Q, np.zeros((2, 3)), np.zeros((2, 3)))}),
                H,
            )
            assert actual.report.accepted
            scene = actual.state
            after = scene.domains["robot"]
            com = forward_kinematics(asset.model, after.q).com_position[1]
            rows.append(measure(k, after.q, after.v, com, (before.q, before.v, Q), load, motor))
    return rows


def newton_cpu():
    import newton
    import warp as wp

    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
    b.default_shape_cfg.density = 0
    b.add_urdf(
        str(ASSET),
        floating=False,
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
        joint_damping_mode="passive",
        up_axis=newton.Axis.Z,
    )
    labels = [str(name).rsplit("/", 1)[-1] for name in b.joint_label]
    joints = [labels.index(n) for n in ("yaw", "slide")]
    iq = [b.joint_q_start[j] for j in joints]
    iv = [b.joint_qd_start[j] for j in joints]
    for j in iv:
        for name in (
            "joint_armature",
            "joint_damping",
            "joint_target_ke",
            "joint_target_kd",
            "joint_limit_ke",
            "joint_limit_kd",
        ):
            getattr(b, name)[j] = 0.0
    geo = b.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(*CENTER), wp.quat(*matrix_quaternion(ROT))),
        hx=HALF[0],
        hy=HALF[1],
        hz=HALF[2],
        cfg=newton.ModelBuilder.ShapeConfig(density=0, gap=0, margin=0),
        label="calibrated_table",
    )
    model = b.finalize(device="cpu")
    state, after, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    native = model.joint_child.numpy()[joints]
    masses = model.body_mass.numpy()[native]
    np.testing.assert_allclose(masses, [1.0, 0.5])
    kinds = model.shape_type.numpy()
    owners = model.shape_body.numpy()
    sx = model.shape_transform.numpy().astype(float)
    scale = model.shape_scale.numpy().astype(float)
    (marker,) = np.flatnonzero((kinds == newton.GeoType.SPHERE) & (owners == native[1]))
    np.testing.assert_allclose(scale[marker, 0], 1 / 32)
    assert kinds[geo] == newton.GeoType.BOX and owners[geo] == -1
    link = LinkSolid("marker", "tool", 1, "sphere", [sx[marker, :3]], scale[marker, 0])
    table = FixedTableContacts(
        [link],
        center_m=sx[geo, :3],
        half_extents_m=scale[geo],
        orientation_xyzw=sx[geo, 3:],
        material=LAW,
    )
    local_com = model.body_com.numpy()[native].astype(float)
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    rows = []
    for k in range(STEPS):
        q, v = state.joint_q.numpy()[iq].astype(float), state.joint_qd.numpy()[iv].astype(float)
        pose = state.body_q.numpy()[native].astype(float)
        R = quaternion_matrix(pose[:, 3:])
        com = pose[:, :3] + np.einsum("nij,nj->ni", R, local_com)
        load = table.evaluate(pose[:, :3], R, com, state.body_qd.numpy()[native].astype(float))
        motor = MOTOR if k < HOLD else np.zeros(2)
        # Read yaw/slide virtual work at the real structural material point.
        # Only the body wrenches, not Q, are applied along with the two motor efforts.
        Q = np.zeros(2)
        for hit in load.contacts:
            p = np.array(hit["point_world_m"])
            F = np.array(hit["force_on_link_N"])
            Q[0] += F @ np.cross([0.0, 0.0, 1.0], p - np.array([0.25, -0.5, 0.125]))
            Q[1] += F @ np.array([-np.sin(q[0]), np.cos(q[0]), 0.0])
        state.clear_forces()
        w = np.zeros_like(state.body_f.numpy())
        w[native] = load.link_wrenches
        state.body_f.assign(w)
        command = np.zeros_like(control.joint_f.numpy())
        command[iv] = motor
        control.joint_f.assign(command)
        solver.step(state, after, control, None, H)
        state, after = after, state
        end_pose = state.body_q.numpy()[native[1]].astype(float)
        pos = end_pose[:3] + quaternion_matrix(end_pose[None, 3:])[0] @ local_com[1]
        rows.append(
            measure(
                k,
                state.joint_q.numpy()[iq],
                state.joint_qd.numpy()[iv],
                pos,
                (q, v, Q),
                load,
                motor,
            )
        )
    return rows


def check(rows):
    assert len(rows) == STEPS and all(r["link_table_contacts"] for r in rows[:HOLD])
    for r in rows[:HOLD]:
        np.testing.assert_allclose(
            r["robot_on_table_wrench"], [*(-2 * N), 0.06, 0.08, -0.075], rtol=0, atol=4e-4
        )
        np.testing.assert_allclose(r["table_joint_load"], -MOTOR, rtol=0, atol=4e-4)
        np.testing.assert_allclose(r["tool_position_m"], SPHERE, rtol=0, atol=1e-6)
        np.testing.assert_allclose(r["q"], [0, 0], atol=2e-6)
    assert rows[HOLD]["v"][0] < -1e-5 and rows[HOLD]["v"][1] > 1e-4
    active = [r for r in rows if any(h["normal_load_N"] > 0 for h in r["link_table_contacts"])]
    assert len(active) > HOLD and not rows[-1]["link_table_contacts"]
    work = sum(sum(h["link_power_W"] for h in r["link_table_contacts"]) * H for r in rows)
    assert work > 0 and all(
        h["table_power_W"] == 0 for r in active for h in r["link_table_contacts"]
    )
    residual = max(
        abs(
            np.dot(r["table_joint_load"], r["pre_v"])
            - sum(h["link_power_W"] for h in r["link_table_contacts"])
        )
        for r in rows
    )
    assert residual < 1e-6
    return dict(
        urdf_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(),
        steps=len(rows),
        hold_steps=HOLD,
        hold_reaction_impulse_Ns=(
            H * np.array([r["robot_on_table_wrench"][:3] for r in rows[:HOLD]]).sum(axis=0)
        ).tolist(),
        first_table_wrench=rows[0]["robot_on_table_wrench"],
        release_first_velocity=rows[HOLD]["v"],
        last_loaded_time_s=active[-1]["t"],
        link_work_J=work,
        table_work_J=0.0,
        max_virtual_power_error_W=residual,
        final_q=rows[-1]["q"],
        final_velocity=rows[-1]["v"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=("phymeo", "newton"), required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = phy() if args.engine == "phymeo" else newton_cpu()
    with args.out.open("x", encoding="utf8") as f:
        json.dump(rows, f, allow_nan=False)
    print(json.dumps(check(rows), allow_nan=False))
