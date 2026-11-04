"""Independently step a hinged finite rod on a carriage and an eccentric moving free box."""

import argparse
import json
from pathlib import Path

import numpy as np

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.rigid import RigidBodyInputs, RigidBodyModel, RigidBodyState
from phymeo.core.robot import RobotInputs, RobotModel, RobotState
from phymeo.core.scene import SceneInputs, SceneState
from phymeo.dynamics.robot_dynamics import robot_dynamics_terms
from phymeo.dynamics.rotation import quaternion_matrix
from phymeo.solvers.box_link_contact import BoxLinkContacts
from phymeo.solvers.rigid_semi_implicit import RigidSemiImplicitSolver
from phymeo.solvers.robot import make_robot_runtime
from phymeo.solvers.sphere_link_contact import LinkSolid

HALF = np.array([0.125, 0.0625, 0.25])
R = 1 / 32
ORIGIN = np.array([1 / 32, HALF[1] + R - 1 / 1024, 0.0])
V0 = np.array([0.02, 0.125, 0.0])
H = 1 / 1024
STEPS = 96
LAW = RobotContactMaterial(2048.0, 8.0, 0.5, 0.04)
BOX_I = (np.sum(HALF**2) - HALF**2) / 3
# A uniform 2 kg spherocylinder of barrel length 1 m and radius R.
M_CYL = 2 / (1 + 4 * R / 3)
M_CAP = 2 - M_CYL
I_SIDE = M_CYL * (3 * R**2 + 1) / 12 + M_CAP * (2 * R**2 / 5 + 0.25 + 3 * R / 8)
ROD_I = np.array([I_SIDE, I_SIDE, M_CYL * R**2 / 2 + M_CAP * 2 * R**2 / 5])


def registry(radius, ends):
    return BoxLinkContacts([LinkSolid("test_rod", "hinge", 1, "capsule", ends, radius)])


def evaluate(law, origins, rotations, twists, box, quaternion, speed, spin, half=HALF):
    return law.evaluate(
        origins,
        rotations,
        origins,
        twists,
        box,
        half,
        speed,
        spin,
        LAW,
        payload_rotation=quaternion_matrix(quaternion[None])[0],
    )


def record(k, load, q, v, box, quat, speed, omega):
    return dict(
        t=(k + 1) * H,
        q=q.tolist(),
        v=v.tolist(),
        box=box.tolist(),
        xyzw=quat.tolist(),
        velocity=speed.tolist(),
        omega=omega.tolist(),
        applied_payload_wrench=load.payload_wrench.tolist(),
        link_wrenches=load.link_wrenches.tolist(),
        structural_contacts=load.contacts,
    )


def phy():
    model = RobotModel(
        np.array([-1, 0]),
        ("prismatic", "revolute"),
        np.array([ORIGIN, [0, 0, 0]]),
        np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([1.0, 2.0]),
        np.zeros((2, 3)),
        np.array([[0.01] * 3, ROD_I]),
        np.zeros(3),
    )
    law = registry(R, [[0, 0, -0.5], [0, 0, 0.5]])
    arm = SceneState({"robot": RobotState(np.zeros(2), np.zeros(2))})
    rigid = RigidBodyModel(np.ones(1), BOX_I[None], np.zeros(3))
    body = RigidBodyState(
        np.zeros((1, 3)), np.array([[0.0, 0.0, 0.0, 1.0]]), V0[None], np.zeros((1, 3))
    )
    integrator = RigidSemiImplicitSolver()
    rows = []
    with make_robot_runtime(model) as rt:
        for k in range(STEPS):
            before = arm.domains["robot"]
            fk, J, A, _, _ = robot_dynamics_terms(model, before)
            load = evaluate(
                law,
                fk.position,
                fk.rotation,
                np.column_stack([J @ before.v, A @ before.v]),
                body.position[0],
                body.orientation[0],
                body.velocity[0],
                body.angular_velocity[0],
            )
            ext = RobotInputs(load.joint_load(J, A), np.zeros((2, 3)), np.zeros((2, 3)))
            moved = rt.step(arm, SceneInputs({"robot": ext}), H)
            after = integrator.step(
                rigid,
                body,
                RigidBodyInputs(load.payload_wrench[None, :3], load.payload_wrench[None, 3:]),
                H,
            )
            assert moved.report.accepted and after.report.accepted
            arm, body = moved.state, after.state
            robot = arm.domains["robot"]
            rows.append(
                record(
                    k,
                    load,
                    robot.q,
                    robot.v,
                    body.position[0],
                    body.orientation[0],
                    body.velocity[0],
                    body.angular_velocity[0],
                )
            )
    return rows


def newton_cpu():
    import newton
    import warp as wp

    builder = newton.ModelBuilder(gravity=(0, 0, 0), up_axis=newton.Axis.Z)

    def link(name, mass, moments):
        return builder.add_link(
            mass=mass, inertia=wp.mat33(*np.diag(moments).ravel()), lock_inertia=True, label=name
        )

    car = link("carriage", 1.0, [0.01] * 3)
    rod = link("hinge", 2.0, ROD_I)
    drive = dict(armature=0.0, damping=0.0, target_ke=0.0, target_kd=0.0)
    rail = builder.add_joint_prismatic(
        -1,
        car,
        axis=(0, 1, 0),
        parent_xform=wp.transform(wp.vec3(*ORIGIN), wp.quat_identity()),
        **drive,
    )
    hinge = builder.add_joint_revolute(car, rod, axis=(0, 0, 1), **drive)
    builder.add_articulation([rail, hinge])
    cfg = newton.ModelBuilder.ShapeConfig(density=0, gap=0, margin=0)
    cap = builder.add_shape_capsule(rod, radius=R, half_height=0.5, cfg=cfg)
    obj = builder.add_body(
        mass=1.0, inertia=wp.mat33(*np.diag(BOX_I).ravel()), lock_inertia=True, label="box"
    )
    box = builder.add_shape_box(obj, hx=HALF[0], hy=HALF[1], hz=HALF[2], cfg=cfg)
    model = builder.finalize(device="cpu")
    assert list(model.shape_type.numpy()[[cap, box]]) == [
        newton.GeoType.CAPSULE,
        newton.GeoType.BOX,
    ]
    scale = model.shape_scale.numpy().astype(float)
    radius, axial = scale[cap, :2]
    np.testing.assert_allclose([radius, axial], [R, 0.5])
    np.testing.assert_allclose(
        model.body_inertia.numpy()[[rod, obj]], [np.diag(ROD_I), np.diag(BOX_I)], rtol=2e-6
    )
    law = registry(radius, [[0, 0, -axial], [0, 0, axial]])
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    state, after, control = model.state(), model.state(), model.control()
    v = state.joint_qd.numpy()
    v[2:5] = V0
    state.joint_qd.assign(v)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    rows = []
    for k in range(STEPS):
        pose = state.body_q.numpy().astype(float)
        tw = state.body_qd.numpy().astype(float)
        load = evaluate(
            law,
            pose[[car, rod], :3],
            quaternion_matrix(pose[[car, rod], 3:]),
            tw[[car, rod]],
            pose[obj, :3],
            pose[obj, 3:],
            tw[obj, :3],
            tw[obj, 3:],
            scale[box],
        )
        state.clear_forces()
        w = np.zeros_like(tw)
        w[[car, rod]] = load.link_wrenches
        w[obj] = load.payload_wrench
        state.body_f.assign(w.astype(np.float32))
        solver.step(state, after, control, None, H)
        state, after = after, state
        pose = state.body_q.numpy()[obj]
        tw = state.body_qd.numpy()[obj]
        rows.append(
            record(
                k,
                load,
                state.joint_q.numpy()[:2],
                state.joint_qd.numpy()[:2],
                pose[:3],
                pose[3:],
                tw[:3],
                tw[3:],
            )
        )
    return rows


def check(rows, epsilon):
    first = rows[0]
    assert len(first["structural_contacts"]) == 2 and all(
        row["weight"] == 0.5 for row in first["structural_contacts"]
    )
    np.testing.assert_allclose(first["applied_payload_wrench"][:3], [-0.75, -3, 0], atol=epsilon)
    y = HALF[1] - 0.5 / 1024
    Tz = -3 * ORIGIN[0] + 0.75 * y
    np.testing.assert_allclose(first["omega"][2], H * Tz / BOX_I[2], atol=epsilon)
    np.testing.assert_allclose(first["v"], [H, H * 0.75 * (ORIGIN[1] - y) / ROD_I[2]], atol=epsilon)
    np.testing.assert_allclose(
        first["velocity"], V0 + H * np.array([-0.75, -3.0, 0.0]), atol=epsilon
    )
    moment = max(abs(3 * r["v"][0] + r["velocity"][1] - V0[1]) for r in rows)
    assert moment < max(epsilon, 1e-12)
    active = [r for r in rows if r["structural_contacts"]]
    assert len(active) > 1 and not rows[-1]["structural_contacts"]
    assert abs(rows[-1]["q"][1]) > 1e-4 and abs(rows[-1]["omega"][2]) > 1e-3
    return dict(
        steps=len(rows),
        active_steps=len(active),
        max_y_momentum_error_Ns=moment,
        last_contact_s=active[-1]["t"],
        last_velocity=rows[-1]["velocity"],
        last_spin=rows[-1]["omega"],
        final_rail_and_hinge=rows[-1]["q"],
        first_box_wrench=first["applied_payload_wrench"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=("phymeo", "newton"), required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = phy() if args.engine == "phymeo" else newton_cpu()
    with args.out.open("x", encoding="utf8") as f:
        json.dump(rows, f, allow_nan=False)
    print(json.dumps(check(rows, 2e-9 if args.engine == "phymeo" else 2e-6), allow_nan=False))
