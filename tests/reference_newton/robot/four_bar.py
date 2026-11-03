"""Newton 1.7 XPBD four-joint ring, including its isolated inbound closure wrench."""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
from four_bar_oracle import case, mechanics, physical_work, points


def build(newton, wp, row, device):
    u, d, theta, omega = row["frame"], row["origin"], row["theta"], row["speed"]
    tilt = wp.quat(0, np.sqrt((1 - u[0, 0]) / 2), 0, np.sqrt((1 + u[0, 0]) / 2))
    hinge = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), wp.float32(theta))
    rod_quat = wp.mul(tilt, hinge)
    builder = newton.ModelBuilder(gravity=tuple(u @ row["gravity"]))
    moving_left = points(row, theta)[0]
    ids = []
    for label, position, rot, com, mass, moment in (
        ("left", d, rod_quat, 0.25, 1, 0.0625),
        ("coupler", moving_left, tilt, 0.5, 2, 0.125),
        ("right", d + u[:, 0], rod_quat, 0.25, 1, 0.0625),
    ):
        ids.append(
            builder.add_link(
                label=label,
                xform=wp.transform(wp.vec3(*position), rot),
                mass=mass,
                com=(com, 0, 0),
                inertia=moment * np.eye(3),
                lock_inertia=True,
            )
        )
    a, c, b = ids
    identity = wp.quat_identity()

    def xf(x):
        return wp.transform(wp.vec3(x, 0, 0), identity)

    def rj(parent, child, parent_xform, child_xform):
        return builder.add_joint_revolute(
            parent,
            child,
            axis=(0, 0, 1),
            parent_xform=parent_xform,
            child_xform=child_xform,
            target_ke=0.0,
            target_kd=0.0,
            damping=0.0,
            armature=0.0,
        )

    drive = rj(-1, a, wp.transform(wp.vec3(*d), tilt), xf(0))
    builder.add_articulation([drive])
    right = rj(-1, b, wp.transform(wp.vec3(*(d + u[:, 0])), tilt), xf(0))
    close = builder.add_joint_ball(b, c, parent_xform=xf(0.5), child_xform=xf(1))
    builder.add_articulation([right, close])
    # Reverse the left/coupler hinge so the coupler's ONLY inbound joint
    # is the closing ball. Its BODY_PARENT_F then isolates that real closure.
    rj(c, a, xf(0), xf(0.5))
    dof = builder.joint_qd_start[drive]
    model = builder.finalize(device=device)
    np.testing.assert_array_equal(np.flatnonzero(model.joint_child.numpy() == c), [close])
    state = model.state()
    tangent = u @ np.array([-np.sin(theta), np.cos(theta), 0])
    spin = u[:, 2] * omega
    state.body_qd.assign(
        np.array(
            [
                np.r_[0.25 * omega * tangent, spin],
                np.r_[0.5 * omega * tangent, np.zeros(3)],
                np.r_[0.25 * omega * tangent, spin],
            ]
        )
    )
    return model, state, dof


def observe(wp, state, row, wrench):
    u, d = row["frame"], row["origin"]
    poses, twists = state.body_q.numpy().astype(float), state.body_qd.numpy().astype(float)
    x = np.array([np.asarray(wp.quat_rotate(wp.quat(*q[3:]), wp.vec3(1, 0, 0))) for q in poses])
    origins = poses[:, :3]
    left = origins[0] + 0.5 * x[0]
    right = origins[2] + 0.5 * x[2]
    tip = origins[1] + x[1]
    gaps = [
        np.linalg.norm(origins[0] - d),
        np.linalg.norm(origins[2] - (d + u[:, 0])),
        np.linalg.norm(origins[1] - left),
        np.linalg.norm(tip - right),
    ]
    vtip = twists[1, :3] + np.cross(twists[1, 3:], 0.5 * x[1])
    vright = twists[2, :3] + np.cross(twists[2, 3:], 0.25 * x[2])
    angle = np.arctan2(u[:, 1] @ x[0], u[:, 0] @ x[0])
    kinetic = 0.5 * (
        np.array([1.0, 2, 1]) @ np.sum(twists[:, :3] ** 2, axis=1)
        + np.array([0.0625, 0.125, 0.0625]) @ np.sum(twists[:, 3:] ** 2, axis=1)
    )
    return dict(
        theta=angle,
        omega=float(u[:, 2] @ twists[0, 3:]),
        gaps=gaps,
        slip=float(np.linalg.norm(vtip - vright)),
        force=wrench[:3].tolist(),
        moment=wrench[3:].tolist(),
        closure_power=float(wrench[:3] @ (vtip - vright)),
        first_power=float(wrench[:3] @ vtip),
        second_power=float(-wrench[:3] @ vright),
        kinetic=float(kinetic),
        moving_points=[left.tolist(), right.tolist()],
    )


def run(newton, wp, row, device, relaxation):
    model, state, dof = build(newton, wp, row, device)
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=32,
        joint_linear_relaxation=relaxation[0],
        joint_angular_relaxation=relaxation[1],
        angular_damping=0.0,
    )
    observables = solver.observables(kinds={newton.solvers.SolverBase.ObservableKind.BODY_PARENT_F})
    target, control = model.state(), model.control()
    effort = np.zeros(model.joint_dof_count)
    effort[dof] = row["drive"]
    control.joint_f.assign(effort)
    loads = np.column_stack(
        (row["forces"] @ row["frame"].T, row["moments"][:, None] * row["frame"][:, 2])
    )
    theta, speed, h = row["theta"], row["speed"], row["h"]
    rows = []
    for _ in range(row["steps"]):
        a, force, power = mechanics(row, theta, speed)
        speed += h * a
        theta += h * speed
        state.clear_forces()
        state.body_f.assign(loads)
        solver.step(state, target, control, None, h, observables=observables)
        state, target = target, state
        item = observe(wp, state, row, observables.body_parent_f.numpy()[1].astype(float))
        item.update(
            analytic_force=force.tolist(),
            analytic_power=power,
            analytic_theta=theta,
            analytic_omega=speed,
        )
        rows.append(item)
    mean_force_error = np.mean([np.array(r["force"]) - r["analytic_force"] for r in rows], axis=0)
    return dict(
        device=str(model.device),
        steps=len(rows),
        rows=rows,
        max_gap=max(max(r["gaps"]) for r in rows),
        max_slip=max(r["slip"] for r in rows),
        theta_error=max(abs(r["theta"] - r["analytic_theta"]) for r in rows),
        omega_error=max(abs(r["omega"] - r["analytic_omega"]) for r in rows),
        mean_force_error=mean_force_error.tolist(),
        constraint_work=h * sum(r["closure_power"] for r in rows),
        drive_external_work=physical_work(row, rows[-1]["theta"]),
        kinetic_change=rows[-1]["kinetic"] - 0.375 * row["speed"] ** 2,
    )


def main():
    import newton
    import warp as wp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--relaxation", choices=("paired", "default"), default="paired")
    options = parser.parse_args()
    relaxation = (0.5, 0.5) if options.relaxation == "paired" else (0.7, 0.4)
    if version("newton") != "1.7.0":
        raise RuntimeError("This native loop comparison targets Newton 1.7.0")
    wp.config.kernel_cache_dir = str(Path(sys.prefix) / "warp-kernels")
    wp.init()
    with wp.ScopedDevice(options.device):
        cases = {
            name: run(newton, wp, case(tilt), options.device, relaxation)
            for name, tilt in (("level", False), ("tilted", True))
        }
    print(
        json.dumps(
            dict(
                newton=version("newton"),
                warp=version("warp-lang"),
                relaxation=relaxation,
                cases=cases,
            ),
            allow_nan=False,
        )
    )
    for name, r in cases.items():
        for metric, limit in (("max_gap", 4e-4), ("theta_error", 3e-3), ("omega_error", 1.5e-2)):
            assert r[metric] <= limit, (name, metric, r[metric], limit)
        assert np.linalg.norm(r["mean_force_error"]) <= 0.08, (name, r["mean_force_error"])
        assert abs(r["constraint_work"]) <= 4e-4, (name, r["constraint_work"])
        assert abs(r["kinetic_change"] - r["drive_external_work"]) <= 3e-3, (name, "energy/work")


if __name__ == "__main__":
    main()
