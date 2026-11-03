"""Newton 1.7 XPBD CONTACT_F and SensorContact on a two-shape body resting or sliding."""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np


def build(newton, wp, device):
    from newton.sensors import SensorContact

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -1.0), up_axis=newton.Axis.Z)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.25, restitution=0.0)
    ground = builder.add_ground_plane(cfg=cfg, label="floor")
    body = builder.add_link(
        label="beam",
        xform=wp.transform(wp.vec3(0, 0, 0.125), wp.quat_identity()),
        mass=2.0,
        inertia=np.eye(3) * 0.125,
        lock_inertia=True,
    )
    free = builder.add_joint_free(body)
    builder.add_articulation([free])
    pads = [
        builder.add_shape_sphere(
            body,
            radius=0.125,
            xform=wp.transform(wp.vec3(x, 0, 0), wp.quat_identity()),
            cfg=cfg,
            label=name,
        )
        for x, name in ((-0.25, "left"), (0.25, "right"))
    ]
    far = builder.add_shape_sphere(
        -1,
        radius=0.1,
        xform=wp.transform(wp.vec3(5, 0, 5), wp.quat_identity()),
        cfg=cfg,
        label="far",
    )
    model = builder.finalize(device=device)
    shapes = [ground, *pads, far]
    a = SensorContact(
        model,
        sensing_bodies=[body],
        counterpart_shapes=[ground, far],
        request_contact_attributes=False,
    )
    b = SensorContact(
        model, sensing_shapes=shapes, counterpart_shapes=shapes, request_contact_attributes=False
    )
    c = SensorContact(
        model, sensing_shapes=[ground], counterpart_bodies=[body], request_contact_attributes=False
    )
    return model, body, pads, ground, far, (a, b, c)


def filtered(sensor, target, source, field):
    row = list(sensor.sensing_indices).index(target)
    col = list(sensor.counterpart_indices[row]).index(source)
    return getattr(sensor, field).numpy()[row, col].astype(float)


def read_points(wp, model, state, contacts, obs, body, pads, reference):
    n = int(contacts.rigid_contact_count.numpy()[0])
    side = [contacts.rigid_contact_shape0.numpy()[:n], contacts.rigid_contact_shape1.numpy()[:n]]
    local = [
        contacts.rigid_contact_point0.numpy()[:n] + contacts.rigid_contact_offset0.numpy()[:n],
        contacts.rigid_contact_point1.numpy()[:n] + contacts.rigid_contact_offset1.numpy()[:n],
    ]
    forces = obs.contact_f.numpy()[:n, :3].astype(float)
    normals = contacts.rigid_contact_normal.numpy()[:n].astype(float)
    pose = state.body_q.numpy()[body]
    tf = wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:]))
    rows = []
    force = np.zeros(3)
    moment = np.zeros(3)
    friction = np.zeros(3)
    about_com = np.zeros(3)
    for k, f in enumerate(forces):
        if side[0][k] in pads:
            slot = 0
        elif side[1][k] in pads:
            slot = 1
        else:
            continue
        signed = f if slot == 0 else -f
        effective = np.asarray(wp.transform_point(tf, wp.vec3(*local[slot][k])), dtype=float)
        normal = normals[k] / np.linalg.norm(normals[k])
        tangent = signed - normal * (signed @ normal)
        force += signed
        friction += tangent
        moment += np.cross(effective - reference, signed)
        about_com += np.cross(effective - pose[:3], signed)
        rows.append(
            dict(
                shapes=[int(side[0][k]), int(side[1][k])],
                pad_index=int(side[slot][k]),
                point_world_m=effective.tolist(),
                force_n=signed.tolist(),
            )
        )
    return force, friction, moment, about_com, rows


def window(newton, wp, device, sliding):
    model, body, pads, floor, far, sensors = build(newton, wp, device)
    beam, shapes, reverse = sensors
    pipe = newton.CollisionPipeline(model, rigid_contact_max=32, soft_contact_max=0)
    contacts = pipe.contacts()
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=16,
        rigid_contact_relaxation=0.75,
        rigid_contact_con_weighting=False,
        angular_damping=0.0,
    )
    obs = solver.observables(kinds=beam.solver_observable_kinds, contacts=contacts)
    state, target = model.state(), model.state()
    state.joint_qd.assign([0.5 if sliding else 0.0, 0, 0, 0, 0, 0])
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    h = 1 / 64
    reference = np.array([-0.25, 0, 0])
    initial = state.body_qd.numpy()[body].astype(float)
    rows = []
    physical_impulse = np.zeros(3)
    spin_impulse = np.zeros(3)
    for k in range(1, 17):
        state.clear_forces()
        pipe.collide(state, contacts)
        solver.step(state, target, None, contacts, h, observables=obs)
        state, target = target, state
        for sensor in sensors:
            sensor.update(state, contacts, observables=obs)
        native = beam.total_force.numpy()[0].astype(float)
        tangent = beam.total_force_friction.numpy()[0].astype(float)
        direct, friction, moment, spin, points = read_points(
            wp, model, state, contacts, obs, body, pads, reference
        )
        np.testing.assert_allclose(native, direct, rtol=0, atol=2e-5)
        np.testing.assert_allclose(tangent, friction, rtol=0, atol=2e-5)
        np.testing.assert_allclose(
            filtered(beam, body, floor, "force_matrix"), native, rtol=0, atol=2e-5
        )
        np.testing.assert_array_equal(filtered(beam, body, far, "force_matrix"), np.zeros(3))
        np.testing.assert_array_equal(
            shapes.total_force.numpy()[list(shapes.sensing_indices).index(far)], np.zeros(3)
        )
        from_pads = [filtered(shapes, p, floor, "force_matrix") for p in pads]
        for p, v in zip(pads, from_pads, strict=True):
            np.testing.assert_allclose(
                filtered(shapes, floor, p, "force_matrix"), -v, rtol=0, atol=2e-5
            )
        np.testing.assert_allclose(np.sum(from_pads, axis=0), native, rtol=0, atol=2e-5)
        np.testing.assert_allclose(
            filtered(reverse, floor, body, "force_matrix"), -native, rtol=0, atol=2e-5
        )
        physical_impulse += h * (native + [0, 0, -2])
        spin_impulse += h * spin
        twist = state.body_qd.numpy()[body].astype(float)
        rows.append(
            dict(
                k=k,
                world_force_n=native.tolist(),
                world_friction_n=tangent.tolist(),
                pad_forces_n=[f.tolist() for f in from_pads],
                native_weighted_midpoint_m=filtered(beam, body, floor, "position_matrix").tolist(),
                points=points,
                derived_torque_nm=moment.tolist(),
                twist=twist.tolist(),
            )
        )
    avg = np.mean([r["world_force_n"] for r in rows], axis=0)
    avg_moment = np.mean([r["derived_torque_nm"] for r in rows], axis=0)
    return dict(
        device=str(beam.total_force.device),
        sliding=sliding,
        h=h,
        steps=len(rows),
        reference_world_m=reference.tolist(),
        average_force_n=avg.tolist(),
        average_torque_nm=avg_moment.tolist(),
        delta_p_error=(physical_impulse - 2 * (twist[:3] - initial[:3])).tolist(),
        delta_h_error=(spin_impulse - 0.125 * (twist[3:] - initial[3:])).tolist(),
        rows=rows,
    )


def main():
    import newton
    import warp as wp

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    args = p.parse_args()
    if version("newton") != "1.7.0":
        raise RuntimeError("Targets Newton 1.7.0 SensorContact")
    wp.config.kernel_cache_dir = str(Path(sys.prefix) / "warp-kernels")
    wp.init()
    with wp.ScopedDevice(args.device):
        runs = [window(newton, wp, args.device, s) for s in (False, True)]
    print(
        json.dumps(
            dict(newton=version("newton"), warp=version("warp-lang"), windows=runs), allow_nan=False
        )
    )
    rest, slide = runs
    np.testing.assert_allclose(rest["average_force_n"], [0, 0, 2], rtol=0, atol=0.015)
    np.testing.assert_allclose(rest["average_torque_nm"], [0, -0.5, 0], rtol=0, atol=0.012)
    assert -0.55 <= slide["average_force_n"][0] < -0.05, slide["average_force_n"]
    assert abs(slide["average_force_n"][2] - 2) < 0.06
    for row in runs:
        assert np.linalg.norm(row["delta_p_error"]) < 8e-4, row["delta_p_error"]
        assert np.linalg.norm(row["delta_h_error"]) < 0.015, row["delta_h_error"]


if __name__ == "__main__":
    main()
