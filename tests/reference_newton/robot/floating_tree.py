"""Newton 1.7 Featherstone free-tree dynamics from real end-of-step public body states."""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np


def build(newton, wp, row, spec, device):
    builder = newton.ModelBuilder(gravity=tuple(row["gravity"]))
    root = builder.add_link(
        mass=2.0, com=spec["root_com"], inertia=0.5 * np.eye(3), lock_inertia=True, label="root"
    )
    child = builder.add_link(
        mass=1.0, com=spec["com"], inertia=0.25 * np.eye(3), lock_inertia=True, label="child"
    )
    free = builder.add_joint_free(root)
    add = builder.add_joint_prismatic if spec["kind"] == "prismatic" else builder.add_joint_revolute
    joint = add(
        root,
        child,
        axis=spec["axis"],
        parent_xform=wp.transform(spec["anchor"], wp.quat_identity()),
        armature=0.0,
        damping=0.0,
        target_ke=0.0,
        target_kd=0.0,
        limit_ke=0.0,
        limit_kd=0.0,
    )
    builder.add_articulation([free, joint])
    return builder.finalize(device=device)


def read(newton, wp, model, state):
    q, v = state.joint_q.numpy().astype(float), state.joint_qd.numpy().astype(float)
    poses, twists = state.body_q.numpy().astype(float), state.body_qd.numpy().astype(float)
    centers = []
    for pose, local_com in zip(poses, model.body_com.numpy(), strict=True):
        frame = wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:]))
        centers.append(np.asarray(wp.transform_point(frame, wp.vec3(*local_com)), dtype=float))
    mass = model.body_mass.numpy()[:, None]
    momentum = twists[:, :3] * mass
    # Both fixtures have spherical COM inertia. The world attitude drops out.
    spin_momentum = model.body_inertia.numpy()[:, 0, 0, None] * twists[:, 3:]
    angular = np.cross(centers, momentum) + spin_momentum
    kinetic = (np.sum(momentum * twists[:, :3]) + np.sum(spin_momentum * twists[:, 3:])) / 2
    native_mass = newton.eval_mass_matrix(model, state).numpy()[0].astype(float)
    np.testing.assert_allclose(v @ native_mass @ v / 2, kinetic, rtol=0, atol=4e-6)
    return dict(
        q=q,
        v=v,
        poses=poses,
        twists=twists,
        com=np.array(centers),
        momentum=momentum.sum(axis=0),
        angular=angular.sum(axis=0),
        energy=float(kinetic),
        mass=native_mass,
    )


def observe(newton, wp, catalog, row, device):
    from newton.solvers import SolverFeatherstone

    model = build(newton, wp, row, catalog["models"][row["model"]], device)
    state, candidate, control = model.state(), model.state(), model.control()
    state.joint_q.assign(row["q"])
    state.joint_qd.assign(row["v"])
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    first = read(newton, wp, model, state)
    if row["model"] == "rail":
        np.testing.assert_allclose(
            first["mass"][np.ix_([0, 6], [0, 6])], [[3, 1], [1, 1]], rtol=0, atol=2e-5
        )
    solver = SolverFeatherstone(model, angular_damping=0.0)
    h, accel = row["h"], np.array(row["acceleration"])
    control.joint_f.assign([0] * 6 + [row["effort"]])
    rows = []
    for k in range(1, row["steps"] + 1):
        state.clear_forces()
        state.body_f.assign(np.column_stack([row["force"], row["torque"]]))
        solver.step(state, candidate, control, None, h)
        state, candidate = candidate, state
        observed = read(newton, wp, model, state)
        wanted_v = first["v"] + k * h * accel
        if k == 1:
            np.testing.assert_allclose((observed["v"] - first["v"]) / h, accel, rtol=0, atol=3e-4)
        np.testing.assert_allclose(observed["v"], wanted_v, rtol=0, atol=2e-5)
        scalar = first["q"][7] + k * h * first["v"][6] + h * h * k * (k + 1) / 2 * accel[6]
        np.testing.assert_allclose(observed["q"][7], scalar, rtol=0, atol=2e-5)
        root_c = first["com"][0] + k * h * first["v"][:3] + h * h * k * (k + 1) / 2 * accel[:3]
        np.testing.assert_allclose(observed["com"][0], root_c, rtol=0, atol=2e-5)
        if row["steps"] > 1:
            root_v, child_v = wanted_v[:3], wanted_v[:3] + np.array([wanted_v[6], 0, 0])
            np.testing.assert_allclose(
                observed["twists"][:, :3], [root_v, child_v], rtol=0, atol=2e-5
            )
            force = np.sum(row["force"], axis=0) + 3 * np.array(row["gravity"])
            np.testing.assert_allclose(
                observed["momentum"], first["momentum"] + k * h * force, rtol=0, atol=2e-5
            )
            np.testing.assert_allclose(
                observed["energy"], root_v @ root_v + child_v @ child_v / 2, rtol=0, atol=2e-5
            )
        elif row["id"] == "oblique-hinge":
            axis, radial, tangent = (
                np.array([0, 0.6, 0.8]),
                np.array([1, 0, 0]),
                np.array([0, 0.8, -0.6]),
            )
            theta = 2 * np.arctan(-2.5 * h * h / 2) + 5.5 * h * h
            child_center = root_c + 0.5 * (radial * np.cos(theta) + tangent * np.sin(theta))
            child_v = h * accel[:3] + 1.5 * h * (tangent * np.cos(theta) - radial * np.sin(theta))
            np.testing.assert_allclose(observed["com"][1], child_center, rtol=0, atol=2e-5)
            np.testing.assert_allclose(observed["twists"][1, :3], child_v, rtol=0, atol=2e-5)
            np.testing.assert_allclose(
                observed["twists"][:, 3:], [-2.5 * h * axis, 3 * h * axis], rtol=0, atol=2e-5
            )
            np.testing.assert_allclose(observed["momentum"], 0, rtol=0, atol=1e-6)
            np.testing.assert_allclose(observed["angular"], 0, rtol=0, atol=1e-6)
            np.testing.assert_allclose(
                observed["energy"], row["effort"] * accel[6] * h * h / 2, rtol=0, atol=5e-9
            )
        else:
            # Newton advances its FREE quaternion with normalized Euler.
            w = wanted_v[5]
            angle = 2 * np.arctan(h * w / 2)
            axis = np.array([np.cos(angle), np.sin(angle), 0])
            tangent = np.array([-np.sin(angle), np.cos(angle), 0])
            child_v = wanted_v[:3] + wanted_v[6] * axis + w * scalar * tangent
            expected_h = (
                np.cross(root_c, 2 * wanted_v[:3])
                + np.cross(root_c + scalar * axis, child_v)
                + [0, 0, 0.75 * w]
            )
            np.testing.assert_allclose(
                observed["momentum"], 2 * wanted_v[:3] + child_v, rtol=0, atol=2e-5
            )
            np.testing.assert_allclose(observed["angular"], expected_h, rtol=0, atol=2e-5)
            np.testing.assert_allclose(
                observed["energy"],
                wanted_v[:3] @ wanted_v[:3] + (child_v @ child_v + 0.75 * w * w) / 2,
                rtol=0,
                atol=2e-5,
            )
        rows.append(
            {
                name: (value.tolist() if isinstance(value, np.ndarray) else value)
                for name, value in observed.items()
                if name != "mass"
            }
        )
    return dict(
        id=row["id"], device=str(model.device), initial_mass=first["mass"].tolist(), rows=rows
    )


def main():
    import newton
    import warp as wp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    options = parser.parse_args()
    if version("newton") != "1.7.0":
        raise RuntimeError("These coupled dynamics cases target Newton 1.7.0")
    wp.config.kernel_cache_dir = str(Path(sys.prefix) / "warp-kernels")
    wp.init()
    catalog = json.loads(
        Path(__file__).with_name("floating_tree_cases.json").read_text(encoding="utf-8")
    )
    with wp.ScopedDevice(options.device):
        records = [observe(newton, wp, catalog, row, options.device) for row in catalog["cases"]]
    print(
        json.dumps(
            dict(newton=version("newton"), warp=version("warp-lang"), records=records),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
