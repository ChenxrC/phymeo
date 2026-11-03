"""Newton 1.7 FREE-root FK and native COM Jacobian, with a shifted tool-point readout."""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np


def _model(newton, wp, device, branched):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    root = builder.add_link(
        label="root",
        mass=2.0,
        com=(0.125, 0, 0),
        inertia=np.diag([0.25, 0.375, 0.5]),
        lock_inertia=True,
    )
    joints = [builder.add_joint_free(root)]
    if branched:
        a, b, c = [
            builder.add_link(label=name, mass=1.0, com=com, inertia=np.eye(3), lock_inertia=True)
            for name, com in (
                ("A", (0.125, 0, 0)),
                ("B", (0.0625, 0.0625, 0)),
                ("C", (0, 0.03125, 0)),
            )
        ]
        for add, parent, child, axis, offset in (
            (builder.add_joint_revolute, root, a, (0, 0, 1), (0.5, 0, 0)),
            (builder.add_joint_prismatic, a, b, (1, 0, 0), (0.25, 0, 0)),
            (builder.add_joint_prismatic, root, c, (0, 1, 0), (0, 0, 0.25)),
        ):
            joints.append(
                add(parent, child, axis=axis, parent_xform=wp.transform(offset, wp.quat_identity()))
            )
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def _observe(newton, wp, model, q, v, body, local_point):
    state = model.state()
    state.joint_q.assign(q)
    state.joint_qd.assign(v)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    jac = newton.eval_jacobian(model, state)
    native = jac.numpy()[0].astype(float).reshape(model.body_count, 6, len(v))
    frames, twists = state.body_q.numpy(), state.body_qd.numpy()
    np.testing.assert_allclose(native @ v, twists, rtol=0, atol=2e-5)
    tf = wp.transform(wp.vec3(*frames[body, :3]), wp.quat(*frames[body, 3:]))
    center = np.asarray(wp.transform_point(tf, wp.vec3(*model.body_com.numpy()[body])), dtype=float)
    point = np.asarray(wp.transform_point(tf, wp.vec3(*local_point)), dtype=float)
    lin, ang = native[body, :3], native[body, 3:]
    tool_jac = lin + np.cross(ang.T, point - center).T
    point_v = twists[body, :3] + np.cross(twists[body, 3:], point - center)
    np.testing.assert_allclose(tool_jac @ v, point_v, rtol=0, atol=2e-5)
    return dict(
        device=str(jac.device),
        body_origins=frames[:, :3].tolist(),
        com_twists=twists.tolist(),
        com_m=center.tolist(),
        point_m=point.tolist(),
        point_velocity=point_v.tolist(),
        point_linear_j=tool_jac.tolist(),
        native_angular_j=ang.tolist(),
    )


def _run(newton, wp, device):
    model = _model(newton, wp, device, False)
    q, v = [0, 0, 0, 0, 0, 0, 1], np.array([0, 0.25, 0, 0, 0, 2])
    zero = _observe(newton, wp, model, q, v, 0, (0, 0, 0))
    np.testing.assert_allclose(zero["com_twists"], [v], rtol=0, atol=2e-5)
    np.testing.assert_allclose(zero["com_m"], [0.125, 0, 0], rtol=0, atol=2e-5)
    np.testing.assert_allclose(zero["point_velocity"], [0, 0, 0], rtol=0, atol=2e-5)
    q[3:] = [0.5, 0.5, 0.5, 0.5]
    rotated = _observe(
        newton, wp, model, q, np.array([0.25, -0.375, 0.125, 1, -0.5, 0.75]), 0, (0.25, 0, 0)
    )
    np.testing.assert_allclose(rotated["com_m"], [0, 0.125, 0], rtol=0, atol=2e-5)
    np.testing.assert_allclose(rotated["point_m"], [0, 0.25, 0], rtol=0, atol=2e-5)
    np.testing.assert_allclose(
        rotated["point_velocity"], [0.15625, -0.375, 0.25], rtol=0, atol=2e-5
    )

    model = _model(newton, wp, device, True)
    v = np.array([0, 0.25, 0, 0, 0, 2, 0.5, -0.25, 0.125])
    branch = _observe(
        newton,
        wp,
        model,
        [0.25, -0.5, 0.125, 0, 0, 0, 1, np.pi / 2, 0.125, 0.25],
        v,
        2,
        (0, 0.125, 0.0625),
    )
    expected = [
        [1, 0, 0, 0, 0.0625, -0.375, -0.375, 0, 0],
        [0, 1, 0, -0.0625, 0, 0.25, -0.125, 1, 0],
        [0, 0, 1, 0.375, -0.25, 0, 0, 0, 0],
    ]
    lin, ang = np.array(branch["point_linear_j"]), np.array(branch["native_angular_j"])
    np.testing.assert_allclose(lin, expected, rtol=0, atol=2e-5)
    want_ang = np.zeros((3, 9))
    want_ang[:, 3:6] = np.eye(3)
    want_ang[2, 6] = 1
    np.testing.assert_allclose(ang, want_ang, rtol=0, atol=2e-5)
    np.testing.assert_allclose(
        branch["body_origins"],
        [[0.25, -0.5, 0.125], [0.75, -0.5, 0.125], [0.75, -0.125, 0.125], [0.25, -0.25, 0.375]],
        rtol=0,
        atol=2e-5,
    )
    np.testing.assert_allclose(branch["point_m"], [0.625, -0.125, 0.1875], rtol=0, atol=2e-5)
    np.testing.assert_allclose(branch["point_velocity"], [-0.9375, 0.4375, 0], rtol=0, atol=2e-5)
    wrench = lin.T @ [2, -3, 4] + ang.T @ [0.5, -0.25, 0.75]
    np.testing.assert_allclose(
        wrench, [2, -3, 4, 2.1875, -1.125, -0.75, 0.375, -3, 0], rtol=0, atol=2e-5
    )
    np.testing.assert_allclose(wrench @ v, -1.3125, rtol=0, atol=2e-5)
    branch.update(generalized_wrench=wrench.tolist(), power_w=float(wrench @ v))
    return dict(origin=zero, rotated=rotated, branched=branch)


def main():
    import newton
    import warp as wp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda:0"))
    options = parser.parse_args()
    if version("newton") != "1.7.0":
        raise RuntimeError("These public twist fixtures target Newton 1.7.0")
    wp.config.kernel_cache_dir = str(Path(sys.prefix) / "warp-kernels")
    wp.init()
    with wp.ScopedDevice(options.device):
        records = _run(newton, wp, options.device)
    print(
        json.dumps(
            dict(newton=version("newton"), warp=version("warp-lang"), records=records),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
