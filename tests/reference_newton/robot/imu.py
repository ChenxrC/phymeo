"""Newton 1.7 native site IMU on true simulation end states and measured acceleration."""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np


def build(newton, wp, device):
    from newton.sensors import SensorIMU

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -4.0))
    newton.solvers.SolverKamino.register_custom_attributes(builder)
    identity = wp.quat_identity()
    z90 = wp.quat(0, 0, np.sqrt(0.5), np.sqrt(0.5))
    x90 = wp.quat(np.sqrt(0.5), 0, 0, np.sqrt(0.5))
    xyz_cycle = wp.quat(0.5, 0.5, 0.5, 0.5)
    ids, joints = [], []
    for label, x in (("fall", 0.0), ("support", 1.0), ("drive", 2.0), ("coast", 3.0)):
        com = (0.0, 0, 0) if label == "support" else (0.125, 0, 0)
        tf = wp.transform(wp.vec3(x, 0, 0), identity)
        body = builder.add_link(
            label=label,
            xform=tf,
            mass=2.0,
            com=com,
            inertia=np.diag([0.25, 0.375, 0.5]),
            lock_inertia=True,
        )
        joint = (
            builder.add_joint_revolute(
                -1,
                body,
                axis=(0, 0, 1),
                parent_xform=tf,
                armature=0.0,
                damping=0.0,
                target_ke=0.0,
                target_kd=0.0,
            )
            if label == "support"
            else builder.add_joint_free(body)
        )
        builder.add_articulation([joint])
        ids.append(body)
        joints.append(joint)
    sites = []
    for label, body, p, rotation in (
        ("fall", ids[0], (0.375, 0, 0), z90),
        ("support", ids[1], (0.25, -0.125, 0), x90),
        ("drive", ids[2], (0.375, 0, 0), z90),
        ("drive_com", ids[2], (0.125, 0, 0), identity),
        ("coast", ids[3], (0.375, 0, 0), z90),
        ("world", -1, (1, 2, 3), xyz_cycle),
    ):
        sites.append(
            builder.add_site(body, label="imu_" + label, xform=wp.transform(wp.vec3(*p), rotation))
        )
    model = builder.finalize(device=device)
    sensor = SensorIMU(model, sites=sites, request_state_attributes=False)
    state = model.state()
    v = state.joint_qd.numpy()
    for joint, angular in ((joints[0], 0.0), (joints[2], 2.0), (joints[3], -2.0)):
        start = builder.joint_qd_start[joint]
        v[start : start + 6] = [0.25, -0.5, 0, 0, 0, angular]
    state.joint_qd.assign(v)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, sensor, state, ids


def run(newton, wp, device):
    model, imu, state, ids = build(newton, wp, device)
    solver = newton.solvers.SolverKamino(model)
    observables = solver.observables(kinds=imu.solver_observable_kinds)
    target = model.state()
    inputs = model.control()
    loads = np.zeros((4, 6))
    loads[ids[2]] = [2, 0, 0, 0, 0, 1.5]
    h = 1 / 64
    rows = []
    for k in range(1, 9):
        previous = state.body_qd.numpy().astype(float)
        state.clear_forces()
        state.body_f.assign(loads)
        solver.step(state, target, inputs, None, h, observables=observables)
        state, target = target, state
        imu.update(state, observables=observables)
        accelerometer, gyro = (
            imu.accelerometer.numpy().astype(float),
            imu.gyroscope.numpy().astype(float),
        )
        poses, twists = state.body_q.numpy(), state.body_qd.numpy()
        raw = observables.body_qdd.numpy().astype(float)
        np.testing.assert_allclose(raw, (twists - previous) / h, rtol=0, atol=3e-5)
        # The known constant force is world +X; project it using the actual end attitude.
        # Convective and tangential terms have exact site-axis components at this principal-axis spin.
        orientation = wp.quat(*poses[ids[2], 3:])
        world_x = np.asarray(wp.quat_rotate_inv(orientation, wp.vec3(1, 0, 0)), dtype=float)
        omega = 2 + 3 * k * h
        expected = [
            [0.0, 0, 0],
            [0, 4, 0],
            [0.75 + world_x[1], 0.25 * omega**2 - world_x[0], 0],
            world_x,
            [0, 1, 0],
            [0, 4, 0],
        ]
        expected_gyro = np.zeros((6, 3))
        expected_gyro[2:4, 2] = omega
        expected_gyro[4, 2] = -2.0
        rows.append(
            dict(
                k=k,
                time_s=k * h,
                accelerometer=accelerometer.tolist(),
                gyro=gyro.tolist(),
                body_q=poses.tolist(),
                body_qd=twists.tolist(),
                body_qdd=raw.tolist(),
                max_force_error=float(np.max(np.abs(accelerometer - expected))),
                max_gyro_error=float(np.max(np.abs(gyro - expected_gyro))),
            )
        )
    return dict(
        device=str(imu.accelerometer.device),
        source="SolverKamino.BODY_QDD",
        step_duration=h,
        rows=rows,
    )


def main():
    import newton
    import warp as wp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda:0"))
    args = parser.parse_args()
    if version("newton") != "1.7.0":
        raise RuntimeError("The site IMU benchmark targets Newton 1.7.0")
    wp.config.kernel_cache_dir = str(Path(sys.prefix) / "warp-kernels")
    wp.init()
    with wp.ScopedDevice(args.device):
        result = run(newton, wp, args.device)
    print(
        json.dumps(
            dict(newton=version("newton"), warp=version("warp-lang"), **result), allow_nan=False
        )
    )
    for row in result["rows"]:
        assert row["max_force_error"] <= 5e-4, (row["k"], row["max_force_error"])
        assert row["max_gyro_error"] <= 2e-5, (row["k"], row["max_gyro_error"])


if __name__ == "__main__":
    main()
