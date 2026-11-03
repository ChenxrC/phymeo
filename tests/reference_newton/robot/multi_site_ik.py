"""Newton 1.7 native multi-objective IK and world-frame observations of mounted tools."""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
from site_ik_fixture import fixture, qmatrix


def model_and_sites(newton, wp, parents, kinds, anchors, frames, axes, lower, upper, sites, device):
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    links = []
    joints = []
    for i, (parent, kind) in enumerate(zip(parents, kinds, strict=True)):
        link = b.add_link(label=f"link{i}", mass=1.0, inertia=np.eye(3), lock_inertia=True)
        add = b.add_joint_revolute if kind == "revolute" else b.add_joint_prismatic
        joint = add(
            -1 if parent < 0 else links[parent],
            link,
            axis=tuple(axes[i]),
            parent_xform=wp.transform(wp.vec3(*anchors[i]), wp.quat(*frames[i])),
            limit_lower=float(lower[i]),
            limit_upper=float(upper[i]),
            armature=0.0,
            damping=0.0,
        )
        links.append(link)
        joints.append(joint)
    b.add_articulation(joints)
    site_ids = [
        b.add_site(links[link], label=f"tool{k}", xform=wp.transform(wp.vec3(*pos), wp.quat(*quat)))
        for k, (link, pos, quat) in enumerate(sites)
    ]
    world = b.add_site(-1, label="world", xform=wp.transform_identity())
    model = b.finalize(device=device)
    sensor = newton.sensors.SensorFrameTransform(model, shapes=site_ids, reference_sites=[world])
    return model, sensor


def solve(newton, wp, model, sensor, objectives, seed):
    from newton.ik import IKJacobianType, IKSolver

    state = model.state()
    state.joint_q.assign(seed)
    batch = state.joint_q.reshape((1, model.joint_coord_count))
    optimizer = IKSolver(
        model=model, n_problems=1, objectives=objectives, jacobian_mode=IKJacobianType.ANALYTIC
    )
    optimizer.step(batch, batch, iterations=64)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    sensor.update(state)
    return state.joint_q.numpy().astype(float), sensor.transforms.numpy().astype(float)


def bimanual(newton, wp, device, mirror):
    from newton.ik import IKObjectiveJointLimit, IKObjectivePosition, IKObjectiveRotation

    f = fixture(mirror)
    sites = [(link, f["local"][k], f["mount"][k]) for k, link in enumerate((2, 4))]
    model, sensor = model_and_sites(
        newton,
        wp,
        f["parents"],
        f["kinds"],
        f["anchors"],
        f["frames"],
        f["axes"],
        f["lower"],
        f["upper"],
        sites,
        device,
    )
    goals = []
    for k, (link, offset, rot) in enumerate(sites):
        goals += [
            IKObjectivePosition(
                link,
                wp.vec3(*offset),
                wp.array([f["goals"][k]], dtype=wp.vec3),
                weight=float(2 + k),
            ),
            IKObjectiveRotation(
                link,
                wp.quat(*rot),
                wp.array([f["rotations"][k]], dtype=wp.vec4),
                weight=float(1 + k),
            ),
        ]
    goals.append(
        IKObjectiveJointLimit(model.joint_limit_lower, model.joint_limit_upper, weight=2.0)
    )
    q, tf = solve(newton, wp, model, sensor, goals, f["seed"])
    errors = []
    for k, row in enumerate(tf):
        rotation = qmatrix(row[3:] / np.linalg.norm(row[3:]))
        delta = rotation.T @ qmatrix(f["rotations"][k])
        sine = (
            np.linalg.norm(
                [delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]]
            )
            / 2
        )
        errors.append(
            [np.linalg.norm(row[:3] - f["goals"][k]), np.arctan2(sine, (np.trace(delta) - 1) / 2)]
        )
    np.testing.assert_allclose(q, f["truth"], rtol=0, atol=3e-4)
    assert np.max(errors) <= 2e-4, errors
    return dict(
        q=q.tolist(),
        site_transforms=tf.tolist(),
        errors=np.asarray(errors).tolist(),
        device=str(sensor.transforms.device),
    )


def scalar(newton, wp, device, soft_limit):
    from newton.ik import IKObjectiveJointLimit, IKObjectivePosition

    limits = (-0.25, 0.25) if soft_limit else (-2.0, 2.0)
    m, sensor = model_and_sites(
        newton,
        wp,
        [-1],
        ["prismatic"],
        [[0, 0, 0]],
        [[0, 0, 0, 1]],
        [[1, 0, 0]],
        [limits[0]],
        [limits[1]],
        [(0, (0, 0, 0), (0, 0, 0, 1))],
        device,
    )

    def position(x, weight):
        return IKObjectivePosition(
            0, wp.vec3(0, 0, 0), wp.array([(x, 0, 0)], dtype=wp.vec3), weight=weight
        )

    goals = (
        [
            position(1.0, 1.0),
            IKObjectiveJointLimit(m.joint_limit_lower, m.joint_limit_upper, weight=2.0),
        ]
        if soft_limit
        else [position(-1.0, 1.0), position(1.0, 2.0)]
    )
    q, pose = solve(newton, wp, m, sensor, goals, [-0.125])
    wanted = 0.4 if soft_limit else 0.6
    np.testing.assert_allclose([q[0], pose[0, 0]], wanted, rtol=0, atol=4e-4)
    return dict(
        q=float(q[0]),
        point=pose[0, :3].tolist(),
        upper=limits[1],
        violation=max(0.0, float(q[0]) - limits[1]),
    )


def fixed_origin_direction(newton, wp, device):
    from newton.ik import IKObjectivePosition

    # A unit offset from a fixed pivot is geometrically equivalent to pointing
    # the local x axis. The independent x-roll DOF leaves this point invariant.
    m, sensor = model_and_sites(
        newton,
        wp,
        [-1, 0],
        ["revolute"] * 2,
        np.zeros((2, 3)),
        np.tile([0, 0, 0, 1], (2, 1)),
        [[0, 0, 1], [1, 0, 0]],
        [-3.2, -1.0],
        [3.2, 1.0],
        [(1, (1, 0, 0), (0, 0, 0, 1))],
        device,
    )
    obj = IKObjectivePosition(1, wp.vec3(1, 0, 0), wp.array([(0.0, 1.0, 0.0)], dtype=wp.vec3))
    q, pose = solve(newton, wp, m, sensor, [obj], [-0.35, 0.625])
    np.testing.assert_allclose(q, [np.pi / 2, 0.625], rtol=0, atol=3e-4)
    np.testing.assert_allclose(pose[0, :3], [0, 1, 0], rtol=0, atol=2e-4)
    return dict(q=q.tolist(), unit_point=pose[0, :3].tolist())


def main():
    import newton
    import warp as wp

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    args = p.parse_args()
    if version("newton") != "1.7.0":
        raise RuntimeError("This IK observation targets Newton 1.7.0")
    wp.config.kernel_cache_dir = str(Path(sys.prefix) / "warp-kernels")
    wp.init()
    with wp.ScopedDevice(args.device):
        rows = dict(
            branched=bimanual(newton, wp, args.device, False),
            mirror=bimanual(newton, wp, args.device, True),
            conflict=scalar(newton, wp, args.device, False),
            soft_limit=scalar(newton, wp, args.device, True),
            fixed_pivot_direction=fixed_origin_direction(newton, wp, args.device),
        )
    print(
        json.dumps(
            dict(
                newton=version("newton"),
                warp=version("warp-lang"),
                device=args.device,
                records=rows,
            ),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
