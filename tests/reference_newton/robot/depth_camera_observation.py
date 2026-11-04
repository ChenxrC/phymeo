"""Compare depth, normal and shape identity to Newton SensorCamera on actual moving state."""

import argparse
import json
from importlib.metadata import version
from itertools import product
from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton.sensors import SensorCamera

import phymeo as ph


def run(device):
    width, height = 35, 27
    fov = 1.03
    eye = np.array((0.013, -0.047, 2.9))
    center = np.array((0.11, -0.04, 0.72))
    table_center = np.array((0.03, 0.06, -0.028))
    half = np.array((1.40, 0.78, 0.028))
    cap = np.array((0.59, 0.21, 0.48))
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -1.0), up_axis=newton.Axis.Z)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0, restitution=0.0)
    ball = builder.add_body(
        xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(*np.diag([0.04] * 3).ravel()),
        lock_inertia=True,
        label="observed-ball",
    )
    ball_id = builder.add_shape_sphere(ball, radius=0.31, cfg=cfg, label="ball")
    table_id = builder.add_shape_box(
        -1,
        hx=half[0],
        hy=half[1],
        hz=half[2],
        xform=wp.transform(wp.vec3(*table_center), wp.quat_identity()),
        cfg=cfg,
        label="table",
    )
    cap_id = builder.add_shape_capsule(
        -1,
        radius=0.12,
        half_height=0.27,
        xform=wp.transform(wp.vec3(*cap), wp.quat(0.0, 2**-0.5, 0.0, 2**-0.5)),
        cfg=cfg,
        label="barrel",
    )
    model = builder.finalize(device=device)
    state, target = model.state(), model.state()
    state.joint_qd.assign([1.3, 0.18, 0.0, 0.0, 0.0, 0.0])
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    native = SensorCamera(model, load_textures=False)
    rays = SensorCamera.compute_camera_rays_pinhole(
        width, height, camera_fov=fov, device=model.device
    )
    xforms = wp.array(
        [wp.transform(wp.vec3(*eye), wp.quat_identity())], dtype=wp.transformf, device=model.device
    )
    image = native.allocate_image_depth(1, width, height)
    forward = native.allocate_image_depth_forward(1, width, height)
    labels = native.allocate_image_shape_index(1, width, height)
    normals = native.allocate_image_normal(1, width, height)
    keys = [ph.CollisionKey(0, idx, idx) for idx in (ball_id, table_id, cap_id)]
    shapes = [
        ph.CollisionShape(keys[0], "sphere", (0,), radius=0.31),
        ph.CollisionShape(keys[1], "hull", tuple(range(8))),
        ph.CollisionShape(keys[2], "capsule", (0, 1), radius=0.12),
    ]
    registered = ph.CollisionModel(shapes)
    poses = {
        keys[1]: ph.CollisionMotion(
            table_center + np.asarray(list(product((-1.0, 1.0), repeat=3))) * half
        ),
        keys[2]: ph.CollisionMotion(np.array([cap - [0.27, 0, 0], cap + [0.27, 0, 0]])),
    }
    cam = ph.DepthCamera(width, height, fov)
    ph_model = ph.RigidBodyModel(np.ones(1), np.full((1, 3), 0.04), np.array([0.0, 0.0, -1.0]))
    ph_state = ph.RigidBodyState(
        center[None, :],
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[1.3, 0.18, 0.0]]),
        np.zeros((1, 3)),
    )
    ph_solver = ph.RigidSemiImplicitSolver()
    ph_inputs = ph.RigidBodyInputs.zeros(1)
    frames = []
    for tick in range(25):
        if tick in (0, 12, 24):
            pos = state.body_q.numpy()[ball, :3].copy()
            np.testing.assert_allclose(ph_state.position[0], pos, rtol=0, atol=2e-6)
            poses[keys[0]] = ph.CollisionMotion(ph_state.position.copy())
            frame = cam.capture(registered, poses, world=0, position_m=eye)
            model.bvh_refit_shapes(state)
            native.update(
                state,
                xforms,
                rays,
                depth_image=image,
                forward_depth_image=forward,
                shape_index_image=labels,
                normal_image=normals,
            )
            ids = labels.numpy()[0].astype(np.int64)
            ids[ids == 0xFFFFFFFF] = -1
            np.testing.assert_array_equal(ids, frame.shape_index)
            assert set(np.unique(ids)) == {-1, ball_id, table_id, cap_id}
            hit = ids >= 0
            actual = image.numpy()[0]
            ahead = forward.numpy()[0]
            vec = normals.numpy()[0]
            np.testing.assert_allclose(frame.ray_depth_m[hit], actual[hit], rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(frame.forward_depth_m[hit], ahead[hit], rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(frame.world_normals[hit], vec[hit], rtol=2e-4, atol=2e-4)
            assert np.isinf(frame.ray_depth_m[~hit]).all() and np.all(actual[~hit] == 0)
            frames.append(
                dict(
                    tick=tick,
                    position=pos.tolist(),
                    phymeo_position=ph_state.position[0].tolist(),
                    labels=ids.tolist(),
                    ray_depth_m=actual.tolist(),
                    forward_depth_m=ahead.tolist(),
                    world_normals=vec.tolist(),
                    max_ray_error=float(np.max(abs(frame.ray_depth_m[hit] - actual[hit]))),
                    max_forward_error=float(np.max(abs(frame.forward_depth_m[hit] - ahead[hit]))),
                    max_normal_error=float(np.max(abs(frame.world_normals[hit] - vec[hit]))),
                )
            )
        if tick < 24:
            ph_state = ph_solver.step(ph_model, ph_state, ph_inputs, 1 / 128).state
            state.clear_forces()
            solver.step(state, target, None, None, 1 / 128)
            state, target = target, state
    assert (
        frames[0]["labels"] != frames[-1]["labels"]
        and frames[-1]["position"][0] > frames[0]["position"][0] + 0.2
    )
    return frames


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    args.add_argument("--out", type=Path, required=True)
    config = args.parse_args()
    if version("newton") != "1.7.0" or config.out.exists():
        raise ValueError("expected Newton 1.7 and a new report path")
    wp.init()
    with wp.ScopedDevice(config.device):
        frames = run(config.device)
    with config.out.open("x", encoding="utf8") as out:
        json.dump(
            dict(device=config.device, width=35, height=27, frames=frames), out, allow_nan=False
        )
    print(
        "compared every pixel and matched all three shape masks and background in",
        len(frames),
        "frames",
    )
