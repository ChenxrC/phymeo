"""Independent parallel beam origins and curved analytic shapes through Newton SensorCamera."""

import argparse
import json
from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton.sensors import SensorCamera

import phymeo as ph


def run(device, target):
    builder = newton.ModelBuilder()
    registrations = (
        ("ellipsoid", (-1.08, 0.07, -0.4), (0.35, 0.19, 0.51), (0, 1, 1), 0.7),
        ("cylinder", (0.03, -0.09, -0.22), (0.31, 0.49), (1, 2, -1), 1.0),
        ("cone", (1.06, 0.035, -0.35), (0.41, 0.56), (1, -2, 0.3), 1.2),
    )
    shapes, poses = [], {}
    for kind, center, size, axis, angle in registrations:
        direction = np.asarray(axis, float) / np.linalg.norm(axis)
        rot = np.r_[np.sin(angle / 2) * direction, np.cos(angle / 2)]
        transform = wp.transform(wp.vec3(*center), wp.quat(*rot))
        if kind == "ellipsoid":
            index = builder.add_shape_ellipsoid(
                -1, rx=size[0], ry=size[1], rz=size[2], xform=transform
            )
        else:
            add = builder.add_shape_cylinder if kind == "cylinder" else builder.add_shape_cone
            index = add(-1, radius=size[0], half_height=size[1], xform=transform)
        key = ph.CollisionKey(0, index, 0)
        shapes.append(
            ph.CollisionShape(key, kind, (0,), curved=ph.CurvedSolid(kind, size, tuple(rot)))
        )
        poses[key] = ph.CollisionMotion([center])
    model = builder.finalize(device=device)
    state = model.state()
    native = SensorCamera(model, load_textures=False)
    width, height = 97, 43
    x, y = np.meshgrid(np.linspace(-1.9, 1.9, width), np.linspace(-0.72, 0.72, height))
    origin = np.stack((x, y + 0.06 * x, 2 + 0.03 * x), axis=-1).astype(np.float32)
    dirs = np.stack((0.027 * y, 0.019 * x, -np.ones_like(x)), axis=-1)
    dirs = (dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)).astype(np.float32)
    rays = wp.array(
        np.stack((origin, dirs), axis=-2)[:, :, None, :, :], dtype=wp.vec3f, device=device
    )
    xforms = wp.array([wp.transform_identity()], dtype=wp.transformf, device=device)
    ids = native.allocate_image_shape_index(1, width, height)
    depth = native.allocate_image_depth(1, width, height)
    normal = native.allocate_image_normal(1, width, height)
    model.bvh_refit_shapes(state)
    native.update(
        state, xforms, rays, depth_image=depth, shape_index_image=ids, normal_image=normal
    )
    ph_result = ph.raycast_scene(
        ph.CollisionModel(shapes), poses, origin.reshape(-1, 3), dirs.reshape(-1, 3)
    )
    index = ids.numpy()[0].astype(np.int64)
    index[index == 0xFFFFFFFF] = -1
    z = depth.numpy()[0]
    normals = normal.numpy()[0]
    ours = ph_result.shape_index.reshape(height, width)
    same_hit = (index >= 0) & (index == ours)
    matched_normal = np.max(
        abs(normals[same_hit] - ph_result.world_normals.reshape(height, width, 3)[same_hit]),
        initial=0.0,
    )
    depth_error = np.max(
        abs(z[same_hit] - ph_result.distance_m.reshape(height, width)[same_hit]), initial=0.0
    )
    summary = dict(
        mismatched_ids=int(np.count_nonzero(index != ours)),
        depth_error_m=float(depth_error),
        normal_component_error=float(matched_normal),
        pixels=[int(np.count_nonzero(index == i)) for i in range(3)],
    )
    with target.open("x", encoding="utf8") as stream:
        json.dump(
            dict(
                device=device,
                summary=summary,
                shape_index=index.tolist(),
                phymeo_shape_index=ours.tolist(),
                native_depth_m=z.tolist(),
                native_normals=normals.tolist(),
            ),
            stream,
            allow_nan=False,
        )
    print(json.dumps(summary))
    assert summary["mismatched_ids"] == 0 and min(summary["pixels"]) > 0
    assert summary["depth_error_m"] < 1e-4 and summary["normal_component_error"] < 1e-3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("choose a new result path")
    with wp.ScopedDevice(args.device):
        run(args.device, args.out)
