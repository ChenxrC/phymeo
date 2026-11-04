"""Compare Newton's two URDF rubber faces at accepted Phymeo grasp coordinates."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import newton
import numpy as np
import warp as wp


def compare(urdf, trace_path, device):
    frames = json.loads(trace_path.read_text(encoding="utf8"))["trace"]
    builder = newton.ModelBuilder()
    builder.add_urdf(
        urdf,
        floating=False,
        collapse_fixed_joints=False,
        parse_visuals_as_colliders=False,
        enable_self_collisions=False,
    )
    model = builder.finalize(device=device)
    state = model.state()

    def named(labels, key):
        matches = [i for i, value in enumerate(labels) if value.rsplit("/", 1)[-1] == key]
        if len(matches) != 1:
            raise ValueError(f"{key} must resolve to exactly one identifier: {matches}")
        return matches[0]

    offsets = model.joint_q_start.numpy()
    joints = [f"panda_joint{i}" for i in range(1, 8)] + [
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    indices = [offsets[named(model.joint_label, name)] for name in joints]
    body_ids = model.shape_body.numpy()
    scales = model.shape_scale.numpy()
    local = model.shape_transform.numpy()
    kind = model.shape_type.numpy()
    pads = []
    for side in ("left", "right"):
        owner = named(model.body_label, f"panda_{side}finger")
        (shape,) = np.flatnonzero(
            (body_ids == owner)
            & (kind == int(newton.GeoType.BOX))
            & np.all(np.isclose(scales, [0.00875, 0.0076, 0.00925], atol=1e-7), axis=1)
        )
        pads.append((owner, shape))
    results = []
    for i in np.linspace(0, len(frames) - 1, 17, dtype=int):
        frame = frames[i]
        q = model.joint_q.numpy().copy()
        q[indices] = frame["q_before"]
        state.joint_q.assign(q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        transforms = state.body_q.numpy()
        centers = []
        matrices = []
        for owner, shape in pads:
            body = transforms[owner]
            body_tf = wp.transform(wp.vec3(*body[:3]), wp.quat(*body[3:]))
            elem = local[shape]
            total = wp.transform_multiply(
                body_tf, wp.transform(wp.vec3(*elem[:3]), wp.quat(*elem[3:]))
            )
            centers.append(np.array(tuple(wp.transform_get_translation(total))))
            quat = wp.transform_get_rotation(total)
            matrices.append(
                np.column_stack([tuple(wp.quat_rotate(quat, wp.vec3(*axis))) for axis in np.eye(3)])
            )
        span = centers[0] - centers[1]
        axis = span / np.linalg.norm(span)
        normals = (axis, -axis)
        faces = [
            c - n * float(np.abs(n @ rot) @ scales[shape])
            for c, n, rot, (_, shape) in zip(centers, normals, matrices, pads, strict=True)
        ]
        measured = np.array(frame["pad_points_before_m"]).reshape(2, 4, 3).mean(axis=1)
        residual = np.linalg.norm(np.array(faces) - measured, axis=1)
        if np.max(residual) > 2e-6:
            raise AssertionError(f"step {i}: face center gaps {residual}")
        results.append(dict(step=int(i), errors_m=residual.tolist()))
    return dict(
        newton=version("newton"),
        warp=version("warp-lang"),
        device=str(model.device),
        urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
        samples=results,
        max_face_error_m=max(max(row["errors_m"]) for row in results),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("urdf", "trace", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    args = parser.parse_args()
    with args.out.open("x", encoding="utf8") as stream:
        json.dump(compare(args.urdf, args.trace, args.device), stream, indent=2)
