"""Check multi-joint Panda pad paths against independent Newton FK and a finite tabletop."""

import argparse
import json
from fractions import Fraction
from itertools import product
from pathlib import Path

import newton
import numpy as np
import warp as wp

import phymeo as ph
from phymeo.builders.panda import PANDA_URDF, load_panda
from phymeo.dynamics.robot_kinematics import forward_kinematics
from phymeo.geometry.articulated_hull_clearance import link_hull_motion, require_cloth_clearance


def comparison(record, device):
    start = np.asarray(record["events"]["close"]["q"])
    end = np.asarray(record["events"]["transfer"]["q"])
    assert np.all(start != end)
    asset = load_panda()
    fks = [forward_kinematics(asset.model, q) for q in (start, end)]
    builder = newton.ModelBuilder()
    builder.add_urdf(
        PANDA_URDF,
        floating=False,
        collapse_fixed_joints=False,
        parse_visuals_as_colliders=False,
        enable_self_collisions=False,
    )
    model = builder.finalize(device=device)
    state = model.state()

    def lookup(labels, name):
        rows = [i for i, key in enumerate(labels) if key.rsplit("/", 1)[-1] == name]
        if len(rows) != 1:
            raise ValueError(f"ambiguous native symbol {name}")
        return rows[0]

    offsets = model.joint_q_start.numpy()
    indices = [offsets[lookup(model.joint_label, n)] for n in asset.joint_names]
    q = model.joint_q.numpy().copy()
    left = next(c for c in asset.collisions if c.name == "panda_leftfinger_pad")
    owner = lookup(model.body_label, left.link_name)
    native_ids = np.flatnonzero(
        (model.shape_body.numpy() == owner)
        & (model.shape_type.numpy() == int(newton.GeoType.BOX))
        & np.all(
            np.isclose(model.shape_scale.numpy(), [0.00875, 0.0076, 0.00925], atol=1e-7), axis=1
        )
    )
    (native_shape,) = native_ids.tolist()
    offsets_xyz = (
        np.array(list(product((-1, 1), repeat=3))) * model.shape_scale.numpy()[native_shape]
    )
    native_shape_tf = model.shape_transform.numpy()[native_shape]
    local = asset.authored_to_model_points(left.link_name, left.local_vertices)
    tool = ph.LinkHullTool(left.name, "arm", left.link_index, local, ph.NormalLayer(0.005, 100))
    support, slack = link_hull_motion(asset.model, tool, left.shape, start, end, *fks)
    samples = []
    for k in range(9):
        t = Fraction(k, 8)
        q[indices] = start + float(t) * (end - start)
        state.joint_q.assign(q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        b = state.body_q.numpy()[owner]
        tf = wp.transform_multiply(
            wp.transform(wp.vec3(*b[:3]), wp.quat(*b[3:])),
            wp.transform(wp.vec3(*native_shape_tf[:3]), wp.quat(*native_shape_tf[3:])),
        )
        expected = np.array(
            [tuple(wp.transform_point(tf, wp.vec3(*corner))) for corner in offsets_xyz]
        )
        actual, err = support.pose(t)
        distances = np.linalg.norm(actual.start[:, None] - expected, axis=2)
        match = np.argmin(distances, axis=1)
        assert len(set(match.tolist())) == 8
        discrepancy = float(np.max(distances[np.arange(8), match]))
        assert discrepancy < 2e-6
        samples.append(
            dict(
                t=float(t),
                phymeo=actual.start.tolist(),
                newton=expected[match].tolist(),
                error_m=discrepancy,
                ideal_to_proposal_error_m=float(err),
            )
        )
    center = np.asarray(record["table_center_m"])
    half = np.asarray(record["table_half_extents_m"])
    verts = center + half * np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    for ids in ((0, 1, 2), (0, 2, 3)):
        face = ph.CollisionShape(ph.CollisionKey(0, 100, 0), "triangle", ids)
        require_cloth_clearance(support, face, ph.CollisionMotion(verts[list(ids)]), 0.0005, slack)
    return dict(
        device=device,
        start=start.tolist(),
        end=end.tolist(),
        samples=samples,
        obstacle="two finite top triangles, 0.5 mm coating",
        comparison="affine q between two measured poses",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("choose a new report file")
    with wp.ScopedDevice(args.device):
        report = comparison(json.loads(args.record.read_text(encoding="utf8")), args.device)
    with args.out.open("x", encoding="utf8") as f:
        json.dump(report, f, allow_nan=False)
    print(
        "all 72 corners and continuous finite top-face clearance passed; max distance m",
        max(row["error_m"] for row in report["samples"]),
    )
