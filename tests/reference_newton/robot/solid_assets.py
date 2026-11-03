"""Newton 1.7 native URDF/MJCF primitive import, device FK and actual ground contacts."""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np


def ry(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def named(labels, name):
    ids = [i for i, s in enumerate(labels) if s == name or s.endswith("/" + name)]
    if len(ids) != 1:
        raise ValueError((name, labels))
    return ids[0]


def expected(q):
    c, s = np.cos(q), np.sin(q)
    return {
        "barrel": 0.25 - 0.125 * s - 0.25 * abs(s) - 0.125 * abs(c),
        "capsule": 0.25 - 0.1875 * c - 0.25 * abs(s) - 0.0625,
        "oval": 0.25 + 0.0625 * s - 0.125 * c - np.hypot(0.25 * c, 0.0625 * s),
    }


def run(newton, wp, filename, device):
    b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
    b.rigid_gap = 0.0
    path = Path(__file__).with_name(filename)
    urdf = filename.endswith(".urdf")
    if urdf:
        b.add_urdf(str(path), floating=False, joint_damping_mode="passive", up_axis=newton.Axis.Z)
    else:
        b.add_mjcf(str(path), floating=False)
    ground = b.add_ground_plane(
        cfg=newton.ModelBuilder.ShapeConfig(gap=0.0, margin=0.0), label="world_floor"
    )
    model = b.finalize(device=device)
    s = model.state()
    carrier = named(model.body_label, "carrier")
    base = named(model.body_label, "base")
    joint = named(model.joint_label, "tilt")
    owner = model.shape_body.numpy()
    flags = model.shape_flags.numpy()
    kinds = model.shape_type.numpy()
    colliders = (flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
    active = np.flatnonzero((owner == carrier) & colliders)
    kind_names = {
        int(newton.GeoType.CYLINDER): "barrel",
        int(newton.GeoType.CAPSULE): "capsule",
        int(newton.GeoType.ELLIPSOID): "oval",
    }
    ids = {kind_names[int(kinds[i])]: int(i) for i in active}
    assert len(ids) == len(active) and set(ids) == (
        {"barrel"} if urdf else {"barrel", "capsule", "oval"}
    )
    root_ids = np.flatnonzero((owner == base) & colliders)
    assert len(root_ids) == 1 and int(kinds[root_ids[0]]) == int(newton.GeoType.CYLINDER)
    scale = model.shape_scale.numpy()
    want_scales = dict(barrel=[0.125, 0.25], capsule=[0.0625, 0.25], oval=[0.25, 0.125, 0.0625])
    for name, i in ids.items():
        want = want_scales[name]
        np.testing.assert_allclose(scale[i, : len(want)], want, rtol=0, atol=3e-7)
    np.testing.assert_allclose(scale[root_ids[0], :2], [0.0625, 0.125], rtol=0, atol=3e-7)
    np.testing.assert_allclose(model.body_mass.numpy()[carrier], 2, rtol=0, atol=3e-6)
    np.testing.assert_allclose(
        model.body_com.numpy()[carrier], [0.0625, -0.03125, 0.125], rtol=0, atol=3e-7
    )
    np.testing.assert_allclose(
        model.body_inertia.numpy()[carrier], np.diag([0.125, 0.1875, 0.25]), rtol=0, atol=3e-6
    )
    qs = model.joint_q_start.numpy()[joint]
    ds = model.joint_qd_start.numpy()[joint]
    np.testing.assert_allclose(model.joint_axis.numpy()[ds], [0, 1, 0], rtol=0, atol=3e-7)
    np.testing.assert_allclose(
        [model.joint_limit_lower.numpy()[ds], model.joint_limit_upper.numpy()[ds]],
        [-np.pi, np.pi],
        rtol=0,
        atol=3e-6,
    )
    pipe = newton.CollisionPipeline(model, rigid_contact_max=96, soft_contact_max=0)
    contacts = pipe.contacts()
    local = model.shape_transform.numpy()
    offsets = dict(
        barrel=[0.125, 0, 0], capsule=[0, -0.125, -0.1875], oval=[-0.0625, 0.3125, -0.125]
    )
    rotations = dict(barrel=ry(np.pi / 2), capsule=ry(np.pi / 2), oval=ry(-np.pi / 2))
    rows = []
    for angle in (0.0, np.arctan2(0.6, 0.8), np.pi / 2, -np.pi / 2, np.pi - 2**-24, np.pi):
        q = s.joint_q.numpy()
        q[qs] = angle
        s.joint_q.assign(q)
        newton.eval_fk(model, s.joint_q, s.joint_qd, s)
        pipe.collide(s, contacts)
        n = int(contacts.rigid_contact_count.numpy()[0])
        sides = [
            contacts.rigid_contact_shape0.numpy()[:n],
            contacts.rigid_contact_shape1.numpy()[:n],
        ]
        pts = [
            contacts.rigid_contact_point0.numpy()[:n] + contacts.rigid_contact_offset0.numpy()[:n],
            contacts.rigid_contact_point1.numpy()[:n] + contacts.rigid_contact_offset1.numpy()[:n],
        ]
        body = s.body_q.numpy()[carrier]
        tf = wp.transform(wp.vec3(*body[:3]), wp.quat(*body[3:]))
        entries = {}
        for name, i in ids.items():
            pose = wp.transform_multiply(
                tf, wp.transform(wp.vec3(*local[i, :3]), wp.quat(*local[i, 3:]))
            )
            center = np.asarray(pose.p, dtype=float)
            orientation = np.asarray(wp.quat_to_matrix(pose.q), dtype=float).reshape(3, 3)
            np.testing.assert_allclose(
                center, [0.25, -0.125, 0.25] + ry(angle) @ offsets[name], rtol=0, atol=4e-6
            )
            np.testing.assert_allclose(orientation, ry(angle) @ rotations[name], rtol=0, atol=5e-6)
            gap = expected(angle)[name]
            support = []
            for k in range(n):
                if sides[0][k] == i and sides[1][k] == ground:
                    side = 0
                elif sides[1][k] == i and sides[0][k] == ground:
                    side = 1
                else:
                    continue
                point = wp.transform_point(tf, wp.vec3(*pts[side][k]))
                support.append(float(point[2]))
            entries[name] = dict(
                center=center.tolist(),
                axis=orientation[:, 2].tolist(),
                shape_id=i,
                body_id=carrier,
                actual_contact_z=sorted(support),
                analytic_gap=gap,
            )
        rows.append(dict(angle=angle, entries=entries))
    return dict(
        source=filename,
        device=str(model.device),
        body=carrier,
        base_shape=int(root_ids[0]),
        rows=rows,
    )


def main():
    import newton
    import warp as wp

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    args = p.parse_args()
    if version("newton") != "1.7.0":
        raise RuntimeError("The reference targets Newton 1.7.0")
    wp.config.kernel_cache_dir = str(Path(sys.prefix) / "warp-kernels")
    wp.init()
    with wp.ScopedDevice(args.device):
        records = [
            run(newton, wp, file, args.device) for file in ("solid_arm.urdf", "solid_arm.mjcf.xml")
        ]
    print(
        json.dumps(
            dict(newton=version("newton"), warp=version("warp-lang"), records=records),
            allow_nan=False,
        )
    )
    for record in records:
        for row in record["rows"]:
            for name, item in row["entries"].items():
                gap = item["analytic_gap"]
                zs = item["actual_contact_z"]
                if gap < -1e-4:
                    assert zs, (record["source"], row["angle"], name, "missing contact")
                    np.testing.assert_allclose(min(zs), gap, rtol=0, atol=1e-4)
                elif gap > 1e-4:
                    assert not zs, (record["source"], row["angle"], name, zs)


if __name__ == "__main__":
    main()
