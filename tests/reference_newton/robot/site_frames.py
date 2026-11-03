"""Newton 1.7 native relative poses and independently derived COM-to-site velocities."""

import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np


def _transform(wp, row):
    return wp.transform(wp.vec3(*row[:3]), wp.quat(*row[3:]))


def _rotation(wp, quaternion):
    return np.asarray(wp.quat_to_matrix(quaternion), dtype=float).reshape(3, 3)


def _build(newton, wp, fixture, row, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    links, roots, groups = [], [], {}
    for i, parent in enumerate(fixture["parents"]):
        body = builder.add_link(
            label=f"link_{i}",
            mass=fixture["mass_kg"][i],
            com=fixture["com_m"][i],
            inertia=np.eye(3),
            lock_inertia=True,
        )
        links.append(body)
        add_joint = (
            builder.add_joint_revolute
            if fixture["kinds"][i] == "revolute"
            else builder.add_joint_prismatic
        )
        joint = add_joint(
            -1 if parent < 0 else links[parent],
            body,
            axis=fixture["axes"][i],
            parent_xform=wp.transform(fixture["anchors"][i], wp.quat_identity()),
        )
        root = i if parent < 0 else roots[parent]
        roots.append(root)
        groups.setdefault(root, []).append(joint)
    for tree in groups.values():
        builder.add_articulation(tree)
    site_ids = []
    for role in ("reference", "target"):
        index, point, rotation = row[role]
        site_ids.append(
            builder.add_site(
                -1 if index is None else links[index],
                label=role,
                xform=wp.transform(point, wp.quat(*rotation)),
            )
        )
    return builder.finalize(device=device), site_ids


def _observe(newton, wp, model, site_ids, q, v):
    from newton.sensors import SensorFrameTransform

    state = model.state()
    state.joint_q.assign(q)
    state.joint_qd.assign(v)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    ref, target = site_ids
    sensor = SensorFrameTransform(model, shapes=[target], reference_sites=[ref])
    sensor.update(state)
    native = sensor.transforms.numpy()[0]
    native_fields = dict(
        position_m=native[:3].tolist(), rotation=_rotation(wp, wp.quat(*native[3:])).tolist()
    )

    owners = model.shape_body.numpy()
    local = model.shape_transform.numpy()
    frames, twists, com = state.body_q.numpy(), state.body_qd.numpy(), model.body_com.numpy()

    def site_world(site):
        tf = _transform(wp, local[site])
        body = owners[site]
        return tf if body < 0 else wp.transform_multiply(_transform(wp, frames[body]), tf)

    # Evaluate both rigid velocity fields directly at the same world point.
    # Newton stores world COM velocity followed by angular velocity in body_qd.
    point = np.asarray(site_world(target).p, dtype=float)

    def field(site):
        body = owners[site]
        if body < 0:
            return np.zeros(3), np.zeros(3)
        center = np.asarray(
            wp.transform_point(_transform(wp, frames[body]), wp.vec3(*com[body])), dtype=float
        )
        velocity, spin = twists[body, :3].astype(float), twists[body, 3:].astype(float)
        return velocity + np.cross(spin, point - center), spin

    w_to_r = _rotation(wp, site_world(ref).q).T
    ref_field, ref_spin = field(ref)
    target_field, target_spin = field(target)
    derived = dict(
        linear_velocity_m_s=(w_to_r @ (target_field - ref_field)).tolist(),
        angular_velocity_rad_s=(w_to_r @ (target_spin - ref_spin)).tolist(),
    )
    return native_fields, derived


def main():
    import argparse

    import newton
    import warp as wp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    options = parser.parse_args()
    if version("newton") != "1.7.0":
        raise RuntimeError("The reference fixtures are for Newton 1.7.0")
    wp.config.kernel_cache_dir = str(Path(sys.prefix) / "warp-kernels")
    wp.init()
    catalog = json.loads(Path(__file__).with_name("site_frames.json").read_text(encoding="utf-8"))
    records = []
    with wp.ScopedDevice(options.device):
        for row in catalog["cases"]:
            model, sites = _build(
                newton, wp, catalog["fixtures"][row["model"]], row, options.device
            )
            for sign in (1, -1) if row["id"] == "ROTATING-REF" else (1,):
                native, derived = _observe(
                    newton, wp, model, sites, row["q"], sign * np.array(row["v"])
                )
                for name, value in native.items():
                    np.testing.assert_allclose(
                        value, row[name], rtol=0, atol=2e-5, err_msg=row["id"] + "/" + name
                    )
                for name, value in derived.items():
                    np.testing.assert_allclose(
                        value,
                        sign * np.array(row[name]),
                        rtol=0,
                        atol=2e-5,
                        err_msg=row["id"] + "/" + name,
                    )
                records.append(
                    dict(
                        id=row["id"],
                        velocity_sign=sign,
                        device=str(model.device),
                        native_frame=native,
                        derived_from_body_qd=derived,
                    )
                )
    print(
        json.dumps(
            dict(newton=version("newton"), warp=version("warp-lang"), records=records),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
