"""Exercise Newton contact detection and dynamics on one explicitly selected device."""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import newton
import numpy as np
import warp as wp


def run(device_name):
    wp.config.kernel_cache_dir = str(Path(sys.prefix) / "warp-kernels")
    wp.init()
    device = wp.get_device(device_name)
    if str(device) != device_name or (device_name.startswith("cuda") and not device.is_cuda):
        raise RuntimeError(f"Requested {device_name}, got {device}")
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        ground = builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(-1.0, 0.0, 0.24), wp.quat_identity()))
        sphere = builder.add_shape_sphere(body, radius=0.25)
        falling = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.8), wp.quat_identity()))
        builder.add_shape_sphere(falling, radius=0.25)
        model = builder.finalize(device=device)
        first, second = model.state(), model.state()
        collision = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts = collision.contacts()
        collision.collide(first, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        pairs = {
            tuple(sorted((int(a), int(b))))
            for a, b in zip(
                contacts.rigid_contact_shape0.numpy()[:count],
                contacts.rigid_contact_shape1.numpy()[:count],
                strict=True,
            )
        }
        if pairs != {tuple(sorted((ground, sphere)))}:
            raise AssertionError(f"Unexpected sphere-plane contact pairs {pairs}")
        z0 = float(first.body_q.numpy()[falling, 2])
        first.clear_forces()
        newton.solvers.SolverXPBD(model).step(first, second, model.control(), contacts, 1.0 / 120.0)
        wp.synchronize_device(device)
        if str(second.body_q.device) != device_name:
            raise AssertionError("Solver output was published on a different device")
        positions = second.body_q.numpy()
        z1 = float(positions[falling, 2])
        if not np.isfinite(positions).all() or not 0.0 < z1 < z0:
            raise AssertionError(f"Free sphere failed the gravity control: {z0} -> {z1}")
    return dict(
        newton=version("newton"),
        warp=version("warp-lang"),
        numpy=version("numpy"),
        device=str(device),
        device_name=device.name,
        python=sys.version.split()[0],
        contact_count=count,
        shape_pairs=sorted(pairs),
        free_sphere_z_m=[z0, z1],
        package_path=newton.__file__,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda:0"))
    args = parser.parse_args()
    print(json.dumps(run(args.device), ensure_ascii=False, allow_nan=False))
