"""Compare every device ray against the frozen 97×43 native Newton SensorCamera capture."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import phymeo as ph
from phymeo.backends.tyx_raycast import TyxRaycaster


def frozen_input():
    # Independent replay of tests/reference_newton/robot/curved_ray_observation.py.
    specs = (
        ("ellipsoid", (-1.08, 0.07, -0.4), (0.35, 0.19, 0.51), (0, 1, 1), 0.7),
        ("cylinder", (0.03, -0.09, -0.22), (0.31, 0.49), (1, 2, -1), 1.0),
        ("cone", (1.06, 0.035, -0.35), (0.41, 0.56), (1, -2, 0.3), 1.2),
    )
    shapes, poses = [], {}
    for i, (kind, center, dimensions, axis, angle) in enumerate(specs):
        unit = np.asarray(axis) / np.linalg.norm(axis)
        orientation = np.r_[np.sin(angle / 2) * unit, np.cos(angle / 2)]
        key = ph.CollisionKey(0, i, 0)
        shapes.append(
            ph.CollisionShape(
                key, kind, (0,), curved=ph.CurvedSolid(kind, dimensions, tuple(orientation))
            )
        )
        poses[key] = ph.CollisionMotion([center])
    x, y = np.meshgrid(np.linspace(-1.9, 1.9, 97), np.linspace(-0.72, 0.72, 43))
    origin = np.stack((x, y + 0.06 * x, 2 + 0.03 * x), axis=-1).astype(np.float32)
    rays = np.stack((0.027 * y, 0.019 * x, -np.ones_like(x)), axis=-1)
    rays = (rays / np.linalg.norm(rays, axis=-1, keepdims=True)).astype(np.float32)
    return ph.CollisionModel(shapes), poses, origin.reshape(-1, 3), rays.reshape(-1, 3)


def compare(result, index, depth, normals):
    comparison = {
        "label_mismatches": int(np.count_nonzero(result.shape_index != index)),
        "shapes": {},
    }
    for i, name in enumerate(("ellipsoid", "cylinder", "cone")):
        mask = (result.shape_index == i) & (index == i)
        distance = abs(result.distance_m[mask] - depth[mask])
        component = np.max(abs(result.world_normals[mask] - normals[mask]), axis=1)
        comparison["shapes"][name] = dict(
            matched_rays=int(mask.sum()),
            distance_error_m=float(np.max(distance, initial=0)),
            normal_component_error=float(np.max(component, initial=0)),
            distance_above_1e_4_m=int(np.count_nonzero(distance >= 1e-4)),
            normal_above_1e_3=int(np.count_nonzero(component >= 1e-3)),
        )
    return comparison


def main():
    import tyx as tx

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--native", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    cli.add_argument("--cache", required=True, type=Path)
    args = cli.parse_args()
    if args.output.exists():
        cli.error("choose a new result path")
    raw = args.native.read_bytes()
    native = json.loads(raw)
    index = np.asarray(native["shape_index"]).reshape(-1)
    model, poses, origin, directions = frozen_input()
    host = ph.raycast_scene(model, poses, origin, directions)
    np.testing.assert_array_equal(
        host.shape_index, np.asarray(native["phymeo_shape_index"]).reshape(-1)
    )
    np.testing.assert_array_equal([np.count_nonzero(index == i) for i in range(3)], [179, 491, 398])
    with TyxRaycaster(args.mode, cache_root=args.cache) as cast:
        before = tx.runtime_stats()
        got = cast.query(model, poses, origin, directions)
        delta = tx.runtime_stats() - before
        counts = cast.stats()
    miss = got.shape_index == -1
    assert np.isposinf(got.distance_m[miss]).all() and not np.any(got.world_normals[miss])
    ncmp = compare(
        got,
        index,
        np.asarray(native["native_depth_m"]).reshape(-1),
        np.asarray(native["native_normals"]).reshape(-1, 3),
    )
    hostcmp = compare(got, host.shape_index, host.distance_m, host.world_normals)
    report = dict(
        mode=args.mode,
        native_device=native["device"],
        native_sha256=hashlib.sha256(raw).hexdigest(),
        native=ncmp,
        host=hostcmp,
        device_work=asdict(delta),
        counters=counts,
        shape_index=got.shape_index.tolist(),
        distance_m=[None if m else float(d) for d, m in zip(got.distance_m, miss, strict=True)],
        world_normals=got.world_normals.tolist(),
    )
    with args.output.open("x", encoding="utf8") as f:
        json.dump(report, f, allow_nan=False)
    print(json.dumps({k: report[k] for k in ("mode", "native", "host", "device_work")}))
    assert delta.cuda_kernel_launches == int(args.mode == "cuda")
    for comparison in (ncmp, hostcmp):
        assert comparison["label_mismatches"] == 0
        assert all(
            row["matched_rays"]
            and row["distance_above_1e_4_m"] == 0
            and row["normal_above_1e_3"] == 0
            for row in comparison["shapes"].values()
        )


if __name__ == "__main__":
    main()
