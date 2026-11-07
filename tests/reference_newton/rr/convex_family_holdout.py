"""Four frozen native Newton 1.7 convex contact poses; no dynamics or CCD."""

import argparse
import json
import sys
import traceback
from importlib.metadata import version
from pathlib import Path

import numpy as np

# Fixed before the first run. Every coordinate magnitude is at most 0.6 m.
TOLERANCES = {
    "length_m": 2e-6,
    "normal_l2": 2e-5,
    "unit_normal": 2e-5,
    "quaternion_l2": 2e-6,
}
RING_XY_M = ((-0.5, -0.1), (0.5, -0.1), (0.6, 0.0), (0.3, 0.1), (-0.5, 0.1))
CONTACT_CAPACITY = 256


def _capsule(center, axis):
    axis = np.asarray(axis, dtype=float)
    quat = np.r_[np.cross((0.0, 0.0, 1.0), axis), 1.0 + axis[2]]
    quat /= np.linalg.norm(quat)
    return {
        "kind": "capsule",
        "body_position_m": list(center),
        "body_xyzw": quat.tolist(),
        "axis_world": axis.tolist(),
        "radius_m": 0.05,
        "half_height_m": 0.5,
    }


def _polytope(kind, height):
    result = {
        "kind": kind,
        "body_position_m": [0.0, 0.0, height],
        "body_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    if kind == "box":
        result["half_extents_m"] = [0.1, 0.1, 0.1]
    else:
        result["vertices_m"] = [(x, y, z) for x, y in RING_XY_M for z in (-0.1, 0.1)]
        # CCW ring: reverse bottom, forward top, then outward side triangles.
        triangles = []
        for i in range(1, len(RING_XY_M) - 1):
            triangles.extend(((0, 2 * (i + 1), 2 * i), (1, 2 * i + 1, 2 * (i + 1) + 1)))
        for i in range(len(RING_XY_M)):
            j = (i + 1) % len(RING_XY_M)
            a, b, c, d = 2 * i, 2 * j, 2 * j + 1, 2 * i + 1
            triangles.extend(((a, b, c), (a, c, d)))
        result["triangles"] = triangles
    return result


def frozen_cases():
    """Return authored SI inputs, independent of either engine's observations."""
    pairs = (
        (
            "CC-PARALLEL",
            "CC",
            (_capsule((0, 0, 0), (1, 0, 0)), _capsule((0, 0, 0.09), (1, 0, 0))),
        ),
        (
            "CC-SKEW",
            "CC",
            (_capsule((0, 0, 0), (1, 0, 0)), _capsule((0, 0, 0.09), (3 / 5, 4 / 5, 0))),
        ),
        ("BB-FACE", "BB", (_polytope("box", 0), _polytope("box", 0.19))),
        ("HH-PENTAGON", "HH", (_polytope("hull", 0), _polytope("hull", 0.19))),
    )
    return tuple(
        {
            "id": name,
            "group": group,
            "shapes": shapes,
            "expected_gap_m": -0.01,
            "expected_normal_a_to_b": [0.0, 0.0, 1.0],
            "expected_intersection": True,
        }
        for name, group, shapes in pairs
    )


def _add_shape(newton, wp, builder, spec, cfg, label):
    pose = wp.transform(wp.vec3(*spec["body_position_m"]), wp.quat(*spec["body_xyzw"]))
    body = builder.add_body(xform=pose, label=label)
    if spec["kind"] == "capsule":
        shape = builder.add_shape_capsule(
            body, radius=spec["radius_m"], half_height=spec["half_height_m"], cfg=cfg
        )
    elif spec["kind"] == "box":
        hx, hy, hz = spec["half_extents_m"]
        shape = builder.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=cfg)
    else:
        mesh = newton.Mesh(
            np.asarray(spec["vertices_m"], dtype=np.float32),
            np.asarray(spec["triangles"], dtype=np.int32).ravel(),
        )
        shape = builder.add_shape_convex_hull(body, mesh=mesh, cfg=cfg)
    return body, shape


def _world_point(wp, pose, point):
    tf = wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:]))
    return np.asarray(wp.transform_point(tf, wp.vec3(*point)), dtype=float)


def _world_vector(wp, pose, vector):
    return np.asarray(wp.quat_rotate(wp.quat(*pose[3:]), wp.vec3(*vector)), dtype=float)


def _surface_error(spec, point):
    """Distance residual against the authored finite surface, not native hull output."""
    local = np.asarray(point, dtype=float) - np.asarray(spec["body_position_m"])
    if spec["kind"] == "capsule":
        axis = np.asarray(spec["axis_world"])
        along = np.clip(local @ axis, -spec["half_height_m"], spec["half_height_m"])
        return float(abs(np.linalg.norm(local - along * axis) - spec["radius_m"]))
    if spec["kind"] == "box":
        planes = np.r_[local - spec["half_extents_m"], -local - spec["half_extents_m"]]
    else:
        planes = [local[2] - 0.1, -local[2] - 0.1]
        for i, start in enumerate(RING_XY_M):
            edge = np.subtract(RING_XY_M[(i + 1) % len(RING_XY_M)], start)
            outward = np.asarray((edge[1], -edge[0])) / np.linalg.norm(edge)
            planes.append(float((local[:2] - start) @ outward))
        planes = np.asarray(planes)
    return float(max(0.0, np.max(planes), np.min(np.abs(planes))))


def _patch_error(spec, point, side):
    """Check the analytic +Z/-Z support patch, including finite capsule ends."""
    local = np.asarray(point, dtype=float) - np.asarray(spec["body_position_m"])
    sign = 1 if side == 0 else -1
    if spec["kind"] == "capsule":
        axis = np.asarray(spec["axis_world"])
        along = np.clip(local @ axis, -spec["half_height_m"], spec["half_height_m"])
        supported = along * axis + (0.0, 0.0, sign * spec["radius_m"])
        return float(np.linalg.norm(local - supported))
    return float(abs(local[2] - sign * 0.1))


def _maximum(rows, name):
    return max((row[name] for row in rows), default=None)


def _read_pose(newton, wp, case, device, record):
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        cfg = builder.ShapeConfig(margin=0.0, gap=0.0, has_particle_collision=False)
        registrations = [
            _add_shape(newton, wp, builder, spec, cfg, label)
            for spec, label in zip(case["shapes"], ("holdout_A", "holdout_B"), strict=True)
        ]
        body_ids, shape_ids = map(list, zip(*registrations, strict=True))
        model = builder.finalize(device=device)
        state = model.state()
        pipeline = newton.CollisionPipeline(
            model, broad_phase="nxn", reduce_contacts=False, rigid_contact_max=CONTACT_CAPACITY
        )
        contacts = pipeline.contacts()
        distance = wp.empty(contacts.rigid_contact_max, dtype=float, device=device)
        points = [
            wp.empty(contacts.rigid_contact_max, dtype=wp.vec3, device=device) for _ in range(2)
        ]
        pipeline.collide(state, contacts)
        newton.eval_rigid_contact_kinematics(
            model,
            state,
            contacts,
            out_distance=distance,
            out_point0_world=points[0],
            out_point1_world=points[1],
        )
        wp.synchronize_device(device)

        counters = contacts.contact_counters.numpy()
        count = int(contacts.rigid_contact_count.numpy()[0])
        active = max(0, min(count, contacts.rigid_contact_max))
        poses = state.body_q.numpy()
        shape_bodies = model.shape_body.numpy()
        native_types = [newton.GeoType(int(t)).name for t in model.shape_type.numpy()[shape_ids]]
        expected_types = [
            {"capsule": "CAPSULE", "box": "BOX", "hull": "CONVEX_MESH"}[s["kind"]]
            for s in case["shapes"]
        ]
        arrays = {
            "shape0": contacts.rigid_contact_shape0.numpy()[:active],
            "shape1": contacts.rigid_contact_shape1.numpy()[:active],
            "point0_body_m": contacts.rigid_contact_point0.numpy()[:active],
            "point1_body_m": contacts.rigid_contact_point1.numpy()[:active],
            "offset0_body_m": contacts.rigid_contact_offset0.numpy()[:active],
            "offset1_body_m": contacts.rigid_contact_offset1.numpy()[:active],
            "margin0_m": contacts.rigid_contact_margin0.numpy()[:active],
            "margin1_m": contacts.rigid_contact_margin1.numpy()[:active],
            "normal0_to1": contacts.rigid_contact_normal.numpy()[:active],
            "gap_m": distance.numpy()[:active],
            "point0_world_m": points[0].numpy()[:active],
            "point1_world_m": points[1].numpy()[:active],
            "point_id": contacts.rigid_contact_point_id.numpy()[:active],
            "triangle_id": contacts.rigid_contact_tids.numpy()[:active],
        }
        raw_rows = [
            {"buffer_index": i, **{name: values[i].tolist() for name, values in arrays.items()}}
            for i in range(active)
        ]
        record.update(
            actual_shape_ids=shape_ids,
            actual_body_ids=body_ids,
            actual_shape_types=native_types,
            shape_body=shape_bodies[shape_ids].tolist(),
            shape_scale=model.shape_scale.numpy()[shape_ids].tolist(),
            shape_transform=model.shape_transform.numpy()[shape_ids].tolist(),
            body_pose_xyz_xyzw=poses[body_ids].tolist(),
            body_mass_kg=model.body_mass.numpy()[body_ids].tolist(),
            contact_count=count,
            contact_capacity=contacts.rigid_contact_max,
            contact_counters=counters.tolist(),
            raw_rows=raw_rows,
            array_devices={
                "body_pose": str(state.body_q.device),
                "shape_type": str(model.shape_type.device),
                "count": str(contacts.rigid_contact_count.device),
                "normal": str(contacts.rigid_contact_normal.device),
                "distance": str(distance.device),
                "point0_world": str(points[0].device),
                "point1_world": str(points[1].device),
            },
        )
        checks = {
            "native_shape_types": native_types == expected_types,
            "two_real_owners": shape_bodies[shape_ids].tolist() == body_ids
            and len(set(body_ids)) == 2
            and min(body_ids) >= 0,
            "actual_device": all(d == str(device) for d in record["array_devices"].values()),
            "positive_body_mass": bool(np.all(model.body_mass.numpy()[body_ids] > 0)),
            "capacity_not_exhausted": 0 <= count < contacts.rigid_contact_max,
            "no_reduction_overflow": int(counters[2]) == 0,
            "intersection": count > 0,
        }
        requested = [s["body_position_m"] + s["body_xyzw"] for s in case["shapes"]]
        checks["authored_positions"] = bool(
            np.all(
                np.linalg.norm(poses[body_ids, :3] - np.asarray(requested)[:, :3], axis=1)
                <= TOLERANCES["length_m"]
            )
        )
        checks["authored_orientations"] = bool(
            np.all(
                np.linalg.norm(poses[body_ids, 3:] - np.asarray(requested)[:, 3:], axis=1)
                <= TOLERANCES["quaternion_l2"]
            )
        )
        record["checks"] = checks
        canonical = []
        for raw in raw_rows:
            native_ids = [int(raw["shape0"]), int(raw["shape1"])]
            if set(native_ids) != set(shape_ids):
                raise ValueError(f"unexpected native shape pair: {native_ids}")
            anchors, offsets, skins, margins = [], [], [], []
            anchor_errors = []
            for side, shape_id in enumerate(native_ids):
                pose = poses[int(shape_bodies[shape_id])]
                anchor = np.asarray(raw[f"point{side}_world_m"])
                local_anchor = raw[f"point{side}_body_m"]
                offset = _world_vector(wp, pose, raw[f"offset{side}_body_m"])
                reconstructed = _world_point(wp, pose, local_anchor)
                anchor_errors.append(float(np.linalg.norm(anchor - reconstructed)))
                anchors.append(anchor)
                offsets.append(offset)
                skins.append(anchor + offset)
                margins.append(float(raw[f"margin{side}_m"]))
            n = np.asarray(raw["normal0_to1"])
            if native_ids[0] != shape_ids[0]:
                anchors.reverse()
                offsets.reverse()
                skins.reverse()
                margins.reverse()
                n = -n
            gap = float(raw["gap_m"])
            skin_errors = [
                _surface_error(spec, skin) for spec, skin in zip(case["shapes"], skins, strict=True)
            ]
            patch_errors = [
                _patch_error(spec, skin, side)
                for side, (spec, skin) in enumerate(zip(case["shapes"], skins, strict=True))
            ]
            finite = np.isfinite(
                [gap, *n, *anchors[0], *anchors[1], *offsets[0], *offsets[1], *margins]
            ).all()
            canonical.append(
                {
                    "buffer_index": raw["buffer_index"],
                    "shape_ids_a_b": shape_ids,
                    "gap_m": gap,
                    "normal_a_to_b": n.tolist(),
                    "anchors_world_m": [a.tolist() for a in anchors],
                    "offsets_world_m": [v.tolist() for v in offsets],
                    "skins_world_m": [s.tolist() for s in skins],
                    "contact_margins_m": margins,
                    "finite": bool(finite),
                    "gap_error_m": abs(gap - case["expected_gap_m"]),
                    "normal_error_l2": float(np.linalg.norm(n - case["expected_normal_a_to_b"])),
                    "unit_normal_error": float(abs(np.linalg.norm(n) - 1.0)),
                    "anchor_reconstruction_error_m": max(anchor_errors),
                    "offset_thickness_error_m": float(
                        max(
                            np.linalg.norm(offsets[0] - margins[0] * n),
                            np.linalg.norm(offsets[1] + margins[1] * n),
                        )
                    ),
                    "skin_gap_identity_error_m": float(abs((skins[1] - skins[0]) @ n - gap)),
                    "surface_errors_m": skin_errors,
                    "support_patch_errors_m": patch_errors,
                    "surface_error_m": max(skin_errors),
                    "support_patch_error_m": max(patch_errors),
                }
            )
        record["canonical_rows"] = canonical
        checks["finite_rows"] = bool(canonical) and all(row["finite"] for row in canonical)
        gates = {
            "gap_error_m": TOLERANCES["length_m"],
            "normal_error_l2": TOLERANCES["normal_l2"],
            "unit_normal_error": TOLERANCES["unit_normal"],
            "anchor_reconstruction_error_m": TOLERANCES["length_m"],
            "offset_thickness_error_m": TOLERANCES["length_m"],
            "skin_gap_identity_error_m": TOLERANCES["length_m"],
            "surface_error_m": TOLERANCES["length_m"],
            "support_patch_error_m": TOLERANCES["length_m"],
        }
        record["maximum_errors"] = {key: _maximum(canonical, key) for key in gates}
        for key, tolerance in gates.items():
            checks[key] = bool(canonical) and all(row[key] <= tolerance for row in canonical)
        if case["id"] == "CC-SKEW":
            expected = np.asarray(((0.0, 0.0, 0.05), (0.0, 0.0, 0.04)))
            checks["unique_skew_skin_points"] = bool(canonical) and all(
                np.max(np.linalg.norm(np.asarray(row["skins_world_m"]) - expected, axis=1))
                <= TOLERANCES["length_m"]
                for row in canonical
            )


def observe_pose(case, *, device):
    """Measure one native pose, preserving readbacks before enforcing its fixed gates."""
    import newton
    import warp as wp

    record = {
        "id": case["id"],
        "group": case["group"],
        "inputs": case,
        "checks": {},
        "execution_error": None,
    }
    try:
        _read_pose(newton, wp, case, device, record)
    except Exception as exc:
        record["execution_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    record["failed_checks"] = [key for key, passed in record["checks"].items() if not passed]
    record["passed"] = (
        record["execution_error"] is None
        and bool(record["checks"])
        and all(record["checks"].values())
    )
    return record


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def main():
    import warp as wp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if version("newton") != "1.7.0":
        parser.error("this frozen comparison requires Newton 1.7.0")
    if args.out.exists():
        parser.error("output must be a new path; earlier results are never overwritten")
    if not args.out.parent.is_dir():
        parser.error("create the evidence directory before running this script")
    wp.config.kernel_cache_dir = str(Path(sys.prefix) / "warp-kernels")
    wp.init()
    device = wp.get_device(args.device)
    if str(device) != args.device or (args.device != "cpu" and not device.is_cuda):
        parser.error(f"requested device unavailable: {args.device}")
    results = [observe_pose(case, device=device) for case in frozen_cases()]
    report = {
        "schema_version": 1,
        "scope": "Four native static convex poses; no dynamics, CCD, area weights or tracking claim",
        "newton": version("newton"),
        "warp": version("warp-lang"),
        "numpy": version("numpy"),
        "python": sys.version,
        "executable": sys.executable,
        "device": str(device),
        "group_count": 3,
        "pose_count": 4,
        "shape_margin_m": 0.0,
        "shape_gap_m": 0.0,
        "tolerances": TOLERANCES,
        "results": results,
        "passed": len(results) == 4 and all(row["passed"] for row in results),
    }
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(_json_value(report), stream, allow_nan=False, indent=2)
        stream.write("\n")
    summary = {
        "device": str(device),
        "poses": len(results),
        "passed": report["passed"],
        "results": [
            {
                "id": row["id"],
                "contact_count": row.get("contact_count"),
                "passed": row["passed"],
                "failed_checks": row["failed_checks"],
                "execution_error": row["execution_error"],
            }
            for row in results
        ],
    }
    print(json.dumps(_json_value(summary), allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
