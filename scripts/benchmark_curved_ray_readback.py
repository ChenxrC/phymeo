"""Optional frozen 4,171-beam CUDA readback benchmark; no product or input tuning.

Run each engine in its own environment and process. See the companion benchmark
protocol for the three fixed pair orders and the required exclusive execution slot.
"""

import argparse
import hashlib
import json
import os
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import distribution, version
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from zipfile import ZipFile

import numpy as np

WIDTH, HEIGHT = 97, 43
COUNT = WIDTH * HEIGHT
WARM, SAMPLES = 12, 21
KINDS = ("ellipsoid", "cylinder", "cone")
DEPTH_GATE, NORMAL_GATE = 1e-4, 1e-3
NATIVE_SHA = "bdc40eb0dffeb3b196421c540fb3d3a3e58ef0def59651fba587bf331af079c4"
WHEEL_SHA = "759d47efaefad1012a6e8302e4ecaa71a2cfe65f20cb74e3a8c4c5364beb50c6"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def frozen_inputs():
    """Same authoring values as G7; one common FP32 asset and ray bundle."""
    specs = (
        ((-1.08, 0.07, -0.4), (0.35, 0.19, 0.51), (0, 1, 1), 0.7),
        ((0.03, -0.09, -0.22), (0.31, 0.49, 0), (1, 2, -1), 1.0),
        ((1.06, 0.035, -0.35), (0.41, 0.56, 0), (1, -2, 0.3), 1.2),
    )
    centers, dimensions, rotations = [], [], []
    for center, size, axis, angle in specs:
        unit = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        centers.append(center)
        dimensions.append(size)
        rotations.append(np.r_[np.sin(angle / 2) * unit, np.cos(angle / 2)])
    x, y = np.meshgrid(np.linspace(-1.9, 1.9, WIDTH), np.linspace(-0.72, 0.72, HEIGHT))
    origins = np.stack((x, y + 0.06 * x, 2 + 0.03 * x), axis=-1).astype(np.float32)
    directions = np.stack((0.027 * y, 0.019 * x, -np.ones_like(x)), axis=-1)
    directions = (directions / np.linalg.norm(directions, axis=-1, keepdims=True)).astype(
        np.float32
    )
    return dict(
        centers_m=np.asarray(centers, np.float32),
        dimensions_m=np.asarray(dimensions, np.float32),
        quaternion_xyzw=np.asarray(rotations, np.float32),
        origins_m=origins.reshape(COUNT, 3),
        directions=directions.reshape(COUNT, 3),
    )


def input_digest(data):
    result = hashlib.sha256()
    for key, array in sorted(data.items()):
        result.update(key.encode("ascii"))
        result.update(array.dtype.str.encode("ascii"))
        result.update(str(array.shape).encode("ascii"))
        result.update(array.tobytes())
    return result.hexdigest()


def unit_rotations(data):
    # CurvedSolid's public contract requires a unit quaternion within 1e-12.
    # Lift the COMMON FP32 asset to a legal unit quaternion once for BOTH adapters.
    q = data["quaternion_xyzw"].astype(float)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def reference(path):
    require(digest(path) == NATIVE_SHA, "this is not the frozen G7 native capture")
    raw = json.loads(path.read_text(encoding="utf8"))
    labels = np.asarray(raw["shape_index"], np.int64).reshape(COUNT)
    require(raw["device"] == "cuda:0", "the frozen native record must be CUDA")
    require(
        np.array_equal(labels, np.asarray(raw["phymeo_shape_index"]).reshape(COUNT)),
        "frozen native/HOST masks disagree",
    )
    require(
        [int(np.count_nonzero(labels == i)) for i in range(3)] == [179, 491, 398],
        "frozen hit counts changed",
    )
    return (
        labels,
        np.asarray(raw["native_depth_m"], float).reshape(COUNT),
        np.asarray(raw["native_normals"], float).reshape(COUNT, 3),
    )


def common_frame(raw):
    """Common owned host fields; preserve each engine's raw arrays separately."""
    ids, distance, normals = raw
    labels = np.asarray(ids).astype(np.int64).reshape(COUNT)
    labels[labels == 0xFFFFFFFF] = -1
    meters = np.asarray(distance).astype(np.float64).reshape(COUNT)
    vectors = np.asarray(normals).astype(np.float64).reshape(COUNT, 3)
    meters[labels < 0] = np.inf
    return labels, meters, vectors


def gate(frame, expected):
    labels, distance, normals = frame
    ids, depth, normal = expected
    match = (labels >= 0) & (labels == ids)
    miss = labels == -1
    finite = np.isfinite(distance[match]).all() and np.isfinite(normals).all()
    errors = {}
    for i, name in enumerate(KINDS):
        at = match & (ids == i)
        dd = np.abs(distance[at] - depth[at])
        dn = np.max(np.abs(normals[at] - normal[at]), axis=1)
        errors[name] = dict(
            hits=int(at.sum()),
            maximum_depth_error_m=float(np.max(dd, initial=0)) if finite else None,
            maximum_normal_component_error=float(np.max(dn, initial=0)) if finite else None,
            depth_failures=int(np.count_nonzero(dd >= DEPTH_GATE)),
            normal_failures=int(np.count_nonzero(dn >= NORMAL_GATE)),
        )
    invalid_labels = int(np.count_nonzero((labels < -1) | (labels > 2)))
    mismatches = int(np.count_nonzero(labels != ids))
    miss_valid = bool(np.isposinf(distance[miss]).all() and not np.any(normals[miss]))
    passed = bool(
        finite
        and miss_valid
        and not invalid_labels
        and not mismatches
        and all(not s["depth_failures"] and not s["normal_failures"] for s in errors.values())
    )
    return dict(
        passed=passed,
        id_mismatches=mismatches,
        invalid_labels=invalid_labels,
        miss_fields_valid=miss_valid,
        finite_hit_fields=bool(finite),
        shapes=errors,
    )


def verified_installed_wheel(args):
    site = args.wheel_site.resolve(strict=True)
    require(sys.flags.isolated, "use python -I to isolate the installed package")
    require(digest(args.wheel) == WHEEL_SHA, "wrong frozen wheel")
    require(not (Path.cwd() / "src" / "phymeo").exists(), "run outside a source checkout")
    sys.path.insert(0, str(site))
    import phymeo as ph

    installed = distribution("phymeo")
    require(Path(ph.__file__).resolve().is_relative_to(site), "source Phymeo was imported")
    require(Path(installed.locate_file("")).resolve() == site, "wrong installed distribution")
    direct = json.loads(installed.read_text("direct_url.json") or "{}")
    archive = direct.get("archive_info", {})
    recorded = archive.get("hashes", {}).get("sha256")
    if recorded is None:
        algorithm, _, recorded = archive.get("hash", "").partition("=")
        require(algorithm == "sha256", "missing installed wheel digest")
    require(recorded.lower() == WHEEL_SHA, "installed wheel origin differs")
    return ph, dict(
        site=str(site),
        wheel=str(args.wheel.resolve()),
        sha256=WHEEL_SHA,
        installed_from=direct.get("url"),
        module=str(Path(ph.__file__).resolve()),
    )


def verify_loaded_files(site, wheel):
    paths = sorted(
        {
            Path(module.__file__).resolve()
            for name, module in tuple(sys.modules.items())
            if (name == "phymeo" or name.startswith("phymeo."))
            and getattr(module, "__file__", None)
        }
    )
    with ZipFile(wheel) as archive:
        for path in paths:
            require(path.is_relative_to(site), "a Phymeo module escaped the installed site")
            require(
                path.read_bytes() == archive.read(path.relative_to(site).as_posix()),
                "installed code differs from the frozen wheel",
            )
    return [str(path) for path in paths]


class Phymeo:
    def __init__(self, args, data):
        ph, origin = verified_installed_wheel(args)
        import tyx as tx

        from phymeo.backends.tyx_raycast import TyxRaycaster

        self.tx = tx
        shapes, self.poses = [], {}
        q = unit_rotations(data)
        for i, kind in enumerate(KINDS):
            key = ph.CollisionKey(0, i, 0)
            size = data["dimensions_m"][i, : 3 if i == 0 else 2]
            shapes.append(
                ph.CollisionShape(
                    key, kind, (0,), curved=ph.CurvedSolid(kind, tuple(size), tuple(q[i]))
                )
            )
            self.poses[key] = ph.CollisionMotion([data["centers_m"][i]])
        self.model = ph.CollisionModel(shapes)
        self.origin, self.rays = data["origins_m"], data["directions"]
        self.operator = TyxRaycaster("cuda", cache_root=args.out / "cache")
        self.meta = dict(
            engine="phymeo",
            package=version("phymeo"),
            runtime=version("tyx"),
            device="cuda:0",
            wheel=origin,
            precision="FP32 compute; public i32 labels and float64 metric/normal arrays",
            synchronization="query synchronizes and downloads both output buffers internally",
            scene_builds=1,
            operator_builds=1,
            input_quaternion=q.tolist(),
            io_timing="not separable through the public query; included in call wall clock",
        )

    def sample(self):
        before = self.tx.runtime_stats()
        start = perf_counter_ns()
        hit = self.operator.query(self.model, self.poses, self.origin, self.rays)
        read = perf_counter_ns()
        raw = (hit.shape_index, hit.distance_m, hit.world_normals)
        frame = common_frame(raw)
        end = perf_counter_ns()
        counts = asdict(self.tx.runtime_stats() - before)
        require(counts["cuda_kernel_launches"] == 1, "expected one actual TYX CUDA launch")
        require(counts["cuda_d2h_transfers"] == 2, "expected both TYX output downloads")
        return (
            raw,
            frame,
            dict(
                public_to_raw_read_ns=read - start,
                public_to_common_frame_ns=end - start,
                public_call_ns=read - start,
                common_conversion_ns=end - read,
                io=counts,
            ),
        )

    def close(self):
        self.operator.close()


class Newton:
    def __init__(self, args, data):
        import newton
        import warp as wp
        from newton.sensors import SensorCamera

        require(version("newton") == "1.7.0", "only the frozen Newton version is eligible")
        self.wp = wp
        wp.config.kernel_cache_dir = str(args.out / "cache")
        wp.init()
        self.device = wp.get_device("cuda:0")
        require(self.device.is_cuda, "Newton is not on CUDA")
        builder = newton.ModelBuilder()
        q = unit_rotations(data)
        for i, kind in enumerate(KINDS):
            center, size = data["centers_m"][i], data["dimensions_m"][i]
            xform = wp.transform(wp.vec3(*center), wp.quat(*q[i]))
            if kind == "ellipsoid":
                index = builder.add_shape_ellipsoid(
                    -1, rx=float(size[0]), ry=float(size[1]), rz=float(size[2]), xform=xform
                )
            else:
                add = builder.add_shape_cylinder if kind == "cylinder" else builder.add_shape_cone
                index = add(-1, radius=float(size[0]), half_height=float(size[1]), xform=xform)
            require(index == i, "native shape registration order changed")
        self.model = builder.finalize(device=self.device)
        self.state = self.model.state()
        self.sensor = SensorCamera(self.model, load_textures=False)
        beams = np.stack(
            (
                data["origins_m"].reshape(HEIGHT, WIDTH, 3),
                data["directions"].reshape(HEIGHT, WIDTH, 3),
            ),
            axis=-2,
        )[:, :, None, :, :]
        self.rays = wp.array(beams, dtype=wp.vec3f, device=self.device)
        self.xform = wp.array([wp.transform_identity()], dtype=wp.transformf, device=self.device)
        self.ids = self.sensor.allocate_image_shape_index(1, WIDTH, HEIGHT)
        self.depth = self.sensor.allocate_image_depth(1, WIDTH, HEIGHT)
        self.normals = self.sensor.allocate_image_normal(1, WIDTH, HEIGHT)
        # Static scene: refit ONCE during setup, never inject rebuilds into timed calls.
        self.model.bvh_refit_shapes(self.state)
        wp.synchronize_device(self.device)
        self.meta = dict(
            engine="newton",
            package=version("newton"),
            runtime=version("warp-lang"),
            module=str(Path(newton.__file__).resolve()),
            device=str(self.device),
            device_name=self.device.name,
            precision="native uint32 labels, float32 metric depth and vec3f world normals",
            synchronization="update, one device synchronization, then three numpy downloads",
            scene_builds=1,
            sensor_builds=1,
            output_buffer_sets=1,
            setup_bvh_refits=1,
            input_quaternion=q.tolist(),
            stored_shape_transform=self.model.shape_transform.numpy().tolist(),
            setup_beam_and_camera_h2d_bytes=int(beams.nbytes + 7 * 4),
            io_timing="explicit update/compute wait/three synchronous array reads",
        )

    def sample(self):
        start = perf_counter_ns()
        self.sensor.update(
            self.state,
            self.xform,
            self.rays,
            depth_image=self.depth,
            shape_index_image=self.ids,
            normal_image=self.normals,
        )
        submitted = perf_counter_ns()
        self.wp.synchronize_device(self.device)
        synchronized = perf_counter_ns()
        ids = self.ids.numpy()
        ids_read = perf_counter_ns()
        depth = self.depth.numpy()
        depth_read = perf_counter_ns()
        normals = self.normals.numpy()
        read = perf_counter_ns()
        raw = (ids, depth, normals)
        frame = common_frame(raw)
        end = perf_counter_ns()
        return (
            raw,
            frame,
            dict(
                public_to_raw_read_ns=read - start,
                public_to_common_frame_ns=end - start,
                public_call_ns=submitted - start,
                compute_wait_ns=synchronized - submitted,
                d2h_ids_ns=ids_read - synchronized,
                d2h_depth_ns=depth_read - ids_read,
                d2h_normals_ns=read - depth_read,
                common_conversion_ns=end - read,
                io=dict(
                    requested_d2h_calls=3,
                    requested_d2h_bytes=int(ids.nbytes + depth.nbytes + normals.nbytes),
                    caller_h2d_calls=0,
                    caller_h2d_bytes=0,
                    scope="explicit caller arrays; not a profiler of hidden Warp driver operations",
                ),
            ),
        )

    def close(self):
        self.wp.synchronize_device(self.device)


def percentiles(samples, field):
    values = [record[field] for record in samples]
    return dict(
        median_ns=median(values),
        minimum_ns=min(values),
        maximum_ns=max(values),
        p10_ns=float(np.percentile(values, 10)),
        p90_ns=float(np.percentile(values, 90)),
    )


def measure(args):
    require(sys.flags.isolated and __debug__, "run with -I and without -O")
    require(not args.out.exists(), "choose a fresh receipt directory")
    args.out.mkdir(parents=True)
    with np.load(args.inputs, allow_pickle=False) as source:
        data = {name: source[name] for name in source.files}
    template = frozen_inputs()
    require(set(data) == set(template), "unexpected input asset fields")
    for key, array in data.items():
        require(array.dtype == np.float32, "all common authored inputs must be FP32")
        require(array.shape == template[key].shape, "wrong frozen input array shape")
        require(np.array_equal(array, template[key]), "input differs from the frozen authoring")
    expected = reference(args.native)
    with (args.out / "inputs.npz").open("xb") as stream:
        np.savez(stream, **data)
    report = dict(
        protocol="g7-4171-public-readback-v1",
        session=args.session,
        engine=args.engine,
        started_utc=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(),
        executable=sys.executable,
        python=sys.version,
        cwd=str(Path.cwd()),
        driver_sha256=digest(__file__),
        numpy=version("numpy"),
        device="cuda:0",
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        input_sha256=input_digest(data),
        input_file_sha256=digest(args.inputs),
        frozen_native_sha256=NATIVE_SHA,
        depth_gate_m=DEPTH_GATE,
        normal_component_gate=NORMAL_GATE,
        counts=dict(cold=1, warm=WARM, formal=SAMPLES),
        comparison="same original error gates; not bitwise identical native precision",
        rows=[],
        success=False,
    )
    engine = None
    staged = []
    try:
        start = perf_counter_ns()
        engine = (Phymeo if args.engine == "phymeo" else Newton)(args, data)
        report["setup_ns"] = perf_counter_ns() - start
        report["implementation"] = engine.meta
        phases = [("cold", 0)] + [("warm", i) for i in range(WARM)]
        phases += [("formal", i) for i in range(SAMPLES)]
        for phase, ordinal in phases:
            raw, frame, timing = engine.sample()
            check_start = perf_counter_ns()
            checked = gate(frame, expected)
            validation_ns = perf_counter_ns() - check_start
            name = f"{phase}-{ordinal:02d}.npz"
            # Every call returns owned host arrays. Retain them without disk traffic
            # between timed calls; save even the first failed gate in the finally block.
            staged.append((raw, frame))
            report["rows"].append(
                dict(
                    phase=phase,
                    ordinal=ordinal,
                    arrays=name,
                    validation_ns=validation_ns,
                    gate=checked,
                    **timing,
                )
            )
            require(checked["passed"], f"{phase} {ordinal}: frozen per-ray gate failed")
        if args.engine == "phymeo":
            report["verified_installed_modules"] = verify_loaded_files(
                args.wheel_site.resolve(), args.wheel
            )
        formal = [r for r in report["rows"] if r["phase"] == "formal"]
        report["statistics"] = {
            field: percentiles(formal, field)
            for field in ("public_to_raw_read_ns", "public_to_common_frame_ns")
        }
        report["success"] = True
    except BaseException as error:
        report["failure"] = dict(type=type(error).__name__, message=str(error))
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        try:
            if engine is not None:
                engine.close()
        finally:
            for row, (raw, frame) in zip(report["rows"], staged, strict=True):
                save_start = perf_counter_ns()
                with (args.out / row["arrays"]).open("xb") as stream:
                    np.savez(
                        stream,
                        raw_shape_index=raw[0],
                        raw_distance_m=raw[1],
                        raw_world_normals=raw[2],
                        shape_index=frame[0],
                        distance_m=frame[1],
                        world_normals=frame[2],
                    )
                row["file_io_ns"] = perf_counter_ns() - save_start
            report["ended_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(args.out / "receipt.json", report)
    print(
        json.dumps(
            dict(
                engine=args.engine,
                session=args.session,
                passed=report["success"],
                statistics=report["statistics"],
            )
        )
    )


def compare(args):
    """Offline paired analysis, including every saved formal ray, not only medians."""
    require(not args.out.exists(), "choose a fresh comparison result")
    records = {}
    for path in args.receipts:
        raw = json.loads(path.read_text(encoding="utf8"))
        key = (raw["session"], raw["engine"])
        require(key not in records, "duplicate paired session")
        require(raw["success"], f"unqualified run: {path}")
        require(raw["protocol"] == "g7-4171-public-readback-v1", "mixed protocols")
        require(raw["counts"] == dict(cold=1, warm=WARM, formal=SAMPLES), "mixed sample counts")
        require(raw["frozen_native_sha256"] == NATIVE_SHA, "mixed frozen references")
        require(raw["input_sha256"] == input_digest(frozen_inputs()), "mixed FP32 inputs")
        require(raw["device"] == "cuda:0", "mixed devices")
        require(raw["depth_gate_m"] == DEPTH_GATE, "changed depth gate")
        require(raw["normal_component_gate"] == NORMAL_GATE, "changed normal gate")
        records[key] = (path, raw)
    require(
        set(records) == {(i, e) for i in (1, 2, 3) for e in ("phymeo", "newton")},
        "need three complete independent paired sessions",
    )
    require(len({r[1]["input_file_sha256"] for r in records.values()}) == 1, "mixed input files")
    require(len({r[1]["driver_sha256"] for r in records.values()}) == 1, "mixed driver sources")
    require(
        len({r[1]["cuda_visible_devices"] for r in records.values()}) == 1,
        "different CUDA device visibility",
    )
    reference_frames = reference(args.native)
    pairs, cross_gates = [], []
    timeline = []
    for session in (1, 2, 3):
        phy_path, phy = records[session, "phymeo"]
        new_path, new = records[session, "newton"]
        p_rows = [r for r in phy["rows"] if r["phase"] == "formal"]
        n_rows = [r for r in new["rows"] if r["phase"] == "formal"]
        require(len(p_rows) == len(n_rows) == SAMPLES, "missing formal samples")
        for ordinal, (p_row, n_row) in enumerate(zip(p_rows, n_rows, strict=True)):
            require(p_row["ordinal"] == n_row["ordinal"] == ordinal, "reordered samples")
            frames = []
            for path, row in ((phy_path, p_row), (new_path, n_row)):
                with np.load(path.parent / row["arrays"], allow_pickle=False) as array:
                    frame = tuple(array[k] for k in ("shape_index", "distance_m", "world_normals"))
                require(
                    gate(frame, reference_frames)["passed"], "saved sample failed original gate"
                )
                frames.append(frame)
            check = gate(frames[0], frames[1])
            cross_gates.append(dict(session=session, ordinal=ordinal, **check))
        pair = dict(session=session, receipts=[str(phy_path), str(new_path)])
        for field in ("public_to_raw_read_ns", "public_to_common_frame_ns"):
            p = median(r[field] for r in p_rows)
            n = median(r[field] for r in n_rows)
            pair[field] = dict(
                phymeo_median_ns=p,
                newton_median_ns=n,
                phymeo_over_newton_throughput=n / p,
                ratio_definition="Newton elapsed / Phymeo elapsed for the same 4171 beams",
            )
        pairs.append(pair)
        ordered = (new, phy) if session in (1, 3) else (phy, new)
        require(ordered[0]["ended_utc"] <= ordered[1]["started_utc"], "pair order or overlap")
        timeline.extend(ordered)
    require(
        all(a["ended_utc"] <= b["started_utc"] for a, b in zip(timeline, timeline[1:])),
        "sessions overlapped or prescribed independent pair order changed",
    )
    qualified = all(row["passed"] for row in cross_gates)
    output = dict(
        qualified=qualified,
        comparison="same original ID/depth/normal gates, not bitwise output equivalence",
        pairs=pairs,
        all_formal_cross_engine_gates=cross_gates,
        reporting_rule="no speed claim unless all frozen AND paired gates pass",
    )
    write_json(args.out, output)
    require(qualified, "paired per-ray error gates failed; retain times but make no speed claim")
    print(json.dumps(dict(qualified=qualified, pairs=pairs)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="command", required=True)
    prepare = modes.add_parser("prepare")
    prepare.add_argument("--out", type=Path, required=True)
    run = modes.add_parser("measure")
    run.add_argument("--engine", choices=("phymeo", "newton"), required=True)
    run.add_argument("--session", type=int, choices=(1, 2, 3), required=True)
    run.add_argument("--native", type=Path, required=True)
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--wheel-site", type=Path)
    run.add_argument("--wheel", type=Path)
    collect = modes.add_parser("compare")
    collect.add_argument("--native", type=Path, required=True)
    collect.add_argument("--out", type=Path, required=True)
    collect.add_argument("receipts", type=Path, nargs=6)
    args = cli.parse_args()
    if args.command == "prepare":
        with args.out.open("xb") as stream:
            np.savez(stream, **frozen_inputs())
    elif args.command == "measure":
        if args.engine == "phymeo" and (args.wheel is None or args.wheel_site is None):
            cli.error("installed Phymeo requires --wheel and --wheel-site")
        measure(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
