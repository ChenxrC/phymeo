"""Measure G1 static collision queries. One engine and explicit device per process."""

import argparse
import csv
import gc
import json
import math
import os
import platform
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "scale"))
from cases import BASE_X_M, HOLDOUT_X_M, SCALES, PhymeoGroup, world_ids  # noqa: E402


def gpu_memory():
    """Whole-device NVML/SMI snapshot, separate from process or allocator peaks."""
    try:
        data = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        )
        rows = []
        for i, name, driver, total, free in csv.reader(
            data.stdout.splitlines(), skipinitialspace=True
        ):
            rows.append(
                dict(
                    index=int(i), name=name, driver=driver, total_mib=int(total), free_mib=int(free)
                )
            )
        return rows
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


class CapacityLimit(Exception):
    pass


def stats(groups, method):
    rows = [getattr(group, method)() for group in groups]
    return {
        key: sum(row[key] for row in rows) if type(value) is int else value
        for key, value in rows[0].items()
    }


def observed(groups, raw, moved):
    parts = [g.inspect(out, moved) for g, out in zip(groups, raw, strict=True)]
    keys = (
        "published_pair_rows",
        "same_world_rows",
        "cross_world_filtered_rows",
        "candidate_pairs",
    )
    snapshot = {key: None if parts[0][key] is None else sum(p[key] for p in parts) for key in keys}
    snapshot["contacts"] = sorted(
        (row for part in parts for row in part["contacts"]), key=lambda row: row["world"]
    )
    return snapshot


def run_configuration(args, factory, device, count, layout):
    ids = world_ids(count)
    partitions = [ids] if layout == "combined" else [(w,) for w in ids]
    runtime_stats = None
    if args.engine == "phymeo" and device == "cuda":
        from tyx import runtime_stats
    runtime_before = runtime_stats() if runtime_stats else None
    with ExitStack() as resources:
        snapshots = []
        monitor = args.device == "cuda:0"

        def check_memory(open_workspaces):
            rows = gpu_memory()
            snapshots.append(dict(open_workspaces=open_workspaces, devices=rows))
            if layout == "serial" and count >= 32:
                if len(rows) != 1 or rows[0]["free_mib"] < args.reserve_mib:
                    raise CapacityLimit(
                        f"W={count}, {open_workspaces} workspaces, reserve={args.reserve_mib} MiB, SMI={rows}"
                    )

        if monitor:
            check_memory(0)
        groups = []
        build_ns = 0
        for part in partitions:
            start = perf_counter_ns()
            g = factory(part, device, args.cache_root)
            build_ns += perf_counter_ns() - start
            resources.callback(g.close)
            groups.append(g)
            if monitor and (
                len(groups) == len(partitions) or (count >= 32 and len(groups) % 8 == 0)
            ):
                check_memory(len(groups))
        initial = stats(groups, "stats")
        runtime_build = asdict(runtime_stats() - runtime_before) if runtime_stats else None

        def query_all():
            return [g.query() for g in groups]

        def timed(window):
            before = stats(groups, "stats")
            start_runtime = runtime_stats() if runtime_stats else None
            t0 = perf_counter_ns()
            for _ in range(window):
                raw = query_all()
            wall = perf_counter_ns() - t0
            after = stats(groups, "stats")
            delta = {k: v - before[k] for k, v in after.items() if type(v) is int}
            actual = asdict(runtime_stats() - start_runtime) if runtime_stats else None
            return raw, dict(
                wall_ns=wall, query_sets=window, stat_deltas=delta, tyx_process_delta=actual
            )

        cold_raw, cold = timed(1)
        base = observed(groups, cold_raw, None)
        held = ids[count // 2]
        for g in groups:
            g.select(held)
        holdout = observed(groups, query_all(), held)
        assert holdout["contacts"] == [c for c in base["contacts"] if c["world"] != held]
        expanded = None
        if args.engine == "phymeo":
            expanded = next(g for g in groups if held in g.ids).near_control(held)
        for g in groups:
            g.select(None)
        restored = observed(groups, query_all(), None)
        assert restored == base
        for _ in range(args.warmup):
            assert observed(groups, query_all(), None) == base
        samples = []
        for _ in range(args.samples):
            raw, sample = timed(args.window)
            assert observed(groups, raw, None) == base
            samples.append(sample)
        if args.engine == "phymeo" and device == "cuda":
            for run in [cold, *samples]:
                delta = run["stat_deltas"]
                assert run["tyx_process_delta"]["cuda_kernel_launches"] > 0
                assert all(
                    delta[k] > 0
                    for k in (
                        "completed_launches",
                        "synchronizations",
                        "uploaded_bytes",
                        "downloaded_bytes",
                    )
                ), delta
        durations = sorted(row["wall_ns"] for row in samples)
        return dict(
            world_count=count,
            world_ids=ids,
            moved_world_id=held,
            layout=layout,
            workspace_count=len(groups),
            shapes=4 * count,
            possible_world_local_pairs=6 * count,
            storage=stats(groups, "storage"),
            stats_after_build=initial,
            stats_at_close=stats(groups, "stats"),
            binding=[g.binding() for g in groups],
            tyx_process_setup_delta=runtime_build,
            gpu_memory_during_build=snapshots,
            setup_ns=build_ns,
            cold=cold,
            basestate=base,
            holdout=holdout,
            restored_equals_base=True,
            expanded_phymeo_positive_gap=expanded,
            warm_samples=samples,
            warm_p50_window_ns=median(durations),
            warm_p95_window_ns=durations[math.ceil(0.95 * len(durations)) - 1],
        )


def save(path, data):
    with NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False, encoding="utf-8"
    ) as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(f.name, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=("phymeo", "newton"), required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    p.add_argument("--worlds", type=int, nargs="+", choices=SCALES, default=list(SCALES))
    p.add_argument(
        "--layouts", nargs="+", choices=("serial", "combined"), default=["serial", "combined"]
    )
    p.add_argument("--reserve-mib", type=int, default=512)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--window", type=int, default=1)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if not __debug__ or min(a.samples, a.window) < 1 or a.warmup < 0:
        p.error(
            "enable assertions and select positive sample/window counts with nonnegative warmup"
        )
    if len(set(a.worlds)) != len(a.worlds) or len(set(a.layouts)) != len(a.layouts):
        p.error("duplicate world count or layout")
    if a.out.exists():
        p.error("choose a new report path")
    cache_was_populated = a.cache_root.is_dir() and any(a.cache_root.iterdir())
    a.cache_root.mkdir(exist_ok=True, parents=True)
    start = perf_counter_ns()
    metadata = dict(
        python=sys.version,
        interpreter=sys.executable,
        platform=platform.platform(),
        numpy=version("numpy"),
    )
    if a.engine == "newton":
        import newton
        import warp as wp
        from newton_g1 import NewtonGroup

        if version("newton") != "1.7.0":
            p.error("this G1 reference is frozen for Newton 1.7.0")
        wp.config.kernel_cache_dir = str(a.cache_root.resolve())
        wp.init()
        selected = wp.get_device(a.device)
        assert str(selected) == a.device and (a.device == "cpu" or selected.is_cuda)
        wp.set_device(selected)
        metadata.update(
            newton=version("newton"),
            warp=version("warp-lang"),
            module=newton.__file__,
            device_name=selected.name,
        )
        factory, device = NewtonGroup, a.device
        semantics = (
            "CollisionPipeline.collide, eval_rigid_contact_kinematics, explicit device sync and active "
            "candidate/contact array reads. Report output has active pairs only. host_read_bytes counts "
            "the source payload observed by this runner, including CPU views; it is not measured PCIe traffic."
        )
    else:
        import phymeo as ph

        metadata.update(phymeo=version("phymeo"), module=ph.__file__)
        if a.device == "cuda:0":
            metadata["tyx"] = version("tyx-compute")
        factory = PhymeoGroup
        device = {"cpu": "host", "cuda:0": "cuda"}[a.device]
        semantics = (
            "Full CollisionDetection pair table. The CUDA backend reduces AABBs and classifies pairs on "
            "the pinned tyx runtime, synchronizes and downloads bounds and flags; envelope construction, "
            "static sphere witnesses and full-row publication execute on HOST."
        )
    init_ns = perf_counter_ns() - start
    report = dict(
        schema="phymeo-g1-world-cost/1",
        engine=a.engine,
        device=a.device,
        runtime=metadata,
        semantics=semantics,
        framework_init_ns=init_ns,
        prior_cache_entries_present=cache_was_populated,
        reserve_mib=a.reserve_mib,
        cache_root=str(a.cache_root.resolve()),
        setup="fresh model/workspace per layout and scale",
        cold_definition="first query on that workspace; compiler/disk/process caches may be warm",
        timed="static queries per window, including mandatory API readback, excluding inspection, state edits and JSON",
        aggregate_stats="sum of numeric fields across retained workspaces; deltas are differences of these sums",
        peak_device_memory_bytes=None,
        device_event_ns=None,
        dynamics_steps=0,
        base_x_m=BASE_X_M,
        moved_second_x_m=HOLDOUT_X_M,
        warmup=a.warmup,
        samples=a.samples,
        window=a.window,
        requested_worlds=a.worlds,
        requested_layouts=a.layouts,
        configurations=[],
    )
    with a.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream)
    for count in a.worlds:
        reference = None
        for layout in a.layouts:
            try:
                record = run_configuration(a, factory, device, count, layout)
                key = (record["basestate"]["contacts"], record["holdout"]["contacts"])
                if reference is not None:
                    assert key == reference, "combined and serial material witnesses differ"
                reference = key
                report["configurations"].append(record)
                save(a.out, report)
                print(
                    f"G1 W={count} {layout} passed, warm p50={record['warm_p50_window_ns'] / 1e6:.3f} ms",
                    flush=True,
                )
            except CapacityLimit as error:
                report.setdefault("skipped_for_capacity", []).append(
                    dict(worlds=count, layout=layout, reason=str(error))
                )
                save(a.out, report)
            except Exception as error:
                report["failed_configuration"] = dict(
                    worlds=count, layout=layout, error=f"{type(error).__name__}: {error}"
                )
                save(a.out, report)
                raise
            gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
