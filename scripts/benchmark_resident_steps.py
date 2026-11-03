"""Measure end-to-end accepted G2 world steps with identical TYX output boundaries."""

import argparse
import hashlib
import json
from contextlib import ExitStack
from dataclasses import asdict, is_dataclass
from importlib.metadata import version
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter_ns

import tyx

from phymeo import FreeSphere, TyxWorldBatch

STEPS = 32
DT = 1 / 128
WARMUP = 5
WINDOWS = 20
COUNTERS = (
    "accepted_world_steps",
    "rejected_world_steps",
    "device_allocations",
    "topology_uploaded_bytes",
    "control_uploaded_bytes",
    "status_downloaded_bytes",
    "summary_downloaded_bytes",
    "snapshot_downloaded_bytes",
    "submitted_kernels",
    "synchronizations",
)
NATIVE = ("cuda_kernel_launches", "cuda_h2d_bytes", "cuda_d2h_bytes", "cuda_d2d_bytes")


def counters(groups):
    local = [batch.stats() for batch in groups]
    backend = tyx.runtime_stats()
    return {k: sum(row[k] for row in local) for k in COUNTERS} | {
        k: getattr(backend, k) for k in NATIVE
    }


def encode(payload):
    def record(value):
        if is_dataclass(value):
            return asdict(value)
        raise TypeError(type(value).__name__)

    return json.dumps(
        payload, sort_keys=True, default=record, allow_nan=False, separators=(",", ":")
    ).encode("utf8")


def verify_physical(tick, snapshot):
    z = 2 - tick * (tick + 1) / 4096
    vz = -tick / 16
    for key, row in snapshot.items():
        assert row.tick == tick and row.world == key
        assert row.position == (0.0, 0.0, z) and row.velocity == (0.0, 0.0, vz), (key, tick, row)


def collect(groups, method):
    readings = {}
    for group in groups:
        readings.update(getattr(group, method)())
    return readings


def advance(groups, mode):
    # Include classification and materialisation of all requested public outputs in wall time.
    status = {}
    for group in groups:
        status.update(group.step(DT))
    if mode == "flags":
        return status, None
    return status, collect(groups, mode)


def measure(args, method, mode):
    sphere = FreeSphere(mass=2.0, radius=1 / 8, position=(0, 0, 2), gravity=(0, 0, -8))
    keys = [17 + 3 * i for i in reversed(range(args.worlds))]
    specs = [(key, sphere) for key in keys]
    packs = [specs] if method == "batched" else [[item] for item in specs]
    with ExitStack() as stack:
        started = perf_counter_ns()
        groups = [
            stack.enter_context(TyxWorldBatch(pack, "cuda", cache_root=args.out.parent / "cache"))
            for pack in packs
        ]
        setup_ns = perf_counter_ns() - started
        started = perf_counter_ns()
        flags, _ = advance(groups, mode)
        first_ns = perf_counter_ns() - started
        expected = dict.fromkeys(keys, True)
        assert flags == expected
        tick = 1
        for _ in range(WARMUP * STEPS):
            status, _ = advance(groups, mode)
            assert status == expected
            tick += 1
        verify_physical(tick, collect(groups, "snapshot"))
        baseline = counters(groups)
        durations, window_hashes = [], []
        for _ in range(WINDOWS):
            started = perf_counter_ns()
            frames = [advance(groups, mode) for _ in range(STEPS)]
            durations.append(perf_counter_ns() - started)
            # Validation, digesting and the extra end-of-series snapshot are outside the timer.
            digest = hashlib.sha256()
            for status, reading in frames:
                assert status == expected
                tick += 1
                if mode == "snapshot":
                    verify_physical(tick, reading)
                elif mode == "summary":
                    assert set(reading) == set(keys)
                    for value in reading.values():
                        assert value["bottom_z_m"] == 2 - tick * (tick + 1) / 4096 - 1 / 8
                        assert value["kinetic_energy_j"] == (tick / 16) ** 2
                digest.update(encode((status, reading)))
            window_hashes.append(digest.hexdigest())
            del frames
        end = counters(groups)
        delta = {k: end[k] - baseline[k] for k in baseline}
        accepted = WINDOWS * STEPS * args.worlds
        assert delta["accepted_world_steps"] == accepted and delta["rejected_world_steps"] == 0
        assert (
            delta["device_allocations"]
            == delta["topology_uploaded_bytes"]
            == delta["cuda_h2d_bytes"]
            == 0
        )
        assert delta["status_downloaded_bytes"] == accepted * 4
        if mode != "snapshot":
            assert delta["snapshot_downloaded_bytes"] == 0
        if mode != "summary":
            assert delta["summary_downloaded_bytes"] == 0
        assert delta["submitted_kernels"] == delta["cuda_kernel_launches"] > 0
        final = collect(groups, "snapshot")
        verify_physical(tick, final)
        return dict(
            method=method,
            output=mode,
            worlds=args.worlds,
            setup_ns=setup_ns,
            first_step_and_output_ns=first_ns,
            raw_ns=durations,
            measured=delta,
            active_owned_bytes=sum(x.stats()["device_owned_bytes"] for x in groups),
            window_sha256=window_hashes,
            final_sha256=hashlib.sha256(encode(final)).hexdigest(),
            warm_wall_p50_ns=median(durations),
            warm_wall_p95_ns=quantiles(durations, n=20, method="inclusive")[18],
            accepted_world_steps_per_s=accepted * 1e9 / sum(durations),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, choices=(1, 32), required=True)
    parser.add_argument("--order", choices=("batch-first", "serial-first"), required=True)
    parser.add_argument("--output", choices=("flags", "summary", "snapshot", "all"), default="all")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.parent.is_dir():
        parser.error("provide a fresh output inside an existing evidence directory")
    modes = ("flags", "summary", "snapshot") if args.output == "all" else (args.output,)
    methods = ("batched", "serial") if args.order == "batch-first" else ("serial", "batched")
    measurements = []
    for mode in modes:
        data = {method: measure(args, method, mode) for method in methods}
        batch, serial = data["batched"], data["serial"]
        assert (
            batch["window_sha256"] == serial["window_sha256"]
            and batch["final_sha256"] == serial["final_sha256"]
        )
        speedup = batch["accepted_world_steps_per_s"] / serial["accepted_world_steps_per_s"]
        measurements.append(dict(output=mode, batched_over_sequential=speedup, **data))
        print(f"{mode} W={args.worlds} {speedup:.4f}x", flush=True)
    with args.out.open("x", encoding="utf8") as stream:
        json.dump(
            dict(
                engine="Phymeo",
                tyx=version("tyx-compute"),
                backend="cuda",
                order=args.order,
                warmup_windows=WARMUP,
                windows=WINDOWS,
                steps_per_window=STEPS,
                step_s=DT,
                workload="G2 resident freely translating sphere, mass 2 kg, g=-8, all worlds advance",
                timing="End-to-end synchronous wall clock. Setup and first call shown separately. Same cache location within process. No isolated kernel time or allocator peak.",
                measurements=measurements,
            ),
            stream,
            indent=2,
            allow_nan=False,
        )


if __name__ == "__main__":
    main()
