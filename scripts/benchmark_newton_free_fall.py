"""G2 cross-engine CUDA timing with a checked sorted position/velocity frame at every step."""

import argparse
import json
import math
import sys
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "scale"))
from g2_cross_engine_steps import (
    DT,
    STEPS,
    ApplicationOutput,
    KinematicFall,
    load_engine,
    setup_rows,
)  # noqa: E402


def cache_state(root):
    files = [p for p in root.rglob("*") if p.is_file()] if root.is_dir() else []
    return dict(files=len(files), bytes=sum(p.stat().st_size for p in files))


def delta(before, after):
    return {k: v - before[k] for k, v in after.items()}


def checked_window(adapter, app, tick, steps):
    before = adapter.counters()
    start = perf_counter_ns()
    trajectory = np.empty((steps, len(app.keys), 6), np.float32)
    for index in range(steps):
        tick += 1
        trajectory[index] = app.read(tick, *adapter.advance(tick))
    wall = perf_counter_ns() - start
    counts = delta(before, adapter.counters())
    digest = sha256(
        np.asarray(app.keys, dtype="<i4").tobytes() + trajectory.astype("<f4", copy=False).tobytes()
    ).hexdigest()
    return (
        tick,
        trajectory,
        dict(
            wall_ns=wall, world_steps=steps * len(app.keys), output_sha256=digest, counters=counts
        ),
    )


def measure(a, modules, factory, count, asymmetric=False):
    keys, z, v = setup_rows(count, asymmetric)
    setup = perf_counter_ns()
    adapter = factory(modules, keys, z, v, a.device, a.cache_root)
    built = perf_counter_ns() - setup
    try:
        if a.engine == "newton":
            adapter.inspect_unused = a.phase == "qualify"
        audit = perf_counter_ns()
        physical = adapter.binding()
        app = ApplicationOutput(tuple(adapter.keys), keys, z, v)
        initial = app.read(0, *adapter.read(0))
        binding_ns = perf_counter_ns() - audit
        after_build = cache_state(a.cache_root)
        tick, first, cold = checked_window(adapter, app, 0, 1)
        result = dict(
            worlds=count,
            asymmetric=asymmetric,
            dt_s=DT,
            registration_order=keys,
            application_order=app.keys,
            binding=physical,
            setup_ns=built,
            binding_and_initial_read_ns=binding_ns,
            initial_positions_velocities=initial.tolist(),
            cold=cold,
            first_positions_velocities=first[-1].tolist(),
            cache_after_build=after_build,
            cache_after_first=cache_state(a.cache_root),
        )
        if a.phase == "qualify":
            tick, subsequent, remainder = checked_window(adapter, app, tick, STEPS - 1)
            states = np.concatenate((first, subsequent))
            result.update(trajectory=states.tolist(), remainder=remainder, qualification_tick=tick)
        else:
            warm = []
            for _ in range(a.warmup):
                tick, _, one = checked_window(adapter, app, tick, STEPS)
                warm.append(one["wall_ns"])
            observations = []
            for _ in range(a.samples):
                tick, states, one = checked_window(adapter, app, tick, STEPS)
                observations.append(one)
            walls = sorted(row["wall_ns"] for row in observations)
            confirmed = count * STEPS * a.samples
            total = {
                k: sum(row["counters"][k] for row in observations)
                for k in observations[0]["counters"]
            }
            assert total["solver_calls"] == STEPS * a.samples
            if a.engine == "phymeo":
                assert (
                    total["accepted_world_steps"] == confirmed
                    and total["rejected_world_steps"] == 0
                )
                if a.device == "cuda:0":
                    assert total["status_downloaded_bytes"] == confirmed * 4
                    if a.phymeo_observation == "kinematics":
                        assert total["kinematics_downloaded_bytes"] == confirmed * 52
                        assert total["snapshot_downloaded_bytes"] == 0
                    else:
                        assert total["snapshot_downloaded_bytes"] == confirmed * 52
                    assert (
                        total["cuda_kernel_launches"]
                        == total["synchronizations"]
                        == STEPS * a.samples
                    )
                    assert (
                        total["cuda_h2d_bytes"]
                        == total["cuda_d2d_bytes"]
                        == total["device_allocations"]
                        == 0
                    )
                    assert total["cuda_d2h_bytes"] == confirmed * 56
            elif a.device == "cuda:0":
                assert total["explicit_state_d2h_bytes"] == confirmed * 52
            result.update(
                warmup_window_ns=warm,
                measurements=observations,
                warm_totals=total,
                warm_p50_window_ns=median(walls),
                warm_p95_window_ns=walls[math.ceil(0.95 * len(walls)) - 1],
                verified_world_steps=confirmed,
                verified_world_steps_per_s=confirmed * 1e9 / sum(walls),
                median_window_world_steps_per_s=count * STEPS * 1e9 / median(walls),
            )
        result.update(
            final_tick=tick,
            last_positions_velocities=states[-1].tolist(),
            max_position_error_m=app.max_position_error,
            max_velocity_error_m_s=app.max_velocity_error,
            application_bytes_per_world_step=6 * np.dtype(np.float32).itemsize,
            cache_after_last=cache_state(a.cache_root),
        )
        result["extra_native_probe"] = adapter.events(app, tick + 1)
        return result
    finally:
        adapter.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=("phymeo", "newton"), required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    p.add_argument("--phase", choices=("qualify", "measure"), required=True)
    p.add_argument("--worlds", choices=(1, 32), type=int, default=1)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--expect-empty-cache", action="store_true")
    p.add_argument(
        "--phymeo-observation", choices=("checkpoint", "kinematics"), default="checkpoint"
    )
    a = p.parse_args()
    if a.warmup < 1 or a.samples < 2:
        p.error("warmup must be positive and at least two samples are required")
    if a.out.exists():
        p.error("provide a fresh receipt")
    initial_cache = cache_state(a.cache_root)
    if a.expect_empty_cache and initial_cache["files"]:
        raise ValueError("cold qualification requires an empty cache")
    a.cache_root.mkdir(parents=True, exist_ok=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter_ns()
    modules, factory, meta = load_engine(a.engine, a.device, a.cache_root)
    initialized = perf_counter_ns() - started
    if a.engine == "phymeo" and a.phymeo_observation == "kinematics":
        factory = KinematicFall
    report = dict(
        schema="phymeo-newton-g2-observed-free-fall/1",
        engine=a.engine,
        device=a.device,
        phase=a.phase,
        observation=a.phymeo_observation if a.engine == "phymeo" else "native_rigid_state",
        interpreter=sys.executable,
        python=sys.version.split()[0],
        numpy=version("numpy"),
        runtime=meta,
        startup_ns=initialized,
        cache_root=str(a.cache_root),
        cache_before_framework=initial_cache,
        cache_after_framework=cache_state(a.cache_root),
        warmup_windows=a.warmup,
        samples=a.samples,
        steps_per_window=STEPS,
        semantics="End-to-end actual physical step, synchronized public state read, same key order, copied FP32 XYZ and COM velocity trajectory and per-step finite/analytic checks. Hashing is outside the clock.",
        runs=[],
    )
    cases = ((1, False), (2, True)) if a.phase == "qualify" else ((a.worlds, False),)
    for count, asymmetric in cases:
        row = measure(a, modules, factory, count, asymmetric)
        report["runs"].append(row)
        a.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
        print(
            f"{a.engine} {a.device} W{count} {a.phase} tick {row['final_tick']}, max errors {row['max_position_error_m']}/{row['max_velocity_error_m_s']}, p50 ns {row.get('warm_p50_window_ns')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
