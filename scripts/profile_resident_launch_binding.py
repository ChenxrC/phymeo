"""Measure the resident CUDA launch bindings while returning every original G2 frame."""

import argparse
import json
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch

import numpy as np
from benchmark_newton_free_fall import checked_window

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "scale"))
from g2_cross_engine_steps import STEPS, ApplicationOutput, KinematicFall, load_engine, setup_rows  # noqa: E402


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--cache-root", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    a = cli.parse_args()
    if a.out.exists():
        cli.error("fresh output required")
    from tyx.backends.cuda_runtime import CompiledCudaKernel, PreparedCudaLaunch

    ref = np.load(a.reference, mmap_mode="r", allow_pickle=False)
    keys, z, v = setup_rows(128, True)
    modules, _, meta = load_engine("phymeo", "cuda:0", a.cache_root)
    adapter = KinematicFall(modules, keys, z, v, "cuda:0", a.cache_root)
    work = adapter.work
    app = ApplicationOutput(adapter.keys, keys, z, v)
    try:
        app.read(0, *adapter.read(0))
        tick, _, _ = checked_window(adapter, app, 0, STEPS)
        records = []
        for label in ("plain", "trace", "trace", "plain", "plain", "trace"):
            counts, times, directions = Counter(), Counter(), Counter()
            prepare = CompiledCudaKernel._prepare_reusable
            match = PreparedCudaLaunch._launch_if_matches
            launch = work._launch

            def rebuilding(compiled, args):
                source = 0 if args["state"] is work._states[0] else 1
                assert args["state"] is work._states[source] and args["tick"] is work._ticks[source]
                assert (
                    args["candidate"] is work._states[1 - source]
                    and args["next_tick"] is work._ticks[1 - source]
                )
                t = perf_counter_ns()
                result = prepare(compiled, args)
                times["rebuild"] += perf_counter_ns() - t
                assert result is not None
                counts["rebuilt"] += 1
                directions[f"{source}->{1 - source}"] += 1
                return result

            def matching(bound, args, **kwargs):
                t = perf_counter_ns()
                yes = match(bound, args, **kwargs)
                if not yes:
                    times["miss"] += perf_counter_ns() - t
                counts["hit" if yes else "miss"] += 1
                return yes

            def physical(*args):
                t = perf_counter_ns()
                try:
                    return launch(*args)
                finally:
                    times["all_physical_submissions"] += perf_counter_ns() - t
                    counts["submissions"] += 1

            with ExitStack() as hooks:
                if label == "trace":
                    hooks.enter_context(
                        patch.object(CompiledCudaKernel, "_prepare_reusable", rebuilding)
                    )
                    hooks.enter_context(
                        patch.object(PreparedCudaLaunch, "_launch_if_matches", matching)
                    )
                    hooks.enter_context(patch.object(work, "_launch", physical))
                old_tick = tick
                tick, states, obs = checked_window(adapter, app, old_tick, STEPS)
            np.testing.assert_array_equal(
                states.view(np.uint32), ref[old_tick + 1 : tick + 1].view(np.uint32)
            )
            assert (
                obs["counters"]["cuda_kernel_launches"]
                == obs["counters"]["synchronizations"]
                == STEPS
            )
            if label == "trace":
                assert counts["submissions"] == STEPS
                times["other_submission"] = (
                    times["all_physical_submissions"] - times["rebuild"] - times["miss"]
                )
                assert min(times.values()) >= 0
            records.append(
                dict(
                    label=label,
                    **obs,
                    timings_ns=dict(times),
                    binding_counts=dict(counts),
                    directions=dict(directions),
                )
            )
    finally:
        adapter.close()
    with a.out.open("x", encoding="utf8") as f:
        json.dump(
            dict(runtime=meta, records=records, reference=str(a.reference)),
            f,
            allow_nan=False,
            indent=2,
        )
    for row in records:
        if row["label"] == "trace":
            print(row["binding_counts"], row["directions"], row["timings_ns"], row["wall_ns"])


if __name__ == "__main__":
    main()
