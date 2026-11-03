"""Measure exclusive stages of the original checked G2 snapshot path on actual CUDA."""

import argparse
import json
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "scale"))
from benchmark_newton_free_fall import checked_window  # noqa: E402
from g2_cross_engine_steps import STEPS, ApplicationOutput, load_engine, setup_rows  # noqa: E402


def instrument(stack, obj, method, name, times, calls):
    original = getattr(obj, method)

    def wrapped(*args, **kwargs):
        category = name(*args) if callable(name) else name
        start = perf_counter_ns()
        result = original(*args, **kwargs)
        times[category] += perf_counter_ns() - start
        calls[category] += 1
        return result

    stack.enter_context(patch.object(obj, method, wrapped))


def capture(adapter, app, tick):
    batch = adapter.work
    times, calls = Counter(), Counter()

    def read_name(buffer, category):
        if category == "snapshot":
            return "state_rows_read" if len(buffer.shape) == 2 else "ticks_read"
        return category + "_read"

    with ExitStack() as stack:
        for obj, m, name in [
            (batch, "step", "step"),
            (batch, "_launch", "physical_kernel_submit"),
            (batch, "_sync", "physical_sync"),
            (batch, "_read_status", "status_read_and_validation"),
            (batch, "_read", read_name),
            (batch, "snapshot", "snapshot"),
            (type(batch._bindings[0]), "checkpoint", "checkpoint_rows"),
            (adapter, "read", "public_to_position_velocity"),
            (app, "read", "common_reorder_check"),
        ]:
            instrument(stack, obj, m, name, times, calls)
        tick, _, obs = checked_window(adapter, app, tick, STEPS)
    result = {
        key: times[key]
        for key in (
            "physical_kernel_submit",
            "physical_sync",
            "status_read",
            "state_rows_read",
            "ticks_read",
            "checkpoint_rows",
            "common_reorder_check",
        )
    }
    result["status_validation"] = times["status_read_and_validation"] - times["status_read"]
    result["step_setup_accept_map"] = (
        times["step"]
        - times["physical_kernel_submit"]
        - times["physical_sync"]
        - times["status_read_and_validation"]
    )
    result["snapshot_keys_dictionary"] = (
        times["snapshot"]
        - times["state_rows_read"]
        - times["ticks_read"]
        - times["checkpoint_rows"]
    )
    result["extract_raw_position_velocity"] = (
        times["public_to_position_velocity"] - times["snapshot"]
    )
    result["window_store_and_dispatch"] = (
        obs["wall_ns"]
        - times["step"]
        - times["public_to_position_velocity"]
        - times["common_reorder_check"]
    )
    assert min(result.values()) >= 0 and sum(result.values()) == obs["wall_ns"]
    assert (
        calls["checkpoint_rows"] == STEPS * len(app.keys)
        and calls["state_rows_read"]
        == calls["ticks_read"]
        == calls["physical_kernel_submit"]
        == STEPS
    )
    return tick, dict(**obs, exclusive_ns=result, stage_calls=dict(calls))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worlds", choices=(1, 32), type=int, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error("provide a fresh receipt")
    a.cache_root.mkdir(parents=True, exist_ok=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    modules, factory, meta = load_engine("phymeo", "cuda:0", a.cache_root)
    keys, z, v = setup_rows(a.worlds)
    adapter = factory(modules, keys, z, v, "cuda:0", a.cache_root)
    try:
        app = ApplicationOutput(tuple(adapter.keys), keys, z, v)
        app.read(0, *adapter.read(0))
        tick = 0
        for _ in range(2):
            tick, _, _ = checked_window(adapter, app, tick, STEPS)
        rows = []
        for _ in range(5):
            tick, row = capture(adapter, app, tick)
            rows.append(row)
        names = rows[0]["exclusive_ns"]
        result = dict(
            worlds=a.worlds,
            backend=meta,
            steps_per_window=STEPS,
            prewarm_windows=2,
            instrumented_windows=rows,
            final_tick=tick,
            max_position_error_m=app.max_position_error,
            max_velocity_error_m_s=app.max_velocity_error,
            exclusive_stage_median_ms={
                n: median(x["exclusive_ns"][n] for x in rows) / 1e6 for n in names
            },
            exclusive_stage_share={
                n: sum(x["exclusive_ns"][n] for x in rows) / sum(x["wall_ns"] for x in rows)
                for n in names
            },
            median_wall_ms=median(x["wall_ns"] for x in rows) / 1e6,
            scope="Instrumented diagnostic only; nested times subtracted within each window. Per-call wrappers and counters perturb the original timings.",
        )
        a.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
        print(f"W{a.worlds}: {result['median_wall_ms']:.3f} ms / 32 audited steps")
        print(result["exclusive_stage_median_ms"])
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
