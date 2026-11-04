"""Pair the fixed and directional dispatch paths using the same native G2 observation contract."""

import argparse
import importlib.util
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import numpy as np
from benchmark_newton_free_fall import checked_window

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "scale"))
from g2_cross_engine_steps import STEPS, ApplicationOutput, KinematicFall, load_engine, setup_rows  # noqa: E402


def prior(path):
    name = "phymeo.backends._frozen_resident_world_step"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--baseline-module", type=Path, required=True)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--cache-root", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--first", choices=("baseline", "candidate"), required=True)
    a = cli.parse_args()
    if a.out.exists():
        cli.error("fresh output path required")
    frozen = np.load(a.reference, mmap_mode="r", allow_pickle=False)
    assert frozen.shape == (802, 128, 6) and frozen.dtype == np.float32
    (ph, tx), _, versions = load_engine("phymeo", "cuda:0", a.cache_root)
    archived = prior(a.baseline_module)
    apis = {
        "baseline": SimpleNamespace(
            FreeSphere=archived.FreeSphere, TyxWorldBatch=archived.TyxWorldBatch
        ),
        "candidate": ph,
    }
    order = (a.first, "baseline" if a.first == "candidate" else "candidate")
    keys, z, v = setup_rows(128, True)
    with ExitStack() as resources:
        lanes, checks = {}, {}
        for name in order:
            lane = KinematicFall((apis[name], tx), keys, z, v, "cuda:0", a.cache_root)
            resources.callback(lane.close)
            lane.binding()
            app = ApplicationOutput(lane.keys, keys, z, v)
            np.testing.assert_array_equal(
                app.read(0, *lane.read(0)).view(np.uint32), frozen[0].view(np.uint32)
            )
            lanes[name], checks[name] = lane, app
        tick, records = 0, []
        for index in range(26):
            steps = 1 if index == 0 else STEPS
            sequence = order if index % 2 == 0 else order[::-1]
            row = {"order": sequence}
            for name in sequence:
                end, values, sample = checked_window(lanes[name], checks[name], tick, steps)
                np.testing.assert_array_equal(
                    values.view(np.uint32), frozen[tick + 1 : end + 1].view(np.uint32)
                )
                row[name] = sample
            assert row["baseline"]["output_sha256"] == row["candidate"]["output_sha256"]
            assert row["baseline"]["counters"] == row["candidate"]["counters"]
            counts = row["candidate"]["counters"]
            assert counts["cuda_kernel_launches"] == counts["synchronizations"] == steps
            assert (
                counts["accepted_world_steps"] == steps * 128 and not counts["rejected_world_steps"]
            )
            assert counts["cuda_d2h_bytes"] == steps * 128 * 56 and counts["cuda_h2d_bytes"] == 0
            records.append(row)
            tick = end
        assert tick == 801
    summary = {}
    for name in order:
        samples = [r[name]["wall_ns"] for r in records[6:]]
        summary[name] = dict(
            p50_ms=median(samples) / 1e6,
            p95_ms=sorted(samples)[18] / 1e6,
            world_steps_s=81920 * 1e9 / sum(samples),
        )
    report = dict(
        versions=versions,
        first=a.first,
        baseline_module=str(a.baseline_module),
        reference=str(a.reference),
        qualification_and_warmup=records[:6],
        measurements=records[6:],
        summary=summary,
        candidate_wins=sum(
            r["candidate"]["wall_ns"] < r["baseline"]["wall_ns"] for r in records[6:]
        ),
    )
    with a.out.open("x", encoding="utf8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    print(summary, report["candidate_wins"])


if __name__ == "__main__":
    main()
