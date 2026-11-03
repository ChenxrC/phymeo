"""Matched mixed physical windows with active cloth/link loads and all checkpoint fields."""

import argparse
import json
import sys
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "scale"))
from mixed_physical_cases import WINDOW, H, MixedRun, population, signature  # noqa: E402


def window(runner, reset):
    restored = 0
    if reset:
        start = perf_counter_ns()
        runner.reset()
        restored = perf_counter_ns() - start
    before = runner.counts()
    start = perf_counter_ns()
    frames = [runner.advance(index) for index in range(WINDOW)]
    wall = perf_counter_ns() - start
    counts = {k: v - before[k] for k, v in runner.counts().items()}
    n = len(runner.keys)
    nr = 1 if runner.group is not None else n // 2
    assert (
        counts["resident_submitted_kernels"] == nr * WINDOW
        and counts["material_completed_kernels"] == 6 * nr * WINDOW
    )
    assert counts["cuda_kernel_launches"] == 7 * nr * WINDOW
    assert counts["resident_accepted_world_steps"] == n // 2 * WINDOW
    assert counts["resident_rejected_world_steps"] == 0 and counts["cuda_d2d_bytes"] == 0
    assert counts["resident_status_downloaded_bytes"] == 4 * n // 2 * WINDOW
    assert counts["resident_snapshot_downloaded_bytes"] == 52 * n // 2 * WINDOW
    assert counts["material_downloaded_bytes"] == 292 * n // 2 * WINDOW
    assert counts["cuda_d2h_bytes"] == (56 + 292) * n // 2 * WINDOW
    forces = []
    for index, frame in enumerate(frames):
        assert len(frame.flags) == n and all(dict(frame.flags).values())
        row = []
        for key, result in frame.material:
            force = result.audit.sample.normal_force_n
            if runner.case.roles[key] == "contact":
                assert 1 <= len(result.audit.sample.sites) <= 2 and force > 0
                if index == 0:
                    assert len(result.audit.sample.sites) == 2 and abs(2 - force) < 1e-6
            else:
                assert result.audit.sample.sites == () and force == 0
            row.append((key, force))
        forces.append(row)
    return frames, dict(
        wall_ns=wall,
        reset_outside_clock_ns=restored,
        counts=counts,
        normal_forces_n=forces,
        site_counts=[{k: len(v.audit.sample.sites) for k, v in frame.material} for frame in frames],
        sha256=signature(frames),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worlds", type=int, choices=(4, 16), required=True)
    p.add_argument("--first", choices=("scene", "separate"), required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--samples", type=int, default=7)
    args = p.parse_args()
    if args.warmups < 1 or args.samples < 3 or args.out.exists():
        p.error("require warmup, at least three measured windows and a fresh receipt")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    case = population(args.worlds)
    order = (args.first, "separate" if args.first == "scene" else "scene")
    runners = {}
    setup = {}
    rows = []
    with ExitStack() as stack:
        for lane in order:
            start = perf_counter_ns()
            runners[lane] = MixedRun(case, lane == "scene", "cuda", args.cache_root, stack)
            setup[lane] = perf_counter_ns() - start
        for i in range(1 + args.warmups + args.samples):
            direction = order if i % 2 == 0 else tuple(reversed(order))
            samples = {}
            actual = {}
            for lane in direction:
                actual[lane], samples[lane] = window(runners[lane], i != 0)
            assert (
                actual["scene"] == actual["separate"]
                and samples["scene"]["sha256"] == samples["separate"]["sha256"]
            )
            if rows:
                assert samples["scene"]["sha256"] == rows[0]["scene"]["sha256"]
            rows.append(dict(order=direction, **samples))
        end = dict(actual["scene"][-1].material)
    measurements = rows[1 + args.warmups :]
    summary = {}
    for lane in order:
        times = sorted(r[lane]["wall_ns"] for r in measurements)
        totals = {
            k: sum(r[lane]["counts"][k] for r in measurements)
            for k in measurements[0][lane]["counts"]
        }
        assert (
            totals["material_device_allocations"]
            == totals["resident_device_allocations"]
            == totals["material_static_uploaded_bytes"]
            == 0
        )
        summary[lane] = dict(
            p50_ms=median(times) / 1e6,
            min_ms=times[0] / 1e6,
            max_ms=times[-1] / 1e6,
            world_steps=len(times) * args.worlds * WINDOW,
            world_steps_s=len(times) * args.worlds * WINDOW * 1e9 / sum(times),
            totals=totals,
        )
    result = dict(
        worlds=args.worlds,
        first=args.first,
        h_s=H,
        steps_per_window=WINDOW,
        roles=case.roles,
        setup_ns=setup,
        cold=rows[0],
        warmups=rows[1 : 1 + args.warmups],
        measurements=measurements,
        summary=summary,
        throughput_scene_over_separate=summary["scene"]["world_steps_s"]
        / summary["separate"]["world_steps_s"],
        window_sha256=rows[0]["scene"]["sha256"],
        end_material_reports={k: asdict(v) for k, v in end.items()},
    )
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    print(
        f"W{args.worlds}: grouped {summary['scene']['p50_ms']:.3f} ms, separate {summary['separate']['p50_ms']:.3f} ms / {WINDOW} steps; throughput {result['throughput_scene_over_separate']:.3f}x"
    )


if __name__ == "__main__":
    main()
