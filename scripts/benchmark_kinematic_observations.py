"""Alternate complete and typed observations for identical CUDA G2 physical windows."""

import argparse
import json
import math
import sys
from contextlib import ExitStack
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "scale"))
from benchmark_newton_free_fall import checked_window  # noqa: E402
from g2_cross_engine_steps import STEPS, ApplicationOutput, PhymeoFall, load_engine, setup_rows  # noqa: E402


class CountedFall(PhymeoFall):
    def counters(self):
        return dict(
            super().counters(),
            kinematics_downloaded_bytes=self.work.stats()["kinematics_downloaded_bytes"],
        )


class KinematicFall(CountedFall):
    def read(self, tick):
        observed = self.work.kinematics()
        if (
            observed.keys != self.keys
            or observed.widths != (3,) * len(self.keys)
            or not np.all(observed.ticks == tick)
        ):
            raise AssertionError("typed observation did not report the expected committed worlds")
        return observed.values[:, :3], observed.values[:, 3:]


def summarize(measurements, worlds, lane):
    times = [x["wall_ns"] for x in measurements]
    counts = {
        name: sum(x["counters"][name] for x in measurements) for name in measurements[0]["counters"]
    }
    w = len(times) * STEPS * worlds
    assert (
        counts["solver_calls"]
        == counts["cuda_kernel_launches"]
        == counts["synchronizations"]
        == len(times) * STEPS
    )
    assert counts["accepted_world_steps"] == w and counts["status_downloaded_bytes"] == w * 4
    assert (
        counts["cuda_d2h_bytes"] == w * 56
        and counts["cuda_h2d_bytes"] == counts["cuda_d2d_bytes"] == 0
    )
    assert (
        counts["device_allocations"]
        == counts["rejected_world_steps"]
        == counts["control_uploaded_bytes"]
        == counts["topology_uploaded_bytes"]
        == 0
    )
    assert counts["snapshot_downloaded_bytes"] == (w * 52 if lane == "full" else 0)
    assert counts["kinematics_downloaded_bytes"] == (w * 52 if lane == "typed" else 0)
    return dict(
        warm_counters=counts,
        p50_ms=median(times) / 1e6,
        p95_ms=sorted(times)[math.ceil(0.95 * len(times)) - 1] / 1e6,
        world_steps=w,
        world_steps_per_s=w * 1e9 / sum(times),
    )


def existing(path, worlds, engine):
    report = json.loads(path.read_text(encoding="utf8"))
    (run,) = report["runs"]
    assert (
        report["engine"] == engine and report["phase"] == "measure" and report["device"] == "cuda:0"
    )
    assert run["worlds"] == worlds and run["final_tick"] == 801
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worlds", type=int, choices=(1, 32), required=True)
    p.add_argument("--first", choices=("full", "typed"), required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--previous-newton", type=Path, required=True)
    p.add_argument("--previous-full", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error("provide a new output file")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    modules, _, versions = load_engine("phymeo", "cuda:0", args.cache_root)
    keys, z, v = setup_rows(args.worlds)
    refs = [
        existing(args.previous_newton, args.worlds, "newton"),
        existing(args.previous_full, args.worlds, "phymeo"),
    ]
    order = (args.first, "typed" if args.first == "full" else "full")
    with ExitStack() as stack:
        lanes = {}
        apps = {}
        for lane in order:
            factory = CountedFall if lane == "full" else KinematicFall
            branch = factory(modules, keys, z, v, "cuda:0", args.cache_root)
            stack.callback(branch.close)
            branch.binding()
            app = ApplicationOutput(branch.keys, keys, z, v)
            initial = app.read(0, *branch.read(0))
            assert all(
                app.keys == tuple(r["application_order"])
                and initial.tolist() == r["initial_positions_velocities"]
                for r in refs
            )
            lanes[lane], apps[lane] = branch, app
        records = []
        tick = 0
        for index in range(26):
            # Both worlds continue independently through a cold step, five warmups, and twenty measured windows.
            steps = 1 if index == 0 else STEPS
            direction = order if index % 2 == 0 else tuple(reversed(order))
            samples = {}
            trajectories = {}
            for lane in direction:
                end, trajectories[lane], samples[lane] = checked_window(
                    lanes[lane], apps[lane], tick, steps
                )
            tick = end
            np.testing.assert_array_equal(trajectories["full"], trajectories["typed"])
            digest = samples["full"]["output_sha256"]
            assert digest == samples["typed"]["output_sha256"]
            if index == 0:
                assert all(digest == r["cold"]["output_sha256"] for r in refs)
            elif index >= 6:
                assert all(digest == r["measurements"][index - 6]["output_sha256"] for r in refs)
            records.append(dict(order=direction, full=samples["full"], typed=samples["typed"]))
        final = trajectories["typed"][-1]
        assert tick == 801 and all(final.tolist() == r["last_positions_velocities"] for r in refs)
        assert max(max(a.max_position_error, a.max_velocity_error) for a in apps.values()) == 0
        summary = {
            lane: summarize([entry[lane] for entry in records[6:]], args.worlds, lane)
            for lane in lanes
        }
    data = dict(
        worlds=args.worlds,
        dt_s=1 / 128,
        first=args.first,
        runtime=versions,
        final_tick=tick,
        final=final.tolist(),
        cold=records[0],
        warmups=records[1:6],
        measurements=records[6:],
        summary=summary,
        p50_full_over_typed=summary["full"]["p50_ms"] / summary["typed"]["p50_ms"],
        throughput_typed_over_full=summary["typed"]["world_steps_per_s"]
        / summary["full"]["world_steps_per_s"],
        prior_newton_receipt=str(args.previous_newton),
        prior_full_receipt=str(args.previous_full),
        previous_newton_world_steps_s=refs[0]["verified_world_steps_per_s"],
        matched_trajectories=True,
        max_position_error_m=0.0,
        max_velocity_error_m_s=0.0,
    )
    args.out.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf8")
    print(
        f"W{args.worlds}: full {summary['full']['p50_ms']:.3f} ms, typed {summary['typed']['p50_ms']:.3f} ms; paired throughput typed/full {data['throughput_typed_over_full']:.3f}"
    )


if __name__ == "__main__":
    main()
