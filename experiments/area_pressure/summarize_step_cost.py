"""Print the cost of complete CUDA steps from the paired profiling experiment."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def summarize(case):
    runs = case["runs"]
    if [r["detailed"] for r in runs] != [False, True, True, False]:
        raise ValueError("expected four complete ordinary / detailed / detailed / ordinary runs")
    steps = len(runs[0]["rows"])
    if steps < 2 or any(len(r["rows"]) != steps for r in runs):
        raise ValueError("incomplete or inconsistent step counts")
    ordinary, detailed = [], []
    for run in runs:
        (detailed if run["detailed"] else ordinary).extend(run["rows"][1:])
    costs = Counter()
    for row in detailed:
        if (
            min(row["exclusive_ns"].values()) < 0
            or sum(row["exclusive_ns"].values()) != row["scene_ns"]
        ):
            raise ValueError("exclusive stages must partition the full step")
        costs.update(row["exclusive_ns"])
    total = sum(costs.values())

    def avg_ms(rows, key):
        return mean(row[key] for row in rows) / 1e6

    ordinary_observed = sum(
        avg_ms(ordinary, key) for key in ("scene_ns", "refresh_ns", "readback_ns", "snapshot_ns")
    )
    return dict(
        case=case["case"],
        worlds=case["worlds"],
        buckets=case["buckets"],
        steps_per_run=steps,
        ordinary_first_scene_ms=mean(r["rows"][0]["scene_ns"] for r in runs if not r["detailed"])
        / 1e6,
        ordinary_scene_ms=avg_ms(ordinary, "scene_ns"),
        ordinary_observed_ms=ordinary_observed,
        world_steps_per_s=1000 * case["worlds"] / ordinary_observed,
        detailed_scene_ms=avg_ms(detailed, "scene_ns"),
        stages={
            k: dict(ms=ns / len(detailed) / 1e6, percent=100 * ns / total)
            for k, ns in costs.most_common()
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profiles", type=Path, nargs="+", help="scene-step-profile.json from passing cases"
    )
    args = parser.parse_args()
    for path in args.profiles:
        print(json.dumps(summarize(json.loads(path.read_text(encoding="utf-8"))), allow_nan=False))


if __name__ == "__main__":
    main()
