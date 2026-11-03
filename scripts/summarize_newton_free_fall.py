"""Match G2 guarded receipts by physical outputs and by the actual paired process order."""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import median


def compare(root, phymeo_observation="checkpoint"):
    qualifications = {}
    pairs = {}
    intervals = []
    for path in root.glob("g2x-*/report.json"):
        guard_path = path.with_name("final.json")
        if not guard_path.exists():
            continue
        guard = json.loads(guard_path.read_text(encoding="utf-8-sig"))
        if guard["raw_exit"] != 0:
            continue
        receipt = json.loads(path.read_text(encoding="utf8"))
        engine = receipt["engine"]
        expected = phymeo_observation if engine == "phymeo" else "native_rigid_state"
        legacy = "checkpoint" if engine == "phymeo" else "native_rigid_state"
        assert receipt.get("observation", legacy) == expected
        if receipt["phase"] == "qualify":
            key = engine, receipt["device"]
            assert key not in qualifications and [r["worlds"] for r in receipt["runs"]] == [1, 2]
            qualifications[key] = receipt["runs"]
            continue
        match = re.fullmatch(
            r"g2x-round-([123])-w(1|32)-(newton|phymeo)(?:-v\d+)?", path.parent.name
        )
        assert match is not None and match[3] == engine and receipt["device"] == "cuda:0"
        round_id, worlds = int(match[1]), int(match[2])
        (run,) = receipt["runs"]
        assert run["worlds"] == worlds and run["verified_world_steps"] == 640 * worlds
        assert (
            receipt["warmup_windows"],
            receipt["samples"],
            receipt["steps_per_window"],
            run["final_tick"],
        ) == (5, 20, 32, 801)
        assert run["max_position_error_m"] == run["max_velocity_error_m_s"] == 0
        if engine == "phymeo" and phymeo_observation == "kinematics":
            counts = run["warm_totals"]
            assert counts["snapshot_downloaded_bytes"] == 0
            assert counts["kinematics_downloaded_bytes"] == worlds * 640 * 52
            assert counts["cuda_kernel_launches"] == 640
        slot = pairs.setdefault((round_id, worlds), {})
        assert engine not in slot
        slot[engine] = (path.parent.name, run)
        intervals.append(
            (
                datetime.fromisoformat(guard["start"]),
                datetime.fromisoformat(guard["end"]),
                round_id,
                worlds,
                engine,
            )
        )

    expected = {(e, d) for e in ("newton", "phymeo") for d in ("cpu", "cuda:0")}
    assert set(qualifications) == expected
    ref = qualifications["newton", "cpu"]
    for runs in qualifications.values():
        for a, b in zip(ref, runs, strict=True):
            assert b["max_position_error_m"] == b["max_velocity_error_m_s"] == 0
            assert (
                a["application_order"] == b["application_order"]
                and a["trajectory"] == b["trajectory"]
            )
        assert (
            runs[1]["asymmetric"]
            and runs[1]["initial_positions_velocities"][0]
            != runs[1]["initial_positions_velocities"][1]
        )

    ordered = sorted(intervals)
    assert all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:]))
    planned = [
        (r, w, e)
        for r in (1, 2, 3)
        for w in ((32, 1) if r == 2 else (1, 32))
        for e in (("newton", "phymeo") if r == 2 else ("phymeo", "newton"))
    ]
    assert [tail[2:] for tail in ordered] == planned
    signatures = {}
    observations = []
    for round_id, worlds in sorted(pairs):
        result = dict(round=round_id, worlds=worlds)
        for engine in ("phymeo", "newton"):
            tag, run = pairs[round_id, worlds][engine]
            item = (
                run["application_order"],
                run["initial_positions_velocities"],
                run["cold"]["output_sha256"],
                [w["output_sha256"] for w in run["measurements"]],
                run["last_positions_velocities"],
            )
            assert signatures.setdefault(worlds, item) == item
            result[engine] = dict(
                tag=tag,
                cold_ms=run["cold"]["wall_ns"] / 1e6,
                p50_ms=run["warm_p50_window_ns"] / 1e6,
                p95_ms=run["warm_p95_window_ns"] / 1e6,
                world_steps_s=run["verified_world_steps_per_s"],
                median_window_world_steps_s=run["median_window_world_steps_per_s"],
            )
        result["p50_newton_over_phymeo"] = result["newton"]["p50_ms"] / result["phymeo"]["p50_ms"]
        result["throughput_newton_over_phymeo"] = (
            result["newton"]["world_steps_s"] / result["phymeo"]["world_steps_s"]
        )
        observations.append(result)
    return dict(
        ordered_processes=planned,
        phymeo_observation=phymeo_observation,
        qualifying_trajectories="all 32 W1 and asymmetric W2 steps match on all four backends",
        output_signatures={str(w): dict(cold=v[2], measured=v[3]) for w, v in signatures.items()},
        pairs=observations,
        paired_p50_newton_over_phymeo_median={
            str(w): median(r["p50_newton_over_phymeo"] for r in observations if r["worlds"] == w)
            for w in (1, 32)
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipts-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--phymeo-observation", choices=("checkpoint", "kinematics"), default="checkpoint"
    )
    args = p.parse_args()
    if args.out.exists():
        p.error("provide a fresh output path")
    result = compare(args.receipts_root, args.phymeo_observation)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    for row in result["pairs"]:
        print(
            f"round {row['round']} W{row['worlds']}: Phymeo {row['phymeo']['p50_ms']:.3f} ms, Newton {row['newton']['p50_ms']:.3f} ms"
        )


if __name__ == "__main__":
    main()
