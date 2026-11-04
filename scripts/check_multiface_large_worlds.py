"""Check paired multiface scale receipts against complete archived scene frames."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def physical(path):
    with gzip.open(path, "rt", encoding="utf8") as source:
        frames = json.load(source)
    for world in frames:
        for frame in world:
            assert frame["report"]["accepted"]
            for sub in frame["report"]["substeps"]:
                sub["timings"] = dict.fromkeys(sub["timings"], 0.0)
    return frames


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--receipts", nargs="+", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if args.out.exists():
        cli.error("use a fresh output receipt")
    groups = defaultdict(list)
    for path in args.receipts:
        report = json.loads(path.read_text(encoding="utf8"))
        guard = json.loads(path.with_name("final.json").read_text(encoding="utf-8-sig"))
        assert guard["raw_exit"] == 0 and report["accepted_per_window"] == 4 * report["worlds"]
        groups[(report["worlds"], report["density"])].append((path, report))
    assert set(groups) == {(w, d) for w in (16, 32) for d in ("dense", "mixed")}
    summary = {}
    for (worlds, density), group in sorted(groups.items()):
        assert len(group) == 2 and {r["first"] for _, r in group} == {"ordinary", "grouped"}
        expected_hashes = reference = None
        entries = []
        for path, r in group:
            assert r["registered_faces"] == worlds * (8 if density == "dense" else 5)
            assert r["loaded_faces_each_midpoint"] == worlds * (8 if density == "dense" else 3)
            assert len(r["per_world_proof"]) == worlds and len(r["pairs"]) == 5
            for lane in ("ordinary", "grouped"):
                frames = physical(path.with_name(r["full_cold_frames"][lane]))
                assert len(frames) == worlds and all(len(w) == 4 for w in frames)
                if reference is None:
                    reference = frames
                    expected_hashes = r["cold"][lane]["hashes"]
                else:
                    assert frames == reference
                assert r["cold"][lane]["hashes"] == expected_hashes
                for pair in r["pairs"]:
                    assert pair[lane]["hashes"] == expected_hashes
                    assert pair[lane]["counts"] == r["cold"][lane]["counts"]
            entries.append(
                dict(
                    receipt=str(path),
                    first=r["first"],
                    p50_ms=r["p50_ms"],
                    rate=r["verified_world_steps_s"],
                    grouped_wins=sum(
                        p["grouped"]["wall_ns"] < p["ordinary"]["wall_ns"] for p in r["pairs"]
                    ),
                    cold_window_ms={k: v["wall_ns"] / 1e6 for k, v in r["cold"].items()},
                    setup_and_cold_window_ms={
                        k: v["setup_plus_first_window_ns"] / 1e6 for k, v in r["setup"].items()
                    },
                    counts={k: v["counts"] for k, v in r["cold"].items()},
                )
            )
        summary[f"W{worlds}-{density}"] = entries
    with args.out.open("x", encoding="utf8") as stream:
        json.dump(summary, stream, allow_nan=False, indent=2)
    print({case: [v["p50_ms"] for v in entries] for case, entries in summary.items()})


if __name__ == "__main__":
    main()
