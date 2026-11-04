"""Compare two nearest-feature paths in exactly the same loaded CUDA worlds."""

import argparse
import json
import platform
from contextlib import ExitStack, nullcontext
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from statistics import median
from unittest.mock import patch

from profile_loaded_worlds import WINDOW, MixedRun, make_case, measure

from phymeo.constraints.collision_shapes import CollisionMotion
from phymeo.geometry._capsule_triangle_sweep import capsule_face_query
from phymeo.solvers.cloth_link_contact import ClothLinkContacts


def sweep_witness(self, cap, face, ends, tri):
    query = capsule_face_query(
        cap, self.triangles[face], CollisionMotion(ends), CollisionMotion(tri)
    )
    return query.sample(0)[0]


def compare(worlds, pairs, cache, reference):
    prior = json.loads(reference.read_text(encoding="utf8"))
    if (prior["worlds"], prior["backend"], prior["steps_per_window"]) != (worlds, "cuda", WINDOW):
        raise ValueError("the reference must use the same CUDA workload")
    frame_path = reference.parent / prior["frame_file"]
    digest = sha256(frame_path.read_bytes()).hexdigest()
    if digest != prior["sha256"]:
        raise ValueError("the retained complete frame payload differs from its reference")
    expected_counts = prior["pairs"][0]["plain"]["device_counts"]
    rows = []
    with ExitStack() as stack:
        runner = MixedRun(make_case(worlds), True, "cuda", cache, stack)

        def window(mode, *, cold=False):
            ctx = (
                patch.object(ClothLinkContacts, "_witness", sweep_witness)
                if mode == "sweep"
                else nullcontext()
            )
            with ctx:
                frames, sample = measure(runner, False)
            if sample["sha256"] != digest:
                raise AssertionError(f"{mode} changed complete frame digest: {sample['sha256']}")
            if not cold and sample["device_counts"] != expected_counts:
                raise AssertionError(
                    f"{mode} changed steady device work: {sample['device_counts']}"
                )
            return frames, sample["wall_ns"]

        warm, _ = window("sweep", cold=True)
        candidate_warm, _ = window("single_pose")
        if candidate_warm != warm:
            raise AssertionError("the two warmups disagree")
        for i in range(pairs):
            order = ("sweep", "single_pose") if i % 2 == 0 else ("single_pose", "sweep")
            row = {"order": order}
            for mode in order:
                frames, row[mode] = window(mode)
                if frames != warm:
                    raise AssertionError(f"{mode} changed state or history after reset")
            rows.append(row)
    paired = [row["sweep"] / row["single_pose"] for row in rows]
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        numpy=version("numpy"),
        tyx=version("tyx-compute"),
        worlds=worlds,
        steps_per_window=WINDOW,
        backend="cuda",
        reference=str(reference),
        full_frame_sha256=digest,
        device_counts=expected_counts,
        sweep_p50_ms=median(row["sweep"] for row in rows) / 1e6,
        single_pose_p50_ms=median(row["single_pose"] for row in rows) / 1e6,
        paired_speedup_p50=median(paired),
        paired_speedup_range=[min(paired), max(paired)],
        windows_ns=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, choices=(1, 4, 16, 32), required=True)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.pairs < 4 or args.pairs % 2:
        parser.error("choose a fresh output and an even number of pairs >= 4")
    report = compare(args.worlds, args.pairs, args.cache_root, args.reference)
    with args.out.open("x", encoding="utf8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(f"W{args.worlds}: {report['paired_speedup_p50']:.3f}x median paired speedup")


if __name__ == "__main__":
    main()
