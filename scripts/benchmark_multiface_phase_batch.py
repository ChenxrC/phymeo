"""Compare the same four physical steps with per-world or cross-world CUDA normal-law pages."""

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import phymeo as ph

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "scale"))
from multiface_load_cases import H, STEPS, certify, make_world, native, physical_hash, whole_window  # noqa: E402


def summary(frames):
    return [[physical_hash(frame) for frame in one] for one in frames]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 1 or args.out.exists():
        parser.error("positive pair count and a fresh receipt path are required")
    settings = ph.StepSettings(substeps=1, position_iterations=1, velocity_iterations=1, record_substeps=True)
    with ExitStack() as resources:
        layer = resources.enter_context(ph.TyxNormalLayer("cuda", cache_root=args.cache_root))
        worlds = [make_world(i, 8, layer) for i in range(4)]
        steppers = [resources.enter_context(w.program.stepper(settings)) for w in worlds]
        batch = resources.enter_context(ph.TyxMultifaceContactBatch(
            [(i, ph.ClothLinkWorld(w.program, w.initial)) for i, w in enumerate(worlds)],
            "cuda", settings=settings, cache_root=args.cache_root,
        ))
        controls = [{i: w.controls[t] for i, w in enumerate(worlds)} for t in range(STEPS)]

        def measure(grouped):
            if grouped:
                batch.reset()
            op = batch if grouped else layer
            before = op.stats()
            start = perf_counter_ns()
            if grouped:
                frames = [[] for _ in worlds]
                for control in controls:
                    out = batch.step(H, control)
                    for key, frame in out.items():
                        frames[key].append(frame)
            else:
                frames = whole_window(worlds, steppers)
            elapsed = perf_counter_ns() - start
            after = op.stats()
            counts = {k: v - before[k] for k, v in after.items() if k != "backend"}
            assert all(f.report.accepted for one in frames for f in one)
            assert counts["sites"] == 416
            assert counts["completed_launches"] == counts["uploads"] == counts["downloads"] == (12 if grouped else 48)
            clocks = [[[dict(s.timings) for s in f.report.substeps] for f in one] for one in frames]
            return frames, dict(wall_ns=elapsed, counters=counts, physical_hashes=summary(frames), raw_clocks_s=clocks)

        base_frames, base = measure(False)
        batch_frames, grouped = measure(True)
        assert grouped["physical_hashes"] == base["physical_hashes"]
        proofs = [certify(w, f) for w, f in zip(worlds, batch_frames, strict=True)]
        samples = []
        for i in range(args.pairs):
            order = ("ordinary", "grouped") if i % 2 == 0 else ("grouped", "ordinary")
            pair = {"order": order}
            for label in order:
                _, sample = measure(label == "grouped")
                assert sample["physical_hashes"] == base["physical_hashes"]
                pair[label] = sample
            samples.append(pair)
    p50 = {k: median(pair[k]["wall_ns"] for pair in samples) / 1e6 for k in ("ordinary", "grouped")}
    receipt = dict(
        worlds=4, faces_per_world=8, physical_steps=STEPS, stepsize_s=H, paired=samples,
        p50_ms=p50, ordinary_frames=native(base_frames), grouped_frames=native(batch_frames),
        physical_checks=native(proofs), last_phases=native(batch.last_phases),
        counters={"ordinary": base["counters"], "grouped": grouped["counters"]},
    )
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(receipt, f, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"p50_ms": p50, "counters": receipt["counters"]}))


if __name__ == "__main__":
    main()
