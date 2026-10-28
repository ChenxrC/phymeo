"""Serial host affine-sweep observations; no response, FK or GPU timing."""

import argparse
import hashlib
import json
import traceback
from fractions import Fraction
from pathlib import Path
from time import perf_counter

import numpy as np

import phymeo as ph
from phymeo.experiments.rigid_cloth import provenance


def cases():
    plate = np.array([[0.0, 0, 0], [2.0, 0, 0], [0.0, 2, 0]])
    point = np.array([[0.5, 0.5, 1.0]])
    first_rod = np.array([[0.0, 0, -0.25], [0.0, 0, 0.25]])
    return (
        ("separated_probe", point, point + [0, 0, 1], plate, plate, 0.25, "certified"),
        ("crossing_probe", point, point - [0, 0, 2], plate, plate, 0.25, "violation"),
        (
            "subdivided_safe_rods",
            first_rod,
            first_rod,
            np.array([[1.0, -2, 0], [1.0, 2, 0]]),
            np.array([[2.0, 1, 0], [-2.0, 1, 0]]),
            0.5,
            "certified",
        ),
    )


def operation(case):
    name, a0, a1, b0, b1, clearance, expected = case
    result = ph.linear_feature_sweep(
        a0, a1, b0, b1, clearance_m=clearance, max_intervals=128, max_depth=20
    )
    if (
        result.status != expected
        or result.method != "affine-convex-separation-v1"
        or not 1 <= result.checked_intervals <= 128
    ):
        raise RuntimeError(f"analytic status or bounded work failed for {name}")
    pieces = sorted(result.certified_intervals + result.unresolved_intervals)
    cursor = 0.0
    for lo, hi in pieces:
        if lo != cursor or not lo < hi:
            raise RuntimeError("incomplete or overlapping time partition")
        cursor = hi
    if cursor != 1.0:
        raise RuntimeError("partition does not cover complete path")
    if expected == "certified" and result.unresolved_intervals:
        raise RuntimeError("certified result retains unresolved intervals")
    if name == "subdivided_safe_rods" and result.checked_intervals <= 1:
        raise RuntimeError("whole-root fixed plane cannot certify this analytic motion")
    if expected == "violation":
        t = Fraction(result.witness_time)
        if not 0 <= t <= 1 or (1 - 2 * t) ** 2 >= Fraction(clearance) ** 2:
            raise RuntimeError("violation time fails independent point-plane oracle")
    return (
        result.status,
        result.checked_intervals,
        result.certified_intervals,
        result.unresolved_intervals,
        result.witness_time,
        result.method,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1 or args.output.exists():
        parser.error("positive counts and new output path required")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record = {
        "source": provenance(),
        "script_sha256": script_hash,
        "backend": "host-fp64-input-exact-rational-proof",
        "status": "running",
        "scope": "sweep plus analytic/partition/repetition checks; excludes fixture setup",
        "samples_per_batch": args.samples,
        "repeats": args.repeats,
        "warmups_per_case": 3,
        "accepted_measured_calls": 0,
        "successful_calls_including_warmups": 0,
        "cases": [],
        "not_measured": ["contact response", "robot FK arcs", "GPU CCD", "Warp", "peak memory"],
    }
    phase = "setup"
    try:
        for case in cases():
            current = {"name": case[0], "expected_status": case[-1], "batches": []}
            record["cases"].append(current)
            reference = None
            for warmup in range(3):
                phase = f"{case[0]}:warmup{warmup}"
                start = perf_counter()
                result = operation(case)
                elapsed = perf_counter() - start
                if reference is None:
                    reference = result
                    current.update(first_call_seconds=elapsed, checked_intervals=result[1])
                elif result != reference:
                    raise RuntimeError("non-deterministic sweep result")
                record["successful_calls_including_warmups"] += 1
            for repeat in range(args.repeats):
                phase = f"{case[0]}:batch{repeat}"
                seconds = []
                batch = {"seconds": seconds}
                current["batches"].append(batch)
                for _ in range(args.samples):
                    start = perf_counter()
                    result = operation(case)
                    if result != reference:
                        raise RuntimeError("non-deterministic sweep result")
                    seconds.append(perf_counter() - start)
                    record["accepted_measured_calls"] += 1
                    record["successful_calls_including_warmups"] += 1
                batch["median_seconds"] = float(np.median(seconds))
                batch["p95_seconds"] = float(np.percentile(seconds, 95))
        record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", phase=phase, error_type=type(error).__name__)
        traceback.print_exc()
    if (
        provenance()["source_sha256"] != record["source"]["source_sha256"]
        or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != script_hash
    ):
        record["status"] = "source_changed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(record["status"], args.output)
    return 0 if record["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
