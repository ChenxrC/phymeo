"""Serial host distance-certificate observations, excluding closest-point generation."""

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np

import phymeo as ph
from phymeo.experiments.rigid_cloth import provenance


def cases():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    return (
        (
            "probe_triangle",
            np.array([[0.5, 0.5, 3.0]]),
            np.array([[0.0, 0, 0], [2.0, 0, 0], [0.0, 2, 0]]),
            np.ones(1),
            np.array([0.5, 0.25, 0.25]),
            3.0,
        ),
        (
            "skew_rods",
            np.array([[-1.0, 0, 0], [1.0, 0, 0]]),
            np.array([[0.0, -1, 2], [0.0, 1, 2]]),
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            2.0,
        ),
        (
            "subnormal_point_hull",
            np.array([[0.0, 0, tiny]]),
            np.zeros((3, 3)),
            np.ones(1),
            np.array([1.0, 0, 0]),
            tiny,
        ),
    )


def operation(case):
    name, first, second, weights_a, weights_b, expected = case
    result = ph.feature_distance_bounds(
        first, second, first_weights=weights_a, second_weights=weights_b
    )
    if (
        result.lower_m != expected
        or result.upper_m != expected
        or result.method != "convex-kkt-distance-v1"
    ):
        raise RuntimeError(f"independent analytic result or method failed for {name}")
    return result.lower_m, result.upper_m, result.method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1 or args.output.exists():
        parser.error("positive sample counts and a new output path are required")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record = {
        "source": provenance(),
        "script_sha256": script_hash,
        "backend": "host-fp64-exact-rational",
        "scope": "certificate plus analytic/repetition checks; excludes case setup and closest witness generation",
        "status": "running",
        "samples_per_batch": args.samples,
        "repeats": args.repeats,
        "warmups_per_case": 3,
        "accepted_measured_calls": 0,
        "successful_calls_including_warmups": 0,
        "cases": [],
        "not_measured": ["GPU geometry", "CCD", "continuous trajectory", "Warp", "peak memory"],
    }
    phase = "setup"
    try:
        for case in cases():
            current = {"name": case[0], "expected_m": case[-1], "batches": []}
            record["cases"].append(current)
            expected_result = None
            for warmup in range(3):
                phase = f"{case[0]}:warmup{warmup}"
                start = perf_counter()
                result = operation(case)
                elapsed = perf_counter() - start
                if expected_result is None:
                    expected_result = result
                    current["first_call_seconds"] = elapsed
                elif result != expected_result:
                    raise RuntimeError("non-deterministic certificate")
                record["successful_calls_including_warmups"] += 1
            for repeat in range(args.repeats):
                phase = f"{case[0]}:batch{repeat}"
                seconds = []
                batch = {"seconds": seconds}
                current["batches"].append(batch)
                for _ in range(args.samples):
                    start = perf_counter()
                    result = operation(case)
                    if result != expected_result:
                        raise RuntimeError("non-deterministic certificate")
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
