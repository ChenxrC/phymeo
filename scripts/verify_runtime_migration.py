"""Compare every numeric audit/trajectory array against a completed legacy run."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from phymeo.experiments.rigid_cloth import load_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    report = {
        "status": "failed",
        "reference_path": str(args.reference),
        "candidate_path": str(args.candidate),
        "array_comparisons": {},
        "comparison_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        reference, candidate = load_run(args.reference), load_run(args.candidate)
        config = dict(candidate.config)
        if config.pop("scheduler", None) != "scene":
            raise ValueError("candidate must use the unified scene scheduler")
        if config != reference.config:
            raise ValueError("reference and candidate physical/execution configurations differ")
        if set(reference.arrays) != set(candidate.arrays):
            raise ValueError("numeric field sets differ")
        expected_substeps = config["frames"] * config["substeps"]
        for run in (reference, candidate):
            if (
                run.metrics["accepted_frames"] != config["frames"]
                or run.metrics["audited_substeps"] != expected_substeps
            ):
                raise ValueError("missing accepted frame or independent substep evidence")
        all_equal = True
        for name, left in reference.arrays.items():
            right = candidate.arrays[name]
            compatible = left.shape == right.shape and left.dtype == right.dtype
            equal = compatible and np.array_equal(left, right)
            error = (
                None
                if not compatible
                else float(
                    np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)), initial=0)
                )
            )
            report["array_comparisons"][name] = {
                "same_shape_dtype": compatible,
                "exactly_equal": equal,
                "max_absolute_difference": error,
            }
            all_equal = all_equal and equal
        for label, directory, run in (
            ("reference", args.reference, reference),
            ("candidate", args.candidate, candidate),
        ):
            raw = json.loads((directory / "run.json").read_text(encoding="utf-8"))
            report[label] = {
                "provenance": raw["provenance"],
                "config": run.config,
                "metrics": run.metrics,
            }
        if not all_equal:
            raise ValueError("same-order numerical migration is not exact")
        report["status"] = "passed"
    except (ValueError, KeyError, OSError) as error:
        report["error"] = str(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "arrays": len(report["array_comparisons"]),
                "error": report.get("error"),
            }
        )
    )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
