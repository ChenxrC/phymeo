"""Verify complete P5 runs, exact numerical repetitions and observable feedback."""

import argparse
import json
from pathlib import Path

import numpy as np

from phymeo.experiments.articulated import load_articulated_run
from phymeo.experiments.prescribed_articulated import load_prescribed_run

NAMES = ("S2", "S3", "S4", "S2-driver", "S3-driver")


def verify(first_root, repeat_root):
    first_root, repeat_root = Path(first_root), Path(repeat_root)
    if first_root.resolve() == repeat_root.resolve():
        # Still validate the horizon before explaining the independent-run requirement.
        load_articulated_run(first_root / "S2", require_complete=True)
        raise ValueError("repetition requires a separately executed output directory")
    runs, records, repetitions, sources = {}, {}, {}, set()
    for name in NAMES:
        loader = load_prescribed_run if name.endswith("-driver") else load_articulated_run
        pair = []
        for root in (first_root, repeat_root):
            run = loader(root / name, require_complete=True)
            metadata = json.loads((root / name / "run.json").read_text(encoding="utf-8"))
            source = metadata["provenance"]
            sources.add(
                tuple(source.get(key) for key in ("source_sha256", "python", "numpy", "platform"))
            )
            records[f"{name}/{'first' if root == first_root else 'repeat'}"] = metadata
            pair.append(run)
        first, repeat = pair
        if first.config != repeat.config or set(first.arrays) != set(repeat.arrays):
            raise ValueError(f"{name}: repeated configuration or fields differ")
        for field, value in first.arrays.items():
            other = repeat.arrays[field]
            if (
                value.dtype != other.dtype
                or value.shape != other.shape
                or not np.array_equal(value, other)
            ):
                raise ValueError(f"{name}: repeated array {field} differs")
        repetitions[name] = {
            "exact": True,
            "array_count": len(first.arrays),
            "element_count": sum(value.size for value in first.arrays.values()),
            "trajectory_sha256": first.metrics["trajectory_sha256"],
        }
        runs[name] = first
    if len(sources) != 1:
        raise ValueError(
            "controlled runs require identical package source and numerical environment"
        )
    feedback = {}
    for name in ("S2", "S3"):
        two, driver = runs[name], runs[name + "-driver"]
        if not np.array_equal(two.arrays["time"], driver.arrays["time"]):
            raise ValueError("feedback runs must share the same time samples")
        domain = two.config["node_domains"][0]
        q_difference = float(abs(two.arrays["robot_q"][-1, 0] - driver.arrays["source_q"][-1, 0]))
        node_difference = float(
            np.linalg.norm(
                two.arrays[domain + "_position"] - driver.arrays[domain + "_position"], axis=-1
            ).max()
        )
        if q_difference < 1e-5 or node_difference < 1e-6:
            raise ValueError(
                f"{name}: bidirectional feedback does not meet frozen observable thresholds"
            )
        feedback[name] = {
            "final_q_difference_m": q_difference,
            "max_node_difference_m": node_difference,
        }
    return {
        "status": "complete",
        "scope": "P5 physical traces and counterfactual repetition; not final branch acceptance",
        "thresholds": {"final_q_difference_m": 1e-5, "max_node_difference_m": 1e-6},
        "repetition": repetitions,
        "feedback": feedback,
        "runs": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    record = verify(args.first, args.repeat)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps({"repetition": record["repetition"], "feedback": record["feedback"]}, indent=2)
    )


if __name__ == "__main__":
    main()
