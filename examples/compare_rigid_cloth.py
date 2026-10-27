"""Run the colored FP32 reference, device repetition and FP64 comparison."""

import argparse
import json
from pathlib import Path

from phymeo.experiments.backend_comparison import run_backend_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("interpret", "cpu_cpp", "cuda"), default="cuda")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/tyx-cache"))
    parser.add_argument("--fp64-reference", type=Path)
    args = parser.parse_args()

    def progress(name, frame, report):
        if frame % 30 == 0 or frame == args.frames:
            print(f"{name} frame {frame}: accepted, residual={report.max_residual:.3e}", flush=True)

    record = run_backend_suite(
        args.output,
        frames=args.frames,
        mode=args.mode,
        tolerance=args.tolerance,
        cache_root=args.cache_root,
        fp64_reference=args.fp64_reference,
        progress=progress,
    )
    print(
        json.dumps(
            {
                key: record[key]
                for key in (
                    "status",
                    "matched_fp32",
                    "device_repetition_identical",
                    "fp64_difference",
                )
            },
            indent=2,
        )
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
