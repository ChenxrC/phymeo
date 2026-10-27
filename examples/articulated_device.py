"""Run the explicit P6 device path or its same-settings FP64 reference."""

import argparse
import json
from pathlib import Path

from phymeo.experiments.articulated_device import (
    run_articulated_device,
    run_articulated_reference,
    save_articulated_device_run,
)
from phymeo.experiments.rigid_cloth import provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("S2", "S3", "S4"), default="S2")
    parser.add_argument(
        "--mode", choices=("interpret", "cpu_cpp", "cuda", "reference"), default="cuda"
    )
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    source = provenance()

    def progress(frame, report):
        if frame % 25 == 0:
            print(f"frame {frame}: residual {report.max_residual:.3g}", flush=True)

    if args.mode == "reference":
        run = run_articulated_reference(scene=args.scene, frames=args.frames, progress=progress)
    else:
        run = run_articulated_device(
            scene=args.scene, mode=args.mode, frames=args.frames, progress=progress
        )
    current = provenance()
    if any(
        current.get(key) != source.get(key)
        for key in ("source_sha256", "python", "numpy", "platform")
    ):
        raise RuntimeError("source or numerical environment changed during simulation")
    save_articulated_device_run(run, args.output, source=source)
    print(json.dumps(run.metrics, indent=2))


if __name__ == "__main__":
    main()
