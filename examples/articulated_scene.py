"""Run frozen P5 robot/cloth/softbody scenes and save actual audited traces."""

import argparse
import json
from pathlib import Path

from phymeo.experiments.articulated import run_articulated_scene, save_articulated_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("S2", "S3", "S4"), default="S4")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")

    def progress(frame, report):
        if frame % 25 == 0:
            print(f"frame {frame}: residual {report.max_residual:.3g}", flush=True)

    run = run_articulated_scene(scene=args.scene, frames=args.frames, progress=progress)
    save_articulated_run(run, args.output)
    print(json.dumps(run.metrics, indent=2))


if __name__ == "__main__":
    main()
