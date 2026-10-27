"""Run frozen P5 robot/cloth/softbody scenes and save actual audited traces."""

import argparse
import json
from pathlib import Path

from phymeo.experiments.articulated import run_articulated_scene, save_articulated_run
from phymeo.experiments.prescribed_articulated import run_prescribed_scene, save_prescribed_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("S2", "S3", "S4"), default="S4")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--experiment", choices=("two-way", "prescribed"), default="two-way")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")

    def progress(frame, report):
        if frame % 25 == 0:
            print(f"frame {frame}: residual {report.max_residual:.3g}", flush=True)

    runner, save = (
        (run_articulated_scene, save_articulated_run)
        if args.experiment == "two-way"
        else (run_prescribed_scene, save_prescribed_run)
    )
    run = runner(scene=args.scene, frames=args.frames, progress=progress)
    save(run, args.output)
    print(json.dumps(run.metrics, indent=2))


if __name__ == "__main__":
    main()
