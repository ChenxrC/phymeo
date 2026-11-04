"""Run the classic Panda tabletop pick-and-place, optionally saving all physical substeps."""

import argparse
import json
from pathlib import Path

from phymeo.experiments.panda_cube import run_panda_cube


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--friction", type=float, default=0.65, help="finger/cube coefficient")
    parser.add_argument("--mass", type=float, default=0.10, help="cube mass in kg")
    parser.add_argument("--urdf", type=Path, help="compatible local classic Panda asset")
    parser.add_argument("--trace", type=Path, help="save full state and forces to a new JSON file")
    args = parser.parse_args()
    if args.trace is not None and args.trace.exists():
        parser.error("choose a new trace path")
    result = run_panda_cube(
        friction=args.friction,
        mass_kg=args.mass,
        asset_path=args.urdf,
        record=args.trace is not None,
    )
    if args.trace is not None:
        with args.trace.open("x", encoding="utf-8") as stream:
            json.dump(result, stream)
    result.pop("trace")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
