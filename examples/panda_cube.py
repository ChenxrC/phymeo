"""Run the classic Panda tabletop pick-and-place, optionally saving all physical substeps."""

import argparse
import json
from pathlib import Path

from phymeo.experiments.panda_cube import run_panda_cube


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--friction", type=float, default=0.65, help="finger/cube coefficient")
    parser.add_argument("--mass", type=float, default=0.10, help="cube mass in kg")
    parser.add_argument(
        "--box-size",
        type=float,
        nargs=3,
        default=(0.05, 0.05, 0.05),
        metavar=("GRIP", "TRANSVERSE", "HEIGHT"),
        help="full size in metres",
    )
    parser.add_argument(
        "--closed-width", type=float, default=0.029, help="nominal closing command in metres"
    )
    parser.add_argument(
        "--initial-xy-offset",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        help="object offset on the fixed table, world metres",
    )
    parser.add_argument(
        "--grip-above-com",
        type=float,
        default=0.012,
        help="target rubber face center height above the payload COM, metres",
    )
    parser.add_argument("--urdf", type=Path, help="compatible local classic Panda asset")
    parser.add_argument("--trace", type=Path, help="save full state and forces to a new JSON file")
    parser.add_argument(
        "--inputs-out", type=Path, help="save the initial state and control knots for independent dynamics"
    )
    args = parser.parse_args()
    if args.trace is not None and args.trace.exists():
        parser.error("choose a new trace path")
    if args.inputs_out is not None and (
        args.inputs_out.exists() or (args.trace is not None and args.inputs_out.resolve() == args.trace.resolve())
    ):
        parser.error("choose a new input manifest path distinct from the trace")
    result = run_panda_cube(
        friction=args.friction,
        mass_kg=args.mass,
        asset_path=args.urdf,
        box_size_m=args.box_size,
        closed_width_m=args.closed_width,
        grip_above_com_m=args.grip_above_com,
        initial_xy_offset_m=args.initial_xy_offset,
        record=args.trace is not None,
    )
    if args.trace is not None:
        with args.trace.open("x", encoding="utf-8") as stream:
            json.dump(result, stream)
    if args.inputs_out is not None:
        with args.inputs_out.open("x", encoding="utf-8") as stream:
            json.dump(result["inputs"], stream)
    result.pop("trace")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
