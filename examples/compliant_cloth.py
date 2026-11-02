"""Run a compliant-cloth case and save the actual motion and contact samples."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from phymeo.experiments.cloth_drape import run_drape
from phymeo.experiments.compliant_cloth import run_impact
from phymeo.experiments.compliant_handoff import run_handoff
from phymeo.experiments.sheet_fabric import fabric_preset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", required=True, choices=("impact", "support", "handoff", "no-grip", "drape")
    )
    parser.add_argument(
        "--robot",
        choices=("cartesian", "articulated"),
        default="cartesian",
        help="gripper layout for handoff/no-grip",
    )
    parser.add_argument(
        "--fabric",
        choices=("reference", "soft"),
        help="handoff or drape fabric; reference preserves the original handoff",
    )
    parser.add_argument(
        "--bend-compliance",
        type=float,
        help="override per-hinge compliance in 1/(N m); larger bends more easily",
    )
    parser.add_argument(
        "--initial-tilt-deg",
        type=float,
        default=0.0,
        help="physical initial sheet angle from vertical in handoff/no-grip",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    handoff = args.case in ("handoff", "no-grip")
    if not handoff and (args.robot != "cartesian" or args.initial_tilt_deg != 0.0):
        parser.error("robot layout and sheet tilt apply only to handoff/no-grip")
    if args.case in ("impact", "support") and (
        args.fabric is not None or args.bend_compliance is not None
    ):
        parser.error("fabric options apply to handoff/no-grip and drape")
    fabric = fabric_preset(args.fabric or ("soft" if args.case == "drape" else "reference"))
    if args.bend_compliance is not None:
        try:
            fabric = replace(fabric, bend_compliance=args.bend_compliance)
        except ValueError as exc:
            parser.error(str(exc))
    if args.case in ("impact", "support"):
        trace = run_impact(support=args.case == "support")
    elif args.case == "drape":
        trace = run_drape(fabric=fabric)
    else:

        def progress(phase, time):
            print(f"{time:.3f} s  {phase}", flush=True)

        trace = run_handoff(
            grip_friction=0.0 if args.case == "no-grip" else 0.9,
            robot=args.robot,
            progress=progress,
            fabric=fabric,
            initial_tilt_deg=args.initial_tilt_deg,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(trace, stream, allow_nan=False, separators=(",", ":"))
    print(
        json.dumps(
            {
                k: trace[k]
                for k in ("case", "completed", "metrics", "failure_phase", "failure", "time_s")
                if k in trace
            },
            ensure_ascii=False,
        )
    )
    expected = not trace["completed"] and trace.get("failure_phase") == "lift_A"
    return 0 if (expected if args.case == "no-grip" else trace["completed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
