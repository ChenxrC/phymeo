"""Run a compliant-cloth case and save the actual motion and contact samples."""

import argparse
import gzip
import json
from dataclasses import replace
from pathlib import Path

from phymeo.experiments.cloth_drape import run_drape
from phymeo.experiments.compliant_cloth import run_impact
from phymeo.experiments.compliant_handoff import run_handoff
from phymeo.experiments.sheet_fabric import fabric_preset
from phymeo.experiments.single_gripper_bend import run_single_gripper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        required=True,
        choices=("impact", "support", "handoff", "no-grip", "drape", "single-grip", "single-slip"),
    )
    parser.add_argument(
        "--robot",
        choices=("cartesian", "articulated", "articulated-palm"),
        help="gripper layout; the single-gripper benchmark uses an articulated arm",
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
        help="physical initial sheet angle from vertical",
    )
    parser.add_argument(
        "--finger-pad-radius-m",
        type=float,
        help="finite gripper pad rounding behind the original front, in metres",
    )
    parser.add_argument(
        "--finger-backing-clearance-m",
        type=float,
        help="optional unilateral clearance ahead of the finger middle surface, in metres",
    )
    parser.add_argument(
        "--receiver-control",
        choices=("snapshot", "material", "cooperative"),
        help="hold the initial target, follow the material, or acquire and transport cooperatively",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.receiver_control is not None and args.case not in ("handoff", "no-grip"):
        parser.error("receiver control requires a two-gripper case")
    if args.output.exists():
        parser.error("output already exists")
    single = args.case in ("single-grip", "single-slip")
    gripper_case = single or args.case in ("handoff", "no-grip")
    args.robot = args.robot or ("articulated" if single else "cartesian")
    if args.initial_tilt_deg is None:
        args.initial_tilt_deg = 2.0 if single else 0.0
    if single:
        if (
            args.robot != "articulated"
            or args.fabric not in (None, "soft")
            or args.bend_compliance is not None
        ):
            parser.error(
                "the single-gripper control requires the original soft sheet and articulated arm"
            )
        if args.finger_pad_radius_m is None:
            args.finger_pad_radius_m = 0.002
        if args.finger_backing_clearance_m is None:
            args.finger_backing_clearance_m = args.finger_pad_radius_m / 16

    if args.finger_pad_radius_m is not None and (
        not gripper_case or not args.finger_pad_radius_m > 0
    ):
        parser.error("a positive finger pad radius applies only to gripper cases")
    if args.finger_backing_clearance_m is not None and (
        not gripper_case
        or args.finger_pad_radius_m is None
        or not 0 < args.finger_backing_clearance_m <= args.finger_pad_radius_m
    ):
        parser.error("finger backing clearance must be positive and inside an authored finite pad")
    if not gripper_case and (args.robot != "cartesian" or args.initial_tilt_deg != 0.0):
        parser.error("robot layout and sheet tilt apply only to gripper cases")
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
    elif single:

        def single_progress(phase, time, values):
            print(f"{time:.3f} s  {phase}  jaw support={values['jaw_up_n']:.4f} N", flush=True)

        trace = run_single_gripper(
            grip_friction=0.9 if args.case == "single-grip" else 0.0,
            initial_tilt_deg=args.initial_tilt_deg,
            finger_pad_radius_m=args.finger_pad_radius_m,
            finger_backing_clearance_m=args.finger_backing_clearance_m,
            progress=single_progress,
        )
    else:

        def progress(phase, time):
            print(f"{time:.3f} s  {phase}", flush=True)

        trace = run_handoff(
            grip_friction=0.0 if args.case == "no-grip" else 0.9,
            robot=args.robot,
            progress=progress,
            receiver_control=args.receiver_control or "snapshot",
            fabric=fabric,
            initial_tilt_deg=args.initial_tilt_deg,
            finger_pad_radius_m=args.finger_pad_radius_m,
            finger_backing_clearance_m=args.finger_backing_clearance_m,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stream = (
        gzip.open(args.output, "xt", encoding="utf-8", compresslevel=3)
        if args.output.suffix == ".gz"
        else args.output.open("x", encoding="utf-8")
    )
    with stream:
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
    if args.case == "single-slip":
        return (
            0 if expected and trace.get("failure") == "gripper lost the sheet to the table" else 1
        )
    return 0 if (expected if args.case == "no-grip" else trace["completed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
