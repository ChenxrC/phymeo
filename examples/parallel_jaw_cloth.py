"""Record an actual finite-force gripper run through the public physics runtime."""

import argparse
from pathlib import Path

from phymeo.experiments.parallel_jaw_cloth import run_gripper_case, save_gripper_run
from phymeo.experiments.rigid_cloth import provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("hold", "slip", "capture", "material"), required=True)
    parser.add_argument("--mode", choices=("numpy", "interpret", "cuda", "cpu_cpp"), required=True)
    parser.add_argument("--h", type=float, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    before = provenance()
    run = run_gripper_case(args.case, mode=args.mode, h=args.h, duration_s=args.duration)
    after = provenance()
    if any(before[key] != after[key] for key in ("source_sha256", "python", "numpy", "platform")):
        raise RuntimeError("source or numerical environment changed during simulation")
    save_gripper_run(run, args.output, source=before)
    print(f"accepted={run.accepted}, recorded_steps={len(run.records)}, output={args.output}")
    return 0 if run.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
