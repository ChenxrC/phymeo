"""Run a compliant-cloth case and save the actual motion and contact samples."""

import argparse
import json
from pathlib import Path

from phymeo.experiments.compliant_cloth import run_impact
from phymeo.experiments.compliant_handoff import run_handoff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", required=True, choices=("impact", "support", "handoff", "no-grip")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.case in ("impact", "support"):
        trace = run_impact(support=args.case == "support")
    else:

        def progress(phase, time):
            print(f"{time:.3f} s  {phase}", flush=True)

        trace = run_handoff(grip_friction=0.0 if args.case == "no-grip" else 0.9, progress=progress)
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
