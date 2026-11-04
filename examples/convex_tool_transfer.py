"""Run the articulated tray task, or render an existing exact trajectory without replaying physics."""

import argparse
import gzip
import json
from pathlib import Path

from phymeo.experiments.convex_tool_replay import write_tool_replay
from phymeo.experiments.convex_tool_transfer import build_tool_transfer, run_tool_transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fabric", choices=("soft", "reference"), default="soft")
    parser.add_argument("--footprint", choices=("rectangle", "taper"), default="rectangle")
    parser.add_argument("--friction", type=float, default=0.75)
    parser.add_argument("--grid", type=int, nargs=2, metavar=("COLUMNS", "ROWS"), default=(2, 2))
    parser.add_argument("--output", type=Path, help="Write per-step physical JSON or JSON.GZ")
    parser.add_argument("--html", type=Path, help="Write an offline scrub/play replay")
    parser.add_argument(
        "--from-trace", type=Path, help="Render a previously saved physical history"
    )
    args = parser.parse_args()
    if args.from_trace:
        if not args.html or args.output:
            parser.error("--from-trace requires --html without --output")
        op = gzip.open if args.from_trace.suffix == ".gz" else open
        with op(args.from_trace, "rt", encoding="utf8") as stream:
            trace = json.load(stream)
    else:
        if not args.output:
            parser.error("physical runs require --output to keep their measurements")
        scene = build_tool_transfer(
            fabric=args.fabric,
            footprint=args.footprint,
            friction=args.friction,
            grid=tuple(args.grid),
        )
        trace = run_tool_transfer(scene)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        op = gzip.open if args.output.suffix == ".gz" else open
        with op(args.output, "wt", encoding="utf8") as stream:
            json.dump(trace, stream, allow_nan=False, separators=(",", ":"))
    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        write_tool_replay(trace, args.html)
    print(
        json.dumps(
            {key: trace[key] for key in ("status", "qualified", "reason", "events")},
            ensure_ascii=False,
        )
    )
    return 0 if trace["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
