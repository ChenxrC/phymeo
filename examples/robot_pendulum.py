"""Save owned q/v/energy from the fixed-base robot CPU reference scenario."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from phymeo.experiments.rigid_cloth import provenance, save_run
from phymeo.experiments.robot import run_pendulum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=1 / 2000)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or Path("artifacts") / (
        "robot-pendulum-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    if destination.exists():
        parser.error(f"output already exists: {destination}")
    source = provenance()
    try:
        run = run_pendulum(steps=args.steps, dt=args.dt, count=args.count)
        save_run(run, destination, source=source)
    except (ValueError, TypeError, RuntimeError) as error:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "failure.json").write_text(
            json.dumps({"status": "failed", "error": str(error), "provenance": source}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        raise SystemExit(str(error)) from error
    print(json.dumps(run.metrics, indent=2))
    print(f"saved {destination}")


if __name__ == "__main__":
    main()
