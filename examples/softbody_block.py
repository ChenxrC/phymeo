"""Save the real bounded tetrahedral loading experiment and numeric trace."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from phymeo.experiments.rigid_cloth import provenance, save_run
from phymeo.experiments.softbody import run_soft_block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or Path("artifacts") / (
        "soft-block-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    if destination.exists():
        parser.error(f"output already exists: {destination}")
    source = provenance()
    try:
        run = run_soft_block(steps=args.steps)
        save_run(run, destination, source=source)
    except (ValueError, TypeError, RuntimeError, AssertionError) as error:
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
