"""Save a verified cloth/sphere trajectory or the full comparison experiment."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from phymeo.experiments.rigid_cloth import provenance, run_scene, run_suite, save_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--dt", type=float, default=1 / 120)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument(
        "--mode",
        choices=("numpy", "numpy_colored", "interpret", "cpu_cpp", "cuda"),
        default="numpy",
    )
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/tyx-cache"))
    parser.add_argument("--sphere-mass", type=float, default=0.05)
    parser.add_argument("--one-way", action="store_true")
    parser.add_argument(
        "--suite",
        action="store_true",
        help="repeat, one-way, half-step and two mass controls (FP64)",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    destination = args.output or Path("artifacts") / (
        "rigid-cloth-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    if destination.exists():
        parser.error(f"output already exists: {destination}")
    if args.suite and (args.one_way or args.dtype != "float64" or args.mode != "numpy"):
        parser.error("--suite requires float64 and selects its own one-way control")
    source = provenance()

    def progress(frame, report):
        if frame % 30 == 0 or frame == args.frames:
            print(
                f"frame {frame}/{args.frames}: accepted, residual={report.max_residual:.3e}",
                flush=True,
            )

    try:
        arguments = dict(
            frames=args.frames,
            dt=args.dt,
            substeps=args.substeps,
            iterations=args.iterations,
            tolerance=args.tolerance,
            dtype=np.dtype(args.dtype),
            mode=args.mode,
            cache_root=args.cache_root,
            sphere_mass=args.sphere_mass,
        )
        if args.suite:

            def suite_progress(name, frame, report):
                if frame % 30 == 0:
                    print(
                        f"{name} frame {frame}: accepted, residual={report.max_residual:.3e}",
                        flush=True,
                    )

            result = run_suite(destination, **arguments, progress=suite_progress)
            summary = {
                key: result[key]
                for key in (
                    "status",
                    "deterministic_repetition",
                    "comparison",
                    "step_halving",
                    "analytic_free_fall",
                )
            }
        else:
            run = run_scene(**arguments, reaction=not args.one_way, progress=progress)
            save_run(run, destination, source=source)
            summary = run.metrics
    except (ValueError, TypeError, RuntimeError, AssertionError) as error:
        destination.mkdir(parents=True, exist_ok=True)
        if not (destination / "failure.json").exists():
            (destination / "failure.json").write_text(
                json.dumps(
                    {"status": "failed", "error": str(error), "provenance": source}, indent=2
                )
                + "\n",
                encoding="utf-8",
            )
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, indent=2))
    print(f"saved {destination}")


if __name__ == "__main__":
    main()
