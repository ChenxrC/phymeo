"""Save independently audited rigid/cloth motion through the unified scene runtime."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from phymeo.experiments.rigid_cloth import provenance, run_scene, save_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--dt", type=float, default=1 / 120)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument(
        "--mode",
        choices=("numpy", "numpy_colored", "interpret", "cpu_cpp", "cuda"),
        default="numpy",
    )
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/tyx-cache"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    numpy_mode = args.mode in ("numpy", "numpy_colored")
    dtype = args.dtype or ("float64" if numpy_mode else "float32")
    tolerance = args.tolerance if args.tolerance is not None else (1e-7 if numpy_mode else 1e-6)
    destination = args.output or Path("artifacts") / (
        "coupled-runtime-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    if destination.exists():
        parser.error(f"output already exists: {destination}")
    source = provenance()

    def progress(frame, report):
        if frame % 30 == 0 or frame == args.frames:
            print(
                f"frame {frame}/{args.frames}: accepted, residual={report.max_residual:.3e}",
                flush=True,
            )

    try:
        run = run_scene(
            frames=args.frames,
            dt=args.dt,
            substeps=args.substeps,
            iterations=args.iterations,
            tolerance=tolerance,
            dtype=np.dtype(dtype),
            mode=args.mode,
            runtime="scene",
            cache_root=args.cache_root,
            progress=progress,
        )
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
