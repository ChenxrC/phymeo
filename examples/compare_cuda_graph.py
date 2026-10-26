"""Compare native CUDA graph replay with direct launches on identical cloth steps."""

import argparse
import json
import platform
from importlib.metadata import version
from time import perf_counter

import numpy as np

from phymeo import Plane, StepInputs, TyxXPBDSolver, cloth_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=400)
    args = parser.parse_args()
    if args.steps < 2 or args.iterations < 1:
        parser.error("--steps must be at least 2 and --iterations positive")
    patch = cloth_grid(
        4,
        4,
        spacing=0.15,
        density=0.5,
        structural_compliance=1e-3,
        shear_compliance=1e-3,
        pinned=[0, 3],
        origin=(0, 0.3, 0),
        dtype=np.float32,
    )
    options = dict(
        mode="cuda",
        iterations=args.iterations,
        tolerance=1e-5,
        plane=Plane(),
        cache_root=".tyx-cache",
    )
    inputs = StepInputs.zeros(patch.model.particle_count, dtype=np.float32)
    state, times, iterations = patch.initial_state.copy(), {"direct": [], "graph": []}, []
    with (
        TyxXPBDSolver(patch.constraints, **options) as direct,
        TyxXPBDSolver(patch.constraints, cuda_graph=True, **options) as graph,
    ):
        for frame in range(args.steps):
            # Alternate order to reduce a systematic warm-cache/clock advantage.
            ordered = [("direct", direct), ("graph", graph)]
            if frame % 2:
                ordered.reverse()
            results = {}
            for name, solver in ordered:
                start = perf_counter()
                results[name] = solver.step(patch.model, state, inputs, 0.02)
                times[name].append(perf_counter() - start)
                if not results[name].report.accepted:
                    raise RuntimeError(f"frame {frame}, {name}: {results[name].report}")
            a, b = results["direct"], results["graph"]
            np.testing.assert_array_equal(a.state.position, b.state.position)
            np.testing.assert_array_equal(a.state.velocity, b.state.velocity)
            if a.report != b.report:
                raise AssertionError(f"frame {frame}: graph changed the iteration/report contract")
            np.testing.assert_array_equal(
                a.state.position[[0, 3]], patch.initial_state.position[[0, 3]]
            )
            if np.min(a.state.position[:, 1]) < 0:
                raise AssertionError("plane penetration")
            iterations.append(a.report.iterations)
            state = a.state
        colors = direct.color_count
    measured = {
        name: {
            "first_step_seconds": samples[0],
            "later_mean_seconds": float(np.mean(samples[1:])),
            "later_median_seconds": float(np.median(samples[1:])),
            "later_p95_seconds": float(np.percentile(samples[1:], 95)),
        }
        for name, samples in times.items()
    }
    print(
        json.dumps(
            {
                "backend": "tyx:cuda",
                "python": platform.python_version(),
                "numpy": np.__version__,
                "tyx": version("tyx-compute"),
                "steps": args.steps,
                "dt": 0.02,
                "tolerance": 1e-5,
                "particles": patch.model.particle_count,
                "edges": len(patch.constraints.indices),
                "colors": colors,
                "max_iterations": max(iterations),
                "states_and_reports_identical": True,
                "timings": measured,
                "measured_direct_over_graph_ratio": measured["direct"]["later_mean_seconds"]
                / measured["graph"]["later_mean_seconds"],
                "timing_scope": "complete solver step including allocations, transfers, capture, per-iteration host residual checks and cleanup",
                "caveat": "single device and small scene; not a general throughput claim; first steps can include compilation",
            }
        )
    )


if __name__ == "__main__":
    main()
