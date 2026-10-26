"""Compare a small colored tyx cloth rollout to the independent FP64 reference."""

import argparse
import json
from time import perf_counter

import numpy as np

from phymeo import (
    DistanceConstraints,
    ParticleModel,
    ParticleState,
    Plane,
    StepInputs,
    TyxXPBDSolver,
    XPBDSolver,
    cloth_grid,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=TyxXPBDSolver.MODES, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args()
    if args.cuda_graph and args.mode != "cuda":
        parser.error("--cuda-graph requires --mode cuda")
    if args.steps < 1 or args.iterations < 1:
        parser.error("--steps and --iterations must be positive")
    pins, dt, tolerance = [0, 3], 0.02, 1e-5
    patch = cloth_grid(
        4,
        4,
        spacing=0.15,
        density=0.5,
        structural_compliance=1e-3,
        shear_compliance=1e-3,
        pinned=pins,
        origin=(0, 0.3, 0),
        dtype=np.float32,
    )
    reference_edges = DistanceConstraints(
        patch.constraints.indices,
        patch.constraints.rest_length.astype(np.float64),
        patch.constraints.compliance.astype(np.float64),
    )
    reference_model = ParticleModel(
        patch.model.inverse_mass.astype(np.float64), patch.model.gravity.astype(np.float64)
    )
    reference = XPBDSolver(reference_edges, iterations=2000, tolerance=1e-8, plane=Plane())
    state = patch.initial_state.copy()
    reference_state = ParticleState(
        state.position.astype(np.float64), state.velocity.astype(np.float64)
    )
    inputs, reference_inputs = StepInputs.zeros(16, dtype=np.float32), StepInputs.zeros(16)
    times, iterations = [], []
    max_residual, max_difference, minimum_height = 0.0, 0.0, float("inf")
    with TyxXPBDSolver(
        patch.constraints,
        mode=args.mode,
        iterations=args.iterations,
        tolerance=tolerance,
        plane=Plane(),
        cache_root=".tyx-cache",
        cuda_graph=args.cuda_graph,
    ) as solver:
        for frame in range(args.steps):
            start = perf_counter()
            result = solver.step(patch.model, state, inputs, dt)
            times.append(perf_counter() - start)
            baseline = reference.step(reference_model, reference_state, reference_inputs, dt)
            if not result.report.accepted or not baseline.report.accepted:
                raise RuntimeError(
                    f"Frame {frame}: tyx={result.report}; reference={baseline.report}"
                )
            state, reference_state = result.state, baseline.state
            np.testing.assert_array_equal(state.position[pins], patch.initial_state.position[pins])
            minimum_height = min(minimum_height, float(np.min(state.position[:, 1])))
            if minimum_height < 0:
                raise AssertionError("plane penetration")
            iterations.append(result.report.iterations)
            max_residual = max(max_residual, result.report.max_residual)
            max_difference = max(
                max_difference,
                float(np.max(np.linalg.norm(state.position - reference_state.position, axis=1))),
            )
        color_count = solver.color_count
    print(
        json.dumps(
            {
                "backend": f"tyx:{args.mode}",
                "cuda_graph": args.cuda_graph,
                "dtype": "float32",
                "steps": args.steps,
                "dt": dt,
                "tolerance": tolerance,
                "particles": 16,
                "edges": 42,
                "colors": color_count,
                "max_iterations": max(iterations),
                "max_normalized_residual": max_residual,
                "minimum_height_m": minimum_height,
                "max_position_difference_from_numpy_fp64_m": max_difference,
                "first_step_seconds_including_possible_compile": times[0],
                "later_steps_mean_seconds": float(np.mean(times[1:])) if len(times) > 1 else None,
                "timing_scope": "tyx step including allocations, transfers, per-iteration host checks",
                "comparison": "nonlinear ordering and precision differ; difference is a measured metric",
                "model": "stretch_shear_without_bending_or_self_collision",
            }
        )
    )


if __name__ == "__main__":
    main()
