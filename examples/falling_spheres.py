"""Discrete sphere/plane contact, resting support, and unaffected spherical spin."""

import argparse
import json

import numpy as np

from phymeo import (
    RigidBodyInputs,
    RigidBodyModel,
    RigidBodyState,
    RigidSpherePlaneSolver,
    SphereShapes,
    Stepper,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    args = parser.parse_args()
    dtype, dt, steps = np.dtype(args.dtype), 0.002, 1000
    radii = np.array([0.2, 0.35, 0.15], dtype=dtype)
    inertia = np.zeros((3, 3), dtype=dtype)
    inertia[0] = 0.4 * radii[0] ** 2  # explicit uniform-sphere inertia for this example
    inertia[1] = 0.4 * 2 * radii[1] ** 2
    model = RigidBodyModel(np.array([1, 0.5, 0], dtype=dtype), inertia)
    state = RigidBodyState(
        np.array([[-1, 2, 0], [1, 3, 0], [0, radii[2], 2]], dtype=dtype),
        np.tile(np.array([0, 0, 0, 1], dtype=dtype), (3, 1)),
        np.zeros((3, 3), dtype=dtype),
        np.array([[0, 2, 0], [1, 0, 0], [0, 0, 0]], dtype=dtype),
    )
    initial = state.copy()
    stepper = Stepper(RigidSpherePlaneSolver(SphereShapes(radii)))
    inputs = RigidBodyInputs.zeros(3, dtype=dtype)
    min_gap, max_residual = float("inf"), 0.0
    for frame in range(steps):
        result = stepper.step(model, state, inputs, dt)
        if not result.report.accepted:
            raise RuntimeError(f"Frame {frame}: {result.report}")
        state = result.state
        gap = state.position[:, 1].astype(np.float64) - radii.astype(np.float64)
        min_gap = min(min_gap, float(np.min(gap)))
        if min_gap < 0:
            raise AssertionError("sphere penetrated the plane")
        np.testing.assert_array_equal(state.position[2], initial.position[2])
        np.testing.assert_array_equal(state.orientation[2], initial.orientation[2])
        max_residual = max(max_residual, result.report.max_residual)
    np.testing.assert_array_equal(state.position[:, 1], radii)
    np.testing.assert_array_equal(state.velocity[:, 1], np.zeros(3))
    time = steps * float(dtype.type(dt))
    expected = np.array(
        [
            [0, np.sin(time), 0, np.cos(time)],
            [np.sin(time / 2), 0, 0, np.cos(time / 2)],
            [0, 0, 0, 1],
        ]
    )
    error = float(np.max(np.abs(state.orientation - expected)))
    if error > (2e-5 if dtype == np.float32 else 1e-12):
        raise AssertionError(f"spin changed unexpectedly: {error}")
    print(
        json.dumps(
            {
                "steps": steps,
                "dt": dt,
                "dtype": args.dtype,
                "backend": "numpy",
                "minimum_gap_m": min_gap,
                "max_normalized_residual": max_residual,
                "final_normal_velocities": state.velocity[:, 1].tolist(),
                "spin_quaternion_error": error,
                "contact_law": "discrete sphere-plane, frictionless, restitution=0; no CCD",
            }
        )
    )


if __name__ == "__main__":
    main()
