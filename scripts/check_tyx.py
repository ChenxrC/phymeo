"""Execute a real integration kernel; exit nonzero if the requested mode cannot run."""

import argparse
import json
from importlib.metadata import version

import numpy as np

from phymeo import ParticleModel, ParticleState, StepInputs, TyxSemiImplicitSolver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=TyxSemiImplicitSolver.MODES)
    parser.add_argument("--require-device", action="store_true")
    parser.add_argument("--cache-root")
    args = parser.parse_args()
    if args.require_device and args.mode != "cuda":
        parser.error("--require-device requires --mode=cuda")
    model = ParticleModel(np.ones(1, dtype=np.float32), np.array([0.0, -10.0, 0.0]))
    state = ParticleState(
        np.array([[0.0, 1.0, 0.0]], dtype=np.float32), np.zeros((1, 3), dtype=np.float32)
    )
    with TyxSemiImplicitSolver(mode=args.mode, cache_root=args.cache_root) as solver:
        result = solver.step(model, state, StepInputs.zeros(1, dtype=np.float32), 0.1)
    if not result.report.accepted:
        raise RuntimeError(result.report.reason)
    np.testing.assert_allclose(result.state.position, [[0.0, 0.9, 0.0]], atol=1e-7)
    np.testing.assert_allclose(result.state.velocity, [[0.0, -1.0, 0.0]], atol=1e-7)
    print(
        json.dumps(
            {
                "available": True,
                "backend": result.report.backend,
                "tyx_version": version("tyx-compute"),
                "dtype": "float32",
                "executed_nonempty_kernel": True,
                "position_y": float(result.state.position[0, 1]),
            }
        )
    )


if __name__ == "__main__":
    main()
