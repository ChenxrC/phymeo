"""Serial three-domain impulse graph observations with analytic validation."""

import argparse
import hashlib
import json
import platform
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from benchmark_surface_foundation import observe

from phymeo import (
    BackendContext,
    MultiDomainPointOperator,
    ParticleModel,
    ParticlePointPort,
    ParticleState,
    StepInputs,
    StepSettings,
    TyxClothParticipant,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("numpy", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = [
        "src/phymeo/coupling/domain_graph.py",
        "src/phymeo/coupling/surface_port.py",
        "src/phymeo/coupling/point_port.py",
        "src/phymeo/backends/tyx_participants.py",
        "scripts/benchmark_domain_point_graph.py",
        "scripts/benchmark_surface_foundation.py",
    ]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}
    with ExitStack() as resources:
        ports, workspaces = {}, {}
        for name, inverse_mass in (("A", 1), ("B", 2), ("cloth", 4)):
            model = ParticleModel(np.array([float(inverse_mass)]), np.zeros(3))
            state = ParticleState(np.zeros((1, 3)), np.zeros((1, 3)))
            if args.mode == "numpy":
                ports[name] = ParticlePointPort(model, state, [0], domain=name)
            else:
                factory = TyxClothParticipant(model, None, "cuda")
                ws = factory.create_workspace(
                    name,
                    state,
                    StepInputs.zeros(1),
                    0.01,
                    StepSettings(tolerance=2e-6),
                    BackendContext("tyx:cuda", np.float64),
                )
                resources.callback(ws.close)
                ws.predict()
                ports[name], workspaces[name] = ws.point_port(np.array([0])), ws
        operator = MultiDomainPointOperator(
            ports, [("A", 0), ("B", 0)], [("cloth", 0), ("cloth", 0)]
        )
        p = np.array([[0.1, 0.2, 0], [-0.2, 0.1, 0]])
        expected = np.array([[5.0, 4], [4.0, 6]]) @ p

        def operation():
            np.testing.assert_allclose(operator.response(p), expected, atol=2e-5, rtol=2e-5)
            states = operator.apply_impulses(p)
            np.testing.assert_allclose(states["A"].velocity, p[:1], atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(states["B"].velocity, 2 * p[1:], atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(
                states["cloth"].velocity, -4 * (p[:1] + p[1:]), atol=2e-5, rtol=2e-5
            )

        before = {name: dict(ws.diagnostics()["execution"]) for name, ws in workspaces.items()}
        measurement = observe(operation, 20, 3)
        after = {name: dict(ws.diagnostics()["execution"]) for name, ws in workspaces.items()}
        for name in workspaces:
            if after[name]["completed_launches"] <= before[name]["completed_launches"]:
                raise RuntimeError("a domain did not execute")
    for name, digest in hashes.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("source changed during observation")
    result = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mode": args.mode,
        "scope": "3 independent one-node domains/2 edges; response+candidate+assertions",
        "source_hashes": hashes,
        "measurement": measurement,
        "execution_before": before,
        "execution_after": after,
        "not_measured": ["full step", "cold process", "peak memory", "bytes", "Warp"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved serial graph observation to {args.output}")


if __name__ == "__main__":
    main()
