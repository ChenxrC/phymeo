"""Serial fixed-configuration two-robot contact solves, not a cloth handoff."""

import argparse
import hashlib
import json
import platform
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np
from benchmark_surface_foundation import observe

from phymeo import (
    BackendContext,
    MultiDomainPointOperator,
    ParticleModel,
    ParticlePointPort,
    ParticleState,
    RobotInputs,
    RobotPointPort,
    RobotState,
    StepInputs,
    StepSettings,
    TyxClothParticipant,
    TyxRobotParticipant,
    pendulum_chain,
    solve_contact_impulses,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("numpy", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src/phymeo").rglob("*.py")) + [
        Path(__file__).resolve(),
        root / "scripts/benchmark_surface_foundation.py",
    ]
    hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    with ExitStack() as resources:
        ports, workspaces = {}, {}
        for name in ("A", "B", "cloth"):
            if name == "cloth":
                model = ParticleModel(np.array([4.0]), np.zeros(3))
                state = ParticleState(np.array([[0.0, -1, 0]]), np.zeros((1, 3)))
                inputs, indices, local = StepInputs.zeros(1), np.array([0]), None
                preload = np.array([[0.0, -0.005, 0]])
                if args.mode == "numpy":
                    base = ParticlePointPort(model, state, indices, domain=name)
                    ports[name] = ParticlePointPort(
                        model, base.apply_impulses(preload), indices, domain=name
                    )
                else:
                    factory = TyxClothParticipant(model, None, "cuda")
            else:
                model, _ = pendulum_chain(2)
                model = replace(model, gravity=np.zeros(3))
                state, inputs = RobotState(np.zeros(2), np.zeros(2)), RobotInputs.zeros(2)
                indices, local = np.array([1]), np.array([[0.0, -0.5, 0]])
                preload = np.array([[0.02 if name == "A" else -0.02, 0, 0]])
                if args.mode == "numpy":
                    base = RobotPointPort(model, state, indices, local, domain=name)
                    ports[name] = RobotPointPort(
                        model, base.apply_impulses(preload), indices, local, domain=name
                    )
                else:
                    factory = TyxRobotParticipant(model, "cuda")
            if args.mode != "numpy":
                ws = factory.create_workspace(
                    name,
                    state,
                    inputs,
                    0.01,
                    StepSettings(tolerance=2e-6),
                    BackendContext("tyx:cuda", np.float64),
                )
                resources.callback(ws.close)
                workspaces[name] = ws
                ws.predict()
                ws.prepare_velocity()
                ws.apply_velocity_impulses(indices, local, preload)
                ports[name] = ws.point_port(indices, local)
        graph = MultiDomainPointOperator(ports, [("A", 0), ("B", 0)], [("cloth", 0), ("cloth", 0)])
        result = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mode": args.mode,
            "source_hashes": hashes,
            "scope": "two2-DOFrobots+oneclothnode,2closedcontacts,fixedconfiguration",
            "setup_included": False,
            "cases": {},
            "not_measured": ["time step", "CCD", "cold process", "peak memory", "bytes", "Warp"],
        }
        result["execution_before"] = {
            name: dict(ws.diagnostics()["execution"]) for name, ws in workspaces.items()
        }
        for name, mu, expected in (("sticking", 0.8, 0.0), ("sliding", 0.0001, -0.019984)):
            iterations, residuals, energies = [], [], []

            def operation():
                tolerance = 1e-10 if args.mode == "numpy" else 2e-5
                response_tolerance = 1e-12 if args.mode == "numpy" else 2e-5
                value = solve_contact_impulses(
                    graph,
                    np.array([[-1.0, 0, 0], [1.0, 0, 0]]),
                    np.full(2, mu),
                    np.full(2, mu),
                    velocity_tolerance=tolerance,
                    response_tolerance=response_tolerance,
                )
                np.testing.assert_allclose(
                    value.normal_impulses, [0.02, 0.02], atol=2e-5, rtol=2e-5
                )
                np.testing.assert_allclose(
                    value.states["cloth"].velocity, [[0, expected, 0]], atol=2e-5, rtol=2e-5
                )
                if value.residual_mps > tolerance or value.energy_change >= 0:
                    raise RuntimeError("joint contact physical validation failed")
                iterations.append(value.iterations)
                residuals.append(value.residual_mps)
                energies.append(value.energy_change)

            measurement = observe(operation, 10, 3)
            measurement.update(
                iterations_min_max=[min(iterations), max(iterations)],
                max_residual_mps=max(residuals),
                max_energy_change=max(energies),
            )
            result["cases"][name] = measurement
        result["execution_after"] = {
            name: dict(ws.diagnostics()["execution"]) for name, ws in workspaces.items()
        }
        for name in workspaces:
            if (
                result["execution_after"][name]["completed_launches"]
                <= result["execution_before"][name]["completed_launches"]
            ):
                raise RuntimeError("one contact domain did not execute")
    for name, digest in hashes.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("source changed during observation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved serial contact observations to {args.output}")


if __name__ == "__main__":
    main()
