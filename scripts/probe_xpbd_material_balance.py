"""Diagnostic XPBD momentum and final-multiplier counterexamples, not a solver."""

import argparse
import json
from pathlib import Path

import numpy as np

from phymeo import (
    BackendContext,
    ClothParticipant,
    DistanceConstraints,
    ParticleModel,
    ParticleState,
    StepInputs,
    StepSettings,
    TyxClothParticipant,
)
from phymeo.experiments.rigid_cloth import provenance


def evaluate(mode, name, x, v, edges, rest, compliance, h):
    model = ParticleModel(np.ones(len(x)), np.zeros(3))
    constraints = DistanceConstraints(
        np.array(edges), np.array(rest, float), np.array(compliance, float)
    )
    participant = (
        ClothParticipant(model, constraints)
        if mode == "numpy"
        else TyxClothParticipant(model, constraints, mode)
    )
    initial = ParticleState(np.array(x, float), np.array(v, float))
    tolerance = 1e-8 if mode == "numpy" else 2e-5
    workspace = participant.create_workspace(
        "cloth",
        initial,
        StepInputs.zeros(len(x)),
        h,
        StepSettings(substeps=1, tolerance=tolerance, position_iterations=1000),
        BackendContext(participant.capabilities.backend),
    )
    try:
        workspace.predict()
        predicted_x, predicted_v = workspace.configuration(), workspace.velocity()
        for iteration in range(1, 1001):
            workspace.position_sweep()
            metrics = workspace.position_metrics()
            if metrics.max_residual <= tolerance:
                break
        else:
            raise RuntimeError("material projection failed to converge")
        workspace.prepare_velocity()
        final, diagnostics = workspace.result(), workspace.diagnostics()
        multipliers = diagnostics["distance_multipliers"]
        delta = (
            final.position[constraints.indices[:, 0]] - final.position[constraints.indices[:, 1]]
        )
        direction = delta / np.linalg.norm(delta, axis=1)[:, None]
        guessed = np.zeros_like(final.velocity)
        for k, (i, j) in enumerate(edges):
            guessed[i] += multipliers[k] * direction[k] / h
            guessed[j] -= multipliers[k] * direction[k] / h

        def energy(state):
            kinetic = float(np.sum(state.velocity**2) / 2)
            if np.all(constraints.compliance > 0):
                d = (
                    state.position[constraints.indices[:, 0]]
                    - state.position[constraints.indices[:, 1]]
                )
                strain = np.linalg.norm(d, axis=1) - constraints.rest_length
                elastic = float(np.sum(strain**2 / (2 * constraints.compliance)))
            else:
                elastic = None  # Hard constraints do not have finite compliance energy.
            return dict(kinetic=kinetic, elastic=elastic)

        result = dict(
            name=name,
            mode=mode,
            h=h,
            iterations=iteration,
            tolerance=tolerance,
            residual=metrics.max_residual,
            initial_position=initial.position.tolist(),
            initial_velocity=initial.velocity.tolist(),
            final_position=final.position.tolist(),
            final_velocity=final.velocity.tolist(),
            initial_linear=np.sum(initial.velocity, axis=0).tolist(),
            final_linear=np.sum(final.velocity, axis=0).tolist(),
            initial_angular=np.cross(initial.position, initial.velocity).sum(0).tolist(),
            final_angular=np.cross(final.position, final.velocity).sum(0).tolist(),
            initial_energy=energy(initial),
            final_energy=energy(final),
            final_gradient_impulse_error=float(
                np.max(np.abs((final.position - predicted_x) / h - guessed))
            ),
            velocity_reconstruction_error=float(
                np.max(np.abs(final.velocity - predicted_v - (final.position - predicted_x) / h))
            ),
            edges=edges,
            rest=rest,
            compliance=compliance,
            execution=dict(diagnostics.get("execution", {})),
        )
        np.testing.assert_allclose(result["final_linear"], result["initial_linear"], atol=2e-5)
        if mode != "numpy":
            assert result["execution"]["completed_launches"] > 0
        if name == "rotating_hard_edge":
            expected = 2 / np.sqrt(1 + h * h)
            np.testing.assert_allclose(result["final_angular"][2], expected, atol=2e-6, rtol=0)
            assert 2 - result["final_angular"][2] > 1e-4
            result["independent_expected_final_Lz"] = expected
        else:
            assert result["final_gradient_impulse_error"] > 0.05
        return result
    finally:
        workspace.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["numpy", "interpret", "cuda"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new path")
    rows = [
        evaluate(
            args.mode,
            "rotating_hard_edge",
            [[-1, 0, 0], [1, 0, 0]],
            [[0, -1, 0], [0, 1, 0]],
            [[0, 1]],
            [2],
            [0],
            h,
        )
        for h in (0.1, 0.05, 0.025)
    ]
    rows.append(
        evaluate(
            args.mode,
            "shared_nonlinear_edges",
            [[0, 0, 0], [1.2, 0.2, 0], [1.5, 1.4, 0]],
            [[0.2, 0.8, 0], [-0.3, 0.1, 0], [0.1, -0.5, 0]],
            [[0, 1], [1, 2]],
            [1, 1],
            [0.001, 0.002],
            0.1,
        )
    )
    record = dict(
        status="counterexamples_confirmed",
        source=provenance(),
        rows=rows,
        scope="one native material substep, no contacts or robot; not a performance measurement",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(record["status"], args.mode)


if __name__ == "__main__":
    main()
