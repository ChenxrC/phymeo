"""Experimental material/contact replay: analytic compressed edge, not production API."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

import phymeo as ph
from phymeo.experiments.rigid_cloth import provenance
from phymeo.solvers._robot_replay_ledger import FrozenRobotLedger


def probe(mode, h):
    robot = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.zeros((1, 3)),
        np.array([[0.0, 0, 0, 1]]),
        np.array([[0.0, 1, 0]]),
        np.array([2.0]),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )
    model = ph.ParticleModel(np.ones(2), np.zeros(3))
    nodes = ph.ParticleState(np.array([[0.0, 0, 0], [0.0, 0.8, 0]]), np.zeros((2, 3)))
    constraints = ph.DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.array([0.01]))
    scene = ph.SceneModel(
        (
            ph.DomainBinding("robot", "robot", robot),
            ph.DomainBinding("nodes", "particles", model),
        )
    )
    state = ph.SceneState({"robot": ph.RobotState(np.zeros(1), np.zeros(1)), "nodes": nodes})
    inputs = ph.SceneInputs({"robot": ph.RobotInputs.zeros(1), "nodes": ph.StepInputs.zeros(2)})
    if mode == "numpy":
        participants = {
            "robot": ph.RobotParticipant(robot),
            "nodes": ph.ClothParticipant(model, constraints, record_material_path=True),
        }
        factory = ph.ArticulatedCouplingProgram
    else:
        participants = {
            "robot": ph.TyxRobotParticipant(robot, mode),
            "nodes": ph.TyxClothParticipant(model, constraints, mode, record_material_path=True),
        }
        factory = ph.TyxArticulatedCouplingProgram
    base = factory(
        scene,
        participants,
        ph.RobotSphereAttachments(np.empty(0, int), np.empty((0, 3)), np.empty(0)),
        [],
    )
    settings = ph.StepSettings(substeps=1, tolerance=1e-8 if mode == "numpy" else 2e-5)
    context = ph.BackendContext(base.capabilities.backend)
    book = ph.MaterialPathLedger(
        model, constraints, nodes, inputs.domains["nodes"], h, settings, context
    )
    robot_book = FrozenRobotLedger(
        robot, state.domains["robot"], inputs.domains["robot"], h, settings, 1.0
    )
    impulse, free_kinetic, history = 0.0, None, []
    with ph.SceneLoadReplay(
        scene,
        base,
        state,
        inputs,
        h,
        channels={"robot": ("effort",), "nodes": ("force",)},
        settings=settings,
        context=context,
        registry=base.registry,
    ) as replay:
        for iteration in range(50):
            requested = np.zeros((2, 3))
            requested[0, 1] = impulse
            trial = replay.evaluate(
                {"robot": {"effort": np.array([-impulse])}, "nodes": {"force": requested}}
            )
            if not trial.result.report.accepted:
                raise AssertionError(trial.result.report.reason)
            sub = trial.result.report.substeps[0]
            final = trial.result.state
            audit = book.require(
                final.domains["nodes"],
                trial.effective_inputs.domains["nodes"],
                requested,
                sub.diagnostics["participants"]["nodes"],
                sweeps=sub.position_iterations,
            )
            robot_audit = robot_book.trial_audit(
                final.domains["robot"],
                trial.effective_inputs.domains["robot"],
                [-Fraction(impulse)],
            )
            assert max(robot_audit.values()) <= settings.tolerance
            v = final.domains["nodes"].velocity[:, 1]
            rv = final.domains["robot"].v[0]
            relative = v[0] - rv
            kinetic = (
                float(robot_book.kinetic(final.domains["robot"].v))
                + audit["energy"]["kinetic_after"]
            )
            if free_kinetic is None:
                free_kinetic = kinetic
            history.append(dict(impulse=impulse, relative_velocity=relative, kinetic=kinetic))
            if abs(relative) <= settings.tolerance:
                break
            # Fixed free response is only an iteration preconditioner. Each
            # candidate includes the complete shared material response.
            impulse = max(0.0, impulse - relative / 1.5)
        else:
            raise AssertionError("prototype outer did not converge")
    # Independent 1-D solution: A=2+c/h^2, b=(1-d)/(h*A),
    # u=(3/2-1/A)*p-b. Neither final lambda nor a free-particle law is substituted.
    step = Fraction(str(h))
    a = 2 + Fraction(1, 100) / step**2
    b = Fraction(1, 5) / (step * a)
    exact_p = b / (Fraction(3, 2) - 1 / a)
    exact_rv = -exact_p / 2
    exact_v = [exact_rv, exact_p / a + b]
    atol = 4 * settings.tolerance
    np.testing.assert_allclose(impulse, float(exact_p), rtol=0, atol=atol)
    np.testing.assert_allclose(v, [float(x) for x in exact_v], rtol=0, atol=atol)
    np.testing.assert_allclose(rv, float(exact_rv), rtol=0, atol=atol)
    gap = final.domains["nodes"].position[0, 1] - final.domains["robot"].q[0]
    assert gap >= -settings.tolerance * h
    momentum = float(v.sum() + 2 * rv)
    assert abs(momentum) <= atol
    energy_after = kinetic + audit["energy"]["elastic_after"][1]
    assert energy_after < 2.0
    if h == 0.1:
        assert kinetic - free_kinetic > 0.04
    return dict(
        h=h,
        impulse=impulse,
        exact_impulse=float(exact_p),
        node_velocities=v.tolist(),
        robot_velocity=rv,
        replays=len(history),
        gap=gap,
        linear_momentum=momentum,
        initial_total_energy=2.0,
        final_total_energy_upper=energy_after,
        final_kinetic=kinetic,
        free_kinetic=free_kinetic,
        kinetic_change_from_free=kinetic - free_kinetic,
        material_residual=audit["max_residual"],
        completed_launches=sub.diagnostics.get("execution", {}).get("completed_launches", 0),
        iterations=history,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["numpy", "interpret", "cuda"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new path")
    result = dict(
        source=provenance(),
        mode=args.mode,
        scope="experimental compressed two-node edge against one prismatic plane; normal-only prototype, not a production contact program or full physical certification",
        cases=[probe(args.mode, h) for h in (0.1, 0.05, 0.025)],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "cases": [
                    {k: v for k, v in c.items() if k != "iterations"} for c in result["cases"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
