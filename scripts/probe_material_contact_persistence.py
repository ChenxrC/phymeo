"""Probe whether the closed-at-start material-contact prototype persists at rest."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

import phymeo as ph
from phymeo.experiments.rigid_cloth import provenance
from phymeo.solvers._robot_replay_ledger import FrozenRobotLedger
from phymeo.solvers.particle_plane_contact import _BorrowedProgram


def probe(mode, h, stopping_rule="velocity"):
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
    model = ph.ParticleModel(np.ones(2), np.array([0.0, -1, 0]))
    nodes = ph.ParticleState(np.array([[0.0, 0, 0], [0.0, 255 / 256, 0]]), np.zeros((2, 3)))
    constraints = ph.DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.array([1 / 256]))
    scene = ph.SceneModel(
        (ph.DomainBinding("robot", "robot", robot), ph.DomainBinding("nodes", "particles", model))
    )
    state = ph.SceneState({"robot": ph.RobotState(np.zeros(1), np.zeros(1)), "nodes": nodes})
    inputs = ph.SceneInputs(
        {
            "robot": ph.RobotInputs(np.array([2.0]), np.zeros((1, 3)), np.zeros((1, 3))),
            "nodes": ph.StepInputs.zeros(2),
        }
    )
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
    length_tol = settings.tolerance * h
    record = dict(
        h=h,
        target_time=2.0,
        target_steps=round(2 / h),
        length_tolerance=length_tol,
        expected_impulse_per_step=2 * h,
        frames=[],
        status="running",
    )
    try:
        for step in range(round(2 / h)):
            before_gap = state.domains["nodes"].position[0, 1] - state.domains["robot"].q[0]
            if abs(before_gap) > length_tol:
                record.update(
                    status="initial_closure_rejected", failed_step=step, initial_gap=before_gap
                )
                break
            book = ph.MaterialPathLedger(
                model,
                constraints,
                state.domains["nodes"],
                inputs.domains["nodes"],
                h,
                settings,
                context,
            )
            rb = FrozenRobotLedger(
                robot, state.domains["robot"], inputs.domains["robot"], h, settings, 1.0
            )
            impulse = 0.0
            with ph.SceneLoadReplay(
                scene,
                _BorrowedProgram(base),
                state,
                inputs,
                h,
                channels={"robot": ("effort",), "nodes": ("force",)},
                settings=settings,
                context=context,
                registry=base.registry,
            ) as replay:
                for iteration in range(50):
                    force = np.zeros((2, 3))
                    force[0, 1] = impulse
                    trial = replay.evaluate(
                        {"robot": {"effort": np.array([-impulse])}, "nodes": {"force": force}}
                    )
                    if not trial.result.report.accepted:
                        raise AssertionError(trial.result.report.reason)
                    sub = trial.result.report.substeps[0]
                    final = trial.result.state
                    audit = book.require(
                        final.domains["nodes"],
                        trial.effective_inputs.domains["nodes"],
                        force,
                        sub.diagnostics["participants"]["nodes"],
                        sweeps=sub.position_iterations,
                    )
                    ra = rb.trial_audit(
                        final.domains["robot"],
                        trial.effective_inputs.domains["robot"],
                        [-Fraction(impulse)],
                    )
                    assert max(ra.values()) <= settings.tolerance
                    u = final.domains["nodes"].velocity[0, 1] - final.domains["robot"].v[0]
                    trial_gap = final.domains["nodes"].position[0, 1] - final.domains["robot"].q[0]
                    if abs(u) <= settings.tolerance and (
                        stopping_rule == "velocity" or trial_gap >= -length_tol
                    ):
                        break
                    impulse = max(0.0, impulse - u / 1.5)
                else:
                    record.update(
                        status="outer_iteration_limit",
                        failed_step=step,
                        last_relative_velocity=float(u),
                        last_gap=float(trial_gap),
                    )
                    return record
            gap = final.domains["nodes"].position[0, 1] - final.domains["robot"].q[0]
            record["frames"].append(
                dict(
                    step=step,
                    impulse=impulse,
                    replays=iteration + 1,
                    relative_velocity=u,
                    gap=gap,
                    robot_q=float(final.domains["robot"].q[0]),
                    max_node_drift=float(
                        np.max(np.abs(final.domains["nodes"].position - nodes.position))
                    ),
                    material_residual=audit["max_residual"],
                )
            )
            if gap < -length_tol:
                record.update(status="final_penetration_rejected", failed_step=step)
                break
            state = final
        else:
            record["status"] = "completed_window"
    finally:
        base.close()
    record["completed_steps"] = int(sum(f["gap"] >= -length_tol for f in record["frames"]))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["numpy", "interpret", "cuda"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stopping-rule", choices=["velocity", "geometry"], default="velocity")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    result = dict(
        source=provenance(),
        mode=args.mode,
        stopping_rule=args.stopping_rule,
        scope="experimental prototype reuse only; a rejected persistence window is evidence against immediate production promotion, not a failure of the already shipped free-node program",
        cases=[probe(args.mode, h, args.stopping_rule) for h in (1 / 16, 1 / 32)],
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "cases": [{k: v for k, v in c.items() if k != "frames"} for c in result["cases"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
