"""Serial dynamic plane replay observation, including physical acceptance gates."""

import argparse
import hashlib
import json
import traceback
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from phymeo import (
    ArticulatedCouplingProgram,
    BackendContext,
    ClothParticipant,
    DomainBinding,
    ParticleModel,
    ParticlePlaneContactProgram,
    ParticlePlaneContacts,
    ParticleState,
    RobotInputs,
    RobotParticipant,
    RobotSphereAttachments,
    RobotState,
    SceneInputs,
    SceneModel,
    SceneState,
    SceneStepper,
    StepInputs,
    StepSettings,
    TyxArticulatedCouplingProgram,
    TyxClothParticipant,
    TyxRobotParticipant,
    pendulum_chain,
)
from phymeo.experiments.rigid_cloth import provenance


def make_case(mode):
    robot, _ = pendulum_chain(1)
    robot = replace(robot, gravity=np.zeros(3))
    node = ParticleModel(np.ones(1), np.array([0.0, -9.81, 0]))
    scene = SceneModel(
        (DomainBinding("robot", "robot", robot), DomainBinding("cloth", "particles", node))
    )
    state = SceneState(
        {
            "robot": RobotState(np.array([0.2]), np.array([0.3])),
            "cloth": ParticleState(np.zeros((1, 3)), np.zeros((1, 3))),
        }
    )
    inputs = SceneInputs(
        {"robot": RobotInputs.zeros(1), "cloth": StepInputs(np.array([[-20.0, 0, 0]]))}
    )
    attachments = RobotSphereAttachments(np.empty(0, int), np.empty((0, 3)), np.empty(0))
    if mode == "numpy":
        base = ArticulatedCouplingProgram(
            scene,
            {"robot": RobotParticipant(robot), "cloth": ClothParticipant(node, None)},
            attachments,
            [],
        )
    else:
        base = TyxArticulatedCouplingProgram(
            scene,
            {
                "robot": TyxRobotParticipant(robot, mode),
                "cloth": TyxClothParticipant(node, None, mode),
            },
            attachments,
            [],
        )
    contact = ParticlePlaneContacts(
        "cloth", np.array([0]), np.array([[1.0, 0, 0]]), np.zeros(1), np.array([0.6])
    )
    program = ParticlePlaneContactProgram(scene, base, (contact,), registry=base.registry)
    runtime = SceneStepper(
        scene,
        program,
        StepSettings(substeps=1, tolerance=2e-5),
        BackendContext(base.capabilities.backend),
        base.registry,
    )
    return runtime, state, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["numpy", "interpret", "cpu_cpp", "cuda"], required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1 or args.output.exists():
        parser.error("positive samples/repeats and a new output path are required")
    source = provenance()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record = dict(
        scope="one full dynamic plane step from the same initial snapshot, one revolute joint plus one supported node; setup/solve/replay/close/validation/assertions included; outer factory construction excluded",
        mode=args.mode,
        source=source,
        script_sha256=script_hash,
        status="running",
        first_call_seconds=None,
        warmups=2,
        batches=[],
        successful_calls_including_warmups=0,
        accepted_measured_calls=0,
        failed_calls=0,
        calls=[],
        not_measured=[
            "robot contact",
            "mesh CCD",
            "full handoff",
            "Warp",
            "peak memory",
            "cold process/cache",
            "profiler attribution",
        ],
    )
    phase = "construction"
    try:
        runtime, initial, inputs = make_case(args.mode)
        with runtime:

            def operation():
                result = runtime.step(initial, inputs, 0.01)
                if not result.report.accepted:
                    raise RuntimeError(result.report.reason)
                state = result.state
                np.testing.assert_allclose(state.domains["robot"].q, [0.203], atol=2e-5)
                np.testing.assert_allclose(state.domains["robot"].v, [0.3], atol=2e-5)
                np.testing.assert_allclose(
                    state.domains["cloth"].position, np.zeros((1, 3)), atol=2e-7
                )
                np.testing.assert_allclose(
                    state.domains["cloth"].velocity, np.zeros((1, 3)), atol=2e-5
                )
                report = result.report.substeps[0]
                diag = report.diagnostics["plane_contact"]
                np.testing.assert_allclose(diag["world_impulses"], [[0.2, 0.0981, 0]], atol=2e-5)
                execution = report.diagnostics["execution"]
                if args.mode != "numpy" and execution["completed_launches"] <= 0:
                    raise RuntimeError("missing actual backend execution")
                record["calls"].append(
                    dict(
                        replays=diag["replays"],
                        position_iterations=report.position_iterations,
                        residual=report.max_residual,
                        contact_work_j=diag["contact_work_j"],
                        **execution,
                    )
                )
                record["successful_calls_including_warmups"] += 1

            phase = "first"
            start = perf_counter()
            operation()
            record["first_call_seconds"] = perf_counter() - start
            phase = "warmup"
            for _ in range(2):
                operation()
            for repeat in range(args.repeats):
                phase = f"batch{repeat}"
                samples = []
                batch = dict(seconds=samples)
                record["batches"].append(batch)
                for _ in range(args.samples):
                    start = perf_counter()
                    operation()
                    samples.append(perf_counter() - start)
                    record["accepted_measured_calls"] += 1
                batch.update(
                    median=float(np.median(samples)), p95=float(np.percentile(samples, 95))
                )
            np.testing.assert_array_equal(initial.domains["robot"].q, [0.2])
            np.testing.assert_array_equal(initial.domains["cloth"].position, np.zeros((1, 3)))
        record["status"] = "complete"
    except Exception as error:
        record.update(
            status="failed",
            failed_calls=1,
            failure_phase=phase,
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
    finally:
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != script_hash:
            record.update(status="source_changed")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(record["status"], args.output)
    return 0 if record["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
