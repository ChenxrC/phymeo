"""Serial full replay observations; geometry/contact outer solves are absent."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from benchmark_surface_foundation import observe

from phymeo import (
    ArticulatedCouplingProgram,
    BackendContext,
    ClothParticipant,
    DomainBinding,
    ParticleModel,
    ParticleState,
    RobotInputs,
    RobotModel,
    RobotParticipant,
    RobotSphereAttachments,
    RobotState,
    SceneInputs,
    SceneLoadReplay,
    SceneModel,
    SceneState,
    StepInputs,
    StepSettings,
    TyxArticulatedCouplingProgram,
    TyxClothParticipant,
    TyxRobotParticipant,
)
from phymeo.experiments.rigid_cloth import provenance


def make_replay(mode):
    robot = RobotModel(
        parents=np.array([-1]),
        joint_types=("prismatic",),
        joint_position=np.zeros((1, 3)),
        joint_orientation=np.array([[0.0, 0, 0, 1]]),
        joint_axis=np.array([[1.0, 0, 0]]),
        mass=np.array([2.0]),
        com=np.zeros((1, 3)),
        inertia_diagonal=np.full((1, 3), 0.1),
        gravity=np.zeros(3),
    )
    node = ParticleModel(np.ones(1), np.zeros(3))
    scene = SceneModel(
        (DomainBinding("robot", "robot", robot), DomainBinding("node", "particles", node))
    )
    state = SceneState(
        {
            "robot": RobotState(np.array([0.2]), np.array([0.3])),
            "node": ParticleState(np.zeros((1, 3)), np.zeros((1, 3))),
        }
    )
    inputs = SceneInputs({"robot": RobotInputs.zeros(1), "node": StepInputs.zeros(1)})
    attachments = RobotSphereAttachments(np.empty(0, int), np.empty((0, 3)), np.empty(0))
    if mode == "numpy":
        program = ArticulatedCouplingProgram(
            scene,
            {"robot": RobotParticipant(robot), "node": ClothParticipant(node, None)},
            attachments,
            [],
        )
    else:
        program = TyxArticulatedCouplingProgram(
            scene,
            {
                "robot": TyxRobotParticipant(robot, mode),
                "node": TyxClothParticipant(node, None, mode),
            },
            attachments,
            [],
        )
    return SceneLoadReplay(
        scene,
        program,
        state,
        inputs,
        0.01,
        channels={"robot": ("effort",), "node": ("force",)},
        registry=program.registry,
        context=BackendContext(program.capabilities.backend),
        settings=StepSettings(substeps=1, tolerance=2e-6),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["numpy", "interpret", "cpu_cpp", "cuda"], required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1:
        parser.error("samples/repeats must be positive")
    root = Path(__file__).resolve().parents[1]
    files = [
        Path(__file__),
        root / "scripts/benchmark_surface_foundation.py",
        root / "src/phymeo/runtime/load_replay.py",
    ]
    hashes = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }
    launches = []
    source = provenance()
    with make_replay(args.mode) as replay:

        def operation():
            trial = replay.evaluate(
                {"robot": {"effort": np.array([0.04])}, "node": {"force": np.array([[0.1, 0, 0]])}}
            )
            if not trial.result.report.accepted:
                raise RuntimeError(trial.result.report.reason)
            state = trial.result.state
            np.testing.assert_allclose(state.domains["robot"].v, [0.32], atol=2e-5)
            np.testing.assert_allclose(state.domains["robot"].q, [0.2032], atol=2e-5)
            np.testing.assert_allclose(state.domains["node"].velocity, [[0.1, 0, 0]], atol=2e-5)
            np.testing.assert_allclose(state.domains["node"].position, [[0.001, 0, 0]], atol=2e-5)
            if args.mode != "numpy":
                execution = trial.result.report.substeps[0].diagnostics["execution"]
                if execution["completed_launches"] <= 0:
                    raise RuntimeError("missing actual backend execution")
                launches.append(execution["completed_launches"])

        record = observe(operation, args.samples, args.repeats)
        np.testing.assert_array_equal(replay.snapshot().domains["robot"].q, [0.2])
    for name, expected in hashes.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("source changed during measurement")
    data = {
        "scope": "one held-load replay, one prismatic joint plus one particle; input composition, all original stages, ownership, close and assertions included",
        "mode": args.mode,
        "source": source,
        "source_hashes": hashes,
        "record": record,
        "completed_launches_per_call_including_first_and_warmups": launches,
        "not_measured": [
            "contact outer convergence",
            "mesh geometry",
            "CCD",
            "full handoff",
            "Warp",
            "memory",
            "cold process/cache",
            "profiler attribution",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
