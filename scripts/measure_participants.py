"""Serial P5 adapter/contact microbenchmark with an independent slider solution."""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from phymeo import (
    ArticulatedCouplingProgram,
    ClothParticipant,
    DomainBinding,
    NodeRobotContact,
    ParticleModel,
    ParticleState,
    RobotInputs,
    RobotModel,
    RobotParticipant,
    RobotSphereAttachments,
    RobotState,
    SceneInputs,
    SceneModel,
    SceneState,
    StepInputs,
    StepSettings,
    make_articulated_runtime,
)
from phymeo.experiments.rigid_cloth import provenance


def build(count):
    model = RobotModel(
        parents=np.array([-1]),
        joint_types=("prismatic",),
        joint_position=np.zeros((1, 3)),
        joint_orientation=np.array([[0.0, 0, 0, 1]]),
        joint_axis=np.array([[0.0, 1, 0]]),
        mass=np.array([2.0]),
        com=np.zeros((1, 3)),
        inertia_diagonal=np.full((1, 3), 0.01),
        gravity=np.zeros(3),
    )
    bindings = [DomainBinding("robot", "robot", model)]
    participants = {"robot": RobotParticipant(model)}
    states = {"robot": RobotState(np.zeros(1), np.zeros(1))}
    inputs = {"robot": RobotInputs.zeros(1)}
    contacts = []
    for index in range(count):
        name = f"nodes{index}"
        nodes = ParticleModel(np.ones(1), np.zeros(3))
        bindings.append(DomainBinding(name, "particles", nodes))
        participants[name] = ClothParticipant(nodes, None)
        states[name] = ParticleState(np.array([[0.0, 0.09, 0]]), np.zeros((1, 3)))
        inputs[name] = StepInputs.zeros(1)
        contacts.append(NodeRobotContact(name, np.array([0]), np.array([0])))
    program = ArticulatedCouplingProgram(
        SceneModel(tuple(bindings)),
        participants,
        RobotSphereAttachments(np.array([0]), np.zeros((1, 3)), np.array([0.1])),
        contacts,
    )
    runtime = make_articulated_runtime(program, settings=StepSettings(substeps=1, tolerance=1e-11))
    return runtime, SceneState(states), SceneInputs(inputs)


def stats(values):
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}


def measure(count):
    start = perf_counter()
    runtime, state, inputs = build(count)
    construction_ms = (perf_counter() - start) * 1000
    maximum_error, maximum_residual, max_pi, max_vi = 0.0, 0.0, 0, 0
    phases, repeats = {}, []
    expected_q = -count * 0.01 / (2 + count)

    def call():
        nonlocal maximum_error, maximum_residual, max_pi, max_vi
        start = perf_counter()
        result = runtime.step(state, inputs, 0.01)
        duration = (perf_counter() - start) * 1000
        if not result.report.accepted:
            raise RuntimeError(result.report.reason)
        robot = result.state.domains["robot"]
        error = max(abs(robot.q[0] - expected_q), abs(robot.v[0] - expected_q / 0.01))
        for name, node in result.state.domains.items():
            if name != "robot":
                error = max(
                    error,
                    abs(node.position[0, 1] - (0.1 + expected_q)),
                    abs(node.velocity[0, 1] - (0.01 + expected_q) / 0.01),
                )
        if error > 2e-9:
            raise RuntimeError(f"independent slider solution error {error}")
        maximum_error = max(maximum_error, error)
        report = result.report.substeps[0]
        maximum_residual = max(maximum_residual, report.max_residual)
        max_pi, max_vi = (
            max(max_pi, report.position_iterations),
            max(max_vi, report.velocity_iterations),
        )
        return duration, report

    try:
        first_ms, _ = call()
        for _ in range(200):
            call()
        for repeat in range(3):
            times = []
            for _ in range(100):
                duration, report = call()
                times.append(duration)
                for name, value in report.timings.items():
                    phases.setdefault(name, []).append(value * 1000)
            repeats.append({"repeat": repeat + 1, "calls": 100, **stats(times)})
    finally:
        runtime.close()
    return {
        "node_domains": count,
        "dof": 1,
        "pairs": count,
        "construction_ms": construction_ms,
        "first_step_ms": first_ms,
        "repeats": repeats,
        "phase_timings": {k: stats(v) for k, v in phases.items()},
        "accepted_measured_steps": 300,
        "max_residual": maximum_residual,
        "max_analytic_error": maximum_error,
        "max_position_iterations": max_pi,
        "max_velocity_iterations": max_vi,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    report = {
        "provenance": provenance(),
        "measurement_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "protocol": {
            "serial": True,
            "warmups": 200,
            "repeats": 3,
            "calls_per_repeat": 100,
            "dt": 0.01,
            "substeps": 1,
            "tolerance": 1e-11,
            "input": "restore same initial state per call; robot mass2; each node mass1; y=.09; radius=.1; zero forces/gravity/velocities",
            "timing": "full production step includes all checks and diagnostics; independent analytic check follows timer",
        },
        "limitations": [
            "no continuous scene throughput",
            "no standalone profiling or peak memory/allocation counts",
            "no controlled BLAS threads or OS background scheduling",
            "no tyx/Warp comparison",
            "first call is after Python imports",
        ],
        "workloads": [],
    }
    try:
        for count in (1, 2):
            report["workloads"].append(measure(count))
        report.update(status="complete", failed_calls=0)
    except Exception as error:
        report.update(status="failed", failed_calls=1, error=str(error))
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(report["workloads"], indent=2))


if __name__ == "__main__":
    main()
