"""Serial full material-contact outer measurements; no performance optimization."""

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np

import phymeo as ph
from phymeo.experiments.rigid_cloth import provenance


def make_case(mode):
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
    nodes = ph.ParticleModel(np.ones(2), np.zeros(3))
    edges = ph.DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.array([0.01]))
    scene = ph.SceneModel(
        (ph.DomainBinding("robot", "robot", robot), ph.DomainBinding("nodes", "particles", nodes))
    )
    if mode == "numpy":
        participants = {
            "robot": ph.RobotParticipant(robot),
            "nodes": ph.ClothParticipant(nodes, edges, record_material_path=True),
        }
        factory = ph.ArticulatedCouplingProgram
    else:
        participants = {
            "robot": ph.TyxRobotParticipant(robot, mode),
            "nodes": ph.TyxClothParticipant(nodes, edges, mode, record_material_path=True),
        }
        factory = ph.TyxArticulatedCouplingProgram
    base = factory(
        scene,
        participants,
        ph.RobotSphereAttachments(np.empty(0, int), np.empty((0, 3)), np.empty(0)),
        [],
    )
    contacts = (
        ph.RobotNodePlaneContacts(
            "nodes",
            np.array([0]),
            np.array([0]),
            np.zeros((1, 3)),
            np.array([[0.0, 1, 0]]),
            np.zeros(1),
        ),
    )
    program = ph.InitialRobotMaterialContactProgram(
        scene,
        base,
        contacts,
        materials={"nodes": edges},
        budgets=ph.MaterialContactBudgets(1e-6, 0.001),
        registry=base.registry,
    )
    runtime = ph.SceneStepper(
        scene,
        program,
        ph.StepSettings(
            substeps=1, tolerance=1e-8 if mode == "numpy" else 2e-5, position_iterations=60
        ),
        ph.BackendContext(base.capabilities.backend),
        base.registry,
    )
    state = ph.SceneState(
        {
            "robot": ph.RobotState(np.zeros(1), np.zeros(1)),
            "nodes": ph.ParticleState(np.array([[0.0, 0, 0], [0.0, 0.8, 0]]), np.zeros((2, 3))),
        }
    )
    inputs = ph.SceneInputs({"robot": ph.RobotInputs.zeros(1), "nodes": ph.StepInputs.zeros(2)})
    return runtime, state, inputs


def operation(case, expected=None):
    runtime, state, inputs = case
    result = runtime.step(state, inputs, 0.1)
    if not result.report.accepted:
        raise RuntimeError(result.report.reason)
    sub = result.report.substeps[0]
    d = sub.diagnostics["initial_robot_material_contact"]
    actual = (
        result.state.domains["robot"].q.copy(),
        result.state.domains["robot"].v.copy(),
        result.state.domains["nodes"].position.copy(),
        result.state.domains["nodes"].velocity.copy(),
        d["world_impulses"].copy(),
    )
    np.testing.assert_allclose(
        actual[4], [[0, 4 / 7, 0]], atol=5 * runtime.settings.tolerance, rtol=0
    )
    np.testing.assert_allclose(actual[1], [-2 / 7], atol=5 * runtime.settings.tolerance, rtol=0)
    if expected is not None:
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, b)
    execution = sub.diagnostics["execution"]
    if runtime.context.backend != "numpy" and execution["completed_launches"] <= 0:
        raise RuntimeError("missing actual native execution")
    return actual, dict(
        outer_iterations=sub.position_iterations,
        replays=d["replays"],
        tolerance=runtime.settings.tolerance,
        max_residual=result.report.max_residual,
        energy_gain_upper_j=d["physical_budgets"]["frozen_model_energy"]["positive_gain_upper_j"],
        angular_defect_nms=d["physical_budgets"]["material_angular_defect"]["l1_sum_nms"],
        outer_stage_seconds=dict(sub.timings),
        **execution,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["numpy", "interpret", "cuda"], required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1 or args.output.exists():
        parser.error("positive counts and a new output are required")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record = dict(
        mode=args.mode,
        source=provenance(),
        script_sha256=script_hash,
        status="running",
        scope="identical compressed two-node edge and mass2 Y robot; full outer, all native replays, exact material/physical audits, owned diagnostics, analytic and bitwise repetition checks, workspace close; factory construction/final close excluded",
        batches=[],
        calls=[],
        accepted_measured_calls=0,
        successful_calls_including_warmups=0,
        not_measured=[
            "continuous trajectory",
            "Warp",
            "C++ endurance",
            "cold cache",
            "peak memory",
        ],
    )
    case = None
    phase = "construction"
    try:
        case = make_case(args.mode)
        expected = None
        for warm in range(3):
            phase = f"warm{warm}"
            start = perf_counter()
            actual, stats = operation(case, expected)
            elapsed = perf_counter() - start
            if expected is None:
                expected = actual
            record["calls"].append(stats)
            record["successful_calls_including_warmups"] += 1
            if warm == 0:
                record["first_call_seconds"] = elapsed
        for repeat in range(args.repeats):
            phase = f"batch{repeat}"
            samples = []
            batch = dict(seconds=samples)
            record["batches"].append(batch)
            for _ in range(args.samples):
                start = perf_counter()
                _, stats = operation(case, expected)
                samples.append(perf_counter() - start)
                record["calls"].append(stats)
                record["accepted_measured_calls"] += 1
                record["successful_calls_including_warmups"] += 1
            batch.update(median=float(np.median(samples)), p95=float(np.percentile(samples, 95)))
        record["status"] = "complete"
    except Exception as error:
        record.update(
            status="failed", failure_phase=phase, error_type=type(error).__name__, failed_calls=1
        )
        traceback.print_exc()
    finally:
        if case is not None:
            case[0].close()
        if (
            provenance()["source_sha256"] != record["source"]["source_sha256"]
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != script_hash
        ):
            record["status"] = "source_changed"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(record["status"], args.output)
    return 0 if record["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
