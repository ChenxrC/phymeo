"""Serial observations of automatic split friction; not prescribed impulses."""

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np

import phymeo as ph
from phymeo.experiments.rigid_cloth import provenance


def make_case(mode, chain):
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
    count = 2 if chain else 1
    nodes = ph.ParticleModel(np.ones(count), np.array([0.0, 0 if chain else -1, 0]))
    edges = (
        ph.DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.array([1 / 16]))
        if chain
        else None
    )
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
    settings = ph.StepSettings(
        substeps=1,
        position_iterations=128,
        velocity_iterations=128,
        tolerance=1e-8 if mode == "numpy" else 2e-5,
    )
    try:
        program = ph.DynamicRobotFrictionContactProgram(
            scene,
            base,
            (ph.RobotPlaneCollider("support", 0, np.zeros(3), np.array([0.0, 1, 0])),),
            friction_by_collider={"support": 0.5 if chain else 2.0},
            materials={"nodes": edges},
            budgets=ph.MaterialContactBudgets(1e-6, 1e-6),
            registry=base.registry,
            inner_settings=settings,
        )
        runtime = ph.SceneStepper(
            scene, program, settings, ph.BackendContext(base.capabilities.backend), base.registry
        )
    except Exception:
        base.close()
        raise
    state = ph.SceneState(
        {
            "robot": ph.RobotState(np.zeros(1), np.zeros(1)),
            "nodes": ph.ParticleState(
                np.array([[0.0, 1 / 8, 0], [0.0, 9 / 8, 0]]) if chain else np.zeros((1, 3)),
                np.tile([1.0, -1, 0], (2, 1)) if chain else np.zeros((1, 3)),
            ),
        },
        program.initial_history(),
    )
    robot_input = ph.RobotInputs.zeros(1)
    if not chain:
        robot_input.effort[0] = 1
    inputs = ph.SceneInputs(
        {
            "robot": robot_input,
            "nodes": ph.StepInputs(np.zeros((2, 3)) if chain else np.array([[1.0, 0, 0]])),
        }
    )
    return runtime, state, inputs


def operation(case, chain, expected=None):
    runtime, initial, inputs = case
    result = runtime.step(initial, inputs, 1 / 4)
    if not result.report.accepted:
        raise RuntimeError(result.report.reason)
    sub = result.report.substeps[0]
    diagnostic = sub.diagnostics["dynamic_robot_friction_contact"]
    robot, nodes = result.state.domains["robot"], result.state.domains["nodes"]
    atol = 5 * runtime.settings.tolerance
    if chain:
        position = [[1 / 4, -3 / 56, 0], [1 / 4, 51 / 56, 0]]
        velocity = [[13 / 21, -8 / 21, 0], [1, -6 / 7, 0]]
        q, rv = [-3 / 56], [-8 / 21]
        held, terminal = [[0, 3 / 7, 0], [0, 0, 0]], [[-8 / 21, 1 / 3, 0], [0, 0, 0]]
        expected_energy, expected_momentum = 1135 / 882, -2
    else:
        position = velocity = [[0, 0, 0]]
        q = rv = [0]
        held, terminal = [[-1 / 4, 1 / 4, 0]], [[0, 0, 0]]
        expected_energy = expected_momentum = 0
    for actual, reference in (
        (robot.q, q),
        (robot.v, rv),
        (nodes.position, position),
        (nodes.velocity, velocity),
        (diagnostic["held_world"], held),
        (diagnostic["terminal_world"], terminal),
    ):
        np.testing.assert_allclose(actual, reference, rtol=0, atol=atol)
    energy = float(robot.v @ robot.v + np.sum(nodes.velocity**2) / 2)
    if chain:
        energy += float((np.linalg.norm(nodes.position[1] - nodes.position[0]) - 1) ** 2 / (2 / 16))
    momentum = float(2 * robot.v[0] + np.sum(nodes.velocity[:, 1]))
    np.testing.assert_allclose(
        [energy, momentum], [expected_energy, expected_momentum], rtol=0, atol=atol
    )
    actual = (
        robot.q.copy(),
        robot.v.copy(),
        nodes.position.copy(),
        nodes.velocity.copy(),
        diagnostic["held_world"].copy(),
        diagnostic["terminal_world"].copy(),
        *(value.copy() for _, value in sorted(result.state.history.items())),
    )
    if expected is not None:
        if len(actual) != len(expected):
            raise RuntimeError("repetition history schema changed")
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, b)
    execution = sub.diagnostics["execution"]
    if runtime.context.backend != "numpy" and execution["completed_launches"] <= 0:
        raise RuntimeError("missing actual native execution")
    physical = diagnostic["physical_budgets"]
    return actual, dict(
        case="chain" if chain else "support",
        position_iterations=sub.position_iterations,
        velocity_iterations=sub.velocity_iterations,
        held_replays=diagnostic["held_replays"],
        terminal_replays=diagnostic["terminal_replays"],
        tolerance=runtime.settings.tolerance,
        max_residual=result.report.max_residual,
        energy_gain_upper_j=physical["frozen_model_energy"]["positive_gain_upper_j"],
        angular_defect_nms=physical["material_angular_defect"]["l1_sum_nms"],
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
        parser.error("positive counts and a new output path required")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record = dict(
        mode=args.mode,
        source=provenance(),
        script_sha256=script_hash,
        status="running",
        scope="fresh-state automatic split friction, alternating sticking support and oblique distance-chain impact; whole step includes native replays, both exact ledgers/physical audits, owned diagnostics, analytic and bitwise repetition assertions, workspace close; factory setup/final close excluded",
        fixed_settings=dict(
            h=0.25,
            outer_position_iterations=128,
            outer_velocity_iterations=128,
            inner_iterations=128,
            energy_budget_j=1e-6,
            angular_budget_nms=1e-6,
        ),
        calls=[],
        batches=[],
        accepted_measured_calls=0,
        successful_calls_including_warmups=0,
        case_measured_calls={"support": 0, "chain": 0},
        first_call_seconds={},
        not_measured=[
            "continuous trajectory",
            "FK CCD",
            "finite-feature friction",
            "Warp",
            "C++ endurance",
            "cold cache",
            "peak memory",
        ],
    )
    cases, expected = {}, {}
    phase = "construction"
    try:
        for chain in (False, True):
            cases[chain] = make_case(args.mode, chain)
            for warm in range(3):
                phase = f"{'chain' if chain else 'support'}_warm{warm}"
                start = perf_counter()
                actual, stats = operation(cases[chain], chain, expected.get(chain))
                elapsed = perf_counter() - start
                if chain not in expected:
                    expected[chain] = actual
                    record["first_call_seconds"][stats["case"]] = elapsed
                record["calls"].append(dict(stats, measured=False, seconds=elapsed))
                record["successful_calls_including_warmups"] += 1
        for repeat in range(args.repeats):
            batch = dict(seconds=[], cases=[])
            record["batches"].append(batch)
            for index in range(args.samples):
                chain = bool(index % 2)
                phase = f"batch{repeat}_sample{index}"
                start = perf_counter()
                _, stats = operation(cases[chain], chain, expected[chain])
                elapsed = perf_counter() - start
                batch["seconds"].append(elapsed)
                batch["cases"].append(stats["case"])
                record["calls"].append(dict(stats, measured=True, seconds=elapsed))
                record["accepted_measured_calls"] += 1
                record["case_measured_calls"][stats["case"]] += 1
                record["successful_calls_including_warmups"] += 1
            batch.update(
                median=float(np.median(batch["seconds"])),
                p95=float(np.percentile(batch["seconds"], 95)),
            )
        record["by_case"] = {}
        for name in ("support", "chain"):
            durations = [
                call["seconds"]
                for call in record["calls"]
                if call["measured"] and call["case"] == name
            ]
            record["by_case"][name] = dict(
                samples=len(durations),
                median=None if not durations else float(np.median(durations)),
                p95=None if not durations else float(np.percentile(durations, 95)),
            )
        record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", failure_phase=phase, error_type=type(error).__name__)
        traceback.print_exc()
    finally:
        close_errors = []
        for chain, case in cases.items():
            try:
                case[0].close()
            except Exception as error:
                close_errors.append(
                    dict(case="chain" if chain else "support", error_type=type(error).__name__)
                )
                traceback.print_exc()
        if close_errors:
            record.update(status="close_failed", close_errors=close_errors)
        record["source_after"] = provenance()
        if (
            record["source_after"]["source_sha256"] != record["source"]["source_sha256"]
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
