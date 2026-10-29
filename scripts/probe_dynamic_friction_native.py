"""Prescribed split-friction vectors through existing native APIs, not an automatic law.

The constants are independent rational scene solutions documented in
p65-dynamic-friction-candidate.md. No contact program produces these impulses.
"""

import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

import numpy as np

import phymeo as ph
from phymeo.core.scene import fingerprint
from phymeo.experiments.rigid_cloth import provenance
from phymeo.solvers._robot_replay_ledger import FrozenRobotLedger


def array(values):
    return np.asarray(values, dtype=np.float64)


def make_case(name, mode):
    chain = name == "material_impact"
    cross = name.startswith("cross_")
    n, joints, h = 2 if chain else 1, 2 if cross else 1, 1.0 if cross else 0.25
    normals = array([[0, 1, 0], [-F(3, 5), F(4, 5), 0]])[:joints]
    anchors = np.zeros((joints, 3))
    if cross:
        anchors[1] = -float(F(28, 5)) * normals[1]
    robot = ph.RobotModel(
        np.full(joints, -1, dtype=int),
        ("prismatic",) * joints,
        anchors,
        np.tile([0.0, 0.0, 0.0, 1.0], (joints, 1)),
        normals,
        np.ones(joints) if cross else array([2]),
        np.zeros((joints, 3)),
        np.ones((joints, 3)),
        np.zeros(3),
    )
    support = not chain and not cross
    nodes = ph.ParticleModel(np.ones(n), array([0, -1 if support else 0, 0]))
    material = (
        ph.DistanceConstraints(np.array([[0, 1]]), array([1]), array([F(1, 16)])) if chain else None
    )
    position = array([[0, F(1, 8), 0], [0, F(9, 8), 0]]) if chain else np.zeros((1, 3))
    velocity = (
        array([[1, -1, 0], [1, -1, 0]])
        if chain
        else array([[F(21, 10), -F(9, 5), 0]])
        if cross
        else np.zeros((1, 3))
    )
    initial = ph.SceneState(
        {
            "robot": ph.RobotState(np.zeros(joints), array([1, 6]) if cross else np.zeros(1)),
            "nodes": ph.ParticleState(position, velocity),
        }
    )
    robot_input = ph.RobotInputs.zeros(joints)
    if support:
        robot_input.effort[0] = 1
    inputs = ph.SceneInputs(
        {
            "robot": robot_input,
            "nodes": ph.StepInputs(array([[1, 0, 0]]) if support else np.zeros((n, 3))),
        }
    )
    scene = ph.SceneModel(
        (ph.DomainBinding("robot", "robot", robot), ph.DomainBinding("nodes", "particles", nodes))
    )
    if mode == "numpy":
        participants = {
            "robot": ph.RobotParticipant(robot),
            "nodes": ph.ClothParticipant(nodes, material, record_material_path=True),
        }
        factory = ph.ArticulatedCouplingProgram
    else:
        participants = {
            "robot": ph.TyxRobotParticipant(robot, mode),
            "nodes": ph.TyxClothParticipant(nodes, material, mode, record_material_path=True),
        }
        factory = ph.TyxArticulatedCouplingProgram
    base = factory(
        scene,
        participants,
        ph.RobotSphereAttachments(np.empty(0, int), np.empty((0, 3)), np.empty(0)),
        (),
    )
    settings = ph.StepSettings(
        substeps=1,
        position_iterations=128,
        velocity_iterations=128,
        tolerance=1e-8 if mode == "numpy" else 2e-5,
    )
    context = ph.BackendContext(base.capabilities.backend)
    held_rows, terminal_rows = np.zeros((joints, 3)), np.zeros((joints, 3))
    if chain:
        held_rows[0, 1] = float(F(3, 7))
        terminal_rows[0] = array([-F(8, 21), F(1, 3), 0])
        expected = dict(
            x=[[F(1, 4), -F(3, 56), 0], [F(1, 4), F(51, 56), 0]],
            q=[-F(3, 56)],
            pre_v=[[1, -F(5, 7), 0], [1, -F(6, 7), 0]],
            pre_rv=[-F(3, 14)],
            v=[[F(13, 21), -F(8, 21), 0], [1, -F(6, 7), 0]],
            rv=[-F(8, 21)],
            energy=F(1135, 882),
        )
    elif cross:
        t = -F(1, 2) if name == "cross_phasewise" else F(201, 550)
        p1, p2 = (13 - 2 * t) / 14, (33 + 10 * t) / 28
        held_rows = array([float(p1) * normals[0] + array([t, 0, 0]), float(p2) * normals[1]])
        terminal_rows[1] = float(F(14, 5)) * normals[1]
        pre_x = F(21, 10) + t - F(3, 5) * p2
        pre_y = -F(9, 5) + p1 + F(4, 5) * p2
        expected = dict(
            x=[[pre_x, pre_y, 0]],
            q=[1 - p1, 6 - p2],
            pre_v=[[pre_x, pre_y, 0]],
            pre_rv=[1 - p1, 6 - p2],
            v=[[pre_x - F(42, 25), pre_y + F(56, 25), 0]],
            rv=[1 - p1, 6 - p2 - F(14, 5)],
        )
    else:
        t = -F(1, 4) if name == "support_stick" else -F(1, 16) if name == "support_slide" else F(0)
        held_rows[0] = array([t, F(1, 4), 0])
        if name == "support_terminal_only":
            terminal_rows[0, 0] = -0.25
        pre_vx = F(1, 4) + t
        final_vx = pre_vx - (F(1, 4) if name == "support_terminal_only" else 0)
        expected = dict(
            x=[[pre_vx / 4, 0, 0]],
            q=[0],
            pre_v=[[pre_vx, 0, 0]],
            pre_rv=[0],
            v=[[final_vx, 0, 0]],
            rv=[0],
        )
    return (
        scene,
        initial,
        inputs,
        base,
        settings,
        context,
        material,
        h,
        normals,
        held_rows,
        terminal_rows,
        expected,
    )


def run_case(name, mode):
    (
        scene,
        initial,
        inputs,
        base,
        settings,
        context,
        material,
        h,
        normals,
        held_rows,
        terminal_rows,
        expected,
    ) = make_case(name, mode)
    initial_signature = fingerprint((initial, inputs))
    held = np.zeros_like(initial.domains["nodes"].position)
    held[0] = held_rows.sum(axis=0)
    generalized = -np.sum(normals * held_rows, axis=1)
    loads = {"nodes": {"force": held}, "robot": {"effort": generalized}}

    def replay(requests):
        wrapper = ph.TerminalImpulseProgram(base, requests, registry=base.registry)
        with ph.SceneLoadReplay(
            scene,
            wrapper,
            initial,
            inputs,
            h,
            channels={"nodes": ("force",), "robot": ("effort",)},
            settings=settings,
            context=context,
            registry=base.registry,
        ) as runner:
            trial = runner.evaluate(loads)
        if not trial.result.report.accepted:
            raise RuntimeError(trial.result.report.reason)
        return trial

    try:
        held_trial = replay(())
        selected_pre = held_trial.result.state
        robot_model, node_model = scene.domains[0].model, scene.domains[1].model
        origin = robot_model.joint_position + normals * selected_pre.domains["robot"].q[:, None]
        relative = selected_pre.domains["nodes"].position[0] - origin
        local = relative - np.sum(relative * normals, axis=1)[:, None] * normals
        requests = (
            ph.PointVelocityImpulses("nodes", np.zeros(len(normals), dtype=int), terminal_rows),
            ph.PointVelocityImpulses("robot", np.arange(len(normals)), -terminal_rows, local),
        )
        trial = replay(requests)
        diagnostics = trial.result.report.substeps[0].diagnostics
        ev = diagnostics["terminal_impulse"]
        pre, post = ev["pre_impulse_state"], trial.result.state
        assert fingerprint(pre) == fingerprint(selected_pre)
        assert fingerprint((initial, inputs)) == initial_signature
        assert ev["applications"] == 1
        atol = 2e-12 if mode == "numpy" else 2e-6
        pairs = (
            (pre.domains["nodes"].position, expected["x"]),
            (pre.domains["robot"].q, expected["q"]),
            (pre.domains["nodes"].velocity, expected["pre_v"]),
            (pre.domains["robot"].v, expected["pre_rv"]),
            (post.domains["nodes"].velocity, expected["v"]),
            (post.domains["robot"].v, expected["rv"]),
        )
        for actual, reference in pairs:
            np.testing.assert_allclose(actual, array(reference), rtol=0, atol=atol)
        for domain, field in (("robot", "q"), ("nodes", "position")):
            np.testing.assert_array_equal(
                getattr(pre.domains[domain], field), getattr(post.domains[domain], field)
            )
        material_report = ph.MaterialPathLedger(
            node_model,
            material,
            initial.domains["nodes"],
            inputs.domains["nodes"],
            h,
            settings,
            context,
        ).require(
            pre.domains["nodes"],
            trial.effective_inputs.domains["nodes"],
            held,
            ev["pre_impulse_diagnostics"]["participants"]["nodes"],
            sweeps=ev["pre_position_iterations"],
        )
        robot_report = FrozenRobotLedger(
            robot_model, initial.domains["robot"], inputs.domains["robot"], h, settings, 1.0
        ).trial_audit(
            pre.domains["robot"],
            trial.effective_inputs.domains["robot"],
            [F(float(v)) for v in generalized],
        )
        assert max(robot_report.values()) <= F(settings.tolerance)
        execution = {}
        for domain, participant in diagnostics["participants"].items():
            if mode != "numpy":
                stats = dict(participant["execution"])
                assert stats["mode"] == mode and stats["compute_dtype"] == "float32"
                assert stats["completed_launches"] > 0
                for value in (
                    (post.domains[domain].q, post.domains[domain].v)
                    if domain == "robot"
                    else (post.domains[domain].position, post.domains[domain].velocity)
                ):
                    np.testing.assert_array_equal(
                        value, value.astype(np.float32).astype(np.float64)
                    )
                execution[domain] = stats
        energy_pre = float(
            np.sum(pre.domains["nodes"].velocity ** 2) / 2
            + np.sum(robot_model.mass * pre.domains["robot"].v ** 2) / 2
        )
        energy_post = float(
            np.sum(post.domains["nodes"].velocity ** 2) / 2
            + np.sum(robot_model.mass * post.domains["robot"].v ** 2) / 2
        )
        checks = dict(terminal_energy_change_j=energy_post - energy_pre)
        if name == "material_impact":
            x = post.domains["nodes"].position
            energy_post += float((np.linalg.norm(x[1] - x[0]) - 1) ** 2 / (2 / 16))
            py = float(post.domains["nodes"].velocity[:, 1].sum() + 2 * post.domains["robot"].v[0])
            np.testing.assert_allclose(
                [energy_post, py], [float(expected["energy"]), -2], rtol=0, atol=atol
            )
            checks.update(energy_with_material_j=energy_post, y_momentum_ns=py)
        if name.startswith("cross_"):
            held_power = held_rows[0, 0] * pre.domains["nodes"].velocity[0, 0]
            final_power = held_rows[0, 0] * post.domains["nodes"].velocity[0, 0]
            np.testing.assert_allclose(energy_post - energy_pre, -196 / 25, rtol=0, atol=10 * atol)
            if name == "cross_phasewise":
                assert held_power < 0 and final_power > 0
                np.testing.assert_allclose(final_power, 17 / 50, rtol=0, atol=atol)
            else:
                assert held_power > 0
                np.testing.assert_allclose(held_power, 4221 / 6875, rtol=0, atol=atol)
            checks.update(
                held_tangent_dot_pre_slip=held_power, total_tangent_dot_final_slip=final_power
            )
        return dict(
            case=name,
            status="prescribed_native_verified",
            h=h,
            held_world_ns=held_rows.tolist(),
            terminal_world_ns=terminal_rows.tolist(),
            position=post.domains["nodes"].position.tolist(),
            velocity=post.domains["nodes"].velocity.tolist(),
            robot_q=post.domains["robot"].q.tolist(),
            robot_v=post.domains["robot"].v.tolist(),
            pre_position_sweeps=ev["pre_position_iterations"],
            robot_pre_residuals={k: float(v) for k, v in robot_report.items()},
            material_scope=material_report["scope"],
            terminal_residual=ev["ledger"]["max_residual"],
            configuration_preserved=ev["ledger"]["configuration_preserved"],
            replays=2,
            final_replay_execution=execution,
            checks=checks,
        )
    finally:
        base.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["numpy", "interpret", "cuda"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new evidence output path required")
    source = provenance()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = dict(
        status="running",
        mode=args.mode,
        scope="six prescribed native scene checks, not automatic contact or friction acceptance and not a performance measurement",
        source=source,
        script_sha256=script_hash,
        cases=[],
    )
    try:
        for name in (
            "support_stick",
            "support_slide",
            "support_terminal_only",
            "material_impact",
            "cross_phasewise",
            "cross_endpoint",
        ):
            result["cases"].append(run_case(name, args.mode))
        result["source_after"] = provenance()
        assert source["source_sha256"] == result["source_after"]["source_sha256"]
        assert script_hash == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["status"] = "complete"
    except Exception as error:
        result["status"], result["error_type"] = "failed", type(error).__name__
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(f"{args.mode}: six prescribed native cases verified; no automatic friction solver")


if __name__ == "__main__":
    main()
