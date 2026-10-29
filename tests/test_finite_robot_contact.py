"""Automatic finite patch scene construction and exact platform controls."""

import math
from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest
from test_finite_patch_geometry import SQUARE, TRIANGLE, model, patch

import phymeo as ph


def build(
    mode="numpy",
    *,
    chain=False,
    rotor=False,
    X=1.5,
    gain=False,
    robot=None,
    vertices=None,
    base_transform=None,
    inner_settings=None,
    budgets=None,
    position=None,
    velocity=None,
    colliders=None,
    q=None,
    omega=None,
):
    assert hasattr(ph, "DynamicRobotFiniteContactProgram"), "missing integrated finite contact API"
    robot = model(rotor) if robot is None else robot
    theta = math.atan(0.1)
    alpha = 0.25
    V = X / 10 + 0.125 * (math.sqrt(101) / 10 - 1)
    p = (V - theta) / 2 if gain else 1 / 16

    def rotate(x):
        return np.array(
            [
                [math.cos(alpha), -math.sin(alpha), 0.0],
                [math.sin(alpha), math.cos(alpha), 0.0],
                [0.0, 0.0, 1.0],
            ]
        ) @ np.asarray(x)

    if position is None:
        position = (
            [rotate([1.0, 0.125, 0.0])]
            if rotor
            else ([[0.0, 0.25, 0.0], [0.0, 1.25, 0.0]] if chain else [[-0.75, 0.25, 0.25]])
        )
    position = np.array(position, dtype=float).reshape(-1, 3)
    if velocity is None:
        velocity = (
            [rotate([X - 1, V - p, 0.0])]
            if rotor
            else np.tile([1.0, -1.0, 0.0], (len(position), 1))
        )
    velocity = np.array(velocity, dtype=float).reshape(-1, 3)
    nodes = ph.ParticleModel(np.ones(len(position)), np.zeros(3))
    material = (
        ph.DistanceConstraints(np.array([[0, 1]]), np.array([1.0]), np.array([1 / 16]))
        if chain
        else None
    )
    scene = ph.SceneModel(
        (ph.DomainBinding("robot", "robot", robot), ph.DomainBinding("nodes", "particles", nodes))
    )
    state = ph.SceneState(
        {
            "robot": ph.RobotState(
                np.array([alpha] if rotor else np.zeros(robot.dof_count))
                if q is None
                else np.asarray(q, float),
                np.array([theta + p] if rotor else np.zeros(robot.dof_count))
                if omega is None
                else np.asarray(omega, float),
            ),
            "nodes": ph.ParticleState(position, velocity),
        }
    )
    inputs = ph.SceneInputs(
        {
            "robot": ph.RobotInputs.zeros(robot.dof_count),
            "nodes": ph.StepInputs.zeros(len(position)),
        }
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
    if base_transform:
        base = base_transform(base)
    c = patch((TRIANGLE if rotor else SQUARE) if vertices is None else vertices)
    program = ph.DynamicRobotFiniteContactProgram(
        scene,
        base,
        (c,) if colliders is None else colliders,
        materials={"nodes": material},
        budgets=budgets
        or ph.MaterialContactBudgets(
            1e-3 if chain else (1e-8 if mode == "numpy" else 1e-5),
            1e-3 if chain else (1e-8 if mode == "numpy" else 1e-5),
        ),
        inner_settings=inner_settings,
    )
    history = {f.name: np.zeros(f.shape, f.dtype) for f in program.history_fields}
    state = ph.SceneState(state.domains, history)
    settings = ph.StepSettings(
        substeps=1,
        position_iterations=128,
        velocity_iterations=128,
        tolerance=1e-8 if mode == "numpy" else 2e-5,
        record_substeps=True,
    )
    return scene, state, inputs, program, settings, ph.BackendContext(program.capabilities.backend)


def step(case, dt=0.25, accepted=True):
    scene, state, inputs, program, settings, context = case
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        with np.errstate(all="raise"):
            result = runtime.step(state, inputs, dt)
    assert result.report.accepted == accepted, result.report.reason
    return result


def test_platform_automatic_impact():
    case = build()
    r = step(case)
    d = r.report.substeps[0].diagnostics["dynamic_robot_finite_contact"]
    np.testing.assert_allclose(d["held_impulses"], [1 / 3], atol=1e-8, rtol=0)
    np.testing.assert_allclose(d["terminal_impulses"], [1 / 3], atol=1e-8, rtol=0)
    np.testing.assert_allclose(r.state.domains["robot"].q, [-1 / 24], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        r.state.domains["nodes"].position, [[-0.5, 1 / 12, 0.25]], atol=1e-8, rtol=0
    )
    np.testing.assert_allclose(
        r.state.domains["nodes"].velocity, [[1.0, -1 / 3, 0.0]], atol=1e-8, rtol=0
    )
    assert d["ccd"][0].status == "certified" and tuple(d["transitions"]) == ("enter",)


def test_unsupported_models_rejected():
    valid = model(True)
    for robot in (
        replace(valid, com=np.array([[0.1, 0.0, 0.0]])),
        replace(valid, joint_axis=np.array([[0.6, 0.8, 0.0]])),
    ):
        with pytest.raises(ValueError, match="supports"):
            build(robot=robot)


def test_signature_and_history_ownership():
    case = build()
    p = case[3]
    assert p.capabilities.features == frozenset({"dynamic_robot_finite_split"})
    assert {f.name: f.shape for f in p.history_fields} == {
        "finite_contact/active": (1,),
        "finite_contact/feature": (1, 2),
        "finite_contact/phase_world": (1, 2, 3),
    }
    from phymeo.core.scene import fingerprint

    sig = fingerprint(p.problem_signature())
    own = p.initial_history()
    own["finite_contact/feature"][:] = 9
    assert fingerprint(p.problem_signature()) == sig
    p.close()


def platform_window(mode, h):
    case = build(mode)
    scene, state, inputs, program, settings, context = case
    dt = F(h)
    x, y, qrobot, vx, vy, vr = F(-3, 4), F(1, 4), F(0), F(1), F(-1), F(0)
    events = []
    features = []
    first_enter_x = None
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        total = int(F(2) / dt) + int(F(1, 4) / dt) + 1
        for n in range(total):
            reverse = n == int(F(2) / dt)
            force = np.array([[-2 / h, 0.0, 0.0]]) if reverse else np.zeros((1, 3))
            inp = ph.SceneInputs({**inputs.domains, "nodes": ph.StepInputs(force)})
            if reverse:
                vx -= 2
            # Independent Fraction implicit-Euler/terminal oracle. All active
            # witnesses here are interior or exactly on the square boundary.
            free_y = y + dt * vy
            free_q = qrobot + dt * vr
            nextx = x + dt * vx
            p = (
                max(F(0), -(free_y - free_q - F(1, 8)) / (dt * F(3, 2)))
                if abs(nextx) <= 1
                else F(0)
            )
            y1 = free_y + dt * p
            q1 = free_q - dt * p / 2
            vm = vy + p
            rm = vr - p / 2
            gap2 = (max(abs(nextx) - 1, F(0))) ** 2 + (y1 - q1) ** 2
            eta = max(F(0), -(vm - rm) / F(3, 2)) if gap2 == F(1, 64) else F(0)
            vy, vr = vm + eta, rm - eta / 2
            x, y, qrobot = nextx, y1, q1
            with np.errstate(all="raise"):
                result = runtime.step(state, inp, h)
            assert result.report.accepted, (n, result.report.reason)
            d = result.report.substeps[0].diagnostics["dynamic_robot_finite_contact"]
            tol = 1e-8 if mode == "numpy" else 2e-5
            np.testing.assert_allclose(
                result.state.domains["nodes"].position,
                [[float(x), float(y), 0.25]],
                atol=tol,
                rtol=0,
            )
            np.testing.assert_allclose(
                result.state.domains["robot"].q, [float(qrobot)], atol=tol, rtol=0
            )
            np.testing.assert_allclose(d["held_impulses"], [float(p)], atol=tol, rtol=0)
            np.testing.assert_allclose(d["terminal_impulses"], [float(eta)], atol=tol, rtol=0)
            assert len(d["row_keys"]) == 1
            events.append(d["transitions"][0])
            if first_enter_x is None and events[-1] == "enter":
                first_enter_x = float(x)
            features.append(tuple(d["final_feature"][0]))
            if events[-1] == "exit":
                assert not np.any(result.state.history["finite_contact/phase_world"])
                assert not np.any(result.state.history["finite_contact/feature"])
            state = result.state
    assert "persist" in events and "exit" in events
    assert events[int(F(7, 4) / dt)] == "exit"
    assert events[int(F(2) / dt) + int(F(1, 4) / dt) - 1] == "enter"
    assert features[int(F(7, 4) / dt) - 1][0] == 2
    assert events[-1] == "persist" and features[-1][0] == 1
    return state, abs(first_enter_x + 5 / 8)


def test_platform_seam_exit_reenter():
    errors = []
    for h in (0.25, 0.125, 0.0625):
        first, error = platform_window("numpy", h)
        second, _ = platform_window("numpy", h)
        np.testing.assert_array_equal(
            first.domains["nodes"].position, second.domains["nodes"].position
        )
        np.testing.assert_array_equal(first.domains["robot"].q, second.domains["robot"].q)
        errors.append(error)
    assert errors == sorted(errors, reverse=True)


def test_shared_tree_rows_and_transform_order():
    # Two particles share the same moving body; the implicit normal solve is
    # coupled through its mass, so independently solving then summing fails.
    positions = [[-0.5, 0.25, 0.25], [0.5, 0.25, 0.25]]
    first = step(build(position=positions, velocity=[[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]]))
    second = step(
        build(
            position=positions[::-1],
            velocity=[[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]],
            vertices=SQUARE[::-1],
        )
    )
    np.testing.assert_allclose(
        first.state.domains["nodes"].position,
        second.state.domains["nodes"].position[::-1],
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_allclose(first.state.domains["robot"].q, [-1 / 16], atol=1e-8, rtol=0)
    d = first.report.substeps[0].diagnostics["dynamic_robot_finite_contact"]
    np.testing.assert_allclose(d["held_impulses"], [0.25, 0.25], atol=1e-8, rtol=0)
    assert len(d["row_keys"]) == 2


def test_multiple_revolute_and_mixed_trees_rejected():
    for kinds in (("revolute", "revolute"), ("prismatic", "revolute")):
        robot = ph.RobotModel(
            np.array([-1, 0]),
            kinds,
            np.zeros((2, 3)),
            np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
            np.tile([0.0, 0.0, 1.0], (2, 1)),
            np.ones(2),
            np.zeros((2, 3)),
            np.ones((2, 3)),
            np.zeros(3),
        )
        with pytest.raises(ValueError, match="supports"):
            build(robot=robot)


def test_shared_ancestor_branches_and_fixed_world_frame():
    robot = ph.RobotModel(
        np.array([-1, 0, 0]),
        ("prismatic",) * 3,
        np.zeros((3, 3)),
        np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)),
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([2.0, 1.0, 1.0]),
        np.zeros((3, 3)),
        np.ones((3, 3)),
        np.zeros(3),
    )
    vertices_a = SQUARE + np.array([-3.0, 0.0, 0.0])
    vertices_b = SQUARE + np.array([3.0, 0.0, 0.0])
    ca = ph.RobotFinitePatchCollider("a", 1, vertices_a, radius_m=0.125)
    cb = ph.RobotFinitePatchCollider("b", 2, vertices_b, radius_m=0.125)
    positions = np.array([[-3.0, 0.25, 0.25], [3.0, 0.25, 0.25]])
    velocities = np.array([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]])
    first = step(build(robot=robot, position=positions, velocity=velocities, colliders=(ca, cb)))
    d = first.report.substeps[0].diagnostics["dynamic_robot_finite_contact"]
    np.testing.assert_allclose(d["held_impulses"], [1 / 3, 0.0, 0.0, 1 / 3], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        first.state.domains["robot"].q, [-1 / 24, 0.0, 0.0], atol=1e-8, rtol=0
    )
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    shift = np.array([3.0, 4.0, 5.0])
    orientation = robot.joint_orientation.copy()
    orientation[0] = [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]
    offsets = robot.joint_position.copy()
    offsets[0] = shift
    transformed = replace(robot, joint_orientation=orientation, joint_position=offsets)
    reversed_a = ph.RobotFinitePatchCollider("a", 1, vertices_a[::-1], radius_m=0.125)
    reversed_b = ph.RobotFinitePatchCollider("b", 2, vertices_b[::-1], radius_m=0.125)
    second = step(
        build(
            robot=transformed,
            position=(positions[::-1] @ R.T) + shift,
            velocity=velocities[::-1] @ R.T,
            colliders=(reversed_b, reversed_a),
        )
    )
    np.testing.assert_allclose(
        second.state.domains["robot"].q, first.state.domains["robot"].q, atol=1e-8, rtol=0
    )
    np.testing.assert_allclose(
        second.state.domains["nodes"].position,
        (first.state.domains["nodes"].position[::-1] @ R.T) + shift,
        atol=1e-8,
        rtol=0,
    )
