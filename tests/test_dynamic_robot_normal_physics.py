"""Independent exact examples, finite windows, budgets and coupled response."""

from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest
from test_dynamic_robot_normal_contact import build, custom, step

import phymeo as ph


def diagnostics(result):
    return result.report.substeps[-1].diagnostics["dynamic_robot_normal_contact"]


def lifecycle(mode="numpy"):
    case = build(mode)
    scene, state, inputs, program, settings, context = case
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        first = runtime.step(state, inputs, 0.25)
        assert first.report.accepted, first.report.reason
        second = runtime.step(first.state, inputs, 0.25)
        assert second.report.accepted, second.report.reason
        np.testing.assert_array_equal(diagnostics(second)["held_impulses"], 0.0)
        np.testing.assert_array_equal(diagnostics(second)["terminal_impulses"], 0.0)
        assert diagnostics(second)["events"] == ("persist",)
        pull = ph.SceneInputs(
            {**inputs.domains, "nodes": ph.StepInputs(np.array([[0.0, 1.0, 0.0]]))}
        )
        third = runtime.step(second.state, pull, 0.25)
        assert third.report.accepted, third.report.reason
        assert diagnostics(third)["events"] == ("exit",)
        np.testing.assert_array_equal(third.state.history["normal_contact/phase_impulses"], 0.0)
        fourth = runtime.step(third.state, inputs, 0.25)
        assert fourth.report.accepted, fourth.report.reason
        assert diagnostics(fourth)["events"] == ("inactive",)
        np.testing.assert_array_equal(fourth.state.history["normal_contact/phase_impulses"], 0.0)
    return first, second, third, fourth


def test_enter_persist_exit_and_still_open_approach():
    lifecycle()
    result = step(custom(position=[[0.0, 1.0, 0.0]]))
    np.testing.assert_array_equal(diagnostics(result)["terminal_impulses"], 0.0)
    assert diagnostics(result)["relative_normal_velocity"][0] == -1


def support(mode, h):
    material = ph.DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.array([1 / 256]))
    case = custom(
        mode=mode,
        position=[[0.0, 0.0, 0.0], [0.0, 255 / 256, 0.0]],
        velocity=np.zeros((2, 3)),
        gravity=(0.0, -1.0, 0.0),
        effort=2.0,
        material=material,
    )
    scene, state, inputs, program, settings, context = case
    start = state.domains["nodes"].position.copy()
    maxdrift = 0.0
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        for _ in range(round(2 / h)):
            result = runtime.step(state, inputs, h)
            assert result.report.accepted, result.report.reason
            state = result.state
            d = diagnostics(result)
            np.testing.assert_allclose(
                d["held_impulses"], [2 * h, 0.0], atol=2e-6 if mode != "numpy" else 1e-8, rtol=0
            )
            np.testing.assert_array_equal(d["terminal_impulses"], 0.0)
            maxdrift = max(
                maxdrift,
                float(np.max(np.abs(state.domains["nodes"].position - start))),
                float(np.max(np.abs(state.domains["robot"].q))),
            )
    assert maxdrift <= (1e-7 if mode == "numpy" else 1e-5)
    return state


@pytest.mark.parametrize("h", [1 / 16, 1 / 32])
def test_fixed_two_second_material_support(h):
    support("numpy", h)


def impact(mode, h):
    outputs = []
    for _ in range(2):
        scene, state, inputs, program, settings, context = build(mode)
        with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
            for _ in range(round(1 / h)):
                result = runtime.step(state, inputs, h)
                assert result.report.accepted, result.report.reason
                state = result.state
        target = -F(7, 24)
        limit = 1e-7 if mode == "numpy" else 1e-5
        np.testing.assert_allclose(
            state.domains["nodes"].position[:, 1], [float(target)], atol=limit, rtol=0
        )
        np.testing.assert_allclose(state.domains["robot"].q, [float(target)], atol=limit, rtol=0)
        np.testing.assert_allclose(
            state.domains["nodes"].velocity[:, 1], [-1 / 3], atol=limit, rtol=0
        )
        outputs.append(state)
    from phymeo.core.scene import fingerprint

    assert fingerprint(outputs[0]) == fingerprint(outputs[1])


@pytest.mark.parametrize("h", [0.25, 0.125, 0.0625])
def test_three_stepsizes_independent_inelastic_toi_endpoint(h):
    impact("numpy", h)


def test_tangential_drift_terminal_moment_uses_final_position_and_original_work(monkeypatch):
    case = custom(velocity=[[1.0, -1.0, 0.0]])
    scene, state, inputs, program, settings, context = case
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        result = runtime.step(state, inputs, 0.25)
    assert result.report.accepted, result.report.reason
    d = diagnostics(result)
    np.testing.assert_allclose(result.state.domains["nodes"].position[0, 0], 0.25, atol=1e-12)
    np.testing.assert_allclose(d["terminal_world_impulses"][0, 1] * 0.25, 1 / 12, atol=1e-8)
    assert d["physical_budgets"]["material_angular_defect"]["l1_sum_nms"] < 1e-8
    assert d["physical_budgets"]["frozen_model_energy"]["original_input_work_j"] == 0
    from phymeo.solvers._material_contact_balance import MaterialContactBalance

    balance = MaterialContactBalance(
        {x.name: x.model for x in scene.domains},
        program.materials,
        state,
        inputs,
        "robot",
        0.25,
        program.budgets,
    )
    from phymeo.solvers._robot_replay_ledger import FrozenRobotLedger

    ledger = FrozenRobotLedger(
        program.model, state.domains["robot"], inputs.domains["robot"], 0.25, settings, 1.0
    )
    held = {"nodes": [[F(0), F(float(d["held_impulses"][0])), F(0)]]}
    terminal = {"nodes": [[F(0), F(float(d["terminal_impulses"][0])), F(0)]]}
    good = balance.require_metrics(
        result.state, ledger, held, settings.tolerance, terminal=terminal
    )
    legacy = balance.require_metrics(result.state, ledger, held, settings.tolerance)
    assert good[0]["material_angular_defect"] <= settings.tolerance
    assert legacy[0]["material_angular_defect"] > settings.tolerance
    np.testing.assert_allclose(
        legacy[1]["material_angular_defect"]["l1_sum_nms"], 1 / 12, atol=1e-8
    )
    # A wrong-time ledger subtracts x0 cross eta=0 and must reject the same candidate.
    original = MaterialContactBalance.require_metrics

    def wrong_time(self, final, robot, requests, tolerance, *, terminal=None):
        return original(self, final, robot, requests, tolerance)

    monkeypatch.setattr(MaterialContactBalance, "require_metrics", wrong_time)
    case = custom(velocity=[[1.0, -1.0, 0.0]])
    with ph.SceneStepper(case[0], case[3], case[4], case[5], case[3].registry) as runtime:
        bad = runtime.step(case[1], case[2], 0.25)
    assert not bad.report.accepted


def test_budget_exhaustion_and_position_only_negative_controls(monkeypatch):
    from phymeo.solvers.dynamic_robot_normal_contact import _DynamicNormalWorkspace

    original = _DynamicNormalWorkspace.velocity_sweep
    monkeypatch.setattr(_DynamicNormalWorkspace, "velocity_sweep", lambda self: None)
    case = build()
    with ph.SceneStepper(case[0], case[3], case[4], case[5], case[3].registry) as runtime:
        result = runtime.step(case[1], case[2], 0.25)
    assert not result.report.accepted and "velocity" in result.report.reason
    monkeypatch.setattr(_DynamicNormalWorkspace, "velocity_sweep", original)
    # A hard over-stretched material creates energy even without contact.
    material = ph.DistanceConstraints(np.array([[0, 1]]), np.array([1.0]), np.array([0.0]))
    case = custom(
        position=[[0.0, 2.0, 0.0], [0.0, 2.5, 0.0]], velocity=np.zeros((2, 3)), material=material
    )
    with ph.SceneStepper(case[0], case[3], case[4], case[5], case[3].registry) as runtime:
        result = runtime.step(case[1], case[2], 0.25)
    assert not result.report.accepted


def test_shared_robot_triangle_and_row_order_independent_mass_oracle():
    # Three masses share one mass2 slider: solve the full four-mass equality
    # system independently. Equal gaps/speeds give p=eta=1/5 per node.
    from test_terminal_impulse import fixture

    robot = replace(fixture()[0].domains[0].model, joint_axis=np.array([[0.0, 0.0, 1.0]]))
    points = np.array([[0.0, 0.0, 1 / 8], [1.0, 0.0, 1 / 8], [0.0, 1.0, 1 / 8]])
    material = ph.DistanceConstraints(
        np.array([[0, 1], [1, 2], [2, 0]]), np.array([1.0, np.sqrt(2), 1.0]), np.full(3, 1 / 16)
    )
    collider = ph.RobotPlaneCollider("z", 0, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    result = step(
        custom(
            position=points,
            velocity=np.tile([0.0, 0.0, -1.0], (3, 1)),
            robot=robot,
            material=material,
            colliders=(collider,),
        )
    )
    d = diagnostics(result)
    np.testing.assert_allclose(d["held_impulses"], np.full(3, 1 / 5), atol=1e-8, rtol=0)
    np.testing.assert_allclose(d["terminal_impulses"], np.full(3, 1 / 5), atol=1e-8, rtol=0)
    momentum = 2 * result.state.domains["robot"].v[0] + sum(
        result.state.domains["nodes"].velocity[:, 2]
    )
    assert abs(momentum + 3) < 1e-8


def test_same_node_two_planes_shared_ancestor_and_representable_transform():
    robot = ph.RobotModel(
        np.array([-1, 0]),
        ("prismatic", "prismatic"),
        np.zeros((2, 3)),
        np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([2.0, 3.0]),
        np.zeros((2, 3)),
        np.ones((2, 3)),
        np.zeros(3),
    )
    x = ph.RobotPlaneCollider("x", 1, np.zeros(3), np.array([1.0, 0.0, 0.0]))
    y = ph.RobotPlaneCollider("y", 1, np.zeros(3), np.array([0.0, 1.0, 0.0]))
    outputs = []
    for colliders in ((x, y), (y, x)):
        result = step(
            custom(
                position=[[1 / 8, 1 / 8, 0.0]],
                velocity=[[-1.0, -1.0, 0.0]],
                robot=robot,
                colliders=colliders,
            )
        )
        d = diagnostics(result)
        # M=diag(5,3), independent scalar equations for each normal.
        np.testing.assert_allclose(d["held_impulses"], [5 / 12, 3 / 8], atol=1e-8, rtol=0)
        np.testing.assert_allclose(d["terminal_impulses"], [5 / 12, 3 / 8], atol=1e-8, rtol=0)
        np.testing.assert_allclose(
            result.state.domains["robot"].v, [-1 / 6, -1 / 4], atol=1e-8, rtol=0
        )
        outputs.append(result.state)
    from phymeo.core.scene import fingerprint

    assert fingerprint(outputs[0]) == fingerprint(outputs[1])
    # Exact axis permutation and translation preserve scalar contact responses.
    shifted = replace(robot, joint_position=np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 0.0]]))
    result = step(
        custom(
            position=[[1 / 8, 1 / 8, 4.0]],
            velocity=[[-1.0, -1.0, 0.0]],
            robot=shifted,
            colliders=(y, x),
        )
    )
    np.testing.assert_allclose(
        result.state.domains["robot"].v, outputs[0].domains["robot"].v, atol=1e-12, rtol=0
    )


def test_terminal_robot_application_point_tracks_tangential_drift():
    result = step(custom(velocity=[[1.0, -1.0, 0.0]]))
    evidence = diagnostics(result)["terminal_evidence"]
    requests = [r for r in evidence["requests"] if r.domain == "robot"]
    assert len(requests) == 1
    np.testing.assert_allclose(requests[0].local_points, [[0.25, 0.0, 0.0]], atol=1e-12, rtol=0)


def chain_window(mode, h):
    """Independent implicit-Euler reduced mass system and continuous oscillator."""
    import math

    hf = F(h)
    omega2 = F(64, 3)
    z = F(0)
    u = -F(2, 3)
    for _ in range(round((0.5 - 0.125) / h)):
        z, u = (z + hf * u) / (1 + hf * hf * omega2), (u - hf * omega2 * z) / (1 + hf * hf * omega2)
    duration = F(3, 8)
    a = -duration / 2 - z / 4
    b = 1 - duration / 2 + 3 * z / 4
    va = -F(1, 2) - u / 4
    vb = -F(1, 2) + 3 * u / 4
    omega = math.sqrt(float(omega2))
    zc = -2 / (3 * omega) * math.sin(omega * float(duration))
    uc = -2 / 3 * math.cos(omega * float(duration))
    continuous_x = np.array([-float(duration) / 2 - zc / 4, 1 - float(duration) / 2 + 3 * zc / 4])
    continuous_v = np.array([-0.5 - uc / 4, -0.5 + 3 * uc / 4])
    scene, state, inputs, program, settings, context = build(mode, chain=True)
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        for _ in range(round(0.5 / h)):
            result = runtime.step(state, inputs, h)
            assert result.report.accepted, result.report.reason
            state = result.state
    target_x = np.array([float(a), float(b)])
    target_v = np.array([float(va), float(vb)])
    tolerance = 1e-8 if mode == "numpy" else 2e-5
    np.testing.assert_allclose(
        state.domains["nodes"].position[:, 1], target_x, atol=tolerance, rtol=0
    )
    np.testing.assert_allclose(
        state.domains["nodes"].velocity[:, 1], target_v, atol=tolerance, rtol=0
    )
    np.testing.assert_allclose(state.domains["robot"].q, [float(a)], atol=tolerance, rtol=0)
    np.testing.assert_allclose(state.domains["robot"].v, [float(va)], atol=tolerance, rtol=0)
    assert (
        abs(2 * state.domains["robot"].v[0] + sum(state.domains["nodes"].velocity[:, 1]) + 2)
        < tolerance
    )
    return float(np.max(np.abs(target_x - continuous_x))), float(
        np.max(np.abs(target_v - continuous_v))
    )


def test_material_three_stepsizes_discrete_oracle_and_nonzero_continuous_error():
    errors = [chain_window("numpy", h) for h in (1 / 8, 1 / 16, 1 / 32)]
    assert errors[0][0] > errors[1][0] > errors[2][0] > 0
    assert errors[0][1] > errors[1][1] > errors[2][1] > 0
    for actual, limits in zip(
        errors, ((0.037, 0.081), (0.023, 0.036), (0.013, 0.015)), strict=True
    ):
        assert all(x <= limit for x, limit in zip(actual, limits, strict=True))


def test_original_inputs_stay_external_and_default_balance_is_unchanged():
    from phymeo.core.scene import fingerprint
    from phymeo.solvers._material_contact_balance import MaterialContactBalance
    from phymeo.solvers._robot_replay_ledger import FrozenRobotLedger

    scene, state, inputs, program, settings, context = build()
    force = ph.SceneInputs({**inputs.domains, "nodes": ph.StepInputs(np.array([[0.0, 0.25, 0.0]]))})
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        result = runtime.step(state, force, 0.25)
    assert result.report.accepted, result.report.reason
    d = diagnostics(result)
    expected = 0.25 * (
        result.state.domains["nodes"].position[0, 1] - state.domains["nodes"].position[0, 1]
    )
    assert (
        abs(d["physical_budgets"]["frozen_model_energy"]["original_input_work_j"] - expected)
        < 1e-15
    )
    # Tiny positive exact energy excess hidden by displayed FP64 rounding.
    epsilon = 2.0**-52
    initial = ph.SceneState(
        {
            "robot": state.domains["robot"],
            "nodes": ph.ParticleState(np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0]])),
        }
    )
    final = ph.SceneState(
        {
            "robot": state.domains["robot"],
            "nodes": ph.ParticleState(
                np.zeros((1, 3)), np.array([[np.nextafter(1.0, np.inf), 0.0, 0.0]])
            ),
        }
    )
    balance = MaterialContactBalance(
        {x.name: x.model for x in scene.domains},
        {"nodes": None},
        initial,
        inputs,
        "robot",
        0.25,
        ph.MaterialContactBudgets(epsilon, 1e-6),
    )
    ledger = FrozenRobotLedger(
        program.model, initial.domains["robot"], inputs.domains["robot"], 0.25, settings, 1.0
    )
    before = balance.require_metrics(
        final, ledger, {"nodes": [[0.0, 0.0, 0.0]]}, settings.tolerance
    )
    explicit = balance.require_metrics(
        final, ledger, {"nodes": [[0.0, 0.0, 0.0]]}, settings.tolerance, terminal=None
    )
    assert fingerprint(before[:2]) == fingerprint(explicit[:2]) and before[2] == explicit[2]
    assert before[0]["frozen_model_energy"] > settings.tolerance
    exact = F(epsilon) + F(epsilon) ** 2 / 2
    assert float(exact) == epsilon and exact > F(epsilon)


def test_multiple_selected_domains_shared_robot_and_domain_order():
    from test_terminal_impulse import fixture

    outputs = []
    for reverse in (False, True):
        original = fixture()
        robot = original[0].domains[0].model
        original[3].close()
        node = ph.ParticleModel(np.ones(1), np.zeros(3))
        bindings = (
            ph.DomainBinding("robot", "robot", robot),
            ph.DomainBinding("z_nodes", "particles", node),
            ph.DomainBinding("a_nodes", "particles", node),
        )
        if reverse:
            bindings = tuple(reversed(bindings))
        scene = ph.SceneModel(bindings)
        participants = {
            d.name: ph.RobotParticipant(robot)
            if d.family == "robot"
            else ph.ClothParticipant(node, None, record_material_path=True)
            for d in bindings
        }
        base = ph.ArticulatedCouplingProgram(
            scene,
            participants,
            ph.RobotSphereAttachments(np.empty(0, int), np.empty((0, 3)), np.empty(0)),
            (),
        )
        collider = ph.RobotPlaneCollider("floor", 0, np.zeros(3), np.array([0.0, 1.0, 0.0]))
        materials = dict(
            (name, None) for name in (("z_nodes", "a_nodes") if reverse else ("a_nodes", "z_nodes"))
        )
        program = ph.DynamicRobotNormalContactProgram(
            scene,
            base,
            (collider,),
            materials=materials,
            budgets=ph.MaterialContactBudgets(1e-6, 1e-6),
        )
        state = ph.SceneState(
            {
                d.name: ph.RobotState(np.zeros(1), np.zeros(1))
                if d.family == "robot"
                else ph.ParticleState(np.array([[0.0, 1 / 8, 0.0]]), np.array([[0.0, -1.0, 0.0]]))
                for d in bindings
            },
            program.initial_history(),
        )
        inputs = ph.SceneInputs(
            {
                d.name: ph.RobotInputs.zeros(1) if d.family == "robot" else ph.StepInputs.zeros(1)
                for d in bindings
            }
        )
        settings = ph.StepSettings(
            substeps=1, position_iterations=128, velocity_iterations=128, tolerance=1e-8
        )
        result = step((scene, state, inputs, program, settings, ph.BackendContext()))
        d = diagnostics(result)
        assert d["row_keys"] == (("a_nodes", 0, "floor"), ("z_nodes", 0, "floor"))
        np.testing.assert_allclose(d["held_impulses"], [1 / 4, 1 / 4], atol=1e-8, rtol=0)
        np.testing.assert_allclose(d["terminal_impulses"], [1 / 4, 1 / 4], atol=1e-8, rtol=0)
        np.testing.assert_allclose(result.state.domains["robot"].v, [-0.5], atol=1e-8, rtol=0)
        assert len(d["terminal_evidence"]["batches"]) == 3
        outputs.append(result.state)
    from phymeo.core.scene import fingerprint

    assert fingerprint(outputs[0]) == fingerprint(outputs[1])


def test_representable_axis_permutation_preserves_analytic_solution():
    from test_terminal_impulse import fixture

    robot = fixture()[0].domains[0].model
    rotation = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    translation = np.array([4.0, -2.0, 1.0])
    rotated = replace(
        robot,
        joint_axis=robot.joint_axis @ rotation.T,
        joint_position=robot.joint_position @ rotation.T + translation,
    )
    normal = rotation @ np.array([0.0, 1.0, 0.0])
    collider = ph.RobotPlaneCollider("floor", 0, np.zeros(3), normal)
    result = step(
        custom(
            robot=rotated,
            position=[rotation @ np.array([0.0, 1 / 8, 0.0]) + translation],
            velocity=[rotation @ np.array([0.0, -1.0, 0.0])],
            colliders=(collider,),
        )
    )
    np.testing.assert_allclose(result.state.domains["robot"].v, [-1 / 3], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        result.state.domains["nodes"].velocity,
        [rotation @ np.array([0.0, -1 / 3, 0.0])],
        atol=1e-8,
        rtol=0,
    )
