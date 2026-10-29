"""Automatic native thick contact: independent A/B/C and material oracles."""

from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest
from test_thick_patch_geometry import SQUARE, TRI, model

import phymeo as ph


def build(
    mode="numpy",
    *,
    scenario="B",
    mu=0.5,
    material=False,
    h=0.25,
    program_options=None,
    base_transform=None,
    shared=False,
    reorder=False,
    rotate=False,
    start_position=None,
    start_velocity=None,
    patch_vertices=None,
):
    assert hasattr(ph, "DynamicRobotThickContactProgram"), "missing automatic thick-contact program"
    robot = model()
    position = TRI + np.array([0.0, 0.125, 0.0])
    velocity = np.tile([1.0, -1.0, 0.0], (3, 1))
    gravity = np.zeros(3)
    vertices = SQUARE if scenario == "C" else SQUARE * 8
    if scenario == "A":
        robot = replace(
            robot,
            parents=np.array([-1, -1]),
            joint_types=("prismatic", "prismatic"),
            joint_position=np.array([[0.0, -0.125, 0.0], [0.0, 0.125, 0.0]]),
            joint_orientation=np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
            joint_axis=np.tile([0.0, 1.0, 0.0], (2, 1)),
            mass=np.full(2, 2.0),
            com=np.zeros((2, 3)),
            inertia_diagonal=np.ones((2, 3)),
        )
        position = TRI - np.array([0.0, 0.125, 0.0])
        velocity = np.zeros((3, 3))
        gravity = np.array([0.0, 0.0, -1.0])
    elif scenario == "C":
        velocity = np.tile([0.0, -1.0, 0.0], (3, 1))
    if material:
        velocity = np.tile([0.0, -1.0, 0.0], (3, 1))
        mu = 0.0
    links = list(range(robot.dof_count))
    effort = np.array([2.0, -2.0]) if scenario == "A" else np.zeros(robot.dof_count)
    if shared:
        assert scenario == "A"
        robot = replace(
            robot,
            parents=np.array([-1, 0, 0]),
            joint_types=("prismatic",) * 3,
            joint_position=np.array([[0.0, 0.0, 0.0], [0.0, -0.125, 0.0], [0.0, 0.125, 0.0]]),
            joint_orientation=np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)),
            joint_axis=np.tile([0.0, 1.0, 0.0], (3, 1)),
            mass=np.array([1.0, 2.0, 2.0]),
            com=np.zeros((3, 3)),
            inertia_diagonal=np.ones((3, 3)),
        )
        links = [1, 2]
        effort = np.array([0.0, 2.0, -2.0])
    if rotate:
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        position = position @ rotation.T + np.array([2.0, -3.0, 1.0])
        velocity = velocity @ rotation.T
        gravity = rotation @ gravity
        robot = replace(
            robot,
            joint_position=robot.joint_position @ rotation.T + np.array([2.0, -3.0, 1.0]),
            joint_orientation=np.tile([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], (robot.dof_count, 1)),
        )
    if reorder:
        position = position[::-1].copy()
        velocity = velocity[::-1].copy()
    if start_position is not None:
        position = np.array(start_position, float)
    if start_velocity is not None:
        velocity = np.array(start_velocity, float)
    if patch_vertices is not None:
        vertices = np.array(patch_vertices, float)
    nodes = ph.ParticleModel(np.ones(3), gravity)
    constraints = (
        ph.DistanceConstraints(np.array([[0, 2]]), np.array([2 - 1 / 64]), np.array([1 / 16]))
        if material
        else None
    )
    scene = ph.SceneModel(
        (ph.DomainBinding("robot", "robot", robot), ph.DomainBinding("nodes", "particles", nodes))
    )
    state = ph.SceneState(
        {
            "robot": ph.RobotState(np.zeros(robot.dof_count), np.zeros(robot.dof_count)),
            "nodes": ph.ParticleState(position, velocity),
        }
    )
    inputs = ph.SceneInputs(
        {
            "robot": ph.RobotInputs(
                effort, np.zeros((robot.dof_count, 3)), np.zeros((robot.dof_count, 3))
            ),
            "nodes": ph.StepInputs.zeros(3),
        }
    )
    if mode == "numpy":
        participants = {
            "robot": ph.RobotParticipant(robot),
            "nodes": ph.ClothParticipant(nodes, constraints, record_material_path=True),
        }
        factory = ph.ArticulatedCouplingProgram
    else:
        participants = {
            "robot": ph.TyxRobotParticipant(robot, mode),
            "nodes": ph.TyxClothParticipant(nodes, constraints, mode, record_material_path=True),
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
    colliders = tuple(
        ph.RobotContactPatch("patch" + str(i), i, vertices[::-1] if reorder else vertices)
        for i in links
    )
    program = ph.DynamicRobotThickContactProgram(
        scene,
        base,
        (ph.TrianglePrismSurface("sheet", "nodes", (0, 1, 2), half_thickness_m=0.125),),
        tuple(reversed(colliders)) if reorder else colliders,
        friction_by_collider={c.key: mu for c in colliders},
        materials={"nodes": constraints},
        budgets=ph.MaterialContactBudgets(
            1e-3 if material else 1e-8 if mode == "numpy" else 1e-5,
            1e-3 if material else 1e-8 if mode == "numpy" else 1e-5,
        ),
        **(program_options or {}),
    )
    state = ph.SceneState(
        state.domains, {f.name: np.zeros(f.shape, f.dtype) for f in program.history_fields}
    )
    settings = ph.StepSettings(
        substeps=1,
        position_iterations=512,
        velocity_iterations=512,
        tolerance=1e-8 if mode == "numpy" else 2e-5,
        record_substeps=True,
    )
    return scene, state, inputs, program, settings, ph.BackendContext(program.capabilities.backend)


def step(case, h=0.25, accepted=True):
    scene, state, inputs, program, settings, context = case
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        with np.errstate(all="raise"):
            r = runtime.step(state, inputs, h)
    assert r.report.accepted == accepted, r.report.reason
    return r


def diag(result):
    return result.report.substeps[0].diagnostics["dynamic_robot_thick_contact"]


@pytest.mark.parametrize("mu", [1.0, 0.25])
def test_two_jaws_static_and_low_friction_single_step(mu):
    h = F(1, 4)
    case = build(scenario="A", mu=mu)
    r = step(case)
    d = diag(r)
    np.testing.assert_allclose(d["held_impulses"], np.full(6, float(2 * h / 3)), atol=1e-8, rtol=0)
    t = float(h / 2 if mu == 1.0 else h / 6)
    np.testing.assert_allclose(d["held_tangent"], np.tile([0.0, 0.0, t], (6, 1)), atol=1e-8, rtol=0)
    np.testing.assert_allclose(d["terminal_world"], 0, atol=1e-8)
    np.testing.assert_allclose(
        r.state.domains["nodes"].velocity,
        np.tile([0.0, 0.0, 0.0 if mu == 1.0 else -float(2 * h / 3)], (3, 1)),
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_allclose(r.state.domains["robot"].q, 0, atol=1e-8)
    assert all(c["status"] == "certified" for c in d["ccd"])


def test_open_impact_full_three_corner_oracle():
    r = step(build())
    d = diag(r)
    order = np.argsort(np.argmax(d["anchor_material"][:, :3], axis=1))
    np.testing.assert_allclose(d["held_impulses"], [1 / 5] * 3, atol=1e-8, rtol=0)
    np.testing.assert_allclose(d["held_tangent"], 0, atol=0)
    np.testing.assert_allclose(
        d["terminal_impulses"][order], [13 / 80, 1 / 5, 19 / 80], atol=1e-8, rtol=0
    )
    np.testing.assert_allclose(
        d["terminal_tangent"][order, 0], [-29 / 160, -1 / 5, -7 / 32], atol=1e-8, rtol=0
    )
    pre = d["held_state"]
    final = r.state
    np.testing.assert_allclose(
        pre.domains["nodes"].velocity, np.tile([1.0, -0.8, 0.0], (3, 1)), atol=1e-8, rtol=0
    )
    np.testing.assert_allclose(
        final.domains["nodes"].velocity,
        [[131 / 160, -0.6, 0.0], [4 / 5, -0.6, 0.0], [25 / 32, -0.6, 0.0]],
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_allclose(final.domains["robot"].v, [-0.6], atol=1e-8, rtol=0)
    np.testing.assert_allclose(final.domains["robot"].q, [-3 / 40], atol=1e-8, rtol=0)

    def kinetic(s):
        return 0.5 * np.sum(s.domains["nodes"].velocity ** 2) + np.sum(s.domains["robot"].v ** 2)

    assert abs(kinetic(pre) - float(F(51, 20))) < 1e-8
    assert abs(kinetic(final) - float(F(1905, 1024))) < 1e-8
    assert (
        abs(sum(final.domains["nodes"].velocity[:, 1]) + 2 * final.domains["robot"].v[0] + 3) < 1e-8
    )
    assert d["ccd"][0]["status"] == "certified"


def test_distance_material_pre_ledger_and_terminal_post():
    r = step(build(material=True))
    d = diag(r)
    pre = d["held_state"]
    np.testing.assert_allclose(
        pre.domains["nodes"].position[:, 0], [-1 + 1 / 192, 0.0, 1 - 1 / 192], atol=1e-8, rtol=0
    )
    np.testing.assert_allclose(
        pre.domains["nodes"].velocity[:, 0], [1 / 48, 0.0, -1 / 48], atol=1e-8, rtol=0
    )
    np.testing.assert_allclose(d["held_impulses"], [0.2] * 3, atol=1e-8, rtol=0)
    np.testing.assert_allclose(d["terminal_impulses"], [0.2] * 3, atol=1e-8, rtol=0)
    np.testing.assert_allclose(r.state.domains["nodes"].velocity[:, 1], -0.6, atol=1e-8, rtol=0)
    assert d["material_reports"]["nodes"]["max_residual"] <= 1e-8


def test_anchor_birth_never_copies_held_capacity():
    case = build(scenario="C", mu=0.0)
    r = step(case)
    d = diag(r)
    assert len(d["held_impulses"]) == 4
    np.testing.assert_allclose(d["held_impulses"], 0.15, atol=1e-8, rtol=0)
    np.testing.assert_allclose(d["terminal_impulses"], 0.15, atol=1e-8, rtol=0)
    # A later input-driven slide changes clipped material coordinates. Any birth is a new p=0 row.
    scene, state, inputs, program, settings, context = build(scenario="C", mu=0.0)
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        first = runtime.step(state, inputs, 0.25)
        assert first.report.accepted, first.report.reason
        slide = ph.SceneInputs(
            {**inputs.domains, "nodes": ph.StepInputs(np.tile([4.0, 0.0, 0.0], (3, 1)))}
        )
        second = runtime.step(first.state, slide, 0.25)
        assert second.report.accepted, second.report.reason
        e = diag(second)
        assert len(e["held_impulses"]) > 4
        assert all(
            p == 0 for p, born in zip(e["held_impulses"], e["terminal_only"], strict=True) if born
        )
        assert np.all(e["terminal_world"][~e["anchor_active"]] == 0)
        assert len({tuple(m) for m in e["anchor_material"]}) == len(e["anchor_material"])


def lifecycle(mode="numpy", h=0.25, rebuild=0):
    scene, state, inputs, program, settings, context = build(mode, scenario="C", mu=0.0)
    tol = 1e-8 if mode == "numpy" else 2e-5
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        impact = runtime.step(state, inputs, 0.25)
        assert impact.report.accepted, impact.report.reason
        d = diag(impact)
        assert abs(np.sum(d["held_impulses"]) - 0.6) < 4 * tol
        assert abs(np.sum(d["terminal_impulses"]) - 0.6) < 4 * tol
        state = impact.state
        origin = state.domains["nodes"].position.copy()
        events = []
        states = []
        for i in range(round(3 / h)):
            force = np.tile(
                [1 / h if i == 0 else -2 / h if i == round(2 / h) else 0.0, 0.0, 0.0], (3, 1)
            )
            control = ph.SceneInputs({**inputs.domains, "nodes": ph.StepInputs(force)})
            r = runtime.step(state, control, h)
            assert r.report.accepted, (i, r.report.reason)
            state = r.state
            d = diag(r)
            t = (i + 1) * h
            x = t if t <= 2 else 4 - t
            np.testing.assert_allclose(
                state.domains["nodes"].position[:, 0], origin[:, 0] + x, atol=4 * tol, rtol=0
            )
            np.testing.assert_allclose(
                state.domains["nodes"].velocity[:, 1], -0.6, atol=4 * tol, rtol=0
            )
            np.testing.assert_allclose(state.domains["robot"].v, -0.6, atol=4 * tol, rtol=0)
            assert np.max(np.abs(d["held_world"])) < 4 * tol if d["held_world"].size else True
            assert (
                np.max(np.abs(d["terminal_world"])) < 4 * tol if d["terminal_world"].size else True
            )
            # Independent triangle half-planes against the square: max admissible X is4/3.
            if x > 4 / 3 + tol:
                assert not np.any(d["anchor_active"])
            if x < 4 / 3 - tol:
                assert np.any(d["anchor_active"])
            events.extend(d["transitions"])
            states.append((x, tuple(d["transitions"])))
        assert "exit" in events and "enter" in events
        assert np.any(diag(r)["anchor_active"])
    return states


def test_partial_overlap_exit_and_input_driven_reentry():
    lifecycle()


def test_exact_grazing_reentry_prefers_patch_face_axis():
    from phymeo.geometry.prism_patch_manifold import triangle_prism_patch_manifold

    # Binary exact scaled R2 boundary: whole prism just touches on the cap and a side.
    x = np.array([[-3.0, 0.125, -3.0], [0.0, 0.125, 6.0], [3.0, 0.125, -3.0]]) + [4.0, 0.0, 0.0]
    p = ph.RobotContactPatch("p", 0, SQUARE * 3)
    m = triangle_prism_patch_manifold(
        model(), np.zeros(1), p, x, half_thickness_m=0.125, geometric_budget_m=0.0
    )
    np.testing.assert_array_equal(m.normal, [0.0, 1.0, 0.0])
    assert m.selected_axis.patch_face and m.gap_bounds == (0.0, 0.0)
    scene, state, inputs, program, settings, context = build(
        scenario="C",
        mu=0.0,
        start_position=x + [0.25, 0.0, 0.0],
        start_velocity=np.tile([-1.0, 0.0, 0.0], (3, 1)),
        patch_vertices=SQUARE * 3,
    )
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        for _ in range(2):
            r = runtime.step(state, inputs, 0.25)
            assert r.report.accepted, r.report.reason
            state = r.state
            d = diag(r)
            np.testing.assert_allclose(d["held_world"], 0.0, atol=1e-8, rtol=0)
            np.testing.assert_allclose(d["terminal_world"], 0.0, atol=1e-8, rtol=0)
            np.testing.assert_allclose(
                d["terminal_normal"][d["anchor_active"]],
                np.tile([0.0, 1.0, 0.0], (np.sum(d["anchor_active"]), 1)),
                atol=1e-14,
                rtol=0,
            )


def test_reordering_and_shared_ancestor_are_equivalent():
    original = step(build(scenario="A", mu=0.25))
    inverse_rotation = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    for kwargs in ({"reorder": True}, {"shared": True}, {"rotate": True}):
        result = step(build(scenario="A", mu=0.25, **kwargs))
        x = result.state.domains["nodes"].position.copy()
        v = result.state.domains["nodes"].velocity.copy()
        if kwargs.get("reorder"):
            x = x[::-1]
            v = v[::-1]
        if kwargs.get("rotate"):
            x = (x - [2.0, -3.0, 1.0]) @ inverse_rotation.T
            v = v @ inverse_rotation.T
        np.testing.assert_allclose(x, original.state.domains["nodes"].position, atol=1e-8, rtol=0)
        np.testing.assert_allclose(v, original.state.domains["nodes"].velocity, atol=1e-8, rtol=0)
        np.testing.assert_allclose(
            diag(result)["held_impulses"], diag(original)["held_impulses"], atol=1e-8, rtol=0
        )


def jaw_window(mode, h, mu):
    scene, state, inputs, program, settings, context = build(mode, scenario="A", mu=mu)
    total = 0.5 if mode == "numpy" else 0.25
    tol = 1e-8 if mode == "numpy" else 2e-5
    original = state.domains["nodes"].position.copy()
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        for i in range(round(total / h)):
            r = runtime.step(state, inputs, h)
            assert r.report.accepted, (i, r.report.reason)
            state = r.state
            t = F(i + 1) * F(h)
            hz = F(h)
            vz = F(0) if mu == 1.0 else -2 * t / 3
            dz = F(0) if mu == 1.0 else -(t * t + t * hz) / 3
            np.testing.assert_allclose(
                state.domains["nodes"].velocity[:, 2], float(vz), atol=4 * tol, rtol=0
            )
            np.testing.assert_allclose(
                state.domains["nodes"].position,
                original + [0.0, 0.0, float(dz)],
                atol=4 * tol,
                rtol=0,
            )
            np.testing.assert_allclose(state.domains["robot"].v, 0.0, atol=4 * tol, rtol=0)
            d = diag(r)
            assert (
                d["physical_budgets"]["frozen_model_energy"]["positive_gain_upper_j"]
                <= program.budgets.max_energy_gain_j
            )
    return abs(float(dz) + (0 if mu == 1.0 else total * total / 3))


def test_two_jaw_window_halving_and_rebuild():
    for mu in (1.0, 0.25):
        errors = []
        for h in (0.125, 0.0625):
            values = [jaw_window("numpy", h, mu) for _ in range(2)]
            assert values[0] == values[1]
            errors.append(values[0])
        if mu == 0.25:
            assert abs(errors[0] - 2 * errors[1]) < 1e-15
    for h in (0.25, 0.125):
        for rebuild in range(2):
            lifecycle("numpy", h, rebuild)
