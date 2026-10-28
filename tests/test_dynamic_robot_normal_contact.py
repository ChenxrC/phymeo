"""Automatic normal rows and independent physical examples."""

from fractions import Fraction

import numpy as np
import pytest
from test_terminal_impulse import fixture

import phymeo as ph


def build(mode="numpy", chain=False, *, colliders=None, budgets=None, base_transform=None):
    assert hasattr(ph, "RobotPlaneCollider"), "missing owned plane collider API"
    assert hasattr(ph, "DynamicRobotNormalContactProgram"), "missing automatic normal program API"
    scene, state, inputs, base, settings, context, material = fixture(mode, chain)
    if base_transform:
        base = base_transform(base)
    settings = ph.StepSettings(
        substeps=1,
        position_iterations=128,
        velocity_iterations=128,
        tolerance=settings.tolerance,
        record_substeps=True,
    )
    collider = ph.RobotPlaneCollider("floor", 0, np.zeros(3), np.array([0.0, 1.0, 0.0]))
    program = ph.DynamicRobotNormalContactProgram(
        scene,
        base,
        (collider,) if colliders is None else colliders,
        materials={"nodes": material},
        budgets=budgets or ph.MaterialContactBudgets(1e-6, 1e-6),
    )
    state = ph.SceneState(state.domains, program.initial_history())
    return scene, state, inputs, program, settings, context


def step(case, dt=0.25):
    scene, state, inputs, program, settings, context = case
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        result = runtime.step(state, inputs, dt)
    assert result.report.accepted, result.report.reason
    return result


def test_collider_owned_schema():
    assert hasattr(ph, "RobotPlaneCollider"), "missing owned plane collider API"
    p, n = np.zeros(3), np.array([0.0, 2.0, 0.0])
    c = ph.RobotPlaneCollider("floor", 0, p, n)
    p[:] = 7
    n[:] = 8
    np.testing.assert_array_equal(c.local_point, np.zeros(3))
    np.testing.assert_array_equal(c.local_normal, [0.0, 1.0, 0.0])
    assert not c.local_point.flags.writeable and not c.local_normal.flags.writeable


@pytest.mark.parametrize("chain", [False, True])
def test_automatic_free_and_material_fraction_oracle(chain):
    case = build(chain=chain)
    result = step(case)
    d = result.report.substeps[0].diagnostics["dynamic_robot_normal_contact"]
    f = Fraction
    p = f(3, 7) if chain else f(1, 3)
    np.testing.assert_allclose(d["held_impulses"], [float(p)] + ([0.0] if chain else []), atol=1e-8)
    np.testing.assert_allclose(
        d["terminal_impulses"], [1 / 3] + ([0.0] if chain else []), atol=1e-8
    )
    target_q = -f(3, 56) if chain else -f(1, 24)
    target_v = -f(8, 21) if chain else -f(1, 3)
    np.testing.assert_allclose(result.state.domains["robot"].q, [float(target_q)], atol=1e-8)
    np.testing.assert_allclose(result.state.domains["robot"].v, [float(target_v)], atol=1e-8)
    velocity = result.state.domains["nodes"].velocity[:, 1]
    np.testing.assert_allclose(velocity, [float(target_v)] + ([-6 / 7] if chain else []), atol=1e-8)
    assert (
        abs(float(2 * result.state.domains["robot"].v[0] + sum(velocity)) + (2 if chain else 1))
        < 1e-8
    )
    assert tuple(d["events"]) == ("enter",) + (("inactive",) if chain else ())
    assert "terminal_point_impulses" not in case[3].capabilities.features
    assert case[3].capabilities.history == "explicit"


def custom(
    *,
    mode="numpy",
    position=None,
    velocity=None,
    gravity=(0.0, 0.0, 0.0),
    effort=0.0,
    material=None,
    colliders=None,
    inverse_mass=None,
    robot=None,
    trace=True,
    budgets=None,
    inner_settings=None,
    base_transform=None,
):
    """Only scene construction, independent of the normal solver's private helpers."""
    scene0, _, _, base0, _, _, _ = fixture(mode)
    robot = scene0.domains[0].model if robot is None else robot
    base0.close()
    position = np.array([[0.0, 1 / 8, 0.0]] if position is None else position, dtype=float).reshape(
        -1, 3
    )
    n = len(position)
    velocity = (
        np.tile([0.0, -1.0, 0.0], (n, 1))
        if velocity is None
        else np.array(velocity, float).reshape(-1, 3)
    )
    nodes = ph.ParticleModel(
        np.ones(n) if inverse_mass is None else np.asarray(inverse_mass, float),
        np.asarray(gravity, float),
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
                np.full(robot.dof_count, effort),
                np.zeros((robot.dof_count, 3)),
                np.zeros((robot.dof_count, 3)),
            ),
            "nodes": ph.StepInputs.zeros(n),
        }
    )
    participants = (
        {
            "robot": ph.RobotParticipant(robot),
            "nodes": ph.ClothParticipant(nodes, material, record_material_path=trace),
        }
        if mode == "numpy"
        else {
            "robot": ph.TyxRobotParticipant(robot, mode),
            "nodes": ph.TyxClothParticipant(nodes, material, mode, record_material_path=trace),
        }
    )
    factory = ph.ArticulatedCouplingProgram if mode == "numpy" else ph.TyxArticulatedCouplingProgram
    base = factory(
        scene,
        participants,
        ph.RobotSphereAttachments(np.empty(0, int), np.empty((0, 3)), np.empty(0)),
        (),
    )
    if base_transform:
        base = base_transform(base)
    collider = ph.RobotPlaneCollider("floor", 0, np.zeros(3), np.array([0.0, 1.0, 0.0]))
    program = ph.DynamicRobotNormalContactProgram(
        scene,
        base,
        (collider,) if colliders is None else colliders,
        materials={"nodes": material},
        budgets=budgets or ph.MaterialContactBudgets(1e-6, 1e-6),
        inner_settings=inner_settings,
    )
    history = program.initial_history()
    history.update({f.name: np.zeros(f.shape, f.dtype) for f in base.history_fields})
    state = ph.SceneState(state.domains, history)
    settings = ph.StepSettings(
        substeps=1,
        position_iterations=128,
        velocity_iterations=128,
        tolerance=1e-8 if mode == "numpy" else 2e-5,
        record_substeps=True,
    )
    return scene, state, inputs, program, settings, ph.BackendContext(program.capabilities.backend)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"key": " floor"},
        {"key": ""},
        {"key": 7},
        {"link_index": True},
        {"link_index": -1},
        {"link_index": 0.0},
        {"local_point": np.zeros(3, dtype=np.float32)},
        {"local_normal": np.zeros(3)},
        {"local_normal": np.array([0.0, np.inf, 0.0])},
        {"radius_m": 1},
        {"radius_m": -1.0},
        {"radius_m": float("nan")},
        {"friction": 0.1},
        {"friction": True},
    ],
)
def test_collider_rejects_unsupported_schema(kwargs):
    args = dict(
        key="floor", link_index=0, local_point=np.zeros(3), local_normal=np.array([0.0, 1.0, 0.0])
    )
    args.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        ph.RobotPlaneCollider(**args)


def test_collider_requires_builtin_key():
    class Name(str):
        pass

    with pytest.raises((TypeError, ValueError)):
        ph.RobotPlaneCollider(Name("floor"), 0, np.zeros(3), np.array([0.0, 1.0, 0.0]))


def test_unsupported_scope_and_missing_trace():
    with pytest.raises(ValueError, match="dynamic"):
        custom(inverse_mass=[0.0])
    case = custom(trace=False)
    with ph.SceneStepper(case[0], case[3], case[4], case[5], case[3].registry) as runtime:
        result = runtime.step(case[1], case[2], 0.25)
    assert not result.report.accepted and "path" in result.report.reason
    from dataclasses import replace

    robot = fixture()[0].domains[0].model
    with pytest.raises(ValueError, match="all-prismatic"):
        custom(robot=replace(robot, joint_types=("revolute",)))
    collider = ph.RobotPlaneCollider("same", 0, np.zeros(3), np.array([0.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match="unique"):
        custom(colliders=(collider, collider))
    from test_terminal_impulse import ObservedProgram

    case = custom(base_transform=lambda base: ObservedProgram(base, missing=True))
    with ph.SceneStepper(case[0], case[3], case[4], case[5], case[3].registry) as runtime:
        result = runtime.step(case[1], case[2], 0.25)
    assert not result.report.accepted and "callable" in result.report.reason
    assert all(ws.counts["close"] == 1 for ws in case[3].base.workspaces)


def test_empty_selected_domain_and_binding_signature_ownership():
    case = custom(position=np.empty((0, 3)), velocity=np.empty((0, 3)))
    result = step(case)
    d = result.report.substeps[0].diagnostics["dynamic_robot_normal_contact"]
    assert d["row_keys"] == () and d["held_impulses"].shape == (0,)
    case = build(chain=True)
    from phymeo.core.scene import fingerprint

    signature = fingerprint(case[3].problem_signature())
    assert signature == fingerprint(case[3].problem_signature())
    assert not case[3].materials["nodes"].rest_length.flags.writeable
    case[3].close()


def test_history_validation_and_prefix_collision():
    from test_terminal_impulse import ObservedProgram

    case = custom(base_transform=ObservedProgram)
    case[3].base.history_fields += (ph.HistoryField("base/active", (1,), bool),)
    with pytest.raises(ValueError, match="conflict"):
        ph.DynamicRobotNormalContactProgram(
            case[0],
            case[3].base,
            case[3].colliders,
            materials={"nodes": None},
            budgets=ph.MaterialContactBudgets(1e-6, 1e-6),
            history_prefix="base",
        )
    case[3].close()
    for kind in ("negative", "inactive", "separated", "shape"):
        case = build()
        history = case[3].initial_history()
        if kind == "negative":
            history["normal_contact/phase_impulses"][0, 0] = -1
        if kind == "inactive":
            history["normal_contact/phase_impulses"][0, 0] = 1
        if kind == "separated":
            history["normal_contact/active"][0] = True
        if kind == "shape":
            history["normal_contact/active"] = np.zeros(2, bool)
        state = ph.SceneState(case[1].domains, history)
        with ph.SceneStepper(case[0], case[3], case[4], case[5], case[3].registry) as runtime:
            if kind == "shape":
                with pytest.raises(ValueError, match="shape"):
                    runtime.step(state, case[2], 0.25)
                continue
            result = runtime.step(state, case[2], 0.25)
        assert not result.report.accepted


def test_radius_side_deep_penetration_and_zero_gap_normal():
    c = ph.RobotPlaneCollider("radius", 0, np.zeros(3), np.array([0.0, 1.0, 0.0]), radius_m=0.5)
    result = step(custom(position=[[0.0, 5 / 8, 0.0]], colliders=(c,)))
    np.testing.assert_allclose(
        result.state.domains["nodes"].position[:, 1] - result.state.domains["robot"].q,
        [0.5],
        atol=1e-8,
    )
    result = step(custom(position=[[0.0, 0.0, 0.0]], velocity=[[0.0, 0.0, 0.0]]))
    assert result.report.substeps[0].diagnostics["dynamic_robot_normal_contact"]["events"] == (
        "enter",
    )
    case = custom(position=[[0.0, -0.01, 0.0]])
    with ph.SceneStepper(case[0], case[3], case[4], case[5], case[3].registry) as runtime:
        result = runtime.step(case[1], case[2], 0.25)
    assert not result.report.accepted and "penetration" in result.report.reason
    c = ph.RobotPlaneCollider("down", 0, np.zeros(3), np.array([0.0, -1.0, 0.0]))
    result = step(custom(position=[[0.0, -1 / 8, 0.0]], velocity=[[0.0, 1.0, 0.0]], colliders=(c,)))
    np.testing.assert_allclose(result.state.domains["robot"].v, [1 / 3], atol=1e-8)


def test_revolute_off_contact_branch_and_missing_native_capability_are_rejected():
    from dataclasses import replace

    robot = ph.RobotModel(
        np.array([-1, -1]),
        ("prismatic", "revolute"),
        np.zeros((2, 3)),
        np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.ones(2),
        np.zeros((2, 3)),
        np.ones((2, 3)),
        np.zeros(3),
    )
    with pytest.raises(ValueError, match="all-prismatic"):
        custom(robot=robot)

    def no_capability(base):
        base.capabilities = replace(
            base.capabilities, features=base.capabilities.features - {"terminal_point_impulses"}
        )
        return base

    with pytest.raises(ValueError, match="capability"):
        custom(base_transform=no_capability)


def test_uncertain_closure_is_rejected_without_host_repair(monkeypatch):
    import phymeo.solvers.dynamic_robot_normal_contact as module

    original = module.geometry
    case = custom(position=[[0.0, 1.0, 0.0]])
    initial = case[1].domains["nodes"].position.copy()

    def uncertain(workspace, state):
        result = original(workspace, state)
        if getattr(workspace, "trial", None) is not None:
            # An enclosure straddling the closed/open threshold is deliberately
            # insufficient evidence; do not classify from its midpoint.
            bounds = ((workspace.bound / 2, workspace.bound * 3 / 2),)
            return (*result[:3], np.array([float(workspace.bound)]), bounds)
        return result

    monkeypatch.setattr(module, "geometry", uncertain)
    with ph.SceneStepper(case[0], case[3], case[4], case[5], case[3].registry) as runtime:
        result = runtime.step(case[1], case[2], 0.25)
    assert not result.report.accepted and "uncertain" in result.report.reason
    np.testing.assert_array_equal(result.state.domains["nodes"].position, initial)


def test_inner_native_budget_is_independent_and_outer_exhaustion_is_explicit():
    from dataclasses import replace

    inner = ph.StepSettings(
        substeps=1, position_iterations=128, velocity_iterations=128, tolerance=1e-8
    )
    material = ph.DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.array([1 / 16]))
    case = custom(
        position=[[0.0, 1 / 8, 0.0], [0.0, 9 / 8, 0.0]],
        velocity=np.tile([0.0, -1.0, 0.0], (2, 1)),
        material=material,
        inner_settings=inner,
    )
    outer = replace(case[4], position_iterations=1)
    with ph.SceneStepper(case[0], case[3], outer, case[5], case[3].registry) as runtime:
        result = runtime.step(case[1], case[2], 0.25)
    assert not result.report.accepted and "position iteration budget" in result.report.reason
    case = custom(
        position=[[0.0, 1 / 8, 0.0], [0.0, 9 / 8, 0.0]],
        velocity=np.tile([0.0, -1.0, 0.0], (2, 1)),
        material=material,
        inner_settings=inner,
    )
    result = step(
        (
            case[0],
            case[1],
            case[2],
            case[3],
            replace(case[4], position_iterations=64, velocity_iterations=64),
            case[5],
        )
    )
    assert result.report.accepted
    assert diagnostics_count(result) == 1


def diagnostics_count(result):
    return result.report.substeps[0].diagnostics["dynamic_robot_normal_contact"][
        "terminal_evidence"
    ]["pre_position_iterations"]


def test_capabilities_do_not_borrow_base_laws_or_live_operations():
    from dataclasses import replace

    def extra_feature(base):
        base.capabilities = replace(
            base.capabilities,
            features=base.capabilities.features | {"borrowed_unsupported_feature"},
        )
        return base

    case = build(base_transform=extra_feature)
    assert case[3].capabilities.features == frozenset({"dynamic_robot_normal_split"})
    assert case[3].capabilities.history == "explicit"
    case[3].close()
