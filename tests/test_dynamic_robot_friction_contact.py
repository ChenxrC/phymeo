"""Public split-friction contract and independent scene construction."""

from dataclasses import replace

import numpy as np
import pytest
from test_dynamic_robot_normal_contact import custom as normal_custom

import phymeo as ph


def build(mode="numpy", *, mu=2.0, chain=False, **kwargs):
    if chain:
        kwargs.update(
            position=[[0, 1 / 8, 0], [0, 9 / 8, 0]],
            velocity=[[1, -1, 0]] * 2,
            material=ph.DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.array([1 / 16])),
        )
    elif not kwargs:
        kwargs.update(position=[[0, 0, 0]], velocity=[[0, 0, 0]], gravity=(0, -1, 0), effort=1.0)
    scene, state, inputs, old, settings, context = normal_custom(mode=mode, **kwargs)
    assert hasattr(ph, "DynamicRobotFrictionContactProgram"), "missing automatic split-friction API"
    program = ph.DynamicRobotFrictionContactProgram(
        scene,
        old.base,
        old.colliders,
        friction_by_collider=mu if isinstance(mu, dict) else {c.key: mu for c in old.colliders},
        materials=old.materials,
        budgets=old.budgets,
    )
    if not chain and kwargs.get("gravity") == (0, -1, 0):
        force = np.tile([1.0, 0.0, 0.0], (len(state.domains["nodes"].position), 1))
        inputs = ph.SceneInputs({**inputs.domains, "nodes": ph.StepInputs(force)})
    history = {k: v for k, v in state.history.items() if not k.startswith("normal_contact/")}
    state = ph.SceneState(state.domains, {**history, **program.initial_history()})
    return scene, state, inputs, program, settings, context


def step(case, dt=0.25):
    scene, state, inputs, program, settings, context = case
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        result = runtime.step(state, inputs, dt)
    assert result.report.accepted, result.report.reason
    return result


def diagnostics(result):
    return result.report.substeps[-1].diagnostics["dynamic_robot_friction_contact"]


def test_public_owned_schema_and_history():
    case = build()
    p = case[3]
    assert p.capabilities.features == frozenset({"dynamic_robot_friction_split"})
    assert set(p.initial_history()) == {
        "friction_contact/active",
        "friction_contact/initial_closed",
        "friction_contact/phase_world",
    }
    assert p.initial_history()["friction_contact/phase_world"].shape == (1, 2, 3)
    assert p.problem_signature()["friction_by_collider"] == {"floor": 2.0}
    with pytest.raises(TypeError):
        p.friction_by_collider["floor"] = 1.0
    p.close()


@pytest.mark.parametrize(
    "mapping",
    [
        None,
        [],
        {},
        {"floor": 2.0, "extra": 0.0},
        {"floor": True},
        {"floor": 1},
        {"floor": np.float64(1)},
        {"floor": -1.0},
        {"floor": float("nan")},
        {"floor": float("inf")},
    ],
)
def test_invalid_mu_mapping(mapping):
    scene, _, _, old, _, _ = normal_custom()
    assert hasattr(ph, "DynamicRobotFrictionContactProgram"), "missing automatic split-friction API"
    try:
        with pytest.raises((TypeError, ValueError)):
            ph.DynamicRobotFrictionContactProgram(
                scene,
                old.base,
                old.colliders,
                friction_by_collider=mapping,
                materials=old.materials,
                budgets=old.budgets,
            )
    finally:
        old.close()


def test_nonzero_radius_is_rejected_even_at_zero_mu():
    scene, _, _, old, _, _ = normal_custom()
    assert hasattr(ph, "DynamicRobotFrictionContactProgram"), "missing automatic split-friction API"
    try:
        with pytest.raises(ValueError, match="radius"):
            ph.DynamicRobotFrictionContactProgram(
                scene,
                old.base,
                [replace(old.colliders[0], radius_m=0.01)],
                friction_by_collider={"floor": 0.0},
                materials=old.materials,
                budgets=old.budgets,
            )
    finally:
        old.close()


def test_empty_selected_domain_has_no_pseudo_contacts():
    result = step(build(position=np.empty((0, 3)), velocity=np.empty((0, 3))))
    d = diagnostics(result)
    assert d["row_keys"] == ()
    assert d["held_world"].shape == (0, 3)
    assert d["terminal_world"].shape == (0, 3)


@pytest.mark.parametrize(
    "kind", ["inactive", "classification", "negative", "cone", "separated", "shape", "dtype", "nan"]
)
def test_history_validation(kind):
    case = build(position=[[0.0, 0.0, 0.0]], velocity=[[0.0, 0.0, 0.0]])
    history = case[3].initial_history()
    if kind not in ("inactive", "classification", "shape", "dtype", "nan"):
        history["friction_contact/active"][:] = True
    if kind == "inactive":
        history["friction_contact/phase_world"][0, 0, 1] = 1
    if kind == "classification":
        history["friction_contact/initial_closed"][:] = True
    if kind == "negative":
        history["friction_contact/phase_world"][0, 0, 1] = -1
    if kind == "cone":
        history["friction_contact/phase_world"][0, 0, 0] = 1
    if kind == "nan":
        history["friction_contact/phase_world"][0, 0, 0] = np.nan
    if kind == "shape":
        history["friction_contact/phase_world"] = np.zeros((1, 2))
    if kind == "dtype":
        history["friction_contact/phase_world"] = np.zeros((1, 2, 3), np.float32)
    domains = dict(case[1].domains)
    if kind == "separated":
        domains["nodes"] = ph.ParticleState(np.array([[0.0, 1.0, 0.0]]), np.zeros((1, 3)))
    state = ph.SceneState(domains, history)
    with ph.SceneStepper(case[0], case[3], case[4], case[5], case[3].registry) as runtime:
        if kind in ("shape", "dtype", "nan"):
            with pytest.raises((TypeError, ValueError)):
                runtime.step(state, case[2], 0.25)
        else:
            result = runtime.step(state, case[2], 0.25)
            assert not result.report.accepted


def test_mu_owned_input_and_signature_and_no_history_warmstart():
    from phymeo.core.scene import fingerprint

    scene, state, inputs, old, settings, context = normal_custom(
        position=[[0.0, 0.0, 0.0]], velocity=[[0.0, 0.0, 0.0]]
    )
    mapping = {"floor": 2.0}
    p = ph.DynamicRobotFrictionContactProgram(
        scene,
        old.base,
        old.colliders,
        friction_by_collider=mapping,
        materials=old.materials,
        budgets=old.budgets,
    )
    before = fingerprint(p.problem_signature())
    mapping["floor"] = 50.0
    assert fingerprint(p.problem_signature()) == before
    h = p.initial_history()
    h["friction_contact/active"][:] = True
    h["friction_contact/initial_closed"][:] = True
    h["friction_contact/phase_world"][0, 0] = [0.5, 1.0, 0.0]
    result = step((scene, ph.SceneState(state.domains, h), inputs, p, settings, context))
    np.testing.assert_array_equal(diagnostics(result)["total_world"], 0.0)


def test_history_prefix_conflict_and_missing_trace():
    from test_terminal_impulse import ObservedProgram

    case = build(base_transform=ObservedProgram)
    try:
        with pytest.raises(ValueError, match="conflict"):
            ph.DynamicRobotFrictionContactProgram(
                case[0],
                case[3].base,
                case[3].colliders,
                friction_by_collider={"floor": 0.5},
                materials=case[3].materials,
                budgets=case[3].budgets,
                history_prefix="base",
            )
    finally:
        case[3].close()
    case = build(trace=False)
    with ph.SceneStepper(case[0], case[3], case[4], case[5], case[3].registry) as runtime:
        result = runtime.step(case[1], case[2], 0.25)
    assert not result.report.accepted and "path" in result.report.reason


def test_unsupported_domains_and_all_prismatic_side_branch():
    with pytest.raises(ValueError, match="dynamic"):
        build(inverse_mass=[0.0])
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
        build(robot=robot)
    c = ph.RobotPlaneCollider("same", 0, np.zeros(3), np.array([0.0, 2.0, 0.0]))
    with pytest.raises(ValueError, match="unique"):
        build(colliders=(c, c))


def test_initial_uncertain_interval_is_rejected(monkeypatch):
    import phymeo.solvers._dynamic_friction_workspace as module

    geometry = module.geometry

    def uncertain(w, state):
        result = geometry(w, state)
        bound = F(w.settings.tolerance) * w.exact_gap_scale
        return (*result[:3], np.array([float(bound)]), ((bound / 2, bound * 3 / 2),))

    from fractions import Fraction as F

    monkeypatch.setattr(module, "geometry", uncertain)
    case = build()
    with ph.SceneStepper(case[0], case[3], case[4], case[5], case[3].registry) as runtime:
        result = runtime.step(case[1], case[2], 0.25)
    assert not result.report.accepted and "uncertain" in result.report.reason


def test_final_uncertain_interval_is_rejected(monkeypatch):
    import phymeo.solvers.dynamic_robot_normal_contact as module

    original = module.geometry

    def uncertain(w, state):
        result = original(w, state)
        if getattr(w, "trial", None) is not None:
            return (*result[:3], np.array([float(w.bound)]), ((w.bound / 2, w.bound * 3 / 2),))
        return result

    monkeypatch.setattr(module, "geometry", uncertain)
    c = build(position=[[0.0, 1.0, 0.0]], velocity=[[0.0, 0.0, 0.0]])
    with ph.SceneStepper(c[0], c[3], c[4], c[5], c[3].registry) as runtime:
        result = runtime.step(c[1], c[2], 0.25)
    assert not result.report.accepted and "uncertain" in result.report.reason


def test_non_collider_object_is_a_schema_error():
    scene, _, _, old, _, _ = normal_custom()
    try:
        with pytest.raises((TypeError, ValueError)):
            ph.DynamicRobotFrictionContactProgram(
                scene,
                old.base,
                (object(),),
                friction_by_collider={},
                materials=old.materials,
                budgets=old.budgets,
            )
    finally:
        old.close()
