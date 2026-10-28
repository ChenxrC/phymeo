"""Actual native robot contact and independently checked material side domain."""

import numpy as np
import pytest

from phymeo import (
    ArticulatedCouplingProgram,
    BackendContext,
    ClothParticipant,
    DistanceConstraints,
    DomainBinding,
    ParticleModel,
    ParticleState,
    RobotInputs,
    RobotModel,
    RobotParticipant,
    RobotPlaneContactProgram,
    RobotPlaneContacts,
    RobotSphereAttachments,
    RobotState,
    SceneInputs,
    SceneModel,
    SceneState,
    SceneStepper,
    StepInputs,
    StepSettings,
    TyxArticulatedCouplingProgram,
    TyxClothParticipant,
    TyxRobotParticipant,
)

pytestmark = pytest.mark.tyx


def make_case(mode, *, rotating=False, material=False):
    model = RobotModel(
        np.array([-1]),
        ("revolute" if rotating else "prismatic",),
        np.zeros((1, 3)),
        np.array([[0.0, 0, 0, 1]]),
        np.array([[0.0, 0, 1]]) if rotating else np.array([[0.0, 1, 0]]),
        np.array([1.0 if rotating else 2.0]),
        np.array([[0.0, -0.5, 0]]) if rotating else np.zeros((1, 3)),
        np.full((1, 3), 1 / 12) if rotating else np.ones((1, 3)),
        np.zeros(3) if rotating else np.array([0.0, -9.81, 0]),
    )
    robot_factory = RobotParticipant if mode == "numpy" else lambda m: TyxRobotParticipant(m, mode)
    participants = {"robot": robot_factory(model)}
    domains = [DomainBinding("robot", "robot", model)]
    states = {
        "robot": RobotState(
            np.array([-0.3 if rotating else 0.0]), np.array([1.0 if rotating else 0.0])
        )
    }
    inputs = {"robot": RobotInputs.zeros(1)}
    if material:
        nodes = ParticleModel(np.ones(2), np.array([0.0, -9.81, 0]))
        constraints = DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.array([0.001]))
        domains.append(DomainBinding("cloth", "particles", nodes))
        participants["cloth"] = (
            ClothParticipant(nodes, constraints)
            if mode == "numpy"
            else TyxClothParticipant(nodes, constraints, mode)
        )
        states["cloth"] = ParticleState(np.array([[0.0, 0, 0], [0, 1.1, 0]]), np.zeros((2, 3)))
        inputs["cloth"] = StepInputs.zeros(2)
    scene = SceneModel(tuple(domains))
    attachments = RobotSphereAttachments(np.empty(0, int), np.empty((0, 3)), np.empty(0))
    factory = ArticulatedCouplingProgram if mode == "numpy" else TyxArticulatedCouplingProgram
    base = factory(scene, participants, attachments, [])
    contact = RobotPlaneContacts(
        "robot",
        np.array([0]),
        np.array([[0.0, -1, 0]]) if rotating else np.zeros((1, 3)),
        np.array([[0.0, 1, 0]]),
        np.array([-np.cos(0.2) if rotating else 0.0]),
        np.zeros(1),
    )
    program = RobotPlaneContactProgram(scene, base, (contact,), registry=base.registry)
    backend = "numpy" if mode == "numpy" else "tyx:" + mode
    runtime = SceneStepper(
        scene,
        program,
        StepSettings(substeps=1, tolerance=2e-6 if rotating else 2e-5),
        BackendContext(backend),
        base.registry,
    )
    return runtime, SceneState(states), SceneInputs(inputs)


def test_actual_bounded_prismatic_support(tyx_mode):
    runtime, state, inputs = make_case(tyx_mode)
    with runtime:
        result = runtime.step(state, inputs, 0.01)
    assert result.report.accepted, result.report.reason
    report = result.report.substeps[0]
    diag = report.diagnostics["robot_plane_contact"]
    assert diag["coefficient_backend"] == "canonical_host_float64"
    assert diag["native_backend"] == "tyx:" + tyx_mode
    assert report.diagnostics["execution"]["completed_launches"] > 0
    np.testing.assert_allclose(result.state.domains["robot"].q, [0], atol=2e-7)
    np.testing.assert_allclose(result.state.domains["robot"].v, [0], atol=2e-5)
    np.testing.assert_allclose(diag["world_impulses"], [[0, 0.1962, 0]], atol=2e-5)
    assert diag["replays"] == 2


def test_actual_rotating_final_jacobian_contact(tyx_mode):
    runtime, state, inputs = make_case(tyx_mode, rotating=True)
    with runtime:
        result = runtime.step(state, inputs, 0.15)
    assert result.report.accepted, result.report.reason
    assert result.report.substeps[0].diagnostics["execution"]["completed_launches"] > 0
    np.testing.assert_allclose(result.state.domains["robot"].q, [-0.2], atol=2e-5)
    np.testing.assert_allclose(result.state.domains["robot"].v, [2 / 3], atol=2e-5)
    expected = (1 / 3) * (2 / 3 - 1) / np.sin(-0.2)
    diag = result.report.substeps[0].diagnostics["robot_plane_contact"]
    np.testing.assert_allclose(diag["world_impulses"], [[0, expected, 0]], atol=3e-5)
    np.testing.assert_allclose(diag["environment_moments"][0, 2], 1 / 9, atol=3e-5)


def verify_material_side(mode):
    runtime, state, inputs = make_case(mode, material=True)
    with runtime:
        result = runtime.step(state, inputs, 0.01)
    assert result.report.accepted, result.report.reason
    alpha = 0.001 / 0.01**2
    multiplier = -0.1 / (2 + alpha)
    q = np.array(
        [[0.0, -multiplier - 9.81 * 0.01**2, 0], [0, 1.1 + multiplier - 9.81 * 0.01**2, 0]]
    )
    np.testing.assert_allclose(result.state.domains["cloth"].position, q, atol=2e-6)
    np.testing.assert_allclose(result.state.domains["robot"].q, [0], atol=2e-7)
    native = (
        result.report.substeps[0].diagnostics["robot_plane_contact"]["inner_report"].substeps[0]
    )
    np.testing.assert_allclose(
        native.diagnostics["participants"]["cloth"]["distance_multipliers"], [multiplier], atol=2e-6
    )
    if mode != "numpy":
        assert result.report.substeps[0].diagnostics["execution"]["completed_launches"] > 0


def test_actual_robot_contact_preserves_native_cloth_material(tyx_mode):
    verify_material_side(tyx_mode)
