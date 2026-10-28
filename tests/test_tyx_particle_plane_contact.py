"""Actual tyx node contact/material with the robot domain advancing natively."""

from dataclasses import replace

import numpy as np
import pytest

from phymeo import (
    BackendContext,
    DistanceConstraints,
    DomainBinding,
    ParticleModel,
    ParticlePlaneContactProgram,
    ParticlePlaneContacts,
    ParticleState,
    RobotInputs,
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
    pendulum_chain,
)

pytestmark = pytest.mark.tyx


def actual_case(mode, *, mu=0.6, material=False):
    robot, _ = pendulum_chain(1)
    robot = replace(robot, gravity=np.zeros(3))
    nodes = ParticleModel(np.ones(2 if material else 1), np.array([0.0, -9.81, 0]))
    constraints = (
        DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.array([0.001])) if material else None
    )
    scene = SceneModel(
        (DomainBinding("robot", "robot", robot), DomainBinding("cloth", "particles", nodes))
    )
    positions = np.array([[0.0, 0, 0], [0, 1.0, 0]]) if material else np.zeros((1, 3))
    state = SceneState(
        {
            "robot": RobotState(np.array([0.2]), np.array([0.3])),
            "cloth": ParticleState(positions, np.zeros_like(positions)),
        }
    )
    force = np.zeros_like(positions)
    if not material:
        force[0, 0] = -20
    inputs = SceneInputs({"robot": RobotInputs.zeros(1), "cloth": StepInputs(force)})
    attachments = RobotSphereAttachments(np.empty(0, int), np.empty((0, 3)), np.empty(0))
    base = TyxArticulatedCouplingProgram(
        scene,
        {
            "robot": TyxRobotParticipant(robot, mode),
            "cloth": TyxClothParticipant(nodes, constraints, mode),
        },
        attachments,
        [],
    )
    contact = ParticlePlaneContacts(
        "cloth",
        np.array([0]),
        np.array([[0.0, 1, 0]]) if material else np.array([[1.0, 0, 0]]),
        np.zeros(1),
        np.array([0.0 if material else mu]),
    )
    program = ParticlePlaneContactProgram(scene, base, (contact,), registry=base.registry)
    runtime = SceneStepper(
        scene,
        program,
        StepSettings(substeps=1, tolerance=2e-5),
        BackendContext("tyx:" + mode),
        base.registry,
    )
    return runtime, state, inputs


@pytest.mark.parametrize("mu,acceleration", [(0.6, 0.0), (0.2, -5.81)])
def test_short_actual_support_and_sliding(tyx_mode, mu, acceleration):
    runtime, state, inputs = actual_case(tyx_mode, mu=mu)
    with runtime:
        for index in range(1, 9):
            result = runtime.step(state, inputs, 0.01)
            assert result.report.accepted, result.report.reason
            state = result.state
            diag = result.report.substeps[0].diagnostics
            assert diag["execution"]["completed_launches"] > 0
            assert diag["plane_contact"]["inner_report"].backend == "tyx:" + tyx_mode
            np.testing.assert_allclose(
                state.domains["cloth"].position,
                [[0, acceleration * 0.01**2 * index * (index + 1) / 2, 0]],
                atol=2e-5,
            )
            np.testing.assert_allclose(
                state.domains["cloth"].velocity, [[0, acceleration * 0.01 * index, 0]], atol=2e-5
            )
            np.testing.assert_allclose(
                state.domains["robot"].q, [0.2 + 0.3 * 0.01 * index], atol=2e-5
            )
            impulse = diag["plane_contact"]["world_impulses"]
            np.testing.assert_allclose(
                impulse, [[0.2, 0.0981 if mu == 0.6 else 0.04, 0]], atol=2e-5
            )


def test_short_actual_material_and_plane_joint_residual(tyx_mode):
    runtime, state, inputs = actual_case(tyx_mode, material=True)
    with runtime:
        result = runtime.step(state, inputs, 0.01)
    assert result.report.accepted, result.report.reason
    diag = result.report.substeps[0].diagnostics
    assert diag["execution"]["completed_launches"] > 0
    alpha = 0.001 / 0.01**2
    multiplier = 9.81 * 0.01**2 / (1 + alpha)
    np.testing.assert_allclose(
        result.state.domains["cloth"].position,
        [[0, 0, 0], [0, 1 - 9.81 * 0.01**2 + multiplier, 0]],
        atol=2e-6,
    )
    np.testing.assert_allclose(
        diag["plane_contact"]["world_impulses"],
        [[0, 9.81 * 0.01 * (2 + alpha) / (1 + alpha), 0]],
        atol=2e-5,
    )
    native = diag["plane_contact"]["inner_report"].substeps[0]
    assert native.max_residual <= 2e-5
    np.testing.assert_allclose(
        native.diagnostics["participants"]["cloth"]["distance_multipliers"], [multiplier], atol=2e-6
    )


def test_actual_fp32_prediction_range_gate_rolls_back(tyx_mode):
    runtime, state, inputs = actual_case(tyx_mode)
    state.domains["cloth"].velocity[0, 1] = -1e8
    with runtime:
        result = runtime.step(state, inputs, 0.01)
    assert not result.report.accepted
    assert "roundoff budget" in result.report.reason
    np.testing.assert_array_equal(result.state.domains["cloth"].velocity, [[0, -1e8, 0]])
    np.testing.assert_array_equal(result.state.domains["robot"].q, [0.2])
