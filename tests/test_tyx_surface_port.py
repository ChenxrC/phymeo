"""Real backend base responses with explicitly host-owned FP64 surface maps."""

import numpy as np
import pytest
from test_robot_dynamics import planar_mass
from test_robot_kinematics import robot_model

from phymeo import (
    ParticleModel,
    ParticleState,
    PointPairOperator,
    RobotInputs,
    RobotState,
    StepInputs,
    TyxClothParticipant,
    TyxRobotParticipant,
    WeightedPointPort,
)
from phymeo.core.scene import BackendContext, StepSettings

pytestmark = pytest.mark.tyx


def workspace(factory, state, inputs, mode):
    return factory.create_workspace(
        "surface",
        state,
        inputs,
        0.01,
        StepSettings(tolerance=2e-6),
        BackendContext("tyx:" + mode, np.float64),
    )


def test_real_node_weighted_pair_response_and_unchanged_snapshot(tyx_mode):
    model = ParticleModel(np.array([1.0, 2, 0]), np.zeros(3))
    state = ParticleState(np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]]), np.zeros((3, 3)))
    ws = workspace(TyxClothParticipant(model, None, tyx_mode), state, StepInputs.zeros(3), tyx_mode)
    try:
        ws.predict()
        base = ws.point_port(np.array([0, 1, 2]))
        port = WeightedPointPort(base, np.array([[0.5, 0.5, 0], [0, 0.25, 0.75]]))
        pair = PointPairOperator(port, np.array([[0, 1]]))
        before = ws.diagnostics()["execution"]["completed_launches"]
        p = np.array([[0.1, -0.2, 0.3], [0.2, 0.1, -0.1]])
        np.testing.assert_allclose(
            port.response(p), np.array([[0.75, 0.25], [0.25, 0.125]]) @ p, atol=2e-5, rtol=2e-5
        )
        np.testing.assert_allclose(pair.response(p[:1]), 0.375 * p[:1], atol=2e-5, rtol=2e-5)
        candidate = pair.apply_impulses(p[:1])
        np.testing.assert_allclose(
            candidate.velocity, [p[0] / 2, p[0] / 2, np.zeros(3)], atol=2e-5, rtol=2e-5
        )
        np.testing.assert_array_equal(candidate.position, base.snapshot().position)
        np.testing.assert_array_equal(ws.velocity(), state.velocity)
        candidate.velocity[:] = 99
        np.testing.assert_array_equal(pair.snapshot().velocity, state.velocity)
        assert ws.diagnostics()["execution"]["completed_launches"] > before
    finally:
        ws.close()


def test_real_robot_weighted_pair_shared_ancestor_response(tyx_mode):
    state = RobotState(np.zeros(2), np.zeros(2))
    factory = TyxRobotParticipant(robot_model(gravity=np.zeros(3)), tyx_mode)
    ws = workspace(factory, state, RobotInputs.zeros(2), tyx_mode)
    try:
        ws.predict()
        base = ws.point_port(np.array([0, 1]), np.array([[0.0, -0.5, 0], [0.0, -0.5, 0]]))
        port = WeightedPointPort(base, np.array([[1.0, 0], [0.25, 0.75]]))
        pair = PointPairOperator(port, np.array([[0, 1]]))
        before = ws.diagnostics()["execution"]["completed_launches"]
        j = np.array([[0.5, 0.0], [0.875, 0.375]])
        mass = planar_mass(np.zeros(2))
        p = np.array([[0.1, 0, 0], [-0.2, 0, 0]])
        np.testing.assert_allclose(
            port.response(p)[:, 0], j @ np.linalg.solve(mass, j.T @ p[:, 0]), atol=2e-5, rtol=2e-5
        )
        relative_j = np.array([-0.375, -0.375])
        delta = np.linalg.solve(mass, relative_j * p[0, 0])
        np.testing.assert_allclose(
            pair.response(p[:1]), [[relative_j @ delta, 0, 0]], atol=2e-5, rtol=2e-5
        )
        candidate = pair.apply_impulses(p[:1])
        np.testing.assert_allclose(candidate.v, delta, atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(candidate.q, state.q)
        np.testing.assert_array_equal(ws.velocity(), state.v)
        candidate.v[:] = 99
        np.testing.assert_array_equal(pair.snapshot().v, state.v)
        assert ws.diagnostics()["execution"]["completed_launches"] > before
    finally:
        ws.close()
