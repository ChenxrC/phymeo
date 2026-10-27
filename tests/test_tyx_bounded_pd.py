"""Held effort drives actual robot prediction; controller predictions are never installed."""

import numpy as np
import pytest
from test_robot_kinematics import robot_model

from phymeo import (
    BackendContext,
    RobotInputs,
    RobotState,
    StepSettings,
    TyxRobotParticipant,
    solve_bounded_pd,
)
from phymeo.backends.tyx_robot_math import TyxRobotMath
from phymeo.dynamics.robot_dynamics import bias_forces, mass_matrix

pytestmark = pytest.mark.tyx


@pytest.mark.parametrize("limit", [0.1, 2.0])
def test_real_device_mass_bias_and_bounded_effort_prediction(limit, tyx_mode):
    model = robot_model()
    state = RobotState(np.array([0.2, -0.3]), np.array([-0.1, 0.2]))
    target, target_v, kp, kd = np.array([0.25, -0.2]), np.zeros(2), np.full(2, 6.0), np.full(2, 0.4)
    engine = TyxRobotMath(model, tyx_mode)
    try:
        before = engine.stats()["completed_launches"]
        mass, bias = engine.mass(state.q), engine.bias(state.q, state.v)
        assert engine.stats()["completed_launches"] > before
        command = solve_bounded_pd(
            mass,
            bias,
            state.q,
            state.v,
            target,
            target_v,
            kp,
            kd,
            np.full(2, limit),
            0.001,
            mass_tolerance=2e-5,
        )
    finally:
        engine.close()
    inputs = RobotInputs(command.effort, np.zeros((2, 3)), np.zeros((2, 3)))
    factory = TyxRobotParticipant(model, tyx_mode)
    ws = factory.create_workspace(
        "arm", state, inputs, 0.001, StepSettings(), BackendContext("tyx:" + tyx_mode, np.float64)
    )
    try:
        before = ws.engine.stats()["completed_launches"]
        ws.predict()
        after = ws.result()
        assert ws.engine.stats()["completed_launches"] > before
        residual = (
            mass_matrix(model, state.q) @ ((after.v - state.v) / 0.001)
            + bias_forces(model, state)
            - command.effort
        )
        np.testing.assert_allclose(residual, 0.0, atol=2e-5, rtol=2e-5)
        expected_effort = np.clip(
            kp * (target - after.q) + kd * (target_v - after.v), -limit, limit
        )
        np.testing.assert_allclose(command.effort, expected_effort, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(after.v, command.predicted_velocity, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(after.q, command.predicted_configuration, atol=2e-5, rtol=2e-5)
        assert np.all(np.abs(inputs.effort) <= limit)
        np.testing.assert_array_equal(state.q, [0.2, -0.3])
        np.testing.assert_array_equal(state.v, [-0.1, 0.2])
    finally:
        ws.close()
