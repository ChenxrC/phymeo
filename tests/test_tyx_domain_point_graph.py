"""Two actual robot runtimes and a node runtime participate in one impulse graph."""

from contextlib import ExitStack

import numpy as np
import pytest
from test_robot_dynamics import planar_mass
from test_robot_kinematics import robot_model

from phymeo import (
    BackendContext,
    MultiDomainPointOperator,
    ParticleModel,
    ParticleState,
    RobotInputs,
    RobotState,
    StepInputs,
    StepSettings,
    TyxClothParticipant,
    TyxRobotParticipant,
)

pytestmark = pytest.mark.tyx


def test_actual_two_robot_and_cloth_domains_with_shared_response(tyx_mode):
    with ExitStack() as resources:
        workspaces, ports = {}, {}
        context = BackendContext("tyx:" + tyx_mode, np.float64)
        for name in ("A", "B", "cloth"):
            if name == "cloth":
                model = ParticleModel(np.array([4.0]), np.zeros(3))
                state = ParticleState(np.array([[1.0, 2, 3]]), np.zeros((1, 3)))
                inputs = StepInputs.zeros(1)
                factory = TyxClothParticipant(model, None, tyx_mode)
            else:
                model = robot_model(gravity=np.zeros(3))
                state, inputs = RobotState(np.zeros(2), np.zeros(2)), RobotInputs.zeros(2)
                factory = TyxRobotParticipant(model, tyx_mode)
            ws = factory.create_workspace(
                name, state, inputs, 0.01, StepSettings(tolerance=2e-6), context
            )
            resources.callback(ws.close)
            workspaces[name] = ws
            ws.predict()
            ports[name] = (
                ws.point_port(np.arange(1))
                if name == "cloth"
                else ws.point_port(np.arange(2), np.array([[0.0, -0.5, 0], [0.0, -0.5, 0]]))
            )
        operator = MultiDomainPointOperator(
            ports, [("A", 0), ("A", 1), ("B", 1)], [("cloth", 0), ("B", 0), ("cloth", 0)]
        )
        before = {
            name: ws.diagnostics()["execution"]["completed_launches"]
            for name, ws in workspaces.items()
        }
        snapshots = operator.snapshot()
        p = np.array([[0.1, 0, 0], [0.2, 0, 0], [-0.3, 0, 0]])
        mass, j = planar_mass(np.zeros(2)), np.array([[0.5, 0], [1.0, 0.5]])
        da = np.linalg.solve(mass, j.T @ p[:2, 0])
        db = np.linalg.solve(mass, j.T @ np.array([-p[1, 0], p[2, 0]]))
        dc = -4 * (p[0, 0] + p[2, 0])
        expected = np.array(
            [[j[0] @ da - dc, 0, 0], [j[1] @ da - j[0] @ db, 0, 0], [j[1] @ db - dc, 0, 0]]
        )
        np.testing.assert_allclose(operator.response(p), expected, atol=2e-5, rtol=2e-5)
        result = operator.apply_impulses(p)
        np.testing.assert_allclose(result["A"].v, da, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(result["B"].v, db, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(result["cloth"].velocity, [[dc, 0, 0]], atol=2e-5, rtol=2e-5)
        for name, ws in workspaces.items():
            assert ws.diagnostics()["execution"]["completed_launches"] > before[name]
            if name == "cloth":
                np.testing.assert_array_equal(ws.velocity(), snapshots[name].velocity)
                np.testing.assert_array_equal(result[name].position, snapshots[name].position)
            else:
                np.testing.assert_array_equal(ws.velocity(), snapshots[name].v)
                np.testing.assert_array_equal(result[name].q, snapshots[name].q)
