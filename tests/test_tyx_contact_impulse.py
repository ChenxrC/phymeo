"""Fixed-configuration pinch impulses using two actual robot runtimes and cloth."""

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
    solve_contact_impulses,
)

pytestmark = pytest.mark.tyx


@pytest.mark.parametrize("mu,expected_vertical", [(0.8, 0.0), (0.0001, -0.019984)])
def test_actual_two_robot_closed_contact_impulses_and_low_mu(mu, expected_vertical, tyx_mode):
    with ExitStack() as resources:
        workspaces, ports = {}, {}
        context = BackendContext("tyx:" + tyx_mode, np.float64)
        for name in ("A", "B", "cloth"):
            if name == "cloth":
                model = ParticleModel(np.array([4.0]), np.zeros(3))
                state = ParticleState(np.array([[0.0, -1, 0]]), np.zeros((1, 3)))
                factory, inputs = TyxClothParticipant(model, None, tyx_mode), StepInputs.zeros(1)
                indices, local = np.array([0]), None
                preload = np.array([[0.0, -0.005, 0]])
            else:
                model = robot_model(gravity=np.zeros(3))
                state, inputs = RobotState(np.zeros(2), np.zeros(2)), RobotInputs.zeros(2)
                factory = TyxRobotParticipant(model, tyx_mode)
                indices, local = np.array([1]), np.array([[0.0, -0.5, 0]])
                preload = np.array([[0.02 if name == "A" else -0.02, 0, 0]])
            ws = factory.create_workspace(
                name, state, inputs, 0.01, StepSettings(tolerance=2e-6), context
            )
            resources.callback(ws.close)
            workspaces[name] = ws
            ws.predict()
            ws.prepare_velocity()
            ws.apply_velocity_impulses(indices, local, preload)
            ports[name] = ws.point_port(indices, local)
        graph = MultiDomainPointOperator(ports, [("A", 0), ("B", 0)], [("cloth", 0), ("cloth", 0)])
        snapshots = graph.snapshot()
        launches = {
            name: ws.diagnostics()["execution"]["completed_launches"]
            for name, ws in workspaces.items()
        }
        result = solve_contact_impulses(
            graph,
            np.array([[-1.0, 0, 0], [1.0, 0, 0]]),
            np.full(2, mu),
            np.full(2, mu),
            velocity_tolerance=2e-5,
            response_tolerance=2e-5,
        )
        np.testing.assert_allclose(result.normal_impulses, [0.02, 0.02], atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(
            result.states["cloth"].velocity, [[0, expected_vertical, 0]], atol=2e-5, rtol=2e-5
        )
        if mu < 0.01:
            assert result.states["cloth"].velocity[0, 1] < -0.019
        assert result.residual_mps <= 2e-5
        assert result.response_asymmetry <= 2e-5
        mass = planar_mass(np.zeros(2))
        energy = 0.0
        for name, ws in workspaces.items():
            assert ws.diagnostics()["execution"]["completed_launches"] > launches[name]
            before, after = snapshots[name], result.states[name]
            if name == "cloth":
                energy += float(np.sum(after.velocity**2 - before.velocity**2)) / 8
                np.testing.assert_array_equal(ws.velocity(), before.velocity)
                np.testing.assert_array_equal(after.position, before.position)
            else:
                energy += float(after.v @ mass @ after.v - before.v @ mass @ before.v) / 2
                np.testing.assert_array_equal(ws.velocity(), before.v)
                np.testing.assert_array_equal(after.q, before.q)
                edge = 0 if name == "A" else 1
                expected_v = before.v + np.linalg.solve(
                    mass, np.array([1.0, 0.5]) * result.world_impulses[edge, 0]
                )
                np.testing.assert_allclose(after.v, expected_v, atol=2e-5, rtol=2e-5)
                point_v = np.array([np.array([1.0, 0.5]) @ after.v, 0.0, 0.0])
                relative_v = point_v - result.states["cloth"].velocity[0]
                np.testing.assert_allclose(
                    relative_v, result.relative_velocity_after[edge], atol=2e-5, rtol=2e-5
                )
        assert energy < 0
        energy_tolerance = 2e-5 * sum(np.linalg.norm(row) for row in result.world_impulses) + 1e-12
        assert abs(energy - result.energy_change) <= energy_tolerance
