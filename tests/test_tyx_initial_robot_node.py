"""Actual backend candidates for initial common-point paired replay."""

import numpy as np
import pytest
from test_initial_robot_node import diag, pair_case

from phymeo import StepSettings

pytestmark = pytest.mark.tyx


def settings():
    return StepSettings(substeps=1, tolerance=2e-5)


def test_actual_bounded_prismatic_pair(tyx_mode):
    runtime, state, inputs = pair_case(mode=tyx_mode, settings=settings())
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert result.report.accepted, result.report.reason
    np.testing.assert_allclose(result.state.domains["robot"].v, [-1 / 3], atol=2e-6)
    np.testing.assert_allclose(result.state.domains["nodes"].velocity, [[0, -1 / 3, 0]], atol=2e-6)
    np.testing.assert_allclose(diag(result)["world_impulses"], [[0, 2 / 3, 0]], atol=2e-6)
    assert diag(result)["replays"] == 2
    assert diag(result)["native_backend"] == "tyx:" + tyx_mode
    assert result.report.substeps[0].diagnostics["execution"]["completed_launches"] > 0


def test_actual_rotor_pair_angular_momentum(tyx_mode):
    runtime, state, inputs = pair_case(mode=tyx_mode, rotor=True, settings=settings())
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert result.report.accepted, result.report.reason
    robot, node = result.state.domains["robot"], result.state.domains["nodes"]
    np.testing.assert_allclose(robot.q, [-0.05], atol=2e-7)
    np.testing.assert_allclose(robot.v, [-0.5], atol=2e-6)
    np.testing.assert_allclose(node.position, [[1, -0.05, 0]], atol=2e-7)
    np.testing.assert_allclose(node.velocity, [[0, -0.5, 0]], atol=2e-6)
    assert abs(robot.v[0] + np.cross(node.position[0], node.velocity[0])[2] + 1) < 2e-6
    assert result.report.substeps[0].diagnostics["execution"]["completed_launches"] > 0


def test_actual_shared_rotor_contacts(tyx_mode):
    runtime, state, inputs = pair_case(
        mode=tyx_mode,
        rotor=True,
        points=(1.0, 2.0),
        node_v=[[0, -2, 0], [0, -3, 0]],
        settings=settings(),
    )
    with runtime:
        result = runtime.step(state, inputs, 0.05)
    assert result.report.accepted, result.report.reason
    robot, node = result.state.domains["robot"], result.state.domains["nodes"]
    np.testing.assert_allclose(diag(result)["world_impulses"][:, 1], [2 / 3, 1 / 3], atol=2e-5)
    assert abs(robot.v[0] + np.sum(np.cross(node.position, node.velocity)[:, 2]) + 8) < 2e-5
    assert result.report.substeps[0].diagnostics["execution"]["completed_launches"] > 0


def verify_material_side(mode):
    runtime, state, inputs = pair_case(mode=mode, side_material=True, settings=settings())
    with runtime:
        result = runtime.step(state, inputs, 0.01)
    assert result.report.accepted, result.report.reason
    multiplier = -0.1 / (2 + 0.001 / 0.01**2)
    expected = np.array(
        [[0.0, -multiplier - 9.81 * 0.01**2, 0], [0, 1.1 + multiplier - 9.81 * 0.01**2, 0]]
    )
    np.testing.assert_allclose(result.state.domains["cloth"].position, expected, atol=2e-6)
    np.testing.assert_allclose(result.state.domains["robot"].v, [-1 / 3], atol=2e-6)
    native = diag(result)["inner_report"].substeps[0]
    np.testing.assert_allclose(
        native.diagnostics["participants"]["cloth"]["distance_multipliers"], [multiplier], atol=2e-6
    )
    if mode != "numpy":
        assert result.report.substeps[0].diagnostics["execution"]["completed_launches"] > 0


def test_actual_material_side_domain(tyx_mode):
    verify_material_side(tyx_mode)
