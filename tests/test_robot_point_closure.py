"""Fixed-base bilateral point closures, reaction work and independent full-window mechanics."""

import json
from dataclasses import replace

import numpy as np
import pytest
from reference_newton.robot.four_bar_oracle import case, mechanics, physical_work, points

from phymeo.core.robot import RobotInputs, RobotModel, RobotState
from phymeo.dynamics.robot_dynamics import mass_matrix
from phymeo.dynamics.robot_kinematics import forward_kinematics, point_jacobians
from phymeo.sensors import RobotSite
from phymeo.solvers.robot_closure import RobotPointClosure, step_robot_closures


def four_bar(row):
    u = row["frame"]
    orient = np.array([0, np.sqrt((1 - u[0, 0]) / 2), 0, np.sqrt((1 + u[0, 0]) / 2)])
    model = RobotModel(
        np.array([-1, 0, -1]),
        ("revolute",) * 3,
        np.array([row["origin"], [0.5, 0, 0], row["origin"] + u[:, 0]]),
        np.array([orient, [0.0, 0, 0, 1], orient]),
        np.tile([0.0, 0, 1], (3, 1)),
        np.array([1.0, 2, 1]),
        np.array([[0.25, 0, 0], [0.5, 0, 0], [0.25, 0, 0]]),
        np.repeat(np.array([0.0625, 0.125, 0.0625])[:, None], 3, axis=1),
        u @ row["gravity"],
    )
    tangent = np.array([1, -1, 1])
    state = RobotState(row["theta"] * tangent, row["speed"] * tangent)
    load = RobotInputs(
        np.array([row["drive"], 0.0, 0]), row["forces"] @ u.T, row["moments"][:, None] * u[:, 2]
    )
    closure = RobotPointClosure(RobotSite(1, (1, 0, 0)), RobotSite(2, (0.5, 0, 0)), u[:, :2].T)
    return model, state, load, closure


@pytest.mark.parametrize(
    ("tilted", "subdivision"),
    [(False, 1), (True, 1), (True, 2)],
    ids=["level-drive", "tilted-coarse", "tilted-refined"],
)
def test_four_bar_full_window_finite_reaction_work_and_one_physical_dof(tilted, subdivision):
    row = case(tilted)
    model, state, load, closure = four_bar(row)
    theta, speed, h = row["theta"], row["speed"], row["h"] / subdivision
    initial = state
    initial_centers = forward_kinematics(model, state.q).com_position
    kinetic0 = state.v @ mass_matrix(model, state.q) @ state.v / 2
    max_gap = max_slip = max_power = max_force_error = 0.0
    first_force = None
    for k in range(row["steps"] * subdivision):
        a, force, power = mechanics(row, theta, speed)
        result = step_robot_closures(model, state, load, [closure], h)
        assert result.report.accepted, result.report.reason
        reaction = result.reaction
        if k == 0:
            first_force = reaction.force_first_world[0].copy()
            if not tilted:
                np.testing.assert_allclose(first_force, [0.25, 0.125, 0], atol=2e-13)
                np.testing.assert_allclose(
                    [reaction.power_first_w[0], reaction.power_second_w[0]],
                    [-1 / 64, 1 / 64],
                    atol=2e-13,
                )
                np.testing.assert_allclose(
                    reaction.generalized_force, [1 / 16, 1 / 8, 1 / 16], atol=2e-13
                )
                np.testing.assert_allclose(
                    points(row, theta), [[0.3, 0.4, 0], [1.3, 0.4, 0]], atol=2e-14
                )
        np.testing.assert_allclose(reaction.acceleration, [a, -a, a], atol=5e-11, rtol=0)
        np.testing.assert_allclose(reaction.force_first_world[0], force, atol=5e-11, rtol=0)
        np.testing.assert_array_equal(reaction.force_second_world, -reaction.force_first_world)
        np.testing.assert_allclose(
            [reaction.power_first_w[0], reaction.power_second_w[0]],
            [power, -power],
            atol=5e-11,
            rtol=0,
        )
        np.testing.assert_allclose(reaction.generalized_force @ state.v, 0, atol=1e-11)
        max_force_error = max(
            max_force_error, np.linalg.norm(reaction.force_first_world[0] - force)
        )
        max_power = max(max_power, abs(reaction.power_first_w[0] + reaction.power_second_w[0]))
        speed += h * a
        theta += h * speed
        state = result.state
        np.testing.assert_allclose(state.q, [theta, -theta, theta], rtol=0, atol=3e-11)
        np.testing.assert_allclose(state.v, [speed, -speed, speed], rtol=0, atol=3e-11)
        geom = forward_kinematics(model, state.q)
        actual = np.stack(
            [geom.position[1], geom.position[1] + geom.rotation[1] @ np.array([1.0, 0, 0])]
        )
        right = geom.position[2] + geom.rotation[2] @ np.array([0.5, 0, 0])
        np.testing.assert_allclose(actual, points(row, theta), rtol=0, atol=3e-11)
        max_gap = max(max_gap, np.linalg.norm(actual[1] - right))
        max_slip = max(max_slip, result.velocity_residual_m_s)
    net_force = load.force + model.mass[:, None] * model.gravity
    changes = state.q - initial.q
    work = (
        load.effort @ changes
        + np.sum(net_force * (geom.com_position - initial_centers))
        + row["moments"] @ np.array([changes[0], changes[0] + changes[1], changes[2]])
    )
    energy_change = state.v @ mass_matrix(model, state.q) @ state.v / 2 - kinetic0
    np.testing.assert_allclose(work, physical_work(row, theta), rtol=0, atol=3e-11)
    np.testing.assert_allclose(
        energy_change, 0.375 * (speed**2 - row["speed"] ** 2), rtol=0, atol=3e-11
    )
    if not tilted or subdivision == 2:
        assert abs(energy_change - work) < 8e-4
    else:
        # The coarser reference is retained as a measured resolution outside the engineering band.
        assert abs(energy_change - work) > 8e-4
    assert max_gap < 1e-10 and max_slip < 1e-10 and max_power < 1e-10
    print(
        json.dumps(
            dict(
                tilted=tilted,
                steps=row["steps"] * subdivision,
                max_gap=max_gap,
                max_slip=max_slip,
                max_power=max_power,
                max_force_error=max_force_error,
                first_force=first_force.tolist(),
                theta=theta,
                speed=speed,
                work=work,
                energy_change=energy_change,
            )
        )
    )


def nonlinear():
    m = RobotModel(
        np.array([-1, -1]),
        ("revolute", "prismatic"),
        np.zeros((2, 3)),
        np.tile([0.0, 0, 0, 1], (2, 1)),
        np.array([[0.0, 0, 1], [1, 0, 0]]),
        np.ones(2),
        np.zeros((2, 3)),
        np.full((2, 3), 0.5),
        np.zeros(3),
    )
    s = RobotState(np.array([np.arctan2(0.8, 0.6), 0.3]), np.array([0.5, -0.2]))
    c = RobotPointClosure(RobotSite(0, (0.5, 0, 0)), RobotSite(1), [[1.0, 0, 0]])
    return m, s, RobotInputs(np.array([0.375, 0]), np.zeros((2, 3)), np.zeros((2, 3))), c


def test_nonlinear_closure_projects_on_mass_metric_and_keeps_tangent_velocity():
    m, s, inputs, c = nonlinear()
    max_it = 0
    for k in range(32):
        result = step_robot_closures(m, s, inputs, [c], 1 / 128)
        assert result.report.accepted, result.report.reason
        if k == 0:
            np.testing.assert_allclose(
                result.reaction.acceleration, [23 / 44, -25 / 88], atol=3e-14
            )
            np.testing.assert_allclose(
                result.reaction.force_first_world, [[25 / 88, 0, 0]], atol=3e-14
            )
            np.testing.assert_allclose(result.reaction.power_first_w, [-5 / 88], atol=3e-14)
            assert np.linalg.norm(result.position_correction) > 1e-8
            assert np.linalg.norm(result.velocity_projection_impulse) > 1e-9
        max_it = max(max_it, result.report.iterations)
        s = result.state
        np.testing.assert_allclose(0.5 * np.cos(s.q[0]), s.q[1], rtol=0, atol=1e-10)
        np.testing.assert_allclose(-0.5 * np.sin(s.q[0]) * s.v[0], s.v[1], rtol=0, atol=1e-10)
    assert max_it > 0


def test_world_anchor_three_independent_rows_and_reversed_endpoint_forces():
    m = RobotModel(
        np.array([-1, 0, 1]),
        ("prismatic",) * 3,
        np.zeros((3, 3)),
        np.tile([0.0, 0, 0, 1], (3, 1)),
        np.eye(3),
        np.ones(3),
        np.zeros((3, 3)),
        np.ones((3, 3)),
        np.zeros(3),
    )
    s = RobotState(np.array([0.125, 0.25, -0.5]), np.zeros(3))
    f = np.array([2.0, -3, 4])
    inputs = RobotInputs(np.zeros(3), np.array([np.zeros(3), np.zeros(3), f]), np.zeros((3, 3)))
    material, world = RobotSite(2), RobotSite(None, tuple(s.q))
    for first, second, sign in ((material, world, -1), (world, material, 1)):
        # Two separate constraints jointly pin xyz.
        constraints = [
            RobotPointClosure(first, second, np.eye(3)[:2]),
            RobotPointClosure(first, second, np.eye(3)[2:]),
        ]
        result = step_robot_closures(m, s, inputs, constraints, 1 / 32)
        assert result.report.accepted, result.report.reason
        np.testing.assert_allclose(
            np.sum(result.reaction.force_first_world, axis=0), sign * f, atol=1e-13
        )
        np.testing.assert_allclose(result.state.q, s.q, atol=1e-14)
        np.testing.assert_allclose(result.state.v, 0, atol=1e-14)


def test_dead_center_dependent_rows_and_projection_failure_preserve_inputs_for_retry():
    m, s, inputs, c = four_bar(case())
    for bad, closures, reason in (
        (RobotState(np.zeros(3), np.zeros(3)), [c], "rank"),
        (s, [c, c], "rank"),
        (RobotState(s.q, np.array([0.5, 0, 0])), [c], "velocity"),
        (RobotState(s.q + np.array([0, 0.1, 0]), s.v), [c], "position"),
    ):
        q, v = bad.q.copy(), bad.v.copy()
        result = step_robot_closures(m, bad, inputs, closures, 1 / 64)
        assert not result.report.accepted and reason in result.report.reason
        assert result.reaction is None
        np.testing.assert_array_equal(result.state.q, q)
        np.testing.assert_array_equal(result.state.v, v)
    m, s, inputs, c = nonlinear()
    rejected = step_robot_closures(m, s, inputs, [c], 1 / 32, max_iterations=0)
    assert not rejected.report.accepted and rejected.reaction is None
    retry = step_robot_closures(m, rejected.state, inputs, [c], 1 / 32)
    clean = step_robot_closures(m, s, inputs, [c], 1 / 32)
    np.testing.assert_array_equal(retry.state.q, clean.state.q)
    np.testing.assert_array_equal(retry.state.v, clean.state.v)
    with pytest.raises(ValueError):
        replace(c, axes_world=[[1, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError):
        step_robot_closures(m, s, inputs, [], 1 / 32)


def test_spatial_hinge_slide_chain_matches_independent_point_velocity_derivative():
    model = RobotModel(
        np.array([-1, 0, 1]),
        ("revolute", "prismatic", "revolute"),
        np.array([[0.25, 0.1, 0.125], [0.3, -0.125, 0.15], [0.2, 0.1, -0.1]]),
        np.array([[0.0, 0, 0, 1], [0.5, 0.5, 0.5, 0.5], [0.6, 0.8, 0, 0]]),
        np.array([[0.6, 0.8, 0], [0.0, -0.8, 0.6], [0.0, 0, 1]]),
        np.array([1.0, 0.625, 0.75]),
        np.array([[0.125, 0, 0], [0, 0.15, 0.1], [0.2, -0.1, 0.05]]),
        np.array([[0.125, 0.16, 0.2], [0.18, 0.13, 0.16], [0.1, 0.12, 0.14]]),
        np.array([0.25, -0.5, -1.0]),
    )
    q = np.array([0.31, 0.2, -0.47])
    local = np.array([[0.2, -0.4, 0.125]])
    normal = np.array([[-0.6, 0, 0.8]])

    def row(configuration):
        return (normal @ point_jacobians(model, configuration, [2], local)[0][0])[0]

    jac = row(q)
    v = np.array([0.25, 0.5, 0.0])
    v[2] = -(jac @ v) / jac[2]
    geo = forward_kinematics(model, q)
    site = RobotSite(2, tuple(local[0]))
    fixed = RobotSite(None, tuple(geo.position[2] + geo.rotation[2] @ local[0]))
    inputs = RobotInputs(np.array([0.3, -0.25, 0.125]), np.zeros((3, 3)), np.zeros((3, 3)))
    result = step_robot_closures(
        model, RobotState(q, v), inputs, [RobotPointClosure(site, fixed, normal)], 1 / 512
    )
    assert result.report.accepted, result.report.reason
    eps = 1e-5
    bias = (row(q + eps * v) @ v - row(q - eps * v) @ v) / (2 * eps)
    assert abs(bias) > 1e-3
    np.testing.assert_allclose(jac @ result.reaction.acceleration, -bias, rtol=0, atol=2e-9)
    np.testing.assert_allclose(
        result.reaction.generalized_force, jac * result.reaction.multipliers[0], atol=2e-13
    )
