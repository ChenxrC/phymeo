"""Site IMU samples from actual fixed/floating robot dynamics at a declared instant."""

import json

import numpy as np
import pytest

from phymeo.core.rigid import RigidBodyModel
from phymeo.core.robot import RobotInputs, RobotModel, RobotState
from phymeo.dynamics.floating_robot import (
    FloatingRobotModel,
    FloatingRobotState,
    floating_kinematics,
)
from phymeo.dynamics.robot_dynamics import forward_dynamics
from phymeo.dynamics.robot_kinematics import forward_kinematics, point_jacobians
from phymeo.dynamics.rotation import quaternion_matrix
from phymeo.sensors import RobotSite
from phymeo.sensors.robot_imu import measure_robot_imu
from phymeo.solvers.floating_tree import (
    FloatingRobotInputs,
    floating_dynamics_terms,
    step_floating_robot,
)
from phymeo.solvers.robot_closure import RobotPointClosure, step_robot_closures


def rail(gravity):
    return RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.zeros((1, 3)),
        np.array([[0.0, 0, 0, 1]]),
        np.array([[0.0, 0, 1]]),
        np.array([2.0]),
        np.zeros((1, 3)),
        np.full((1, 3), 0.5),
        np.array(gravity, dtype=float),
    )


def free_body():
    root = RigidBodyModel(np.array([0.5]), np.array([[0.25, 0.375, 0.5]]), np.array([0.0, 0, -4.0]))
    return FloatingRobotModel(root, np.array([0.125, 0, 0]))


def test_static_supported_sensor_reads_the_same_accepted_closure_force_and_time():
    model = rail((0, 0, -4))
    state = RobotState(np.array([0.125]), np.zeros(1))
    root, anchor = RobotSite(0), RobotSite(None, (0, 0, 0.125))
    c = RobotPointClosure(root, anchor, [[0.0, 0, 1]])
    site = RobotSite(0, (0.25, -0.125, 0), (np.sqrt(0.5), 0, 0, np.sqrt(0.5)))
    h = 1 / 64
    for k in range(4):
        step = step_robot_closures(model, state, RobotInputs.zeros(1), [c], h)
        assert step.report.accepted
        np.testing.assert_array_equal(step.state.q, state.q)
        np.testing.assert_allclose(step.reaction.force_first_world, [[0, 0, 8]], atol=1e-13)
        reading = measure_robot_imu(model, state, step.reaction.acceleration, site, time_s=k * h)
        assert reading.time_s == k * h
        np.testing.assert_allclose(reading.specific_force_m_s2, [0, 4, 0], rtol=0, atol=2e-14)
        np.testing.assert_array_equal(reading.angular_velocity_rad_s, np.zeros(3))
        np.testing.assert_allclose(
            reading.specific_force_m_s2[1],
            step.reaction.force_first_world[0, 2] / 2,
            rtol=0,
            atol=2e-14,
        )
        state = step.state
    print(
        json.dumps(
            dict(
                id="supported",
                time_s=reading.time_s,
                force_n=step.reaction.force_first_world[0].tolist(),
                accelerometer=reading.specific_force_m_s2.tolist(),
            )
        )
    )


@pytest.mark.parametrize("loaded", [False, True], ids=["free-fall", "eccentric-driven"])
def test_floating_root_fall_and_driven_spin_follow_real_eight_step_trajectory(loaded):
    m = free_body()
    omega = 2.0 if loaded else 0.0
    state = FloatingRobotState(
        np.array([0.0, 0, 0, 0, 0, 0, 1]), np.array([0.25, -0.5, 0, 0, 0, omega])
    )
    site = RobotSite(0, (0.375, 0, 0), (0, 0, np.sqrt(0.5), np.sqrt(0.5)))
    loads = FloatingRobotInputs(
        np.empty(0),
        np.array([[2.0 if loaded else 0.0, 0, 0]]),
        np.array([[0.0, 0, 1.5 if loaded else 0.0]]),
    )
    h = 1 / 64
    angle = 0.0
    alpha = 3.0 if loaded else 0.0
    specific_x = 1.0 if loaded else 0.0
    for k in range(9):
        a = floating_dynamics_terms(m, state, loads).acceleration
        reading = measure_robot_imu(m, state, a, site, time_s=k * h)
        want = [
            alpha * 0.25 - specific_x * np.sin(angle),
            0.25 * omega**2 - specific_x * np.cos(angle),
            0,
        ]
        np.testing.assert_allclose(reading.specific_force_m_s2, want, rtol=0, atol=8e-14)
        np.testing.assert_allclose(
            reading.angular_velocity_rad_s, [0, 0, omega], rtol=0, atol=8e-14
        )
        np.testing.assert_allclose(state.v[2], -4 * k * h, rtol=0, atol=2e-15)
        if k < 8:
            result = step_floating_robot(m, state, loads, h)
            assert result.report.accepted
            omega += alpha * h
            angle += omega * h
            state = result.state
    print(
        json.dumps(
            dict(
                id="driven" if loaded else "free-fall",
                time_s=reading.time_s,
                accelerometer=reading.specific_force_m_s2.tolist(),
                gyro=reading.angular_velocity_rad_s.tolist(),
            )
        )
    )


@pytest.mark.parametrize(("speed", "alpha"), [(2.0, 0.0), (-2.0, 0.0), (2.0, 3.0), (2.0, -3.0)])
def test_fixed_revolute_exact_radial_and_tangential_specific_force(speed, alpha):
    m = RobotModel(
        np.array([-1]),
        ("revolute",),
        np.zeros((1, 3)),
        np.array([[0.0, 0, 0, 1]]),
        np.array([[0.0, 0, 1]]),
        np.array([2.0]),
        np.array([[0.125, 0, 0]]),
        np.full((1, 3), 0.25),
        np.zeros(3),
    )
    s = RobotState(np.array([0.73]), np.array([speed]))
    # World z torque about the hinge, effective moment = 1/4 + 2*(1/8)^2.
    inputs = RobotInputs(np.array([alpha * 9 / 32]), np.zeros((1, 3)), np.zeros((1, 3)))
    a = forward_dynamics(m, s, inputs)
    imu = RobotSite(0, (0.5, 0, 0), (0, 0, np.sqrt(0.5), np.sqrt(0.5)))
    obs = measure_robot_imu(m, s, a, imu, time_s=0.25)
    np.testing.assert_allclose(
        obs.specific_force_m_s2, [0.5 * alpha, 0.5 * speed**2, 0], rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(obs.angular_velocity_rad_s, [0, 0, speed], rtol=0, atol=1e-14)


def coupled(floating):
    g = np.array([0.25, -1.0, -4.0])
    children = RobotModel(
        np.array([-1, 0, 1]),
        ("revolute", "prismatic", "revolute"),
        np.array([[0.25, -0.125, 0], [0.25, 0.0625, 0.125], [0.125, 0.125, 0.25]]),
        np.array([[0.5, 0.5, 0.5, 0.5], [0.0, 0, 0, 1], [0.6, 0.8, 0, 0]]),
        np.array([[0.6, 0.8, 0], [0.0, -0.8, 0.6], [0.0, 0, 1]]),
        np.array([1.0, 0.75, 1.25]),
        np.array([[0.125, 0, 0], [0, 0.125, -0.125], [0.25, 0.0625, 0.125]]),
        np.ones((3, 3)),
        g,
    )
    cq = np.array([0.375, -0.125, -0.47])
    cv = np.array([0.25, -0.5, 0.375])
    ca = np.array([-0.25, 0.75, 0.5])
    if not floating:
        return children, RobotState(cq, cv), ca
    m = FloatingRobotModel(
        RigidBodyModel(np.array([0.5]), np.array([[0.25, 0.375, 0.5]]), g),
        np.array([0.125, -0.25, 0.0625]),
        children,
    )
    return (
        m,
        FloatingRobotState(
            np.r_[0.25, -0.5, 0.125, 0.5, 0.5, 0.5, 0.5, cq],
            np.r_[0.25, 0.125, -0.5, 0.75, -0.375, 1.0, cv],
        ),
        np.r_[0.5, -0.125, 0.25, -0.4, 0.875, 0.25, ca],
    )


@pytest.mark.parametrize("floating", [False, True], ids=["fixed-chain", "moving-root-chain"])
def test_oblique_nested_chain_imu_against_independent_point_velocity_derivative(floating):
    m, s, a = coupled(floating)
    site = RobotSite(3 if floating else 2, (0.125, -0.25, 0.375), (0.5, -0.5, 0.5, 0.5))
    p = np.array(site.local_position_m)

    def at(t):
        if not floating:
            q = s.q + t * s.v
            geom = forward_kinematics(m, q)
            linear, angular = point_jacobians(m, q, [2], p[None])
            return linear[0] @ (s.v + t * a), angular[0] @ (s.v + t * a), geom.rotation[2]
        r0 = quaternion_matrix(s.q[None, 3:7])[0]
        angle = t * np.linalg.norm(s.v[3:6])
        axis = s.v[3:6] / np.linalg.norm(s.v[3:6])
        xyz = s.q[3:6]
        w = s.q[6]
        dv = axis * np.sin(angle / 2)
        dw = np.cos(angle / 2)
        quat = np.r_[dw * xyz + w * dv + np.cross(dv, xyz), dw * w - dv @ xyz]
        r = quaternion_matrix(quat[None])[0]
        origin = s.q[:3] + r0 @ m.root_com_m + t * s.v[:3] - r @ m.root_com_m
        sim = FloatingRobotState(np.r_[origin, quat, s.q[7:] + t * s.v[6:]], s.v + t * a)
        geom = floating_kinematics(m, sim)
        return (
            geom.origin_velocity[3] + np.cross(geom.angular_velocity[3], geom.rotation[3] @ p),
            geom.angular_velocity[3],
            geom.rotation[3],
        )

    _, omega, r = at(0.0)
    h = 1e-5
    acc = (at(h)[0] - at(-h)[0]) / (2 * h)
    gravity = m.root.gravity if floating else m.gravity
    site_to_world = r @ quaternion_matrix(np.array([site.local_orientation_xyzw]))[0]
    actual = measure_robot_imu(m, s, a, site, time_s=0.375)
    np.testing.assert_allclose(
        actual.specific_force_m_s2, site_to_world.T @ (acc - gravity), rtol=0, atol=4e-10
    )
    np.testing.assert_allclose(
        actual.angular_velocity_rad_s, site_to_world.T @ omega, rtol=0, atol=2e-14
    )


def test_world_site_ignores_body_motion_and_rejects_missing_acceleration_or_invalid_time():
    m, s, a = coupled(True)
    world = RobotSite(None, (1.0, 2.0, 3.0), (0.5, 0.5, 0.5, 0.5))
    actual = measure_robot_imu(m, s, a, world, time_s=4.0)
    np.testing.assert_allclose(actual.specific_force_m_s2, [1.0, 4.0, -0.25], rtol=0, atol=2e-14)
    np.testing.assert_array_equal(actual.angular_velocity_rad_s, np.zeros(3))
    for invalid in (np.zeros(3), np.full(len(s.v), np.nan)):
        with pytest.raises(ValueError):
            measure_robot_imu(m, s, invalid, world, time_s=0.0)
    with pytest.raises(TypeError):
        measure_robot_imu(m, s, a.astype(np.float32), world, time_s=0.0)
    with pytest.raises(ValueError):
        measure_robot_imu(m, s, a, RobotSite(4), time_s=0.0)
    with pytest.raises(ValueError):
        measure_robot_imu(m, s, a, world, time_s=np.inf)
    with pytest.raises(TypeError):
        measure_robot_imu(m, RobotState(np.zeros(3), np.zeros(3)), a, world, time_s=0.0)
