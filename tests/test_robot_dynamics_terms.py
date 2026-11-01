"""Batched robot terms must represent the same tree mechanics as independent virtual work."""

import numpy as np

from phymeo.core.robot import RobotModel, RobotState
from phymeo.dynamics import robot_dynamics as dynamics
from phymeo.dynamics.robot_kinematics import forward_kinematics


def test_moving_branched_robot_terms_against_world_motion():
    model = RobotModel(
        np.array([-1, 0, 0, 1]),
        ("revolute", "prismatic", "revolute", "revolute"),
        np.array([[0.2, -0.1, 0.3], [0.13, 0.2, 0.0], [0.0, -0.2, 0.3], [0.1, 0.0, -0.2]]),
        np.tile([0.0, 0.0, 0.0, 1.0], (4, 1)),
        np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([0.6, 1.3, 0.27, 0.45]),
        np.array([[0.1, 0.0, 0.0], [0.0, 0.05, -0.1], [0.17, -0.1, 0.04], [0.1, -0.2, 0.05]]),
        np.tile([0.13, 0.17, 0.19], (4, 1)),
        np.array([0.4, -0.8, -9.81]),
    )
    q = np.array([0.51, 0.07, -0.32, 0.49])
    v = np.array([-0.24, 0.08, 0.15, 0.44])
    geo, linear, angular, mass, bias = dynamics.robot_dynamics_terms(model, RobotState(q, v))
    assert np.array_equal(geo.com_position, forward_kinematics(model, q).com_position)
    # Independent finite-difference world COM Jacobian and kinetic energy Hessian.
    ds = 1e-6
    for j, unit in enumerate(np.eye(4)):
        plus = forward_kinematics(model, q + ds * unit)
        minus = forward_kinematics(model, q - ds * unit)
        np.testing.assert_allclose(
            linear[:, :, j], (plus.com_position - minus.com_position) / (2 * ds), atol=2e-10, rtol=0
        )
        s = (plus.rotation - minus.rotation) @ geo.rotation.transpose(0, 2, 1) / (2 * ds)
        np.testing.assert_allclose(angular[:, :, j], s[:, [2, 0, 1], [1, 2, 0]], atol=2e-10, rtol=0)

    def kinetic(configuration):
        g = forward_kinematics(model, configuration)
        jp = []
        jw = []
        for unit in np.eye(4):
            plus = forward_kinematics(model, configuration + ds * unit)
            minus = forward_kinematics(model, configuration - ds * unit)
            jp.append((plus.com_position - minus.com_position) / (2 * ds))
            skew = (plus.rotation - minus.rotation) @ g.rotation.transpose(0, 2, 1) / (2 * ds)
            jw.append(skew[:, [2, 0, 1], [1, 2, 0]])
        jp = np.moveaxis(jp, 0, -1)
        jw = np.moveaxis(jw, 0, -1)
        body_j = g.rotation.transpose(0, 2, 1) @ jw
        return np.einsum("b,bij,bik->jk", model.mass, jp, jp) + np.einsum(
            "bi,bij,bik->jk", model.inertia_diagonal, body_j, body_j
        )

    np.testing.assert_allclose(mass, kinetic(q), atol=2e-10, rtol=0)
    step = 2e-4
    mdot = (kinetic(q + step * v) - kinetic(q - step * v)) / (2 * step)
    grad = np.array(
        [v @ (kinetic(q + step * u) - kinetic(q - step * u)) @ v / (4 * step) for u in np.eye(4)]
    )
    gravity = np.einsum("b,bkj,k->j", model.mass, linear, model.gravity)
    np.testing.assert_allclose(bias, mdot @ v - grad - gravity, atol=5e-7, rtol=0)
