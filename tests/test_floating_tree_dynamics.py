"""Analytic free-tree loads, motion and independent bias/work checks at multiple poses."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from phymeo.core.capabilities import SolverFailure
from phymeo.core.rigid import RigidBodyInputs, RigidBodyModel
from phymeo.core.robot import RobotModel
from phymeo.dynamics.floating_robot import (
    FloatingRobotModel,
    FloatingRobotState,
    floating_kinematics,
)
from phymeo.dynamics.robot_dynamics import _mass_from_terms
from phymeo.solvers.floating_tree import (
    FloatingRobotInputs,
    floating_dynamics_terms,
    step_floating_robot,
)
from phymeo.solvers.free_root import step_free_root

CATALOG = json.loads(
    (Path(__file__).parent / "reference_newton/robot/floating_tree_cases.json").read_text()
)


def system(row):
    fixture = CATALOG["models"][row["model"]]
    g = np.array(row["gravity"], dtype=float)
    root = RigidBodyModel(np.array([0.5]), np.full((1, 3), 0.5), g)
    children = RobotModel(
        np.array([-1]),
        (fixture["kind"],),
        np.array([fixture["anchor"]], dtype=float),
        np.array([[0.0, 0, 0, 1]]),
        np.array([fixture["axis"]], dtype=float),
        np.ones(1),
        np.array([fixture["com"]], dtype=float),
        np.full((1, 3), 0.25),
        g,
    )
    model = FloatingRobotModel(root, np.asarray(fixture["root_com"], dtype=float), children)
    state = FloatingRobotState(np.asarray(row["q"], dtype=float), np.asarray(row["v"], dtype=float))
    inputs = FloatingRobotInputs(
        np.array([float(row["effort"])]),
        np.asarray(row["force"], dtype=float),
        np.asarray(row["torque"], dtype=float),
    )
    return model, state, inputs


def totals(model, s):
    obs = floating_kinematics(model, s)
    masses = np.array([2, 1])[:, None]
    # The catalog uses spherical inertias. Compute laboratory momenta/energy
    # directly from measured body COM velocities, spin and position.
    spins = np.array([0.5, 0.25])[:, None] * obs.angular_velocity
    momenta = masses * obs.com_velocity
    angular = spins + np.cross(obs.com_position, momenta)
    kinetic = (np.sum(momenta * obs.com_velocity) + np.sum(spins * obs.angular_velocity)) / 2
    return momenta.sum(axis=0), angular.sum(axis=0), kinetic


@pytest.mark.parametrize("row", CATALOG["cases"], ids=lambda row: row["id"])
def test_coupled_public_end_states_energy_and_momentum(row):
    model, initial, inputs = system(row)
    s, h = initial, row["h"]
    a = np.asarray(row["acceleration"])
    terms = floating_dynamics_terms(model, s, inputs)
    np.testing.assert_allclose(terms.acceleration, a, rtol=0, atol=2e-13)
    np.testing.assert_allclose(
        terms.mass_matrix @ a + terms.bias, terms.applied, rtol=0, atol=2e-13
    )
    if row["model"] == "rail":
        np.testing.assert_array_equal(terms.mass_matrix[np.ix_([0, 6], [0, 6])], [[3, 1], [1, 1]])
    p0, h0, e0 = totals(model, s)
    for k in range(1, row["steps"] + 1):
        result = step_floating_robot(model, s, inputs, h)
        assert result.report.accepted, result.report.reason
        s = result.state
        expected_v = initial.v + k * h * a
        np.testing.assert_allclose(s.v, expected_v, rtol=0, atol=2e-13)
        np.testing.assert_allclose(
            s.q[7],
            initial.q[7] + k * h * initial.v[6] + a[6] * h**2 * k * (k + 1) / 2,
            rtol=0,
            atol=2e-14,
        )
        p, angular, energy = totals(model, s)
        if row["steps"] > 1:
            root_v = expected_v[:3]
            child_v = root_v + np.array([expected_v[6], 0, 0])
            expected_energy = root_v @ root_v + (child_v @ child_v) / 2
            np.testing.assert_allclose(energy, expected_energy, atol=2e-14)
            external = inputs.force.sum(axis=0) + 3 * model.root.gravity
            np.testing.assert_allclose(p, p0 + k * h * external, atol=2e-14)
            np.testing.assert_allclose(
                floating_kinematics(model, s).com_velocity, [root_v, child_v], atol=2e-14
            )
            np.testing.assert_allclose(
                s.q[:3],
                initial.q[:3] + k * h * initial.v[:3] + a[:3] * h**2 * k * (k + 1) / 2,
                rtol=0,
                atol=2e-14,
            )
        elif row["id"] == "oblique-hinge":
            # At rest alpha_root=-5u/2, alpha_child=3u. Effective child inertia
            # about the system COM is I1+mu*r²=5/12. Relative acceleration is 11/2.
            axis, radial, tangent = (
                np.array([0, 0.6, 0.8]),
                np.array([1, 0, 0]),
                np.array([0, 0.8, -0.6]),
            )
            theta = 3 * h**2
            root_center = np.array([0.28125, -0.375, 0.0625]) + h**2 * a[:3]
            child_center = root_center + 0.5 * (radial * np.cos(theta) + tangent * np.sin(theta))
            child_v = h * a[:3] + 1.5 * h * (tangent * np.cos(theta) - radial * np.sin(theta))
            obs = floating_kinematics(model, s)
            np.testing.assert_allclose(
                obs.com_position, [root_center, child_center], rtol=0, atol=2e-14
            )
            np.testing.assert_allclose(obs.com_velocity[1], child_v, rtol=0, atol=2e-14)
            np.testing.assert_allclose(
                obs.angular_velocity, [-2.5 * h * axis, 3 * h * axis], rtol=0, atol=2e-14
            )
            root_radial = radial * np.cos(-2.5 * h**2) + tangent * np.sin(-2.5 * h**2)
            np.testing.assert_allclose(obs.rotation[0, :, 2], root_radial, rtol=0, atol=2e-14)
            np.testing.assert_allclose(p, 0, atol=3e-7)
            np.testing.assert_allclose(angular, 0, atol=3e-7)
            np.testing.assert_allclose(energy, 0.5 * inputs.effort[0] * a[6] * h**2, atol=2e-12)
        else:
            # Independently reconstruct both end twists for a rotating rail.
            w = expected_v[5]
            angle = h * w
            axis = np.array([np.cos(angle), np.sin(angle), 0])
            tangent = np.array([-np.sin(angle), np.cos(angle), 0])
            extension = initial.q[7] + h * expected_v[6]
            root_center = initial.q[:3] + h * expected_v[:3]
            child_center = root_center + extension * axis
            child_v = expected_v[:3] + expected_v[6] * axis + w * extension * tangent
            expected_p = 2 * expected_v[:3] + child_v
            expected_h = (
                np.cross(root_center, 2 * expected_v[:3])
                + np.cross(child_center, child_v)
                + [0, 0, 0.75 * w]
            )
            expected_e = expected_v[:3] @ expected_v[:3] + 0.5 * (child_v @ child_v + 0.75 * w * w)
            np.testing.assert_allclose(p, expected_p, rtol=0, atol=2e-14)
            np.testing.assert_allclose(angular, expected_h, rtol=0, atol=2e-14)
            np.testing.assert_allclose(energy, expected_e, rtol=0, atol=2e-14)

    print(
        json.dumps(
            dict(
                id=row["id"],
                accepted_steps=k,
                acceleration=terms.acceleration.tolist(),
                q=s.q.tolist(),
                v=s.v.tolist(),
                start_P=p0.tolist(),
                end_P=p.tolist(),
                start_H=h0.tolist(),
                end_H=angular.tolist(),
                start_T=e0,
                end_T=energy,
            ),
            allow_nan=False,
        )
    )


def test_external_couple_at_child_com_reacts_on_free_root():
    m, s, _ = system(CATALOG["cases"][0])
    inputs = FloatingRobotInputs(
        np.zeros(1), np.zeros((2, 3)), np.array([[0.0, 0, 0], [0, 0, 0.75]])
    )
    result = floating_dynamics_terms(m, s, inputs)
    np.testing.assert_allclose(result.acceleration, [0, 0, 0, 0, 0, 1, 0], atol=2e-14)
    np.testing.assert_array_equal(result.applied, [0, 0, 0, 0, 0, 0.75, 0])


def deep_tree():
    gravity = np.array([0.5, -2, -3])
    root = RigidBodyModel(np.array([0.5]), np.array([[0.25, 0.375, 0.5]]), gravity)
    c = RobotModel(
        np.array([-1, 0, -1]),
        ("revolute", "prismatic", "revolute"),
        np.array([[0.25, 0, 0.125], [0.375, 0.0625, 0], [0.125, -0.25, 0.125]]),
        np.array([[0.0, 0, 0, 1], [0.5, 0.5, 0.5, 0.5], [0.6, 0.8, 0, 0]]),
        np.array([[0.6, 0.8, 0], [0, -0.8, 0.6], [0, 0, 1]]),
        np.array([1.0, 0.75, 1.25]),
        np.array([[0.25, 0, 0.125], [0.0625, 0.125, -0.25], [-0.125, 0.25, 0.125]]),
        np.array([[0.25, 0.375, 0.5], [0.5, 0.25, 0.375], [0.1, 0.14, 0.18]]),
        gravity,
    )
    m = FloatingRobotModel(root, np.array([0.125, -0.0625, 0.25]), c)
    s = FloatingRobotState(
        np.array([0.125, -0.25, 0.375, 0.5, 0.5, 0.5, 0.5, -0.37, 0.25, 0.19]),
        np.array([0.25, -0.5, 0.125, 0.75, -0.5, 1.0, 0.25, -0.375, 0.125]),
    )
    return m, s


def test_oblique_moving_branched_bias_against_measured_com_and_angular_accelerations():
    model, s = deep_tree()
    obs = floating_kinematics(model, s)
    epsilon = 1e-5
    pair = []
    axis = s.v[3:6] / np.linalg.norm(s.v[3:6])
    for dt in (-epsilon, epsilon):
        angle = dt * np.linalg.norm(s.v[3:6])
        dv = axis * np.sin(angle / 2)
        dw = np.cos(angle / 2)
        xyz, w = s.q[3:6], s.q[6]
        quat = np.r_[dw * xyz + w * dv + np.cross(dv, xyz), dw * w - dv @ xyz]
        offset = obs.rotation[0] @ model.root_com_m
        rotated = (
            offset * np.cos(angle)
            + np.cross(axis, offset) * np.sin(angle)
            + axis * (axis @ offset) * (1 - np.cos(angle))
        )
        origin = obs.com_position[0] + dt * s.v[:3] - rotated
        curve = FloatingRobotState(np.r_[origin, quat, s.q[7:] + dt * s.v[6:]], s.v)
        pair.append(floating_kinematics(model, curve))
    da = (pair[1].com_velocity - pair[0].com_velocity) / (2 * epsilon)
    dw = (pair[1].angular_velocity - pair[0].angular_velocity) / (2 * epsilon)
    masses = np.r_[2.0, model.children.mass]
    diag = np.vstack([model.root.inertia_diagonal, model.children.inertia_diagonal])
    inertia = np.array([r @ np.diag(i) @ r.T for r, i in zip(obs.rotation, diag, strict=True)])
    forces = masses[:, None] * (da - model.root.gravity)
    torques = np.array(
        [
            i @ a + np.cross(w, i @ w)
            for i, a, w in zip(inertia, dw, obs.angular_velocity, strict=True)
        ]
    )
    inputs = FloatingRobotInputs(
        np.array([-0.25, 0.5, 0.125]),
        np.arange(12, dtype=float).reshape(4, 3) / 8,
        np.arange(-6, 6, dtype=float).reshape(4, 3) / 16,
    )
    terms = floating_dynamics_terms(model, s, inputs)
    for j, velocity in enumerate(np.eye(9)):
        response = floating_kinematics(model, FloatingRobotState(s.q, velocity))
        bias = np.sum(response.com_velocity * forces) + np.sum(response.angular_velocity * torques)
        applied = np.sum(response.com_velocity * inputs.force) + np.sum(
            response.angular_velocity * inputs.torque
        )
        np.testing.assert_allclose(terms.bias[j], bias, rtol=0, atol=8e-10)
        np.testing.assert_allclose(
            terms.applied[j], applied + velocity[6:] @ inputs.effort, atol=2e-14
        )
    kinetic = (
        np.sum(masses[:, None] * obs.com_velocity**2)
        + sum(w @ i @ w for w, i in zip(obs.angular_velocity, inertia, strict=True))
    ) / 2
    np.testing.assert_allclose(s.v @ terms.mass_matrix @ s.v / 2, kinetic, atol=2e-14)
    result = step_floating_robot(model, s, inputs, 1 / 256)
    assert result.report.accepted
    np.testing.assert_allclose(result.state.v, s.v + terms.acceleration / 256, atol=2e-14)


def test_zero_children_routes_to_single_rigid_solver_and_invalid_domains_are_named():
    m, s = deep_tree()
    m = replace(m, children=None)
    s = FloatingRobotState(s.q[:7], s.v[:6])
    inputs = FloatingRobotInputs(
        np.empty(0), np.array([[0.25, -0.375, 0.5]]), np.array([[1.0, -2, 0.5]])
    )
    rigid = step_free_root(m, s, RigidBodyInputs(inputs.force, inputs.torque), 1 / 64)
    tree = step_floating_robot(m, s, inputs, 1 / 64)
    np.testing.assert_array_equal(tree.state.q, rigid.state.q)
    np.testing.assert_array_equal(tree.state.v, rigid.state.v)
    np.testing.assert_allclose(
        s.v + floating_dynamics_terms(m, s, inputs).acceleration / 64, tree.state.v, atol=2e-14
    )
    m, s = deep_tree()
    with pytest.raises(ValueError, match="gravity"):
        step_floating_robot(
            replace(m, children=replace(m.children, gravity=np.zeros(3))),
            s,
            FloatingRobotInputs.zeros(3),
            1 / 64,
        )
    with pytest.raises(ValueError):
        step_floating_robot(m, s, FloatingRobotInputs.zeros(1), 1 / 64)
    with pytest.raises(ValueError):
        step_floating_robot(m, s, FloatingRobotInputs.zeros(3), 0)
    with pytest.raises((ValueError, TypeError)):
        FloatingRobotInputs.zeros(True)
    with pytest.raises(ValueError):
        FloatingRobotInputs(np.zeros(1), np.zeros((1, 3)), np.zeros((1, 3)))


def test_numerical_rejection_preserves_state_and_retries_without_phantom_drive():
    m, s, inputs = system(CATALOG["cases"][0])
    rejected = step_floating_robot(
        m, s, FloatingRobotInputs(np.array([1e308]), np.zeros((2, 3)), np.zeros((2, 3))), 16.0
    )
    assert not rejected.report.accepted and rejected.report.reason
    np.testing.assert_array_equal(s.q, rejected.state.q)
    np.testing.assert_array_equal(s.v, rejected.state.v)
    clean = step_floating_robot(m, s, inputs, 1 / 32)
    retry = step_floating_robot(m, rejected.state, inputs, 1 / 32)
    np.testing.assert_array_equal(clean.state.q, retry.state.q)
    np.testing.assert_array_equal(clean.state.v, retry.state.v)


def test_isotropic_free_hinge_in_an_arbitrary_world_attitude():
    m, s, _ = system(CATALOG["cases"][0])
    child = replace(
        m.children,
        joint_types=("revolute",),
        joint_axis=np.array([[0.6, 0.8, 0]]),
        inertia_diagonal=np.full((1, 3), 0.7),
    )
    m = replace(m, root=replace(m.root, inertia_diagonal=np.full((1, 3), 0.3)), children=child)
    quat = np.array([1.0, 2, 3, 4]) / np.sqrt(30)
    s = FloatingRobotState(np.r_[s.q[:3], quat, 0.0], s.v)
    terms = floating_dynamics_terms(
        m, s, FloatingRobotInputs(np.ones(1), np.zeros((2, 3)), np.zeros((2, 3)))
    )
    np.testing.assert_allclose(terms.mass_matrix[3:6, 3:6], np.eye(3), rtol=0, atol=3e-15)
    # Rotate the hinge axis using the independent quaternion vector formula.
    axis = child.joint_axis[0]
    world_axis = axis + 2 * np.cross(quat[:3], np.cross(quat[:3], axis) + quat[3] * axis)
    np.testing.assert_allclose(terms.acceleration[3:6], -world_axis / 0.3, rtol=0, atol=1e-14)
    np.testing.assert_allclose(terms.acceleration[6], 1 / 0.3 + 1 / 0.7, rtol=0, atol=1e-14)


def test_mass_symmetry_gate_rejects_an_asymmetric_physical_scale_defect():
    # Artificial invalid inertia. Its 1e-8 kg m² cross-entry is many orders
    # above the O(eps) roundoff of unit-sized angular diagonals.
    inertia = np.eye(3)[None]
    inertia[0, 0, 1] = 1e-8
    with pytest.raises(SolverFailure, match="symmetr"):
        _mass_from_terms(np.ones(1), np.zeros((1, 3, 3)), np.eye(3)[None], inertia)
