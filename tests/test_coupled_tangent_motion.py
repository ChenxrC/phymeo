"""Actual native motion after a coupled tangent solve, with explicitly prescribed normal capacity.

These are material-port mechanics tests, not an automatic contact-geometry adapter.
"""

import importlib

import numpy as np
import pytest

from phymeo.core.model import ParticleModel
from phymeo.core.rigid import RigidBodyInputs, RigidBodyModel, RigidBodyState
from phymeo.core.robot import RobotModel, RobotState
from phymeo.core.state import ParticleState, StepInputs
from phymeo.coupling.robot_port import RobotPointPort
from phymeo.dynamics.rotation import quaternion_matrix
from phymeo.solvers.rigid_semi_implicit import RigidSemiImplicitSolver
from phymeo.solvers.semi_implicit import SemiImplicitSolver


def solve(old, free, W, h, normal, stiffness, friction):
    module = importlib.import_module("phymeo.solvers.coupled_tangent")
    return module.solve_coupled_tangent(
        np.asarray(old, dtype=float), np.asarray(free, dtype=float), W, h,
        np.asarray(normal, dtype=float), stiffness_n_per_m=np.asarray(stiffness, dtype=float),
        friction=np.asarray(friction, dtype=float),
    )


@pytest.mark.parametrize("turned", [False, True])
def test_two_rigid_bodies_receive_opposite_multisite_forces_once(turned):
    # Two parallel material couplers between COM ports; all cross blocks survive.
    R = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]]) if turned else np.eye(3)
    T = R[:, :2]
    shift = np.array([4., -2, .5]) if turned else np.zeros(3)
    boost = R @ np.array([0., .25, 0])
    h, mass = .1, np.array([2., 3.])
    model = RigidBodyModel(1/mass, np.ones((2, 3)), np.zeros(3))
    initial = RigidBodyState(
        np.array([[-.5, 0, 0], [.5, 0, 0]]) @ R.T + shift,
        np.tile([0., 0, 0, 1], (2, 1)), np.tile(boost, (2, 1)), np.zeros((2, 3)),
    )
    external = np.array([[-3., 0, 0], [3, 0, 0]]) @ R.T
    solver = RigidSemiImplicitSolver()
    free = solver.step(model, initial, RigidBodyInputs(external, np.zeros((2, 3))), h)
    assert free.report.accepted
    relative_free = h * (free.state.velocity[1] - free.state.velocity[0]) @ T
    W = (1/2 + 1/3) * np.kron(np.ones((2, 2)), np.eye(2))
    tangent = solve(np.zeros((2, 2)), np.tile(relative_free, (2, 1)), W, h,
                    [4, 4], [120, 120], [.5, .5])
    np.testing.assert_allclose(tangent.force_n, [[-1, 0], [-1, 0]], atol=1e-11)
    on_second = tangent.force_n.sum(axis=0) @ T.T
    forces = external + np.stack((-on_second, on_second))
    result = solver.step(model, initial, RigidBodyInputs(forces, np.zeros((2, 3))), h)
    assert result.report.accepted
    after = result.state
    np.testing.assert_allclose(after.velocity - boost, np.array([[-.05, 0, 0], [1/30, 0, 0]]) @ R.T,
                               atol=1e-12)
    dx = after.position - initial.position
    actual = np.tile((dx[1] - dx[0]) @ T, (2, 1))
    receipt = tangent.verify_motion(actual, tolerance_m=2e-13)
    assert receipt.accepted
    np.testing.assert_allclose((mass[:, None]*after.velocity).sum(axis=0), mass.sum()*boost, atol=1e-12)
    np.testing.assert_array_equal(initial.velocity, np.tile(boost, (2, 1)))
    kinetic = .5*np.sum(mass[:, None]*(after.velocity**2-initial.velocity**2))
    drift_loss = .5*np.sum(mass[:, None]*(after.velocity-initial.velocity)**2)
    external_work = np.sum(external*dx)
    assert kinetic + drift_loss == pytest.approx(external_work + receipt.work_j.sum(), abs=2e-12)
    assert kinetic + tangent.stored_energy_j.sum() + drift_loss + (
        tangent.plastic_dissipation_j + tangent.implicit_dissipation_j
    ).sum() == pytest.approx(external_work, abs=2e-12)


def test_particle_64_support_steps_onset_slip_unload_and_reentry_have_energy_receipts():
    model = ParticleModel(np.ones(1), np.zeros(3))
    state = ParticleState(np.zeros((1, 3)), np.zeros((1, 3)))
    solver, h, old, previous_energy = SemiImplicitSolver(), 1/16, np.zeros((1, 2)), 0.0
    releases = []
    held_velocity = None
    for tick in range(75):
        applied = 1.0 if tick < 64 else 3.0 if tick < 66 else 0.0
        normal = 0.0 if 66 <= tick < 74 else 4.0
        force = np.array([[applied, 0., 0.]])
        free_v = state.velocity + h*force
        tangent = solve(old, h*free_v[:, :2], np.eye(2), h,
                        [normal], [256], [.5])
        total = force.copy()
        total[:, :2] += tangent.force_n
        result = solver.step(model, state, StepInputs(total), h)
        assert result.report.accepted
        after = result.state
        actual = after.position[:, :2] - state.position[:, :2]
        receipt = tangent.verify_motion(actual, tolerance_m=2e-13)
        assert receipt.accepted and abs(after.position[0, 0]) < .5
        kinetic = .5 * np.sum(after.velocity**2 - state.velocity**2)
        drift_loss = .5 * np.sum((after.velocity-state.velocity)**2)
        balance = kinetic + tangent.stored_energy_j.sum() - previous_energy + drift_loss + (
            tangent.plastic_dissipation_j + tangent.implicit_dissipation_j
            + tangent.released_energy_j
        ).sum() - np.sum(force*(after.position-state.position))
        assert abs(balance) < 2e-12
        if tick == 63:
            assert abs(after.velocity[0, 0]) < 1e-9
            assert after.position[0, 0] == pytest.approx(1/256, abs=1e-9)
            np.testing.assert_allclose(tangent.force_n, [[-1, 0]], atol=1e-8)
        if tick == 65:
            assert tangent.regime == ("slide",)
            np.testing.assert_allclose(tangent.force_n, [[-2, 0]], atol=1e-8)
            assert after.velocity[0, 0] == pytest.approx(1/8, abs=1e-8)
            held_velocity = after.velocity.copy()
        if 66 <= tick < 74:
            releases.append(float(tangent.released_energy_j[0]))
            np.testing.assert_array_equal(tangent.force_n, [[0, 0]])
            np.testing.assert_array_equal(tangent.elastic_displacement_m, [[0, 0]])
            np.testing.assert_array_equal(after.velocity, held_velocity)
        if tick == 74:
            assert tangent.regime == ("stick",)
            np.testing.assert_allclose(tangent.force_n, [[-1, 0]], atol=1e-8)
        state, old = after, tangent.elastic_displacement_m
        previous_energy = float(tangent.stored_energy_j.sum())
    assert releases[0] == pytest.approx(1/128, abs=1e-9)
    assert releases[1:] == [0.0]*7


def test_actual_robot_point_port_preserves_shared_ancestor_response():
    model = RobotModel(
        np.array([-1]), ("prismatic",), np.zeros((1, 3)), np.array([[0., 0, 0, 1]]),
        np.array([[1., 0, 0]]), np.ones(1), np.zeros((1, 3)), np.ones((1, 3)), np.zeros(3),
    )
    state = RobotState(np.zeros(1), np.zeros(1))
    indices, points = np.array([0, 0]), np.array([[0., .1, 0], [0, -.1, 0]])
    port = RobotPointPort(model, state, indices, points)
    columns = []
    for column in range(4):
        impulse = np.zeros((2, 3))
        impulse[column//2, column % 2] = 1
        columns.append(port.response(impulse)[:, :2].ravel())
    W = np.column_stack(columns)
    expected = np.zeros((4, 4))
    expected[np.ix_([0, 2], [0, 2])] = 1
    np.testing.assert_allclose(W, expected, atol=1e-13)
    h = .1
    free_state = port.apply_impulses(np.array([[.3, 0, 0], [0, 0, 0]]))
    free_port = RobotPointPort(model, free_state, indices, points)
    tangent = solve(np.zeros((2, 2)), h*free_port.velocities()[:, :2], W, h,
                    [4, 4], [100, 100], [.5, .5])
    impulse = np.column_stack((h*tangent.force_n, np.zeros(2)))
    candidate = free_port.apply_impulses(impulse)
    np.testing.assert_allclose(candidate.v, [.1], atol=1e-12)
    np.testing.assert_array_equal(candidate.q, state.q)
    np.testing.assert_array_equal(state.v, [0])
    # An instantaneous port impulse leaves q unchanged; no finite-drift claim here.


def test_true_exp_map_motion_is_not_mislabeled_as_linear_point_displacement():
    h = 1/8
    model = RigidBodyModel(np.ones(1), np.ones((1, 3)), np.zeros(3))
    state = RigidBodyState(np.zeros((1, 3)), np.array([[0., 0, 0, 1]]),
                           np.array([[1/8, -1/4, 0.]]), np.zeros((1, 3)))
    tangent = solve([[0, 0]], [[1/64, -1/32]], np.array([[2., 1], [1, 2]]),
                    h, [4], [64], [.5])
    force, point = np.r_[tangent.force_n[0], 0], np.array([1., -1, 0])
    result = RigidSemiImplicitSolver().step(
        model, state, RigidBodyInputs(force[None, :], np.cross(point, force)[None, :]), h,
    )
    assert result.report.accepted
    np.testing.assert_allclose(result.state.angular_velocity, [[0, 0, 1/32]], atol=1e-12)
    R = quaternion_matrix(result.state.orientation)[0]
    actual = (result.state.position[0] + R @ point - point)[:2][None, :]
    check = tangent.verify_motion(actual, tolerance_m=1e-12)
    assert not check.accepted and check.maximum_error_m > 1e-6
    np.testing.assert_allclose(check.discrepancy_m, actual-tangent.predicted_displacement_m, atol=1e-15)
    np.testing.assert_array_equal(state.orientation, [[0, 0, 0, 1]])
    np.testing.assert_array_equal(state.position, [[0, 0, 0]])
