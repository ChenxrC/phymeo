"""Whole-object device responses, lifecycle and configuration/impulse units."""

import importlib
import importlib.util

import numpy as np
import pytest
from test_robot_dynamics import planar_mass
from test_robot_kinematics import robot_model
from test_tetrahedral_material import unit_tetra

from phymeo.core.model import ParticleModel
from phymeo.core.robot import RobotInputs, RobotState
from phymeo.core.scene import BackendContext, StepSettings
from phymeo.core.state import ParticleState, StepInputs

pytestmark = pytest.mark.tyx


def api():
    name = "phymeo.backends.tyx_participants"
    assert importlib.util.find_spec(name) is not None, "P6 device participants missing"
    return importlib.import_module(name)


def workspace(factory, state, inputs, mode, h=0.01):
    return factory.create_workspace(
        "body",
        state,
        inputs,
        h,
        StepSettings(tolerance=2e-6),
        BackendContext("tyx:" + mode, np.float64),
    )


def test_robot_shared_ancestor_cross_response_and_instantaneous_q(tyx_mode):
    model = robot_model(gravity=np.zeros(3))
    state = RobotState(np.zeros(2), np.zeros(2))
    ws = workspace(
        api().TyxRobotParticipant(model, tyx_mode), state, RobotInputs.zeros(2), tyx_mode
    )
    try:
        ws.predict()
        port = ws.point_port(np.array([0, 1]), np.array([[0.0, -0.5, 0.0], [0.0, -0.5, 0.0]]))
        j = np.array([[0.5, 0.0], [1.0, 0.5]])
        impulse = np.array([[0.1, 0.0, 0.0], [-0.2, 0.0, 0.0]])
        expected_delta = np.linalg.solve(planar_mass(np.zeros(2)), j.T @ impulse[:, 0])
        response = port.response(impulse)
        np.testing.assert_allclose(response[:, 0], j @ expected_delta, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(response[:, 1:], 0.0, atol=2e-5)
        candidate = port.validate_candidate(port.apply_impulses(impulse))
        np.testing.assert_allclose(candidate.v, expected_delta, atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(candidate.q, state.q)
        np.testing.assert_array_equal(ws.velocity(), state.v)
        candidate.v[:] = 99
        np.testing.assert_array_equal(port.snapshot().v, state.v)
        moved = port.snapshot()
        moved.q[0] = 1
        with pytest.raises(ValueError, match="configuration"):
            port.validate_candidate(moved)
    finally:
        ws.close()


def test_robot_duplicate_cancellation_and_world_local_roundtrip(tyx_mode):
    state = RobotState(np.array([0.2, -0.4]), np.zeros(2))
    factory = api().TyxRobotParticipant(robot_model(gravity=np.zeros(3)), tyx_mode)
    ws = workspace(factory, state, RobotInputs.zeros(2), tyx_mode)
    try:
        ws.predict()
        indices = np.array([1, 1])
        local = np.array([[0.1, -0.5, 0.2], [0.1, -0.5, 0.2]])
        port = ws.point_port(indices, local)
        np.testing.assert_allclose(ws.world_to_local(indices, port.positions()), local, atol=2e-5)
        loads = np.array([[0.1, 0.2, -0.3], [-0.1, -0.2, 0.3]])
        np.testing.assert_allclose(port.response(loads), 0.0, atol=2e-5)
        np.testing.assert_allclose(ws.generalized_load(indices, local, loads), 0.0, atol=2e-5)
        before = port.positions()
        ws.apply_position_load(np.array([1]), local[:1], np.array([[0.001, 0.0, 0.0]]))
        np.testing.assert_array_equal(port.positions(), before)
    finally:
        ws.close()


def test_position_load_reconstructs_once_and_velocity_impulse_never_moves_q(tyx_mode):
    model = robot_model(
        1,
        joint_types=("prismatic",),
        joint_axis=np.array([[0.0, 1.0, 0.0]]),
        mass=np.array([2.0]),
        gravity=np.zeros(3),
    )
    ws = workspace(
        api().TyxRobotParticipant(model, tyx_mode),
        RobotState(np.zeros(1), np.zeros(1)),
        RobotInputs.zeros(1),
        tyx_mode,
        h=0.01,
    )
    try:
        ws.predict()
        ws.apply_position_load(np.array([0]), np.zeros((1, 3)), np.array([[0.0, 0.02, 0.0]]))
        np.testing.assert_allclose(ws.configuration(), [0.01], atol=2e-5)
        np.testing.assert_array_equal(ws.velocity(), [0.0])
        ws.prepare_velocity()
        np.testing.assert_allclose(ws.velocity(), [1.0], atol=2e-5)
        with pytest.raises(RuntimeError, match="prepared"):
            ws.prepare_velocity()
        q = ws.configuration()
        ws.apply_velocity_impulses(np.array([0]), np.zeros((1, 3)), np.array([[0.0, 0.2, 0.0]]))
        np.testing.assert_allclose(ws.velocity(), [1.1], atol=2e-5)
        np.testing.assert_array_equal(ws.configuration(), q)
        with pytest.raises(RuntimeError, match="prediction"):
            ws.predict()
        diagnostics = ws.diagnostics()
        assert diagnostics["compute_dtype"] == "float32"
        assert diagnostics["storage_dtype"] == "float64"
        assert diagnostics["execution"]["completed_launches"] > 0
        with pytest.raises(ValueError, match="read-only"):
            diagnostics["predicted_q"][:] = 900
        assert ws.diagnostics()["predicted_q"][0] == 0
    finally:
        ws.close()


@pytest.mark.parametrize("kind", ["cloth", "tetrahedral"])
def test_node_fixed_duplicate_response_and_state_isolation(kind, tyx_mode):
    if kind == "cloth":
        model = ParticleModel(np.array([0.0, 4.0, 4.0, 4.0]), np.zeros(3))
        factory = api().TyxClothParticipant(model, None, tyx_mode)
        position = unit_tetra().rest_position.copy()
    else:
        model = unit_tetra()
        factory = api().TyxTetrahedralParticipant(model, tyx_mode)
        position = model.rest_position.copy()
    position[0] += 0.000000001
    state = ParticleState(position, np.zeros((4, 3)))
    ws = workspace(factory, state, StepInputs.zeros(4), tyx_mode, h=1e-4)
    try:
        ws.predict()
        port = ws.point_port(np.array([0, 1, 1]))
        impulse = np.array([[9.0, 9.0, 9.0], [0.1, 0.0, 0.0], [-0.025, 0.0, 0.0]])
        np.testing.assert_allclose(
            port.response(impulse), [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.0, 0.0]], atol=2e-5
        )
        candidate = port.validate_candidate(port.apply_impulses(impulse))
        np.testing.assert_array_equal(candidate.position, port.snapshot().position)
        np.testing.assert_array_equal(candidate.position[0], state.position[0])
        np.testing.assert_array_equal(candidate.velocity[0], [0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="offset"):
            ws.point_port(np.array([1]), np.ones((1, 3)))
        saved = ws.configuration()
        returned = ws.result()
        returned.position[:] = 99
        np.testing.assert_array_equal(ws.configuration(), saved)
    finally:
        ws.close()


@pytest.mark.parametrize(
    "bad_context",
    [
        BackendContext(),
        BackendContext("tyx:invalid", np.float64),
        BackendContext("tyx:cuda", np.float32),
    ],
)
def test_participant_rejects_context_mismatch(bad_context, tyx_mode):
    factory = api().TyxRobotParticipant(robot_model(1), tyx_mode)
    with pytest.raises(ValueError, match="backend|precision"):
        factory.create_workspace(
            "body",
            RobotState(np.zeros(1), np.zeros(1)),
            RobotInputs.zeros(1),
            0.01,
            StepSettings(),
            bad_context,
        )


def test_signature_binds_ordered_constraints_mode_and_precision(tyx_mode):
    from phymeo.constraints.distance import DistanceConstraints
    from phymeo.core.scene import fingerprint

    model = ParticleModel(np.ones(3), np.zeros(3))
    first = DistanceConstraints(np.array([[0, 1], [1, 2]]), np.ones(2), np.zeros(2))
    second = DistanceConstraints(first.indices[::-1], first.rest_length, first.compliance)
    factory = api().TyxClothParticipant(model, first, tyx_mode)
    signature = factory.problem_signature()
    assert signature["compute_dtype"] == "float32"
    assert signature["storage_dtype"] == "float64"
    assert fingerprint(signature) != fingerprint(
        api().TyxClothParticipant(model, second, tyx_mode).problem_signature()
    )


def test_node_position_units_reconstruct_only_once_and_reject_bad_state(tyx_mode):
    model = ParticleModel(np.array([0.0, 2.0]), np.zeros(3))
    factory = api().TyxClothParticipant(model, None, tyx_mode)
    state = ParticleState(np.array([[0.123456789, 0.0, 0.0], [1.0, 0.0, 0.0]]), np.zeros((2, 3)))
    ws = workspace(factory, state, StepInputs.zeros(2), tyx_mode)
    try:
        ws.predict()
        ws.apply_position_load(np.array([1]), None, np.array([[0.01, 0.0, 0.0]]))
        np.testing.assert_allclose(ws.configuration()[1], [1.02, 0.0, 0.0], atol=2e-5)
        ws.prepare_velocity()
        np.testing.assert_allclose(ws.velocity()[1], [2.0, 0.0, 0.0], atol=2e-5)
        before = ws.configuration()
        ws.apply_velocity_impulses(np.array([1]), None, np.array([[0.1, 0.0, 0.0]]))
        np.testing.assert_allclose(ws.velocity()[1], [2.2, 0.0, 0.0], atol=2e-5)
        np.testing.assert_array_equal(ws.configuration(), before)
        np.testing.assert_array_equal(ws.configuration()[0], state.position[0])
        with pytest.raises(RuntimeError, match="prepared"):
            ws.prepare_velocity()
        port = ws.point_port(np.array([1]))
        corrupt = port.snapshot()
        corrupt.position[0, 0] += 1
        with pytest.raises(ValueError, match="fixed"):
            port.validate_candidate(corrupt)
    finally:
        ws.close()
    with pytest.raises(RuntimeError, match="closed"):
        ws.velocity()
    with pytest.raises((ValueError, TypeError)):
        workspace(factory, RobotState(np.zeros(2), np.zeros(2)), StepInputs.zeros(2), tyx_mode)
    with pytest.raises((ValueError, TypeError)):
        workspace(factory, state, StepInputs.zeros(3), tyx_mode)


def test_port_snapshot_does_not_alias_original_model_state_or_attachments(tyx_mode):
    model = robot_model(1, gravity=np.zeros(3))
    state = RobotState(np.zeros(1), np.zeros(1))
    ws = workspace(
        api().TyxRobotParticipant(model, tyx_mode), state, RobotInputs.zeros(1), tyx_mode
    )
    indices, local = np.array([0]), np.array([[0.0, -0.5, 0.0]])
    try:
        ws.predict()
        port = ws.point_port(indices, local)
        indices[:] = 99
        local[:] = 99
        state.q[:] = 99
        model.mass.flags.writeable = True
        model.mass[:] = 99
        np.testing.assert_allclose(port.positions(), [[0.0, -0.5, 0.0]], atol=2e-5)
        np.testing.assert_allclose(
            port.response(np.array([[2.0, 0.0, 0.0]])), [[6.0, 0.0, 0.0]], atol=2e-5
        )
        assert port.domain == "body" and port.dtype == np.float64
    finally:
        ws.close()


def test_cloth_workspace_uses_ordered_constraints_and_device_reconstruction(tyx_mode):
    from phymeo.constraints.distance import DistanceConstraints, project_distances

    constraints = DistanceConstraints(
        np.array([[0, 1], [1, 2]]), np.ones(2), np.array([1e-5, 2e-5])
    )
    model = ParticleModel(np.array([0.0, 1.0, 2.0]), np.zeros(3))
    initial = ParticleState(
        np.array([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [2.4, 0.0, 0.0]]), np.zeros((3, 3))
    )
    factory = api().TyxClothParticipant(model, constraints, tyx_mode)
    ws = workspace(factory, initial, StepInputs.zeros(3), tyx_mode)
    expected, multipliers = initial.position.copy(), np.zeros(2)
    try:
        ws.predict()
        for _ in range(3):
            ws.position_sweep()
            project_distances(
                expected,
                model.inverse_mass,
                constraints,
                constraints.compliance / 0.01**2,
                multipliers,
            )
        np.testing.assert_allclose(ws.configuration(), expected, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(ws.diagnostics()["distance_multipliers"], multipliers, atol=2e-5)
        ws.prepare_velocity()
        np.testing.assert_allclose(
            ws.velocity(), (expected - initial.position) / 0.01, atol=2e-5, rtol=2e-5
        )
        assert np.isfinite(list(ws.position_metrics().residuals.values())).all()
    finally:
        ws.close()



def test_multiple_cloth_participants_share_exact_state_family_contract(tyx_mode):
    model = ParticleModel(np.ones(1), np.zeros(3))
    first = api().TyxClothParticipant(model, None, tyx_mode)
    second = api().TyxClothParticipant(model, None, tyx_mode)
    assert first.family == second.family
