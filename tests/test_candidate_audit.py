"""Inspect published candidates independently of response-query predictions."""

import numpy as np
import pytest
from test_contact_impulse import node, solve

from phymeo import (
    MultiDomainPointOperator,
    ParticleModel,
    ParticlePointPort,
    ParticleState,
    SolverFailure,
    WeightedPointPort,
)


def test_candidate_audit_is_public_and_owned():
    from phymeo import PointCandidateAudit

    values = np.zeros((2, 3))
    audit = PointCandidateAudit(values, values, 0.0)
    values[:] = 4
    np.testing.assert_array_equal(audit.initial_velocities, 0)
    assert not audit.candidate_velocities.flags.writeable
    with pytest.raises(SolverFailure):
        PointCandidateAudit(np.zeros((1, 3)), np.zeros((1, 3)), np.inf)


def test_weighted_graph_audits_whole_domain_energy_once():
    state = ParticleState(np.zeros((2, 3)), np.array([[2.0, 1, 0], [-1.0, 0, 3]]))
    base = ParticlePointPort(
        ParticleModel(np.array([0.5, 2.0]), np.zeros(3)), state, [0, 1], domain="body"
    )
    weights = np.array([[0.25, 0.75], [0.75, 0.25]])
    body = WeightedPointPort(base, weights)
    floor = node("floor", 0, [0, 0, 0])
    graph = MultiDomainPointOperator(
        {"body": body, "floor": floor}, [("body", 0), ("body", 1)], [("floor", 0), ("floor", 0)]
    )
    impulse = np.array([[1.0, -2, 0], [-0.3, 0.2, 1]])
    states = graph.apply_impulses(impulse)
    audit = graph.audit_candidates(states)
    expected = state.velocity + np.array([0.5, 2.0])[:, None] * (weights.T @ impulse)
    np.testing.assert_allclose(audit.candidate_velocities, weights @ expected)
    expected_energy = np.sum((expected**2 - state.velocity**2) / np.array([0.5, 2.0])[:, None]) / 2
    assert audit.energy_change == pytest.approx(expected_energy, abs=1e-12)


def test_hidden_unattached_candidate_energy_change_is_rejected():
    class Corrupt(ParticlePointPort):
        def apply_impulses(self, p):
            state = super().apply_impulses(p)
            state.velocity[1, 0] = 2.0
            return state

    state = ParticleState(np.zeros((2, 3)), np.array([[0.0, -1, 0], [0.0, 0, 0]]))
    body = Corrupt(ParticleModel(np.ones(2), np.zeros(3)), state, [0], domain="body")
    graph = MultiDomainPointOperator(
        {"body": body, "floor": node("floor", 0, [0, 0, 0])}, [("body", 0)], [("floor", 0)]
    )
    with pytest.raises(SolverFailure, match="candidate energy"):
        solve(graph, [[0.0, 1, 0]])
    np.testing.assert_array_equal(body.snapshot().velocity, state.velocity)


def test_missing_candidate_audit_cannot_claim_certified_contact():
    class Legacy(ParticlePointPort):
        candidate_audit = None

    body = Legacy(
        ParticleModel(np.ones(1), np.zeros(3)),
        ParticleState(np.zeros((1, 3)), np.array([[0.0, -1, 0]])),
        [0],
        domain="body",
    )
    graph = MultiDomainPointOperator(
        {"body": body, "floor": node("floor", 0, [0, 0, 0])}, [("body", 0)], [("floor", 0)]
    )
    with pytest.raises(SolverFailure, match="lacks candidate audit"):
        solve(graph, [[0.0, 1, 0]])


def test_equal_spurious_domain_velocity_changes_cannot_cancel_in_relative_audit():
    class Flip(ParticlePointPort):
        def apply_impulses(self, p):
            state = super().apply_impulses(p)
            state.velocity[:, 0] *= -1
            return state

    ports = {
        name: Flip(
            ParticleModel(np.ones(1), np.zeros(3)),
            ParticleState(np.zeros((1, 3)), np.array([[1.0, vy, 0]])),
            [0],
            domain=name,
        )
        for name, vy in (("a", -1), ("b", 1))
    }
    graph = MultiDomainPointOperator(ports, [("a", 0)], [("b", 0)])
    with pytest.raises(SolverFailure, match="candidate velocity"):
        solve(graph, [[0.0, 1, 0]])


def test_rigid_rotated_inertia_candidate_energy_and_points():
    from phymeo import RigidBodyModel, RigidBodyState, RigidPointPort

    root = np.sqrt(0.5)
    model = RigidBodyModel(np.array([0.5]), np.array([[2.0, 3, 5]]), np.zeros(3))
    before = RigidBodyState(
        np.zeros((1, 3)),
        np.array([[0.0, 0, root, root]]),
        np.array([[1.0, 2, 3]]),
        np.array([[2.0, 1, -1]]),
    )
    local = np.array([[1.0, 0, 0], [0.0, 1, 0]])
    port = RigidPointPort(model, before, [0, 0], local)
    impulse = np.array([[1.0, -0.5, 2], [0.2, -0.1, 0.3]])
    after = port.apply_impulses(impulse)
    audit = port.candidate_audit(after)
    arms = np.array([[0.0, 1, 0], [-1.0, 0, 0]])
    expected = after.velocity + np.cross(after.angular_velocity, arms)
    np.testing.assert_allclose(audit.candidate_velocities, expected, atol=1e-14)
    # A 90-degree z rotation swaps the x/y principal moments.
    energy = np.sum(after.velocity**2 - before.velocity**2)
    energy += np.sum((after.angular_velocity**2 - before.angular_velocity**2) * [3, 2, 5]) / 2
    assert audit.energy_change == pytest.approx(energy, abs=1e-12)


def test_large_normal_offset_rotated_features_certify_or_explicitly_refuse():
    from phymeo import closest_point_triangle, closest_segment_segment

    rotation, _ = np.linalg.qr(np.array([[1.0, 2, 3], [-2.0, 3, 1], [4.0, 1, -1]]))
    a = np.array([[0.0, 0, 0], [1.0, 0, 0]]) @ rotation.T
    b = np.array([[0.5, 2, 1e8], [0.5, 3, 1e8]]) @ rotation.T
    tri = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]]) @ rotation.T
    point = np.array([0.5, 2, 1e10]) @ rotation.T
    for operation, args, expected_a, expected_b in (
        (closest_point_triangle, (point, tri), point, tri[2]),
        (closest_segment_segment, (a, b), np.array([0.5, 0, 0]) @ rotation.T, b[0]),
    ):
        try:
            result = operation(*args)
        except SolverFailure as error:
            assert "optimality certificate" in str(error)
        else:
            np.testing.assert_allclose(result.first, expected_a, atol=1e-6, rtol=0)
            np.testing.assert_allclose(result.second, expected_b, atol=1e-6, rtol=0)


def test_tetrahedral_candidate_energy_counts_mass_once_with_duplicate_attachments():
    from test_tetrahedral_ports import port

    solid = port()
    impulses = np.array([[99.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [0.0, 1, 0]])
    candidate = solid.apply_impulses(impulses)
    audit = solid.candidate_audit(candidate)
    np.testing.assert_array_equal(
        audit.candidate_velocities, [[0, 0, 0], [12, 0, 0], [12, 0, 0], [0, 4, 0]]
    )
    assert audit.energy_change == pytest.approx((12**2 + 4**2) * 0.25 / 2)
