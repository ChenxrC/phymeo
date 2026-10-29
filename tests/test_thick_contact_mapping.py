"""Independent offset mapping, aggregate response, and phase-law controls."""

from fractions import Fraction as F

import numpy as np
import pytest
from test_thick_patch_geometry import TRI

import phymeo as ph
from phymeo.coupling.ports import ParticlePointPort


def data(weights=None):
    weights = np.eye(3) if weights is None else np.array(weights, float)
    state = ph.ParticleState(TRI, np.array([[0.2, 0.3, -0.1], [-0.4, 0.2, 0.7], [0.6, -0.2, 0.1]]))
    model = ph.ParticleModel(np.ones(3), np.zeros(3))
    p = ph.TriangleOffsetPointPort(
        ParticlePointPort(model, state, np.arange(3), domain="nodes"),
        np.tile([0, 1, 2], (len(weights), 1)),
        weights,
        np.full(len(weights), -0.125),
    )
    return state, model, p


def ledger_api():
    from phymeo.validation.thick_contact_mapping import ThickContactMappingLedger

    return ThickContactMappingLedger


def test_full_offset_mapping_preserves_actual_force_moment_power():
    state, model, p = data()
    s = np.array([[0.2, 0.3, -0.1], [-0.1, 0.4, 0.2], [0.6, 0.7, -0.3]])
    ledger = ledger_api()(state.position, p.triangles, p.weights, p.offsets)
    node = ledger.aggregate(p.jacobians, s, len(TRI))
    report = ledger.require(p.jacobians, s, node, state.velocity, tolerance=1e-8)
    assert max(report.values()) < 1e-12
    np.testing.assert_allclose(np.sum(node, axis=0), s.sum(axis=0), atol=1e-15)
    np.testing.assert_allclose(
        np.cross(TRI, node).sum(axis=0), np.cross(p.positions(), s).sum(axis=0), atol=1e-15
    )


def test_shared_nodes_and_robot_keep_cross_terms():
    _, _, p = data([[1, 0, 0], [1, 0, 0]])
    ledger = ledger_api()(TRI, p.triangles, p.weights, p.offsets)
    w = ledger.response(
        p.jacobians,
        np.ones(3),
        np.tile(np.array([[0.0], [1.0], [0.0]]), (2, 1, 1)),
        np.array([[0.5]]),
    )
    assert w.shape == (2, 2, 3, 3)
    np.testing.assert_allclose(w[0, 1], w[0, 0], atol=1e-15)
    impulses = np.array([[1.0, 2.0, 3.0], [-0.5, -1.0, -1.5]])
    node = ledger.aggregate(p.jacobians, impulses, 3)
    aggregate_energy = 0.5 * np.sum(node * node) + 0.25 * np.sum(impulses[:, 1]) ** 2
    separated_energy = sum(
        0.5
        * np.sum(
            ledger.aggregate(
                p.jacobians[i : i + 1], impulses[i : i + 1], 3, triangles=p.triangles[i : i + 1]
            )
            ** 2
        )
        + 0.25 * impulses[i, 1] ** 2
        for i in range(2)
    )
    assert abs(aggregate_energy - separated_energy) > 0.1


def test_missing_thickness_block_and_centroid_only_are_rejected():
    state, _, p = data()
    s = np.tile([-0.2, 0.0, 0.0], (3, 1))
    ledger = ledger_api()(TRI, p.triangles, p.weights, p.offsets)
    bad = p.weights[:, :, None, None] * np.eye(3)[None, None, :, :]
    node = ledger.aggregate(bad, s, 3)
    with pytest.raises(ph.SolverFailure, match="mapping|moment"):
        ledger.require(bad, s, node, state.velocity, tolerance=1e-8)
    # Independent full cap oracle: the centroid-only admissible energy misses an approaching extreme.
    vy = -0.8
    vr = -0.3
    total = np.array([-0.6, 0.6, 0.0])
    _, _, centroid = data([[1 / 3] * 3])
    loads = ledger_api()(TRI, centroid.triangles, centroid.weights, centroid.offsets).aggregate(
        centroid.jacobians, total[None, :], 3
    )
    post = np.tile([1.0, vy, 0.0], (3, 1)) + loads
    u = (
        np.einsum("rijk,rk->rij", p.jacobians, post[p.triangles])
        if False
        else np.array(
            [sum(j @ v for j, v in zip(block, post, strict=True)) for block in p.jacobians]
        )
    )
    assert min(u[:, 1] - (vr - 0.5 * total[1])) < -0.01


def test_fp32_requests_and_actual_native_response_are_independent():
    ledger = ledger_api()
    stored = np.array([0.1], np.float32).astype(float)
    ledger.require_fp32(stored)
    with pytest.raises(ph.SolverFailure, match="FP32"):
        ledger.require_fp32(np.nextafter(stored, np.inf))
    # Actual live pre/post response is separately exercised by native Task1 tests.


def test_phase_normal_capacity_exact_sum_and_row_scale_invariance():
    from phymeo.validation.thick_contact_mapping import phase_capacity, row_scales

    c = phase_capacity([1, 0, 0], [1, 0, 0], [0, 1, 0], [0, F(1, 2**53), 0], 1.0, False, True)
    assert c == (F(1) + F(1, 2**53),) * 2 and c[0] > F(float(c[0]))
    assert (
        phase_capacity([1, 0, 0], [1, 0, 0], [0, 1, 0], [0, F(1, 2**53), 0], 1.0, True, True)
        == (F(1, 2**53),) * 2
    )
    a = np.diag([2.0, 3.0, 4.0])
    k, kappa = row_scales(a, np.array([0.0, 1.0, 0.0]))
    assert (k, kappa) == (3.0, 9.0)
    blocks = np.zeros((2, 2, 3, 3))
    blocks[0, 0] = a
    assert row_scales(blocks[0, 0], np.array([0.0, 1.0, 0.0])) == (k, kappa)


def test_phase_audit_keeps_exact_relative_velocity_and_paired_moment():
    from phymeo.validation.thick_contact_mapping import paired_moment_residual, phase_audit

    u = F(1) + F(1, 2**53)
    r, _, _ = phase_audit([0, 1, 0], [1, 0, 0], [u, 0, 0], (F(1), F(1)), 3.0, 1.0)
    assert r["positive_work"] > 3.0
    assert (
        paired_moment_residual([[0.0, 0.01, 0.0]], [[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], 1e-8, 1e-8)
        > 1e-8
    )


def review_force_data(shared=False):
    positions = np.array([[-1.0, 0.0, -1.0], [0.0, 0.0, 2.0], [1.0, 0.0, -1.0]])
    triangles = np.tile([0, 1, 2], (2 if shared else 1, 1))
    weights = np.tile([0.25, 0.5, 0.25], (len(triangles), 1))
    ledger = ledger_api()(positions, triangles, weights, np.zeros(len(triangles)))
    blocks = weights[:, :, None, None] * np.eye(3)[None, None, :, :]
    world = np.tile([2.0**20, 0.0, 0.0], (len(triangles), 1))
    return ledger, blocks, world


@pytest.mark.parametrize("defect", ["translation", "canceling"])
def test_review_resultant_and_weighted_mapping(defect):
    ledger, blocks, world = review_force_data()
    delta = 2.0**-36
    if defect == "translation":
        blocks += delta * np.eye(3)[None, None, :, :]
    else:
        blocks[0, 0, 0, 0] += delta
        blocks[0, 2, 0, 0] -= delta
    actual = ledger.aggregate(blocks, world, 3)
    force = actual.sum(axis=0) - world.sum(axis=0)
    if defect == "translation":
        assert abs(force[0]) > 1e-5
    else:
        np.testing.assert_array_equal(force, 0.0)
    with pytest.raises(ph.SolverFailure, match="mapping|force"):
        ledger.require(blocks, world, actual, np.zeros((3, 3)), tolerance=1e-8)


def test_review_valid_large_impulse_and_shared_nodes():
    for shared in (False, True):
        ledger, blocks, world = review_force_data(shared)
        actual = ledger.aggregate(blocks, world, 3)
        report = ledger.require(blocks, world, actual, np.zeros((3, 3)), tolerance=1e-8)
        assert max(report.values()) == 0.0


def test_review_shared_nodes_canceling_weighted_mapping():
    ledger, blocks, world = review_force_data(True)
    bad = blocks.copy()
    bad[0, 0, 0, 0] += 2.0**-36
    bad[1, 2, 0, 0] -= 2.0**-36
    actual = ledger.aggregate(bad, world, 3)
    np.testing.assert_array_equal(actual.sum(axis=0), world.sum(axis=0))
    with pytest.raises(ph.SolverFailure, match="mapping|force"):
        ledger.require(bad, world, actual, np.zeros((3, 3)), tolerance=1e-8)


def test_review_zero_impulse_keeps_coefficient_guard():
    ledger, blocks, world = review_force_data()
    world[:] = 0.0
    blocks[0, 0, 0, 0] += 1e-6
    with pytest.raises(ph.SolverFailure, match="mapping"):
        ledger.require(
            blocks, world, ledger.aggregate(blocks, world, 3), np.zeros((3, 3)), tolerance=1e-8
        )
