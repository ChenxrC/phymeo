"""Actual backend execution under host FP64 thickness maps."""

import numpy as np
import pytest
from test_tyx_surface_port import workspace

from phymeo import (
    ParticleModel,
    ParticleState,
    StepInputs,
    TriangleOffsetPointPort,
    TyxClothParticipant,
)

pytestmark = pytest.mark.tyx


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_real_thickness_load_torque_response_energy_and_execution(tyx_mode, sign):
    x = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
    inverse_mass = np.array([1.0, 2, 3])
    state = ParticleState(x, np.zeros((3, 3)))
    ws = workspace(
        TyxClothParticipant(ParticleModel(inverse_mass, np.zeros(3)), None, tyx_mode),
        state,
        StepInputs.zeros(3),
        tyx_mode,
    )
    try:
        ws.predict()
        p = TriangleOffsetPointPort(
            ws.point_port(np.arange(3)),
            np.array([[0, 1, 2]]),
            np.array([[0.2, 0.3, 0.5]]),
            np.array([sign * 0.2]),
        )
        count = ws.diagnostics()["execution"]["completed_launches"]
        impulse = np.array([[1.0, 0, 0]])
        np.testing.assert_allclose(
            p.response(impulse), [[1.09, 0.04, -sign * 0.08]], atol=2e-5, rtol=2e-5
        )
        after = p.apply_impulses(impulse)
        loads = after.velocity / inverse_mass[:, None]
        np.testing.assert_allclose(
            loads, [[0.2, 0, sign * 0.2], [0.3, 0, -sign * 0.2], [0.5, 0, 0]], atol=2e-5
        )
        np.testing.assert_allclose(loads.sum(axis=0), impulse[0], atol=2e-5)
        np.testing.assert_allclose(np.cross(x, loads).sum(axis=0), [0, sign * 0.2, -0.5], atol=2e-5)
        actual_energy = np.sum(after.velocity**2 / inverse_mass[:, None]) / 2
        assert actual_energy == pytest.approx(0.545, abs=2e-5, rel=2e-5)
        audit = p.candidate_audit(after)
        assert audit.energy_change == pytest.approx(actual_energy, abs=2e-12)
        np.testing.assert_allclose(audit.candidate_velocities, p.response(impulse), atol=2e-5)
        np.testing.assert_array_equal(ws.velocity(), state.velocity)
        np.testing.assert_array_equal(after.position, p.snapshot().position)
        assert ws.diagnostics()["execution"]["completed_launches"] > count
    finally:
        ws.close()


def test_real_shared_thickness_attachments_sum_once(tyx_mode):
    x = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
    state = ParticleState(x, np.zeros((3, 3)))
    ws = workspace(
        TyxClothParticipant(ParticleModel(np.ones(3), np.zeros(3)), None, tyx_mode),
        state,
        StepInputs.zeros(3),
        tyx_mode,
    )
    try:
        ws.predict()
        p = TriangleOffsetPointPort(
            ws.point_port(np.arange(3)),
            np.array([[0, 1, 2], [0, 1, 2]]),
            np.array([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]),
            np.array([0.2, -0.2]),
        )
        # Equal opposing tangential impulses form a pure couple about +Y.
        impulse = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
        count = ws.diagnostics()["execution"]["completed_launches"]
        after = p.apply_impulses(impulse)
        np.testing.assert_allclose(
            after.velocity, [[0, 0, 0.4], [0, 0, -0.4], [0, 0, 0]], atol=2e-5
        )
        np.testing.assert_allclose(after.velocity.sum(axis=0), 0, atol=2e-5)
        np.testing.assert_allclose(np.cross(x, after.velocity).sum(axis=0), [0, 0.4, 0], atol=2e-5)
        assert p.candidate_audit(after).energy_change == pytest.approx(0.16, abs=2e-5)
        np.testing.assert_allclose(
            p.candidate_audit(after).candidate_velocities, p.response(impulse), atol=2e-5
        )
        np.testing.assert_array_equal(ws.velocity(), state.velocity)
        assert ws.diagnostics()["execution"]["completed_launches"] > count
    finally:
        ws.close()
