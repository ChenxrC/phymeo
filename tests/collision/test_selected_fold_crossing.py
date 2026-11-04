"""Observe material and spatial feature ownership across both signed folds in the full program."""

import json
from dataclasses import asdict

import numpy as np
import pytest
from test_folded_material_seam import H, fixture, ref_event

import phymeo as ph
from phymeo.solvers.folded_contact_program import FoldedContactProgram


@pytest.mark.parametrize("sign", [-1, 1], ids=["ridge", "valley"])
def test_geometry_selected_site_transfers_one_loaded_episode_between_real_wings(sign, tmp_path):
    _, initial, permutation, masses, rotation = fixture(sign, rotated=sign > 0)
    base = initial.particles
    x, v = base.position[permutation], base.velocity[permutation]
    # The four other material nodes form a second, independently registered finite hinge.
    state = ph.ParticleState(np.vstack((x, x[1:] + rotation @ [4.0, 0, 0])), np.vstack((v, v[1:])))
    model = ph.ParticleModel(1 / np.r_[masses, masses[1:]], np.zeros(3))
    probe = ph.CollisionShape(ph.CollisionKey(5, 61, 11), "particle", (2,))
    shapes = [probe]
    bindings = {probe.key: ph.ParticleAttachment((0,))}
    for body, offset in ((9, 0), (12, 4)):
        for key, ids, rows in ((20, (7, 17, 31), (3, 1, 2)), (30, (17, 49, 31), (1, 4, 2))):
            face = ph.CollisionShape(ph.CollisionKey(5, body, key), "triangle", ids)
            shapes.append(face)
            bindings[face.key] = ph.ParticleAttachment(tuple(i + offset for i in rows))
    program = FoldedContactProgram(
        model,
        ph.CollisionModel(shapes[::-1] if sign > 0 else shapes),
        bindings,
        layer=ph.NormalLayer(1 / 16, 64, friction=0.25),
        tangent_stiffness_n_per_m=16,
        static_friction=0.5,
    )
    before = program.initial_state(state)
    load = ph.StepInputs.zeros(9)
    load.force[0] = rotation @ [0.0, 0.0, -2.0]
    result = program.step(before, load, H)
    assert result.report.accepted and result.selection.eligible == 2
    assert result.selection.broad_rejected == 1
    (contact,) = result.contacts
    (event,) = result.events
    assert event.kind == "enter" and event.fraction == 0
    assert len(result.state.active) == 1 and result.state.active[0] == contact.history
    assert contact.history.regime == "slide" and contact.history.plastic_slip_m > 0
    assert contact.normal_force_n == pytest.approx(2.0, abs=2e-9)
    left, right = contact.legs
    end = result.state.particles
    cross = ref_event(x @ rotation, end.position[:5] @ rotation)
    assert left.t1 == pytest.approx(cross, abs=2e-9) and right.t0 == left.t1
    assert left.face.shape == 20 and right.face.shape == 30 and contact.key.body == (5, 9)
    for station in (left.end, right.start):
        w = dict(station.material_weights)
        np.testing.assert_allclose([w[17], w[31]], [0.5, 0.5], atol=2e-11)
        assert [s.feature for s in station.spatial] == (
            ["edge", "edge"] if sign < 0 else ["face", "face"]
        )
    np.testing.assert_allclose(left.end.material_point_m, right.start.material_point_m, atol=2e-10)
    if sign < 0:
        assert all(s.distance_m > left.end.normal_gap_m for s in left.end.spatial)
    else:
        a, b = left.end.spatial
        assert np.linalg.norm(np.asarray(a.point_m) - b.point_m) > 1e-4
    m = 1 / model.inverse_mass[:, None]
    impulse = m * (end.velocity - state.velocity) - H * load.force
    np.testing.assert_allclose(result.nodal_contact_impulse_ns, impulse, atol=2e-9)
    np.testing.assert_array_equal(impulse[5:], 0.0)
    dL = np.cross(end.position, m * end.velocity).sum(axis=0) - np.cross(
        state.position, m * state.velocity
    ).sum(axis=0)
    actual = dL - H * np.cross(state.position, load.force).sum(axis=0)
    np.testing.assert_allclose(result.discrete_angular_impulse_nms, actual, atol=2e-10)
    np.testing.assert_allclose(result.integrated_angular_impulse_nms, 0, atol=2e-10)
    (tmp_path / "selected-fold-crossing.json").write_text(
        json.dumps(
            {
                "sign": sign,
                "dt_s": H,
                "eligible": result.selection.eligible,
                "station_key": asdict(contact.key),
                "event_fraction": cross,
                "before": asdict(left.end),
                "after": asdict(right.start),
                "end_history": asdict(contact.history),
                "actual_internal_angular_impulse_nms": actual.tolist(),
                "integrated_internal_angular_impulse_nms": result.integrated_angular_impulse_nms,
                "normal_n": contact.normal_force_n,
                "tangent_n": contact.shear_n,
                "start_x_m": state.position.tolist(),
                "end_x_m": end.position.tolist(),
                "start_v_m_s": state.velocity.tolist(),
                "end_v_m_s": end.velocity.tolist(),
                "nodal_impulses_ns": impulse.tolist(),
            },
            allow_nan=False,
        ),
        encoding="utf8",
    )
