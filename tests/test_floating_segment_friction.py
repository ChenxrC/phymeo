"""Check Coulomb holding and sliding from the same finite cylinder/plane contact line."""

import json

import numpy as np
import pytest
from test_floating_segment_endpoints import IDENTITY, UP, detect_support

import phymeo as ph


@pytest.mark.parametrize("h", [1 / 64, 1 / 128])
@pytest.mark.parametrize(
    "case,speed,force,mu_static,mu_dynamic",
    [
        ("stationary", 0.0, 1.0, 0.5, 0.25),
        ("forward", 0.25, 2.0, 0.5, 0.25),
        ("reverse", -0.25, -2.0, 0.5, 0.25),
        ("frictionless", 0.25, 0.0, 0.0, 0.0),
    ],
)
def test_line_contact_axial_traction(h, case, speed, force, mu_static, mu_dynamic, tmp_path):
    # A force applied at the contact height contributes both F and a COM moment.
    # With an 8 N normal load, sliding traction is 2 N at mu_dynamic=1/4.
    reversed_owners = speed < 0.0
    cylinder, plane, pair, _ = detect_support(curved_first=reversed_owners)
    assert pair.status == "touching" and pair.curved.locus == "segment"
    center = np.array([0.0, 0.5, 0.0])
    principal = 19 / 48
    body = ph.RigidBodyModel(np.ones(1), np.array([[principal, principal, 0.125]]), -8 * UP)
    model = ph.FloatingRobotModel(body, np.zeros(3))
    state = ph.FloatingRobotState(np.r_[center, IDENTITY], np.array([0.0, 0.0, speed, 0, 0, 0]))
    inputs = ph.FloatingRobotInputs(
        np.empty(0), np.array([[0.0, 0.0, force]]), np.array([[-0.5 * force, 0.0, 0.0]])
    )
    floor = ph.RigidBodyModel(np.zeros(1), np.zeros((1, 3)), np.zeros(3))
    rest = ph.RigidBodyState(
        np.zeros((1, 3)), IDENTITY[None, :], np.zeros((1, 3)), np.zeros((1, 3))
    )
    ids = np.zeros(2, dtype=np.int64)
    endpoints = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    sites = {"cylinder": (ids, endpoints - center), "floor": (ids, endpoints)}
    order = (1, 0) if reversed_owners else (0, 1)
    rows = [(pair, {cylinder: ("cylinder", i), plane: ("floor", i)}, i) for i in order]
    result = ph.step_floating_contacts(
        {"cylinder": (model, state, inputs)},
        sites,
        rows,
        [mu_static] * 2,
        [mu_dynamic] * 2,
        h,
        fixed_bodies={"floor": (floor, rest)},
        local_normals={plane: UP},
    )
    assert result.report.accepted, result.report.reason
    contact = result.impulses
    body_impulses = contact.world_impulses * (1 if pair.second == cylinder else -1)
    np.testing.assert_allclose(
        body_impulses.sum(axis=0), [0.0, 8 * h, -force * h], rtol=0, atol=4e-9
    )
    np.testing.assert_allclose(contact.normal_impulses, [4 * h, 4 * h], atol=4e-9, rtol=0)
    tangent = np.linalg.norm(contact.tangent_impulses, axis=1)
    if speed:
        np.testing.assert_allclose(tangent, mu_dynamic * contact.normal_impulses, atol=4e-9, rtol=0)
    else:
        assert np.all(tangent <= mu_static * contact.normal_impulses + 4e-9)
    actual = result.states["cylinder"]
    np.testing.assert_allclose(actual.v, state.v, atol=3e-8, rtol=0)
    expected_q = state.q.copy()
    expected_q[2] += speed * h
    np.testing.assert_allclose(actual.q, expected_q, atol=2e-9, rtol=0)
    free_energy = 0.5 * (
        (8 * h) ** 2 + (speed + force * h) ** 2 + (0.5 * force * h) ** 2 / principal
    )
    assert contact.energy_change == pytest.approx(0.5 * speed**2 - free_energy, abs=3e-9)
    (tmp_path / "traction.json").write_text(
        json.dumps(
            dict(
                case=case,
                h_s=h,
                sliding_m_s=speed,
                applied_n=force,
                contact_force_n=(body_impulses.sum(axis=0) / h).tolist(),
                impulses_ns=body_impulses.tolist(),
                normal_ns=contact.normal_impulses.tolist(),
                regimes=contact.regimes,
                energy_change_j=contact.energy_change,
                final_velocity=actual.v.tolist(),
                final_q=actual.q.tolist(),
            ),
            allow_nan=False,
        ),
        encoding="utf8",
    )
