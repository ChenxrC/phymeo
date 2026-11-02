"""Unilateral backing impulses with all participant masses and retained material owners."""

from dataclasses import replace

import numpy as np
import pytest
from test_compliant_cloth_contact import step, triangle_case

import phymeo as ph
from phymeo.solvers.finite_pad_backing import project_backing_velocity


@pytest.mark.parametrize("duplicates", [1, 2])
def test_two_dynamic_masses_share_only_the_necessary_impulse(duplicates):
    b = np.repeat([[1.0, -1.0]], duplicates, axis=0)
    wbt = np.diag([2.0, 0.5]) @ b.T
    result = project_backing_velocity(b, wbt, np.array([-1.0, 0.0]), np.zeros(duplicates))
    assert len(result.impulses) == duplicates
    assert sum(result.impulses) == pytest.approx(0.4)
    np.testing.assert_allclose(result.velocity, [-0.2, -0.2], atol=1e-12)
    assert result.work_j == pytest.approx(-0.2)
    assert result.residual_m_s <= 1e-8


def test_backing_has_no_impulse_for_allowed_closure_or_separation():
    b = np.array([[1.0, -1.0]])
    for v in ([-0.1, 0.0], [0.5, 0.0]):
        result = project_backing_velocity(b, b.T, np.array(v), np.array([-0.25]))
        np.testing.assert_array_equal(result.velocity, v)
        assert result.impulses.tolist() == [0.0] and result.work_j == 0.0


def test_coupled_corner_rows_include_cross_terms():
    b = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
    wbt = np.diag([1.0, 1.0, 0.5]) @ b.T
    result = project_backing_velocity(b, wbt, np.array([-1.0, -0.5, 0.0]), np.zeros(2))
    np.testing.assert_allclose(result.velocity, [-0.375] * 3, atol=1e-8)
    assert result.impulses == pytest.approx([0.625, 0.125], abs=1e-8)
    assert result.work_j < 0.0


def backed_case(height, speed):
    old, state, inputs = triangle_case(height=height, speed=speed)
    pad = replace(old.patches[0], radius_m=0.002, backing_clearance_m=0.000125)
    program = ph.CompliantClothContactProgram(
        old.scene,
        "sheet",
        old.material,
        old.surfaces,
        state.domains["sheet"].position,
        (pad,),
        max_penetration_m=0.01,
    )
    return program, state, inputs


def test_real_sheet_and_articulated_link_stop_before_pad_middle():
    program, state, inputs = backed_case(0.0082, -0.8)
    h = 0.0005
    out = step(program, state, inputs, h)
    assert out.report.accepted, out.report.reason
    row = out.report.substeps[0].diagnostics["patches"]["support"]
    assert out.report.substeps[0].diagnostics["force_sample"] == "substep-entry"
    assert np.sum(row["backing_impulse_ns"]) > 0.0
    f = np.asarray(row["force_n"])
    v = out.state.domains["sheet"].velocity
    np.testing.assert_allclose(
        0.5 * (v - state.domains["sheet"].velocity).sum(axis=0), h * f.sum(axis=0), atol=1e-13
    )
    assert 2 * out.state.domains["plate"].v[0] == pytest.approx(-h * f[:, 2].sum(), abs=1e-13)
    end = program.observe(out.state)["support"]
    assert min(end["middle_distance_m"]) >= 0.000125 - 1e-10
    assert out.report.substeps[0].diagnostics["backing"]["work_j"] <= 0.0


def test_no_backing_prevents_a_remote_tunneling_negative_from_committing():
    program, state, inputs = backed_case(0.11, -80.0)
    out = step(program, state, inputs, 0.005)
    assert not out.report.accepted and "middle" in out.report.reason
    np.testing.assert_array_equal(
        out.state.domains["sheet"].position, state.domains["sheet"].position
    )
    np.testing.assert_array_equal(out.state.domains["plate"].q, state.domains["plate"].q)


def test_invalid_or_over_capacity_backing_is_not_silently_truncated():
    b = np.ones((129, 1))
    with pytest.raises(ph.SolverFailure, match="capacity"):
        project_backing_velocity(b, b.T, np.array([-1.0]), np.zeros(129))
    with pytest.raises(ph.SolverFailure, match="mobility"):
        project_backing_velocity(b[:1], np.zeros((1, 1)), np.array([-1.0]), np.zeros(1))
    with pytest.raises(ValueError, match="closing"):
        project_backing_velocity(b[:1], b[:1].T, np.array([0.0]), np.ones(1))


def test_already_opening_contact_creates_no_backing_impulse():
    program, state, inputs = backed_case(0.0082, 1.0)
    result = step(program, state, inputs, 0.0005)
    assert result.report.accepted, result.report.reason
    row = result.report.substeps[0].diagnostics["patches"]["support"]
    assert np.all(row["backing_impulse_ns"] == 0.0)
    assert np.all(row["normal_force_n"] == 0.0)
