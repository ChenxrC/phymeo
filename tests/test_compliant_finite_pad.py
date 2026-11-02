"""Finite-thickness material columns at moving, rounded articulated contact patches."""

from dataclasses import replace

import numpy as np
import pytest
from test_compliant_cloth_contact import step, triangle_case

import phymeo as ph
from phymeo.geometry.pad_columns import check_column_segment, rounded_pad_columns

VERTICES = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
NORMAL = np.array([0.0, 0.0, 1.0])


def test_full_thickness_axis_reaches_rim_in_its_interior():
    # Pad middle plane is z=-1/16. A vertical material column spans it outside the right edge.
    middle = np.array([[1 + 3 / 32, 0.0, -1 / 32]])
    half = np.array([[0.0, 0.0, 1 / 8]])
    site = rounded_pad_columns(VERTICES, NORMAL, 1 / 16, middle, half)
    assert site.gap_m == pytest.approx([1 / 32])
    assert site.fraction == pytest.approx([-1 / 4])
    np.testing.assert_allclose(site.normal, [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(site.points_m, [[1 + 3 / 32, 0.0, -1 / 16]])


def test_column_crossing_pad_middle_is_rejected_even_with_clear_ends():
    with pytest.raises(ph.SolverFailure, match="middle"):
        rounded_pad_columns(
            VERTICES, NORMAL, 1 / 16, np.array([[0.0, 0.0, -1 / 16]]), np.array([[0.0, 0.0, 1 / 8]])
        )
    zero = np.zeros((1, 3))
    a = rounded_pad_columns(VERTICES, NORMAL, 1 / 16, np.array([[0.0, 0.0, 0.25]]), zero)
    b = rounded_pad_columns(VERTICES, NORMAL, 1 / 16, np.array([[0.0, 0.0, -0.375]]), zero)
    with pytest.raises(ph.SolverFailure, match="middle"):
        check_column_segment(a.endpoints_m, a.gap_m + 1 / 16, b.endpoints_m, b.gap_m + 1 / 16)


def test_zero_thickness_and_column_direction_reversal():
    middle = np.array([[0.0, 0.0, 3 / 32], [0.0, 0.0, -7 / 32]])
    half = np.array([[0.0, 0.0, 1 / 32], [0.0, 0.0, 1 / 32]])
    a = rounded_pad_columns(VERTICES, NORMAL, 1 / 16, middle, half)
    b = rounded_pad_columns(VERTICES, NORMAL, 1 / 16, middle, -half)
    assert a.gap_m == pytest.approx([1 / 16, 1 / 16])
    np.testing.assert_allclose(a.points_m, b.points_m)
    np.testing.assert_allclose(a.fraction, -b.fraction)
    np.testing.assert_allclose(a.normal, [[0, 0, 1], [0, 0, -1]])


@pytest.mark.parametrize(("height", "direction"), [(0.011, 1), (-0.015, -1)])
def test_link_and_cloth_receive_actual_midpoint_impulses_on_either_face(height, direction):
    old, state, inputs = triangle_case(height=height, speed=0.0)
    pad = replace(old.patches[0], radius_m=0.002)
    program = ph.CompliantClothContactProgram(
        old.scene,
        "sheet",
        old.material,
        old.surfaces,
        state.domains["sheet"].position,
        (pad,),
        max_penetration_m=0.01,
    )
    dt = 0.0005
    out = step(program, state, inputs, dt)
    assert out.report.accepted, out.report.reason
    row = out.report.substeps[0].diagnostics["patches"]["support"]
    f = np.asarray(row["force_n"])
    assert np.all(f[:, 2] * direction > 0)
    np.testing.assert_allclose(row["site_normals"], np.tile([0, 0, direction], (3, 1)), atol=1e-8)
    assert np.all(np.asarray(row["gap_m"]) > 0)
    np.testing.assert_allclose(
        0.5 * (out.state.domains["sheet"].velocity - state.domains["sheet"].velocity).sum(axis=0),
        dt * f.sum(axis=0),
        atol=1e-14,
    )
    assert 2 * out.state.domains["plate"].v[0] == pytest.approx(-dt * f[:, 2].sum(), abs=1e-14)


def test_fast_crossing_rolls_back_in_scene_stepper():
    old, state, inputs = triangle_case(height=0.11, speed=-80.0)
    pad = replace(old.patches[0], radius_m=0.002)
    program = ph.CompliantClothContactProgram(
        old.scene,
        "sheet",
        old.material,
        old.surfaces,
        state.domains["sheet"].position,
        (pad,),
        max_penetration_m=0.01,
    )
    out = step(program, state, inputs, 0.005)
    assert not out.report.accepted and "middle" in out.report.reason
    np.testing.assert_array_equal(
        out.state.domains["sheet"].position, state.domains["sheet"].position
    )
    np.testing.assert_array_equal(out.state.domains["plate"].q, state.domains["plate"].q)
