"""Independent static geometry and bilateral virtual-work checks for RR area points."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from experiments.area_pressure.capsule_box_reference import integrate
from experiments.area_pressure.pusht_entry_area import (
    GRADIENT_PA_M,
    MAX_POINTS,
    _quadrature,
    area_response,
)

# Original pusht-planar compound, not its convex hull. The stem joins the bar
# only across x in [-.018,.018] at y=.020; the remaining bar underside is exposed.
T_BOXES = np.array(
    [
        [[-0.018, -0.070, -0.015], [0.018, 0.020, 0.015]],
        [[-0.065, 0.020, -0.015], [0.065, 0.056, 0.015]],
    ]
)
PLANE_BOX = np.array([[[0.0, -0.08, -0.08], [0.05, 0.08, 0.08]]])


def _area(cells, nv=3):
    count = len(cells)
    return dict(
        points=np.asarray([cell["center"] for cell in cells], dtype=float).reshape(count, 3),
        normals=-np.asarray([cell["normal"] for cell in cells], dtype=float).reshape(count, 3),
        weights=np.asarray([cell["weight"] for cell in cells]),
        stiffness=GRADIENT_PA_M * np.asarray([cell["weight"] for cell in cells]),
        initial_gap=np.asarray([cell["gap"] for cell in cells]),
        jac=np.tile(np.eye(3, nv), (count, 1, 1)),
    )


@pytest.mark.parametrize("indentation", (0.0005, 0.003))
def test_spherical_cap_pressure_matches_analytic_integral_and_area_measure(indentation):
    radius = 0.02
    distance = radius - indentation
    ends = np.array([[-distance, 0.0, 0.0], [-distance, 0.0, 0.0]])
    cells, certificate = _quadrature(ends, PLANE_BOX, radius)
    area = _area(cells)
    q, forces = area_response(area, np.zeros(3), 0.001)
    # Integrate G*(r-sqrt(d^2+rho^2))*2*pi*rho over the circular cap.
    exact_load = np.pi * GRADIENT_PA_M * indentation**2 * (radius - 2 * indentation / 3)
    exact_area = np.pi * (radius**2 - distance**2)
    low, high = certificate["static_normal_load_bounds_n"]
    assert 0 < low <= exact_load <= high
    assert 0 < len(cells) <= certificate["leaf_count"] <= MAX_POINTS
    assert np.all(area["weights"] > 0) and np.all(area["initial_gap"] < 0)
    np.testing.assert_allclose(area["points"][:, 0], 0.0, atol=1e-15)
    np.testing.assert_array_equal(area["normals"], np.tile([1.0, 0.0, 0.0], (len(cells), 1)))
    assert area["weights"].sum() == pytest.approx(exact_area, rel=0.08)
    assert q[0] == pytest.approx(exact_load, rel=0.03)
    np.testing.assert_allclose(q, forces.sum(axis=0), atol=1e-14)
    np.testing.assert_array_equal(q[1:], 0.0)
    assert q[0] == pytest.approx(certificate["static_normal_load_n"], rel=1e-13)
    # At the actual sample points the integrand is analytic, independently of
    # the production closest-segment and stiffness/area assembly helpers.
    pressure = GRADIENT_PA_M * (
        radius - np.sqrt(distance**2 + np.sum(area["points"][:, 1:] ** 2, axis=1))
    )
    np.testing.assert_allclose(forces[:, 0], area["weights"] * pressure, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("case", ("tilted-capsule", "first_load", "peak"))
def test_cropped_sparse_integral_agrees_with_old_full_face_reference_and_t_exterior(case):
    if case == "tilted-capsule":
        # Nonzero normal and both tangential axis components exercise section
        # extrema in the segment interior as well as the spherical endpoint caps.
        ends = np.array([[-0.019, -0.012, -0.015], [-0.014, 0.013, 0.016]])
        boxes, radius = PLANE_BOX, 0.02
    else:
        path = (
            Path(__file__).resolve().parents[1]
            / "experiments/area_pressure/pusht_reference_poses.json"
        )
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["scene"] == "pusht-planar"
        assert (
            saved["source_sha256"]
            == "319b910182e955aeb650f782a5f0c433be4ded9b801ec273e603cb5a15c1bf50"
        )
        assert saved["samples"][case]["tick"] == {"first_load": 750, "peak": 1255}[case]
        ends = np.asarray(saved["samples"][case]["ends"])
        boxes, radius = T_BOXES, 0.018
    cells, certificate = _quadrature(ends, boxes, radius)
    area = _area(cells)
    q, forces = area_response(area, np.zeros(3), 0.001)
    assert 0 < len(cells) <= MAX_POINTS
    reference = integrate(ends, boxes, radius, GRADIENT_PA_M, np.zeros(3), np.eye(3), depth=8)
    bounds = np.asarray(reference["bounds"])
    sparse_bounds = certificate["static_normal_load_bounds_n"]
    assert max(sparse_bounds[0], bounds[1, 0]) <= min(sparse_bounds[1], bounds[1, 1])
    normal_load = np.einsum("ij,ij->", forces, area["normals"])
    allowance = max(1e-9, 0.03 * bounds[1, 1])
    assert bounds[1, 0] - allowance <= normal_load <= bounds[1, 1] + allowance
    assert np.all(q >= bounds[2:5, 0] - allowance)
    assert np.all(q <= bounds[2:5, 1] + allowance)
    # Probe each retained face from both sides: no internal compound seam or
    # convex-hull bridge is allowed to contribute an integration point.
    outward = -area["normals"]
    inside_probe = area["points"] - 1e-8 * outward
    outside_probe = area["points"] + 1e-8 * outward
    inside = (
        (
            (inside_probe[:, None] >= boxes[None, :, 0])
            & (inside_probe[:, None] <= boxes[None, :, 1])
        )
        .all(axis=2)
        .any(axis=1)
    )
    outside = (
        (
            (outside_probe[:, None] >= boxes[None, :, 0])
            & (outside_probe[:, None] <= boxes[None, :, 1])
        )
        .all(axis=2)
        .any(axis=1)
    )
    assert np.all(inside) and not np.any(outside)
    if case != "tilted-capsule":
        seam = (np.abs(area["points"][:, 1] - 0.020) < 1e-12) & (
            np.abs(area["points"][:, 0]) < 0.018
        )
        assert not np.any(seam)
        assert np.any(np.abs(area["points"][:, 0]) > 0.018)


def test_separation_has_empty_measure_and_invalid_law_parameters_are_rejected():
    cells, certificate = _quadrature(
        np.array([[-0.05, 0.0, 0.0], [-0.05, 0.0, 0.01]]), PLANE_BOX, 0.02
    )
    assert cells == [] and certificate["static_normal_load_bounds_n"] == [0, 0]
    empty = _area(cells, nv=4)
    q, forces = area_response(empty, np.zeros(4), 0.001)
    assert forces.shape == (0, 3) and q.shape == (4,)
    np.testing.assert_array_equal(q, 0.0)
    valid = dict(
        points=np.zeros((1, 3)),
        normals=np.array([[1.0, 0.0, 0.0]]),
        weights=np.array([0.25]),
        stiffness=np.array([100.0]),
        initial_gap=np.array([-0.01]),
        jac=np.eye(3)[None],
    )
    for field, value in (
        ("stiffness", [-1.0]),
        ("weights", [-1.0]),
        ("mu", -0.1),
        ("slip", -0.01),
        ("initial_gap", [0.01]),
    ):
        invalid = deepcopy(valid)
        invalid[field] = value
        with pytest.raises(ValueError):
            area_response(invalid, np.zeros(3), 0.001)
    for h in (0.0, -0.001, float("nan")):
        with pytest.raises(ValueError):
            area_response(valid, np.zeros(3), h)
    with pytest.raises(ValueError):
        area_response(valid, np.array([0.0, np.nan, 0.0]), 0.001)


def test_smoothed_friction_dissipates_and_bilateral_point_forces_obey_virtual_work():
    point = np.array([0.3, 0.2, 0.1])
    com_a, com_b = np.zeros(3), np.array([1.0, -0.1, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    ja, jb = np.zeros((3, 6)), np.zeros((3, 6))
    ja[:, :2], jb[:, 3:5] = np.eye(3)[:, :2], np.eye(3)[:, :2]
    ja[:, 2], jb[:, 5] = np.cross(axis, point - com_a), np.cross(axis, point - com_b)
    velocity = np.array([0.2, 0.3, 0.7, 0.1, -0.4, -0.2])
    va = np.r_[velocity[:2], 0.0] + velocity[2] * np.cross(axis, point - com_a)
    vb = np.r_[velocity[3:5], 0.0] + velocity[5] * np.cross(axis, point - com_b)
    relative = va - vb
    area = dict(
        points=point[None],
        normals=np.array([[1.0, 0.0, 0.0]]),
        weights=np.array([0.25]),
        stiffness=np.array([25.0]),
        initial_gap=np.array([-0.02]),
        jac=(ja - jb)[None],
    )
    q, forces = area_response(area, velocity, 0.01)
    fn = max(0.0, -25.0 * (-0.02 + 0.01 * relative[0]))
    tangent = np.array([0.0, relative[1], relative[2]])
    ft = -0.7 * fn * tangent / np.sqrt(tangent @ tangent + 0.01**2)
    expected = np.array([fn, 0.0, 0.0]) + ft
    np.testing.assert_allclose(forces[0], expected, rtol=1e-13, atol=1e-14)
    qa = np.r_[expected[:2], np.cross(point - com_a, expected)[2]]
    qb = np.r_[-expected[:2], np.cross(point - com_b, -expected)[2]]
    np.testing.assert_allclose(q, np.r_[qa, qb], rtol=1e-13, atol=1e-14)
    assert q @ velocity == pytest.approx(expected @ va + (-expected) @ vb, abs=1e-14)
    assert ft @ tangent < 0 and np.linalg.norm(ft) < 0.7 * fn
    # Swapping physical A/B reverses force, while its generalized action in the
    # unchanged full DOF space remains the same.
    swapped = dict(area, jac=-area["jac"], normals=-area["normals"])
    q_swap, f_swap = area_response(swapped, velocity, 0.01)
    np.testing.assert_allclose(f_swap, -forces, atol=1e-14)
    np.testing.assert_allclose(q_swap, q, atol=1e-14)
