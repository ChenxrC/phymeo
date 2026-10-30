"""Independent signed-hinge expectations; authored for the serial test queue."""

from math import atan2, pi

import numpy as np
import pytest

from phymeo.constraints.bending import (
    BendingConstraints,
    bending_angle_gradient,
    bending_energy,
    bending_residual,
    project_bending,
)
from phymeo.core.capabilities import SolverFailure


def flat_hinge():
    return np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, -1, 0]])


def one_hinge(rest=0.0, compliance=2.0):
    return BendingConstraints(
        np.array([[0, 1, 2, 3]], dtype=np.int64),
        np.array([rest], dtype=np.float64),
        np.array([compliance], dtype=np.float64),
    )


def independent_angle(nodes):
    """Scalar triple-product identity, independent of production face normals."""
    a, b, c, d = nodes
    e, u, v = b - a, c - a, d - a
    sine_numerator = -np.linalg.norm(e) * np.linalg.det(np.stack((e, u, v)))
    cosine_numerator = np.dot(e, u) * np.dot(e, v) - np.dot(e, e) * np.dot(u, v)
    return atan2(float(sine_numerator), float(cosine_numerator))


def centered_gradient(function, nodes, step=2e-6):
    result = np.empty_like(nodes)
    for node in range(len(nodes)):
        for axis in range(3):
            left, right = nodes.copy(), nodes.copy()
            left[node, axis] -= step
            right[node, axis] += step
            result[node, axis] = (function(right) - function(left)) / (2 * step)
    return result


def sheet_case():
    x = np.array(
        [[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, -1, -0.4], [1.1, 1.2, 0.5]]
    )
    constraints = BendingConstraints(
        np.array([[0, 1, 2, 3], [1, 2, 0, 4]], dtype=np.int64),
        np.array([-0.1, 0.2]),
        np.array([0.04, 0.07]),
    )
    return x, constraints


def test_flat_hinge_has_literal_gradient_zero_energy_and_zero_correction():
    x = flat_hinge()
    theta, gradient = bending_angle_gradient(x)
    assert theta == 0.0
    np.testing.assert_array_equal(gradient, [[0, 0, 2], [0, 0, 0], [0, 0, -1], [0, 0, -1]])
    constraints, lam = one_hinge(), np.zeros(1)
    assert bending_energy(x, constraints) == 0.0
    before = x.copy()
    traces = project_bending(x, np.ones(4), constraints, np.ones(1), lam)
    np.testing.assert_array_equal(x, before)
    np.testing.assert_array_equal(lam, [0])
    assert len(traces) == 1
    assert traces[0].delta_lambda == 0.0
    assert bending_residual(x, constraints, np.ones(1), lam) == (0.0, 0.0)


def test_positive_right_angle_and_energy_have_independent_exact_values():
    x = flat_hinge()
    x[3] = [0.0, 0, -1]
    theta, gradient = bending_angle_gradient(x)
    assert theta == pi / 2
    np.testing.assert_array_equal(gradient, [[0, -1, 1], [0, 0, 0], [0, 0, -1], [0, 1, 0]])
    assert bending_energy(x, one_hinge()) == pytest.approx(pi**2 / 16, rel=2e-15)


@pytest.mark.parametrize(
    "order, sign", [([0, 1, 3, 2], -1), ([1, 0, 2, 3], -1), ([1, 0, 3, 2], 1)]
)
def test_face_and_edge_orientation_transform_consistently(order, sign):
    x = flat_hinge()
    x[3] = [0.2, -0.7, -0.5]
    theta, gradient = bending_angle_gradient(x)
    other_theta, other_gradient = bending_angle_gradient(x[order])
    assert other_theta == pytest.approx(sign * theta, abs=2e-15)
    np.testing.assert_allclose(other_gradient, sign * gradient[order], atol=2e-15, rtol=2e-15)


@pytest.mark.parametrize("scale", [0.125, 1.0, 7.0, 1e100, 1e-100])
def test_angle_and_gradient_obey_rigid_motion_and_scale(scale):
    x = np.array([[0.2, -0.1, 0.3], [1.1, 0.4, -0.2], [-0.2, 1.3, 0.4], [0.4, -0.8, -0.7]])
    rotation = np.array([[0.0, -1, 0], [0.0, 0, -1], [1.0, 0, 0]])
    theta, gradient = bending_angle_gradient(x)
    transformed = scale * (x @ rotation.T + np.array([2.0, -3.0, 4.0]))
    actual_theta, actual_gradient = bending_angle_gradient(transformed)
    assert actual_theta == pytest.approx(theta, abs=3e-15)
    np.testing.assert_allclose(
        actual_gradient * scale, gradient @ rotation.T, rtol=2e-14, atol=2e-14
    )


@pytest.mark.parametrize(
    "x",
    [
        flat_hinge(),
        np.array([[0.2, -0.1, 0.3], [1.1, 0.4, -0.2], [-0.2, 1.3, 0.4], [0.4, -0.8, -0.7]]),
    ],
)
def test_gradient_matches_independent_centered_difference_and_rigid_mode_identities(x):
    theta, gradient = bending_angle_gradient(x)
    assert theta == pytest.approx(independent_angle(x), abs=3e-15)
    expected = centered_gradient(independent_angle, x)
    np.testing.assert_allclose(gradient, expected, rtol=3e-8, atol=3e-9)
    np.testing.assert_allclose(gradient.sum(axis=0), 0.0, rtol=0, atol=2e-15)
    np.testing.assert_allclose(np.cross(x, gradient).sum(axis=0), 0.0, rtol=0, atol=3e-15)


def test_two_hinge_energy_gradient_accumulates_shared_node_loads():
    x, constraints = sheet_case()

    def independent_energy(nodes):
        return sum(
            (independent_angle(nodes[row]) - rest) ** 2 / (2 * compliance)
            for row, rest, compliance in zip(
                constraints.indices, constraints.rest_angle, constraints.compliance
            )
        )

    expected = centered_gradient(independent_energy, x)
    assembled = np.zeros_like(x)
    for row, rest, compliance in zip(
        constraints.indices, constraints.rest_angle, constraints.compliance
    ):
        theta, gradient = bending_angle_gradient(x[row])
        assembled[row] += ((theta - rest) / compliance) * gradient
    assert bending_energy(x, constraints) == pytest.approx(independent_energy(x), rel=3e-15)
    np.testing.assert_allclose(assembled, expected, rtol=2e-8, atol=1e-8)


def test_residual_is_constitutive_not_only_angular_error():
    x = flat_hinge()
    x[3] = [0.0, 0, -1]
    constraints = one_hinge(rest=0.5)
    alpha = np.array([2.0])
    lam = np.array([-(pi / 2 - 0.5) / 2])
    residual, error = bending_residual(x, constraints, alpha, lam, reference_angle=0.25)
    assert residual == 0.0
    assert error == pi / 2 - 0.5
    residual, _ = bending_residual(x, constraints, alpha, np.zeros(1), reference_angle=2.0)
    assert residual == (pi / 2 - 0.5) / 2


def test_all_fixed_soft_hinge_changes_only_multiplier():
    x = flat_hinge()
    x[3] = [0.0, 0, -1]
    original, lam = x.copy(), np.array([0.25])
    traces = project_bending(x, np.zeros(4), one_hinge(), np.array([2.0]), lam)
    np.testing.assert_array_equal(x, original)
    assert lam[0] == pytest.approx(-pi / 4, abs=2e-16)
    assert traces[0].denominator == 2.0
    assert bending_residual(x, one_hinge(), np.array([2.0]), lam)[0] <= 3e-16


def test_configuration_owns_inputs_and_rejects_writes():
    indices = np.array([[0, 1, 2, 3]], dtype=np.int32)
    rest, compliance = np.array([0.2]), np.array([0.5])
    constraints = BendingConstraints(indices, rest, compliance)
    indices[:] = 7
    rest[:] = 9
    compliance[:] = 0
    np.testing.assert_array_equal(constraints.indices, [[0, 1, 2, 3]])
    np.testing.assert_array_equal(constraints.rest_angle, [0.2])
    np.testing.assert_array_equal(constraints.compliance, [0.5])
    assert constraints.indices.dtype == np.int64
    for array in (constraints.indices, constraints.rest_angle, constraints.compliance):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 5


@pytest.mark.parametrize("second", [[0, 1, 2, 3], [1, 0, 3, 2], [1, 0, 2, 3], [0, 1, 3, 2]])
def test_duplicate_hinge_rejected_for_every_edge_and_face_orientation(second):
    with pytest.raises(ValueError, match="duplicate"):
        BendingConstraints(np.array([[0, 1, 2, 3], second]), np.zeros(2), np.ones(2))


@pytest.mark.parametrize(
    "indices, rest, compliance, error",
    [
        (np.array([[0, 1, 2]]), np.zeros(1), np.ones(1), ValueError),
        (np.array([[0.0, 1, 2, 3]]), np.zeros(1), np.ones(1), TypeError),
        (np.array([[0, 1, 2, -1]]), np.zeros(1), np.ones(1), ValueError),
        (np.array([[0, 1, 2, 2]]), np.zeros(1), np.ones(1), ValueError),
        (np.array([[0, 1, 2, 2**63]], dtype=np.uint64), np.zeros(1), np.ones(1), ValueError),
        (np.array([[0, 1, 2, 3]]), np.array([pi]), np.ones(1), ValueError),
        (np.array([[0, 1, 2, 3]]), np.array([-pi]), np.ones(1), ValueError),
        (np.array([[0, 1, 2, 3]]), np.array([np.nan]), np.ones(1), ValueError),
        (np.array([[0, 1, 2, 3]]), np.zeros(1, np.float32), np.ones(1), TypeError),
        (np.array([[0, 1, 2, 3]]), np.zeros(2), np.ones(1), ValueError),
        (np.array([[0, 1, 2, 3]]), np.zeros(1), np.zeros(1), ValueError),
        (np.array([[0, 1, 2, 3]]), np.zeros(1), -np.ones(1), ValueError),
        (np.array([[0, 1, 2, 3]]), np.zeros(1), np.array([np.inf]), ValueError),
        (np.array([[0, 1, 2, 3]]), np.zeros(1), np.ones(1, np.float32), TypeError),
    ],
)
def test_invalid_configuration_is_rejected(indices, rest, compliance, error):
    with pytest.raises(error):
        BendingConstraints(indices, rest, compliance)


@pytest.mark.parametrize("reference", [0.0, -1.0, np.nan, np.inf, True, "1"])
def test_invalid_reference_angle_is_rejected(reference):
    with pytest.raises((ValueError, TypeError)):
        bending_residual(flat_hinge(), one_hinge(), np.ones(1), np.zeros(1), reference)


@pytest.mark.parametrize("scale", [1.0, 1e80, 1e-80])
def test_dimensionless_face_admissibility_is_scale_independent(scale):
    x = flat_hinge()
    x[2] = [1.0, 0.5e-10, 0.0]
    with pytest.raises(SolverFailure, match="degenerate|ill-conditioned"):
        bending_angle_gradient(x * scale)
    x[2, 1] = 2e-10
    theta, _ = bending_angle_gradient(x * scale)
    assert theta == 0.0


@pytest.mark.parametrize("case", ["edge", "face", "cut", "overflow", "tiny"])
def test_invalid_or_nonrepresentable_geometry_rejected(case):
    x = flat_hinge()
    if case == "edge":
        x[1] = x[0]
    elif case == "face":
        x[2] = [0.5, 0, 0]
    elif case == "cut":
        x[3] = [0, 1, 0]
    elif case == "overflow":
        x[0, 0], x[1, 0] = -1e308, 1e308
    else:
        x *= 1e-310
    with pytest.raises(SolverFailure):
        bending_angle_gradient(x)


def test_empty_sweep_has_no_trace_and_zero_energy_or_residual():
    constraints = BendingConstraints(np.empty((0, 4), np.int64), np.empty(0), np.empty(0))
    x, lam = np.empty((0, 3)), np.empty(0)
    assert project_bending(x, np.empty(0), constraints, np.empty(0), lam) == ()
    assert bending_energy(x, constraints) == 0.0
    assert bending_residual(x, constraints, np.empty(0), lam) == (0.0, 0.0)


@pytest.mark.parametrize(
    "positions",
    [np.zeros((3, 3)), np.zeros((4, 2)), np.zeros((4, 3), np.float32),
     np.zeros((4, 3), np.int64), np.full((4, 3), np.nan), [[0.0] * 3] * 4],
)
def test_angle_rejects_invalid_position_shape_dtype_and_storage(positions):
    with pytest.raises((TypeError, ValueError)):
        bending_angle_gradient(positions)


def test_evaluation_energy_and_residual_reject_unrepresentable_results():
    x = flat_hinge()
    x[3] = [0.0, 0, -1]
    with pytest.raises(SolverFailure, match="representable"):
        bending_energy(x, one_hinge(compliance=1e-310))
    with pytest.raises(SolverFailure, match="representable"):
        bending_residual(x, one_hinge(), np.array([1e308]), np.array([1e308]))
