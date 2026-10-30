"""Projection path reconstruction and all-or-nothing publication tests.

All expectations below are authored for later serial execution. The independent
reference uses centered derivatives of a scalar triple-product angle identity.
"""

from dataclasses import FrozenInstanceError, replace
from math import atan2, pi

import numpy as np
import pytest

from phymeo.constraints.bending import BendingConstraints, project_bending
from phymeo.core.capabilities import SolverFailure


def case():
    x = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, -1, -0.4], [1.1, 1.2, 0.5]])
    constraints = BendingConstraints(
        np.array([[0, 1, 2, 3], [1, 2, 0, 4]], dtype=np.int64),
        np.array([-0.1, 0.2]),
        np.array([0.04, 0.07]),
    )
    return (
        x,
        np.array([1.0, 0.5, 2.0, 0.8, 1.5]),
        constraints,
        np.array([0.8, 1.1]),
        np.array([0.03, -0.02]),
    )


def angle(nodes):
    a, b, c, d = nodes
    e, u, v = b - a, c - a, d - a
    return atan2(
        -np.linalg.norm(e) * np.linalg.det(np.stack((e, u, v))),
        np.dot(e, u) * np.dot(e, v) - np.dot(e, e) * np.dot(u, v),
    )


def numerical_gradient(nodes):
    gradient = np.empty_like(nodes)
    step = 2e-6
    for index in range(4):
        for axis in range(3):
            minus, plus = nodes.copy(), nodes.copy()
            minus[index, axis] -= step
            plus[index, axis] += step
            gradient[index, axis] = (angle(plus) - angle(minus)) / (2 * step)
    return gradient


def test_two_shared_hinge_sweep_matches_independently_accumulated_reference():
    x, w, constraints, alpha, lam = case()
    expected_x, expected_lambda = x.copy(), lam.copy()
    for k, row in enumerate(constraints.indices):
        nodes = expected_x[row]
        gradient = numerical_gradient(nodes)
        c = angle(nodes) - constraints.rest_angle[k]
        denominator = np.sum(w[row, None] * gradient**2) + alpha[k]
        dl = (-c - alpha[k] * expected_lambda[k]) / denominator
        expected_x[row] += w[row, None] * dl * gradient
        expected_lambda[k] += dl
    traces = project_bending(x, w, constraints, alpha, lam)
    assert len(traces) == 2
    np.testing.assert_allclose(x, expected_x, atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(lam, expected_lambda, atol=2e-9, rtol=2e-9)


def test_complete_trace_reconstructs_actual_ordered_updates_and_evaluation_states():
    x, w, constraints, alpha, lam = case()
    replay_x, replay_lambda = x.copy(), lam.copy()
    traces = project_bending(x, w, constraints, alpha, lam)
    assert isinstance(traces, tuple)
    assert [trace.element_index for trace in traces] == [0, 1]
    for k, trace in enumerate(traces):
        row = constraints.indices[k]
        np.testing.assert_array_equal(trace.node_indices, row)
        np.testing.assert_array_equal(trace.positions_before, replay_x[row])
        np.testing.assert_allclose(
            trace.gradient, numerical_gradient(replay_x[row]), atol=3e-9, rtol=3e-8
        )
        assert trace.theta_before == pytest.approx(angle(replay_x[row]), abs=3e-15)
        assert trace.constraint_before == trace.theta_before - constraints.rest_angle[k]
        denominator = np.sum(w[row, None] * trace.gradient**2) + alpha[k]
        assert trace.denominator == pytest.approx(denominator, rel=3e-15)
        assert trace.alpha == alpha[k]
        assert trace.lambda_before == replay_lambda[k]
        expected_dl = (-trace.constraint_before - alpha[k] * replay_lambda[k]) / denominator
        assert trace.delta_lambda == pytest.approx(expected_dl, abs=3e-16)
        replay_x[row] += (w[row] * trace.delta_lambda)[:, None] * trace.gradient
        replay_lambda[k] += trace.delta_lambda
        np.testing.assert_array_equal(trace.positions_after, replay_x[row])
        assert trace.lambda_after == replay_lambda[k]
        assert trace.theta_after == pytest.approx(angle(replay_x[row]), abs=3e-15)
        assert trace.constraint_after == trace.theta_after - constraints.rest_angle[k]
        assert trace.branch_status == "smooth"
    np.testing.assert_array_equal(replay_x, x)
    np.testing.assert_array_equal(replay_lambda, lam)


def test_late_degenerate_shared_hinge_leaves_entire_sweep_unchanged():
    x, w, constraints, alpha, lam = case()
    x[4] = [0.5, 0.5, 0]
    w[1:3] = 0
    before, old_lambda = x.copy(), lam.copy()
    first = BendingConstraints(
        constraints.indices[:1], constraints.rest_angle[:1], constraints.compliance[:1]
    )
    first_trial = x.copy()
    project_bending(first_trial, w, first, alpha[:1].copy(), lam[:1].copy())
    assert not np.array_equal(first_trial, before)
    with pytest.raises(SolverFailure, match="degenerate|ill-conditioned"):
        project_bending(x, w, constraints, alpha, lam)
    np.testing.assert_array_equal(x, before)
    np.testing.assert_array_equal(lam, old_lambda)


def test_late_arithmetic_failure_leaves_positions_and_all_multipliers_unchanged():
    x, w, constraints, alpha, lam = case()
    alpha[1], lam[1] = 1e308, 1e308
    before, old_lambda = x.copy(), lam.copy()
    with pytest.raises(SolverFailure, match="representable"):
        project_bending(x, w, constraints, alpha, lam)
    np.testing.assert_array_equal(x, before)
    np.testing.assert_array_equal(lam, old_lambda)


def single_case(nodes, rest=0.0, multiplier=0.0):
    constraints = BendingConstraints(np.array([[0, 1, 2, 3]]), np.array([rest]), np.ones(1))
    return nodes, np.array([0.0, 0, 0, 1]), constraints, np.ones(1), np.array([multiplier])


def test_explicit_atan2_branch_jump_rejects_even_when_both_endpoints_are_admissible():
    nodes = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 1, -0.1]])
    args = single_case(nodes, multiplier=-10.0)
    before, old_lambda = nodes.copy(), args[-1].copy()
    # For this unit edge g_d=(0,0.1,1)/1.01, derived directly by rotating d.
    gd = np.array([0.0, 0.1, 1.0]) / 1.01
    trial = nodes.copy()
    trial[3] += ((10 - angle(nodes)) / (np.dot(gd, gd) + 1)) * gd
    assert abs(angle(nodes)) < pi and abs(angle(trial)) < pi
    assert abs(angle(trial) - angle(nodes)) >= pi
    with pytest.raises(SolverFailure, match="branch jump"):
        project_bending(*args)
    np.testing.assert_array_equal(nodes, before)
    np.testing.assert_array_equal(args[-1], old_lambda)


@pytest.mark.parametrize("when", ["before", "after"])
def test_angular_error_domain_rejection_is_transactional(when):
    nodes = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, -1, 0]])
    if when == "before":
        nodes[3] = [0, 0, -1]
        args = single_case(nodes, rest=-2.0)
    else:
        args = single_case(nodes, rest=3.0, multiplier=10.0)
    before, old_lambda = nodes.copy(), args[-1].copy()
    with pytest.raises(SolverFailure, match="angular error"):
        project_bending(*args)
    np.testing.assert_array_equal(nodes, before)
    np.testing.assert_array_equal(args[-1], old_lambda)


def test_trace_owns_all_arrays_across_input_mutation_and_another_sweep():
    x, w, constraints, alpha, lam = case()
    traces = project_bending(x, w, constraints, alpha, lam)
    snapshots = [
        tuple(
            getattr(trace, name).copy()
            for name in ("node_indices", "positions_before", "positions_after", "gradient")
        )
        for trace in traces
    ]
    project_bending(x, w, constraints, alpha, lam)
    x[:] = 13
    lam[:] = 7
    for trace, saved in zip(traces, snapshots):
        for name, original in zip(
            ("node_indices", "positions_before", "positions_after", "gradient"), saved
        ):
            array = getattr(trace, name)
            assert not array.flags.writeable
            assert not np.shares_memory(array, x)
            np.testing.assert_array_equal(array, original)
            with pytest.raises(ValueError):
                array.flat[0] = 9
        with pytest.raises(FrozenInstanceError):
            trace.branch_status = "other"


@pytest.mark.parametrize(
    "replacement",
    [
        {"element_index": -1},
        {"element_index": True},
        {"node_indices": np.array([0, 1, 2, 2])},
        {"node_indices": np.array([0.0, 1, 2, 3])},
        {"positions_before": np.zeros((3, 3))},
        {"positions_after": np.zeros((4, 3), np.float32)},
        {"gradient": np.full((4, 3), np.nan)},
        {"theta_before": pi},
        {"constraint_after": -pi},
        {"lambda_after": np.inf},
        {"alpha": 0.0},
        {"denominator": 0.0},
        {"branch_status": "safe"},
    ],
)
def test_malformed_trace_rejected_without_claiming_valid_evidence(replacement):
    trace = project_bending(*case())[0]
    with pytest.raises((ValueError, TypeError)):
        replace(trace, **replacement)


@pytest.mark.parametrize(
    "bad",
    [
        "position_dtype",
        "position_shape",
        "position_nan",
        "mass_dtype",
        "mass_shape",
        "negative_mass",
        "mass_inf",
        "alpha_dtype",
        "alpha_shape",
        "alpha_zero",
        "alpha_nan",
        "lambda_dtype",
        "lambda_shape",
        "lambda_inf",
        "index_bound",
        "position_readonly",
        "lambda_readonly",
        "alias_alpha_lambda",
        "alias_position_mass",
    ],
)
def test_invalid_sweep_inputs_never_publish_mutations(bad):
    x, w, constraints, alpha, lam = case()
    if bad == "position_dtype":
        x = x.astype(np.float32)
    elif bad == "position_shape":
        x = x[:, :2].copy()
    elif bad == "position_nan":
        x[0, 0] = np.nan
    elif bad == "mass_dtype":
        w = w.astype(np.float32)
    elif bad == "mass_shape":
        w = w[:-1]
    elif bad == "negative_mass":
        w[0] = -1
    elif bad == "mass_inf":
        w[0] = np.inf
    elif bad == "alpha_dtype":
        alpha = alpha.astype(np.float32)
    elif bad == "alpha_shape":
        alpha = alpha[:1]
    elif bad == "alpha_zero":
        alpha[0] = 0
    elif bad == "alpha_nan":
        alpha[0] = np.nan
    elif bad == "lambda_dtype":
        lam = lam.astype(np.float32)
    elif bad == "lambda_shape":
        lam = lam[:1]
    elif bad == "lambda_inf":
        lam[0] = np.inf
    elif bad == "index_bound":
        constraints = BendingConstraints(np.array([[0, 1, 2, 8]]), np.zeros(1), np.ones(1))
        alpha, lam = alpha[:1], lam[:1]
    elif bad == "position_readonly":
        x.flags.writeable = False
    elif bad == "lambda_readonly":
        lam.flags.writeable = False
    elif bad == "alias_alpha_lambda":
        alpha = lam = np.ones(2)
    elif bad == "alias_position_mass":
        w = x[:, 0]
    before, old_lambda = x.copy(), lam.copy()
    with pytest.raises((ValueError, TypeError)):
        project_bending(x, w, constraints, alpha, lam)
    np.testing.assert_array_equal(x, before)
    np.testing.assert_array_equal(lam, old_lambda)


def test_disjoint_strided_publication_targets_are_supported():
    original, w, constraints, alpha, lam = case()
    backing, lambda_backing = np.zeros((10, 6)), np.zeros(4)
    x, target_lambda = backing[::2, ::2], lambda_backing[::2]
    x[:], target_lambda[:] = original, lam
    expected_x, expected_lambda = original.copy(), lam.copy()
    project_bending(expected_x, w, constraints, alpha, expected_lambda)
    project_bending(x, w, constraints, alpha, target_lambda)
    np.testing.assert_array_equal(x, expected_x)
    np.testing.assert_array_equal(target_lambda, expected_lambda)
    np.testing.assert_array_equal(backing[1::2], 0)
    np.testing.assert_array_equal(backing[:, 1::2], 0)
    np.testing.assert_array_equal(lambda_backing[1::2], 0)


def test_self_overlapping_publication_target_is_rejected():
    backing = np.zeros(3)
    x = np.lib.stride_tricks.as_strided(backing, shape=(4, 3), strides=(0, 8))
    args = single_case(x)
    with pytest.raises(ValueError, match="overlapping"):
        project_bending(*args)
    np.testing.assert_array_equal(backing, 0)
    np.testing.assert_array_equal(args[-1], 0)
