"""Actual device-generated XPBD projection records and bitwise opt-in parity."""

import numpy as np
import pytest
from test_material_path import material_case, run_material

from phymeo import DistanceConstraints
from phymeo.backends.tyx_node_math import TyxNodeMath

pytestmark = pytest.mark.tyx


@pytest.mark.parametrize("kind", ["shared", "fixed", "coincident", "none", "empty"])
def test_actual_material_path_is_owned_and_bitwise_transparent(tyx_mode, kind):
    plain, old, old_count, *_ = run_material(tyx_mode, trace=False, kind=kind)
    final, diag, count, *_ = run_material(tyx_mode, kind=kind)
    assert count == old_count
    np.testing.assert_array_equal(final.position, plain.position)
    np.testing.assert_array_equal(final.velocity, plain.velocity)
    np.testing.assert_array_equal(diag["distance_multipliers"], old["distance_multipliers"])
    assert "material_path" not in old
    assert diag["execution"]["completed_launches"] == old["execution"]["completed_launches"] > 0
    path = diag["material_path"]
    assert len(path["frames"]) == count
    for frame in path["frames"]:
        assert frame.shape == (0 if kind in ("none", "empty") else 2, 18)
        assert not frame.flags.writeable
        np.testing.assert_array_equal(frame, frame.astype(np.float32).astype(np.float64))
    if tyx_mode == "cuda" and kind not in ("none", "empty"):
        assert diag["execution"]["downloads"] == old["execution"]["downloads"] + count


def test_actual_bounded_single_edge_path(tyx_mode):
    x = np.array([[0.0, 0, 0], [2.0, 0, 0]])
    constraints = DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.zeros(1))
    target = np.full((1, 18), np.nan)
    with TyxNodeMath(tyx_mode) as engine:
        plain, lam0 = engine.project_distances(x, np.ones(2), constraints, np.zeros(1), np.zeros(1))
        actual, lam1 = engine.project_distances(
            x, np.ones(2), constraints, np.zeros(1), np.zeros(1), trace=target
        )
        assert engine.stats()["completed_launches"] == 2
    np.testing.assert_array_equal(actual, plain)
    np.testing.assert_array_equal(lam0, lam1)
    np.testing.assert_array_equal(
        target[0], [0, 0, 0, 2, 0, 0, 0.5, 0, 0, 1.5, 0, 0, 0, -0.5, -1, 0, 0, -0.5]
    )
    np.testing.assert_array_equal(x, [[0, 0, 0], [2, 0, 0]])


def test_actual_zero_denominator_and_coincident_path(tyx_mode):
    constraints = DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.zeros(1))
    target = np.full((1, 18), np.nan)
    with TyxNodeMath(tyx_mode) as engine:
        x, lam = engine.project_distances(
            np.zeros((2, 3)), np.zeros(2), constraints, np.zeros(1), np.array([0.25]), trace=target
        )
    np.testing.assert_array_equal(target[0], [0] * 12 + [0.25, 0.25, 1, 0, 0, 0])
    np.testing.assert_array_equal(x, np.zeros((2, 3)))
    np.testing.assert_array_equal(lam, [0.25])


def test_actual_invalid_trace_is_rejected_before_launch(tyx_mode):
    model, state, constraints = material_case()
    with TyxNodeMath(tyx_mode) as engine:
        for trace in (np.zeros((2, 17)), np.zeros((2, 18), np.float32), np.zeros((2, 36))[:, ::2]):
            with pytest.raises((TypeError, ValueError), match="trace"):
                engine.project_distances(
                    state.position,
                    model.inverse_mass,
                    constraints,
                    np.zeros(2),
                    np.zeros(2),
                    trace=trace,
                )
        assert engine.stats()["completed_launches"] == 0


def test_actual_incremental_path_and_energy_accounting(tyx_mode):
    from test_material_path import audit_material_path

    audit_material_path(tyx_mode)


@pytest.mark.parametrize("h", [0.1, 0.05, 0.025])
def test_actual_rotating_edge_keeps_original_discrete_loss(tyx_mode, h):
    from test_material_path import check_rotating_edge

    check_rotating_edge(tyx_mode, h)


@pytest.mark.parametrize("failure", ["velocity", "close"])
def test_actual_traced_late_failure_and_retry(tyx_mode, failure):
    from test_material_path import check_scene_failure

    check_scene_failure(tyx_mode, failure)
