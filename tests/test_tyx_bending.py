"""Actual new native boundary: no host fallback, actual 45-field device records."""

from math import atan2, sqrt

import numpy as np
import pytest

from phymeo.backends.tyx_bending import TyxBendingMath
from phymeo.constraints.bending import BendingConstraints
from phymeo.core.capabilities import SolverFailure
from test_bending_bounds import flat_nodes

pytestmark = pytest.mark.tyx


def test_actual_bending_first_sweep_and_launch_receipt(tyx_mode):
    x = flat_nodes()
    constraints = BendingConstraints(np.array([[0, 1, 2, 3]]), np.array([0.25]), np.array([0.125]))
    with TyxBendingMath(tyx_mode) as engine:
        after, multipliers, trace = engine.project_bending(x, np.ones(4), constraints, np.array([8.]), np.zeros(1))
        assert engine.stats()["completed_launches"] == 1
        assert trace.launches == trace.completed_launches == 1
        assert trace.rows.shape == (1, 45) and trace.status.tolist() == [0]
        assert multipliers[0] == pytest.approx(1/56, abs=2e-8)
        expected = x.copy()
        expected[:, 2] = (1/28, 0., -1/56, -1/56)
        np.testing.assert_allclose(after, expected, rtol=0, atol=2e-7)
        theta = atan2(6/56*sqrt(1+4/56**2), 1-5/56**2)
        assert trace.rows[0, 37] == pytest.approx(theta, abs=2e-6)
        np.testing.assert_array_equal(after, after.astype(np.float32).astype(np.float64))
        if tyx_mode == "cuda":
            assert engine.stats()["uploads"] > 0 and engine.stats()["downloads"] > 0
    np.testing.assert_array_equal(x, flat_nodes())


def test_native_path_cannot_call_host_bending_implementation(tyx_mode, monkeypatch):
    import phymeo.constraints.bending as host
    def forbidden(*args, **kwargs):
        raise AssertionError("host projection fallback forbidden")
    monkeypatch.setattr(host, "project_bending", forbidden)
    monkeypatch.setattr(host, "bending_angle_gradient", forbidden)
    law = BendingConstraints(np.array([[0, 1, 2, 3]]), np.array([0.25]), np.array([0.125]))
    with TyxBendingMath(tyx_mode) as engine:
        _, _, batch = engine.project_bending(flat_nodes(), np.ones(4), law, np.array([8.]), np.zeros(1))
        assert batch.completed_launches == 1


def test_native_late_row_failure_preserves_all_caller_arrays(tyx_mode):
    x = np.concatenate((flat_nodes(), flat_nodes()+[4., 0., 0.]))
    x[6] = x[4]
    law = BendingConstraints(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]), np.full(2, 0.25), np.full(2, 0.125))
    initial, multipliers = x.copy(), np.zeros(2)
    with TyxBendingMath(tyx_mode) as engine:
        with pytest.raises(SolverFailure, match="row"):
            engine.project_bending(x, np.ones(8), law, np.full(2, 8.), multipliers)
        assert engine.stats()["completed_launches"] == 1
    np.testing.assert_array_equal(x, initial)
    np.testing.assert_array_equal(multipliers, np.zeros(2))


def test_native_conversion_induced_degeneracy_and_cut_do_not_publish(tyx_mode):
    law = BendingConstraints(np.array([[0, 1, 2, 3]]), np.zeros(1), np.ones(1))
    with TyxBendingMath(tyx_mode) as engine:
        for nodes in (flat_nodes()+2**26, np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 1., 0.]])):
            before = nodes.copy()
            with pytest.raises(SolverFailure):
                engine.project_bending(nodes, np.ones(4), law, np.ones(1), np.zeros(1))
            np.testing.assert_array_equal(nodes, before)


def test_empty_native_bending_is_explicit_and_does_not_invent_launches(tyx_mode):
    law = BendingConstraints(np.empty((0, 4), np.int64), np.zeros(0), np.zeros(0))
    with TyxBendingMath(tyx_mode) as engine:
        x, multipliers, batch = engine.project_bending(flat_nodes(), np.ones(4), law, np.zeros(0), np.zeros(0))
        assert engine.stats()["launches"] == 0
        assert batch.rows.shape == (0, 45) and not len(multipliers)
        np.testing.assert_array_equal(x, flat_nodes())
