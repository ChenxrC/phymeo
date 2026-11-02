"""Interior material coordinates inherit the two represented full-thickness endpoints."""

import numpy as np
import pytest

import phymeo as ph
from phymeo.coupling.material_column import TriangleColumnMap
from phymeo.coupling.offset_port import TriangleOffsetPointPort
from phymeo.coupling.ports import ParticlePointPort


def inputs(z=0.25):
    x = np.array([[0.0, 0.0, z], [1.0, 0.0, z], [0.0, 1.0, z]])
    v = np.array([[0.0, 0.0, 0.25], [0.0, 0.5, 1.0], [1.0, 0.0, -0.25]])
    state = ph.ParticleState(x, v)
    model = ph.ParticleModel(np.ones(3), np.zeros(3))
    ids = np.array([[0, 1, 2]])
    weights = np.array([[0.5, 0.25, 0.25]])
    return ParticlePointPort(model, state, np.arange(3), domain="sheet"), ids, weights


def test_fraction_close_to_middle_preserves_director_map_without_a_relative_tiny_height_gate():
    base, ids, w = inputs()
    rho = np.array([1 / 64])
    fraction = np.array([2.0**-54])
    with pytest.raises(ph.SolverFailure, match="not representable"):
        TriangleOffsetPointPort(base, ids, w, rho * fraction)
    column = TriangleColumnMap(base, ids, w, rho)
    x, v, j = column.sample(fraction)
    np.testing.assert_allclose(x, [[0.25, 0.25, 0.25]], rtol=0, atol=np.finfo(float).eps)
    np.testing.assert_allclose(v, np.einsum("kaij,kaj->ki", j, base.velocities()[ids]), atol=2e-15)
    np.testing.assert_allclose(j.sum(axis=1), np.eye(3)[None], atol=2e-15)
    # For this flat face, d n_x / d x_{1,z}=-1. The sub-ulp height still contributes to J.
    assert j[0, 1, 0, 2] == pytest.approx(-rho[0] * fraction[0], rel=1e-10, abs=0.0)


def test_endpoints_retain_original_checked_port_and_bad_fractions_reject():
    base, ids, w = inputs()
    rho = np.array([0.125])
    column = TriangleColumnMap(base, ids, w, rho)
    for a in [-1.0, 1.0]:
        port = TriangleOffsetPointPort(base, ids, w, a * rho)
        x, v, j = column.sample(np.array([a]))
        np.testing.assert_array_equal(x, port.positions())
        np.testing.assert_array_equal(j, port.jacobians)
        np.testing.assert_allclose(v, port.velocities())
    with pytest.raises(ValueError):
        column.sample(np.array([1.01]))
    base, ids, w = inputs(2.0**56)
    with pytest.raises(ph.SolverFailure, match="not representable"):
        TriangleColumnMap(base, ids, w, rho)
