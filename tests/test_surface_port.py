"""Independent load/power checks for barycentric surface attachments."""

import importlib

import numpy as np
import pytest

from phymeo import ParticleModel, ParticleState, SolverFailure
from phymeo.coupling.ports import ParticlePointPort


def api():
    return importlib.import_module("phymeo.coupling.surface_port")


def particles(indices=None, inverse_mass=None):
    inverse_mass = np.array([1.0, 2.0, 0.0]) if inverse_mass is None else inverse_mass
    n = len(inverse_mass)
    state = ParticleState(np.arange(n * 3, dtype=float).reshape(n, 3), np.zeros((n, 3)))
    state.velocity[:] = np.arange(n * 3, dtype=float).reshape(n, 3) / 10
    state.velocity[inverse_mass == 0] = 0
    return ParticlePointPort(
        ParticleModel(inverse_mass, np.zeros(3)),
        state,
        np.arange(n) if indices is None else indices,
        domain="cloth",
    )


def weights():
    return np.array([[0.5, 0.5, 0.0], [0.0, 0.25, 0.75]])


def test_barycentric_positions_and_full_response():
    base = particles()
    port = api().WeightedPointPort(base, weights())
    np.testing.assert_array_equal(port.positions(), [[1.5, 2.5, 3.5], [5.25, 6.25, 7.25]])
    p = np.array([[2.0, -1, 3], [0.5, 4, -2]])
    # Independently evaluated B diag(1,2,0) B.T, including the shared node.
    expected = np.array([[0.75, 0.25], [0.25, 0.125]]) @ p
    np.testing.assert_allclose(port.response(p), expected, atol=1e-12)
    after = port.apply_impulses(p)
    delta = np.array([p[0] / 2, p[0] + p[1] / 2, np.zeros(3)])
    np.testing.assert_allclose(after.velocity - base.snapshot().velocity, delta, atol=1e-12)
    np.testing.assert_array_equal(after.position, base.snapshot().position)
    assert port.domain == "cloth"
    assert port.dtype == np.dtype(np.float64)


def test_duplicate_base_points_fixed_nodes_and_power_duality():
    base = particles(np.array([0, 0, 1, 2]))
    b = np.array([[0.25, 0.25, 0.5, 0.0], [0.0, 0.25, 0.25, 0.5]])
    port = api().WeightedPointPort(base, b)
    p = np.array([[1.0, 2, 3], [-2.0, 1, 0.25]])
    f = np.zeros((3, 3))
    for row in range(2):
        for col, node in enumerate([0, 0, 1, 2]):
            f[node] += b[row, col] * p[row]
    state = base.snapshot()
    after = port.apply_impulses(p)
    np.testing.assert_allclose(after.velocity, state.velocity + f * [[1], [2], [0]])
    assert np.sum(p * port.velocities()) == pytest.approx(
        np.sum(f * state.velocity), abs=1e-12, rel=1e-12
    )
    u = np.array([[0.3, -1, 2], [1.0, 0, -0.2]])
    assert np.sum(u * port.response(p)) == pytest.approx(
        np.sum(p * port.response(u)), abs=1e-12, rel=1e-12
    )
    assert np.sum(p * port.response(p)) >= 0


def test_weighted_port_owns_inputs_queries_and_candidates():
    base, b = particles(), weights()
    port = api().WeightedPointPort(base, b)
    b[:] = 0
    port.weights[:] = 0
    p = np.ones((2, 3))
    before = port.positions()
    port.positions()[:] = 123
    port.velocities()[:] = 123
    port.response(p)[:] = 123
    candidate = port.apply_impulses(p)
    candidate.position[:] = 123
    candidate.velocity[:] = 123
    port.snapshot().velocity[:] = 123
    np.testing.assert_array_equal(port.positions(), before)
    np.testing.assert_array_equal(port.weights, weights())
    np.testing.assert_array_equal(p, np.ones((2, 3)))
    np.testing.assert_array_equal(port.snapshot().velocity, base.snapshot().velocity)
    bad = port.snapshot()
    bad.position[0, 0] += 1
    with pytest.raises(ValueError, match="configuration"):
        port.validate_candidate(bad)


@pytest.mark.parametrize(
    "b",
    [
        np.ones((2, 3), np.float32),
        np.ones((2, 3), int),
        np.ones(3),
        np.ones((1, 2)),
        np.array([[np.nan, 0, 1]]),
        np.array([[-0.1, 0.1, 1]]),
        np.array([[0.5, 0.5, 0.1]]),
        np.array([[1.0, 0.0, np.inf]]),
    ],
)
def test_invalid_weights_are_rejected(b):
    with pytest.raises((ValueError, TypeError)):
        api().WeightedPointPort(particles(), b)


@pytest.mark.parametrize("method", ["response", "apply_impulses"])
@pytest.mark.parametrize(
    "p",
    [
        np.ones((2, 3), np.float32),
        np.ones((1, 3)),
        np.ones((2, 2)),
        np.full((2, 3), np.nan),
        np.ones((2, 3), int),
    ],
)
def test_invalid_impulses_leave_snapshot_unchanged(method, p):
    port = api().WeightedPointPort(particles(), weights())
    before = port.snapshot()
    with pytest.raises((ValueError, TypeError)):
        getattr(port, method)(p)
    np.testing.assert_array_equal(port.snapshot().velocity, before.velocity)


@pytest.mark.parametrize("n", [0, 3])
def test_empty_mapping_and_empty_base(n):
    base = particles(inverse_mass=np.ones(n))
    port = api().WeightedPointPort(base, np.empty((0, n)))
    assert port.positions().shape == port.velocities().shape == (0, 3)
    assert port.response(np.empty((0, 3))).shape == (0, 3)
    np.testing.assert_array_equal(
        port.apply_impulses(np.empty((0, 3))).velocity, base.snapshot().velocity
    )
    if n == 0:
        with pytest.raises(ValueError):
            api().WeightedPointPort(base, np.empty((1, 0)))


def test_fp32_base_is_not_silently_promoted():
    base = ParticlePointPort(
        ParticleModel(np.ones(1, np.float32), np.zeros(3, np.float32)),
        ParticleState(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32)),
        [0],
    )
    with pytest.raises((ValueError, TypeError), match="64"):
        api().WeightedPointPort(base, np.ones((1, 1)))


@pytest.mark.parametrize("method", ["response", "apply_impulses"])
def test_finite_load_sum_overflow_is_a_solver_failure(method):
    base = particles(inverse_mass=np.ones(1))
    port = api().WeightedPointPort(base, np.ones((2, 1)))
    p = np.full((2, 3), 1e308)
    with pytest.raises(SolverFailure):
        getattr(port, method)(p)
    np.testing.assert_array_equal(base.snapshot().velocity, [[0, 0.1, 0.2]])
    np.testing.assert_array_equal(p, np.full((2, 3), 1e308))


def test_bottom_solver_failure_propagates_and_candidate_is_validated():
    class Broken(ParticlePointPort):
        def response(self, impulses):
            raise SolverFailure("base failure")

        def apply_impulses(self, impulses):
            candidate = self.snapshot()
            candidate.position[0, 0] += 1
            return candidate

    base = Broken(
        ParticleModel(np.ones(1), np.zeros(3)),
        ParticleState(np.zeros((1, 3)), np.zeros((1, 3))),
        [0],
    )
    port = api().WeightedPointPort(base, np.ones((1, 1)))
    with pytest.raises(SolverFailure, match="base failure"):
        port.response(np.ones((1, 3)))
    with pytest.raises(ValueError, match="configuration"):
        port.apply_impulses(np.ones((1, 3)))
