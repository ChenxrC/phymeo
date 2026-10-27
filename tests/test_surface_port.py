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


def test_pair_shared_node_cross_terms_and_single_candidate():
    class Counted:
        def __init__(self, base):
            self.base, self.calls = base, 0

        def __getattr__(self, name):
            return getattr(self.base, name)

        def apply_impulses(self, impulses):
            self.calls += 1
            return self.base.apply_impulses(impulses)

    base = Counted(particles(np.array([0, 0, 1, 2]), np.array([1.0, 2, 4])))
    b = np.array([[0.25, 0.25, 0.5, 0], [0, 0, 0.5, 0.5], [0.5, 0.5, 0, 0]])
    port = api().WeightedPointPort(base, b)
    pair = api().PointPairOperator(port, np.array([[0, 1], [1, 2]]))
    # Combined signed weights on the three physical nodes, derived by hand.
    a = np.array([[0.5, 0, -0.5], [-1, 0.5, 0.5]])
    p = np.array([[2.0, -1, 3], [0.5, 4, -2]])
    before = base.snapshot()
    response = a @ np.diag([1.0, 2, 4]) @ a.T @ p
    np.testing.assert_allclose(pair.relative_positions(), a @ before.position, atol=1e-12)
    np.testing.assert_allclose(pair.relative_velocities(), a @ before.velocity, atol=1e-12)
    np.testing.assert_allclose(pair.response(p), response, atol=1e-12)
    candidate = pair.apply_impulses(p)
    assert base.calls == 1
    delta = candidate.velocity - before.velocity
    np.testing.assert_allclose(delta, np.diag([1.0, 2, 4]) @ a.T @ p, atol=1e-12)
    np.testing.assert_allclose(np.sum(delta / [[1], [2], [4]], axis=0), 0, atol=1e-12)
    np.testing.assert_array_equal(candidate.position, before.position)
    np.testing.assert_array_equal(base.snapshot().velocity, before.velocity)
    assert not hasattr(pair, "positions")
    assert pair.domain == "cloth" and pair.dtype == np.dtype(np.float64)


@pytest.mark.parametrize("same", [False, True])
def test_pair_zero_net_load_reversed_or_identical_endpoints(same):
    port = particles(inverse_mass=np.array([1.0, 2, 4]))
    pairs = np.array([[1, 1], [1, 1]]) if same else np.array([[0, 1], [1, 0]])
    pair = api().PointPairOperator(port, pairs)
    p = np.ones((2, 3))
    np.testing.assert_array_equal(pair.response(p), np.zeros((2, 3)))
    np.testing.assert_array_equal(pair.apply_impulses(p).velocity, port.snapshot().velocity)
    if same:
        np.testing.assert_array_equal(pair.relative_positions(), np.zeros((2, 3)))
        np.testing.assert_array_equal(pair.relative_velocities(), np.zeros((2, 3)))


def test_pair_repeated_loads_accumulate_and_cancellation_is_retained():
    port = particles(inverse_mass=np.array([1.0, 2, 4]))
    pair = api().PointPairOperator(port, np.array([[0, 1]] * 3))
    p = np.array([[1e16, 0, 0], [1.0, 0, 0], [-1e16, 0, 0]])
    np.testing.assert_array_equal(pair.response(p), [[3, 0, 0]] * 3)
    after = pair.apply_impulses(p)
    np.testing.assert_allclose(
        after.velocity - port.snapshot().velocity, [[1, 0, 0], [-2, 0, 0], [0, 0, 0]], atol=1e-12
    )


@pytest.mark.parametrize(
    "pairs",
    [
        np.array([[0.0, 1]]),
        np.array([[False, True]]),
        np.array([[0, 3]]),
        np.array([[-1, 0]]),
        np.array([0, 1]),
        np.array([[0, 1, 2]]),
        np.array([[0, 2**64 - 1]], np.uint64),
    ],
)
def test_invalid_pairs_rejected(pairs):
    with pytest.raises((ValueError, TypeError)):
        api().PointPairOperator(particles(), pairs)


def test_pair_empty_ownership_and_impulse_validation():
    base = particles()
    pair = api().PointPairOperator(base, np.empty((0, 2), dtype=int))
    assert pair.response(np.empty((0, 3))).shape == (0, 3)
    np.testing.assert_array_equal(
        pair.apply_impulses(np.empty((0, 3))).velocity, base.snapshot().velocity
    )
    indices = np.array([[0, 1]])
    pair = api().PointPairOperator(base, indices)
    indices[:] = 2
    pair.pairs[:] = 2
    np.testing.assert_array_equal(pair.pairs, [[0, 1]])
    before = pair.relative_positions()
    pair.relative_positions()[:] = 100
    np.testing.assert_array_equal(pair.relative_positions(), before)
    pair.snapshot().velocity[:] = 10
    np.testing.assert_array_equal(pair.snapshot().velocity, base.snapshot().velocity)
    for p in (np.ones((1, 3), np.float32), np.ones((2, 3)), np.full((1, 3), np.nan)):
        with pytest.raises((TypeError, ValueError)):
            pair.response(p)
        with pytest.raises((TypeError, ValueError)):
            pair.apply_impulses(p)


def test_pair_robot_shared_ancestor_analytic_response():
    from test_robot_dynamics import planar_mass
    from test_robot_kinematics import robot_model

    from phymeo.core.robot import RobotState
    from phymeo.coupling.robot_port import RobotPointPort

    base = RobotPointPort(
        robot_model(),
        RobotState(np.zeros(2), np.zeros(2)),
        [0, 1],
        np.array([[0.0, -0.5, 0], [0.0, -0.5, 0]]),
    )
    b = np.array([[1.0, 0], [0.25, 0.75]])
    pair = api().PointPairOperator(api().WeightedPointPort(base, b), np.array([[0, 1]]))
    # First endpoint minus the barycentric second endpoint: .75*(J0-J1).
    relative_jacobian = np.array([-0.375, -0.375])
    delta = np.linalg.solve(planar_mass(np.zeros(2)), relative_jacobian * 2)
    p = np.array([[2.0, 0, 0]])
    np.testing.assert_allclose(pair.response(p), [[relative_jacobian @ delta, 0, 0]], atol=1e-12)
    candidate = pair.apply_impulses(p)
    np.testing.assert_allclose(candidate.v, delta, atol=1e-12)
    np.testing.assert_array_equal(candidate.q, np.zeros(2))
    np.testing.assert_array_equal(base.snapshot().v, np.zeros(2))


@pytest.mark.parametrize("method", ["response", "apply_impulses"])
def test_pair_overflow_never_returns_partial_candidate(method):
    pair = api().PointPairOperator(particles(), np.array([[0, 1], [0, 1]]))
    before = pair.snapshot()
    with pytest.raises(SolverFailure):
        getattr(pair, method)(np.full((2, 3), 1e308))
    np.testing.assert_array_equal(pair.snapshot().velocity, before.velocity)
