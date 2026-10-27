"""Independent multi-domain response, load conservation and candidate isolation."""

import importlib

import numpy as np
import pytest

from phymeo import ParticleModel, ParticlePointPort, ParticleState, SolverFailure


def graph(ports, left, right):
    return importlib.import_module("phymeo.coupling.domain_graph").MultiDomainPointOperator(
        ports, left, right
    )


def test_multidomain_graph_public_export():
    import phymeo

    assert (
        phymeo.MultiDomainPointOperator
        is importlib.import_module("phymeo.coupling.domain_graph").MultiDomainPointOperator
    )


def particle(domain, inverse_mass, velocity=None):
    weights = np.atleast_1d(np.array(inverse_mass, dtype=float))
    velocity = np.zeros((len(weights), 3)) if velocity is None else np.array(velocity, dtype=float)
    return ParticlePointPort(
        ParticleModel(weights, np.zeros(3)),
        ParticleState(np.tile([1.0, 2, 3], (len(weights), 1)), velocity),
        np.arange(len(weights)),
        domain=domain,
    )


def three_ports():
    return {
        "A": particle("A", 1, [[1, 2, 3]]),
        "B": particle("B", 2, [[-1, 0.5, 2]]),
        "cloth": particle("cloth", 4, [[0.2, 0.1, 0.3]]),
    }


def three_graph(ports=None):
    return graph(
        three_ports() if ports is None else ports,
        [("A", 0), ("B", 0)],
        [("cloth", 0), ("cloth", 0)],
    )


def test_shared_cloth_response_power_and_total_momentum():
    ports = three_ports()
    operator = three_graph(ports)
    p = np.array([[2.0, -1, 3], [0.5, 4, -2]])
    np.testing.assert_allclose(operator.response(p), np.array([[5.0, 4], [4.0, 6]]) @ p, atol=1e-12)
    np.testing.assert_array_equal(operator.relative_positions(), np.zeros((2, 3)))
    expected_loads = {"A": p[:1], "B": p[1:], "cloth": -(p[:1] + p[1:])}
    loads = operator.domain_impulses(p)
    assert operator.domains == ("A", "B", "cloth")
    assert operator.dtype == np.dtype(np.float64)
    for name in ports:
        np.testing.assert_array_equal(loads[name], expected_loads[name])
        assert not loads[name].flags.writeable
    power = sum(float(np.sum(loads[name] * ports[name].velocities())) for name in ports)
    assert np.sum(p * operator.relative_velocities()) == pytest.approx(power, abs=1e-12)
    candidates = operator.apply_impulses(p)
    momentum, angular = np.zeros(3), np.zeros(3)
    energy = 0.0
    for name, w in (("A", 1), ("B", 2), ("cloth", 4)):
        before, after = ports[name].snapshot(), candidates[name]
        delta = after.velocity - before.velocity
        np.testing.assert_allclose(delta, w * expected_loads[name], atol=1e-12)
        momentum += delta[0] / w
        angular += np.cross(after.position[0], delta[0] / w)
        energy += float(np.sum(after.velocity**2 - before.velocity**2) / (2 * w))
        np.testing.assert_array_equal(after.position, before.position)
    np.testing.assert_allclose(momentum, 0, atol=1e-12)
    np.testing.assert_allclose(angular, 0, atol=1e-12)
    assert energy == pytest.approx(power + np.sum(p * operator.response(p)) / 2, abs=1e-12)


def test_graph_calls_each_domain_candidate_once_and_snapshots_are_owned():
    class Counted:
        def __init__(self, base):
            self.base, self.calls = base, 0

        def __getattr__(self, name):
            return getattr(self.base, name)

        def apply_impulses(self, p):
            self.calls += 1
            return self.base.apply_impulses(p)

    ports = {key: Counted(value) for key, value in three_ports().items()}
    operator = three_graph(ports)
    before = operator.snapshot()
    result = operator.apply_impulses(np.ones((2, 3)))
    assert all(port.calls == 1 for port in ports.values())
    with pytest.raises(TypeError):
        result["A"] = result["B"]
    result["A"].velocity[:] = 100
    operator.snapshot()["B"].position[:] = 100
    for name in ports:
        np.testing.assert_array_equal(operator.snapshot()[name].velocity, before[name].velocity)
        np.testing.assert_array_equal(operator.snapshot()[name].position, before[name].position)
    operator.relative_velocities()[:] = 100
    np.testing.assert_array_equal(operator.relative_positions(), np.zeros((2, 3)))


def test_same_domain_duplicate_edges_compensated_cancellation_and_self_loop():
    ports = {"cloth": particle("cloth", [1, 2])}
    operator = graph(ports, [("cloth", 0)] * 3, [("cloth", 1)] * 3)
    p = np.array([[1e16, 0, 0], [1.0, 0, 0], [-1e16, 0, 0]])
    np.testing.assert_array_equal(operator.response(p), [[3, 0, 0]] * 3)
    np.testing.assert_array_equal(
        operator.apply_impulses(p)["cloth"].velocity, [[1, 0, 0], [-2, 0, 0]]
    )
    loop = graph(ports, [("cloth", 0)], [("cloth", 0)])
    np.testing.assert_array_equal(loop.response(np.full((1, 3), 1e308)), [[0, 0, 0]])


def test_static_platform_reaction_remains_in_ledger():
    operator = graph(
        {"cloth": particle("cloth", 2), "platform": particle("platform", 0)},
        [("cloth", 0)],
        [("platform", 0)],
    )
    p = np.array([[0.0, 0.3, 0]])
    np.testing.assert_array_equal(operator.domain_impulses(p)["platform"], -p)
    np.testing.assert_array_equal(operator.apply_impulses(p)["platform"].velocity, [[0, 0, 0]])
    np.testing.assert_array_equal(operator.response(p), 2 * p)


@pytest.mark.parametrize("ports", [{}, {"A": particle("A", 1)}])
def test_empty_graph_is_well_defined(ports):
    operator = graph(ports, [], [])
    assert operator.relative_positions().shape == (0, 3)
    assert operator.response(np.empty((0, 3))).shape == (0, 3)
    assert set(operator.apply_impulses(np.empty((0, 3)))) == set(ports)


@pytest.mark.parametrize(
    "bad", [[("missing", 0)], [("A", -1)], [("A", 1)], [("A", True)], [("A", 0.0)], [("A",)], ["A"]]
)
def test_invalid_endpoint_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        graph(three_ports(), bad, [("B", 0)])


def test_mismatched_domain_or_edge_count_and_precision_rejected():
    with pytest.raises((TypeError, ValueError)):
        graph({"wrong": particle("A", 1)}, [], [])
    with pytest.raises((TypeError, ValueError)):
        graph(three_ports(), [("A", 0)], [])
    fp32 = ParticlePointPort(
        ParticleModel(np.ones(1, np.float32), np.zeros(3, np.float32)),
        ParticleState(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32)),
        [0],
        domain="A",
    )
    with pytest.raises((TypeError, ValueError)):
        graph({"A": fp32}, [], [])


def test_graph_configuration_and_impulses_are_owned():
    ports, left, right = three_ports(), [["A", 0], ["B", 0]], [["cloth", 0], ["cloth", 0]]
    operator = graph(ports, left, right)
    ports.clear()
    left[0][0] = "bad"
    right.clear()
    assert operator.left == (("A", 0), ("B", 0))
    assert operator.right == (("cloth", 0), ("cloth", 0))
    p = np.ones((2, 3))
    operator.apply_impulses(p)
    np.testing.assert_array_equal(p, np.ones((2, 3)))
    with pytest.raises(TypeError):
        operator.domain_impulses(p)["A"] = p
    for bad in (np.ones((2, 3), np.float32), np.ones((1, 3)), np.full((2, 3), np.inf)):
        with pytest.raises((TypeError, ValueError)):
            operator.response(bad)
        with pytest.raises((TypeError, ValueError)):
            operator.apply_impulses(bad)


@pytest.mark.parametrize("failure", ["solver", "candidate"])
def test_last_domain_failure_leaves_all_external_state_unchanged(failure):
    class Broken:
        def __init__(self, base):
            self.base = base

        def __getattr__(self, name):
            return getattr(self.base, name)

        def apply_impulses(self, p):
            if failure == "solver":
                raise SolverFailure("last domain failure")
            candidate = self.base.apply_impulses(p)
            candidate.position[:] += 1
            return candidate

    ports = three_ports()
    ports["cloth"] = Broken(ports["cloth"])
    operator = three_graph(ports)
    before = operator.snapshot()
    with pytest.raises((SolverFailure, ValueError)):
        operator.apply_impulses(np.ones((2, 3)))
    for name in before:
        np.testing.assert_array_equal(operator.snapshot()[name].position, before[name].position)
        np.testing.assert_array_equal(operator.snapshot()[name].velocity, before[name].velocity)
    with pytest.raises(ValueError):
        operator.validate_candidates({"A": before["A"]})


def test_overflow_is_rejected_before_any_candidate_is_published():
    operator = three_graph()
    with pytest.raises(SolverFailure):
        operator.apply_impulses(np.full((2, 3), 1e308))


def test_two_robots_and_cloth_preserve_full_shared_ancestor_response():
    from test_robot_dynamics import planar_mass
    from test_robot_kinematics import robot_model

    from phymeo import RobotPointPort, RobotState

    state = RobotState(np.zeros(2), np.zeros(2))
    ports = {
        name: RobotPointPort(
            robot_model(), state, [0, 1], np.array([[0.0, -0.5, 0], [0.0, -0.5, 0]]), domain=name
        )
        for name in ("A", "B")
    }
    ports["cloth"] = particle("cloth", 4)
    operator = graph(ports, [("A", 0), ("A", 1), ("B", 1)], [("cloth", 0), ("B", 0), ("cloth", 0)])
    p = np.array([[0.1, 0, 0], [0.2, 0, 0], [-0.3, 0, 0]])
    mass, j = planar_mass(np.zeros(2)), np.array([[0.5, 0], [1.0, 0.5]])
    da = np.linalg.solve(mass, j.T @ p[:2, 0])
    db = np.linalg.solve(mass, j.T @ np.array([-p[1, 0], p[2, 0]]))
    dc = -4 * (p[0, 0] + p[2, 0])
    expected = np.array(
        [[j[0] @ da - dc, 0, 0], [j[1] @ da - j[0] @ db, 0, 0], [j[1] @ db - dc, 0, 0]]
    )
    np.testing.assert_allclose(operator.response(p), expected, atol=1e-12)
    result = operator.apply_impulses(p)
    np.testing.assert_allclose(result["A"].v, da, atol=1e-12)
    np.testing.assert_allclose(result["B"].v, db, atol=1e-12)
    np.testing.assert_allclose(result["cloth"].velocity, [[dc, 0, 0]], atol=1e-12)
    np.testing.assert_array_equal(result["A"].q, state.q)
    np.testing.assert_array_equal(result["B"].q, state.q)
