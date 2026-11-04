"""Normal restitution of closed material witnesses with the same measured impulses and work."""

import numpy as np
import pytest
from test_contact_impulse import node, plane_graph, solve
from test_floating_witness_impact import build

import phymeo as ph


@pytest.mark.parametrize(
    "speed, coefficient, threshold, desired, normal",
    [
        (-2, 0, 0, 0, 4),
        (-2, 0.5, 0, 1, 6),
        (-2, 1, 0, 2, 8),
        (-0.1, 0.8, 0.2, 0, 0.2),
        (-0.2, 1, 0.2, 0, 0.4),
        (2, 1, 0, 2, 0),
    ],
)
def test_plane_approach_rebound_settle_and_open(speed, coefficient, threshold, desired, normal):
    result = solve(
        plane_graph([0, speed, 0]),
        [[0, 1, 0]],
        mu=0,
        restitution=coefficient,
        bounce_threshold_mps=threshold,
    )
    np.testing.assert_allclose(result.states["body"].velocity[0], [0, desired, 0], atol=1e-10)
    np.testing.assert_allclose(result.world_impulses, [[0, normal, 0]], atol=1e-10)
    assert result.energy_change == pytest.approx(desired**2 - speed**2)


def test_one_impact_exchanges_momentum_between_two_masses_and_allows_slip():
    # 2 kg travels at -1; 1 kg travels at +0.5.  At e=0.5 the impulse is 1.5 Ns.
    left = node("left", 0.5, [3, -1, 0])
    right = node("right", 1, [0, 0.5, 0])
    graph = ph.MultiDomainPointOperator(
        {"left": left, "right": right}, [("left", 0)], [("right", 0)]
    )
    result = solve(graph, [[0, 1, 0]], mu=0.25, restitution=[0.5])
    a, b = (result.states[key].velocity[0] for key in ("left", "right"))
    np.testing.assert_allclose(result.world_impulses, [[-0.375, 1.5, 0]], atol=1e-10)
    np.testing.assert_allclose(2 * a + b, [6, -1.5, 0], atol=1e-10)
    assert a[1] - b[1] == pytest.approx(0.75)
    assert result.energy_change == pytest.approx(np.dot(a, a) + np.dot(b, b) / 2 - 10.125)


def test_distinct_row_materials_in_one_graph():
    ports = {str(i): node(str(i), 1, [0, -1, 0]) for i in range(2)}
    ports["ground"] = node("ground", 0, [0, 0, 0])
    graph = ph.MultiDomainPointOperator(ports, [("0", 0), ("1", 0)], [("ground", 0)] * 2)
    result = solve(graph, [[0, 1, 0]] * 2, mu=0, restitution=[0.0, 0.75])
    np.testing.assert_allclose(result.normal_impulses, [1, 1.75], atol=1e-10)
    np.testing.assert_allclose(result.relative_velocity_after[:, 1], [0, 0.75], atol=1e-10)


@pytest.mark.parametrize("bad", [-0.01, 1.001, True, np.nan, [0.2, 0.2]])
def test_bad_material_does_not_mutate_ports(bad):
    graph = plane_graph([1.0, -2.0, 0.0])
    with pytest.raises((ValueError, TypeError)):
        solve(graph, [[0, 1, 0]], restitution=bad)
    np.testing.assert_array_equal(graph.snapshot()["body"].velocity, [[1, -2, 0]])
    with pytest.raises(ValueError):
        solve(graph, [[0, 1, 0]], bounce_threshold_mps=-1)


def test_swept_floating_arm_and_sphere_recoil_with_restitution():
    _, _, _, _, _, _, ports, hit, bindings, arm, _ = build(True, True)
    result = ph.solve_witness_impulses(
        ports,
        [(hit, bindings)],
        [0.0],
        [0.0],
        sample="first_contact",
        local_vertices={arm: ((0.0, -0.125, 0.0),)},
        restitution=0.5,
    )
    assert result.normal_impulses[0] == pytest.approx(15 / 16)
    assert result.relative_velocity_after[0] @ result.contact_basis[0, :, 0] == pytest.approx(0.5)
    assert result.energy_change == pytest.approx(-15 / 64)
