"""Point response must include shared-body cross terms and conserve exchanged loads."""

import importlib

import numpy as np
import pytest

from phymeo import ParticleModel, ParticleState, RigidBodyModel, RigidBodyState, SolverFailure


def api():
    return importlib.import_module("phymeo.coupling.ports")


def particle(dtype=np.float64, *, inverse_mass=0.5, position=(1, 0, 0)):
    return (
        ParticleModel(np.array([inverse_mass], dtype), np.zeros(3, dtype)),
        ParticleState(np.array([position], dtype), np.zeros((1, 3), dtype)),
    )


def body(dtype=np.float64, *, inverse_mass=0.25):
    return (
        RigidBodyModel(
            np.array([inverse_mass], dtype), np.full((1, 3), 2, dtype), np.zeros(3, dtype)
        ),
        RigidBodyState(
            np.zeros((1, 3), dtype),
            np.array([[0, 0, 0, 1]], dtype),
            np.zeros((1, 3), dtype),
            np.zeros((1, 3), dtype),
        ),
    )


@pytest.mark.parametrize("dtype,atol", [(np.float64, 1e-12), (np.float32, 1e-6)])
def test_same_point_exchange_has_opposite_linear_and_correct_angular_momentum(dtype, atol):
    pmodel, ps = particle(dtype)
    bmodel, bs = body(dtype)
    p = api().ParticlePointPort(pmodel, ps, [0], domain="cloth")
    b = api().RigidPointPort(bmodel, bs, [0], np.array([[1, 0, 0]], dtype), domain="body")
    result = api().exchange_impulses(p, b, np.array([[0, 2, 0]], dtype))
    after_p, after_b = result.states["cloth"], result.states["body"]
    np.testing.assert_allclose(after_p.velocity, [[0, 1, 0]], atol=atol, rtol=0)
    np.testing.assert_allclose(after_b.velocity, [[0, -0.5, 0]], atol=atol, rtol=0)
    np.testing.assert_allclose(after_b.angular_velocity, [[0, 0, -1]], atol=atol, rtol=0)
    np.testing.assert_allclose(2 * after_p.velocity + 4 * after_b.velocity, 0, atol=atol)
    np.testing.assert_allclose(
        np.cross(after_p.position, 2 * after_p.velocity) + 2 * after_b.angular_velocity,
        0,
        atol=atol,
    )
    np.testing.assert_array_equal(result.impulses["cloth"], -result.impulses["body"])
    np.testing.assert_array_equal(ps.velocity, [[0, 0, 0]])
    np.testing.assert_array_equal(bs.angular_velocity, [[0, 0, 0]])
    np.testing.assert_array_equal(after_b.position, bs.position)
    np.testing.assert_array_equal(after_b.orientation, bs.orientation)


def test_shared_rigid_ports_include_negative_cross_response_at_opposite_point():
    model, state = body()
    offsets = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    port = api().RigidPointPort(model, state, [0, 0], offsets)
    impulses = np.array([[0.0, 2, 0], [0.0, 0, 0]])
    expected = [[0, 1.5, 0], [0, -0.5, 0]]
    np.testing.assert_allclose(port.response(impulses), expected, atol=1e-12)
    after = port.apply_impulses(impulses)
    queried = api().RigidPointPort(model, after, [0, 0], offsets)
    np.testing.assert_allclose(queried.velocities() - port.velocities(), expected, atol=1e-12)


def test_repeated_particle_ports_accumulate_once_with_compensated_cancellation():
    model, state = particle(inverse_mass=1)
    port = api().ParticlePointPort(model, state, [0, 0, 0])
    impulses = np.array([[1e16, 0, 0], [1, 0, 0], [-1e16, 0, 0]])
    np.testing.assert_array_equal(port.response(impulses), [[1, 0, 0]] * 3)
    np.testing.assert_array_equal(port.apply_impulses(impulses).velocity, [[1, 0, 0]])


def test_rotated_body_attachment_uses_world_inertia_and_point_velocity():
    model, state = body()
    model = RigidBodyModel(model.inverse_mass, np.array([[2.0, 3.0, 4.0]]), model.gravity)
    state.position[:] = [2, 3, 4]
    state.orientation[:] = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
    port = api().RigidPointPort(model, state, [0], np.array([[1.0, 0, 0]]))
    np.testing.assert_allclose(port.positions(), [[2, 4, 4]], atol=1e-12)
    impulse = np.array([[0.0, 0, 4]])
    np.testing.assert_allclose(port.response(impulse), [[0, 0, 7 / 3]], atol=1e-12)
    np.testing.assert_allclose(
        port.apply_impulses(impulse).angular_velocity, [[4 / 3, 0, 0]], atol=1e-12
    )


def test_particle_to_particle_exchange_uses_the_same_protocol():
    m1, s1 = particle(inverse_mass=0.5, position=(0, 0, 0))
    m2, s2 = particle(inverse_mass=0.25, position=(0, 0, 0))
    left = api().ParticlePointPort(m1, s1, [0], domain="a")
    right = api().ParticlePointPort(m2, s2, [0], domain="b")
    result = api().exchange_impulses(left, right, np.array([[4.0, 0, 0]]))
    np.testing.assert_array_equal(result.states["a"].velocity, [[2, 0, 0]])
    np.testing.assert_array_equal(result.states["b"].velocity, [[-1, 0, 0]])


@pytest.mark.parametrize("kind", ["particle", "rigid"])
def test_ports_and_returned_queries_own_snapshots(kind):
    model, state = particle() if kind == "particle" else body()
    if kind == "particle":
        port = api().ParticlePointPort(model, state, [0])
    else:
        port = api().RigidPointPort(model, state, [0], np.array([[1.0, 0, 0]]))
    before = port.positions()
    state.position[:] = 42
    exposed = port.positions()
    exposed[:] = 99
    port.snapshot().velocity[:] = 33
    np.testing.assert_array_equal(port.positions(), before)
    np.testing.assert_array_equal(port.velocities(), [[0, 0, 0]])


def test_fixed_and_empty_rigid_ports_do_not_divide_by_zero_inertia():
    _, state = body()
    model = RigidBodyModel(np.zeros(1), np.zeros((1, 3)), np.zeros(3))
    port = api().RigidPointPort(model, state, [0], np.array([[1.0, 0, 0]]))
    np.testing.assert_array_equal(port.response(np.ones((1, 3))), np.zeros((1, 3)))
    np.testing.assert_array_equal(
        port.apply_impulses(np.ones((1, 3))).angular_velocity, np.zeros((1, 3))
    )
    empty = api().RigidPointPort(model, state, np.array([], np.int64), np.empty((0, 3)))
    assert empty.response(np.empty((0, 3))).shape == (0, 3)


@pytest.mark.parametrize("indices", [[-1], [1], [True], [[0]]])
def test_bad_port_indices_are_rejected(indices):
    model, state = particle()
    with pytest.raises((TypeError, ValueError)):
        api().ParticlePointPort(model, state, indices)


@pytest.mark.parametrize(
    "impulses", [np.ones((1, 3), np.float32), np.ones((2, 3)), np.array([[np.nan, 0, 0]])]
)
def test_bad_impulses_are_rejected_without_state_changes(impulses):
    model, state = particle()
    port = api().ParticlePointPort(model, state, [0])
    with pytest.raises((TypeError, ValueError)):
        port.apply_impulses(impulses)
    np.testing.assert_array_equal(state.velocity, [[0, 0, 0]])


def test_exchange_rejects_different_world_points_and_same_domain():
    model, state = particle()
    p = api().ParticlePointPort(model, state, [0], domain="a")
    m2, s2 = particle(position=(2, 0, 0))
    other = api().ParticlePointPort(m2, s2, [0], domain="b")
    with pytest.raises(ValueError):
        api().exchange_impulses(p, other, np.ones((1, 3)))
    with pytest.raises(ValueError):
        api().exchange_impulses(p, p, np.ones((1, 3)))


def test_second_side_numerical_failure_never_exposes_partial_exchange():
    pm, ps = particle(position=(0, 0, 0))
    bm, bs = body(inverse_mass=1e308)
    p = api().ParticlePointPort(pm, ps, [0])
    b = api().RigidPointPort(bm, bs, [0], np.zeros((1, 3)))
    with pytest.raises(SolverFailure):
        api().exchange_impulses(p, b, np.array([[2.0, 0, 0]]))
    np.testing.assert_array_equal(p.snapshot().velocity, [[0, 0, 0]])
    np.testing.assert_array_equal(b.snapshot().velocity, [[0, 0, 0]])


def test_exchange_rejects_malformed_extension_candidate_before_returning_either_side():
    pm, ps = particle(position=(0, 0, 0))

    class InvalidCandidate(api().ParticlePointPort):
        def apply_impulses(self, impulses):
            return ParticleState(np.zeros((2, 3)), np.zeros((2, 3)))

    left = api().ParticlePointPort(pm, ps, [0], domain="a")
    right = InvalidCandidate(pm, ps, [0], domain="b")
    with pytest.raises(ValueError):
        api().exchange_impulses(left, right, np.array([[1.0, 0, 0]]))
    np.testing.assert_array_equal(left.snapshot().velocity, [[0, 0, 0]])


def test_unrepresentable_world_attachment_never_returns_infinite_geometry():
    model, state = body()
    state.orientation[:] = [0, 0, np.sin(np.pi / 8), np.cos(np.pi / 8)]
    port = api().RigidPointPort(model, state, [0], np.array([[1.3e308, 1.3e308, 0]]))
    with pytest.raises(SolverFailure):
        port.positions()


def test_impulse_extension_cannot_silently_advance_dynamic_configuration():
    model, state = particle(position=(0, 0, 0))

    class TimeAdvancingPort(api().ParticlePointPort):
        def apply_impulses(self, impulses):
            candidate = super().apply_impulses(impulses)
            candidate.position[:] += 1
            return candidate

    left = api().ParticlePointPort(model, state, [0], domain="left")
    right = TimeAdvancingPort(model, state, [0], domain="right")
    with pytest.raises(ValueError):
        api().exchange_impulses(left, right, np.ones((1, 3)))
