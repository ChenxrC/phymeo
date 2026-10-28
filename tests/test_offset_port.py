"""Independent physical tests for finite-thickness material attachments."""

import importlib

import numpy as np
import pytest

from phymeo import ParticleModel, ParticleState, SolverFailure, WeightedPointPort
from phymeo.coupling.ports import ParticlePointPort

X = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
B = np.array([[0.2, 0.3, 0.5]])
T = np.array([[0, 1, 2]])


def api():
    return importlib.import_module("phymeo.coupling.offset_port").TriangleOffsetPointPort


def base(x=X, v=None, inverse_mass=None, cls=ParticlePointPort):
    n = len(x)
    return cls(
        ParticleModel(np.ones(n) if inverse_mass is None else inverse_mass, np.zeros(3)),
        ParticleState(x, np.zeros_like(x) if v is None else v),
        np.arange(n),
        domain="cloth",
    )


def port(source=None, triangles=T, weights=B, offsets=None):
    return api()(
        base() if source is None else source,
        triangles,
        weights,
        np.array([0.2]) if offsets is None else offsets,
    )


def test_public_api_and_independent_unit_triangle_impulse():
    import phymeo

    assert phymeo.TriangleOffsetPointPort is api()
    p = port(base(inverse_mass=np.array([1.0, 2, 3])))
    force = np.array([[1.0, 0, 0]])
    np.testing.assert_allclose(p.positions(), [[0.3, 0.5, 0.2]], atol=1e-15)
    np.testing.assert_allclose(p.response(force), [[1.09, 0.04, -0.08]], atol=1e-14)
    after = p.apply_impulses(force)
    np.testing.assert_allclose(after.velocity, [[0.2, 0, 0.2], [0.6, 0, -0.4], [1.5, 0, 0]])
    loads = after.velocity / [[1], [2], [3]]
    np.testing.assert_allclose(loads.sum(axis=0), force[0], atol=1e-15)
    np.testing.assert_allclose(np.cross(X, loads).sum(axis=0), [0, 0.2, -0.5], atol=1e-15)
    audit = p.candidate_audit(after)
    assert audit.energy_change == pytest.approx(0.545, abs=1e-14)
    np.testing.assert_allclose(audit.initial_velocities, 0)
    np.testing.assert_allclose(audit.candidate_velocities, p.response(force), atol=1e-14)
    np.testing.assert_array_equal(p.snapshot().velocity, np.zeros((3, 3)))
    assert p.domain == "cloth" and p.dtype == np.dtype(np.float64)


@pytest.mark.parametrize("offset", [-0.2, 0.0, 0.2])
def test_rigid_translation_rotation_and_virtual_power(offset):
    omega, translation = np.array([0.7, -0.4, 0.3]), np.array([-0.2, 0.5, 0.6])
    v = translation + np.cross(omega, X)
    p = port(base(v=v), offsets=np.array([offset]))
    np.testing.assert_allclose(
        p.velocities(), translation + np.cross(omega, p.positions()), atol=1e-14
    )
    force = np.array([[0.4, -0.3, 0.7]])
    loads = p.apply_impulses(force).velocity - v
    assert np.sum(force * p.velocities()) == pytest.approx(np.sum(loads * v), abs=1e-14)
    np.testing.assert_allclose(loads.sum(axis=0), force[0], atol=1e-14)
    np.testing.assert_allclose(
        np.cross(X, loads).sum(axis=0), np.cross(p.positions(), force).sum(axis=0), atol=1e-14
    )


def material_point(x, weights, offset):
    normal = np.cross(x[1] - x[0], x[2] - x[0])
    return weights @ x + offset * normal / np.linalg.norm(normal)


def test_jacobian_against_independent_central_difference_and_energy():
    rng = np.random.default_rng(273)
    for _ in range(24):
        x = X + rng.normal(size=(3, 3)) * 0.1
        v, force = rng.normal(size=(3, 3)), rng.normal(size=(1, 3))
        b = rng.random(3)
        b /= b.sum()
        offset = float(rng.uniform(-0.25, 0.25))
        p = port(base(x, v), weights=b[None], offsets=np.array([offset]))
        h = 1e-6
        derivative = (
            material_point(x + h * v, b, offset) - material_point(x - h * v, b, offset)
        ) / (2 * h)
        np.testing.assert_allclose(p.velocities()[0], derivative, atol=3e-10, rtol=3e-9)
        after = p.apply_impulses(force)
        loads = after.velocity - v
        assert np.sum(force * p.velocities()) == pytest.approx(np.sum(loads * v), abs=2e-14)
        np.testing.assert_allclose(
            np.cross(x, loads).sum(axis=0), np.cross(p.positions(), force)[0], atol=3e-14
        )
        actual_energy = np.sum(after.velocity**2 - v**2) / 2
        predicted_energy = np.sum(force * (p.velocities() + p.response(force) / 2))
        assert actual_energy == pytest.approx(predicted_energy, abs=3e-14, rel=3e-14)


def test_zero_offset_agrees_with_weighted_port():
    source = base(v=np.arange(9, dtype=float).reshape(3, 3))
    p, q = port(source, offsets=np.zeros(1)), WeightedPointPort(source, B)
    force = np.array([[0.3, 0.2, -0.1]])
    for method, args in [("positions", ()), ("velocities", ()), ("response", (force,))]:
        np.testing.assert_allclose(getattr(p, method)(*args), getattr(q, method)(*args), atol=1e-14)
    np.testing.assert_allclose(p.apply_impulses(force).velocity, q.apply_impulses(force).velocity)


class Counted(ParticlePointPort):
    calls = 0

    def apply_impulses(self, impulses):
        self.calls += 1
        return super().apply_impulses(impulses)


def test_shared_nodes_cross_response_and_single_aggregate_candidate():
    source = base(cls=Counted)
    p = port(
        source,
        triangles=np.repeat(T, 2, axis=0),
        weights=np.repeat(B, 2, axis=0),
        offsets=np.array([0.2, -0.1]),
    )
    f = np.array([[0.8, -0.3, 0.4], [-0.2, 0.5, 0.3]])
    g = np.array([[0.2, 0.1, -0.6], [0.4, 0.1, 0.2]])
    after = p.apply_impulses(f)
    assert source.calls == 1
    np.testing.assert_allclose(after.velocity.sum(axis=0), f.sum(axis=0), atol=1e-14)
    np.testing.assert_allclose(
        np.cross(X, after.velocity).sum(axis=0), np.cross(p.positions(), f).sum(axis=0), atol=1e-14
    )
    assert np.sum(g * p.response(f)) == pytest.approx(np.sum(f * p.response(g)), abs=1e-14)
    assert np.sum(f * p.response(f)) >= 0
    np.testing.assert_allclose(
        p.candidate_audit(after).candidate_velocities, p.response(f), atol=1e-14
    )
    assert p.candidate_audit(after).energy_change == pytest.approx(np.sum(after.velocity**2) / 2)
    only_first = p.response(np.array([f[0], np.zeros(3)]))
    assert np.linalg.norm(only_first[1]) > 0.01  # shared-node off-diagonal response


def test_coordinate_rotation_and_length_scale_covariance():
    q, _ = np.linalg.qr(np.random.default_rng(111).normal(size=(3, 3)))
    assert np.linalg.det(q) > 0
    f, v = np.array([[0.4, -0.2, 0.7]]), np.arange(9, dtype=float).reshape(3, 3) / 10
    original = port(base(v=v))
    for scale in [1e-5, 1.0, 1e5]:
        shifted = port(
            base((X @ q.T + [2, -3, 4]) * scale, v @ q.T), offsets=np.array([0.2 * scale])
        )
        np.testing.assert_allclose(
            shifted.positions(), (original.positions() @ q.T + [2, -3, 4]) * scale, atol=1e-9
        )
        np.testing.assert_allclose(shifted.velocities(), original.velocities() @ q.T, atol=2e-14)
        np.testing.assert_allclose(
            shifted.response(f @ q.T), original.response(f) @ q.T, atol=2e-14
        )


def test_input_query_ownership_and_configuration_validation():
    triangles, weights, offsets = T.copy(), B.copy(), np.array([0.2])
    p = port(triangles=triangles, weights=weights, offsets=offsets)
    triangles[:] = 0
    weights[:] = 0
    offsets[:] = 100
    for name in ["triangles", "weights", "offsets", "jacobians"]:
        getattr(p, name)[:] = 99
    for name in ["positions", "velocities"]:
        getattr(p, name)()[:] = 99
    candidate = p.apply_impulses(np.ones((1, 3)))
    candidate.velocity[:] = 99
    np.testing.assert_allclose(p.positions(), [[0.3, 0.5, 0.2]])
    np.testing.assert_allclose(p.weights, B)
    np.testing.assert_array_equal(p.snapshot().velocity, np.zeros((3, 3)))
    candidate.position[0, 0] += 1
    with pytest.raises(ValueError, match="configuration"):
        p.candidate_audit(candidate)


@pytest.mark.parametrize(
    "field,value",
    [
        ("triangles", np.array([[0, 1, 3]])),
        ("triangles", np.array([[0, -1, 2]])),
        ("triangles", np.array([[0.0, 1, 2]])),
        ("triangles", np.array([[0, 1]])),
        ("weights", np.ones((1, 3), np.float32)),
        ("weights", np.ones((1, 3), int)),
        ("weights", np.array([[-0.2, 0.7, 0.5]])),
        ("weights", np.array([[0.2, 0.3, 0.6]])),
        ("weights", np.array([[np.nan, 0.3, 0.5]])),
        ("weights", np.ones((2, 3))),
        ("offsets", np.array([np.inf])),
        ("offsets", np.array([[0.2]])),
        ("offsets", np.array([0.2], np.float32)),
        ("offsets", np.array([], float)),
    ],
)
def test_invalid_constructor_inputs(field, value):
    with pytest.raises((ValueError, TypeError)):
        port(**{field: value})


@pytest.mark.parametrize(
    "x",
    [
        np.zeros((3, 3)),
        np.array([[0.0, 0, 0], [1, 0, 0], [0.5, 1e-12, 0]]),
        np.array([[-1e308, 0, 0], [1e308, 0, 0], [0, 1e308, 0]]),
    ],
)
def test_degenerate_or_unrepresentable_geometry_is_refused(x):
    with pytest.raises(SolverFailure):
        port(base(x))


def test_lost_world_offset_is_refused_by_actual_point_moment_check():
    x = X + [0, 0, 1e8]
    with pytest.raises(SolverFailure, match="represent|moment|kinematic"):
        port(base(x), offsets=np.array([1e-8]))


@pytest.mark.parametrize("method", ["response", "apply_impulses"])
def test_invalid_or_overflowing_impulses_preserve_base(method):
    p = port(offsets=np.array([1000.0]))
    with pytest.raises((TypeError, ValueError)):
        getattr(p, method)(np.ones((1, 3), np.float32))
    with pytest.raises(SolverFailure):
        getattr(p, method)(np.full((1, 3), 1.79e308))
    np.testing.assert_array_equal(p.snapshot().velocity, np.zeros((3, 3)))


@pytest.mark.parametrize("n", [0, 3])
def test_empty_port(n):
    p = port(
        base(np.zeros((n, 3))),
        triangles=np.empty((0, 3), int),
        weights=np.empty((0, 3)),
        offsets=np.empty(0),
    )
    assert p.positions().shape == p.velocities().shape == (0, 3)
    assert p.jacobians.shape == (0, 3, 3, 3)
    assert p.response(np.empty((0, 3))).shape == (0, 3)
    np.testing.assert_array_equal(p.apply_impulses(np.empty((0, 3))).velocity, np.zeros((n, 3)))


def test_robot_surface_agrees_with_direct_rigid_attachment():
    from test_robot_kinematics import robot_model

    from phymeo import RobotPointPort, RobotState

    model = robot_model(1, gravity=np.zeros(3))
    state = RobotState(np.array([0.37]), np.array([-0.4]))
    local = np.array([[0.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    source = RobotPointPort(model, state, [0, 0, 0], local)
    mapped = port(source)
    direct = RobotPointPort(model, state, [0], np.array([[0.2, 0.3, 0.5]]))
    force = np.array([[0.7, -0.2, 0.4]])
    for method, args in [("positions", ()), ("velocities", ()), ("response", (force,))]:
        np.testing.assert_allclose(
            getattr(mapped, method)(*args), getattr(direct, method)(*args), atol=3e-14
        )
    after = mapped.apply_impulses(force)
    np.testing.assert_allclose(after.v, direct.apply_impulses(force).v, atol=3e-14)
    assert mapped.candidate_audit(after).energy_change == pytest.approx(
        direct.candidate_audit(after).energy_change, abs=3e-14
    )


def test_tiny_lost_offsets_cannot_create_a_couple_at_identical_actual_points():
    source = base(X + [0, 0, 1e8])
    with pytest.raises(SolverFailure, match="offset|represent"):
        port(
            source,
            triangles=np.repeat(T, 2, axis=0),
            weights=np.repeat(B, 2, axis=0),
            offsets=np.array([1e-12, -1e-12]),
        )
    np.testing.assert_array_equal(source.snapshot().velocity, np.zeros((3, 3)))


@pytest.mark.parametrize("operation", ["velocities", "apply_impulses"])
def test_nonzero_block_products_cannot_silently_underflow(operation):
    velocity = np.zeros((3, 3))
    velocity[1, 0] = 1e-200
    p = port(base(v=velocity), weights=np.array([[1.0, 1e-200, 0.0]]), offsets=np.zeros(1))
    with pytest.raises(SolverFailure, match="represent"):
        if operation == "velocities":
            p.velocities()
        else:
            p.apply_impulses(np.array([[1e-200, 0, 0]]))
    np.testing.assert_array_equal(p.snapshot().velocity, velocity)
