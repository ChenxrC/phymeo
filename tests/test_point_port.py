"""Independent impulse, angular momentum, effective mass and atomicity checks."""

import numpy as np
import pytest

import phymeo
from phymeo import ParticleModel, ParticleState, RigidBodyModel, RigidBodyState, SolverFailure


def api(name):
    cls = getattr(phymeo, name, None)
    assert cls is not None, f"missing coupling API: {name}"
    return cls


def ports(particles=(0,), bodies=(0,)):
    return api("ParticleRigidPorts")(np.array(particles, np.int64), np.array(bodies, np.int64))


def scene(dtype=np.float64):
    particles = ParticleModel(np.array([0.5], dtype), np.zeros(3))
    particle_state = ParticleState(np.array([[1, 0, 0]], dtype), np.array([[0.2, 0.3, 0.4]], dtype))
    bodies = RigidBodyModel(np.array([0.25], dtype), np.array([[1, 2, 3]], dtype), np.zeros(3))
    body_state = RigidBodyState(
        np.zeros((1, 3), dtype),
        np.array([[0, 0, 0, 1]], dtype),
        np.array([[-0.1, 0.2, 0.3]], dtype),
        np.array([[0.2, 0.3, 0.4]], dtype),
    )
    return particles, particle_state, bodies, body_state


def test_port_indices_are_owned_read_only_and_duplicates_are_allowed():
    a, b = np.array([0, 0, 1]), np.array([1, 1, 0])
    port = api("ParticleRigidPorts")(a, b)
    a[:] = b[:] = 9
    np.testing.assert_array_equal(port.particle_indices, [0, 0, 1])
    np.testing.assert_array_equal(port.body_indices, [1, 1, 0])
    for indices in (port.particle_indices, port.body_indices):
        assert indices.dtype == np.int64 and not indices.flags.writeable


@pytest.mark.parametrize(
    "a,b",
    [
        ([0.0], [0]),
        ([-1], [0]),
        ([0], [-1]),
        ([0, 1], [0]),
        ([[0]], [0]),
        ([True], [0]),
        (np.array([2**63], np.uint64), [0]),
    ],
)
def test_invalid_port_indices_are_rejected(a, b):
    with pytest.raises((TypeError, ValueError)):
        api("ParticleRigidPorts")(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_same_point_impulse_preserves_total_linear_and_angular_momentum(dtype):
    pm, ps, bm, bs = scene(dtype)
    impulse = np.array([[0, 2, 0]], dtype)
    result = ports().apply_impulses(pm, ps, bm, bs, impulse)
    assert result.accepted and result.reason is None
    np.testing.assert_allclose(result.particle_state.velocity, ps.velocity + [[0, 1, 0]], atol=2e-7)
    np.testing.assert_allclose(result.rigid_state.velocity, bs.velocity + [[0, -0.5, 0]], atol=2e-7)
    np.testing.assert_allclose(
        result.rigid_state.angular_velocity, bs.angular_velocity + [[0, 0, -2 / 3]], atol=2e-7
    )
    before_linear = 2 * ps.velocity.astype(float) + 4 * bs.velocity.astype(float)
    after_linear = 2 * result.particle_state.velocity.astype(
        float
    ) + 4 * result.rigid_state.velocity.astype(float)
    before_angular = (
        np.cross(ps.position.astype(float), 2 * ps.velocity)
        + bm.inertia_diagonal * bs.angular_velocity
    )
    after_angular = (
        np.cross(result.particle_state.position.astype(float), 2 * result.particle_state.velocity)
        + bm.inertia_diagonal * result.rigid_state.angular_velocity
    )
    np.testing.assert_allclose(after_linear, before_linear, atol=3e-7)
    np.testing.assert_allclose(after_angular, before_angular, atol=3e-7)
    np.testing.assert_array_equal(result.particle_state.position, ps.position)
    np.testing.assert_array_equal(result.rigid_state.position, bs.position)
    np.testing.assert_array_equal(result.rigid_state.orientation, bs.orientation)
    np.testing.assert_array_equal(result.ledger.particle, impulse)
    np.testing.assert_array_equal(result.ledger.rigid_linear, -impulse)
    np.testing.assert_array_equal(result.ledger.rigid_angular, [[0, 0, -2]])


def test_rotated_principal_axes_change_world_angular_response():
    pm, ps, bm, bs = scene()
    bs.orientation[:] = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
    # r=(1,0,0), Jparticle=(0,0,2): body torque +Y; world Iyy=body Ixx=1.
    result = ports().apply_impulses(pm, ps, bm, bs, np.array([[0.0, 0.0, 2.0]]))
    assert result.accepted
    np.testing.assert_allclose(
        result.rigid_state.angular_velocity - bs.angular_velocity, [[0, 2, 0]], atol=1e-14
    )


@pytest.mark.parametrize("fixed_particle,fixed_body", [(True, False), (False, True), (True, True)])
def test_fixed_objects_keep_configuration_and_ledger_retains_reaction(fixed_particle, fixed_body):
    pm, ps, bm, bs = scene()
    pm = ParticleModel(np.array([0.0 if fixed_particle else 0.5]), np.zeros(3))
    bm = RigidBodyModel(
        np.array([0.0 if fixed_body else 0.25]),
        np.zeros((1, 3)) if fixed_body else bm.inertia_diagonal,
        np.zeros(3),
    )
    result = ports().apply_impulses(pm, ps, bm, bs, np.array([[0.0, 2.0, 0.0]]))
    assert result.accepted
    if fixed_particle:
        np.testing.assert_array_equal(result.particle_state.velocity, 0)
    if fixed_body:
        np.testing.assert_array_equal(result.rigid_state.velocity, 0)
        np.testing.assert_array_equal(result.rigid_state.angular_velocity, 0)
    np.testing.assert_array_equal(result.particle_state.position, ps.position)
    np.testing.assert_array_equal(result.rigid_state.orientation, bs.orientation)
    np.testing.assert_array_equal(result.ledger.particle, [[0, 2, 0]])
    np.testing.assert_array_equal(result.ledger.rigid_linear, [[0, -2, 0]])


def test_duplicate_ports_preserve_small_impulse_between_canceling_large_values():
    pm, ps, bm, bs = scene()
    impulse = np.array([[0, 1e16, 0], [0, 1, 0], [0, -1e16, 0]])
    result = ports([0, 0, 0], [0, 0, 0]).apply_impulses(pm, ps, bm, bs, impulse)
    assert result.accepted
    np.testing.assert_array_equal(result.ledger.particle, [[0, 1, 0]])
    np.testing.assert_array_equal(result.ledger.rigid_linear, [[0, -1, 0]])
    np.testing.assert_array_equal(result.ledger.rigid_angular, [[0, 0, -1]])
    np.testing.assert_allclose(result.particle_state.velocity - ps.velocity, [[0, 0.5, 0]])


def test_ledger_copies_inputs_and_freezes_fp64_storage():
    values = np.ones((2, 3), np.float32)
    ledger = api("ImpulseLedger")(values, values, values)
    values[:] = 9
    for field in (ledger.particle, ledger.rigid_linear, ledger.rigid_angular):
        assert field.dtype == np.float64 and not field.flags.writeable
        np.testing.assert_array_equal(field, 1)


def test_ledger_requires_matching_rigid_counts():
    with pytest.raises(ValueError):
        api("ImpulseLedger")(np.zeros((2, 3)), np.zeros((1, 3)), np.zeros((2, 3)))


def test_effective_inverse_mass_matches_analytic_diagonal_and_velocity_response():
    pm, ps, bm, bs = scene()
    port = ports()
    effective = port.effective_inverse_mass(pm, ps, bm, bs)
    assert effective.dtype == np.float64 and effective.shape == (1, 3, 3)
    np.testing.assert_allclose(effective[0], np.diag([0.75, 0.75 + 1 / 3, 1.25]), atol=1e-14)
    impulse = np.array([[0.3, -0.7, 1.1]])
    result = port.apply_impulses(pm, ps, bm, bs, impulse)
    r = ps.position - bs.position
    before = ps.velocity - bs.velocity - np.cross(bs.angular_velocity, r)
    after = (
        result.particle_state.velocity
        - result.rigid_state.velocity
        - np.cross(result.rigid_state.angular_velocity, r)
    )
    np.testing.assert_allclose(
        after - before, np.einsum("nij,nj->ni", effective, impulse), atol=1e-14
    )


def test_effective_inverse_mass_is_symmetric_positive_and_translation_invariant():
    pm, ps, bm, bs = scene()
    ps.position[:] = [1, 2, 3]
    bs.orientation[:] = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
    value = ports().effective_inverse_mass(pm, ps, bm, bs)
    np.testing.assert_allclose(value, value.swapaxes(1, 2), atol=1e-14)
    assert np.min(np.linalg.eigvalsh(value)) > 0
    ps.position += [3, 5, -2]
    bs.position += [3, 5, -2]
    np.testing.assert_allclose(ports().effective_inverse_mass(pm, ps, bm, bs), value)


def test_both_fixed_port_has_zero_effective_inverse_mass():
    _, ps, _, bs = scene()
    pm = ParticleModel(np.zeros(1))
    bm = RigidBodyModel(np.zeros(1), np.zeros((1, 3)))
    np.testing.assert_array_equal(
        ports().effective_inverse_mass(pm, ps, bm, bs), np.zeros((1, 3, 3))
    )


@pytest.mark.parametrize("empty_states", [False, True])
def test_empty_ports_and_empty_state_families(empty_states):
    pm, ps, bm, bs = scene()
    if empty_states:
        pm, ps = ParticleModel(np.empty(0)), ParticleState(np.empty((0, 3)), np.empty((0, 3)))
        bm = RigidBodyModel(np.empty(0), np.empty((0, 3)))
        bs = RigidBodyState(np.empty((0, 3)), np.empty((0, 4)), np.empty((0, 3)), np.empty((0, 3)))
    port = ports([], [])
    result = port.apply_impulses(pm, ps, bm, bs, np.empty((0, 3)))
    assert result.accepted and result.ledger is not None
    assert port.effective_inverse_mass(pm, ps, bm, bs).shape == (0, 3, 3)
    np.testing.assert_array_equal(result.particle_state.velocity, ps.velocity)
    np.testing.assert_array_equal(result.rigid_state.angular_velocity, bs.angular_velocity)


@pytest.mark.parametrize(
    "bad", ["particle_index", "body_index", "rows", "dtype", "int", "nan", "family"]
)
def test_invalid_exchange_input_is_rejected_before_writing(bad):
    pm, ps, bm, bs = scene()
    port, impulse = ports(), np.zeros((1, 3))
    if bad == "particle_index":
        port = ports([1], [0])
    elif bad == "body_index":
        port = ports([0], [1])
    elif bad == "rows":
        impulse = np.zeros((2, 3))
    elif bad == "dtype":
        impulse = np.zeros((1, 3), np.float32)
    elif bad == "int":
        impulse = np.zeros((1, 3), np.int64)
    elif bad == "nan":
        impulse[:] = np.nan
    else:
        bs = ps
    with pytest.raises((ValueError, TypeError)):
        port.apply_impulses(pm, ps, bm, bs, impulse)


@pytest.mark.parametrize("failure", ["particle", "rigid_angular"])
def test_numeric_failure_returns_both_original_owned_states(failure):
    pm, ps, bm, bs = scene(np.float32)
    if failure == "particle":
        pm = ParticleModel(np.array([10.0], np.float32), np.zeros(3))
    else:
        bm = RigidBodyModel(bm.inverse_mass, np.full((1, 3), 1e-30, np.float32), np.zeros(3))
    result = ports().apply_impulses(pm, ps, bm, bs, np.array([[0, 1e38, 0]], np.float32))
    assert not result.accepted and result.ledger is None and result.reason
    for actual, original in [(result.particle_state, ps), (result.rigid_state, bs)]:
        np.testing.assert_array_equal(actual.position, original.position)
        np.testing.assert_array_equal(actual.velocity, original.velocity)
        assert not np.shares_memory(actual.position, original.position)
        assert not np.shares_memory(actual.velocity, original.velocity)
    np.testing.assert_array_equal(result.rigid_state.angular_velocity, bs.angular_velocity)


def test_unrepresentable_effective_mass_is_explicit_solver_failure():
    pm, ps, bm, bs = scene()
    bm = RigidBodyModel(bm.inverse_mass, np.full((1, 3), np.nextafter(0.0, 1.0)), np.zeros(3))
    with pytest.raises(SolverFailure):
        ports().effective_inverse_mass(pm, ps, bm, bs)


def test_success_keeps_input_buffers_and_returns_owned_state_and_ledger():
    pm, ps, bm, bs = scene()
    impulse = np.array([[0.0, 2.0, 0.0]])
    before_particle, before_body = ps.copy(), bs.copy()
    result = ports().apply_impulses(pm, ps, bm, bs, impulse)
    assert result.accepted
    impulse[:] = 9
    result.particle_state.position[:] = 99
    result.rigid_state.angular_velocity[:] = 88
    np.testing.assert_array_equal(ps.position, before_particle.position)
    np.testing.assert_array_equal(ps.velocity, before_particle.velocity)
    np.testing.assert_array_equal(bs.angular_velocity, before_body.angular_velocity)
    np.testing.assert_array_equal(result.ledger.particle, [[0, 2, 0]])


def test_two_distinct_particle_points_accumulate_force_and_torque_on_one_body():
    _, _, bm, bs = scene()
    pm = ParticleModel(np.array([0.5, 1.0]), np.zeros(3))
    ps = ParticleState(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.zeros((2, 3)))
    bs.velocity[:] = bs.angular_velocity[:] = 0
    result = ports([0, 1], [0, 0]).apply_impulses(
        pm, ps, bm, bs, np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 0.0]])
    )
    assert result.accepted
    np.testing.assert_allclose(result.particle_state.velocity, [[0, 1, 0], [3, 0, 0]])
    np.testing.assert_allclose(result.rigid_state.velocity, [[-0.75, -0.5, 0]])
    np.testing.assert_allclose(result.rigid_state.angular_velocity, [[0, 0, 1 / 3]])
    np.testing.assert_array_equal(result.ledger.rigid_angular, [[0, 0, 1]])


def test_extension_copy_override_cannot_leak_original_buffers_on_failure():
    class ParticleAlias(ParticleState):
        def copy(self):
            return self

    class RigidAlias(RigidBodyState):
        def copy(self):
            return self

    pm, ps, bm, bs = scene(np.float32)
    ps = ParticleAlias(ps.position, ps.velocity)
    bs = RigidAlias(bs.position, bs.orientation, bs.velocity, bs.angular_velocity)
    pm = ParticleModel(np.array([10.0], np.float32), np.zeros(3))
    result = ports().apply_impulses(pm, ps, bm, bs, np.array([[0, 1e38, 0]], np.float32))
    assert not result.accepted
    assert (
        type(result.particle_state) is ParticleState and type(result.rigid_state) is RigidBodyState
    )
    result.particle_state.velocity[:] = 99
    result.rigid_state.angular_velocity[:] = 88
    np.testing.assert_allclose(ps.velocity, [[0.2, 0.3, 0.4]])
    np.testing.assert_allclose(bs.angular_velocity, [[0.2, 0.3, 0.4]])


def test_mixed_state_precision_is_rejected():
    pm, ps, _, _ = scene(np.float32)
    _, _, bm, bs = scene(np.float64)
    with pytest.raises(ValueError, match="dtype"):
        ports().apply_impulses(pm, ps, bm, bs, np.zeros((1, 3), np.float32))


def test_tiny_inertia_ratio_can_be_finite_without_constructing_its_inverse():
    smallest = np.nextafter(0.0, 1.0)
    _, ps, _, bs = scene()
    pm = ParticleModel(np.zeros(1), np.zeros(3))
    bm = RigidBodyModel(np.ones(1), np.full((1, 3), smallest), np.zeros(3))
    bs.velocity[:] = bs.angular_velocity[:] = 0
    result = ports().apply_impulses(pm, ps, bm, bs, np.array([[0.0, smallest, 0.0]]))
    assert result.accepted
    np.testing.assert_array_equal(result.rigid_state.angular_velocity, [[0, 0, -1]])


def test_fixed_body_effective_response_does_not_need_unrepresentable_lever_arm():
    pm, ps, _, bs = scene()
    bm = RigidBodyModel(np.zeros(1), np.zeros((1, 3)))
    ps.position[:] = 1e308
    bs.position[:] = -1e308
    np.testing.assert_array_equal(ports().effective_inverse_mass(pm, ps, bm, bs), [np.eye(3) * 0.5])


def test_zero_impulse_is_noop_even_when_unused_lever_arm_is_unrepresentable():
    pm, ps, bm, bs = scene()
    ps.position[:] = 1e308
    bs.position[:] = -1e308
    result = ports().apply_impulses(pm, ps, bm, bs, np.zeros((1, 3)))
    assert result.accepted
    np.testing.assert_array_equal(result.particle_state.velocity, ps.velocity)
    np.testing.assert_array_equal(result.rigid_state.angular_velocity, bs.angular_velocity)


def test_com_point_effective_mass_ignores_unrepresentable_inverse_inertia():
    _, ps, _, bs = scene()
    pm = ParticleModel(np.ones(1))
    bm = RigidBodyModel(np.ones(1), np.full((1, 3), np.nextafter(0.0, 1.0)))
    ps.position[:] = bs.position[:] = 0
    np.testing.assert_array_equal(ports().effective_inverse_mass(pm, ps, bm, bs), [np.eye(3) * 2])


def test_small_lever_and_inertia_can_have_finite_effective_mass():
    from decimal import Decimal, localcontext

    _, ps, _, bs = scene()
    lever, inertia = 1e-160, 1e-320
    pm = ParticleModel(np.ones(1))
    bm = RigidBodyModel(np.ones(1), np.full((1, 3), inertia))
    ps.position[:] = [lever, 0, 0]
    with localcontext() as context:
        context.prec = 60
        rotation_term = float(Decimal.from_float(lever) ** 2 / Decimal.from_float(inertia))
    actual = ports().effective_inverse_mass(pm, ps, bm, bs)
    np.testing.assert_allclose(
        actual, [np.diag([2, 2 + rotation_term, 2 + rotation_term])], rtol=1e-14, atol=0
    )
