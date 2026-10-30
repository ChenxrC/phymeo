"""Independent same-domain response and audit tests; execution is deferred."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from phymeo import ParticleModel, ParticleState, SolverFailure
from phymeo.coupling.cloth_self_contact import SelfContactBinding, make_self_contact_operator
from phymeo.coupling.domain_graph import MultiDomainPointOperator
from phymeo.coupling.ports import ParticlePointPort


X = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
T = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64)
B = np.array([[0.25, 0.25, 0.5], [0.25, 0.25, 0.5]])
O = np.array([0.25, -0.25])
P = np.array([[0, 1]], dtype=np.int64)


class CountedParticlePort(ParticlePointPort):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests = []

    def apply_impulses(self, impulses):
        self.requests.append(impulses.copy())
        return super().apply_impulses(impulses)


class RejectCandidatePort(CountedParticlePort):
    def validate_candidate(self, state):
        super().validate_candidate(state)
        raise SolverFailure("deliberate candidate rejection")


def base(x=X, velocity=None, inverse_mass=None, cls=CountedParticlePort):
    velocity = np.zeros_like(x) if velocity is None else velocity
    inverse_mass = np.ones(len(x)) if inverse_mass is None else inverse_mass
    return cls(
        ParticleModel(inverse_mass, np.zeros(3)), ParticleState(x, velocity),
        np.arange(len(x), dtype=np.int64), domain="cloth",
    )


def binding(source=None, triangles=T, weights=B, offsets=O, pairs=P):
    return make_self_contact_operator(
        base() if source is None else source, triangles, weights, offsets, pairs
    )


def unit_triangle_blocks(weights, offset):
    # At this particular unit XY triangle, dn/dz_i is exactly (1,1), (-1,0), (0,-1).
    result = np.array([weight * np.eye(3) for weight in weights])
    result[0, 0, 2] += offset
    result[0, 1, 2] += offset
    result[1, 0, 2] -= offset
    result[2, 1, 2] -= offset
    return result


def independent_unit_map(weights, offsets, pairs):
    rows = [unit_triangle_blocks(w, o).transpose(1, 0, 2).reshape(3, 9)
            for w, o in zip(weights, offsets, strict=True)]
    return np.concatenate([rows[a] - rows[b] for a, b in pairs])


def material_point(nodes, weights, offset):
    area = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])
    return weights @ nodes + offset * area / np.linalg.norm(area)


def independent_difference_map(x, triangles, weights, offsets, pairs):
    def relative(points):
        anchors = np.array([material_point(points[t], w, o)
                            for t, w, o in zip(triangles, weights, offsets, strict=True)])
        return (anchors[pairs[:, 0]] - anchors[pairs[:, 1]]).reshape(-1)

    result = np.empty((3 * len(pairs), x.size))
    epsilon = 1e-6
    for dof in range(x.size):
        plus, minus = x.copy(), x.copy()
        plus.flat[dof] += epsilon
        minus.flat[dof] -= epsilon
        result[:, dof] = (relative(plus) - relative(minus)) / (2 * epsilon)
    return result


def test_single_base_apply_and_separated_point_actual_moment():
    source = base(inverse_mass=np.array([1.0, 2.0, 4.0]))
    receipt = binding(source)
    assert isinstance(receipt, SelfContactBinding)
    assert isinstance(receipt.operator, MultiDomainPointOperator)
    assert receipt.operator.domains == ("cloth",)
    assert receipt.operator.left == (("cloth", 0),)
    assert receipt.operator.right == (("cloth", 1),)
    impulse = np.array([[1.0, 2.0, 0.0]])
    expected_map = independent_unit_map(B, O, P)
    loads = receipt.node_impulses(impulse)
    np.testing.assert_allclose(loads.reshape(-1), expected_map.T @ impulse.reshape(-1), atol=1e-15)
    assert source.requests == []  # Audit computation never applies a state.
    states = receipt.operator.apply_impulses(impulse)
    assert len(source.requests) == 1
    np.testing.assert_array_equal(source.requests[0], loads)
    np.testing.assert_allclose(states["cloth"].velocity, loads * [[1.0], [2.0], [4.0]])
    np.testing.assert_allclose(loads.sum(axis=0), 0, atol=1e-15)
    moment = np.cross(X, loads).sum(axis=0)
    np.testing.assert_allclose(moment, [-1.0, 0.5, 0.0], atol=1e-15)
    np.testing.assert_allclose(
        moment, np.cross(receipt.world_points[0] - receipt.world_points[1], impulse[0]), atol=1e-15
    )
    np.testing.assert_array_equal(source.snapshot().velocity, np.zeros((3, 3)))


def test_coincident_opposite_requests_cancel_without_duplicate_state_advance():
    source = base()
    receipt = binding(source, offsets=np.array([0.25, 0.25]))
    impulse = np.array([[0.5, -0.25, 0.75]])
    np.testing.assert_array_equal(receipt.node_impulses(impulse), np.zeros((3, 3)))
    np.testing.assert_array_equal(receipt.operator.response(impulse), np.zeros((1, 3)))
    after = receipt.operator.apply_impulses(impulse)["cloth"]
    assert len(source.requests) == 1
    np.testing.assert_array_equal(after.velocity, np.zeros((3, 3)))
    np.testing.assert_array_equal(after.position, X)


def test_reversed_and_identical_pairs_cancel_before_jacobian_products():
    source = base()
    receipt = binding(source, pairs=np.array([[0, 1], [1, 0], [1, 1]]))
    impulse = np.array([[0.5, -0.25, 0.75], [0.5, -0.25, 0.75], [1.79e308] * 3])
    np.testing.assert_array_equal(receipt.node_impulses(impulse), np.zeros((3, 3)))
    np.testing.assert_array_equal(receipt.operator.apply_impulses(impulse)["cloth"].velocity, 0)
    assert len(source.requests) == 1
    np.testing.assert_array_equal(receipt.pairs, [[0, 1], [1, 0], [1, 1]])


def test_full_shared_node_off_diagonal_response_and_kinetic_work():
    weights = np.array([[0.25, 0.25, 0.5], [0.5, 0.25, 0.25], [0.25, 0.5, 0.25]])
    offsets = np.array([0.25, -0.125, 0.375])
    pairs = np.array([[0, 1], [1, 2]])
    triangles = np.repeat(T[:1], 3, axis=0)
    inverse_mass = np.array([0.5, 1.25, 2.0])
    velocity = np.array([[0.2, -0.1, 0.3], [-0.3, 0.5, 0.1], [0.1, 0.2, -0.4]])
    source = base(velocity=velocity, inverse_mass=inverse_mass)
    receipt = binding(source, triangles, weights, offsets, pairs)
    d = independent_unit_map(weights, offsets, pairs)
    mobility = np.repeat(inverse_mass, 3)
    response = (d * mobility) @ d.T
    assert np.linalg.norm(response[:3, 3:]) > 0.01
    impulse = np.array([[0.4, -0.2, 0.3], [-0.1, 0.5, -0.4]])
    loads = (d.T @ impulse.reshape(-1)).reshape(-1, 3)
    expected_delta = (response @ impulse.reshape(-1)).reshape(-1, 3)
    np.testing.assert_allclose(receipt.node_impulses(impulse), loads, atol=2e-15)
    np.testing.assert_allclose(receipt.operator.response(impulse), expected_delta, atol=2e-15)
    np.testing.assert_allclose(
        receipt.operator.relative_velocities(), (d @ velocity.reshape(-1)).reshape(-1, 3)
    )
    candidate = receipt.operator.apply_impulses(impulse)
    assert len(source.requests) == 1
    actual = candidate["cloth"].velocity
    np.testing.assert_allclose(actual, velocity + inverse_mass[:, None] * loads, atol=2e-15)
    kinetic_change = np.sum((actual**2 - velocity**2) / inverse_mass[:, None]) / 2
    work = np.sum(impulse * (receipt.operator.relative_velocities() + expected_delta / 2))
    assert kinetic_change == pytest.approx(work, abs=3e-15)
    audit = receipt.operator.audit_candidates(candidate, impulses=impulse)
    assert audit.energy_change == pytest.approx(kinetic_change, abs=3e-15)
    np.testing.assert_array_equal(source.snapshot().velocity, velocity)
    np.testing.assert_allclose(
        np.sum(loads * velocity),
        np.sum(impulse * receipt.operator.relative_velocities()),
        atol=2e-15,
    )


def test_adjacent_triangles_match_independent_geometric_differences():
    x = np.array([[0.0, 0, 0], [1, 0, 0.1], [0, 1, -0.1], [1, 1, 0.2]])
    triangles = np.array([[0, 1, 2], [1, 3, 2], [0, 1, 2]])
    weights = np.array([[0.25, 0.25, 0.5], [0.5, 0.25, 0.25], [0.25, 0.5, 0.25]])
    offsets, pairs = np.array([0.125, -0.25, 0.25]), np.array([[0, 1], [1, 2]])
    source = base(x, inverse_mass=np.array([1.0, 0.5, 2.0, 1.5]))
    receipt = binding(source, triangles, weights, offsets, pairs)
    d = independent_difference_map(x, triangles, weights, offsets, pairs)
    impulse = np.array([[0.4, 0.1, -0.3], [-0.2, 0.5, 0.1]])
    expected_loads = (d.T @ impulse.reshape(-1)).reshape(-1, 3)
    np.testing.assert_allclose(receipt.node_impulses(impulse), expected_loads, atol=1e-10, rtol=2e-9)
    expected = d @ (np.repeat([1.0, 0.5, 2.0, 1.5], 3) * expected_loads.reshape(-1))
    np.testing.assert_allclose(
        receipt.operator.response(impulse).reshape(-1), expected, atol=2e-10, rtol=3e-9
    )


def test_binding_owns_immutable_receipts_and_preserves_input_arrays():
    triangles, weights, offsets, pairs = T.copy(), B.copy(), O.copy(), P.copy()
    receipt = binding(triangles=triangles, weights=weights, offsets=offsets, pairs=pairs)
    expected = {name: getattr(receipt, name) for name in
                ("triangles", "weights", "offsets", "pairs", "world_points", "jacobians")}
    triangles[:] = 0
    weights[:] = 0
    offsets[:] = 99
    pairs[:] = 0
    for name, original in expected.items():
        exposed = getattr(receipt, name)
        assert not exposed.flags.writeable and exposed.flags.owndata
        with pytest.raises(ValueError):
            exposed.flat[0] = 99
        exposed.setflags(write=True)
        exposed[:] = 99
        np.testing.assert_array_equal(getattr(receipt, name), original)
    with pytest.raises(FrozenInstanceError):
        receipt.operator = None
    impulse = np.array([[0.5, 0.25, -0.75]])
    result = receipt.node_impulses(impulse)
    assert result.dtype == np.float64 and result.flags.owndata and not result.flags.writeable
    result.setflags(write=True)
    result[:] = 0
    np.testing.assert_array_equal(impulse, [[0.5, 0.25, -0.75]])
    assert np.linalg.norm(receipt.node_impulses(impulse)) > 0


def test_receipt_records_the_public_ports_actual_normalized_weights():
    weights = B.copy()
    weights[0, 0] += 2e-13
    original = weights.copy()
    receipt = binding(weights=weights)
    np.testing.assert_array_equal(weights, original)
    assert receipt.weights[0, 0] != original[0, 0]
    assert receipt.weights[0].sum() == pytest.approx(1.0, abs=1e-15)
    np.testing.assert_allclose(
        receipt.jacobians[0], unit_triangle_blocks(receipt.weights[0], O[0]), atol=1e-15
    )


def test_late_candidate_rejection_publishes_no_state_and_preserves_snapshot():
    source = base(cls=RejectCandidatePort)
    receipt = binding(source)
    before = source.snapshot()
    with pytest.raises(SolverFailure, match="deliberate"):
        receipt.operator.apply_impulses(np.array([[0.5, -0.25, 0.75]]))
    assert len(source.requests) == 1
    np.testing.assert_array_equal(source.snapshot().position, before.position)
    np.testing.assert_array_equal(source.snapshot().velocity, before.velocity)


@pytest.mark.parametrize(
    "field,value",
    [("triangles", np.array([[0, 1, 3], [0, 1, 2]])),
     ("triangles", np.array([[0.0, 1, 2], [0, 1, 2]])),
     ("triangles", np.array([[0, 1, 2**63], [0, 1, 2]], dtype=np.uint64)),
     ("triangles", np.array([[0, 1, 1], [0, 1, 2]])),
     ("weights", B.astype(np.float32)), ("weights", np.full((2, 3), np.nan)),
     ("weights", np.full((2, 3), 1.0)), ("weights", np.array([[-0.25, 0.5, 0.75], B[1]])),
     ("offsets", O.astype(np.float32)), ("offsets", np.array([np.inf, 0.0])),
     ("offsets", np.zeros((2, 1))),
     ("pairs", np.array([[0.0, 1.0]])), ("pairs", np.array([[False, True]])),
     ("pairs", np.array([[0, -1]])), ("pairs", np.array([[0, 2]])),
     ("pairs", np.array([[0, 2**63]], dtype=np.uint64)), ("pairs", np.array([0, 1]))],
)
def test_invalid_attachments_are_rejected_before_any_apply(field, value):
    source = base()
    with pytest.raises((TypeError, ValueError, SolverFailure)):
        binding(source, **{field: value})
    assert source.requests == []
    np.testing.assert_array_equal(source.snapshot().velocity, np.zeros((3, 3)))


@pytest.mark.parametrize("method", ["node_impulses", "response", "apply_impulses"])
@pytest.mark.parametrize(
    "impulses",
    [np.ones((1, 3), dtype=np.float32), np.ones((2, 3)), np.full((1, 3), np.nan)],
)
def test_invalid_impulses_do_not_change_source(method, impulses):
    source = base()
    receipt = binding(source)
    target = receipt if method == "node_impulses" else receipt.operator
    with pytest.raises((TypeError, ValueError)):
        getattr(target, method)(impulses)
    assert source.requests == []


def test_nonrepresentable_audit_request_is_rejected_without_applying():
    source = base()
    receipt = binding(source, offsets=np.array([1000.0, -1000.0]))
    with pytest.raises(SolverFailure, match="represent"):
        receipt.node_impulses(np.full((1, 3), 1.79e308))
    assert source.requests == []
    np.testing.assert_array_equal(source.snapshot().velocity, np.zeros((3, 3)))


def test_empty_attachments_and_pairs_preserve_one_base_candidate():
    source = base()
    receipt = binding(source, np.empty((0, 3), dtype=np.int64), np.empty((0, 3)),
                      np.empty(0), np.empty((0, 2), dtype=np.int64))
    assert receipt.world_points.shape == (0, 3)
    assert receipt.jacobians.shape == (0, 3, 3, 3)
    assert receipt.operator.response(np.empty((0, 3))).shape == (0, 3)
    np.testing.assert_array_equal(receipt.node_impulses(np.empty((0, 3))), np.zeros((3, 3)))
    candidate = receipt.operator.apply_impulses(np.empty((0, 3)))
    assert len(source.requests) == 1
    np.testing.assert_array_equal(candidate["cloth"].velocity, np.zeros((3, 3)))


def test_compensated_pair_accumulation_retains_small_shared_node_request():
    source = base()
    pairs = np.repeat(P, 3, axis=0)
    receipt = binding(source, pairs=pairs)
    impulses = np.array([[1e16, 0.0, 0.0], [1.0, 0.0, 0.0], [-1e16, 0.0, 0.0]])
    expected = independent_unit_map(B, O, P).T @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_array_equal(receipt.node_impulses(impulses).reshape(-1), expected)
    candidate = receipt.operator.apply_impulses(impulses)
    assert len(source.requests) == 1
    np.testing.assert_array_equal(candidate["cloth"].velocity.reshape(-1), expected)


def test_fixed_node_requests_remain_visible_without_claiming_applied_momentum():
    source = base(inverse_mass=np.array([0.0, 1.0, 2.0]))
    receipt = binding(source)
    impulses = np.array([[1.0, 2.0, 0.0]])
    loads = receipt.node_impulses(impulses)
    assert np.linalg.norm(loads[0]) > 0
    after = receipt.operator.apply_impulses(impulses)["cloth"]
    np.testing.assert_array_equal(after.velocity[0], np.zeros(3))
    np.testing.assert_allclose(after.velocity[1:], loads[1:] * [[1.0], [2.0]])
    np.testing.assert_array_equal(source.requests[0], loads)


def test_fp32_base_storage_is_rejected():
    source = ParticlePointPort(
        ParticleModel(np.ones(3, dtype=np.float32), np.zeros(3, dtype=np.float32)),
        ParticleState(X.astype(np.float32), np.zeros((3, 3), dtype=np.float32)),
        np.arange(3), domain="cloth",
    )
    with pytest.raises(TypeError, match="float64"):
        binding(source)


def test_underflow_in_audit_products_is_rejected_without_applying():
    source = base()
    weights = np.array([[1.0, 1e-200, 0.0], [1.0, 0.0, 0.0]])
    receipt = binding(source, weights=weights, offsets=np.zeros(2))
    with pytest.raises(SolverFailure, match="represent"):
        receipt.node_impulses(np.array([[1e-200, 0.0, 0.0]]))
    assert source.requests == []
