"""Independent convex support/material and stable-identity controls."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F

import numpy as np
import pytest

from phymeo.geometry._prism_pair_exact import PairField, PrismPairProofLimits
from phymeo.geometry.prism_prism_manifold import PrismPairIdentity, prism_prism_manifold

TRIANGLE = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])


def identity(faces=(11, 29), nodes=((2, 7, 9), (12, 17, 19))):
    return PrismPairIdentity(
        "cloth", np.array(faces, dtype=np.int64), np.array(nodes, dtype=np.int64)
    )


def manifold(a, b, rho=(0.5, 0.5), ids=None, limits=None):
    return prism_prism_manifold(
        a,
        b,
        identity=identity() if ids is None else ids,
        half_thickness_m=np.array(rho, dtype=np.float64),
        proof_limits=PrismPairProofLimits() if limits is None else limits,
    )


def axis_z(result):
    return next(axis for axis in result.maximal_axes if axis.direction == (0, 0, -1))


def assert_exact_material(result, triangles, rho):
    # Independently reconstruct with the input Gram geometry and field roots.
    for axis in result.maximal_axes:
        for contact in axis.points:
            for point, material, triangle, thickness in zip(
                (contact.point_a, contact.point_b),
                (contact.material_a, contact.material_b),
                triangles,
                rho,
                strict=True,
            ):
                coordinates = tuple(tuple(F.from_float(float(v)) for v in row) for row in triangle)
                e = tuple(coordinates[1][i] - coordinates[0][i] for i in range(3))
                h = tuple(coordinates[2][i] - coordinates[0][i] for i in range(3))
                raw = (
                    e[1] * h[2] - e[2] * h[1],
                    e[2] * h[0] - e[0] * h[2],
                    e[0] * h[1] - e[1] * h[0],
                )
                length = point[0].field.sqrt(sum(x * x for x in raw))
                normal = tuple(v / length for v in raw)
                assert sum(material[:3]) == 1
                assert all(w >= 0 for w in material[:3])
                assert -F(thickness) <= material[3] <= F(thickness)
                rebuilt = tuple(
                    sum(material[j] * coordinates[j][i] for j in range(3)) + material[3] * normal[i]
                    for i in range(3)
                )
                assert rebuilt == point


def test_independent_radicals_support_formula_is_not_a_sampled_oracle():
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    b = np.array([[0.0, 0.0, 4.0], [1.0, -1.0, 4.0], [0.0, 1.0, 3.0]])
    result = manifold(a, b, (0.25, 0.125))
    axis = next(row for row in result.axes if row.direction == (0, 0, -1))
    field = PairField(2, 3)
    expected = 2 - field.sqrt(2) / 8 - field.sqrt(3) / 24
    assert axis.numerator == expected
    assert expected > F(83, 48)
    assert result.sat_complete
    assert result.exact_gap_bounds[0] > 0
    assert result.status in ("separated", "unresolved")
    if result.status == "unresolved":
        assert result.obligation.stage == "clipping"


@pytest.mark.parametrize("kind,count", [("point", 1), ("edge", 2), ("face", 3)])
def test_exact_touching_complete_point_segment_polygon_extrema(kind, count):
    b = TRIANGLE.copy()
    if kind == "point":
        b += np.array([2.0, 0.0, 1.0])
    elif kind == "edge":
        b[:, 1] *= -1
        b[:, 2] += 1
    else:
        b[:, 2] += 1
    result = manifold(TRIANGLE, b)
    assert result.status == "touching"
    assert result.exact_gap_bounds == (0, 0)
    assert len(axis_z(result).points) == count
    assert all(axis.points for axis in result.maximal_axes)
    assert_exact_material(result, (TRIANGLE, b), (0.5, 0.5))


def test_penetration_keeps_both_opposite_maximal_ties():
    result = manifold(TRIANGLE, TRIANGLE.copy(), (0.25, 0.25))
    assert result.status == "penetrating"
    assert result.exact_gap_bounds == (-F(1, 2), -F(1, 2))
    assert {axis.direction for axis in result.maximal_axes} == {(0, 0, -1), (0, 0, 1)}
    assert all(len(axis.points) == 3 for axis in result.maximal_axes)
    assert not hasattr(result, "primary_axis")
    assert_exact_material(result, (TRIANGLE, TRIANGLE), (0.25, 0.25))


def test_unequal_thickness_exact_face_contact():
    b = TRIANGLE + np.array([0.0, 0.0, 0.75])
    result = manifold(TRIANGLE, b, (0.25, 0.5))
    assert result.status == "touching"
    assert all(
        point.material_a[3] == F(1, 4) and point.material_b[3] == -F(1, 2)
        for point in axis_z(result).points
    )


def test_exchange_preserves_keys_and_swaps_oriented_receipts():
    b = TRIANGLE + np.array([0.0, 0.0, 0.75])
    forward = manifold(TRIANGLE, b, (0.25, 0.5))
    backward = manifold(b, TRIANGLE, (0.5, 0.25), identity((29, 11), ((12, 17, 19), (2, 7, 9))))
    assert forward.fingerprint == backward.fingerprint
    assert forward.caller_fingerprint != backward.caller_fingerprint
    assert backward.caller_to_canonical == (1, 0)
    assert [axis.key for axis in forward.maximal_axes] == [
        axis.key for axis in backward.maximal_axes
    ]
    for left, right in zip(forward.maximal_axes, backward.maximal_axes, strict=True):
        assert left.direction == tuple(-x for x in right.direction)
        for p, q in zip(left.points, right.points, strict=True):
            assert p.key == q.key
            assert (p.point_a, p.point_b, p.material_a, p.material_b) == (
                q.point_b,
                q.point_a,
                q.material_b,
                q.material_a,
            )


def test_cyclic_vertex_reorder_preserves_keys_and_remaps_material():
    b = TRIANGLE + np.array([0.0, 0.0, 1.0])
    original = manifold(TRIANGLE, b)
    permutation = [1, 2, 0]
    reordered = manifold(TRIANGLE[permutation], b, ids=identity(nodes=((7, 9, 2), (12, 17, 19))))
    assert reordered.fingerprint == original.fingerprint
    assert reordered.caller_fingerprint != original.caller_fingerprint
    for left, right in zip(axis_z(original).points, axis_z(reordered).points, strict=True):
        assert left.key == right.key
        assert tuple(left.material_a[i] for i in permutation) == right.material_a[:3]
    reverse = manifold(TRIANGLE[[0, 2, 1]], b, ids=identity(nodes=((2, 9, 7), (12, 17, 19))))
    assert reverse.fingerprint != original.fingerprint
    assert reverse.status == original.status
    assert_exact_material(reverse, (TRIANGLE[[0, 2, 1]], b), (0.5, 0.5))


@pytest.mark.parametrize("scale", [2.0**-400, 1.0, 2.0**400])
def test_exact_binary_extremes_and_negative_translation(scale):
    a = (TRIANGLE + np.array([-4.0, -8.0, -16.0])) * scale
    b = a + np.array([0.0, 0.0, scale])
    result = manifold(a, b, (0.5 * scale, 0.5 * scale))
    assert result.status == "touching"
    assert len(axis_z(result).points) == 3
    assert_exact_material(result, (a, b), (0.5 * scale, 0.5 * scale))


def test_proper_signed_coordinate_transform_preserves_gap_and_extrema():
    # (x,y,z) -> (-y,x,z) is a proper rotation with exact binary entries.
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    b = TRIANGLE + np.array([0.0, 0.0, 1.0])
    old = manifold(TRIANGLE, b)
    new = manifold(TRIANGLE @ rotation.T, b @ rotation.T)
    assert new.status == old.status
    assert new.exact_gap_bounds == old.exact_gap_bounds
    expected = {(-p.point_a[1], p.point_a[0], p.point_a[2]) for p in axis_z(old).points}
    assert {p.point_a for p in axis_z(new).points} == expected


def test_identity_and_evidence_own_inputs_without_incidence_exemption():
    faces = np.array([11, 29], dtype=np.int64)
    nodes = np.array([[2, 7, 9], [2, 7, 9]], dtype=np.int64)
    ids = PrismPairIdentity("cloth", faces, nodes)
    result = manifold(TRIANGLE, TRIANGLE.copy(), (0.25, 0.25), ids)
    assert result.status == "penetrating"
    faces[:] = 0
    nodes[:] = 0
    assert tuple(ids.face_ids) == (11, 29)
    with pytest.raises(ValueError):
        ids.node_ids.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        result.status = "separated"
    with pytest.raises(ValueError, match="shared node"):
        manifold(TRIANGLE, TRIANGLE + 1, ids=ids)


def test_fingerprints_bind_identity_orientation_geometry_thickness_policy():
    b = TRIANGLE + np.array([0.0, 0.0, 1.0])
    baseline = manifold(TRIANGLE, b)
    alternatives = [
        manifold(TRIANGLE, b, ids=identity((10, 29))),
        manifold(TRIANGLE, b + 1),
        manifold(TRIANGLE, b, (0.25, 0.5)),
        manifold(TRIANGLE, b, limits=PrismPairProofLimits(max_time_depth=31)),
    ]
    assert all(item.fingerprint != baseline.fingerprint for item in alternatives)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 3), dtype=np.float64),
        np.zeros((3, 3), dtype=np.float32),
        np.full((3, 3), np.nan),
        [[0.0, 0.0, 0.0]] * 3,
    ],
)
def test_invalid_coordinate_shapes_dtype_nonfinite_reject(bad):
    with pytest.raises((TypeError, ValueError)):
        manifold(bad, TRIANGLE)


def test_invalid_identity_and_thickness_reject():
    with pytest.raises(ValueError):
        identity((11, 11))
    with pytest.raises(ValueError):
        identity(nodes=((2, 2, 9), (12, 17, 19)))
    with pytest.raises(ValueError):
        identity(nodes=((-2, 7, 9), (12, 17, 19)))
    with pytest.raises(TypeError):
        PrismPairIdentity("cloth", np.array([1.0, 2.0]), np.zeros((2, 3), dtype=np.int64))
    for rho in ((0.0, 1.0), (-1.0, 1.0), (float("inf"), 1.0)):
        with pytest.raises(ValueError):
            manifold(TRIANGLE, TRIANGLE, rho)


def test_degenerate_geometry_is_unresolved_never_separated():
    result = manifold(np.zeros((3, 3), dtype=np.float64), TRIANGLE + 10)
    assert result.status == "unresolved" and not result.sat_complete
    assert result.obligation is not None


def test_two_independent_normals_exact_edge_edge_touch_material_endpoints():
    # d=(-2,1,1) is perpendicular to both raw normals. Its extremal edges
    # meet at (0,-2,4): A2+3*cA = B1-cB. Thus offsets are 3*sqrt(2),-sqrt(3).
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    b = np.array([[0.0, 0.0, 5.0], [1.0, -1.0, 5.0], [0.0, 1.0, 4.0]])
    result = manifold(a, b, (5.0, 2.0))
    assert result.status == "touching"
    selected = next(
        axis for axis in result.maximal_axes if axis.direction == (1, -F(1, 2), -F(1, 2))
    )
    assert len(selected.points) == 1
    point = selected.points[0]
    field = PairField(2, 3)
    assert point.point_a == point.point_b == tuple(field.scalar(v) for v in (0, -2, 4))
    assert point.material_a == (0, 0, 1, 3 * field.sqrt(2))
    assert point.material_b == (0, 1, 0, -field.sqrt(3))
    assert_exact_material(result, (a, b), (5.0, 2.0))


def test_static_common_root_reduction_and_exact_normalized_tie():
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    b = 2 * a
    result = manifold(a, b, (0.25, 0.25))
    assert result.status == "penetrating"
    assert len(result.maximal_axes) == 2
    # Raw area norms squared2 and32 generate the same reduced quadratic field.
    assert all(axis.numerator.field.roots == (F(2),) for axis in result.axes)
    for axis in result.maximal_axes:
        assert axis.numerator**2 == axis.norm_squared * F(1, 4)
        assert axis.numerator < 0
        assert len(axis.points) == 3
    assert_exact_material(result, (a, b), (0.25, 0.25))
