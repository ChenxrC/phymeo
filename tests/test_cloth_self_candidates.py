"""Independent candidate-only controls; execution requires the parent queue grant."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from phymeo.constraints.cloth_self_contact import ClothSelfContactTopology
from phymeo.geometry.cloth_self_candidates import candidate_self_pairs


TRIANGLES = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
UNIT = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def topology(triangles=TRIANGLES, thickness=None):
    if thickness is None:
        thickness = np.full(len(triangles), 0.125)
    return ClothSelfContactTopology("cloth", triangles, thickness)


def two_layers(gap):
    return np.concatenate((UNIT, UNIT + [0.0, 0.0, gap]))


def test_unequal_thickness_equality_is_inclusive_without_epsilon():
    mesh = topology(thickness=np.array([0.25, 0.5]))
    assert candidate_self_pairs(mesh, two_layers(0.75)) == ((0, 1),)
    assert candidate_self_pairs(mesh, two_layers(np.nextafter(0.75, np.inf))) == ()
    assert candidate_self_pairs(mesh, two_layers(np.nextafter(0.75, -np.inf))) == ((0, 1),)


def test_affine_swept_union_finds_crossing_with_separated_endpoint_boxes():
    start, end = two_layers(2.0), two_layers(-2.0)
    mesh = topology()
    assert candidate_self_pairs(mesh, start) == ()
    assert candidate_self_pairs(mesh, end) == ()
    assert candidate_self_pairs(mesh, start, end) == ((0, 1),)
    assert candidate_self_pairs(mesh, end, start) == ((0, 1),)
    np.testing.assert_array_equal(start, two_layers(2.0))
    np.testing.assert_array_equal(end, two_layers(-2.0))


@pytest.mark.parametrize("translation", [[-128.0, -64.0, -32.0], [2.0**40, -(2.0**39), 0.0]])
def test_negative_and_large_translations_preserve_candidates(translation):
    mesh = topology()
    assert candidate_self_pairs(mesh, two_layers(0.25) + translation) == ((0, 1),)
    assert candidate_self_pairs(mesh, two_layers(0.5) + translation) == ()


def test_exact_bounds_do_not_overflow_or_round_a_small_gap_closed():
    # Exact endpoint + rho is outside FP64 here, but the bounds are rational.
    points = np.concatenate((UNIT + [1e308, 0, 0], UNIT + [-1e308, 0, 0]))
    mesh = topology(thickness=np.array([1e308, 1e308]))
    assert candidate_self_pairs(mesh, points) == ((0, 1),)
    # A rounded float expansion would lose this tiny positive separation.
    points = two_layers(1.0)
    points[3:, 2] = np.nextafter(1.0, np.inf)
    mesh = topology(thickness=np.array([0.5, 0.5]))
    assert candidate_self_pairs(mesh, points) == ()


def test_folded_strip_excludes_all_shared_vertices_and_retains_nonincident_faces():
    faces = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4], [4, 5, 6], [5, 7, 6]])
    points = np.array(
        [[0.0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0],
         [0, 0, 0.1], [0, 1, 0.1], [1, 0, 0.1], [1, 1, 0.1]]
    )
    mesh = topology(faces)
    assert mesh.incidence_policy == "exclude_shared_vertex"
    assert candidate_self_pairs(mesh, points) == ((0, 3), (0, 4), (0, 5), (1, 4), (1, 5), (2, 5))
    # These faces share only vertex 0, not an edge; they are still excluded.
    vertex_only = topology(np.array([[0, 1, 2], [0, 3, 4]]))
    assert candidate_self_pairs(vertex_only, np.zeros((5, 3))) == ()


def test_disconnected_overlapping_sheets_and_aabb_false_positive_remain_candidates():
    assert candidate_self_pairs(topology(), two_layers(0.0)) == ((0, 1),)
    # Mid-surface triangles have x+y <= 1 and x+y >= 1.75 respectively.
    # Their enclosing boxes overlap; S1 returns no collision or safety status.
    points = np.concatenate((UNIT, [[1.0, 1, 0], [0.75, 1, 0], [1, 0.75, 0]]))
    result = candidate_self_pairs(topology(thickness=np.array([0.01, 0.01])), points)
    assert type(result) is tuple
    assert result == ((0, 1),)


def test_face_order_and_complete_winding_reversal_have_deterministic_receipts():
    points = np.concatenate((UNIT, UNIT + [0, 0, 0.125], UNIT + [0, 0, 10.0]))
    faces = np.arange(9).reshape(3, 3)
    assert candidate_self_pairs(topology(faces), points) == ((0, 1),)
    permutation = np.array([2, 0, 1])
    reordered = topology(faces[permutation])
    assert candidate_self_pairs(reordered, points) == ((1, 2),)
    reversed_mesh = topology(faces[permutation, ::-1])
    assert candidate_self_pairs(reversed_mesh, points) == ((1, 2),)
    strip = np.array([[0, 1, 2], [1, 3, 2]])
    np.testing.assert_array_equal(topology(strip[:, ::-1]).triangles, strip[:, ::-1])


def test_intermediate_degeneracy_is_not_reported_safe():
    # The second triangle collapses halfway through this affine path.
    start = two_layers(0.0)
    end = start.copy()
    end[4], end[5] = start[5], start[4]
    assert candidate_self_pairs(topology(), start, end) == ((0, 1),)


def test_topology_owns_read_only_configuration():
    faces, rho = TRIANGLES.copy(), np.array([0.125, 0.25])
    mesh = topology(faces, rho)
    faces[:] = 0
    rho[:] = 99
    np.testing.assert_array_equal(mesh.triangles, TRIANGLES)
    np.testing.assert_array_equal(mesh.half_thickness_m, [0.125, 0.25])
    for name in ("triangles", "half_thickness_m"):
        exposed = getattr(mesh, name)
        assert not exposed.flags.writeable
        assert exposed.flags.owndata
        with pytest.raises(ValueError):
            exposed.flat[0] = 99
        exposed.setflags(write=True)
        exposed[:] = 99
    np.testing.assert_array_equal(mesh.triangles, TRIANGLES)
    np.testing.assert_array_equal(mesh.half_thickness_m, [0.125, 0.25])
    with pytest.raises(FrozenInstanceError):
        mesh.domain = "other"


@pytest.mark.parametrize(
    "faces,match",
    [
        (np.array([[0, 1, 1]]), "distinct"),
        (np.array([[0, 1, 2], [2, 0, 1]]), "duplicate"),
        (np.array([[0, 1, 2], [2, 1, 0]]), "duplicate"),
        (np.array([[0, 1, 2], [1, 2, 3]]), "orientation"),
        (np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]]), "manifold"),
    ],
)
def test_invalid_incidence_is_rejected(faces, match):
    with pytest.raises(ValueError, match=match):
        topology(faces)


@pytest.mark.parametrize(
    "field,value",
    [
        ("domain", ""), ("domain", " cloth"), ("domain", 1),
        ("triangles", np.array([[0.0, 1.0, 2.0]])),
        ("triangles", np.array([[False, True, False]])),
        ("triangles", np.array([[0, -1, 2]])),
        ("triangles", np.array([[0, 1, 2**63]], dtype=np.uint64)),
        ("triangles", np.array([0, 1, 2])),
        ("half_thickness_m", np.array([0.0, 0.125])),
        ("half_thickness_m", np.array([-0.125, 0.125])),
        ("half_thickness_m", np.array([np.inf, 0.125])),
        ("half_thickness_m", np.array([np.nan, 0.125])),
        ("half_thickness_m", np.array([0.125, 0.125], dtype=np.float32)),
        ("half_thickness_m", np.array([1, 1])),
        ("half_thickness_m", np.array([[0.125, 0.125]])),
        ("half_thickness_m", np.array([0.125])),
    ],
)
def test_invalid_topology_inputs(field, value):
    kwargs = dict(domain="cloth", triangles=TRIANGLES, half_thickness_m=np.array([0.125, 0.125]))
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError)):
        ClothSelfContactTopology(**kwargs)


@pytest.mark.parametrize(
    "points",
    [np.zeros((6, 3), dtype=np.float32), np.zeros((6, 3), dtype=int),
     np.zeros((6, 2)), np.zeros((5, 3)), np.zeros(18),
     np.full((6, 3), np.nan), np.full((6, 3), np.inf)],
)
def test_invalid_position_inputs(points):
    with pytest.raises((TypeError, ValueError)):
        candidate_self_pairs(topology(), points)
    with pytest.raises((TypeError, ValueError)):
        candidate_self_pairs(topology(), two_layers(1.0), points)


def test_endpoint_sizes_must_match_and_topology_type_is_explicit():
    with pytest.raises(ValueError):
        candidate_self_pairs(topology(), two_layers(1.0), np.zeros((7, 3)))
    with pytest.raises(TypeError):
        candidate_self_pairs(TRIANGLES, two_layers(1.0))


def test_empty_mesh_is_supported_without_bypassing_input_validation():
    mesh = topology(np.empty((0, 3), dtype=np.int64), np.empty(0))
    assert candidate_self_pairs(mesh, np.empty((0, 3))) == ()
    assert candidate_self_pairs(mesh, np.zeros((4, 3)), np.ones((4, 3))) == ()
    with pytest.raises(TypeError):
        candidate_self_pairs(mesh, np.zeros((0, 3), dtype=np.float32))
