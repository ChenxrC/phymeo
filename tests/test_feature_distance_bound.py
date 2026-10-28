"""Independent convex-hull oracle and exact binary64 endpoint checks."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from itertools import combinations

import numpy as np
import pytest

import phymeo as ph


def certificate(first, second, wa, wb):
    # Resolve at call time: a missing public API is an assertion, not collection error.
    assert callable(getattr(ph, "feature_distance_bounds", None)), "missing public certificate API"
    return ph.feature_distance_bounds(first, second, first_weights=wa, second_weights=wb)


def triangle():
    return np.array([[0.0, 0, 0], [2.0, 0, 0], [0.0, 2, 0]])


def exact_rows(points):
    return [tuple(F(float(x)) for x in row) for row in points]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def linear_solve(matrix, rhs):
    rows = [list(a) + [b] for a, b in zip(matrix, rhs)]
    for column in range(len(rhs)):
        pivot = next((i for i in range(column, len(rows)) if rows[i][column]), None)
        if pivot is None:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [x / divisor for x in rows[column]]
        for i in range(len(rows)):
            if i != column:
                factor = rows[i][column]
                rows[i] = [x - factor * y for x, y in zip(rows[i], rows[column])]
    return [row[-1] for row in rows]


def true_distance_squared(first, second):
    """Minimize norm on Minkowski hull by exact active-face enumeration.

    Caratheodory gives supports of size <=4 in R3. Singular supports have
    equivalent smaller supports, also enumerated. No production KKT helpers.
    """
    vertices = sorted(
        set(
            tuple(x - y for x, y in zip(a, b))
            for a in exact_rows(first)
            for b in exact_rows(second)
        )
    )
    candidates = []
    for size in range(1, min(4, len(vertices)) + 1):
        for support in combinations(vertices, size):
            matrix = [[dot(a, b) for b in support] + [F(1)] for a in support]
            matrix.append([F(1)] * size + [F(0)])
            solution = linear_solve(matrix, [F(0)] * size + [F(1)])
            if solution is not None and all(w >= 0 for w in solution[:size]):
                point = [sum(w * v[k] for w, v in zip(solution, support)) for k in range(3)]
                candidates.append(dot(point, point))
    return min(candidates)


def assert_encloses(result, squared):
    assert F(result.lower_m) ** 2 <= squared <= F(result.upper_m) ** 2
    assert type(result.lower_m) is float and type(result.upper_m) is float
    assert result.method == "convex-kkt-distance-v1"


def assert_tight(result, q, s):
    assert F(result.lower_m) ** 2 <= q
    assert F(result.upper_m) ** 2 >= s
    if result.lower_m < np.finfo(float).max:
        up = np.nextafter(result.lower_m, np.inf)
        assert F(float(up)) ** 2 > q
    if result.upper_m:
        down = np.nextafter(result.upper_m, 0.0)
        assert F(float(down)) ** 2 < s


def test_probe_plate_public_api_exact_and_scalar_record():
    result = certificate(np.array([[0.5, 0.5, 3.0]]), triangle(), [1.0], [0.5, 0.25, 0.25])
    assert isinstance(result, ph.FeatureDistanceBounds)
    assert "FeatureDistanceBounds" in ph.__all__ and "feature_distance_bounds" in ph.__all__
    assert result.lower_m == result.upper_m == 3.0
    assert_encloses(result, F(9))
    assert_tight(result, F(9), F(9))


def test_nonoptimal_probe_witness_has_exact_gap_and_adjacent_endpoints():
    point = np.array([[0.5, 0.5, 3.0]])
    result = certificate(point, triangle(), [1.0], [1.0, 0.0, 0.0])
    assert true_distance_squared(point, triangle()) == 9
    assert_encloses(result, F(9))
    assert_tight(result, F(15, 2), F(19, 2))


@pytest.mark.parametrize(
    "point,weights,squared",
    [
        ([0.5, 0.5, 0.0], [0.5, 0.25, 0.25], 0),
        ([0.5, -1.0, 1.0], [0.75, 0.25, 0.0], 2),
        ([2.0, 2.0, 1.0], [0.0, 0.5, 0.5], 3),
        ([-1.0, -1.0, 2.0], [1.0, 0.0, 0.0], 6),
    ],
)
def test_probe_face_edge_vertex_analytic(point, weights, squared):
    point = np.array([point])
    result = certificate(point, triangle(), [1.0], weights)
    assert true_distance_squared(point, triangle()) == squared
    assert_encloses(result, F(squared))
    assert_tight(result, F(squared), F(squared))


@pytest.mark.parametrize(
    "first,second,wa,wb,squared",
    [
        ([[-1.0, 0, 0], [1.0, 0, 0]], [[0.0, -1, 2], [0.0, 1, 2]], [0.5, 0.5], [0.5, 0.5], 4),
        ([[-1.0, 0, 0], [1.0, 0, 0]], [[0.0, -1, 0], [0.0, 1, 0]], [0.5, 0.5], [0.5, 0.5], 0),
        ([[0.0, 0, 0], [2.0, 0, 0]], [[0.0, 3, 0], [2.0, 3, 0]], [0.5, 0.5], [0.5, 0.5], 9),
        ([[0.0, 0, 0], [2.0, 0, 0]], [[1.0, 0, 0], [3.0, 0, 0]], [0.25, 0.75], [0.75, 0.25], 0),
        ([[0.0, 0, 0], [1.0, 0, 0]], [[2.0, 1, 0], [3.0, 1, 0]], [0.0, 1.0], [1.0, 0.0], 2),
        (
            [[0.0, 0, 0], [1.0, 0, 0]],
            [[0.0, 1, 0], [1.0, 1 + 2.0**-40, 0]],
            [1.0, 0.0],
            [1.0, 0.0],
            1,
        ),
        ([[0.0, 0, 0], [0.0, 0, 0]], [[2.0, 0, 0], [2.0, 0, 0]], [0.5, 0.5], [0.5, 0.5], 4),
    ],
)
def test_independent_rods_analytic_and_exact_convex_oracle(first, second, wa, wb, squared):
    a, b = np.array(first), np.array(second)
    result = certificate(a, b, wa, wb)
    assert true_distance_squared(a, b) == squared
    assert_encloses(result, F(squared))
    assert_tight(result, F(squared), F(squared))


@pytest.mark.parametrize(
    "plate,point,weights,squared",
    [
        ([[0.0, 0, 0]] * 3, [[0.0, 0, 2]], [1.0, 2.0, 3.0], 4),
        ([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]], [[1.0, 0, 2]], [0.25, 0.5, 0.25], 4),
        ([[0.0, 0, 0], [0.0, 0, 0], [2.0, 0, 0]], [[1.0, 0, 2]], [0.25, 0.25, 0.5], 4),
    ],
)
def test_degenerate_triangles_are_convex_hulls(plate, point, weights, squared):
    a, b = np.array(point), np.array(plate)
    result = certificate(a, b, [1.0], weights)
    assert_encloses(result, true_distance_squared(a, b))
    assert_tight(result, F(squared), F(squared))


@pytest.mark.parametrize(
    "weights",
    [
        [np.finfo(float).max] * 3,
        [np.nextafter(0.0, 1.0), np.finfo(float).max, 0.0],
        [np.nextafter(0.0, 1.0), 0.0, 0.0],
    ],
)
def test_exact_normalization_accepts_overflowing_sums_and_tiny_positive_weights(weights):
    a, b = np.array([[0.0, 0, 2]]), np.zeros((3, 3))
    result = certificate(a, b, [np.finfo(float).max], weights)
    assert_tight(result, F(4), F(4))


def test_tiny_weight_is_not_dropped_from_rational_witness():
    tiny = np.nextafter(0.0, 1.0)
    maximum = np.finfo(float).max
    # Weight normalization underflows in FP64, yet the exact witness is positive.
    a, b = np.zeros((1, 3)), np.array([[maximum, 0.0, 0.0], [0.0, 0, 0], [0.0, 0, 0]])
    result = certificate(a, b, [1.0], [tiny, maximum, 0.0])
    witness = F(float(tiny)) * F(float(maximum)) / (F(float(tiny)) + F(float(maximum)))
    assert result.lower_m == 0.0
    assert_tight(result, F(0), witness * witness)


@pytest.mark.parametrize(
    "distance", [np.nextafter(0.0, 1.0), 2.0**-600, 2.0**500, np.finfo(float).max]
)
def test_extreme_representable_distances_have_exact_endpoints(distance):
    result = certificate(np.array([[distance, 0.0, 0.0]]), np.zeros((3, 3)), [1.0], [1.0, 0.0, 0.0])
    assert result.lower_m == result.upper_m == distance
    assert_tight(result, F(float(distance)) ** 2, F(float(distance)) ** 2)


def test_distance_below_minimum_subnormal_has_conservative_adjacent_bounds():
    tiny = np.nextafter(0.0, 1.0)
    # Constant point at zero vs an edge witness at tiny/2; overlap makes Q zero.
    a, b = np.zeros((2, 3)), np.array([[0.0, 0, 0], [tiny, 0, 0]])
    result = certificate(a, b, [0.5, 0.5], [0.5, 0.5])
    assert result.lower_m == 0.0 and result.upper_m == tiny
    assert_tight(result, F(0), (F(float(tiny)) / 2) ** 2)


@pytest.mark.parametrize("y", [2.0**-27, 2.0**-26, np.nextafter(2.0**-26, np.inf), 1.0])
def test_square_root_across_ulp_boundary(y):
    point = np.array([[1.0, y, 0.0]])
    result = certificate(point, np.zeros((3, 3)), [1.0], [1.0, 0.0, 0.0])
    squared = 1 + F(float(y)) ** 2
    assert_tight(result, squared, squared)


def test_common_huge_translation_retains_actual_representable_gap():
    offset = 2.0**500
    gap = np.spacing(offset)
    a, b = np.array([[offset + gap, offset, offset]]), np.full((3, 3), offset)
    result = certificate(a, b, [1.0], [0.5, 0.25, 0.25])
    assert result.lower_m == result.upper_m == gap
    assert_encloses(result, true_distance_squared(a, b))


def test_unrepresentable_upper_bound_fails_then_normal_call_still_works():
    maximum = np.finfo(float).max
    with pytest.raises(ph.SolverFailure, match="represent|finite|range"):
        certificate(
            np.array([[maximum, 0.0, 0.0]]),
            np.array([[-maximum, 0.0, 0.0]] * 3),
            [1.0],
            [1.0, 0.0, 0.0],
        )
    assert (
        certificate(np.array([[0.5, 0.5, 3.0]]), triangle(), [1.0], [0.5, 0.25, 0.25]).lower_m
        == 3.0
    )


@pytest.mark.parametrize("which", range(4))
@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.bool_])
def test_rejects_non_fp64_storage_for_each_argument(which, dtype):
    args = [np.array([[0.5, 0.5, 3.0]]), triangle(), np.ones(1), np.array([0.5, 0.25, 0.25])]
    args[which] = args[which].astype(dtype)
    with pytest.raises(TypeError):
        certificate(*args)


@pytest.mark.parametrize(
    "which,value",
    [
        (0, np.zeros(3)),
        (0, np.zeros((1, 2))),
        (1, np.zeros((4, 3))),
        (1, np.zeros((0, 3))),
        (2, np.ones((1, 1))),
        (3, np.ones(2)),
        (3, np.array([-1.0, 1.0, 1.0])),
        (3, np.zeros(3)),
        (2, np.zeros(1)),
    ],
)
def test_rejects_shapes_combinations_and_infeasible_weights(which, value):
    args = [np.array([[0.5, 0.5, 3.0]]), triangle(), np.ones(1), np.array([0.5, 0.25, 0.25])]
    args[which] = value
    with pytest.raises(ValueError):
        certificate(*args)


@pytest.mark.parametrize("which", range(4))
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_each_argument(which, bad):
    args = [np.array([[0.5, 0.5, 3.0]]), triangle(), np.ones(1), np.array([0.5, 0.25, 0.25])]
    args[which].flat[0] = bad
    with pytest.raises(ValueError):
        certificate(*args)


def test_scalar_record_is_frozen_validated_and_has_no_certification_authority():
    assert hasattr(ph, "FeatureDistanceBounds")
    cls = ph.FeatureDistanceBounds
    value = cls(0.0, 1.0)
    assert value.method == "convex-kkt-distance-v1"
    with pytest.raises(FrozenInstanceError):
        value.lower_m = 0.5
    for lower, upper in [
        (-1.0, 1.0),
        (2.0, 1.0),
        (0.0, np.inf),
        (np.nan, 1.0),
        (True, 1.0),
        (0, 1.0),
    ]:
        with pytest.raises((TypeError, ValueError)):
            cls(lower, upper)
    with pytest.raises(ValueError):
        cls(0.0, 1.0, method="claimed-safe")
    # Shape-valid arbitrary construction is allowed, without proving geometry.
    assert cls(100.0, 100.0).lower_m == 100.0


def test_inputs_results_and_interleaved_calls_have_no_shared_mutable_state():
    args = [np.array([[0.5, 0.5, 3.0]]), triangle(), np.ones(1), np.array([0.5, 0.25, 0.25])]
    copies = [a.copy() for a in args]
    first = certificate(*args)
    certificate(args[0], args[1], args[2], [1.0, 0.0, 0.0])
    assert certificate(*args) == first
    for a, before in zip(args, copies):
        np.testing.assert_array_equal(a, before)
        a[:] = 42.0
    assert first.lower_m == first.upper_m == 3.0
    assert set(vars(first)) == {"lower_m", "upper_m", "method"}


def test_permutations_exchange_and_exactly_representable_transforms():
    a, b, wa, wb = np.array([[0.5, 0.5, 3.0]]), triangle(), [1.0], np.array([1.0, 0.0, 0.0])
    original = certificate(a, b, wa, wb)
    assert certificate(b, a, wb, wa) == original
    order = [2, 0, 1]
    assert certificate(a, b[order], wa, wb[order]) == original
    for scale, translation in [(1.0, 8.0), (4.0, 0.0), (0.125, -2.0)]:
        aa, bb = (
            a[:, [2, 0, 1]] * [-1, 1, -1] * scale + translation,
            b[:, [2, 0, 1]] * [-1, 1, -1] * scale + translation,
        )
        result = certificate(aa, bb, wa, wb)
        assert_encloses(result, true_distance_squared(aa, bb))
        assert result.lower_m == original.lower_m * scale
        assert result.upper_m == original.upper_m * scale


def test_general_rounded_rotation_certifies_actual_stored_input():
    theta = 0.37
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
    )
    a, b = np.array([[0.5, 0.5, 3.0]]) @ rotation, triangle() @ rotation
    result = certificate(a, b, [1.0], [0.5, 0.25, 0.25])
    assert_encloses(result, true_distance_squared(a, b))


@pytest.mark.parametrize(
    "first,second,wa,wb",
    [
        ([[0.5, 0.5, 3.0]], [[0.0, 0, 0], [2.0, 0, 0], [0.0, 2, 0]], [7.0], [1.0, 2.0, 7.0]),
        ([[2.0, -1.0, 1.0]], [[0.0, 0, 0], [2.0, 0, 0], [0.0, 2, 0]], [3.0], [7.0, 1.0, 2.0]),
        ([[-2.0, 0, 0], [2.0, 0, 0]], [[0.0, -2, 1], [0.0, 2, 1]], [2.0, 7.0], [3.0, 1.0]),
        ([[0.0, 0, 0], [1.0, 0, 0]], [[2.0, 1, 0], [3.0, 1, 0]], [1.0, 2.0], [5.0, 3.0]),
    ],
)
def test_arbitrary_nonoptimal_witness_exact_tangent_bound_and_geometric_oracle(
    first, second, wa, wb
):
    a, b = np.array(first), np.array(second)
    result = certificate(a, b, wa, wb)
    # Independently evaluate the tangent affine plane over Minkowski vertices.
    # This is separate from the active-face oracle for the true minimum.
    aa, bb = exact_rows(a), exact_rows(b)
    weights_a, weights_b = [F(w) for w in wa], [F(w) for w in wb]
    witness_a = [sum(w * v[k] for w, v in zip(weights_a, aa)) / sum(weights_a) for k in range(3)]
    witness_b = [sum(w * v[k] for w, v in zip(weights_b, bb)) / sum(weights_b) for k in range(3)]
    d = tuple(x - y for x, y in zip(witness_a, witness_b))
    s = dot(d, d)
    minimum_projection = min(dot(d, tuple(x - y for x, y in zip(v, w))) for v in aa for w in bb)
    q = max(F(0), 2 * minimum_projection - s)
    assert_tight(result, q, s)
    assert_encloses(result, true_distance_squared(a, b))


@pytest.mark.parametrize("n,m", [(1, 1), (1, 2), (2, 1), (2, 3), (3, 2), (3, 3)])
def test_rejects_every_other_nonempty_feature_size_pair(n, m):
    with pytest.raises(ValueError):
        certificate(np.zeros((n, 3)), np.zeros((m, 3)), np.ones(n), np.ones(m))


def test_negative_first_weight_is_not_clamped_and_failure_does_not_poison_next_call():
    a, b = np.zeros((2, 3)), np.ones((2, 3))
    with pytest.raises(ValueError):
        certificate(a, b, [-np.nextafter(0.0, 1.0), 1.0], [0.5, 0.5])
    assert certificate(a, b, [0.5, 0.5], [0.5, 0.5]) == certificate(a, b, [1.0, 1.0], [1.0, 1.0])


def test_finite_coordinates_with_unrepresentable_vector_norm_fail_explicitly():
    maximum = np.finfo(float).max
    with pytest.raises(ph.SolverFailure):
        certificate(np.array([[maximum, maximum, 0.0]]), np.zeros((3, 3)), [1.0], [1.0, 0.0, 0.0])


def test_source_installed_smoke_checks_public_distance_api():
    from pathlib import Path
    from runpy import run_path

    smoke = run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "smoke_installed.py"))
    assert callable(smoke.get("check_feature_distance_bounds")), "missing public distance smoke"
    smoke["check_feature_distance_bounds"]()
