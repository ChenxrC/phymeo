"""Independent analytic affine paths; samples are never a safety oracle."""

import importlib
import runpy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import phymeo as ph


def sweep(a, b, c, d, **kwargs):
    assert callable(getattr(ph, "linear_feature_sweep", None)), "missing public sweep API"
    return ph.linear_feature_sweep(*(np.array(v, dtype=np.float64) for v in (a, b, c, d)), **kwargs)


PLATE = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]


def test_public_sweep_api():
    assert callable(getattr(ph, "linear_feature_sweep", None)), "missing public sweep API"
    assert getattr(ph, "FeatureSweepResult", None) is not None


def test_probe_plate_crossing_safe_and_equality():
    for start, end, status in ((1, -1, "violation"), (1, 2, "certified")):
        result = sweep([[0.5, 0.5, start]], [[0.5, 0.5, end]], PLATE, PLATE, clearance_m=0.25)
        assert result.status == status
        if status == "violation":
            assert abs(Fraction(start) + Fraction(result.witness_time) * (end - start)) < Fraction(
                0.25
            )
    result = sweep([[0.25, 0.25, 0.25]], [[0.75, 0.25, 0.25]], PLATE, PLATE, clearance_m=0.25)
    assert result.status == "certified"


def partition(result):
    intervals = sorted(result.certified_intervals + result.unresolved_intervals)
    assert intervals[0][0] == 0 and intervals[-1][1] == 1
    assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
    assert all(type(t) is float for pair in intervals for t in pair)
    if result.status == "violation":
        assert any(
            lower <= result.witness_time <= upper for lower, upper in result.unresolved_intervals
        )


def test_quarter_crossing_requires_subdivision_and_complete_pending_partition():
    plate = [[-1, -2, -2], [-1, 2, -2], [-1, 0, 2]]
    args = ([[-2, 0, 0]], [[2, 0, 0]], plate, plate)
    root = sweep(*args, clearance_m=0.125, max_intervals=1)
    assert root.status == "unresolved" and root.checked_intervals == 1
    result = sweep(*args, clearance_m=0.125)
    assert result.status == "violation"
    assert abs(4 * Fraction(result.witness_time) - 1) < Fraction(0.125)
    partition(result)


def test_independent_rods_crossing_and_safe_offset():
    first = [[-1, 0, 0], [1, 0, 0]]
    assert (
        sweep(
            first, first, [[0, -1, 1], [0, 1, 1]], [[0, -1, -1], [0, 1, -1]], clearance_m=0.25
        ).status
        == "violation"
    )
    # y=.5 is a separating coordinate throughout this in-plane translation.
    assert (
        sweep(
            first, first, [[-1, 0.5, 0], [1, 0.5, 0]], [[0, 0.5, 0], [2, 0.5, 0]], clearance_m=0.5
        ).status
        == "certified"
    )


def test_turning_rod_safe_but_requires_subdivision():
    a = [[0, 0, -0.25], [0, 0, 0.25]]
    args = (a, a, [[1, -2, 0], [1, 2, 0]], [[2, 1, 0], [-2, 1, 0]])
    assert sweep(*args, clearance_m=0.5, max_intervals=1).status == "unresolved"
    result = sweep(*args, clearance_m=0.5)
    assert result.status == "certified" and result.checked_intervals > 1
    partition(result)
    # Independently: center c=(1-t,t); direction 2*(-t,1-t) is perpendicular.
    # The z-axis segment therefore has distance squared .5+2*(t-.5)^2.
    for lower, upper in result.certified_intervals:
        t = min(max(Fraction(1, 2), Fraction(lower)), Fraction(upper))
        assert (1 - t) ** 2 + t**2 >= Fraction(1, 2) > Fraction(0.5) ** 2


@pytest.mark.parametrize("budget", [1, 2, 3, 4, 7])
def test_budget_and_depth_preserve_full_partition(budget):
    # Degenerate hulls with an interior-only intersection, missed by root samples.
    a = [[-2, 0, 0]]
    b = [[-1, 0, 0]] * 3
    result = sweep(a, [[2, 0, 0]], b, b, clearance_m=0.01, max_intervals=budget, max_depth=52)
    assert result.checked_intervals <= budget
    partition(result)


def test_subnormal_clearance_is_not_squared_in_float():
    tiny = float(np.nextafter(0.0, 1.0))
    point, triangle = [[0, 0, 0]], [[0, 0, 0]] * 3
    assert sweep(point, point, triangle, triangle, clearance_m=tiny).status == "violation"
    displaced = [[tiny, 0, 0]]
    assert sweep(displaced, displaced, triangle, triangle, clearance_m=tiny).status == "certified"


@pytest.mark.parametrize(
    "plane_x,r,status", [(1.0, 2.0**-54, "certified"), (1.0 + 2.0**-52, 3 * 2.0**-54, "violation")]
)
def test_rational_midpoint_not_rounded_geometry_witness(plane_x, r, status):
    plate = [[plane_x, -0.25, -1], [plane_x, 0.25, -1], [plane_x, 0, 1]]
    start, end = [[1.0, -1, 0]], [[1.0 + 2.0**-52, 1, 0]]
    exact_x = (Fraction(start[0][0]) + Fraction(end[0][0])) / 2
    assert exact_x != Fraction(float(exact_x))
    assert abs(exact_x - Fraction(plane_x)) == Fraction(2.0**-53)
    # A safe affine path here can be conservative/unresolved, never false violation.
    result = sweep(start, end, plate, plate, clearance_m=r)
    if status == "certified":
        assert result.status != "violation"
    else:
        assert result.status == status
        t = Fraction(result.witness_time)
        x = Fraction(1) + t * Fraction(2.0**-52)
        assert (x - Fraction(plane_x)) ** 2 < Fraction(r) ** 2


@pytest.mark.parametrize("value", [True, 1, np.float32(0.25), np.float64(0.25), Fraction(1, 4)])
def test_clearance_rejects_other_precision_and_types(value):
    with pytest.raises(TypeError):
        sweep([[0, 0, 1]], [[0, 0, 1]], PLATE, PLATE, clearance_m=value)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_clearance_invalid_values(value):
    with pytest.raises(ValueError):
        sweep([[0, 0, 1]], [[0, 0, 1]], PLATE, PLATE, clearance_m=value)


@pytest.mark.parametrize(
    "name,value,error",
    [
        ("max_depth", True, TypeError),
        ("max_depth", 0.5, TypeError),
        ("max_depth", -1, ValueError),
        ("max_depth", 53, ValueError),
        ("max_intervals", 0, ValueError),
        ("max_intervals", False, TypeError),
        ("max_intervals", 1.0, TypeError),
    ],
)
def test_budget_validation(name, value, error):
    with pytest.raises(error):
        sweep([[0, 0, 1]], [[0, 0, 1]], PLATE, PLATE, clearance_m=0.25, **{name: value})


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize(
    "bad,error",
    [
        (np.zeros((1, 3), np.float32), TypeError),
        (np.zeros((1, 2)), ValueError),
        (np.zeros((1, 1, 3)), ValueError),
        (np.full((1, 3), np.nan), ValueError),
        (np.full((1, 3), np.inf), ValueError),
    ],
)
def test_geometry_input_validation(index, bad, error):
    args = [
        np.array([[0.0, 0, 1]]),
        np.array([[0.0, 0, 1]]),
        np.array(PLATE, float),
        np.array(PLATE, float),
    ]
    args[index] = bad
    with pytest.raises(error):
        ph.linear_feature_sweep(*args, clearance_m=0.25)


def test_unsupported_pair_and_shape_mismatch():
    with pytest.raises(ValueError):
        sweep([[0, 0, 1]], [[0, 0, 1]] * 2, PLATE, PLATE, clearance_m=0.25)
    with pytest.raises(ValueError):
        sweep(PLATE, PLATE, PLATE, PLATE, clearance_m=0.25)


@pytest.mark.parametrize("start,end,expected", [(0.0, 1.0, 0.0), (1.0, 0.0, 1.0)])
def test_initial_and_final_violation_are_witnesses(start, end, expected):
    result = sweep([[0.5, 0.5, start]], [[0.5, 0.5, end]], PLATE, PLATE, clearance_m=0.1)
    assert result.status == "violation" and result.witness_time == expected
    partition(result)


def test_depth_zero_and_numpy_integral_budget():
    a = [[0, 0, -0.25], [0, 0, 0.25]]
    result = sweep(
        a,
        a,
        [[1, -2, 0], [1, 2, 0]],
        [[2, 1, 0], [-2, 1, 0]],
        clearance_m=0.5,
        max_depth=np.int64(0),
        max_intervals=np.int64(128),
    )
    assert result.status == "unresolved" and type(result.checked_intervals) is int
    assert result.checked_intervals == 1


def test_owned_frozen_report_and_constructor_validation():
    a, b = np.array([[0.5, 0.5, 1.0]]), np.array(PLATE, float)
    before_a, before_b = a.copy(), b.copy()
    result = ph.linear_feature_sweep(a, a, b, b, clearance_m=0.25)
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)
    a[:] = b[:] = 0
    assert result.certified_intervals == ((0.0, 1.0),)
    with pytest.raises(FrozenInstanceError):
        result.status = "violation"
    for kwargs, error in [
        ({"status": np.array("certified")}, TypeError),
        ({"method": np.array(result.method)}, TypeError),
        ({"checked_intervals": True}, TypeError),
        ({"certified_intervals": [(0.0, 1.0)]}, TypeError),
        ({"certified_intervals": ((0.0, 0.5),)}, ValueError),
        ({"certified_intervals": ((0.0, 0.75), (0.5, 1.0))}, ValueError),
        ({"status": "unresolved"}, ValueError),
        ({"witness_time": 0.5}, ValueError),
        (
            {
                "status": "violation",
                "certified_intervals": (),
                "unresolved_intervals": ((0.0, 1.0),),
                "witness_time": None,
            },
            TypeError,
        ),
    ]:
        with pytest.raises(error):
            replace(result, **kwargs)


@pytest.mark.parametrize("status", ["certified", "violation"])
def test_swap_permutation_exact_transform_and_common_motion(status):
    start = np.array([[0.5, 0.5, 1.0]])
    end = np.array([[0.5, 0.5, 2.0 if status == "certified" else -1.0]])
    plate = np.array(PLATE, float)
    for order in ([0, 1, 2], [2, 0, 1]):
        for swap in (False, True):
            # Signed coordinate permutation + binary-exact translation.
            def transform(v):
                return v[:, [2, 0, 1]] * [-1, 1, -1] + [8, -4, 2]

            args = [
                transform(start),
                transform(end),
                transform(plate[order]),
                transform(plate[order]),
            ]
            if swap:
                args = args[2:] + args[:2]
            assert ph.linear_feature_sweep(*args, clearance_m=0.25).status == status
    delta = np.array([2.0**45, -(2.0**45), 2.0**45])
    assert (
        ph.linear_feature_sweep(start, start + delta, plate, plate + delta, clearance_m=0.25).status
        == "certified"
    )


def test_degenerate_and_near_parallel_candidates_are_conservative():
    a = [[0, 0, 0], [1, 0, 0]]
    b = [[0, 0.5, 0], [1, 0.5, 2.0**-50]]
    assert sweep(a, a, b, b, clearance_m=0.5).status == "certified"
    a = [[0, 0, 0]] * 2
    b = [[1, 1, 1]] * 2
    assert sweep(a, a, b, b, clearance_m=0.5).status == "certified"
    assert sweep(a, a, a, a, clearance_m=0.5).status == "violation"


def test_declared_candidate_failure_axes_and_unknown_errors_propagate(monkeypatch):
    module = importlib.import_module("phymeo.geometry.linear_sweep")

    def failed(*args):
        raise ph.SolverFailure("candidate representation failure")

    monkeypatch.setattr(module, "closest_point_triangle", failed)
    point = [[0.5, 0.5, 0.25]]
    assert sweep(point, point, PLATE, PLATE, clearance_m=0.25).status == "certified"

    def unknown(*args):
        raise RuntimeError("unexpected resource failure")

    monkeypatch.setattr(module, "closest_point_triangle", unknown)
    with pytest.raises(RuntimeError, match="unexpected resource"):
        sweep(point, point, PLATE, PLATE, clearance_m=0.25)
    monkeypatch.undo()
    assert sweep(point, point, PLATE, PLATE, clearance_m=0.25).status == "certified"


def test_source_smoke_helper_runs_without_claiming_installed_wheel():
    script = Path(__file__).resolve().parents[1] / "scripts" / "smoke_installed.py"
    namespace = runpy.run_path(str(script))
    assert callable(namespace.get("check_linear_feature_sweep")), "missing sweep smoke helper"
    namespace["check_linear_feature_sweep"]()


def test_depth_52_reaches_exact_dyadic_leaves_without_missing_partition():
    # Contact at t=1/3 is not dyadic, and the least positive clearance makes
    # no sampled dyadic a strict witness, even at maximum depth.
    tiny = float(np.nextafter(0.0, 1.0))
    b = [[0, 0, 0]] * 3
    result = sweep(
        [[-1, 0, 0]], [[2, 0, 0]], b, b, clearance_m=tiny, max_intervals=128, max_depth=52
    )
    assert result.status == "unresolved" and result.witness_time is None
    assert result.checked_intervals == 105
    assert len(result.unresolved_intervals) == 1
    lower, upper = result.unresolved_intervals[0]
    assert Fraction(upper) - Fraction(lower) == Fraction(1, 2**52)
    assert Fraction(lower) < Fraction(1, 3) < Fraction(upper)
    for lower, upper in result.certified_intervals:
        x, y = 3 * Fraction(lower) - 1, 3 * Fraction(upper) - 1
        assert x * y > 0 and min(abs(x), abs(y)) >= Fraction(tiny)
    partition(result)


def test_closest_distance_is_not_evidence_and_weights_are_exactly_normalized(monkeypatch):
    module = importlib.import_module("phymeo.geometry.linear_sweep")
    maximum = np.finfo(float).max

    def fake_distance(*args):
        return SimpleNamespace(
            distance=0.0,
            first_weights=np.array([maximum]),
            second_weights=np.array([maximum, maximum, maximum]),
        )

    monkeypatch.setattr(module, "closest_point_triangle", fake_distance)
    assert (
        sweep([[0.5, 0.5, 1]], [[0.5, 0.5, 2]], PLATE, PLATE, clearance_m=0.25).status
        == "certified"
    )

    def fake_large_distance(*args):
        return SimpleNamespace(
            distance=1.0e300,
            first_weights=np.array([2.0]),
            second_weights=np.array([2.0, 1.0, 1.0]),
        )

    monkeypatch.setattr(module, "closest_point_triangle", fake_large_distance)
    result = sweep([[0.5, 0.5, 1]], [[0.5, 0.5, -1]], PLATE, PLATE, clearance_m=0.25)
    assert result.status == "violation" and result.witness_time == 0.5


def test_unknown_value_error_is_not_declared_candidate_failure(monkeypatch):
    module = importlib.import_module("phymeo.geometry.linear_sweep")

    def unknown(*args):
        raise ValueError("unexpected candidate bug")

    monkeypatch.setattr(module, "closest_point_triangle", unknown)
    with pytest.raises(ValueError, match="unexpected candidate bug"):
        sweep([[0.5, 0.5, 1]], [[0.5, 0.5, 2]], PLATE, PLATE, clearance_m=0.25)


def test_huge_common_translation_and_transient_degeneracy():
    # FP64 interpolation candidates may collapse here; the synchronous exact
    # relative x offset is constant, and coordinate-axis proof remains valid.
    a = np.array([[0.0, 0, 0]])
    b = np.array([[1.0, 0, 0], [1.0, 2, 0], [1.0, 0, 2]])
    shift = np.array([0.0, 1.0e308, 0.0])
    assert (
        ph.linear_feature_sweep(a, a + shift, b, b + shift, clearance_m=1.0).status == "certified"
    )
    first = [[-1, 0, 0], [1, 0, 0]]
    # second rod collapses at .5 yet retains z=.5 throughout.
    second = [[0, -1, 0.5], [0, 1, 0.5]]
    assert sweep(first, first, second, second[::-1], clearance_m=0.5).status == "certified"
