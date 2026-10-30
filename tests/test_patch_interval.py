"""Independent interval/curve identities and deterministic duplicate-work controls."""

from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest
from test_finite_patch_geometry import SQUARE, model

import phymeo as ph
from phymeo.core.scene import fingerprint
from phymeo.geometry import _patch_interval as interval


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ((2, 2), (-3, 5), (-6, 10)),
        ((-2, -2), (-3, 5), (-10, 6)),
        ((-2, 3), (-5, 7), (-15, 21)),
        ((-3, 5), (2, 2), (-6, 10)),
        ((-3, 5), (-2, -2), (-10, 6)),
        ((2, 2), (-3, -3), (-6, -6)),
        ((3, -2), (7, -5), (-15, 21)),
        ((2, 2), (5, -3), (-6, 10)),
        ((5, -3), (-2, -2), (-10, 6)),
        ((0, 0), (-(2**1100), 2**1100), (0, 0)),
    ],
)
def test_independent_interval_products(a, b, expected):
    value = interval.mul(tuple(map(F, a)), tuple(map(F, b)))
    assert value == tuple(map(F, expected))
    assert all(type(x) is F for x in value)


@pytest.mark.parametrize("kind", [int, float, np.float64, F])
def test_accepted_numeric_product_types(kind):
    a, b = (kind(2), kind(2)), (kind(-3), kind(5))
    actual = interval.mul(a, b)
    assert actual == (-6, 10)
    assert tuple(type(x) for x in actual) == (type(a[0] * b[0]), type(a[0] * b[1]))


def test_huge_subnormal_zero_and_helpers():
    tiny = F(1, 2**1074)
    huge = F(2**1100)
    assert interval.interval(tiny) == (tiny, tiny)
    assert interval.mul((huge, huge), (-tiny, tiny)) == (-F(2**26), F(2**26))
    assert interval.mul((F(0), F(0)), (-huge, huge)) == (F(0), F(0))
    assert interval.scale((F(-3), F(5)), F(-2)) == (F(-10), F(6))
    assert interval.add((F(-2), F(3)), (F(-5), F(7))) == (F(-7), F(10))
    assert interval.sub((F(-2), F(3)), (F(-5), F(7))) == (F(-9), F(8))
    assert interval.divide((F(2), F(6)), (F(2), F(2))) == (F(1), F(3))
    a = tuple(tuple((F(x), F(x)) for x in row) for row in ((1, 2), (3, 4)))
    b = tuple(tuple((F(x), F(x)) for x in row) for row in ((5, 6), (7, 8)))
    assert interval.matmul(a, b) == tuple(
        tuple((F(x), F(x)) for x in row) for row in ((19, 22), (43, 50))
    )
    assert interval.matvec(a, ((F(1), F(1)), (F(2), F(2)))) == ((F(5), F(5)), (F(11), F(11)))
    assert interval.idot(((F(1), F(1)), (F(-2), F(3))), ((F(3), F(3)), (F(4), F(5)))) == (
        F(-7),
        F(18),
    )


@pytest.mark.parametrize(
    "b,error",
    [
        ((-1, 1), ValueError),
        ((0, 1), ValueError),
        ((-1, 0), ValueError),
        ((1, 0), ZeroDivisionError),
    ],
)
def test_original_zero_divisor_errors(b, error):
    with pytest.raises(error):
        interval.divide((F(1), F(2)), tuple(map(F, b)))


def curve_fixture(kind="coordinate"):
    robot = model(kind == "rotor")
    vertices = SQUARE
    link = 0
    if kind == "noncoordinate":
        robot = replace(robot, joint_axis=np.array([[np.sqrt(0.5), np.sqrt(0.5), 0.0]]))
    if kind == "tree":
        robot = replace(
            robot,
            parents=np.array([-1, 0, 0]),
            joint_types=("prismatic",) * 3,
            joint_position=np.array([[1.0, 2.0, 3.0], [0.25, -0.5, 0.0], [-0.25, 0.0, 0.0]]),
            joint_orientation=np.array(
                [[0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
            ),
            joint_axis=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            mass=np.ones(3),
            com=np.zeros((3, 3)),
            inertia_diagonal=np.ones((3, 3)),
        )
        link = 1
    if kind == "rotor":
        vertices = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    return robot, ph.RobotFinitePatchCollider("math", link, vertices, radius_m=0.125)


@pytest.mark.parametrize("time", [F(0), F(1, 3), F(1), F(-1, 3), F(4, 3)])
def test_coordinate_curve_exact_interpolation_and_extrapolation(time):
    robot, collider = curve_fixture()
    curve = interval.PatchCurve(robot, [0.0], [2.0], collider)
    expected = tuple(
        tuple((F(x) + (2 * time if k == 1 else 0),) * 2 for k, x in enumerate(v))
        for v in collider.local_vertices
    )
    assert curve.at(time) == expected
    assert curve.projected_curvature((F(1), F(2), F(3))) == (F(0),) * 4
    assert curve.plane_lower((F(0),) * 3, (F(0),) * 3, F(0), F(1)) == 0


def test_shared_tree_fixed_frames_and_unused_sibling():
    robot, collider = curve_fixture("tree")
    q0 = np.array([0.5, 0.25, 0.0])
    q1 = np.array([0.5, 0.25, 7.0])
    before = fingerprint((robot, collider, q0, q1))
    curve = interval.PatchCurve(robot, q0, q1, collider)
    expected = tuple(
        ((F(1), F(1)), (F(5, 2) + F(v[0]),) * 2, (F(3) + F(v[2]),) * 2)
        for v in collider.local_vertices
    )
    assert curve.ends == (expected, expected)
    assert curve.at(F(2, 3)) == expected
    assert fingerprint((robot, collider, q0, q1)) == before
    frame = interval.exact_frame((0, 0, 1, 1))
    assert tuple(tuple(x[0] for x in row) for row in frame) == ((0, -1, 0), (1, 0, 0), (0, 0, 1))


def test_noncoordinate_norm_enclosure_and_constant_curve_extrapolation():
    robot, collider = curve_fixture("noncoordinate")
    curve = interval.PatchCurve(robot, [1.0], [1.0], collider)
    for row, local in zip(curve.ends[0], collider.local_vertices, strict=True):
        a, b = row[0][0] - F(local[0]), row[0][1] - F(local[0])
        assert 0 < a <= b and a * a <= F(1, 2) <= b * b
        assert row[1] == (a, b)
    expected = curve.ends[0]
    assert curve.at(F(1, 3)) == expected
    expanded = curve.at(F(-1, 3))
    assert expanded[0][0][1] - expanded[0][0][0] > expected[0][0][1] - expected[0][0][0]


def test_rotor_exact_origin_curve_curvature_and_plane():
    robot, collider = curve_fixture("rotor")
    curve = interval.PatchCurve(robot, [0.0], [0.5], collider)
    expected = tuple(tuple((F(v), F(v)) for v in row) for row in collider.local_vertices)
    assert curve.at(F(0)) == expected
    assert curve.projected_curvature((F(0), F(0), F(1))) == (F(0),) * 3
    assert curve.plane_lower((F(0), F(0), F(3, 2)), (F(0), F(0), F(3, 2)), F(0), F(1)) == F(3, 2)
    point = next(
        i for i, v in enumerate(collider.local_vertices) if np.array_equal(v, [0.0, 1.0, 0.0])
    )
    x = F(1, 6)
    slo = x - x**3 / 6
    shi = slo + x**5 / 120
    clo = 1 - x * x / 2
    chi = clo + x**4 / 24
    observed = curve.at(F(1, 3))[point]
    assert -shi <= observed[0][0] <= observed[0][1] <= -slo
    assert clo <= observed[1][0] <= observed[1][1] <= chi
    with pytest.raises(ValueError, match="remain"):
        curve.at(F(9))
    with pytest.raises(ValueError, match="remain"):
        interval.PatchCurve(robot, [4.25], [0.0], collider)


def test_endpoint_immutability_and_fresh_objects():
    robot, collider = curve_fixture()
    q0 = np.array([0.25])
    q1 = q0.copy()
    before = fingerprint((robot, collider, q0, q1))
    a = interval.PatchCurve(robot, q0, q1, collider)
    b = interval.PatchCurve(robot, q0, q1, collider)
    assert fingerprint((robot, collider, q0, q1)) == before
    assert a.ends == b.ends and a.ends[0] is not b.ends[0]
    with pytest.raises(TypeError):
        a.ends[0][0][0] = (F(1), F(1))
    q0[:] = 9
    assert a.q0 == (F(1, 4),)


@pytest.mark.parametrize(
    "left,right,expected",
    [((2, 2), (-3, -3), 1), ((2, 2), (-3, 5), 2), ((-3, 5), (2, 2), 2), ((-2, 3), (-5, 7), 4)],
)
def test_no_redundant_interval_multiplications(monkeypatch, left, right, expected):
    calls = []
    original = F.__mul__

    def counted(self, other):
        calls.append((self, other))
        return original(self, other)

    monkeypatch.setattr(F, "__mul__", counted)
    a, b = tuple(map(F, left)), tuple(map(F, right))
    interval.mul(a, b)
    assert len(calls) == expected


@pytest.mark.parametrize("kind", ["same", "same_converted", "different"])
def test_prismatic_endpoint_work_count(monkeypatch, kind):
    robot, collider = curve_fixture()
    calls = []
    original = interval.PatchCurve._prismatic

    def observed(self, q):
        calls.append(q)
        return original(self, q)

    monkeypatch.setattr(interval.PatchCurve, "_prismatic", observed)
    q0, q1 = (
        ([0.0], [0.0])
        if kind == "same"
        else ([F(1, 10)], [0.1])
        if kind == "same_converted"
        else ([0.0], [0.25])
    )
    curve = interval.PatchCurve(robot, q0, q1, collider)
    assert len(calls) == (2 if kind == "different" else 1)
    assert calls[0] == curve.q0


@pytest.mark.parametrize("a,b", [((1j, 1j), (2, 2)), ((F(1), complex(1, 0)), (F(2), F(2)))])
def test_non_orderable_products_preserve_type_error(a, b):
    with pytest.raises(TypeError):
        interval.mul(a, b)


def test_previously_accepted_short_interval_and_iterator():
    assert interval.mul((F(2),), (F(3), F(4))) == (F(6), F(8))
    assert interval.mul((x for x in (F(2), F(3))), (F(4), F(5))) == (F(8), F(15))
    assert interval.mul([F(2), F(2)], [F(3), F(3)]) == (F(6), F(6))


@pytest.mark.parametrize("a,b", [((), (F(1), F(2))), ((F(1), F(2)), ())])
def test_empty_legacy_product_retains_value_error(a, b):
    with pytest.raises(ValueError, match="empty"):
        interval.mul(a, b)


def test_fraction_subclass_retains_generic_operator_calls():
    class Legacy(F):
        calls = 0

        def __mul__(self, other):
            Legacy.calls += 1
            return super().__mul__(other)

    assert interval.mul((Legacy(2), Legacy(2)), (Legacy(-3), Legacy(5))) == (-6, 10)
    assert Legacy.calls == 4
