"""Eight frozen dual-turn inputs and independent finite-geometry/time references.

No producer, rotation enclosure, or production closest-feature implementation is
imported. The old test-only exact feature enumerator is reused as an independent
sample oracle; time references below use analytic geometry and rational Taylor
bounds, rather than observations made by the new query.
"""

from dataclasses import dataclass, replace
from fractions import Fraction as F
from math import cos, factorial, pi, sin

import numpy as np
from rotating_capsule_mesh_cases import (
    box,
    segment_triangle_reference,
    stored_pose,
    u_mesh,
)

import phymeo as ph

# Decimal rational bounds for mathematical pi, independent of producer constants.
PI_LOW = F("3.14159265358979323846264338327950288419716939937510")
PI_HIGH = PI_LOW + F(1, 10**50)
A = pi / 2
ZERO = (0.0, 0.0, 0.0)
X = (1.0, 0.0, 0.0)
Z = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class DualCase:
    name: str
    vertices: tuple
    triangles: tuple
    points: tuple
    radius: float
    mesh_angle: float
    round_angle: float
    mesh_axis: tuple = X
    round_axis: tuple = Z
    mesh_pivot_start: tuple = ZERO
    mesh_pivot_end: tuple = ZERO
    round_pivot_start: tuple = ZERO
    round_pivot_end: tuple = ZERO

    @property
    def kind(self):
        return "sphere" if len(self.points) == 1 else "capsule"


B01 = DualCase(
    "B01-noncommuting-capsule",
    *box((-2, -2, 7 / 8), (2, 2, 9 / 8)),
    (ZERO, (0.0, 1.0, 0.0)),
    1 / 8,
    -2 * A,
    A,
)
B02 = replace(B01, name="B02-orbiting-sphere", points=((0.0, 1.0, 0.0),))
B03 = replace(
    B01,
    name="B03-independent-moving-pivots",
    mesh_pivot_end=(1 / 8, -1 / 16, 3 / 32),
    round_pivot_end=(1 / 8, -1 / 16, 7 / 64),
)
B04 = DualCase(
    "B04-shaft-only",
    *box((-1 / 2, -2, 1), (1 / 2, 2, 5 / 4)),
    ((-3.0, 1.0, 0.0), (3.0, 1.0, 0.0)),
    1 / 8,
    1.0,
    3.0,
    mesh_axis=Z,
    round_axis=X,
)
B05 = replace(B01, name="B05-clear-height", vertices=box((-2, -2, 1), (2, 2, 5 / 4))[0])
B06 = DualCase(
    "B06-concave-notch",
    *u_mesh(),
    ((-1 / 4, 0.0, 0.0), (1 / 4, 0.0, 0.0)),
    1 / 8,
    0.9,
    1.3,
)
B07 = DualCase(
    "B07-initial-containment",
    *box((-2,) * 3, (2,) * 3),
    ((-1 / 2, 0.0, 0.0), (1 / 2, 0.0, 0.0)),
    1 / 8,
    0.9,
    1.3,
)
B08 = replace(
    B01,
    name="B08-isolated-grazing",
    mesh_angle=-A,
    vertices=box((-2, -2, 5 / 8), (2, 2, 7 / 8))[0],
)
CASES = (B01, B02, B03, B04, B05, B06, B07, B08)


def public_case(case, *, world=0, reverse=False, reorder=False, round_world=None):
    mk = ph.CollisionKey(world, int(reverse), 0)
    rk = ph.CollisionKey(world if round_world is None else round_world, int(not reverse), 0)
    ids = tuple(range(len(case.vertices)))
    order = tuple(reversed(ids)) if reorder else ids
    mesh = ph.CollisionShape(mk, "mesh", order, triangles=case.triangles)
    round_shape = ph.CollisionShape(
        rk, case.kind, (41,) if case.kind == "sphere" else (41, 93), case.radius
    )
    paths = {
        mk: ph.CollisionMotion(
            [case.vertices[i] for i in order],
            turn=ph.RigidTurn(
                case.mesh_pivot_start, case.mesh_pivot_end, case.mesh_axis, case.mesh_angle
            ),
        ),
        rk: ph.CollisionMotion(
            case.points,
            turn=ph.RigidTurn(
                case.round_pivot_start, case.round_pivot_end, case.round_axis, case.round_angle
            ),
        ),
    }
    return ph.CollisionModel((round_shape, mesh)), paths, mesh, round_shape


def transform(case, matrix, offset):
    matrix = np.asarray(matrix, dtype=float)
    offset = np.asarray(offset, dtype=float)
    sign = 1 if np.linalg.det(matrix) > 0 else -1

    def point(p):
        return tuple(matrix @ p + offset)

    return replace(
        case,
        vertices=tuple(point(p) for p in case.vertices),
        triangles=case.triangles if sign > 0 else tuple((a, c, b) for a, b, c in case.triangles),
        points=tuple(point(p) for p in case.points),
        mesh_axis=tuple(sign * matrix @ case.mesh_axis),
        round_axis=tuple(sign * matrix @ case.round_axis),
        mesh_pivot_start=point(case.mesh_pivot_start),
        mesh_pivot_end=point(case.mesh_pivot_end),
        round_pivot_start=point(case.round_pivot_start),
        round_pivot_end=point(case.round_pivot_end),
    )


def _series(x, *, cosine=False):
    """Rational Taylor enclosure; the next derivative has magnitude at most one."""
    x = F(x)
    assert abs(x) <= 4
    odd = int(not cosine)
    term = x if odd else F(1)
    total = term
    for k in range(1, 28):
        degree = 2 * k + odd
        term *= -x * x / (degree * (degree - 1))
        total += term
    degree = 56 + odd
    error = abs(x) ** degree / factorial(degree)
    return total - error, total + error


def _product(a, b):
    corners = tuple(x * y for x in a for y in b)
    return min(corners), max(corners)


def height_interval(t, angle=A, drift=F(0)):
    """B01/B02/B03 true tip height in the unrotated material box."""
    t, angle = F(t), F(angle)
    sine = _series(2 * angle * t)
    cosine = _series(angle * t, cosine=True)
    lo, hi = _product(sine, cosine)
    extra = _product((F(drift) * t, F(drift) * t), _series(2 * angle * t, cosine=True))
    return lo + extra[0], hi + extra[1]


def _increasing_root(function, target, low, high):
    target, low, high = F(target), F(low), F(high)
    assert function(low)[1] < target < function(high)[0]
    for _ in range(64):
        if high - low <= F(1, 1 << 42):
            return low, high
        mid = (low + high) / 2
        a, b = function(mid)
        if b < target:
            low = mid
        elif a > target:
            high = mid
        else:
            raise AssertionError("independent time reference could not decide its midpoint")
    raise AssertionError("independent time reference exceeded its fixed budget")


def root_interval(case, round_angle=None):
    angle = F(case.round_angle if round_angle is None else round_angle)
    if case.name in (B01.name, B02.name):
        # f=2s(1-s^2), strictly increasing up to s=1/sqrt(3); first f=3/4 is s=1/2.
        # The stored A is not exact mathematical pi/2, so t is not exactly 1/3.
        return PI_LOW / (6 * angle), PI_HIGH / (6 * angle)
    if case.name == B03.name:
        # On [0,1/3], f'(t)=2A*cos(a)*(1-3sin(a)^2) plus the bounded drift derivative.
        # A<pi/2, cos(a)>0.85 and sin(a)^2<1/4 give f'>.425A-(1+2A/3)/64>0.
        assert 2 * angle < PI_LOW
        assert F(17, 40) * angle - (1 + 2 * angle / 3) / 64 > 0
        return _increasing_root(
            lambda t: height_interval(t, angle, F(1, 64)), F(3, 4), F(5, 16), F(1, 3)
        )
    if case.name == B04.name:
        # sin(3t) is increasing over this predeclared interval; the finite face contains its projection.
        return _increasing_root(lambda t: _series(angle * t), F(7, 8), F(1, 4), F(1, 2))
    if case.name == B07.name:
        return F(0), F(0)
    if case.name == B08.name:
        return PI_LOW / (4 * angle), PI_HIGH / (4 * angle)
    raise ValueError("this input has a full-path clearance proof, not an entry root")


def mesh_distance_reference(mesh, round_shape, paths, time):
    """All original finite triangles, exact QPs on independently evaluated stored poses."""
    vertices = dict(zip(mesh.vertex_ids, stored_pose(paths[mesh.key], time), strict=True))
    points = stored_pose(paths[round_shape.key], time)
    ends = points if len(points) == 2 else (points[0], points[0])
    return min(
        (segment_triangle_reference(ends, [vertices[i] for i in face]), face)
        for face in mesh.triangles
    )


def shaft_span_reference(time):
    """Clip the actual finite axis to the rotated bottom rectangle, independent of triangulation."""
    y, phi = cos(B04.round_angle * time), B04.mesh_angle * time
    low, high = -3.0, 3.0
    for coefficient, constant, half in (
        (cos(phi), y * sin(phi), 1 / 2),
        (-sin(phi), y * cos(phi), 2),
    ):
        if coefficient:
            bounds = sorted(((-half - constant) / coefficient, (half - constant) / coefficient))
            low, high = max(low, bounds[0]), min(high, bounds[1])
        else:
            assert abs(constant) <= half
    assert low < high
    return (low + 3) / 6, (high + 3) / 6


def static_paths(paths, time):
    return {key: ph.CollisionMotion(stored_pose(path, time)) for key, path in paths.items()}
