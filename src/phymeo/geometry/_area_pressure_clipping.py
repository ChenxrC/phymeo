"""Small exact-from-FP64 clipping primitives for area-pressure intersection.

Closed half-space clipping follows the barycentric construction already used
by _tetrahedral_surface; there is no motion, CCD, epsilon fallback or solver.
Every rational arithmetic result has a fixed bit budget.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

INPUT_BITS = 256
ARITHMETIC_BITS = 4096
MAX_POLYGON_VERTICES = 7
_ZERO = Fraction(0)
_ONE = Fraction(1)


class AreaPressureIntersectionError(ValueError):
    """An attributable failure of the bounded static intersection contract."""

    def __init__(self, category, message):
        if category not in (
            "invalid_input", "invalid_mesh", "unsupported_gradient",
            "unsupported_shared_face_gradient", "resource_limit", "publication_failure",
        ):
            raise ValueError("unknown area-pressure intersection error category")
        self.category = category
        super().__init__(f"{category}: {message}")


def _bounded(value, bits=ARITHMETIC_BITS):
    if value.numerator.bit_length() > bits or value.denominator.bit_length() > bits:
        raise AreaPressureIntersectionError("resource_limit", "exact rational bit budget exceeded")
    return value


def exact_scalar(value):
    """Convert one already validated stored FP64 number, with the input budget."""
    return _bounded(Fraction(float(value)), INPUT_BITS)


def add(a, b):
    return _bounded(a + b)


def subtract(a, b):
    return _bounded(a - b)


def multiply(a, b):
    return _bounded(a * b)


def divide(a, b):
    if b == 0:
        raise AreaPressureIntersectionError("invalid_mesh", "zero geometric denominator")
    return _bounded(a / b)


def total(values):
    result = _ZERO
    for value in values:
        result = add(result, value)
    return result


def difference(a, b):
    return tuple(subtract(x, y) for x, y in zip(a, b, strict=True))


def dot(a, b):
    return total(multiply(x, y) for x, y in zip(a, b, strict=True))


def cross(a, b):
    return tuple(
        subtract(multiply(a[(i + 1) % 3], b[(i + 2) % 3]), multiply(a[(i + 2) % 3], b[(i + 1) % 3]))
        for i in range(3)
    )


def interpolate(a, b, t):
    return tuple(add(x, multiply(t, subtract(y, x))) for x, y in zip(a, b, strict=True))


@dataclass(frozen=True)
class Tetrahedron:
    points: tuple
    pressures: tuple
    duals: tuple
    determinant: Fraction
    gradient: tuple

    def weights(self, point):
        delta = difference(point, self.points[0])
        tail = tuple(divide(dot(dual, delta), self.determinant) for dual in self.duals)
        return (subtract(_ONE, total(tail)), *tail)

    def pressure(self, point):
        return dot(self.weights(point), self.pressures)


def tetrahedron(points, pressures):
    u, v, w = (difference(point, points[0]) for point in points[1:])
    duals = (cross(v, w), cross(w, u), cross(u, v))
    determinant = dot(u, duals[0])
    if determinant == 0:
        raise AreaPressureIntersectionError("invalid_mesh", "degenerate tetrahedron")
    change = tuple(subtract(p, pressures[0]) for p in pressures[1:])
    gradient = tuple(
        divide(total(multiply(change[i], duals[i][axis]) for i in range(3)), determinant)
        for axis in range(3)
    )
    return Tetrahedron(tuple(points), tuple(pressures), duals, determinant, gradient)


@dataclass(frozen=True)
class Triangle:
    points: tuple
    u: tuple
    v: tuple
    normal: tuple
    denominator: Fraction

    def height(self, point):
        return dot(self.normal, difference(point, self.points[0]))

    def weights(self, point):
        delta = difference(point, self.points[0])
        u = divide(dot(cross(delta, self.v), self.normal), self.denominator)
        v = divide(dot(cross(self.u, delta), self.normal), self.denominator)
        return (subtract(subtract(_ONE, u), v), u, v)


def triangle(points):
    u, v = (difference(point, points[0]) for point in points[1:])
    normal = cross(u, v)
    denominator = dot(normal, normal)
    if denominator == 0:
        raise AreaPressureIntersectionError("invalid_mesh", "degenerate rigid triangle")
    return Triangle(tuple(points), u, v, normal, denominator)


def _clip_segment(a, b, weights_a, weights_b):
    low, high = _ZERO, _ONE
    for start, end in zip(weights_a, weights_b, strict=True):
        slope = subtract(end, start)
        if slope == 0:
            if start < 0:
                return ()
        else:
            root = divide(subtract(_ZERO, start), slope)
            if slope > 0:
                low = max(low, root)
            else:
                high = min(high, root)
            if low > high:
                return ()
    first, second = interpolate(a, b, low), interpolate(a, b, high)
    return (first,) if first == second else (first, second)


def segment_in_tetrahedron(a, b, target):
    return _clip_segment(a, b, target.weights(a), target.weights(b))


def segment_in_triangle(a, b, target):
    ha, hb = target.height(a), target.height(b)
    if ha == 0 and hb == 0:
        return _clip_segment(a, b, target.weights(a), target.weights(b))
    if (ha > 0 and hb > 0) or (ha < 0 and hb < 0):
        return ()
    point = interpolate(a, b, divide(ha, subtract(ha, hb)))
    return (point,) if all(weight >= 0 for weight in target.weights(point)) else ()


def tetrahedron_intersection_vertices(a, b):
    result = set()
    for source, target in ((a, b), (b, a)):
        for i, j in combinations(range(4), 2):
            result.update(segment_in_tetrahedron(source.points[i], source.points[j], target))
    return tuple(sorted(result))


def triangle_intersection_vertices(a, b):
    result = set()
    for source, target in ((a, b), (b, a)):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            result.update(segment_in_triangle(source.points[i], source.points[j], target))
    return tuple(sorted(result))


def check_shared_simplex(first, first_node_ids, second_node_ids, intersection):
    """Every intersection vertex must lie in the shared topological simplex."""
    shared = set(first_node_ids).intersection(second_node_ids)
    for point in intersection:
        weights = first.weights(point)
        if not shared or any(weight != 0 for node, weight in zip(first_node_ids, weights, strict=True) if node not in shared):
            raise AreaPressureIntersectionError("invalid_mesh", "cell/face intersection exceeds its shared-node simplex")


def _unique_polygon(points):
    result = tuple(dict.fromkeys(points))
    if len(result) > MAX_POLYGON_VERTICES:
        raise AreaPressureIntersectionError("resource_limit", "clipped polygon exceeds seven vertices")
    return result


def _canonical_polygon(points):
    points = _unique_polygon(points)
    if len(points) < 3:
        return ()
    corners = tuple(
        point for i, point in enumerate(points)
        if any(cross(difference(point, points[i - 1]), difference(points[(i + 1) % len(points)], point)))
    )
    if len(corners) < 3:
        return ()
    first = min(range(len(corners)), key=lambda i: corners[i])
    return corners[first:] + corners[:first]


def clip_triangle(cell, face):
    """Return a winding-preserving positive-area polygon, or an empty tuple."""
    polygon = face.points
    for plane in range(4):
        output = []
        for i, b in enumerate(polygon):
            a = polygon[i - 1]
            da, db = cell.weights(a)[plane], cell.weights(b)[plane]
            if (da < 0 < db) or (db < 0 < da):
                output.append(interpolate(a, b, divide(da, subtract(da, db))))
            if db >= 0:
                output.append(b)
        polygon = _unique_polygon(output)
        if not polygon:
            return ()
    polygon = _canonical_polygon(polygon)
    if polygon:
        winding = dot(face.normal, cross(difference(polygon[1], polygon[0]), difference(polygon[2], polygon[0])))
        if winding <= 0:
            raise AreaPressureIntersectionError("publication_failure", "clipping did not preserve rigid winding")
    return polygon
