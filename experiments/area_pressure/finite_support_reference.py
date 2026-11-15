"""V05 prescribed-pose finite affine-layer reference, without dynamics.

The top plane is z=0 with force-on-A normal +z. Two strictly convex CCW planar
footprints are intersected, then restricted to the closure of positive pressure
for gap=a*x+b*y+c and p=kappa*max(-gap,0). This is a declared elastic foundation
model, not general closed-body collision, side/back contact, or a reproduction
of a complete hydroelastic engine. Geometry tolerances do not calibrate loads.
"""

import json
from dataclasses import asdict, dataclass
from math import hypot, isfinite
from numbers import Integral, Real

import numpy as np

from .reference import (
    MAX_POINTS,
    MAX_TRIANGLES,
    MAX_VERTICES,
    SurfaceQuadrature,
    _real,
    _tuples,
    triangle_quadrature,
)

_LENGTH_TOLERANCE_M = 1e-12
_AREA_THRESHOLD_M2 = 1e-18
_MAX_POLYGON_VERTICES = 40


def _need(condition, message):
    if not condition:
        raise ValueError("finite affine-layer reference: " + message)


def _numeric(value, shape, name):
    _need(isinstance(value, (tuple, list, np.ndarray)), name + " requires explicit bounded values")
    if isinstance(value, np.ndarray):
        _need(value.shape == shape and value.dtype.kind in "fiu", "invalid " + name)
    else:
        _need(len(value) == shape[0], "invalid " + name)
        rows = value if len(shape) == 2 else (value,)
        for row in rows:
            _need(isinstance(row, (tuple, list, np.ndarray))
                  and (not isinstance(row, np.ndarray) or row.ndim == 1)
                  and len(row) == shape[-1], "invalid " + name)
            _need(all(isinstance(item, Real) and not isinstance(item, (bool, np.bool_))
                      for item in row), name + " must contain real non-boolean values")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.array(value, dtype=np.float64, copy=True).reshape(shape)
    _need(np.isfinite(array).all(), "nonfinite " + name)
    return array


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _difference(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _area(polygon):
    if len(polygon) < 3:
        return 0.
    anchor = polygon[0]
    # Translation-invariant fan avoids subtracting large world-coordinate products.
    return .5 * sum(_cross(_difference(polygon[i], anchor),
                             _difference(polygon[i + 1], anchor))
                     for i in range(1, len(polygon) - 1))


def _polygon(value, name):
    _need(isinstance(value, (tuple, list, np.ndarray))
          and (not isinstance(value, np.ndarray) or value.ndim == 2)
          and 3 <= len(value) <= 16, name + " must have 3..16 vertices")
    points = _numeric(value, (len(value), 2), name)
    _need(np.all(np.abs(points) <= 1.), name + " coordinates exceed 1 m")
    owned = _tuples(points)
    for i, point in enumerate(owned):
        for other in owned[i + 1:]:
            _need(hypot(*_difference(point, other)) > _LENGTH_TOLERANCE_M,
                  name + " has repeated or indistinguishable vertices")
    # A local left-turn check alone admits star traversals. Every other vertex
    # must lie strictly left of EVERY directed edge: the full convex support test.
    for i, start in enumerate(owned):
        end_index = (i + 1) % len(owned)
        edge = _difference(owned[end_index], start)
        length = hypot(*edge)
        for j, point in enumerate(owned):
            if j not in (i, end_index):
                _need(_cross(edge, _difference(point, start)) / length > 0,
                      name + " must be strictly convex, simple and CCW")
    _need(_area(owned) > _AREA_THRESHOLD_M2, name + " area is degenerate")
    return owned


def _append(points, point):
    if points and hypot(*_difference(points[-1], point)) <= _LENGTH_TOLERANCE_M:
        return
    _need(len(points) < _MAX_POLYGON_VERTICES, "clipped polygon exceeds vertex budget")
    points.append((float(point[0]), float(point[1])))


def _clean(polygon):
    points = []
    for point in polygon:
        _append(points, point)
    if len(points) > 1 and hypot(*_difference(points[0], points[-1])) <= _LENGTH_TOLERANCE_M:
        points.pop()
    # Clipping can create collinear boundary vertices. Remove only a vertex
    # within the fixed length tolerance of the segment joining its neighbors.
    changed = True
    while changed and len(points) >= 3:
        changed = False
        for i, point in enumerate(points):
            before, after = points[i - 1], points[(i + 1) % len(points)]
            segment, offset = _difference(after, before), _difference(point, before)
            length = hypot(*segment)
            if length <= _LENGTH_TOLERANCE_M:
                continue
            along = (segment[0] * offset[0] + segment[1] * offset[1]) / length
            if (abs(_cross(segment, offset)) / length <= _LENGTH_TOLERANCE_M
                    and -_LENGTH_TOLERANCE_M <= along <= length + _LENGTH_TOLERANCE_M):
                points.pop(i)
                changed = True
                break
    area = _area(points)
    if abs(area) <= _AREA_THRESHOLD_M2:
        return ()
    _need(area > 0, "clipping lost CCW orientation")
    return tuple(points)


def _clip(polygon, signed_distance):
    if not polygon:
        return ()
    result = []
    start = polygon[-1]
    start_distance = signed_distance(start)
    for end in polygon:
        end_distance = signed_distance(end)
        _need(isfinite(start_distance) and isfinite(end_distance), "nonfinite clip distance")
        # Do not expand a half-plane by the cleanup tolerance: that could create
        # a spurious finite contact area between physically disjoint polygons.
        start_inside, end_inside = start_distance >= 0., end_distance >= 0.
        if start_inside != end_inside:
            fraction = start_distance / (start_distance - end_distance)
            intersection = (start[0] + fraction * (end[0] - start[0]),
                            start[1] + fraction * (end[1] - start[1]))
            _append(result, intersection)
        if end_inside:
            _append(result, end)
        start, start_distance = end, end_distance
    return _clean(result)


def _quadrature(polygon, subdivisions):
    if not polygon:
        return triangle_quadrature((), ())
    fans = len(polygon) - 2
    n = subdivisions
    vertices_count = fans * (n + 1) * (n + 2) // 2
    triangles_count = fans * n * n
    _need(vertices_count <= MAX_VERTICES and triangles_count <= MAX_TRIANGLES
          and 3 * triangles_count <= MAX_POINTS, "subdivision exceeds shared quadrature budget")
    vertices, faces = [], []
    anchor = np.asarray(polygon[0])
    for fan in range(1, len(polygon) - 1):
        first = np.asarray(polygon[fan]) - anchor
        second = np.asarray(polygon[fan + 1]) - anchor
        indices = {}
        for i in range(n + 1):
            for j in range(n + 1 - i):
                point = anchor + (i / n) * first + (j / n) * second
                indices[i, j] = len(vertices)
                vertices.append((float(point[0]), float(point[1]), 0.))
        for i in range(n):
            for j in range(n - i):
                faces.append((indices[i, j], indices[i + 1, j], indices[i, j + 1]))
                if i + j < n - 1:
                    faces.append((indices[i + 1, j], indices[i + 1, j + 1], indices[i, j + 1]))
    return triangle_quadrature(vertices, faces)


@dataclass(frozen=True)
class FiniteSupportPatch:
    qualification: str
    footprint_xy: tuple
    support_xy: tuple
    gap_coefficients: tuple
    stiffness_pa_per_m: float
    layer_depth_m: float
    subdivisions: int
    overlap_polygon_xy: tuple
    active_polygon_xy: tuple
    overlap_area_m2: float
    active_area_m2: float
    quadrature: SurfaceQuadrature
    pressure_pa: tuple

    def to_dict(self):
        """Detached full geometry/pressure record with strict finite JSON output."""
        result = asdict(self)
        json.dumps(result, allow_nan=False)
        return result


def build_finite_patch(footprint_xy, support_xy, *, gap_coefficients,
                       stiffness_pa_per_m=1e7, layer_depth_m=.002, subdivisions=1):
    """Build a prescribed affine-layer patch on the finite top plane z=0.

    Depth is checked only over the footprint/support overlap. The active set is
    the closure of POSITIVE pressure, so an everywhere zero gap has zero active
    area, while its geometric overlap is retained. Overlap and active areas are
    never rescaled to support a mass. No normal/friction dynamics is evaluated.
    """
    footprint = _polygon(footprint_xy, "footprint")
    support = _polygon(support_xy, "support")
    gap = _numeric(gap_coefficients, (3,), "gap coefficients")
    slope = hypot(float(gap[0]), float(gap[1]))
    _need(slope <= .1, "gap slope exceeds the small-slope domain")
    stiffness = _real(stiffness_pa_per_m, "layer stiffness")
    depth = _real(layer_depth_m, "layer depth")
    _need(stiffness > 0 and depth > 0, "positive stiffness and depth required")
    _need(isinstance(subdivisions, Integral) and not isinstance(subdivisions, (bool, np.bool_))
          and 1 <= subdivisions <= 8, "subdivisions must be an integer in 1..8")
    n = int(subdivisions)
    overlap = footprint
    for i, start in enumerate(support):
        edge = _difference(support[(i + 1) % len(support)], start)
        length = hypot(*edge)
        overlap = _clip(overlap, lambda point: _cross(edge, _difference(point, start)) / length)
        if not overlap:
            break
    overlap_area = float(_area(overlap))
    active = ()
    if overlap:
        overlap_points = np.asarray(overlap)
        with np.errstate(over="ignore", invalid="ignore"):
            gaps = overlap_points @ gap[:2] + gap[2]
        _need(np.isfinite(gaps).all(), "nonfinite overlap gap")
        minimum, maximum = float(gaps.min()), float(gaps.max())
        # Affine extrema occur at the overlap vertices; the outside footprint
        # cannot penetrate a finite support that is absent there.
        _need(minimum >= -depth, "overlap penetrates beyond the finite layer depth")
        if minimum < 0:
            if maximum <= 0:
                active = overlap
            else:
                _need(slope > 0, "nonconstant gap required for a sign crossing")
                active = _clip(overlap, lambda point: -(gap[0] * point[0] + gap[1] * point[1] + gap[2]) / slope)
    quadrature = _quadrature(active, n)
    points = np.asarray(quadrature.points_world).reshape(-1, 3)
    with np.errstate(over="ignore", invalid="ignore"):
        local_gap = points[:, :2] @ gap[:2] + gap[2]
        pressure = stiffness * np.maximum(-local_gap, 0.)
    _need(np.isfinite(local_gap).all() and np.isfinite(pressure).all(), "nonfinite affine pressure")
    active_area = float(sum(quadrature.area_weights_m2))
    _need(isfinite(overlap_area) and isfinite(active_area), "nonfinite patch area")
    return FiniteSupportPatch(
        "NUMPY_FINITE_AFFINE_LAYER_REFERENCE", footprint, support, tuple(map(float, gap)),
        stiffness, depth, n, overlap, active, overlap_area, active_area, quadrature,
        tuple(map(float, pressure)),
    )
