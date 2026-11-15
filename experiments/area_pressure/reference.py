"""Independent NumPy V00/V01/V02 static pressure references.

No production patch, contact solver, time integration, Drake or Newton is used.
Normals give force on the first participant; the second receives its opposite
at the SAME quadrature point. All moments use the explicitly shared origin.
Pressure is Pa, area is m^2, coordinates are m, forces are N and moments are Nm.
"""

import json
from dataclasses import asdict, dataclass
from math import ceil, cos, isfinite, pi, sin, sqrt
from numbers import Integral, Real

import numpy as np

MAX_TRIANGLES = 8192
MAX_VERTICES = 8193
MAX_POINTS = 3 * MAX_TRIANGLES
_IDENTITY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


def _need(condition, message):
    if not condition:
        raise ValueError("area-pressure reference: " + message)


def _real(value, name):
    _need(isinstance(value, Real) and not isinstance(value, (bool, np.bool_)), name + " must be real")
    value = float(value)
    _need(isfinite(value), name + " must be finite")
    return value


def _rows(value, width, maximum, name, kind="fiu"):
    _need(isinstance(value, (tuple, list, np.ndarray)), name + " requires explicit bounded rows")
    _need(not isinstance(value, np.ndarray) or value.ndim in (1, 2), name + " has invalid rank")
    _need(len(value) <= maximum, name + " exceeds its row budget")
    array = np.asarray(value)
    if array.shape == (0,):
        array = array.reshape(0, width)
    _need(array.ndim == 2 and array.shape[1] == width and len(array) <= maximum
          and array.dtype.kind in kind and np.isfinite(array).all(), "invalid " + name)
    return np.array(array, dtype=np.int64 if kind == "iu" else np.float64, copy=True)


def _values(value, count, name):
    _need(isinstance(value, (tuple, list, np.ndarray)), name + " requires explicit values")
    _need(not isinstance(value, np.ndarray) or value.ndim in (1, 2), name + " has invalid rank")
    array = np.asarray(value)
    _need(array.shape == (count,) and count <= MAX_POINTS and array.dtype.kind in "fiu"
          and np.isfinite(array).all(), "invalid " + name)
    result = np.array(array, dtype=np.float64, copy=True)
    _need(np.isfinite(result).all(), name + " exceeds FP64")
    return result


def _tuples(rows):
    return tuple(tuple(float(item) for item in row) for row in rows)


def _mesh(vertices, triangles):
    vertices = _rows(vertices, 3, MAX_VERTICES, "vertices")
    if isinstance(triangles, (tuple, list)):
        _need(len(triangles) <= MAX_TRIANGLES, "triangle capacity exceeded")
        _need(all(isinstance(row, (tuple, list, np.ndarray)) and len(row) == 3
                  and all(isinstance(i, Integral) and not isinstance(i, (bool, np.bool_)) for i in row)
                  for row in triangles), "triangle indices must be integers")
    # Empty input has no dtype to infer; preserve the legitimate zero domain.
    faces = np.empty((0, 3), dtype=np.int64) if isinstance(triangles, (tuple, list)) and not triangles else _rows(triangles, 3, MAX_TRIANGLES, "triangles", "iu")
    _need(not len(faces) or (np.all(faces >= 0) and np.all(faces < len(vertices))
          and all(len(set(map(int, row))) == 3 for row in faces)), "invalid triangle vertex IDs")
    _need(np.isfinite(vertices).all(), "vertices exceed FP64")
    return vertices, faces


@dataclass(frozen=True)
class SurfaceQuadrature:
    vertices_world: tuple
    triangles: tuple
    points_world: tuple
    normals_world: tuple
    area_weights_m2: tuple
    triangle_indices: tuple


@dataclass(frozen=True)
class PressureIntegral:
    qualification: str
    quadrature: SurfaceQuadrature
    pressure_pa: tuple
    first_force_per_point_n: tuple
    second_force_per_point_n: tuple
    area_m2: float
    pressure_load_n: float
    origin_world: tuple
    first_force_n: tuple
    second_force_n: tuple
    first_moment_nm: tuple
    second_moment_nm: tuple
    pressure_centroid_world: tuple | None
    cop_world: tuple | None

    def to_dict(self):
        """Detached JSON-compatible values; reject NaN/Infinity rather than emit them."""
        result = asdict(self)
        json.dumps(result, allow_nan=False)
        return result


def triangle_quadrature(vertices_world, triangles):
    """Three positive degree-two-exact points per oriented, nondegenerate triangle.

    The barycentric points are permutations of (2/3,1/6,1/6), each weighted by
    one third of the ACTUAL triangle area. A linear pressure times its moment
    arm is quadratic, so a centroid-only rule would not suffice. This operation
    does not find a contact surface, prove mesh coverage or resolve self-overlap.
    """
    vertices, faces = _mesh(vertices_world, triangles)
    if not len(faces):
        return SurfaceQuadrature(_tuples(vertices), (), (), (), (), ())
    corners = vertices[faces]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        twice_area = np.linalg.norm(raw, axis=1)
        normals = raw / twice_area[:, None]
        barycentric = np.array(((2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3)))
        points = np.einsum("ij,tjk->tik", barycentric, corners).reshape(-1, 3)
        weights = np.repeat(twice_area / 6, 3)
    _need(np.isfinite(points).all() and np.isfinite(normals).all() and np.isfinite(weights).all()
          and np.all(weights > 0), "degenerate or nonfinite triangle quadrature")
    return SurfaceQuadrature(
        _tuples(vertices), tuple(tuple(int(i) for i in row) for row in faces),
        _tuples(points), _tuples(np.repeat(normals, 3, axis=0)), tuple(map(float, weights)),
        tuple(int(i) for i in np.repeat(np.arange(len(faces)), 3)),
    )


def integrate_pressure(quadrature, pressure_pa, *, origin_world=(0., 0., 0.)):
    """Integrate explicit nonnegative pressure; no area normalization or inferred load.

    pressure_centroid_world is the scalar p*dA centroid. cop_world is supplied
    only for a common oriented plane with positive load, where that centroid
    is a center of pressure. It is None for zero load and nonplanar surfaces.
    """
    _need(isinstance(quadrature, SurfaceQuadrature), "SurfaceQuadrature required")
    _, faces = _mesh(quadrature.vertices_world, quadrature.triangles)
    points = _rows(quadrature.points_world, 3, MAX_POINTS, "quadrature points")
    normals = _rows(quadrature.normals_world, 3, MAX_POINTS, "quadrature normals")
    count = len(points)
    areas = _values(quadrature.area_weights_m2, count, "area weights")
    pressure = _values(pressure_pa, count, "pressure")
    origin = _values(origin_world, 3, "moment origin")
    _need(normals.shape == points.shape and np.all(areas > 0) and np.all(pressure >= 0),
          "positive area and nonnegative pressure required")
    _need(len(quadrature.triangle_indices) == count
          and all(isinstance(i, Integral) and not isinstance(i, (bool, np.bool_)) and 0 <= i < len(faces)
                  for i in quadrature.triangle_indices), "invalid quadrature triangle mapping")
    _need(np.allclose(np.linalg.norm(normals, axis=1), 1., rtol=0, atol=1e-10), "unit normals required")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        loads = pressure * areas
        forces = loads[:, None] * normals
        force = forces.sum(axis=0)
        moment = np.cross(points - origin, forces).sum(axis=0)
        area, load = float(areas.sum()), float(loads.sum())
        centroid = (loads[:, None] * points).sum(axis=0) / load if load > 0 else None
    _need(all(np.isfinite(value).all() for value in (forces, force, moment))
          and isfinite(area) and isfinite(load)
          and (centroid is None or np.isfinite(centroid).all()), "nonfinite integrated pressure")
    center = None if centroid is None else tuple(map(float, centroid))
    planar = bool(count) and np.allclose(normals, normals[0], rtol=0, atol=1e-10)
    if planar:
        planar = bool(np.all(np.abs((points - points[0]) @ normals[0]) <= 1e-10))
    # Copy the validated quadrature into owned tuples, including caller-built records.
    owned = SurfaceQuadrature(
        _tuples(np.asarray(quadrature.vertices_world).reshape(-1, 3)),
        tuple(tuple(int(i) for i in row) for row in faces), _tuples(points), _tuples(normals),
        tuple(map(float, areas)), tuple(int(i) for i in quadrature.triangle_indices),
    )
    return PressureIntegral(
        "NUMPY_STATIC_REFERENCE", owned, tuple(map(float, pressure)), _tuples(forces), _tuples(-forces),
        area, load, tuple(map(float, origin)), tuple(map(float, force)), tuple(map(float, -force)),
        tuple(map(float, moment)), tuple(map(float, -moment)), center, center if planar else None,
    )


def _transform(quadrature, position, rotation):
    position = _values(position, 3, "position")
    rotation = _rows(rotation, 3, 3, "rotation")
    _need(rotation.shape == (3, 3) and np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-12)
          and abs(float(np.linalg.det(rotation)) - 1.) <= 1e-12, "proper rotation required")
    vertices = np.asarray(quadrature.vertices_world).reshape(-1, 3)
    points = np.asarray(quadrature.points_world).reshape(-1, 3)
    normals = np.asarray(quadrature.normals_world).reshape(-1, 3)
    with np.errstate(over="ignore", invalid="ignore"):
        vertices, points, normals = vertices @ rotation.T + position, points @ rotation.T + position, normals @ rotation.T
    _need(all(np.isfinite(value).all() for value in (vertices, points, normals)), "nonfinite transformed surface")
    return SurfaceQuadrature(_tuples(vertices), quadrature.triangles, _tuples(points), _tuples(normals),
                             quadrature.area_weights_m2, quadrature.triangle_indices)


def rectangle_case(width_m=.02, height_m=.02, *, subdivisions=(1, 1), pressure_pa=2000.,
                   pressure_gradient_pa_per_m=(0., 0.), position_world=(0., 0., 0.),
                   rotation_world=_IDENTITY, origin_world=(0., 0., 0.)):
    """Manufactured local p=p0+gx*x+gy*y on a centered rectangle; not contact detection."""
    width, height, base = _real(width_m, "width"), _real(height_m, "height"), _real(pressure_pa, "pressure")
    gradient = _values(pressure_gradient_pa_per_m, 2, "pressure gradient")
    _need(width > 0 and height > 0 and base >= 0, "positive dimensions and nonnegative pressure required")
    _need(isinstance(subdivisions, (tuple, list)) and len(subdivisions) == 2
          and all(isinstance(n, Integral) and not isinstance(n, (bool, np.bool_)) and 1 <= n <= 64
                  for n in subdivisions), "subdivisions must be two integers in 1..64")
    nx, ny = map(int, subdivisions)
    _need(2 * nx * ny <= MAX_TRIANGLES and (nx + 1) * (ny + 1) <= MAX_VERTICES, "rectangle mesh budget")
    _need(base - abs(gradient[0]) * width / 2 - abs(gradient[1]) * height / 2 >= 0,
          "manufactured pressure becomes negative on the rectangle")
    vertices = tuple((-width / 2 + width * i / nx, -height / 2 + height * j / ny, 0.)
                     for j in range(ny + 1) for i in range(nx + 1))
    faces = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b, d = a + 1, a + nx + 1
            faces.extend(((a, b, d + 1), (a, d + 1, d)))
    local = triangle_quadrature(vertices, faces)
    pressure = base + np.asarray(local.points_world)[:, :2] @ gradient
    return integrate_pressure(_transform(local, position_world, rotation_world), pressure, origin_world=origin_world)


def sphere_plane_case(radius_m=.02, indentation_m=.0002, modulus_pa=1e5, resolution_m=.0005, *,
                      plane_position_world=(0., 0., 0.), rotation_world=_IDENTITY, origin_world=(0., 0., 0.)):
    """Discretize a rigid-plane cut of the declared radial field p(r)=E*(1-r/R).

    The local plane is z=0, the sphere center is (0,0,R-indentation), and its
    pressure points are on an INSCRIBED polygonal disk. This is not Hertz or an
    external-engine reproduction. Concentric rings and sectors bound every
    triangle edge by resolution_m; that bound is not a pressure-error promise.
    Only 0<=indentation<=R is admitted. No closed-form total force/area is used.
    """
    radius, depth = _real(radius_m, "radius"), _real(indentation_m, "indentation")
    modulus, resolution = _real(modulus_pa, "modulus"), _real(resolution_m, "resolution")
    _need(radius > 0 and modulus > 0 and resolution > 0 and 0 <= depth <= radius, "invalid sphere/plane parameters")
    if depth == 0:
        local = triangle_quadrature((), ())
        return integrate_pressure(_transform(local, plane_position_world, rotation_world), (), origin_world=origin_world)
    squared_radius = depth * (2 * radius - depth)
    _need(isfinite(squared_radius) and squared_radius > 0, "disk radius is not representable")
    disk_radius = sqrt(squared_radius)
    radial_count, angular_count = sqrt(2) * disk_radius / resolution, 2 * pi * sqrt(2) * disk_radius / resolution
    _need(isfinite(radial_count) and isfinite(angular_count) and max(radial_count, angular_count) <= MAX_VERTICES,
          "disk resolution exceeds finite mesh budget")
    rings, sectors = max(1, ceil(radial_count)), max(8, ceil(angular_count))
    _need(1 + rings * sectors <= MAX_VERTICES and sectors * (2 * rings - 1) <= MAX_TRIANGLES, "disk mesh budget")
    vertices = [(0., 0., 0.)]
    for ring in range(1, rings + 1):
        r = disk_radius * ring / rings
        vertices.extend((r * cos(2 * pi * i / sectors), r * sin(2 * pi * i / sectors), 0.) for i in range(sectors))
    faces = [(0, 1 + i, 1 + (i + 1) % sectors) for i in range(sectors)]
    for ring in range(2, rings + 1):
        inner, outer = 1 + (ring - 2) * sectors, 1 + (ring - 1) * sectors
        for i in range(sectors):
            following = (i + 1) % sectors
            faces.extend(((inner + i, outer + i, outer + following),
                          (inner + i, outer + following, inner + following)))
    local = triangle_quadrature(vertices, faces)
    xy = np.asarray(local.points_world)[:, :2]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        radial_squared = np.einsum("ij,ij->i", xy, xy)
        distance = np.sqrt(radial_squared + (radius - depth) * (radius - depth))
        # Algebraically E*(1-distance/R), evaluated without subtracting nearly equal lengths.
        pressure = modulus * (np.maximum(0., squared_radius - radial_squared) / radius) / (radius + distance)
    _need(np.isfinite(radial_squared).all() and np.isfinite(distance).all()
          and np.isfinite(radius + distance).all() and np.isfinite(pressure).all(), "nonfinite radial pressure")
    return integrate_pressure(_transform(local, plane_position_world, rotation_world), pressure, origin_world=origin_world)
