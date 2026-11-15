"""Independent V03 static equal-pressure interface for two soft radial fields.

This is an axis-aligned shallow-overlap reference, not a contact finder, dynamic
solver, Hertz model, or Drake/Newton reproduction. Sphere A is at the origin;
sphere B is on +z. Each field is p=E*(1-r/R). Equal pressure defines a generally
CURVED interface: projected quadrature points are lifted by bounded bisection,
and their actual surface areas and normals use the implicit-surface Jacobian.
The rendered vertex triangles are only a chordal visualization of that surface.
"""

import json
from dataclasses import asdict, dataclass, replace
from math import ceil, cos, isfinite, pi, sin, sqrt
from numbers import Real

import numpy as np

from .reference import (
    MAX_TRIANGLES,
    MAX_VERTICES,
    PressureIntegral,
    SurfaceQuadrature,
    integrate_pressure,
    triangle_quadrature,
)

_PRESSURE_TOLERANCE_PA = 1e-7
_BISECTION_STEPS = 64


def _need(condition, message):
    if not condition:
        raise ValueError("two-sphere static reference: " + message)


def _real(value, name):
    _need(
        isinstance(value, Real) and not isinstance(value, (bool, np.bool_)), name + " must be real"
    )
    result = float(value)
    _need(isfinite(result), name + " must be finite")
    return result


def _tuples(rows):
    return tuple(tuple(float(value) for value in row) for row in rows)


def _fields(rho, z, ra, rb, ea, eb, distance):
    radius_a = np.hypot(rho, z)
    radius_b = np.hypot(rho, distance - z)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        pa = ea * ((ra - radius_a) / ra)
        pb = eb * ((rb - radius_b) / rb)
    _need(
        all(np.isfinite(value).all() for value in (radius_a, radius_b, pa, pb))
        and np.all(radius_a > 0)
        and np.all(radius_b > 0),
        "nonfinite radial field",
    )
    return pa, pb, radius_a, radius_b


def _roots(rho, ra, rb, ea, eb, distance):
    """Unique root on the overlapping caps, with a fixed finite iteration budget."""
    with np.errstate(over="ignore", invalid="ignore"):
        low = distance - np.sqrt(rb * rb - rho * rho)
        high = np.sqrt(ra * ra - rho * rho)
    _need(
        np.isfinite(low).all() and np.isfinite(high).all() and np.all(low <= high),
        "unrepresentable overlapping caps",
    )
    low_a, low_b, _, _ = _fields(rho, low, ra, rb, ea, eb, distance)
    high_a, high_b, _, _ = _fields(rho, high, ra, rb, ea, eb, distance)
    _need(
        np.all(low_a - low_b >= -_PRESSURE_TOLERANCE_PA)
        and np.all(high_a - high_b <= _PRESSURE_TOLERANCE_PA),
        "root is not bracketed",
    )
    # g=pA-pB decreases with z throughout this admitted shallow lens.
    for _ in range(_BISECTION_STEPS):
        middle = low + 0.5 * (high - low)
        pa, pb, _, _ = _fields(rho, middle, ra, rb, ea, eb, distance)
        low = np.where(pa > pb, middle, low)
        high = np.where(pa <= pb, middle, high)
    z = low + 0.5 * (high - low)
    pa, pb, _, _ = _fields(rho, z, ra, rb, ea, eb, distance)
    _need(
        np.all(np.abs(pa - pb) <= _PRESSURE_TOLERANCE_PA), "equal-pressure root tolerance not met"
    )
    return z


@dataclass(frozen=True)
class TwoSphereReference:
    radius_a_m: float
    radius_b_m: float
    modulus_a_pa: float
    modulus_b_pa: float
    overlap_m: float
    resolution_m: float
    centers_world: tuple
    circle_radius_m: float
    circle_height_m: float
    axis_height_m: float | None
    axis_pressure_pa: float | None
    area_projected_m2: float
    projected_area_weights_m2: tuple
    radial_slopes: tuple
    pressure_a_pa: tuple
    pressure_b_pa: tuple
    pressure_residual_pa: tuple
    vertex_pressure_residual_pa: tuple
    max_pressure_residual_pa: float
    integral: PressureIntegral

    def to_dict(self):
        """Owned JSON data including every lifted point, dA, normal and both pressures."""
        result = asdict(self)
        json.dumps(result, allow_nan=False)
        return result


def two_sphere_case(
    radius_a_m=0.02,
    radius_b_m=0.02,
    modulus_a_pa=1e5,
    modulus_b_pa=3e5,
    overlap_m=0.0004,
    resolution_m=0.0005,
    *,
    origin_world=(0.0, 0.0, 0.0),
):
    """Return a bounded static reference; integral.first=A, second=B.

    Only positive finite radii/moduli and 0<=overlap<min(RA,RB) are admitted.
    Centers are fixed at (0,0,0) and (0,0,RA+RB-overlap); arbitrary poses and deep
    overlap are unsupported. resolution_m bounds XY mesh edges, not curved edge
    lengths or integration error. Both the projected and curved areas are the
    quadrature over an inscribed polygonal disk, whose refinement is explicit.

    Normal is the force direction ON A (negative z at the axis). At every lifted
    quadrature point the common pressure is the mean of pA,pB, only after their
    absolute discrepancy passes 1e-7 Pa. No field clipping or normal repair is
    used. Moments are about one shared origin, not about separate sphere COMs.
    """
    ra, rb = _real(radius_a_m, "radius A"), _real(radius_b_m, "radius B")
    ea, eb = _real(modulus_a_pa, "modulus A"), _real(modulus_b_pa, "modulus B")
    overlap, resolution = _real(overlap_m, "overlap"), _real(resolution_m, "resolution")
    _need(
        ra > 0 and rb > 0 and ea > 0 and eb > 0 and resolution > 0 and 0 <= overlap < min(ra, rb),
        "unsupported sphere parameters or overlap",
    )
    _need(
        isinstance(origin_world, (tuple, list, np.ndarray))
        and (not isinstance(origin_world, np.ndarray) or origin_world.ndim == 1)
        and len(origin_world) == 3,
        "moment origin must have three coordinates",
    )
    origin = tuple(_real(value, "moment origin") for value in origin_world)
    distance = ra + rb - overlap
    _need(isfinite(distance) and distance > max(ra, rb), "unrepresentable shallow separation")
    centers = ((0.0, 0.0, 0.0), (0.0, 0.0, distance))
    if overlap == 0:
        integral = integrate_pressure(triangle_quadrature((), ()), (), origin_world=origin)
        return TwoSphereReference(
            ra,
            rb,
            ea,
            eb,
            overlap,
            resolution,
            centers,
            0.0,
            ra,
            None,
            None,
            0.0,
            (),
            (),
            (),
            (),
            (),
            (),
            0.0,
            integral,
        )

    circle_z = 0.5 * (distance + (ra - rb) * (ra + rb) / distance)
    radius_squared = (ra - circle_z) * (ra + circle_z)
    _need(
        isfinite(circle_z) and isfinite(radius_squared) and radius_squared > 0,
        "intersection circle is not representable",
    )
    circle_radius = sqrt(radius_squared)
    radial_count = sqrt(2) * circle_radius / resolution
    angular_count = 2 * pi * sqrt(2) * circle_radius / resolution
    _need(
        isfinite(radial_count)
        and isfinite(angular_count)
        and max(radial_count, angular_count) <= MAX_VERTICES,
        "resolution exceeds mesh budget",
    )
    rings, sectors = max(1, ceil(radial_count)), max(8, ceil(angular_count))
    _need(
        1 + rings * sectors <= MAX_VERTICES and sectors * (2 * rings - 1) <= MAX_TRIANGLES,
        "disk mesh budget exceeded",
    )
    vertices = [(0.0, 0.0, 0.0)]
    for ring in range(1, rings + 1):
        radius = circle_radius * ring / rings
        vertices.extend(
            (radius * cos(2 * pi * i / sectors), radius * sin(2 * pi * i / sectors), 0.0)
            for i in range(sectors)
        )
    faces = [(0, 1 + i, 1 + (i + 1) % sectors) for i in range(sectors)]
    for ring in range(2, rings + 1):
        inner, outer = 1 + (ring - 2) * sectors, 1 + (ring - 1) * sectors
        for i in range(sectors):
            following = (i + 1) % sectors
            faces.extend(
                (
                    (inner + i, outer + i, outer + following),
                    (inner + i, outer + following, inner + following),
                )
            )
    projected = triangle_quadrature(vertices, faces)
    mesh = np.asarray(projected.vertices_world, dtype=np.float64).copy()
    # Boundary vertices lie on both zero-pressure surfaces. Do not root a zero-width
    # bracket reconstructed from rounded sine/cosine coordinates.
    interior_count = 1 + (rings - 1) * sectors
    vertex_rho = np.hypot(mesh[:, 0], mesh[:, 1])
    mesh[:interior_count, 2] = _roots(vertex_rho[:interior_count], ra, rb, ea, eb, distance)
    mesh[interior_count:, 2] = circle_z
    vertex_pa, vertex_pb, _, _ = _fields(vertex_rho, mesh[:, 2], ra, rb, ea, eb, distance)
    vertex_residual = vertex_pa - vertex_pb

    points = np.asarray(projected.points_world, dtype=np.float64).copy()
    rho = np.hypot(points[:, 0], points[:, 1])
    points[:, 2] = _roots(rho, ra, rb, ea, eb, distance)
    pa, pb, field_ra, field_rb = _fields(rho, points[:, 2], ra, rb, ea, eb, distance)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # g_z is strictly negative; z_rho=-g_rho/g_z.
        factor = eb / rb / field_rb - ea / ra / field_ra
        denominator = (
            ea / ra * points[:, 2] / field_ra + eb / rb * (distance - points[:, 2]) / field_rb
        )
        slopes = rho * factor / denominator
        jacobian = np.sqrt(1 + slopes * slopes)
        normals = (
            np.column_stack(
                (
                    points[:, 0] * factor / denominator,
                    points[:, 1] * factor / denominator,
                    -np.ones(len(points)),
                )
            )
            / jacobian[:, None]
        )
        areas = np.asarray(projected.area_weights_m2) * jacobian
        pressure = 0.5 * pa + 0.5 * pb
    _need(
        all(
            np.isfinite(value).all()
            for value in (factor, denominator, slopes, jacobian, normals, areas, pressure)
        )
        and np.all(denominator > 0)
        and np.all(pa >= 0)
        and np.all(pb >= 0),
        "invalid curved field or Jacobian",
    )
    residual = pa - pb
    maximum_residual = max(float(np.max(np.abs(residual))), float(np.max(np.abs(vertex_residual))))
    _need(
        isfinite(maximum_residual) and maximum_residual <= _PRESSURE_TOLERANCE_PA,
        "surface pressure continuity tolerance not met",
    )
    curved = SurfaceQuadrature(
        _tuples(mesh),
        projected.triangles,
        _tuples(points),
        _tuples(normals),
        tuple(map(float, areas)),
        projected.triangle_indices,
    )
    integral = integrate_pressure(curved, pressure, origin_world=origin)
    if ra != rb or ea != eb:
        # Even a nearly planar unequal-field interface is not an exact plane.
        integral = replace(integral, cop_world=None)
    return TwoSphereReference(
        ra,
        rb,
        ea,
        eb,
        overlap,
        resolution,
        centers,
        circle_radius,
        circle_z,
        float(mesh[0, 2]),
        float(0.5 * vertex_pa[0] + 0.5 * vertex_pb[0]),
        float(sum(projected.area_weights_m2)),
        projected.area_weights_m2,
        tuple(map(float, slopes)),
        tuple(map(float, pa)),
        tuple(map(float, pb)),
        tuple(map(float, residual)),
        tuple(map(float, vertex_residual)),
        maximum_residual,
        integral,
    )
