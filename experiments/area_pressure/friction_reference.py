"""V04 prescribed-slip area traction reference, with no state evolution.

Pressure and both participants' point velocities are inputs. This module solves
neither contact nor static friction and does not reproduce an external engine.
Normals point along the normal force on A; all forces on B are their opposites
at the same points. Both participants' moments use one common world origin.
"""

import json
from dataclasses import asdict, dataclass
from math import cos, hypot, isfinite, pi, sin
from numbers import Integral, Real

import numpy as np

from .reference import (
    MAX_POINTS,
    MAX_TRIANGLES,
    MAX_VERTICES,
    PressureIntegral,
    SurfaceQuadrature,
    _real,
    _tuples,
    integrate_pressure,
    triangle_quadrature,
)


def _need(condition, message):
    if not condition:
        raise ValueError("prescribed-slip reference: " + message)


def _numeric(value, shape, name):
    """Own bounded real data without coercing Python booleans into numbers."""
    _need(isinstance(value, (tuple, list, np.ndarray)), name + " requires explicit values")
    if isinstance(value, np.ndarray):
        _need(value.shape == shape and value.dtype.kind in "fiu", "invalid " + name)
    else:
        _need(len(value) == shape[0], "invalid " + name)
        rows = value if len(shape) == 2 else (value,)
        width = shape[-1]
        for row in rows:
            _need(
                isinstance(row, (tuple, list, np.ndarray))
                and (not isinstance(row, np.ndarray) or row.ndim == 1)
                and len(row) == width,
                "invalid " + name,
            )
            _need(
                all(
                    isinstance(item, Real) and not isinstance(item, (bool, np.bool_))
                    for item in row
                ),
                name + " must contain real non-boolean values",
            )
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(value, dtype=np.float64, copy=True).reshape(shape)
    _need(np.isfinite(result).all(), "nonfinite " + name)
    return result


def disk_quadrature(radius_m=0.01, centers_xy=((0.0, 0.0),), *, rings=8, sectors=64):
    """One or two inscribed polygonal disks at z=0, n=+ez, with actual triangle dA.

    Positive-radius interiors must be disjoint; boundary tangency is allowed.
    Each disk retains its own vertex/triangle IDs, even at a coincident boundary
    point. No topology/contact detection or analytic-area renormalization occurs.
    """
    radius = _real(radius_m, "disk radius")
    _need(radius > 0 and isfinite(2 * radius), "positive representable radius required")
    _need(
        isinstance(centers_xy, (tuple, list, np.ndarray))
        and (not isinstance(centers_xy, np.ndarray) or centers_xy.ndim == 2)
        and 1 <= len(centers_xy) <= 2,
        "one or two disk centers required",
    )
    count = len(centers_xy)
    centers = _numeric(centers_xy, (count, 2), "disk centers")
    _need(
        isinstance(rings, Integral)
        and not isinstance(rings, (bool, np.bool_))
        and 1 <= rings <= MAX_VERTICES,
        "rings must be a bounded positive integer",
    )
    _need(
        isinstance(sectors, Integral)
        and not isinstance(sectors, (bool, np.bool_))
        and 8 <= sectors <= MAX_TRIANGLES,
        "sectors must be a bounded integer >=8",
    )
    rings, sectors = int(rings), int(sectors)
    vertices_count = count * (1 + rings * sectors)
    triangles_count = count * sectors * (2 * rings - 1)
    _need(
        vertices_count <= MAX_VERTICES
        and triangles_count <= MAX_TRIANGLES
        and 3 * triangles_count <= MAX_POINTS,
        "disk mesh exceeds shared quadrature budget",
    )
    if count == 2:
        with np.errstate(over="ignore", invalid="ignore"):
            displacement = centers[1] - centers[0]
        separation = hypot(float(displacement[0]), float(displacement[1]))
        _need(
            isfinite(separation) and separation >= 2 * radius,
            "disk interiors overlap or separation is nonfinite",
        )
    vertices, faces = [], []
    for center in centers:
        base = len(vertices)
        vertices.append((float(center[0]), float(center[1]), 0.0))
        for ring in range(1, rings + 1):
            r = radius * ring / rings
            vertices.extend(
                (
                    float(center[0]) + r * cos(2 * pi * i / sectors),
                    float(center[1]) + r * sin(2 * pi * i / sectors),
                    0.0,
                )
                for i in range(sectors)
            )
        faces.extend((base, base + 1 + i, base + 1 + (i + 1) % sectors) for i in range(sectors))
        for ring in range(2, rings + 1):
            inner = base + 1 + (ring - 2) * sectors
            outer = base + 1 + (ring - 1) * sectors
            for i in range(sectors):
                following = (i + 1) % sectors
                faces.extend(
                    (
                        (inner + i, outer + i, outer + following),
                        (inner + i, outer + following, inner + following),
                    )
                )
    return triangle_quadrature(vertices, faces)


@dataclass(frozen=True)
class FrictionIntegral:
    qualification: str
    law: str
    mu: float
    slip_speed_m_s: float
    normal: PressureIntegral
    velocity_a_world: tuple
    velocity_b_world: tuple
    tangential_velocity_world: tuple
    tangential_traction_pa: tuple
    first_friction_force_per_point_n: tuple
    second_friction_force_per_point_n: tuple
    first_friction_force_n: tuple
    second_friction_force_n: tuple
    first_friction_moment_nm: tuple
    second_friction_moment_nm: tuple
    first_total_force_n: tuple
    second_total_force_n: tuple
    first_total_moment_nm: tuple
    second_total_moment_nm: tuple
    friction_power_per_point_w: tuple
    friction_power_w: float

    def to_dict(self):
        """Detached complete JSON-compatible record; no NaN/Infinity emission."""
        result = asdict(self)
        json.dumps(result, allow_nan=False)
        return result


def integrate_friction(
    quadrature,
    pressure_pa,
    *,
    velocity_a_world,
    velocity_b_world,
    mu=0.5,
    law="regularized",
    slip_speed_m_s=0.001,
    origin_world=(0.0, 0.0, 0.0),
):
    """Integrate prescribed normal pressure and relative tangential sliding traction.

    Velocities are explicit [quadrature_point,3] world vectors in m/s, including
    for curved surfaces. vt=(I-n*n.T)*(vA-vB). Coulomb sliding uses -mu*p*vt/|vt|;
    at exactly zero tangential speed it selects zero, not a static-friction
    solution. The regularized law uses sqrt(|vt|^2+vs^2), with positive vs.
    Both laws save and validate vs, although Coulomb does not use it.

    Friction power is the actual sum of dFt dot (vA-vB); normal work is excluded.
    No total force, torque or power is clamped or zeroed by symmetry assumptions.
    """
    coefficient, slip_speed = _real(mu, "friction coefficient"), _real(slip_speed_m_s, "slip speed")
    _need(coefficient >= 0 and slip_speed > 0, "nonnegative mu and positive slip speed required")
    _need(type(law) is str and law in ("coulomb", "regularized"), "unsupported friction law")
    _need(isinstance(quadrature, SurfaceQuadrature), "SurfaceQuadrature required")
    # Reuse the normal integrator's complete geometry, pressure, origin and budget
    # validation. Additional explicit numeric reads reject mixed Python booleans.
    normal = integrate_pressure(quadrature, pressure_pa, origin_world=origin_world)
    count = len(normal.quadrature.points_world)
    _numeric(quadrature.vertices_world, (len(normal.quadrature.vertices_world), 3), "vertices")
    points = _numeric(quadrature.points_world, (count, 3), "quadrature points")
    normals = _numeric(quadrature.normals_world, (count, 3), "quadrature normals")
    areas = _numeric(quadrature.area_weights_m2, (count,), "area weights")
    pressure = _numeric(pressure_pa, (count,), "pressure")
    origin = _numeric(origin_world, (3,), "moment origin")
    velocity_a = _numeric(velocity_a_world, (count, 3), "velocity A")
    velocity_b = _numeric(velocity_b_world, (count, 3), "velocity B")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        relative = velocity_a - velocity_b
        tangent = relative - np.einsum("ij,ij->i", relative, normals)[:, None] * normals
        speed = np.hypot.reduce(tangent, axis=1)
        direction = np.zeros_like(tangent)
        if law == "coulomb":
            np.divide(tangent, speed[:, None], out=direction, where=speed[:, None] != 0)
        else:
            direction = tangent / np.hypot(speed, slip_speed)[:, None]
        traction = -(coefficient * pressure)[:, None] * direction
        forces = traction * areas[:, None]
        force = forces.sum(axis=0)
        moment = np.cross(points - origin, forces).sum(axis=0)
        total_force = np.asarray(normal.first_force_n) + force
        total_moment = np.asarray(normal.first_moment_nm) + moment
        power_per_point = np.einsum("ij,ij->i", forces, relative)
        power = float(power_per_point.sum())
    _need(
        all(
            np.isfinite(value).all()
            for value in (
                relative,
                tangent,
                speed,
                direction,
                traction,
                forces,
                force,
                moment,
                total_force,
                total_moment,
                power_per_point,
            )
        )
        and isfinite(power),
        "nonfinite integrated friction",
    )
    return FrictionIntegral(
        "NUMPY_PRESCRIBED_SLIP_REFERENCE",
        law,
        coefficient,
        slip_speed,
        normal,
        _tuples(velocity_a),
        _tuples(velocity_b),
        _tuples(tangent),
        _tuples(traction),
        _tuples(forces),
        _tuples(-forces),
        tuple(map(float, force)),
        tuple(map(float, -force)),
        tuple(map(float, moment)),
        tuple(map(float, -moment)),
        tuple(map(float, total_force)),
        tuple(map(float, -total_force)),
        tuple(map(float, total_moment)),
        tuple(map(float, -total_moment)),
        tuple(map(float, power_per_point)),
        power,
    )
