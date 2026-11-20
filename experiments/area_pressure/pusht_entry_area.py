"""Bounded frozen-entry area points for the original RR capsule/T-box geometry.

The caller supplies an already forwarded MuJoCo entry; this module never forwards,
steps, modifies native contacts, or fits a gradient to their loads. A is the T block,
B is the tool. The normal and returned point force act on A. Ground remains native.
"""

import heapq
from itertools import product
from types import SimpleNamespace

import mujoco as mj
import numpy as np

from experiments.area_pressure.capsule_box_reference import (
    _residual,
    axis_box_clearance,
    exterior_tiles,
)
from experiments.area_pressure.pusht_saved_contact import GRADIENT_PA_M, PushGeometry

MAX_POINTS = 256
MAX_DEPTH = 12
LOAD_RTOL = 0.02
LOAD_ATOL_N = 1e-9
MU = 0.7
SLIP_M_S = 0.01


def _face_patch(center, half, ends, radius):
    """Conservative rectangle containing the capsule's intersection with this face.

    Optimize each tangential coordinate of the union of section disks along the
    finite capsule axis, including both endpoint spheres. Cropping before splitting
    resolves microscopic penetration strips without sampling the entire T face.
    """
    axis = int(np.flatnonzero(half == 0)[0])
    delta = ends[1] - ends[0]
    z0, dz = float(ends[0, axis] - center[axis]), float(delta[axis])
    left, right = 0.0, 1.0
    if dz == 0:
        if abs(z0) >= radius:
            return None
    else:
        cuts = sorted(((-radius - z0) / dz, (radius - z0) / dz))
        left, right = max(left, cuts[0]), min(right, cuts[1])
        if left >= right:
            return None
    lo, hi = center - half, center + half
    pad = 64 * np.finfo(float).eps * max(1.0, float(np.abs(ends).max()), radius)
    for tangent in np.flatnonzero(half):
        times = [left, right]
        slope = float(delta[tangent])
        if dz != 0:
            extremum = np.sign(dz) * radius * slope / np.hypot(slope, dz)
            for height in (extremum, -extremum):
                t = (height - z0) / dz
                if left <= t <= right:
                    times.append(t)
        t = np.asarray(times)
        z = np.abs(z0 + t * dz)
        disk_radius = np.sqrt(np.maximum(0.0, (radius - z) * (radius + z)))
        coordinate = ends[0, tangent] + t * slope
        lo[tangent] = max(lo[tangent], float((coordinate - disk_radius).min()) - pad)
        hi[tangent] = min(hi[tangent], float((coordinate + disk_radius).max()) + pad)
        if lo[tangent] >= hi[tangent]:
            return None
    return (lo + hi) / 2, (hi - lo) / 2


def _offsets(half):
    return np.asarray(list(product(*[(-h, h) if h else (0.0,) for h in half])))


def _cell(center, half, normal, face, level, ends, radius):
    residual = _residual(center[None], ends)[0]
    distance = float(np.linalg.norm(residual))
    unit = residual / distance if distance else np.zeros(3)
    pad = (
        64 * np.finfo(float).eps * max(1.0, float(np.abs(center).max()), float(np.abs(ends).max()))
    )
    lower = max(0.0, distance - float(np.abs(unit) @ half) - pad)
    if lower >= radius:
        return None
    corners = center + _offsets(half)
    upper = float(np.linalg.norm(_residual(corners, ends), axis=1).max()) + pad
    area = float(4 * half[half > 0].prod())
    low_load = area * GRADIENT_PA_M * max(radius - upper, 0.0)
    high_load = area * GRADIENT_PA_M * max(radius - lower, 0.0)
    return dict(
        center=center,
        half=half,
        normal=normal,
        face=face,
        level=level,
        weight=area,
        gap=distance - radius,
        low=low_load,
        high=high_load,
    )


def _quadrature(ends, boxes, radius):
    leaves, queue = {}, []
    serial = 0

    def add(cell):
        nonlocal serial
        if cell is not None:
            leaves[serial] = cell
            heapq.heappush(queue, (-(cell["high"] - cell["low"]), serial))
            serial += 1

    clearance = axis_box_clearance(ends, boxes, radius)
    if clearance < 0:
        for face, (center, half, normal) in enumerate(exterior_tiles(boxes)):
            patch = _face_patch(center, half, ends, radius)
            if patch is not None:
                add(_cell(*patch, normal, face, 0, ends, radius))
        if not leaves:
            raise ValueError("capsule overlaps the solid without a resolved exterior patch")
    while queue and len(leaves) + 3 <= MAX_POINTS:
        low = sum(cell["low"] for cell in leaves.values())
        high = sum(cell["high"] for cell in leaves.values())
        if high - low <= max(LOAD_ATOL_N, LOAD_RTOL * high):
            break
        _priority, identity = heapq.heappop(queue)
        parent = leaves[identity]
        if parent["level"] >= MAX_DEPTH:
            continue
        del leaves[identity]
        half = parent["half"] / 2
        for offset in _offsets(half):
            add(
                _cell(
                    parent["center"] + offset,
                    half,
                    parent["normal"],
                    parent["face"],
                    parent["level"] + 1,
                    ends,
                    radius,
                )
            )
    cells = list(leaves.values())
    low, high = (sum(cell[key] for cell in cells) for key in ("low", "high"))
    live = [cell for cell in cells if cell["gap"] < 0]
    if not live and high > LOAD_ATOL_N:
        raise ValueError("bounded quadrature did not resolve the nonzero contact patch")
    metadata = dict(
        method="cropped-exterior-adaptive-midpoint-v1",
        max_points=MAX_POINTS,
        max_depth=MAX_DEPTH,
        leaf_count=len(cells),
        point_count=len(live),
        attained_depth=max((cell["level"] for cell in cells), default=0),
        static_normal_load_bounds_n=[low, high],
        static_normal_load_n=sum(-cell["weight"] * GRADIENT_PA_M * cell["gap"] for cell in live),
        relative_load_target=LOAD_RTOL,
        absolute_load_target_n=LOAD_ATOL_N,
        bound_target_met=bool(high - low <= max(LOAD_ATOL_N, LOAD_RTOL * high)),
        clearance_m=clearance,
    )
    return live, metadata


def build_area(model, data):
    """Return frozen world points and both point Jacobians at this exact entry.

    The fixed gradient is the prior static reference's 1e6 Pa/m. Integration is
    bounded to 256 midpoint leaves. Its reported static pressure-load interval is
    a geometric bound, not a convergence claim or a final-velocity error bound.
    Initial exterior points are frozen: newly touching regions require a new build.
    """
    scene = SimpleNamespace(object_geom_names=("object-0", "object-1"), support_geom_name="support")
    geometry = PushGeometry(model, scene)
    if geometry.body == geometry.tool or model.nv < 1:
        raise ValueError("expected distinct movable T-block and capsule bodies")
    owned = tuple(np.flatnonzero(model.geom_bodyid == geometry.body))
    if set(owned) != set(geometry.objects):
        raise ValueError("only the two co-aligned T-block boxes are supported")
    body_ids = (int(geometry.body), int(geometry.tool))
    rotations = [np.asarray(data.xmat[body]).reshape(3, 3) for body in body_ids]
    origins = [np.asarray(data.xpos[body]) for body in body_ids]
    for rotation, origin in zip(rotations, origins, strict=True):
        if (
            not np.isfinite(rotation).all()
            or not np.isfinite(origin).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=3e-6, rtol=0)
            or np.linalg.det(rotation) <= 0
        ):
            raise ValueError("finite rigid entry poses are required")
    rotation, tool_rotation = rotations
    origin, tool_origin = origins
    ends = (geometry.local_ends @ tool_rotation.T + tool_origin - origin) @ rotation
    if not np.isfinite(ends).all() or not np.isfinite(geometry.radius) or geometry.radius <= 0:
        raise ValueError("finite capsule with positive radius required")
    live, metadata = _quadrature(ends, geometry.boxes, float(geometry.radius))
    local_points = np.asarray([cell["center"] for cell in live], dtype=float).reshape(-1, 3)
    local_normals = np.asarray([cell["normal"] for cell in live], dtype=float).reshape(-1, 3)
    points = local_points @ rotation.T + origin
    normals = -local_normals @ rotation.T
    if len(live):
        normals /= np.linalg.norm(normals, axis=1)[:, None]
    weights = np.asarray([cell["weight"] for cell in live], dtype=float)
    initial_gap = np.asarray([cell["gap"] for cell in live], dtype=float)
    jac_a, jac_b = (np.zeros((len(live), 3, model.nv)) for _ in range(2))
    rotational = np.zeros((3, model.nv))
    for i, point in enumerate(points):
        mj.mj_jac(model, data, jac_a[i], rotational, point, body_ids[0])
        mj.mj_jac(model, data, jac_b[i], rotational, point, body_ids[1])
    if not np.isfinite(jac_a).all() or not np.isfinite(jac_b).all():
        raise ValueError("nonfinite same-entry point Jacobians")
    return dict(
        points=points,
        normals=normals,
        weights=weights,
        stiffness=GRADIENT_PA_M * weights,
        initial_gap=initial_gap,
        jac=jac_a - jac_b,
        jac_a=jac_a,
        jac_b=jac_b,
        body_ids=body_ids,
        object_geom_ids=tuple(map(int, geometry.objects)),
        pusher_geom_id=int(geometry.pusher),
        face_ids=np.asarray([c["face"] for c in live], dtype=int),
        gradient_pa_m=GRADIENT_PA_M,
        mu=MU,
        slip=SLIP_M_S,
        quadrature=metadata,
    )


def area_response(area, velocity, h):
    """Substitute the candidate final velocity, returning Q and force on the T.

    u=(J_T-J_tool)v; phi1=phi0+h*n.u; Fn=max(0,-k*phi1).
    Ft=-mu*Fn*u_t/sqrt(u_t.u_t+slip^2); Q=sum (J_T-J_tool)^T(Fn*n+Ft).
    k already includes dA, so neither force nor Q multiplies weights again.
    """
    velocity = np.asarray(velocity, dtype=float)
    jac = np.asarray(area["jac"], dtype=float)
    normals = np.asarray(area["normals"], dtype=float)
    gap = np.asarray(area["initial_gap"], dtype=float)
    stiffness = np.asarray(area["stiffness"], dtype=float)
    points, weights = (np.asarray(area[name], dtype=float) for name in ("points", "weights"))
    mu, slip, h = float(area.get("mu", MU)), float(area.get("slip", SLIP_M_S)), float(h)
    if (
        jac.ndim != 3
        or jac.shape[1] != 3
        or velocity.shape != (jac.shape[2],)
        or normals.shape != jac.shape[:2]
        or points.shape != normals.shape
        or gap.shape != (len(jac),)
        or stiffness.shape != gap.shape
        or weights.shape != gap.shape
        or not all(
            np.isfinite(a).all() for a in (jac, velocity, normals, gap, stiffness, points, weights)
        )
        or not np.isfinite([mu, slip, h]).all()
        or mu < 0
        or slip <= 0
        or h <= 0
        or np.any(stiffness <= 0)
        or np.any(weights <= 0)
        or np.any(gap > 0)
        or not np.allclose(np.sum(normals * normals, axis=1), 1.0, atol=1e-10, rtol=0)
    ):
        raise ValueError("invalid finite frozen-entry area response inputs")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            relative = np.einsum("sij,j->si", jac, velocity)
            normal_speed = np.einsum("si,si->s", normals, relative)
            tangent = relative - normal_speed[:, None] * normals
            fn = np.maximum(0.0, -stiffness * (gap + h * normal_speed))
            denominator = np.sqrt(np.einsum("si,si->s", tangent, tangent) + slip * slip)
            force = fn[:, None] * normals - mu * fn[:, None] * tangent / denominator[:, None]
            generalized = np.einsum("sij,si->j", jac, force)
    except FloatingPointError as error:
        raise ValueError("nonfinite area response algebra") from error
    if not np.isfinite(force).all() or not np.isfinite(generalized).all():
        raise ValueError("nonfinite area response")
    return generalized, force
