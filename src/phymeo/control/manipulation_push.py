"""Finite planar BOX-union push candidates, without IK or physical integration.

Analytic distances are queries on admitted planar rectangles with a declared
1e-9 m tolerance. They are not a mathematical CCD or whole-robot certificate.
Entering direction and the nominal inward-normal force are distinct quantities.
"""

import json
from dataclasses import dataclass
from hashlib import sha256
from math import cos, hypot, isfinite, pi, sin
from numbers import Integral

import numpy as np

from .manipulation_diagnostic import _configuration, _readonly, _require
from .manipulation_path import _items, _model_bytes, _number
from .manipulation_pose import _ancestors, _names

QUERY_TOLERANCE_M = 1e-9
ALIGNMENT_TOLERANCE = 1e-8
MAX_BOUNDARY_EDGES = 16


@dataclass(frozen=True)
class PlanarPushRectangle:
    geometry_id: int
    geometry_name: str
    lower_xy: tuple[float, float]
    upper_xy: tuple[float, float]
    z_interval_world: tuple[float, float]


@dataclass(frozen=True)
class PlanarPushBoundaryEdge:
    index: int
    start_world: tuple[float, float, float]
    end_world: tuple[float, float, float]
    outward_normal_world: tuple[float, float, float]
    owner_geometry_ids: tuple[int, ...]


@dataclass(frozen=True)
class NativePlanarPushContactCandidate:
    index: int
    edge_index: int
    edge_fraction: float
    approach_angle_rad: float
    owner_geometry_ids: tuple[int, ...]
    point_world: tuple[float, float, float]
    outward_normal_world: tuple[float, float, float]
    contact_center_world: tuple[float, float, float]
    precontact_center_world: tuple[float, float, float]
    approach_direction_world: tuple[float, float, float]
    normal_wrench_com_world: tuple[float, ...]
    contact_distances_m: tuple[float, ...]
    entry_distances_m: tuple[float, ...]
    usable: bool
    reason: str


@dataclass(frozen=True)
class NativePlanarPushContacts:
    input_binding: str
    q: np.ndarray
    object_body_id: int
    object_com_world: tuple[float, float, float]
    pusher_geom_id: int
    pusher_radius_m: float
    plane_origin_world: tuple[float, float, float]
    plane_axes_world: np.ndarray
    rectangles: tuple[PlanarPushRectangle, ...]
    boundary: tuple[PlanarPushBoundaryEdge, ...]
    candidates: tuple[NativePlanarPushContactCandidate, ...]
    query_tolerance_m: float


def _coordinates(values):
    coordinates, indices = [], {}
    for value in sorted(values):
        # Coalesce only against the group's first coordinate, never a drifting
        # chain. Raw rectangle bounds remain unchanged for distance reporting.
        if not coordinates or value - coordinates[-1] > QUERY_TOLERANCE_M:
            coordinates.append(value)
        indices[value] = len(coordinates) - 1
    return tuple(coordinates), indices


def _boundary(rectangles):
    """Exterior of the occupied coordinate cells; no union-AABB substitution."""
    xs, xi = _coordinates([v for r in rectangles for v in (r.lower_xy[0], r.upper_xy[0])])
    ys, yi = _coordinates([v for r in rectangles for v in (r.lower_xy[1], r.upper_xy[1])])
    occupied = set()
    for rect in rectangles:
        lo, hi = rect.lower_xy, rect.upper_xy
        _require(xi[lo[0]] < xi[hi[0]] and yi[lo[1]] < yi[hi[1]], "BOX below query resolution")
        occupied.update(
            (i, j) for i in range(xi[lo[0]], xi[hi[0]]) for j in range(yi[lo[1]], yi[hi[1]])
        )
    pieces = []
    for i, j in sorted(occupied):
        for axis, sign, neighbour, fixed, low, high in (
            (0, -1, (i - 1, j), xs[i], ys[j], ys[j + 1]),
            (0, 1, (i + 1, j), xs[i + 1], ys[j], ys[j + 1]),
            (1, -1, (i, j - 1), ys[j], xs[i], xs[i + 1]),
            (1, 1, (i, j + 1), ys[j + 1], xs[i], xs[i + 1]),
        ):
            if neighbour in occupied:
                continue
            other, middle = 1 - axis, 0.5 * (low + high)
            owners = tuple(
                r.geometry_id
                for r in rectangles
                if abs((r.lower_xy if sign < 0 else r.upper_xy)[axis] - fixed)
                <= QUERY_TOLERANCE_M
                and r.lower_xy[other] - QUERY_TOLERANCE_M <= middle
                <= r.upper_xy[other] + QUERY_TOLERANCE_M
            )
            _require(bool(owners), "exterior edge has no source BOX owner")
            pieces.append((axis, sign, fixed, owners, low, high))
    merged = []
    for piece in sorted(pieces):
        axis, sign, fixed, owners, low, high = piece
        if merged and merged[-1][:4] == piece[:4] and merged[-1][5] == low:
            merged[-1] = (axis, sign, fixed, owners, merged[-1][4], high)
        else:
            merged.append(piece)
    _require(0 < len(merged) <= MAX_BOUNDARY_EDGES, "exterior boundary exceeds its edge budget")
    return tuple(merged)


def _point_rectangle(point, rectangle):
    return hypot(
        max(rectangle.lower_xy[0] - point[0], 0.0, point[0] - rectangle.upper_xy[0]),
        max(rectangle.lower_xy[1] - point[1], 0.0, point[1] - rectangle.upper_xy[1]),
    )


def _point_segment(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = hypot(dx, dy)
    if length == 0.0:
        return hypot(point[0] - start[0], point[1] - start[1])
    ux, uy = dx / length, dy / length
    along = min(length, max(0.0, (point[0] - start[0]) * ux + (point[1] - start[1]) * uy))
    return hypot(point[0] - start[0] - along * ux, point[1] - start[1] - along * uy)


def _segment_rectangle(start, end, rectangle):
    # Slab intersection, followed by all endpoint/edge and corner/segment
    # distances. The segment interior is tested analytically, not sampled.
    lower, upper, intersects = 0.0, 1.0, True
    for axis in range(2):
        delta = end[axis] - start[axis]
        if delta == 0:
            if not rectangle.lower_xy[axis] <= start[axis] <= rectangle.upper_xy[axis]:
                intersects = False
                break
        else:
            first = (rectangle.lower_xy[axis] - start[axis]) / delta
            second = (rectangle.upper_xy[axis] - start[axis]) / delta
            lower, upper = max(lower, min(first, second)), min(upper, max(first, second))
            if lower > upper:
                intersects = False
                break
    if intersects:
        return 0.0
    distances = [_point_rectangle(start, rectangle), _point_rectangle(end, rectangle)]
    distances.extend(
        _point_segment((x, y), start, end)
        for x in (rectangle.lower_xy[0], rectangle.upper_xy[0])
        for y in (rectangle.lower_xy[1], rectangle.upper_xy[1])
    )
    return min(distances)


def _tuple3(value):
    result = tuple(map(float, value))
    _require(len(result) == 3 and all(map(isfinite, result)), "nonfinite candidate coordinate")
    return result


def enumerate_native_planar_push_contacts(
    model,
    q,
    *,
    object_geom_names,
    pusher_geom_name,
    edge_fractions=(0.25, 0.5, 0.75),
    approach_angles=(0.0,),
    stand_off_m=0.01,
    max_candidates=128,
):
    """Return every declared edge/point/approach attempt, including blocked ones.

    Object names must cover all 1..2 BOX geoms on one native body, with no
    descendant geometry omitted. Boxes must share planar axes and a Z interval
    within 1e-8. A common yaw, translation, unequal widths and overlapping,
    touching, contained or separated rectangles are supported. Small coordinate
    differences up to 1e-9 m are coalesced only for boundary topology. Original
    rectangle coordinates are used for all reported analytic distances.

    The pusher must be a sphere or vertical capsule. Its actual Z volume must
    overlap the boxes, and its center/axis segment must intersect their shared Z
    interval. Cap-only overlap cannot use the full-radius planar circle and is
    explicitly unsupported. No height or object configuration is changed.

    Fractions are 1..8 distinct explicit values strictly inside (0,1); angles
    contain 1..5 explicit finite values with abs(angle)<pi/2. Each angle rotates
    the inward normal counterclockwise in the world-up plane. The contact-center
    in-plane offset is radius*outward_normal; its height stays at the native
    pusher geom center. The surface point uses the shared side-contact Z slice.
    Precontact is center-stand_off*direction.
    A usable candidate has no circle penetration beyond 1e-9 m at contact or
    anywhere on that straight entry segment against either BOX. Rejected attempts
    are retained with their raw distances and a reason. More than max_candidates
    (hard maximum 128) raises before candidate generation; nothing is truncated.

    COM is native xipos, never the body origin or the union's area centroid.
    The unit normal wrench acts on the object at the contact point about this
    COM. It is a geometric nominal wrench, not a prediction of frictional force,
    yaw response, reachability, whole-robot clearance, or task success.
    """
    import mujoco as mj

    _require(isinstance(model, mj.MjModel), "an original MuJoCo model is required")
    _require(
        0 < model.nq <= 45 and 0 < model.nv <= 35 and model.nbody <= 128 and model.ngeom <= 128,
        "model exceeds planar push dimensions",
    )
    names = _names(object_geom_names, "object_geom_names", 1, 2)
    _require(type(pusher_geom_name) is str and pusher_geom_name, "pusher geom name is required")
    fractions = tuple(_number(v, "edge fraction") for v in _items(edge_fractions, "fractions", 1, 8))
    angles = tuple(_number(v, "approach angle") for v in _items(approach_angles, "angles", 1, 5))
    _require(
        len(set(fractions)) == len(fractions) and all(0 < v < 1 for v in fractions),
        "edge fractions must be distinct and strictly inside 0..1",
    )
    _require(all(abs(v) < pi / 2 for v in angles), "approach must point inward")
    stand_off = _number(stand_off_m, "stand_off_m")
    _require(stand_off > 0, "stand_off_m must be positive")
    _require(
        isinstance(max_candidates, Integral)
        and not isinstance(max_candidates, bool)
        and 1 <= max_candidates <= 128,
        "max_candidates must be an integer in 1..128",
    )
    configuration = _configuration(mj, model, q)
    objects = tuple(
        sorted(int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)) for name in names)
    )
    pusher = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, pusher_geom_name))
    _require(objects[0] >= 0 and pusher >= 0 and pusher not in objects, "unknown or reused geom")
    body = int(model.geom_bodyid[objects[0]])
    _require(body > 0 and all(int(model.geom_bodyid[g]) == body for g in objects), "one object body")
    _require(
        set(objects) == set(map(int, np.flatnonzero(model.geom_bodyid == body))),
        "object geoms must cover the complete native body",
    )
    for geom in range(model.ngeom):
        if geom not in objects:
            _require(
                body not in _ancestors(model, int(model.geom_bodyid[geom])),
                "object descendant geometry is not covered",
            )
    _require(
        all(int(model.geom_type[g]) == int(mj.mjtGeom.mjGEOM_BOX) for g in objects),
        "object union requires BOX geoms",
    )
    kind = int(model.geom_type[pusher])
    _require(
        kind in (int(mj.mjtGeom.mjGEOM_SPHERE), int(mj.mjtGeom.mjGEOM_CAPSULE)),
        "pusher must be a sphere or vertical capsule",
    )
    binary = _model_bytes(mj, model)
    data = mj.MjData(model)
    data.qpos[:] = configuration
    mj.mj_kinematics(model, data)
    _require(data.qpos.tobytes() == configuration.tobytes(), "FK changed candidate q")
    for value in (data.geom_xpos, data.geom_xmat, data.xpos, data.xipos):
        _require(np.isfinite(value).all(), "nonfinite native planar geometry")
    origin = data.xpos[body].copy()
    first_rotation = data.geom_xmat[objects[0]].reshape(3, 3)
    _require(
        np.max(np.abs(first_rotation[:2, 2])) <= ALIGNMENT_TOLERANCE
        and abs(abs(first_rotation[2, 2]) - 1) <= ALIGNMENT_TOLERANCE,
        "object BOX is not planar",
    )
    xaxis = first_rotation[:, 0].copy()
    xaxis[2] = 0.0
    xaxis /= np.linalg.norm(xaxis)
    axes = np.column_stack((xaxis, np.array((-xaxis[1], xaxis[0], 0.0)), np.array((0.0, 0.0, 1.0))))
    rectangles = []
    for geom in objects:
        relative = axes.T @ data.geom_xmat[geom].reshape(3, 3)
        mapping = tuple(map(int, np.argmax(np.abs(relative), axis=0)))
        snapped = np.zeros((3, 3))
        for column, axis in enumerate(mapping):
            snapped[axis, column] = 1.0 if relative[axis, column] >= 0 else -1.0
        _require(
            set(mapping) == {0, 1, 2}
            and mapping[2] == 2
            and np.max(np.abs(relative - snapped)) <= ALIGNMENT_TOLERANCE,
            "BOX axes are not co-aligned",
        )
        size = np.asarray(model.geom_size[geom], dtype=np.float64)
        _require(np.isfinite(size).all() and np.all(size > 0), "invalid BOX dimensions")
        half = np.abs(snapped) @ size
        center = axes.T @ (data.geom_xpos[geom] - origin)
        lo, hi = center[:2] - half[:2], center[:2] + half[:2]
        z = (float(data.geom_xpos[geom, 2] - half[2]), float(data.geom_xpos[geom, 2] + half[2]))
        _require(np.isfinite(lo).all() and np.isfinite(hi).all() and all(map(isfinite, z)), "BOX bounds")
        rectangles.append(
            PlanarPushRectangle(
                geom, mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom),
                tuple(map(float, lo)), tuple(map(float, hi)), z,
            )
        )
    _require(
        all(
            abs(a - b) <= ALIGNMENT_TOLERANCE
            for rect in rectangles for a, b in zip(rect.z_interval_world, rectangles[0].z_interval_world)
        ),
        "BOX geoms do not share a Z interval",
    )
    pusher_center = data.geom_xpos[pusher]
    radius = float(model.geom_size[pusher, 0])
    half_length = float(model.geom_size[pusher, 1]) if kind == int(mj.mjtGeom.mjGEOM_CAPSULE) else 0.0
    _require(isfinite(radius) and radius > 0 and isfinite(half_length) and half_length >= 0, "pusher size")
    if kind == int(mj.mjtGeom.mjGEOM_CAPSULE):
        axis = data.geom_xmat[pusher].reshape(3, 3)[:, 2]
        _require(
            np.max(np.abs(axis[:2])) <= ALIGNMENT_TOLERANCE
            and abs(abs(axis[2]) - 1) <= ALIGNMENT_TOLERANCE,
            "capsule pusher is not vertical",
        )
    zlo = max(r.z_interval_world[0] for r in rectangles)
    zhi = min(r.z_interval_world[1] for r in rectangles)
    cz = float(pusher_center[2])
    _require(max(zlo, cz - half_length - radius) < min(zhi, cz + half_length + radius), "no Z volume overlap")
    contact_lo, contact_hi = max(zlo, cz - half_length), min(zhi, cz + half_length)
    _require(contact_lo <= contact_hi, "cap-only Z overlap is unsupported for a full-radius query")
    contact_z = 0.5 * (contact_lo + contact_hi)
    com = _tuple3(data.xipos[body])
    local_edges = _boundary(rectangles)
    _require(
        len(local_edges) * len(fractions) * len(angles) <= int(max_candidates),
        "candidate budget exceeded; no attempts were dropped",
    )

    def world(xy, height):
        return _tuple3(origin + axes[:, 0] * xy[0] + axes[:, 1] * xy[1] + (0, 0, height - origin[2]))

    boundary, candidates = [], []
    for edge_index, (axis, sign, fixed, owners, low, high) in enumerate(local_edges):
        start, end, normal = np.zeros(2), np.zeros(2), np.zeros(2)
        start[axis], end[axis], normal[axis] = fixed, fixed, sign
        start[1 - axis], end[1 - axis] = low, high
        normal_world = _tuple3(axes[:, :2] @ normal)
        boundary.append(
            PlanarPushBoundaryEdge(edge_index, world(start, contact_z), world(end, contact_z), normal_world, owners)
        )
        for fraction in fractions:
            point = start + fraction * (end - start)
            center = point + radius * normal
            distances = tuple(_point_rectangle(center, rect) for rect in rectangles)
            point_world = world(point, contact_z)
            force = -np.asarray(normal_world)
            wrench = tuple(map(float, np.r_[force, np.cross(np.asarray(point_world) - com, force)]))
            _require(all(map(isfinite, wrench)), "nonfinite nominal wrench")
            for angle in angles:
                direction = np.array(
                    (-cos(angle) * normal[0] + sin(angle) * normal[1],
                     -sin(angle) * normal[0] - cos(angle) * normal[1])
                )
                precontact = center - stand_off * direction
                _require(np.isfinite(precontact).all(), "nonfinite precontact center")
                entry = tuple(_segment_rectangle(precontact, center, rect) for rect in rectangles)
                _require(all(map(isfinite, (*distances, *entry))), "nonfinite analytic distance")
                reason = "clear"
                if not float(normal @ direction) < 0:
                    reason = "non-inward-entry"
                elif np.array_equal(precontact, center):
                    reason = "unresolved-entry-displacement"
                elif any(distance < radius - QUERY_TOLERANCE_M for distance in distances):
                    reason = "contact-occluded"
                elif not any(
                    abs(distances[index] - radius) <= QUERY_TOLERANCE_M
                    for index, rect in enumerate(rectangles) if rect.geometry_id in owners
                ):
                    reason = "no-boundary-contact"
                elif any(distance < radius - QUERY_TOLERANCE_M for distance in entry):
                    reason = "entry-occluded"
                candidates.append(
                    NativePlanarPushContactCandidate(
                        len(candidates), edge_index, fraction, angle, owners, point_world,
                        normal_world, world(center, cz), world(precontact, cz),
                        _tuple3(axes[:, :2] @ direction), wrench, distances, entry,
                        reason == "clear", reason,
                    )
                )
    request = json.dumps(
        dict(
            schema="native-planar-push-contacts-v1", q_fp64=configuration.tobytes().hex(),
            object_geom_names=[r.geometry_name for r in rectangles], pusher_geom_name=pusher_geom_name,
            edge_fractions=fractions, approach_angles=angles, stand_off_m=stand_off,
            max_candidates=int(max_candidates), query_tolerance_m=QUERY_TOLERANCE_M,
            alignment_tolerance=ALIGNMENT_TOLERANCE,
        ),
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    binding = sha256(len(binary).to_bytes(8, "little") + binary + request).hexdigest()
    _require(_model_bytes(mj, model) == binary, "native model changed during push query")
    return NativePlanarPushContacts(
        binding, _readonly(configuration), body, com, pusher, radius, _tuple3(origin),
        _readonly(axes), tuple(rectangles), tuple(boundary), tuple(candidates), QUERY_TOLERANCE_M,
    )
