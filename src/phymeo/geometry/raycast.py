"""Metric ray queries against the registered collision geometry at one physical instant."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionModel, CollisionMotion
from ..core.validation import finite_real
from ..dynamics.rotation import quaternion_matrix
from ._convex_hull import HullShape


@dataclass(frozen=True, eq=False)
class RayHits:
    """First hit per ray in metres. Misses use +inf, shape index −1 and a zero normal."""

    distance_m: np.ndarray
    shape_index: np.ndarray
    world_normals: np.ndarray
    keys: tuple[CollisionKey, ...]


def _terrain_triangles(points, shape):
    height, width = shape
    for y in range(height - 1):
        for x in range(width - 1):
            start = y * width + x
            diagonal = start + width + 1
            yield points[[start, start + 1, diagonal]]
            yield points[[start, diagonal, start + width]]


def _surfaces(model, poses, world):
    if not isinstance(model, CollisionModel) or not isinstance(poses, Mapping):
        raise ValueError("expected a collision model and a mapping of current poses")
    if isinstance(world, (bool, np.bool_)) or not isinstance(world, Integral) or world < 0:
        raise ValueError("expected a nonnegative world index")
    for index, shape in enumerate(model.shapes):
        if shape.key.world != world:
            continue
        motion = poses.get(shape.key)
        if (
            not isinstance(motion, CollisionMotion)
            or motion.moving
            or len(motion.start) != len(shape.vertex_ids)
        ):
            raise ValueError("each observed shape needs one complete current pose")
        p = motion.start
        if shape.kind in ("sphere", "rounded_particle", "capsule") and shape.radius > 0:
            yield index, "round", p, shape.radius
        elif shape.kind == "plane":
            normal = np.asarray(shape.normal, dtype=float)
            scaled = normal / np.max(np.abs(normal))
            yield index, "plane", p, scaled / np.hypot.reduce(scaled)
        elif shape.curved is not None:
            frame = quaternion_matrix(np.asarray([shape.curved.orientation_xyzw]))[0]
            if motion.body_orientation_xyzw is not None:
                frame = quaternion_matrix(np.asarray([motion.body_orientation_xyzw]))[0] @ frame
            yield index, shape.kind, p[0], (frame, np.asarray(shape.curved.dimensions_m))
        elif (
            shape.kind in ("triangle", "mesh", "tetrahedron", "hull", "heightfield")
            and shape.radius == 0
        ):
            if shape.kind == "triangle":
                triangles = [p]
            elif shape.kind == "heightfield":
                if not np.array_equal(p, shape.heightfield.world_vertices()):
                    raise ValueError("terrain must match its registered world vertices")
                triangles = _terrain_triangles(p, np.shape(shape.heightfield.heights_yx))
            elif shape.kind == "tetrahedron":
                triangles = [p[list(face)] for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))]
            else:
                offsets = dict(zip(shape.vertex_ids, range(len(p)), strict=True))
                faces = (
                    shape.triangles
                    if shape.kind == "mesh"
                    else (face.ids for face in HullShape(shape, motion).faces)
                )
                triangles = (
                    p[[offsets[i] for i in (face[0], face[j], face[j + 1])]]
                    for face in faces
                    for j in range(1, len(face) - 1)
                )
            for tri in triangles:
                u, v = tri[1] - tri[0], tri[2] - tri[0]
                normal = np.cross(u, v)
                length = np.hypot.reduce(normal)
                if not np.isfinite(length) or not length:
                    raise ValueError("degenerate triangle in camera pose")
                yield index, "triangle", tri, normal / length
        else:
            raise ValueError(f"unsupported depth geometry: {shape.kind} (radius {shape.radius})")


def _store(t, mask, normals, index, tile, near, far):
    depth, labels, output = tile
    hit = mask & np.isfinite(t) & (t >= near) & (t <= far) & (t < depth)
    if np.any(hit):
        depth[hit] = t[hit]
        labels[hit] = index
        output[hit] = normals if normals.ndim == 1 else normals[hit]


def _ball(center, radius, origin, directions, index, tile, near, far, cap_axis=None):
    rel = center - origin
    along = np.sum(directions * rel, axis=1)
    off = rel - along[:, None] * directions
    gap = radius**2 - np.einsum("ij,ij->i", off, off)
    span = np.sqrt(np.maximum(0, gap))
    for t in (along - span, along + span):
        arm = t[:, None] * directions - rel
        mask = gap >= 0
        if cap_axis is not None:
            mask = mask & (arm @ cap_axis >= 0)
        _store(t, mask, arm / radius, index, tile, near, far)


def _round(p, radius, origin, directions, index, tile, near, far):
    if len(p) == 1 or np.array_equal(p[0], p[1]):
        _ball(p[0], radius, origin, directions, index, tile, near, far)
        return
    axis = p[1] - p[0]
    length = np.hypot.reduce(axis)
    axis = axis / length
    _ball(p[0], radius, origin, directions, index, tile, near, far, -axis)
    _ball(p[1], radius, origin, directions, index, tile, near, far, axis)
    rel = origin - p[0]
    z0 = rel @ axis
    axial = directions @ axis
    radial = rel - z0[..., None] * axis
    d = directions - axial[:, None] * axis
    a = np.einsum("ij,ij->i", d, d)
    mid = np.divide(-np.sum(d * radial, axis=1), a, out=np.zeros_like(a), where=a > 0)
    closest = radial + mid[:, None] * d
    gap = radius**2 - np.einsum("ij,ij->i", closest, closest)
    span = np.sqrt(np.divide(np.maximum(0, gap), a, out=np.zeros_like(a), where=a > 0))
    for t in (mid - span, mid + span):
        z = z0 + t * axial
        mask = (a > 0) & (gap >= 0) & (z >= 0) & (z <= length)
        _store(t, mask, (radial + t[:, None] * d) / radius, index, tile, near, far)


def _flat(kind, p, normal, origin, directions, index, tile, near, far):
    denom = directions @ normal
    t = np.divide((p[0] - origin) @ normal, denom, out=np.zeros_like(denom), where=denom != 0)
    mask = denom != 0
    if kind == "triangle":
        point = origin + t[:, None] * directions
        for a, b in ((0, 1), (1, 2), (2, 0)):
            mask &= np.cross(p[b] - p[a], point - p[a]) @ normal >= 0
    # Two-sided planes and triangles publish a camera-facing geometric normal.
    n = np.where((denom > 0)[:, None], -normal, normal)
    _store(t, mask, n, index, tile, near, far)


def _roots(a, b, c):
    """Return both roots of a*t^2 + 2*b*t + c, retaining linear and tangent hits."""
    disc = b * b - a * c
    ok = (a != 0) & (disc >= 0)
    q = -b - np.copysign(np.sqrt(np.maximum(0.0, disc)), b)
    first = np.divide(q, a, out=np.zeros_like(b), where=ok)
    second = np.divide(c, q, out=first.copy(), where=ok & (q != 0))
    linear = (a == 0) & (b != 0)
    np.divide(-c, 2 * b, out=first, where=linear)
    return ((first, ok | linear), (second, ok))


def _curved(kind, center, parameters, origin, rays, index, tile, near, far):
    frame, size = parameters
    o = (origin - center) @ frame
    d = rays @ frame
    if kind == "ellipsoid":
        u, v = o / size, d / size
        a = np.sum(v * v, axis=1)
        b = np.sum(u * v, axis=1)
        c = np.sum(u * u, axis=-1) - 1
    else:
        radius, half = size
        u, v = o[..., :2], d[:, :2]
        a = np.sum(v * v, axis=1)
        b = np.sum(u * v, axis=1)
        c = np.sum(u * u, axis=-1)
        if kind == "cylinder":
            c = c - radius**2
        else:
            slope2 = (radius / (2 * half)) ** 2
            end = half - o[..., 2]
            a -= slope2 * d[:, 2] ** 2
            b += slope2 * end * d[:, 2]
            c = c - slope2 * end**2
    candidates = list(_roots(a, b, c))
    if kind != "ellipsoid":
        # A ray can lie on a whole cylinder generator or cone generator.
        # Intersect that interval with the true finite axial extent and near plane.
        coincident = (a == 0) & (b == 0) & (c == 0) & (d[:, 2] != 0)
        low = np.divide(-half - o[..., 2], d[:, 2], out=np.zeros(len(d)), where=coincident)
        high = np.divide(half - o[..., 2], d[:, 2], out=np.zeros(len(d)), where=coincident)
        first = np.maximum(near, np.minimum(low, high))
        candidates.append((first, coincident & (first <= np.maximum(low, high))))
    for t, valid in candidates:
        p = o + t[:, None] * d
        if kind == "ellipsoid":
            normal = p / size**2
        else:
            valid = valid & (p[:, 2] >= -half) & (p[:, 2] <= half)
            normal = p.copy()
            normal[:, 2] = 0 if kind == "cylinder" else slope2 * (half - p[:, 2])
        length = np.hypot.reduce(normal, axis=1)[:, None]
        normal = np.divide(normal, length, out=np.zeros_like(normal), where=length != 0)
        _store(t, valid, normal @ frame.T, index, tile, near, far)
    if kind in ("cylinder", "cone"):
        for sign in (-1, 1) if kind == "cylinder" else (-1,):
            nz = d[:, 2] != 0
            t = np.divide(sign * half - o[..., 2], d[:, 2], out=np.zeros(len(d)), where=nz)
            xy = o[..., :2] + t[:, None] * d[:, :2]
            _store(
                t,
                nz & (np.sum(xy * xy, axis=1) <= radius**2),
                sign * frame[:, 2],
                index,
                tile,
                near,
                far,
            )


def _frozen(a):
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _cast_unit(model, poses, origin, rays, world, near, far):
    surfaces = list(_surfaces(model, poses, world))
    n = len(rays)
    depth = np.full(n, np.inf)
    labels = np.full(n, -1, dtype=np.int32)
    normals = np.zeros((n, 3))
    for start in range(0, n, 8192):
        section = slice(start, start + 8192)
        eye = origin if origin.ndim == 1 else origin[section]
        dirs = rays[section]
        tile = depth[section], labels[section], normals[section]
        for index, kind, p, param in surfaces:
            if kind == "round":
                _round(p, param, eye, dirs, index, tile, near, far)
            elif kind in ("plane", "triangle"):
                _flat(kind, p, param, eye, dirs, index, tile, near, far)
            else:
                _curved(kind, p, param, eye, dirs, index, tile, near, far)
    return RayHits(
        _frozen(depth), _frozen(labels), _frozen(normals), tuple(s.key for s in model.shapes)
    )


def _ray_inputs(origins_m, directions, near_m, far_m):
    origins = np.asarray(origins_m)
    rays = np.asarray(directions)
    if (
        rays.ndim != 2
        or rays.shape[1] != 3
        or len(rays) > 2**20
        or origins.shape not in ((3,), rays.shape)
        or origins.dtype.kind not in "fiu"
        or rays.dtype.kind not in "fiu"
    ):
        raise ValueError("expected bounded real 3D ray origins and directions")
    origins, rays = np.asarray(origins, dtype=float), np.asarray(rays, dtype=float)
    if not np.isfinite(origins).all() or not np.isfinite(rays).all():
        raise ValueError("ray origins and directions must be finite")
    scale = np.max(abs(rays), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("each ray needs a nonzero direction")
    rays = rays / scale
    rays /= np.hypot.reduce(rays, axis=1)[:, None]
    near = finite_real(near_m, "near_m")
    far = np.inf if isinstance(far_m, Real) and far_m == np.inf else finite_real(far_m, "far_m")
    if not 0 <= near < far:
        raise ValueError("expected a nonnegative metric clipping interval")
    return origins, rays, near, far


def _ray_batch(model, poses, origins, directions, world, near, far):
    """Flatten physical beams and form a stable routing plan without visiting unselected poses."""
    if not isinstance(model, CollisionModel) or not isinstance(poses, Mapping):
        raise ValueError("expected a collision model and a mapping of current poses")
    d, o = np.asarray(directions), np.asarray(origins)
    if (
        d.ndim not in (2, 3)
        or d.shape[-1] != 3
        or d.size // 3 > 2**20
        or o.shape not in ((3,), d.shape)
    ):
        raise ValueError("expected rows or a matrix of 3D rays and matching origins")
    shape = d.shape[:-1]
    eyes, rays, near, far = _ray_inputs(
        o if o.ndim == 1 else o.reshape(-1, 3), d.reshape(-1, 3), near, far
    )
    if isinstance(world, Integral) and not isinstance(world, (bool, np.bool_)) and world >= 0:
        return shape, eyes, rays, near, far, None
    if isinstance(world, np.ndarray):
        address = np.asarray(world)
        if address.dtype.kind not in "iu":
            raise ValueError("ray world addresses must be nonnegative integers")
    else:
        # Validate elements before array promotion can turn a mixed Boolean/integer list into IDs.
        address = np.asarray(world, dtype=object)
        if any(
            not isinstance(v, Integral) or isinstance(v, (bool, np.bool_)) for v in address.flat
        ):
            raise ValueError("ray world addresses must be nonnegative integers")
    if address.shape != shape or np.any(address < 0):
        raise ValueError("ray world addresses must match exactly the ray layout and be nonnegative")
    flat = address.ravel()
    order = np.argsort(flat, kind="stable")
    sorted_keys = flat[order]
    groups = []
    for key in sorted({s.key.world for s in model.shapes}):
        if flat.dtype.kind in "iu" and key > int(np.iinfo(flat.dtype).max):
            continue
        left = np.searchsorted(sorted_keys, key, side="left")
        right = np.searchsorted(sorted_keys, key, side="right")
        if right > left:
            groups.append((key, order[left:right]))
    return shape, eyes, rays, near, far, groups


def _ray_shape(hit, shape):
    if hit.distance_m.shape == shape:
        return hit
    return RayHits(
        hit.distance_m.reshape(shape),
        hit.shape_index.reshape(shape),
        hit.world_normals.reshape(*shape, 3),
        hit.keys,
    )


def raycast_scene(model, poses, origins_m, directions, *, world=0, near_m=0.0, far_m=np.inf):
    """Query stationary geometry with an arbitrary beam pattern and metric clipping.

    Supply a scalar world or one integer world per ray. Directions may have shape (N,3) or
    (H,W,3); world IDs and results preserve the leading shape. Origins are either shared (3,)
    or match the complete direction array. Directions need not be unit vectors.
    Finite flat faces are two-sided, with their normals facing the incoming ray.
    A cone apex has no unique normal and returns a zero vector. On a distance tie
    the first stable registered shape is retained. Equal rim hits on one analytic
    solid use its side surface before its cap.
    """
    shape, origins, rays, near, far, groups = _ray_batch(
        model, poses, origins_m, directions, world, near_m, far_m
    )
    with np.errstate(over="raise", invalid="raise"):
        if groups is None:
            return _ray_shape(_cast_unit(model, poses, origins, rays, world, near, far), shape)
        depth = np.full(len(rays), np.inf)
        labels = np.full(len(rays), -1, dtype=np.int32)
        normals = np.zeros((len(rays), 3))
        for key, slots in groups:
            eye = origins if origins.ndim == 1 else origins[slots]
            hit = _cast_unit(model, poses, eye, rays[slots], key, near, far)
            depth[slots], labels[slots], normals[slots] = (
                hit.distance_m,
                hit.shape_index,
                hit.world_normals,
            )
        result = RayHits(
            _frozen(depth), _frozen(labels), _frozen(normals), tuple(s.key for s in model.shapes)
        )
        return _ray_shape(result, shape)
