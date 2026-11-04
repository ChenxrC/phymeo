"""Pinhole depth and shape observations on a supplied collision-world pose."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionModel, CollisionMotion
from ..core.validation import finite_real
from ..dynamics.rotation import quaternion_matrix
from ..geometry._convex_hull import HullShape
from .robot_frame import RobotSite


@dataclass(frozen=True, eq=False)
class DepthImage:
    """Pixel arrays use top-left origin. A miss has depth +inf, index -1 and zero normal."""

    ray_depth_m: np.ndarray
    forward_depth_m: np.ndarray
    shape_index: np.ndarray
    world_normals: np.ndarray
    keys: tuple[CollisionKey, ...]


def _surfaces(model, poses, world):
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
            yield index, "plane", p, np.asarray(shape.normal)
        elif shape.kind in ("triangle", "mesh", "tetrahedron", "hull") and shape.radius == 0:
            if shape.kind == "triangle":
                triangles = [p]
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
    along = directions @ rel
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
    radial = rel - z0 * axis
    d = directions - axial[:, None] * axis
    a = np.einsum("ij,ij->i", d, d)
    mid = np.divide(-(d @ radial), a, out=np.zeros_like(a), where=a > 0)
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


class DepthCamera:
    """Capture the first closed-surface ray hit in [near_m, far_m].

    Camera coordinates have +X right, +Y up and -Z forward. Depth limits are
    distances along each unit ray. Image indexing is [y,x], and shape_index
    maps to the original CollisionModel's key tuple, including across worlds.
    Equal distances retain the first stable collision key. Each observation
    reads the supplied pose; it neither advances nor caches simulation state.
    """

    def __init__(self, width, height, vertical_fov_rad, *, near_m=0.0, far_m=np.inf):
        if (
            any(
                isinstance(v, (bool, np.bool_))
                or not isinstance(v, Integral)
                or not 1 <= v <= 2**20
                for v in (width, height)
            )
            or int(width) * int(height) > 2**20
        ):
            raise ValueError("expected positive image dimensions up to one megapixel")
        fov = finite_real(vertical_fov_rad, "vertical_fov_rad")
        near = finite_real(near_m, "near_m")
        far = np.inf if isinstance(far_m, Real) and far_m == np.inf else finite_real(far_m, "far_m")
        if not (0 < fov < np.pi and 0 <= near < far):
            raise ValueError("invalid pinhole field of view or ray clipping interval")
        self.width, self.height = int(width), int(height)
        self.near_m, self.far_m = near, far
        pix = 2 * np.tan(fov / 2) / self.height
        x, y = np.meshgrid(
            (np.arange(width) + 0.5 - width / 2) * pix, (height / 2 - np.arange(height) - 0.5) * pix
        )
        rays = np.stack((x, y, -np.ones_like(x)), axis=-1).reshape(-1, 3)
        self._rays = rays / np.linalg.norm(rays, axis=1)[:, None]

    def capture(
        self, model, poses, *, world=0, position_m=(0, 0, 0), orientation_xyzw=(0, 0, 0, 1)
    ):
        if (
            not isinstance(model, CollisionModel)
            or isinstance(world, (bool, np.bool_))
            or not isinstance(world, Integral)
            or world < 0
        ):
            raise ValueError("expected a collision model and nonnegative world index")
        frame = RobotSite(None, position_m, orientation_xyzw)
        origin = np.asarray(frame.local_position_m)
        rotation = quaternion_matrix(np.asarray([frame.local_orientation_xyzw]))[0]
        surfaces = list(_surfaces(model, poses, world))
        n = len(self._rays)
        depth = np.full(n, np.inf)
        labels = np.full(n, -1, dtype=np.int32)
        normals = np.zeros((n, 3))
        for start in range(0, n, 8192):
            section = slice(start, start + 8192)
            rays = self._rays[section] @ rotation.T
            tile = depth[section], labels[section], normals[section]
            for i, kind, p, param in surfaces:
                if kind == "round":
                    _round(p, param, origin, rays, i, tile, self.near_m, self.far_m)
                else:
                    _flat(kind, p, param, origin, rays, i, tile, self.near_m, self.far_m)
        hw = (self.height, self.width)
        images = (
            depth.reshape(hw),
            (depth * -self._rays[:, 2]).reshape(hw),
            labels.reshape(hw),
            normals.reshape(*hw, 3),
        )
        frozen = [
            np.frombuffer(image.tobytes(), dtype=image.dtype).reshape(image.shape)
            for image in images
        ]
        return DepthImage(*frozen, tuple(shape.key for shape in model.shapes))
