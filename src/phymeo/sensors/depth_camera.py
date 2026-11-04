"""Pinhole depth and shape observations on a supplied collision-world pose."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from ..constraints.collision_shapes import CollisionKey
from ..core.validation import finite_real
from ..dynamics.rotation import quaternion_matrix
from ..geometry.raycast import RayHits, _cast_unit, _frozen
from .robot_frame import RobotSite


@dataclass(frozen=True, eq=False)
class DepthImage:
    """Pixel arrays use top-left origin. A miss has depth +inf, index -1 and zero normal."""

    ray_depth_m: np.ndarray
    forward_depth_m: np.ndarray
    shape_index: np.ndarray
    world_normals: np.ndarray
    keys: tuple[CollisionKey, ...]


def _pose(position_m, orientation_xyzw):
    frame = RobotSite(None, position_m, orientation_xyzw)
    return np.asarray(frame.local_position_m), quaternion_matrix(
        np.asarray([frame.local_orientation_xyzw])
    )[0]


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

    def unproject(self, ray_depth_m, *, position_m=(0, 0, 0), orientation_xyzw=(0, 0, 0, 1)):
        """Recover world coordinates in metres. Missing rays (+inf) produce three NaNs.

        Supply the ray depth and camera pose from the same exposure. Use this
        camera's calibration, including its pixel-centre and image-axis conventions.
        """
        raw = np.asarray(ray_depth_m)
        if raw.shape != (self.height, self.width) or raw.dtype.kind not in "fiu":
            raise ValueError("expected one real ray depth per camera pixel")
        depth = raw.astype(np.float64, copy=False).reshape(-1)
        if np.any(np.isnan(depth)) or np.any(depth < 0):
            raise ValueError("ray depths must be nonnegative metres or +inf")
        origin, rotation = _pose(position_m, orientation_xyzw)
        valid = np.isfinite(depth)
        points = np.full((len(depth), 3), np.nan)
        points[valid] = origin + (self._rays[valid] @ rotation.T) * depth[valid, None]
        return points.reshape(self.height, self.width, 3)

    def capture(
        self,
        model,
        poses,
        *,
        world=0,
        position_m=(0, 0, 0),
        orientation_xyzw=(0, 0, 0, 1),
        operator=None,
    ):
        """Capture at one pose. An explicit ray query operator owns the entire ray calculation."""
        origin, rotation = _pose(position_m, orientation_xyzw)
        rays = self._rays @ rotation.T
        if operator is None:
            hit = _cast_unit(model, poses, origin, rays, world, self.near_m, self.far_m)
        else:
            hit = operator.query(
                model, poses, origin, rays, world=world, near_m=self.near_m, far_m=self.far_m
            )
            count = len(rays)
            if (
                not isinstance(hit, RayHits)
                or hit.keys != tuple(s.key for s in model.shapes)
                or hit.distance_m.shape != (count,)
                or hit.shape_index.shape != (count,)
                or hit.world_normals.shape != (count, 3)
            ):
                raise ValueError(
                    "ray operator returned a different shape registration or image extent"
                )
        shape = (self.height, self.width)
        return DepthImage(
            hit.distance_m.reshape(shape),
            _frozen((hit.distance_m * -self._rays[:, 2]).reshape(shape)),
            hit.shape_index.reshape(shape),
            hit.world_normals.reshape(*shape, 3),
            hit.keys,
        )
