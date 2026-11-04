"""Preserve named authored URDF frames while dynamics uses rigid groups in principal axes."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..constraints.collision_shapes import CollisionMotion
from ..dynamics.robot_kinematics import forward_kinematics
from ..dynamics.rotation import matrix_quaternion, quaternion_matrix
from ..sensors.robot_frame import RobotSite
from ._robot_asset import RobotAsset


def _rotation(q):
    return quaternion_matrix(np.array([q], dtype=float))[0]


@dataclass(frozen=True, eq=False, kw_only=True)
class FoldedURDFAsset(RobotAsset):
    """link_names and joint_names follow moving DOFs; sites includes every authored link.

    A collider's local_vertices are in its named URDF link. Use authored_to_model_points
    to bind original local material to a point port or Jacobian. A fixed root site
    maps directly into world coordinates.
    """

    sites: Mapping[str, RobotSite]

    def authored_to_model_points(self, name, points_m):
        """Map named authored local points to their moving group, or world for a fixed site."""
        site = self.sites[name]
        points = np.asarray(points_m, dtype=float)
        if points.ndim not in (1, 2) or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("authored points must be finite 3-vectors")
        return np.asarray(site.local_position_m) + points @ _rotation(site.local_orientation_xyzw).T

    def _pose(self, name, kinematics):
        site = self.sites[name]
        p, r = np.asarray(site.local_position_m), _rotation(site.local_orientation_xyzw)
        i = site.link_index
        if i is None:
            return p, r
        return kinematics.position[i] + kinematics.rotation[i] @ p, kinematics.rotation[i] @ r

    def authored_link_pose(self, name, q):
        """Return the original named link origin and body-to-world rotation."""
        return self._pose(name, forward_kinematics(self.model, q))

    def collision_motion(self, q, q_end=None):
        fk = forward_kinematics(self.model, q)
        poses = {name: self._pose(name, fk) for name in {c.link_name for c in self.collisions}}
        paths = {}
        for collider in self.collisions:
            p, r = poses[collider.link_name]
            points = p + np.asarray(collider.local_vertices) @ r.T
            orientation = matrix_quaternion(r) if collider.shape.curved is not None else None
            paths[collider.shape.key] = CollisionMotion(points, body_orientation_xyzw=orientation)
        if q_end is not None:
            from ._robot_joint_motion import joint_interval_motion

            return joint_interval_motion(self, q, q_end, fk, paths)
        return paths
