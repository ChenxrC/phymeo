"""A canonical zero-thickness finite robot-local convex contact patch."""

from dataclasses import dataclass

import numpy as np

from .robot_finite_patch_collider import RobotFinitePatchCollider


@dataclass(frozen=True, eq=False)
class RobotContactPatch:
    key: str
    link_index: int
    local_vertices: np.ndarray

    def __post_init__(self):
        canonical = RobotFinitePatchCollider(
            self.key, self.link_index, self.local_vertices, radius_m=1.0
        )
        object.__setattr__(self, "link_index", canonical.link_index)
        object.__setattr__(self, "local_vertices", canonical.local_vertices)
