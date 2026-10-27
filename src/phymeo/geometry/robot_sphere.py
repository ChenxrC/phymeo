"""Ordered, owned sphere geometry attached to robot links."""

from dataclasses import dataclass

import numpy as np

from ..core.robot import robot_array


def attachment_indices(value, name):
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind not in "iu":
        raise TypeError(f"{name} must be rank-one integer indices")
    if np.any(value < 0) or np.any(value > np.iinfo(np.int64).max):
        raise ValueError(f"{name} outside supported index range")
    owned = np.array(value, dtype=np.int64, copy=True)
    owned.flags.writeable = False
    return owned


@dataclass(frozen=True, eq=False)
class RobotSphereAttachments:
    link_indices: np.ndarray
    local_centers: np.ndarray
    radii: np.ndarray

    def __post_init__(self):
        indices = attachment_indices(self.link_indices, "link_indices")
        centers = robot_array(self.local_centers, "local_centers", ndim=2, columns=3)
        radii = robot_array(self.radii, "radii", ndim=1)
        if len(indices) != len(centers) or len(indices) != len(radii):
            raise ValueError("each sphere requires a link, local center and radius")
        if np.any(radii <= 0):
            raise ValueError("sphere radii must be positive")
        centers.flags.writeable = radii.flags.writeable = False
        object.__setattr__(self, "link_indices", indices)
        object.__setattr__(self, "local_centers", centers)
        object.__setattr__(self, "radii", radii)
