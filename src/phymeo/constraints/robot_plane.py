"""Owned link-material-point bindings to static world planes."""

from dataclasses import dataclass

import numpy as np

from .particle_plane import ParticlePlaneContacts, _values


@dataclass(frozen=True, eq=False)
class RobotPlaneContacts:
    domain: str
    indices: np.ndarray
    local_points: np.ndarray
    normals: np.ndarray
    offsets: np.ndarray
    friction: np.ndarray

    def __post_init__(self):
        plane = ParticlePlaneContacts(
            self.domain, self.indices, self.normals, self.offsets, self.friction
        )
        local = _values(self.local_points, "local_points", (len(plane.indices), 3))
        local.flags.writeable = False
        for name in ("indices", "normals", "offsets", "friction"):
            object.__setattr__(self, name, getattr(plane, name))
        object.__setattr__(self, "local_points", local)
