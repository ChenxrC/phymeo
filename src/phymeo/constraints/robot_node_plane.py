"""Initial-contact bindings between free nodes and robot-local material planes."""

from dataclasses import dataclass

import numpy as np

from ..geometry.robot_sphere import attachment_indices
from .robot_plane import RobotPlaneContacts


@dataclass(frozen=True, eq=False)
class RobotNodePlaneContacts:
    node_domain: str
    node_indices: np.ndarray
    robot_indices: np.ndarray
    local_points: np.ndarray
    local_normals: np.ndarray
    friction: np.ndarray

    def __post_init__(self):
        checked = RobotPlaneContacts(
            self.node_domain,
            self.node_indices,
            self.local_points,
            self.local_normals,
            np.zeros(len(self.node_indices)),
            self.friction,
        )
        robot = attachment_indices(self.robot_indices, "robot_indices")
        if robot.shape != checked.indices.shape:
            raise ValueError("node and robot contact rows must match")
        for name, value in (
            ("node_indices", checked.indices),
            ("robot_indices", robot),
            ("local_points", checked.local_points),
            ("local_normals", checked.normals),
            ("friction", checked.friction),
        ):
            object.__setattr__(self, name, value)
