"""C6 HOST contact packets between resident entry and candidate kinematics.

Schema checkpoint only: sampling and finishing are not implemented in this
checkpoint. No integration, device launch or accepted-state publication lives here.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionShape
from ..constraints.robot_contact import RobotContactMaterial
from ..geometry.contact_patch_adapter import ContactSite


@dataclass(frozen=True, eq=False)
class ContactShapeFrame:
    shape: CollisionShape
    world_points_m: np.ndarray
    group_index: int | None
    position_m: np.ndarray
    rotation: np.ndarray
    reference_point_m: np.ndarray
    linear_velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray
    rotation_orthogonality_error: float = field(init=False, default=0.0)


@dataclass(frozen=True, eq=False)
class ResidentContactSite:
    pair: tuple[CollisionKey, CollisionKey]
    geometry: ContactSite
    law: RobotContactMaterial
    first_velocity_m_s: np.ndarray
    second_velocity_m_s: np.ndarray
    first_force_n: np.ndarray
    second_force_n: np.ndarray
    first_reference_point_m: np.ndarray
    second_reference_point_m: np.ndarray
    first_material_displacement_m: np.ndarray | None = None
    second_material_displacement_m: np.ndarray | None = None
    first_application_displacement_m: np.ndarray | None = None
    second_application_displacement_m: np.ndarray | None = None
    application_work_j: float | None = None
    material_work_j: float | None = None


@dataclass(frozen=True, eq=False)
class ResidentStaticReaction:
    reference_point_m: np.ndarray
    force_n: np.ndarray
    torque_nm: np.ndarray


@dataclass(frozen=True, eq=False)
class ContactLoads:
    sites: tuple[ResidentContactSite, ...]
    wrench_fp64: np.ndarray
    static_reactions: Mapping[CollisionKey, ResidentStaticReaction]
    frames: Mapping[CollisionKey, ContactShapeFrame]
    world: int
    finalized: bool = False


class ContactSiteCapacityExceeded(ValueError):
    def __init__(self, count, limit):
        self.count = count
        self.limit = limit
        super().__init__(f"contact site count {count} exceeds capacity {limit}")


def sample_tree_contacts(*, world, frames, materials_by_pair, candidates, group_count, site_capacity):
    raise NotImplementedError("C6 HOST contact sampling is pending implementation")


def finish_contact_loads(loads, candidate_frames):
    raise NotImplementedError("C6 actual stored-frame transport is pending implementation")
