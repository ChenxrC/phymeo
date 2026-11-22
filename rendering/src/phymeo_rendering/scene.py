"""Static visual scene and read-only, time-stamped instance motion."""

from dataclasses import dataclass
from typing import Literal

from .assets import Material, Mesh


@dataclass(frozen=True, slots=True)
class Pose:
    """Child frame in parent coordinates; metres and a unit XYZW quaternion."""

    position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class Instance:
    """Stable positive segmentation ID, with the visual attached to a named node."""

    id: int
    mesh_id: str
    material_id: str
    node_id: str
    local_pose: Pose = Pose()


@dataclass(frozen=True, slots=True)
class DirectionalLight:
    """World-space unit direction of light travel and linear RGB illuminance."""

    direction: tuple[float, float, float]
    illuminance: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class VisualScene:
    """A static asset and instance manifest. A change to its contents gets a new key."""

    key: str
    meshes: tuple[Mesh, ...]
    materials: tuple[Material, ...]
    instances: tuple[Instance, ...]
    lights: tuple[DirectionalLight, ...] = ()


@dataclass(frozen=True, slots=True)
class FrameKey:
    """World epoch changes on reset; revision identifies the published source state."""

    scene_key: str
    world_id: str
    epoch: int
    revision: int
    tick: int
    phase: Literal["authored", "step_input", "accepted_endpoint"]


@dataclass(frozen=True, slots=True)
class NodePose:
    node_id: str
    world_pose: Pose


@dataclass(frozen=True, slots=True)
class SceneFrame:
    key: FrameKey
    time_s: float
    nodes: tuple[NodePose, ...]
