"""Static visual scene and read-only, time-stamped instance motion.

These contracts validate one value or scene/frame pair without retaining history.
They do not read assets, initialize a renderer, or advance a physical source.
"""

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

from .assets import Material, Mesh

_UNIT_TOLERANCE = 1e-6


def _name(value, field):
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} must not be blank")


def _real(value, field):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number, not bool")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{field} must be finite and representable")


def _integer(value, field, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer, not bool")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{field} outside allowed range: {value!r}")


def _vector(value, size, field, *, unit=False):
    if not isinstance(value, tuple):
        raise TypeError(f"{field} must be an immutable tuple")
    if len(value) != size:
        raise ValueError(f"{field} must have {size} components")
    for index, item in enumerate(value):
        _real(item, f"{field}[{index}]")
    if unit and abs(math.hypot(*value) - 1.0) > _UNIT_TOLERANCE:
        raise ValueError(f"{field} must have unit norm within {_UNIT_TOLERANCE}")


def _members(value, expected_type, field):
    if not isinstance(value, tuple):
        raise TypeError(f"{field} must be an immutable tuple")
    for index, item in enumerate(value):
        if not isinstance(item, expected_type):
            raise TypeError(f"{field}[{index}] must be {expected_type.__name__}")


def _unique(items, attribute, field):
    identities = set()
    for item in items:
        identity = getattr(item, attribute)
        if identity in identities:
            raise ValueError(f"duplicate {field}: {identity!r}")
        identities.add(identity)
    return identities


def _static_references(scene):
    for mesh in scene.meshes:
        _name(mesh.id, "mesh.id")
    for material in scene.materials:
        _name(material.id, "material.id")
    meshes = _unique(scene.meshes, "id", "mesh.id")
    materials = _unique(scene.materials, "id", "material.id")
    _unique(scene.instances, "id", "instance.id")
    for instance in scene.instances:
        if instance.mesh_id not in meshes:
            raise ValueError(
                f"instance {instance.id} references missing mesh_id {instance.mesh_id!r}"
            )
        if instance.material_id not in materials:
            raise ValueError(
                f"instance {instance.id} references missing material_id {instance.material_id!r}"
            )


@dataclass(frozen=True, slots=True)
class Pose:
    """Child in parent coordinates: finite metres and a unit XYZW quaternion.

    Norm error up to 1e-6 is admitted. Construction preserves supplied values,
    including quaternion sign; it does not silently normalize source state.
    """

    position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def __post_init__(self):
        _vector(self.position_m, 3, "position_m")
        _vector(self.orientation_xyzw, 4, "orientation_xyzw", unit=True)


@dataclass(frozen=True, slots=True)
class Instance:
    """Stable positive uint32 segmentation ID, attached to a named node."""

    id: int
    mesh_id: str
    material_id: str
    node_id: str
    local_pose: Pose = Pose()

    def __post_init__(self):
        _integer(self.id, "instance.id", 1, 2**32 - 1)
        for field in ("mesh_id", "material_id", "node_id"):
            _name(getattr(self, field), field)
        if not isinstance(self.local_pose, Pose):
            raise TypeError("local_pose must be Pose")


@dataclass(frozen=True, slots=True)
class DirectionalLight:
    """World-space unit direction of light travel and linear RGB illuminance."""

    direction: tuple[float, float, float]
    illuminance: tuple[float, float, float]

    def __post_init__(self):
        _vector(self.direction, 3, "direction", unit=True)
        _vector(self.illuminance, 3, "illuminance")
        if any(value < 0 for value in self.illuminance):
            raise ValueError("illuminance must be nonnegative")


@dataclass(frozen=True, slots=True)
class VisualScene:
    """Static asset and instance manifest; changed contents require a new key.

    Constructors validate references within this manifest. Detecting reuse of an
    uploaded key for different contents belongs to the rendering session.
    """

    key: str
    meshes: tuple[Mesh, ...]
    materials: tuple[Material, ...]
    instances: tuple[Instance, ...]
    lights: tuple[DirectionalLight, ...] = ()

    def __post_init__(self):
        _name(self.key, "scene.key")
        _members(self.meshes, Mesh, "meshes")
        _members(self.materials, Material, "materials")
        _members(self.instances, Instance, "instances")
        _members(self.lights, DirectionalLight, "lights")
        _static_references(self)


@dataclass(frozen=True, slots=True)
class FrameKey:
    """World epoch changes on reset; revision identifies the published source state."""

    scene_key: str
    world_id: str
    epoch: int
    revision: int
    tick: int
    phase: Literal["authored", "step_input", "accepted_endpoint"]

    def __post_init__(self):
        _name(self.scene_key, "scene_key")
        _name(self.world_id, "world_id")
        for field in ("epoch", "revision", "tick"):
            _integer(getattr(self, field), field)
        if not isinstance(self.phase, str):
            raise TypeError("phase must be a string")
        if self.phase not in ("authored", "step_input", "accepted_endpoint"):
            raise ValueError(f"invalid phase: {self.phase!r}")


@dataclass(frozen=True, slots=True)
class NodePose:
    node_id: str
    world_pose: Pose

    def __post_init__(self):
        _name(self.node_id, "node_id")
        if not isinstance(self.world_pose, Pose):
            raise TypeError("world_pose must be Pose")


@dataclass(frozen=True, slots=True)
class SceneFrame:
    key: FrameKey
    time_s: float
    nodes: tuple[NodePose, ...]

    def __post_init__(self):
        if not isinstance(self.key, FrameKey):
            raise TypeError("key must be FrameKey")
        _real(self.time_s, "time_s")
        if self.time_s < 0:
            raise ValueError("time_s must be nonnegative")
        _members(self.nodes, NodePose, "nodes")
        _unique(self.nodes, "node_id", "node_id")


def validate_scene_frame(scene: VisualScene, frame: SceneFrame) -> None:
    """Check static references, scene identity and required node coverage.

    Extra nodes, repeat exposures and replay in arbitrary time order are legal.
    The function neither changes input values nor caches previous frame keys.
    """
    if not isinstance(scene, VisualScene):
        raise TypeError("scene must be VisualScene")
    if not isinstance(frame, SceneFrame):
        raise TypeError("frame must be SceneFrame")
    _static_references(scene)
    if frame.key.scene_key != scene.key:
        raise ValueError(
            f"frame scene_key {frame.key.scene_key!r} differs from scene.key {scene.key!r}"
        )
    nodes = _unique(frame.nodes, "node_id", "node_id")
    for instance in scene.instances:
        if instance.node_id not in nodes:
            raise ValueError(
                f"instance {instance.id} references missing node_id {instance.node_id!r}"
            )


def _unit_rotation(pose):
    norm = math.hypot(*pose.orientation_xyzw)
    return tuple(component / norm for component in pose.orientation_xyzw)


def compose_pose(parent: Pose, local: Pose) -> Pose:
    """Return parent * local, treating admitted quaternions as unit rotations.

    Only calculation copies and the newly composed rotation are normalized. The
    result is a derived visual pose, not a verbatim physical source quaternion.
    """
    if not isinstance(parent, Pose) or not isinstance(local, Pose):
        raise TypeError("parent and local must be Pose")
    x, y, z, w = _unit_rotation(parent)
    a, b, c, d = _unit_rotation(local)
    rotation = (
        w * a + x * d + y * c - z * b,
        w * b - x * c + y * d + z * a,
        w * c + x * b - y * a + z * d,
        w * d - x * a - y * b - z * c,
    )
    norm = math.hypot(*rotation)
    rotation = tuple(component / norm for component in rotation)
    rows = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    try:
        position = tuple(
            math.fsum((offset, *(r * p for r, p in zip(row, local.position_m, strict=True))))
            for offset, row in zip(parent.position_m, rows, strict=True)
        )
    except OverflowError as error:
        raise ValueError("composed position_m is not finite and representable") from error
    return Pose(position, rotation)
