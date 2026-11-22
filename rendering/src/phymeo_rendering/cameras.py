"""Pinhole exposure, independent of the display window or physics environment."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from .scene import FrameKey, Pose, SceneFrame


@dataclass(frozen=True, slots=True)
class Camera:
    """Pose resolved at exposure time; +X right, +Y up and -Z forward.

    Pixel coordinates start at the top-left corner, with first centre (0.5, 0.5).
    The near and far planes are measured along camera forward in metres.
    """

    id: str
    frame_key: FrameKey
    world_pose: Pose
    width: int
    height: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    near_m: float
    far_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TypeError("camera id must be a string")
        if not self.id.strip():
            raise ValueError("camera id must not be blank")
        if not isinstance(self.frame_key, FrameKey):
            raise TypeError("camera frame_key must be a FrameKey")
        if not isinstance(self.world_pose, Pose):
            raise TypeError("camera world_pose must be a Pose")

        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"camera {name} must be an integer")
            if value <= 0:
                raise ValueError(f"camera {name} must be positive")

        for name in ("fx_px", "fy_px", "cx_px", "cy_px", "near_m", "far_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"camera {name} must be a real number")
            try:
                finite = isfinite(value)
            except OverflowError as exc:
                raise ValueError(f"camera {name} must be finite and representable") from exc
            if not finite:
                raise ValueError(f"camera {name} must be finite")

        for name in ("fx_px", "fy_px", "near_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"camera {name} must be positive")
        if self.far_m <= self.near_m:
            raise ValueError("camera far_m must be greater than near_m")


def validate_exposure(frame: SceneFrame, camera: Camera) -> None:
    """Require an exposure bound to this complete source key, without mutation.

    The source frame owns the exposure time. This check does not resolve wrist
    attachments, project points, or verify the scene opened by a render backend.
    """
    if not isinstance(frame, SceneFrame):
        raise TypeError("frame must be a SceneFrame")
    if not isinstance(camera, Camera):
        raise TypeError("camera must be a Camera")
    for name in ("scene_key", "world_id", "epoch", "revision", "tick", "phase"):
        if getattr(frame.key, name) != getattr(camera.frame_key, name):
            raise ValueError(f"exposure frame_key {name} does not match source frame")
