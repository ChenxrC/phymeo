"""Pinhole exposure, independent of the display window or physics environment."""

from dataclasses import dataclass

from .scene import FrameKey, Pose


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
