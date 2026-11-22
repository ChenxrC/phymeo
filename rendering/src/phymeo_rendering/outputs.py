"""Consumer-facing, CPU-owned image planes and their exposure metadata."""

from dataclasses import dataclass
from enum import Enum

from .cameras import Camera
from .scene import FrameKey


class Channel(Enum):
    """Tightly packed, top-left raster order; multi-byte samples are little endian."""

    RGB = "srgb8_rgb"
    FORWARD_DEPTH = "forward_depth_m_f32"  # Positive along -Z; +inf on miss.
    INSTANCE_ID = "instance_id_u32"  # 0 is background; positive values are Instance.id.
    VALID = "valid_u8"  # One for a visible hit, zero for a miss.


@dataclass(frozen=True, slots=True)
class ImagePlane:
    channel: Channel
    data: bytes


@dataclass(frozen=True, slots=True)
class Capture:
    """Planes have the exposure camera's dimensions and belong to this source frame."""

    frame_key: FrameKey
    time_s: float
    camera: Camera
    planes: tuple[ImagePlane, ...]
