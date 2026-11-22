"""Consumer-facing, CPU-owned image planes and their exposure metadata."""

from dataclasses import dataclass
from enum import Enum
from math import inf, isfinite
from numbers import Real
from struct import iter_unpack

from .cameras import Camera
from .scene import FrameKey


class Channel(Enum):
    """Tightly packed, top-left raster order; multi-byte samples are little endian."""

    RGB = "srgb8_rgb"
    FORWARD_DEPTH = "forward_depth_m_f32"  # Positive along -Z; +inf on miss.
    INSTANCE_ID = "instance_id_u32"  # 0 is background; positive values are Instance.id.
    VALID = "valid_u8"  # One for a visible hit, zero for a miss.


_SAMPLE_BYTES = {
    Channel.RGB: 3,
    Channel.FORWARD_DEPTH: 4,
    Channel.INSTANCE_ID: 4,
    Channel.VALID: 1,
}


@dataclass(frozen=True, slots=True)
class ImagePlane:
    """Owned immutable bytes; Capture validates samples against camera dimensions."""

    channel: Channel
    data: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.channel, Channel):
            raise TypeError("image plane channel must be a Channel")
        if not isinstance(self.data, bytes):
            raise TypeError("image plane data must be immutable bytes")


def _visible_samples(plane: ImagePlane):
    """Validate one sensor stream and yield hit flags without expanding the image."""
    if plane.channel is Channel.FORWARD_DEPTH:
        for index, (depth,) in enumerate(iter_unpack("<f", plane.data)):
            if not depth > 0:
                raise ValueError(
                    f"{plane.channel.value} pixel {index} must be positive or +inf"
                )
            yield depth != inf
    elif plane.channel is Channel.INSTANCE_ID:
        for (identity,) in iter_unpack("<I", plane.data):
            yield identity != 0
    elif plane.channel is Channel.VALID:
        for index, valid in enumerate(plane.data):
            if valid not in (0, 1):
                raise ValueError(f"{plane.channel.value} pixel {index} must be 0 or 1")
            yield valid == 1


@dataclass(frozen=True, slots=True)
class Capture:
    """Validated planes from one exposure, with independent immutable host storage.

    Validation preserves plane order and encoded bytes. Scene membership of
    positive instance IDs and equality to a particular request belong to the
    rendering session, which owns that context.
    """

    frame_key: FrameKey
    time_s: float
    camera: Camera
    planes: tuple[ImagePlane, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frame_key, FrameKey):
            raise TypeError("capture frame_key must be a FrameKey")
        if not isinstance(self.camera, Camera):
            raise TypeError("capture camera must be a Camera")
        if self.frame_key != self.camera.frame_key:
            raise ValueError("capture frame_key does not match camera.frame_key")
        if isinstance(self.time_s, bool) or not isinstance(self.time_s, Real):
            raise TypeError("capture time_s must be a real number")
        try:
            finite = isfinite(self.time_s)
        except OverflowError as exc:
            raise ValueError("capture time_s must be finite and representable") from exc
        if not finite or self.time_s < 0:
            raise ValueError("capture time_s must be finite and nonnegative")
        if not isinstance(self.planes, tuple):
            raise TypeError("capture planes must be an immutable tuple")
        if not self.planes:
            raise ValueError("capture planes must not be empty")

        seen = set()
        pixels = self.camera.width * self.camera.height
        for index, plane in enumerate(self.planes):
            if not isinstance(plane, ImagePlane):
                raise TypeError(f"capture planes[{index}] must be an ImagePlane")
            if plane.channel in seen:
                raise ValueError(f"capture planes contain duplicate channel {plane.channel.value}")
            seen.add(plane.channel)
            expected = pixels * _SAMPLE_BYTES[plane.channel]
            if len(plane.data) != expected:
                raise ValueError(
                    f"{plane.channel.value} data length must be {expected} bytes, "
                    f"got {len(plane.data)}"
                )

        # At most three iterators and three flags are live per pixel. Sensor
        # samples never pass through a colour transform, float ID conversion,
        # resampling, or a full-image decoded list.
        streams = tuple(
            _visible_samples(plane) for plane in self.planes if plane.channel is not Channel.RGB
        )
        for index, visible in enumerate(zip(*streams, strict=True)):
            if any(hit != visible[0] for hit in visible[1:]):
                raise ValueError(f"sensor background mismatch at pixel {index}")
