"""Capture intent and deterministic native stage planning, without a device."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .cameras import Camera
from .outputs import Channel


class RenderPath(Enum):
    RASTER = "raster"
    PATH_TRACE = "path_trace"


@dataclass(frozen=True, slots=True)
class CaptureRequest:
    camera: Camera
    path: RenderPath
    channels: tuple[Channel, ...]


class NativePass(Enum):
    VISIBILITY = "visibility"
    SHADOWS = "shadows"
    LIGHTING = "lighting"
    TRACE = "trace"
    DISPLAY_TRANSFORM = "display_transform"
    SENSOR_EXPORT = "sensor_export"


class PassPlanner(Protocol):
    """Choose ordered native stages for one request. Sensor data bypasses display processing."""

    def plan(self, request: CaptureRequest) -> tuple[NativePass, ...]: ...


def validate_capture_request(request: CaptureRequest) -> None:
    """Validate the complete request without rewriting values or probing a device."""
    if not isinstance(request, CaptureRequest):
        raise TypeError("request must be a CaptureRequest")
    if not isinstance(request.camera, Camera):
        raise TypeError("request camera must be a Camera")
    if not isinstance(request.path, RenderPath):
        raise TypeError("request path must be a RenderPath")
    if not isinstance(request.channels, tuple):
        raise TypeError("request channels must be an immutable tuple")
    if not request.channels:
        raise ValueError("request channels must not be empty")
    seen = set()
    for index, channel in enumerate(request.channels):
        if not isinstance(channel, Channel):
            raise TypeError(f"request channels[{index}] must be a Channel")
        if channel in seen:
            raise ValueError(f"request channels contain duplicate channel {channel.value}")
        seen.add(channel)


class DefaultPassPlanner:
    """Pure native-stage selection; the backend must still check its capabilities.

    DISPLAY_TRANSFORM consumes colour only. SENSOR_EXPORT consumes primary
    geometric visibility, not the displayed or temporally accumulated RGB.
    """

    def plan(self, request: CaptureRequest) -> tuple[NativePass, ...]:
        validate_capture_request(request)
        colour = Channel.RGB in request.channels
        sensors = any(channel is not Channel.RGB for channel in request.channels)
        if request.path is RenderPath.RASTER:
            stages = (NativePass.VISIBILITY,)
            if colour:
                stages += (NativePass.SHADOWS, NativePass.LIGHTING)
        else:
            stages = (NativePass.TRACE,)
        if colour:
            stages += (NativePass.DISPLAY_TRANSFORM,)
        if sensors:
            stages += (NativePass.SENSOR_EXPORT,)
        return stages
