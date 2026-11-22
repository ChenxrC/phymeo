"""Capture intent and native pass planning. The planner is an extension point."""

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
