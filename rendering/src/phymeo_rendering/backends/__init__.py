"""Implement this high-level interface to accept scenes and produce camera exposures."""

from typing import Protocol

from ..outputs import Capture
from ..pipeline import CaptureRequest
from ..scene import SceneFrame, VisualScene


class RenderSession(Protocol):
    """Owns one uploaded scene; reject stale/mismatched frames or unsupported requests."""

    def supports(self, request: CaptureRequest) -> bool: ...

    def capture(self, frame: SceneFrame, request: CaptureRequest) -> Capture: ...

    def close(self) -> None: ...


class RenderBackend(Protocol):
    def open_scene(self, scene: VisualScene) -> RenderSession: ...
