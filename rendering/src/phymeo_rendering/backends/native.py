"""Explicit entry point for the future native Vulkan implementation.

G2 provides checked session contracts, not a graphics device. Native raster and
native path tracing require independent device implementations and acceptance.
Importing this module does not load Vulkan, choose a GPU or create resources.
"""

from ..scene import VisualScene
from . import BackendUnavailableError, RenderSession


class NativeBackend:
    """No native rendering paths are currently implemented."""

    def open_scene(self, scene: VisualScene) -> RenderSession:
        if not isinstance(scene, VisualScene):
            raise TypeError("scene must be VisualScene")
        raise BackendUnavailableError(
            "native raster and path_trace are not implemented; "
            "no Vulkan device was initialized and no fallback was selected"
        )
