"""Explicit entry point for an optional, future SAPIEN reference adapter.

G2 does not initialize SAPIEN or substitute its renderer for the native backend.
The pure-rendering raster and path-tracing modes each require real validation.
"""

from ..scene import VisualScene
from . import BackendUnavailableError, RenderSession


class SapienBackend:
    """No reference rendering paths are currently implemented."""

    def open_scene(self, scene: VisualScene) -> RenderSession:
        if not isinstance(scene, VisualScene):
            raise TypeError("scene must be VisualScene")
        raise BackendUnavailableError(
            "SAPIEN reference raster and path_trace adapters are not implemented; "
            "no SAPIEN runtime was imported and no fallback was selected"
        )
