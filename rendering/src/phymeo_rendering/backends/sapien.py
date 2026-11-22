"""Explicit, optional SAPIEN 3.0.3 pure-render reference backend.

Importing this module never imports SAPIEN or NumPy. Each session selects one
real shader pipeline; a failed path is never replaced by another renderer.
"""

from copy import deepcopy
from dataclasses import dataclass

from ..pipeline import RenderPath
from ..scene import VisualScene
from . import BackendUnavailableError, UnsupportedRequestError, ValidatedRenderSession
from ._sapien_studio import SapienStudio


@dataclass(frozen=True, slots=True)
class SapienConfig:
    """Reference settings, independent of the public exposure request.

    PT is RGB-only, with fresh accumulation, no denoiser and no user-settable
    random seed. Its shipped primary-ray clipping requires conservative frame
    admission (all scene vertices strictly inside the requested clip slab).
    """

    path: RenderPath = RenderPath.RASTER
    device: str | None = None
    samples_per_pixel: int = 64
    max_bounces: int = 4
    seed: int | None = None
    denoiser: str = "none"
    studio: SapienStudio | None = None

    def __post_init__(self):
        if not isinstance(self.path, RenderPath):
            raise TypeError("path must be RenderPath")
        if self.device is not None:
            if not isinstance(self.device, str):
                raise TypeError("device must be a SAPIEN device alias string or None")
            if not self.device.strip():
                raise ValueError("device must not be blank")
        for name, minimum in (("samples_per_pixel", 1), ("max_bounces", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, not bool")
            if not minimum <= value <= 2**31 - 1:
                raise ValueError(f"{name} is outside the signed 32-bit shader range")
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise TypeError("seed must be an integer or None")
            raise UnsupportedRequestError("SAPIEN 3.0.3 has no public user seed control")
        if not isinstance(self.denoiser, str):
            raise TypeError("denoiser must be a string")
        if self.denoiser != "none":
            raise UnsupportedRequestError("this reference supports denoiser='none' only")
        if self.studio is not None:
            if not isinstance(self.studio, SapienStudio):
                raise TypeError("studio must be SapienStudio or None")
            if self.path is not RenderPath.PATH_TRACE:
                raise UnsupportedRequestError("studio requires path_trace; no raster approximation")


class _SapienValidatedSession(ValidatedRenderSession):
    def diagnostics(self):
        """Detached, per-session evidence, readable after close as well."""
        with self._lock:
            return self._delegate.diagnostics_snapshot()


class SapienBackend:
    """Owns pure render scenes and exposes JSON-ready initialization evidence.

    ``last_open_diagnostics`` is a detached sidecar snapshot, not a change to
    the three-method RenderSession protocol. It does not claim image accuracy
    merely because the selected shader initialized successfully.
    """

    def __init__(self, config: SapienConfig | None = None):
        if config is not None and not isinstance(config, SapienConfig):
            raise TypeError("config must be SapienConfig or None")
        self.config = SapienConfig() if config is None else config
        self._last_open_diagnostics = None

    @property
    def last_open_diagnostics(self):
        return deepcopy(self._last_open_diagnostics)

    def open_scene(self, scene: VisualScene):
        if not isinstance(scene, VisualScene):
            raise TypeError("scene must be VisualScene")
        self._last_open_diagnostics = None
        try:
            import sapien
        except (ImportError, OSError) as exc:
            raise BackendUnavailableError(
                "SAPIEN 3.0.3 reference dependencies unavailable"
            ) from exc
        if sapien.__version__ != "3.0.3":
            raise BackendUnavailableError(
                f"reference requires SAPIEN 3.0.3, found {sapien.__version__!r}"
            )
        # A successful import of the supported wheel already loads its NumPy dependency.
        import numpy

        from ._sapien_session import SapienSession

        delegate = SapienSession(scene, self.config, sapien, numpy)
        self._last_open_diagnostics = delegate.diagnostics
        return _SapienValidatedSession(scene, delegate)
