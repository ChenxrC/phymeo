"""Explicit, optional SAPIEN 3.0.3 pure-render reference backend.

Importing this module never imports SAPIEN or NumPy. Each session selects one
real shader pipeline; a failed path is never replaced by another renderer.
"""

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from numbers import Real

from ..assets import TexturedMaterial
from ..pipeline import RenderPath
from ..scene import VisualScene
from . import BackendUnavailableError, UnsupportedRequestError, ValidatedRenderSession
from ._sapien_studio import SapienStudio


@dataclass(frozen=True, slots=True)
class SapienConfig:
    """Reference settings, independent of the public exposure request.

    PT is RGB-only with fresh accumulation. The official sampling mode uses
    SAPIEN's shipped sequence. The shifted-halton mode explicitly selects a
    locally derived shader pack with user-seeded path and primary samples.
    The shipped primary-ray clipping requires conservative frame
    admission (all scene vertices strictly inside the requested clip slab).
    display_transform is a reference-only SDR display choice after exposure;
    clip retains the original bytes, while pbr-neutral compresses HDR highlights.
    """

    path: RenderPath = RenderPath.RASTER
    device: str | None = None
    samples_per_pixel: int = 64
    max_bounces: int = 4
    seed: int | None = None
    denoiser: str = "none"
    studio: SapienStudio | None = None
    sampling_mode: str = "official"
    exposure: float = 1.0
    display_transform: str = "clip"

    def __post_init__(self):
        if not isinstance(self.display_transform, str):
            raise TypeError("display_transform must be a string")
        if self.display_transform not in ("clip", "pbr-neutral"):
            raise ValueError("display_transform must be clip or pbr-neutral")
        if isinstance(self.exposure, bool) or not isinstance(self.exposure, Real):
            raise TypeError("exposure must be a real number")
        if not isfinite(self.exposure) or not 0 <= self.exposure <= 3.4028234663852886e38:
            raise ValueError("exposure must be finite nonnegative FP32")
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
        if self.sampling_mode not in ("official", "shifted-halton"):
            raise ValueError("unknown reference sampling mode")
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise TypeError("seed must be an integer or None")
            if self.sampling_mode == "official":
                raise UnsupportedRequestError("the official SAPIEN 3.0.3 shader has a fixed seed")
        if self.sampling_mode == "shifted-halton":
            if self.path is not RenderPath.PATH_TRACE:
                raise UnsupportedRequestError("shifted Halton samples require path tracing")
            if self.seed is None or not 0 <= self.seed <= 2**32-1:
                raise ValueError("shifted Halton sampling requires an explicit uint32 seed")
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
        bound = {instance.material_id for instance in scene.instances}
        for material in scene.materials:
            if material.id in bound and isinstance(material, TexturedMaterial) and material.normal_map is not None:
                raise UnsupportedRequestError(
                    f"material {material.id!r} uses a tangent-space normal map; "
                    "the SAPIEN reference adapter does not support this material input"
                )
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
