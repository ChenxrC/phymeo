"""Backend protocols and checked publication of immutable camera exposures."""

import struct
import sys
from threading import RLock
from typing import Protocol

from ..cameras import validate_exposure
from ..outputs import Capture, Channel
from ..pipeline import CaptureRequest, validate_capture_request
from ..scene import SceneFrame, VisualScene, validate_scene_frame


def _scene_ids_valid(raw: bytes, allowed: frozenset) -> bool:
    """Validate every run of equal IDs, keeping scratch space bounded."""
    np = sys.modules.get("numpy")
    if len(raw) < 8192 * 4 or not hasattr(np, "frombuffer"):
        return all(identity == 0 or identity in allowed for (identity,) in struct.iter_unpack("<I", raw))
    pixels = np.frombuffer(raw, dtype="<u4")
    for start in range(0, len(pixels), 32768):
        block = pixels[start:start + 32768]
        first = int(block[0])
        if first and first not in allowed:
            return False
        changed = block[1:] != block[:-1]
        count = np.count_nonzero(changed)
        if count == 0:
            continue
        # Scanline interiors need a single membership check per run. Sorting
        # is cheaper for noisy label images with many repeated transitions.
        labels = block[1:][changed] if count <= 256 else np.unique(block)
        if any(i != 0 and i not in allowed for i in labels.tolist()):
            return False
    return True

class BackendUnavailableError(RuntimeError):
    """The requested implementation, dependency or device is unavailable."""


class UnsupportedRequestError(RuntimeError):
    """An available session cannot fulfill the complete capture request."""


class SessionClosedError(RuntimeError):
    """The session has started closing and cannot perform further captures."""


class RenderSession(Protocol):
    """Capture from one uploaded scene without stepping a physical source.

    Capability queries cover the complete request. Capture returns CPU-owned
    immutable bytes after completion, or raises without publishing a result.
    """

    def supports(self, request: CaptureRequest) -> bool: ...

    def capture(self, frame: SceneFrame, request: CaptureRequest) -> Capture: ...

    def close(self) -> None: ...


class RenderBackend(Protocol):
    def open_scene(self, scene: VisualScene) -> RenderSession: ...


class ValidatedRenderSession:
    """Wrap a scene's rendering delegate with shared input and publication gates.

    On successful construction this wrapper owns the delegate's close lifecycle.
    The backend must have opened that delegate with this same immutable scene;
    this wrapper neither uploads assets nor verifies external URI file contents.
    Delegates construct results through the public Capture/ImagePlane API, which
    owns pixel validation. This gate adds source, request and scene-ID checks.

    Calls are serialized. Replay may revisit older ticks; there is no implicit
    monotonic clock rule. A device delegate owns any temporal accumulation and
    must invalidate it when exposure or scene identity changes. No fallback,
    device import, physics step or fabricated image is performed here.
    """

    def __init__(self, scene: VisualScene, delegate: RenderSession):
        if not isinstance(scene, VisualScene):
            raise TypeError("scene must be VisualScene")
        for name in ("supports", "capture", "close"):
            if not callable(getattr(delegate, name, None)):
                raise TypeError(f"delegate must provide callable {name}")
        self._scene = scene
        self._delegate = delegate
        self._instance_ids = frozenset(instance.id for instance in scene.instances)
        self._lock = RLock()
        self._closed = False
        self._close_error: BaseException | None = None

    def _check_open(self) -> None:
        if self._closed:
            raise SessionClosedError("render session is closed")

    def _check_request(self, request: CaptureRequest) -> None:
        validate_capture_request(request)
        if request.camera.frame_key.scene_key != self._scene.key:
            raise ValueError("request camera scene_key does not match the opened scene")

    def _supported(self, request: CaptureRequest) -> bool:
        answer = self._delegate.supports(request)
        if type(answer) is not bool:
            raise TypeError("delegate.supports must return bool")
        return answer

    def supports(self, request: CaptureRequest) -> bool:
        """Check a complete, well-formed request without rendering or fallback."""
        with self._lock:
            self._check_open()
            self._check_request(request)
            return self._supported(request)

    def capture(self, frame: SceneFrame, request: CaptureRequest) -> Capture:
        """Publish only a result matching the exact frame, exposure and request."""
        with self._lock:
            self._check_open()
            self._check_request(request)
            validate_scene_frame(self._scene, frame)
            validate_exposure(frame, request.camera)
            if not self._supported(request):
                raise UnsupportedRequestError(
                    f"session does not support complete request: {request.path.value}, "
                    f"channels={tuple(channel.value for channel in request.channels)!r}"
                )
            result = self._delegate.capture(frame, request)
            if not isinstance(result, Capture):
                raise TypeError("delegate.capture must return Capture")
            if result.frame_key != frame.key:
                raise ValueError("capture frame_key does not match the source frame")
            if result.time_s != frame.time_s:
                raise ValueError("capture time_s does not match the source frame")
            if result.camera != request.camera:
                raise ValueError("capture camera does not match the requested exposure")
            if tuple(plane.channel for plane in result.planes) != request.channels:
                raise ValueError("capture channels must match request channels in order")
            # Capture already validates layout, depth, masks and background.
            # Only this scene-owning layer can validate positive instance IDs.
            for plane in result.planes:
                if plane.channel is Channel.INSTANCE_ID and not _scene_ids_valid(plane.data, self._instance_ids):
                    for (identity,) in struct.iter_unpack("<I", plane.data):
                        if identity != 0 and identity not in self._instance_ids:
                            raise ValueError(
                                f"capture instance ID {identity} is absent from the opened scene"
                            )
            return result

    def close(self) -> None:
        """Close at most once; retain and report any uncertain release failure.

        A failed close still closes the public gate. Repeated close reports the
        same error without retrying a potentially partially completed release.
        """
        with self._lock:
            if self._closed:
                if self._close_error is not None:
                    raise self._close_error
                return
            self._closed = True
            try:
                self._delegate.close()
            except BaseException as error:
                self._close_error = error
                raise
