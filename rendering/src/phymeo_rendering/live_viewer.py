"""Observe immutable simulation frames in a local browser without driving the simulation."""
from collections.abc import Mapping
from math import isfinite
from numbers import Real
import re
from threading import Lock, Thread, current_thread
from time import perf_counter

from ._viewer_planes import CHANNEL_IDS, channel_metadata, display_png
from .cameras import validate_exposure
from .outputs import Channel
from .pipeline import CaptureRequest, RenderPath
from .scene import SceneFrame, VisualScene, validate_scene_frame
from .viewer import ViewerFrameUnavailable, ViewerServer


_DEFAULT_CHANNELS = (Channel.RGB, Channel.FORWARD_DEPTH, Channel.INSTANCE_ID)
_HISTORY = 8


class _Publications:
    """The producer exchanges references only. Readers retain a chosen frame past eviction."""
    def __init__(self):
        self.lock = Lock()
        self.frames = {}
        self.next_index = 0
        self.active = False

    def start(self):
        with self.lock:
            self.active = True

    def stop(self):
        with self.lock:
            self.active = False
            self.frames.clear()

    def append(self, frame):
        with self.lock:
            if not self.active:
                raise RuntimeError('start the live scene viewer before publishing')
            i = self.next_index
            self.next_index += 1
            self.frames[i] = frame
            self.frames.pop(i-_HISTORY, None)
            return i

    def latest(self):
        with self.lock:
            return self.next_index-1 if self.frames else None

    def snapshot(self, index):
        with self.lock:
            if index is None:
                index = self.next_index-1
            if index not in self.frames:
                raise ViewerFrameUnavailable('the requested publication is outside the retained frame window')
            return index, self.frames[index]


class _ObserverSource:
    """GPU resources belong to the server's single source worker."""
    def __init__(self, viewer):
        self.viewer = viewer
        self.session = None
        self.backend = None

    def latest_frame(self):
        return self.viewer._frames.latest()

    def describe(self):
        viewer = self.viewer
        return dict(title=viewer._title, mode='observe', fps=viewer._fps, frame_count=None,
                    current_frame=self.latest_frame(), seekable=False, retained_frames=_HISTORY,
                    cameras=[dict(id=k, label=viewer._labels[k], depth_range_m=list(viewer._depth[k]))
                             for k in viewer._cameras], channels=channel_metadata(viewer._channels),
                    capture_ms_scope='camera binding, scene opening if needed, and native capture; excludes PNG')

    def capture(self, camera_id, index, channel_id):
        if type(index) is not int or index < 0:
            raise ValueError('publication index must be a nonnegative integer')
        return self._capture(camera_id, index, channel_id)

    def capture_latest(self, camera_id, channel_id):
        return self._capture(camera_id, None, channel_id)

    def _capture(self, camera_id, index, channel_id):
        viewer = self.viewer
        channel = CHANNEL_IDS.get(channel_id)
        if camera_id not in viewer._cameras or channel not in viewer._channels:
            raise ValueError('unknown observer camera or channel')
        index, frame = viewer._frames.snapshot(index)
        started = perf_counter()
        validate_scene_frame(viewer._scene, frame)
        camera = viewer._cameras[camera_id](frame)
        validate_exposure(frame, camera)
        if camera.id != camera_id:
            raise ValueError(f'camera binding {camera_id!r} returned camera ID {camera.id!r}')
        if self.session is None:
            backend = viewer._renderer_factory()
            try:
                session = backend.open_scene(viewer._scene)
                if not callable(getattr(session, 'capture', None)) or not callable(getattr(session, 'close', None)):
                    if session is not backend and callable(getattr(session, 'close', None)):
                        session.close()
                    raise TypeError('open_scene must return a render session with capture() and close()')
            except BaseException:
                close = getattr(backend, 'close', None)
                if close is not None:
                    close()
                raise
            self.backend, self.session = backend, session
        shot = self.session.capture(frame, CaptureRequest(camera, RenderPath.RASTER, (channel,)))
        ms = (perf_counter()-started)*1000
        if (shot.frame_key, shot.camera, shot.time_s) != (frame.key, camera, frame.time_s):
            raise ValueError('renderer returned a different observer exposure')
        return display_png(shot, channel, viewer._depth[camera_id]), index, float(frame.time_s), ms

    def close(self):
        session, backend = self.session, self.backend
        self.session = self.backend = None
        try:
            if session is not None:
                session.close()
        finally:
            if backend is not None and backend is not session:
                close = getattr(backend, 'close', None)
                if close is not None:
                    close()


def _range(value):
    try:
        low, high = value
    except (TypeError, ValueError) as exc:
        raise ValueError('depth range must be a (near, far) pair in metres') from exc
    if (not all(isinstance(x, Real) and not isinstance(x, bool) and isfinite(x) for x in (low,high))
            or not 0 <= low < high):
        raise ValueError('depth range must be finite with 0 <= near < far')
    return float(low), float(high)


class LiveSceneViewer:
    """Publish from a simulation thread and inspect the selected camera on demand.

    Use as a context manager or call start()/close(). Read url after starting.
    Camera callbacks return the registered camera ID, run on the rendering worker and derive their pose
    solely from the supplied frame (including wrist mounts). The renderer
    factory supplies a fresh backend on its worker's first image request.

    publish() retains the eight most recent immutable frames and returns a
    monotonically increasing viewer index. Simulation epochs, keys and times
    remain those of the producer, including across a physics reset. Closing
    releases retained frames and joins the HTTP and rendering workers. Starting
    this object again opens an empty window and continues the publication index.
    """

    def __init__(self, scene, *, cameras, renderer_factory, title=None, fps=15,
                 channels=_DEFAULT_CHANNELS, port=0, camera_labels=None, depth_ranges=None):
        if not isinstance(scene, VisualScene):
            raise TypeError('scene must be a VisualScene')
        if not isinstance(cameras, Mapping) or not cameras:
            raise ValueError('cameras must map at least one camera ID to a frame callback')
        self._cameras = dict(cameras)
        if not all(isinstance(k,str) and re.fullmatch(r'[A-Za-z0-9_.-]+',k) and callable(v)
                   for k,v in self._cameras.items()):
            raise ValueError('each camera needs a URL-safe ID and a callable frame binding')
        if not callable(renderer_factory):
            raise TypeError('renderer_factory must be callable')
        if title is not None and (not isinstance(title,str) or not title.strip()):
            raise ValueError('title must be a nonempty string')
        if isinstance(fps,bool) or not isinstance(fps,Real) or not isfinite(fps) or not 0 < fps <= 1000:
            raise ValueError('fps must be a finite number in (0, 1000]')
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError('port must be an integer in 0..65535')
        channels = tuple(channels)
        if not channels or not all(isinstance(ch,Channel) for ch in channels) or len(set(channels)) != len(channels):
            raise ValueError('channels must be a nonempty sequence of distinct Channel values')
        for override in (camera_labels, depth_ranges):
            if override is not None and (not isinstance(override, Mapping) or set(override)-self._cameras.keys()):
                raise ValueError('camera overrides must be keyed by declared camera IDs')
        self._labels = {k:(camera_labels or {}).get(k,k) for k in self._cameras}
        if not all(isinstance(v,str) and v.strip() for v in self._labels.values()):
            raise ValueError('camera labels must be nonempty strings')
        self._depth = {k:_range((depth_ranges or {}).get(k,(0.,5.))) for k in self._cameras}
        self._scene, self._renderer_factory = scene, renderer_factory
        self._title, self._fps, self._port, self._channels = title or scene.key, float(fps), port, channels
        self._frames = _Publications()
        self._lifecycle = Lock()
        self._server = self._thread = self._source_thread = None

    @property
    def url(self):
        server = self._server
        if server is None:
            raise RuntimeError('start the live scene viewer to obtain its URL')
        return server.url

    def start(self):
        """Start serving in the background. The renderer opens on the first capture."""
        if current_thread() in (self._thread, self._source_thread):
            raise RuntimeError('start the live viewer from its host, outside a camera or render callback')

        def source_factory():
            self._source_thread = current_thread()
            return _ObserverSource(self)

        with self._lifecycle:
            if self._server is not None:
                raise RuntimeError('live scene viewer is already running')
            server = ViewerServer(source_factory, port=self._port)
            try:
                thread = Thread(target=server.serve_forever, kwargs={'poll_interval': .05},
                                name='phymeo-live-viewer-http', daemon=True)
                self._frames.start()
                thread.start()
            except BaseException:
                self._frames.stop()
                server.server_close()
                self._source_thread = None
                raise
            self._server, self._thread = server, thread
        return self

    def publish(self, frame):
        """Retain an immutable SceneFrame in O(1); scene and camera checks run on capture."""
        if not isinstance(frame, SceneFrame):
            raise TypeError('publish expects an immutable SceneFrame')
        if frame.key.scene_key != self._scene.key:
            raise ValueError('frame belongs to a different visual scene')
        return self._frames.append(frame)

    def close(self):
        if current_thread() in (self._thread,self._source_thread):
            raise RuntimeError('close the live viewer from its host, outside a camera or render callback')
        with self._lifecycle:
            server = self._server
            if server is None:
                return
            self._frames.stop()
            try:
                server.shutdown()
                self._thread.join()
            finally:
                try:
                    server.server_close()
                finally:
                    self._server = self._thread = self._source_thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
