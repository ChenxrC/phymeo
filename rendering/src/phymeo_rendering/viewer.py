"""Local browser viewer for calibrated training and test captures.

Run a ViewerServer as a context manager. Its source factory, describe,
capture and close methods all run on one dedicated thread. A source returns
(PNG bytes, source frame index, simulation seconds, capture milliseconds).
Observe-mode sources also expose a thread-safe latest_frame() index and a
capture_latest(camera, channel) method, which pins a published snapshot.
"""
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import math
import re
from threading import Lock
from urllib.parse import parse_qs, urlsplit


_STATIC = {'/': ('index.html', 'text/html; charset=utf-8'),
           '/viewer.css': ('viewer.css', 'text/css; charset=utf-8'),
           '/viewer.js': ('viewer.js', 'text/javascript; charset=utf-8')}


class ViewerFrameUnavailable(LookupError):
    """Requested observation has not been published or has left the source buffer."""


def _frame_number(text):
    if not re.fullmatch(r'[0-9]{1,16}', text):
        raise ValueError('Expected a nonnegative frame index')
    return int(text)


def _manifest(source):
    # Copy the public metadata, including optional per-camera and channel legends.
    data = json.loads(json.dumps(source.describe(), allow_nan=False))
    if not isinstance(data['title'], str) or not data['title'].strip():
        raise ValueError('viewer needs a scene title')
    if data['mode'] not in ('live', 'replay', 'observe') or not 0 < data['fps'] <= 1000:
        raise ValueError('invalid viewer mode or frame rate')
    count = data['frame_count']
    if not (count is None and data['mode'] in ('live', 'observe')) and (type(count) is not int or count < 1):
        raise ValueError('frame_count must be positive for a recording')
    for field in ('cameras', 'channels'):
        ids = [entry['id'] for entry in data[field]]
        if (not ids or len(set(ids)) != len(ids)
                or not all(re.fullmatch(r'[A-Za-z0-9_.-]+', name) for name in ids)
                or not all(isinstance(entry['label'], str) and entry['label'] for entry in data[field])):
            raise ValueError(f'invalid {field} in viewer metadata')
    index = data.setdefault('current_frame', None if data['mode'] == 'observe' else 0)
    if not (index is None and data['mode'] == 'observe') and (
            type(index) is not int or index < 0 or (count is not None and index >= count)):
        raise ValueError('invalid starting frame')
    return data


class _ViewerHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, format, *args):
        pass  # One local HTTP request per displayed frame.

    def _send(self, code, mime, body, **headers):
        self.send_response(code)
        self.send_header('Content-Type', mime)
        if code != HTTPStatus.NO_CONTENT:
            self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy',
                         "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; "
                         "object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        for key, value in headers.items():
            self.send_header(key, str(value))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, value):
        self._send(code, 'application/json; charset=utf-8',
                   json.dumps(value, ensure_ascii=False, allow_nan=False).encode())

    def do_GET(self):
        try:
            self._route()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
            pass  # A tab can close or change its selection during capture.

    def _route(self):
        port = self.server.server_port
        hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
        origin = self.headers.get('Origin')
        path = urlsplit(self.path)
        if (self.headers.get('Host', '').lower() not in hosts
                or (origin and origin not in {'http://' + h for h in hosts})
                or (path.path.startswith('/api/') and self.headers.get('Sec-Fetch-Site') == 'cross-site')):
            return self._json(403, {'error': 'Local viewer requests only'})
        if path.path == '/favicon.ico':
            return self._send(204, 'image/x-icon', b'')
        if path.path in _STATIC:
            filename, mime = _STATIC[path.path]
            return self._send(200, mime, self.server.assets[filename])
        if path.path == '/api/scene':
            return self._json(200, self.server.metadata())
        if path.path != '/api/frame':
            return self._json(404, {'error': 'Unknown viewer resource'})
        try:
            params = parse_qs(path.query, strict_parsing=True, keep_blank_values=True, max_num_fields=4)
            required = {'camera', 'frame', 'channel'}
            if params.keys() not in (required, required | {'after'}) or any(len(v) != 1 for v in params.values()):
                raise ValueError('Expected camera, frame and channel')
            camera, channel = params['camera'][0], params['channel'][0]
            mode = self.server.scene_info['mode']
            latest = params['frame'][0] == 'latest' and mode == 'observe'
            index = None if latest else _frame_number(params['frame'][0])
            if 'after' in params and not latest:
                raise ValueError('after is only valid with frame=latest')
            after = _frame_number(params['after'][0]) if 'after' in params else None
            self.server.validate_request_values(camera, index, channel)
        except ValueError as exc:
            return self._json(400, {'error': str(exc)})
        if mode == 'observe':
            current = self.server.metadata()['current_frame']
            if latest and (current is None or current == after):
                return self._send(204, 'image/png', b'')
            if not latest and (current is None or index > current):
                return self._json(409, {'error': '这帧尚未发布', 'current_frame': current})
        if not self.server.capture_slot.acquire(blocking=False):
            return self._json(HTTPStatus.TOO_MANY_REQUESTS, {'error': '另一个视图正在采集，请稍后重试'})
        try:
            if mode == 'live':
                current = self.server.metadata()['current_frame']
                if index not in (current, current + 1):
                    return self._json(409, {'error': '现场仿真只能查看当前帧或前进一步', 'current_frame': current})
            try:
                png, actual, seconds, duration = self.server.owner.submit(
                    self.server.capture, camera, index, channel).result()
            except ViewerFrameUnavailable as exc:
                return self._json(410, {'error': str(exc), 'current_frame': self.server.metadata()['current_frame']})
            except Exception as exc:
                return self._json(500, {'error': f'{type(exc).__name__}: {exc}'})
        finally:
            self.server.capture_slot.release()
        self._send(200, 'image/png', png, **{
            'X-Frame-Index': actual, 'X-Sim-Time-S': seconds, 'X-Render-Ms': duration,
            'X-Camera-Id': camera, 'X-Channel-Id': channel})


class ViewerServer(ThreadingHTTPServer):
    """Serve one source on loopback and render only the requested camera/channel."""
    daemon_threads = False

    def __init__(self, source_factory, *, port=0):
        root = files('phymeo_rendering').joinpath('viewer_assets')
        self.assets = {name: root.joinpath(name).read_bytes() for name, _ in _STATIC.values()}
        self.capture_slot = Lock()
        self.state_lock = Lock()
        self.source = None
        super().__init__(('127.0.0.1', port), _ViewerHandler)
        self.owner = ThreadPoolExecutor(max_workers=1, thread_name_prefix='phymeo-viewer-source')
        try:
            self.source = self.owner.submit(source_factory).result()
            self.scene_info = self.owner.submit(_manifest, self.source).result()
            self.current_frame = self.scene_info['current_frame']
            self.cameras = {item['id'] for item in self.scene_info['cameras']}
            self.channels = {item['id'] for item in self.scene_info['channels']}
        except BaseException:
            self.server_close()
            raise

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server_port}/'

    def metadata(self):
        if self.scene_info['mode'] == 'observe':
            return dict(self.scene_info, current_frame=self.source.latest_frame())
        with self.state_lock:
            return dict(self.scene_info, current_frame=self.current_frame)

    def validate_request_values(self, camera, index, channel):
        if camera not in self.cameras or channel not in self.channels:
            raise ValueError('Unknown camera or channel')
        count = self.scene_info['frame_count']
        if count is not None and index is not None and index >= count:
            raise ValueError('Requested frame exceeds the scene duration')

    def capture(self, camera, index, channel):
        if index is None:
            png, actual, seconds, duration = self.source.capture_latest(camera, channel)
        else:
            png, actual, seconds, duration = self.source.capture(camera, index, channel)
        if (type(actual) is not int or actual < 0 or (index is not None and index != actual) or not isinstance(png, bytes)
                or not png.startswith(b'\x89PNG\r\n\x1a\n') or not math.isfinite(seconds) or seconds < 0
                or not math.isfinite(duration) or duration < 0):
            raise ValueError('Source returned invalid image, timing or frame identity')
        if self.scene_info['mode'] != 'observe':
            with self.state_lock:
                self.current_frame = actual
        return png, actual, seconds, duration

    def server_close(self):
        super().server_close()
        owner = getattr(self, 'owner', None)
        if owner is not None:
            try:
                if self.source is not None:
                    owner.submit(self.source.close).result()
            finally:
                owner.shutdown()
                self.owner = None
