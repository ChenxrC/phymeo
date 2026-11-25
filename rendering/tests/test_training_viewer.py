"""Service lifecycle, camera/frame identity and bounded capture scheduling."""
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
import json
from threading import Event, Thread, get_ident
import unittest

from phymeo_rendering.viewer import ViewerFrameUnavailable, ViewerServer


class Source:
    def __init__(self, mode='replay'):
        self.thread = get_ident()
        self.mode = mode
        self.closed = False
        self.calls = []
        self.entered = Event()
        self.release = Event()
        self.release.set()
        self.bad = False

    def describe(self):
        assert get_ident() == self.thread
        return dict(mode=self.mode, title='test', fps=30, frame_count=9,
                    cameras=[dict(id='overview', label='总览'), dict(id='wrist', label='腕部')],
                    channels=[dict(id='rgb', label='RGB'), dict(id='depth', label='深度')])

    def capture(self, camera, frame, channel):
        assert get_ident() == self.thread
        self.calls.append((camera, frame, channel))
        self.entered.set()
        if not self.release.wait(5):
            raise TimeoutError()
        return b'\x89PNG\r\n\x1a\n', frame + int(self.bad), frame / 30., 2.5

    def close(self):
        assert get_ident() == self.thread
        self.closed = True


class ViewerTests(unittest.TestCase):
    def start(self, mode='replay', factory=None):
        self.server = ViewerServer(factory or (lambda: Source(mode)))
        thread = Thread(target=self.server.serve_forever, kwargs={'poll_interval': .02})
        thread.start()
        def finish():
            self.server.shutdown()
            thread.join(3)
            self.server.server_close()
            self.assertTrue(self.server.source.closed)
        self.addCleanup(finish)
        return self.server.source

    def get(self, path, headers=None):
        conn = HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        try:
            conn.request('GET', path, headers=headers or {})
            response = conn.getresponse()
            return response.status, response.headers, response.read()
        finally:
            conn.close()

    def test_resources_metadata_and_single_requested_view(self):
        source = self.start()
        for path in ('/', '/viewer.css', '/viewer.js'):
            self.assertEqual(self.get(path)[0], 200)
        metadata = json.loads(self.get('/api/scene')[2])
        self.assertEqual(metadata, dict(source_data(), current_frame=0))
        self.assertEqual(source.calls, [])
        status, headers, _ = self.get('/api/frame?camera=wrist&frame=8&channel=depth')
        self.assertEqual((status, headers['X-Camera-Id'], headers['X-Channel-Id'],
                          headers['X-Frame-Index'], headers['X-Render-Ms']), (200,'wrist','depth','8','2.5'))
        self.assertEqual(source.calls, [('wrist',8,'depth')])
        self.assertEqual(self.get('/api/frame?frame=0&camera=overview&channel=rgb')[0], 200)
        for path in ('/api/frame?frame=9&camera=overview&channel=rgb',
                     '/api/frame?frame=-1&camera=overview&channel=rgb',
                     '/api/frame?frame=3&camera=absent&channel=id',
                     '/api/frame?camera=wrist&frame=0&frame=1&channel=rgb'):
            self.assertEqual(self.get(path)[0], 400)
        self.assertEqual(self.get('/api/scene', {'Host':'other.example'})[0], 403)
        self.assertEqual(self.get('/api/scene', {'Origin':'https://other.example'})[0], 403)
        self.assertEqual(len(source.calls), 2)

    def test_live_advances_once_and_does_not_publish_mismatched_result(self):
        source = self.start('live')
        path = '/api/frame?camera=overview&channel=rgb&frame='
        self.assertEqual(self.get(path+'2')[0], 409)
        for i in (0,1,1):
            self.assertEqual(self.get(path+str(i))[0], 200)
        self.assertEqual(self.get(path+'0')[0], 409)
        source.bad = True
        self.assertEqual(self.get(path+'2')[0], 500)
        self.assertEqual(json.loads(self.get('/api/scene')[2])['current_frame'], 1)

    def test_concurrent_requests_do_not_build_a_render_queue(self):
        source = self.start()
        path = '/api/frame?camera=overview&frame=0&channel=rgb'
        source.release.clear()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self.get, path)
            try:
                self.assertTrue(source.entered.wait(2))
                self.assertEqual(self.get(path)[0], 429)
            finally:
                source.release.set()
            self.assertEqual(pending.result()[0], 200)
        self.assertEqual(len(source.calls), 1)

    def test_passive_monitor_only_renders_published_snapshots(self):
        class Observer(Source):
            latest = None

            def describe(self):
                return dict(super().describe(), mode='observe', current_frame=None, frame_count=None)

            def latest_frame(self):
                return self.latest

            def capture_latest(self, camera, channel):
                return self.capture(camera, self.latest, channel)

            def capture(self, camera, frame, channel):
                if self.latest - frame >= 2:
                    raise ViewerFrameUnavailable('expired')
                return super().capture(camera, frame, channel)

        source = self.start(factory=Observer)
        path = '/api/frame?camera=wrist&channel=depth&frame='
        self.assertEqual(self.get(path+'latest')[0], 204)
        self.assertIsNone(json.loads(self.get('/api/scene')[2])['current_frame'])
        source.latest = 23
        self.assertEqual(json.loads(self.get('/api/scene')[2])['current_frame'], 23)
        self.assertEqual(self.get(path+'latest&after=23')[0], 204)
        self.assertEqual(self.get(path+'24')[0], 409)
        self.assertEqual(self.get(path+'20')[0], 410)
        self.assertEqual(self.get(path+'22&after=21')[0], 400)
        self.assertEqual(source.calls, [])
        self.assertEqual(self.get(path+'22')[1]['X-Frame-Index'], '22')
        self.assertEqual(self.get(path+'latest&after=10')[1]['X-Frame-Index'], '23')
        self.assertEqual(source.calls, [('wrist',22,'depth'), ('wrist',23,'depth')])
        self.assertEqual(json.loads(self.get('/api/scene')[2])['current_frame'], 23)


def source_data():
    return Source().describe()
