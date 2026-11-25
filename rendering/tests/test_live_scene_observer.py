"""An independent simulation producer and a recording render backend, without a GPU."""
from struct import pack
from threading import Event, Thread, get_ident
import unittest

from phymeo_rendering.cameras import Camera
from phymeo_rendering.live_viewer import LiveSceneViewer
from phymeo_rendering.outputs import Capture, Channel, ImagePlane
from phymeo_rendering.scene import FrameKey, Pose, SceneFrame, VisualScene
from phymeo_rendering.viewer import ViewerFrameUnavailable


SCENE = VisualScene('training', (), (), ())


def frame(tick, seconds, *, epoch=0, scene_key=SCENE.key):
    return SceneFrame(FrameKey(scene_key, 'world', epoch, tick, tick, 'accepted_endpoint'), seconds, ())


def camera(frame, name):
    return Camera(name, frame.key, Pose(), 1, 1, 1., 1., .5, .5, .01, 10.)


class Recorder:
    def __init__(self):
        self.calls = []
        self.waiting = self.release = None

    def factory(self):
        self.calls.append(('factory', get_ident()))
        return self

    def open_scene(self, scene):
        self.calls.append(('open', get_ident(), scene.key))
        return self

    def capture(self, published, request):
        self.calls.append(('capture', get_ident(), published, request))
        if self.waiting is not None:
            self.waiting.set()
            if not self.release.wait(5):
                raise TimeoutError('test did not release capture')
        channel, = request.channels
        data = {Channel.RGB:b'\x09\x54\xff', Channel.FORWARD_DEPTH:pack('<f', 1.),
                Channel.INSTANCE_ID:pack('<I', 37), Channel.VALID:b'\x01'}[channel]
        return Capture(published.key, published.time_s, request.camera, (ImagePlane(channel,data),))

    def close(self):
        self.calls.append(('close', get_ident()))


class LiveSceneObserverTests(unittest.TestCase):
    def viewer(self, device, **kwargs):
        return LiveSceneViewer(SCENE, cameras={'fixed':lambda f:camera(f,'fixed'), 'wrist':lambda f:camera(f,'wrist')},
                               renderer_factory=device.factory, **kwargs)

    def test_publish_is_independent_of_clients_and_observer_indices_survive_physics_reset(self):
        device = Recorder()
        host = get_ident()
        with self.viewer(device, camera_labels={'wrist':'Arm wrist'}, depth_ranges={'wrist':(.2,8.)}) as view:
            server = view._server
            source = server.source
            self.assertTrue(view.url.startswith('http://127.0.0.1:'))
            self.assertIsNone(server.metadata()['current_frame'])
            with self.assertRaises(ViewerFrameUnavailable):
                server.owner.submit(source.capture_latest,'fixed','rgb').result()
            self.assertEqual(view.publish(frame(915, .915)), 0)
            self.assertEqual(view.publish(frame(0, 0., epoch=1)), 1)
            self.assertEqual(source.latest_frame(), 1)
            self.assertEqual(server.metadata()['cameras'][1]['depth_range_m'], [.2,8.])
            self.assertEqual(device.calls, [])
            for camera_id, index, channel, time in [('wrist',1,'depth',0.), ('fixed',0,'rgb',.915)]:
                png, got, seconds, ms = server.owner.submit(source.capture,camera_id,index,channel).result()
                self.assertTrue(png.startswith(b'\x89PNG'))
                self.assertEqual((got,seconds), (index,time))
                self.assertGreaterEqual(ms, 0.)
            self.assertEqual([row[3].channels for row in device.calls if row[0]=='capture'],
                             [(Channel.FORWARD_DEPTH,), (Channel.RGB,)])
            self.assertEqual(server.metadata()['current_frame'], 1)
        self.assertEqual([row[0] for row in device.calls], ['factory','open','capture','capture','close'])
        threads = {row[1] for row in device.calls}
        self.assertEqual(len(threads), 1)
        self.assertNotIn(host, threads)

    def test_latest_capture_retains_its_snapshot_while_producer_evicts_the_old_index(self):
        device = Recorder()
        device.waiting, device.release = Event(), Event()
        with self.viewer(device) as view:
            server = view._server
            source = server.source
            view.publish(frame(4, .004))
            capture = server.owner.submit(source.capture_latest, 'fixed','rgb')
            try:
                self.assertTrue(device.waiting.wait(3))
                for n in range(1,15):
                    view.publish(frame(n+4, (n+4)*.001))
                self.assertEqual(source.latest_frame(),14)
            finally:
                device.release.set()
            self.assertEqual(capture.result(timeout=3)[1:3], (0,.004))
            with self.assertRaises(ViewerFrameUnavailable):
                server.owner.submit(source.capture,'wrist',0,'depth').result()
            self.assertEqual(server.owner.submit(source.capture_latest,'wrist','id').result()[1], 14)

    def test_shutdown_waits_for_owner_and_restart_opens_an_empty_window(self):
        device = Recorder()
        device.waiting, device.release = Event(), Event()
        view = self.viewer(device).start()
        try:
            view.publish(frame(0,0.))
            server = view._server
            running = server.owner.submit(server.source.capture_latest, 'fixed','rgb')
            self.assertTrue(device.waiting.wait(3))
            closing = Thread(target=view.close)
            closing.start()
            device.release.set()
            self.assertEqual(running.result(timeout=3)[1],0)
            closing.join(3)
            self.assertFalse(closing.is_alive())
            self.assertEqual(device.calls[-1][0], 'close')
            with self.assertRaises(RuntimeError):
                view.publish(frame(2,.002))
            with view:
                self.assertIsNone(view._server.metadata()['current_frame'])
                self.assertEqual(view.publish(frame(0,0.,epoch=1)),1)
                with self.assertRaises(RuntimeError):
                    view.start()
        finally:
            device.release.set()
            view.close()

    def test_checks_static_inputs_and_distinguishes_bad_camera_from_eviction(self):
        for opt in (dict(camera_labels={'missing':'unknown'}), dict(depth_ranges={'fixed':(2.,1.)}),
                    dict(channels=(Channel.RGB, Channel.RGB)), dict(fps=True), dict(port=65536)):
            with self.subTest(opt=opt), self.assertRaises(ValueError):
                self.viewer(Recorder(), **opt)
        device = Recorder()
        def bad_camera(_):
            raise KeyError('invalid simulated mount')
        with LiveSceneViewer(SCENE, cameras={'mount':bad_camera, 'wrong':lambda f:camera(f,'another')},
                             renderer_factory=device.factory) as view:
            with self.assertRaises(ValueError):
                view.publish(frame(0,0.,scene_key='another training world'))
            view.publish(frame(0,0.))
            server = view._server
            with self.assertRaisesRegex(KeyError, 'invalid simulated mount'):
                server.owner.submit(server.source.capture_latest,'mount','rgb').result()
            with self.assertRaisesRegex(ValueError, 'returned camera ID'):
                server.owner.submit(server.source.capture_latest,'wrong','rgb').result()
            self.assertEqual(device.calls, [])
