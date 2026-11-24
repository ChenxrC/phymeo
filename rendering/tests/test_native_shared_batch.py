"""CPU coverage for public shared batch admission, ABI layout, and atomic publication."""

import ctypes as ct
from dataclasses import replace
from threading import RLock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from phymeo_rendering.backends import BackendUnavailableError, UnsupportedRequestError
from phymeo_rendering.backends._native_camera_batch import FIXED_CAMERA, ResidentBatchView
from phymeo_rendering.backends._native_device import Parameters
from phymeo_rendering.backends._native_resident import ResidentDevice
from phymeo_rendering.backends._native_shared_matrices import SharedMatrices
from phymeo_rendering.backends.native import DevicePoseExposure, NativeRenderSession, _NativeDelegate
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import FrameKey, Pose, VisualScene


class NativeSharedBatchTests(unittest.TestCase):
    def setUp(self):
        self.scene = VisualScene('test', (), (), ())
        self.key = FrameKey('test', 'world', 0, 1, 1, 'accepted_endpoint')
        self.token = DevicePoseExposure(self.key, 0.01)
        self.lease = object()
        self.mount = Pose((0., 1., 0.))
        self.active = SimpleNamespace(exposure=self.token, lease=self.lease, frame=object(), workspace=Mock(),
                                      binding=Mock(), camera_poses={0: self.mount})
        self.active.binding.camera_slot.side_effect = lambda name: 0 if name == 'mount' else None
        exchange = SharedMatrices.__new__(SharedMatrices)
        exchange._closed, exchange._active, exchange._captures = False, self.lease, 0
        exchange._lock = RLock()
        device = ResidentDevice.__new__(ResidentDevice)
        self.device = device
        device._lib = SimpleNamespace(pr_resident_shared_batch_abi_version=Mock(return_value=1),
                                      pr_resident_render_shared_batch=Mock(side_effect=self.native_write))
        device._handle, device._uploaded, device._instance_count = 7, True, 0
        device._shared_camera_capacity, device._shared_exchange = 1, exchange
        device._shared_batch_calls = device._shared_batch_bytes = device._shared_batch_views = 0
        device._shared_batch_draw, device._output_capacity = None, 0
        device._last_info = dict(max_width=4096, max_height=4096)
        device._render_planes_function = Mock(side_effect=AssertionError('single-view fallback'))
        device._error = lambda: 'native capture failed'
        delegate = _NativeDelegate.__new__(_NativeDelegate)
        delegate.scene, delegate.device, delegate.limits = self.scene, device, device._last_info
        delegate.exposure, delegate.triangles, delegate.effective_lights = 1., 0, ()
        delegate.diagnostics = lambda: {}
        self.session = NativeRenderSession(self.scene, delegate, shared_cuda_device=0)
        self.session._active_device_poses = self.active
        self.fixed = self.request('fixed', (Channel.VALID, Channel.RGB), Pose((2., 0., 0.)), 2)
        self.mounted = self.request('mount', (Channel.INSTANCE_ID,), self.mount, 1)
        # Fixed view has 2 RGBA pixels and 2 valid bytes; mounted view has 1 ID.
        self.payload = bytes((2, 3, 4, 255, 6, 7, 8, 255, 1, 0, 0, 0, 0, 0))

    def request(self, name, channels, pose, width):
        camera = Camera(name, self.key, pose, width, 1, 1., 1., 0.5, 0.5, 0.1, 10.)
        return CaptureRequest(camera, RenderPath.RASTER, channels)

    def native_write(self, handle, views, count, out, size):
        self.assertEqual((handle, count, size), (7, 2, len(self.payload)))
        self.assertEqual([(v.width, v.height, v.channel_mask, v.camera_index) for v in views],
                         [(2, 1, 9, FIXED_CAMERA), (1, 1, 4, 0)])
        self.assertEqual(tuple(views[0].params.camera)[:3], (2., 0., 0.))
        self.assertEqual(tuple(views[1].params.camera)[:3], (0., 0., 0.))
        ct.memmove(out, self.payload, size)
        return 0

    def capture(self):
        return self.session.capture_device_poses_batch(self.token, iter((self.fixed, self.mounted)))

    def test_one_abi_call_mixed_views_and_owned_unaligned_planes(self):
        self.assertEqual(ct.sizeof(ResidentBatchView), ct.sizeof(Parameters) + 16)
        first, second = self.capture()
        self.assertEqual((first.camera, second.camera), (self.fixed.camera, self.mounted.camera))
        self.assertEqual([p.channel for p in first.planes], list(self.fixed.channels))
        self.assertEqual(tuple(p.data for p in first.planes), (bytes((1, 0)), bytes((2, 3, 4, 6, 7, 8))))
        self.assertEqual(second.planes[0].data, bytes(4))
        ct.memset(self.device._output_buffer, 255, self.device._output_capacity)
        self.assertEqual(second.planes[0].data, bytes(4))
        self.device._lib.pr_resident_render_shared_batch.assert_called_once()
        self.assertEqual((self.device._shared_exchange._captures, self.device._shared_batch_views,
                          self.device._shared_batch_calls, self.device._shared_batch_bytes), (2, 2, 1, 14))
        self.assertEqual(self.session.diagnostics()['published_device_pose_batches'], 1)

    def test_invalid_late_request_never_dispatches(self):
        other_key = replace(self.key, revision=2)
        bad_mount = replace(self.mounted, camera=replace(self.mounted.camera, world_pose=Pose()))
        too_large = replace(self.fixed, camera=replace(self.fixed.camera, width=4096, height=4096))
        for views, error in (((), ValueError), ((self.fixed,) * 17, ValueError), ({self.fixed}, TypeError),
                             ((self.fixed, None), TypeError), ((self.fixed, bad_mount), ValueError),
                             ((self.fixed, replace(self.mounted, camera=replace(self.mounted.camera,
                                                                                frame_key=other_key))), ValueError),
                             ((self.fixed, replace(self.mounted, path=RenderPath.PATH_TRACE)), UnsupportedRequestError),
                             ((self.fixed, too_large), ValueError)):
            with self.subTest(error=error, views=views), self.assertRaises(error):
                self.session.capture_device_poses_batch(self.token, views)
        with self.assertRaisesRegex(ValueError, 'foreign or expired'):
            self.session.capture_device_poses_batch(replace(self.token), (self.fixed,))
        self.device._lib.pr_resident_render_shared_batch.assert_not_called()
        self.assertEqual(self.session.diagnostics()['published_device_pose_batches'], 0)

    def test_old_abi_and_native_error_have_no_partial_result_or_fallback(self):
        native = self.device._lib.pr_resident_render_shared_batch
        del self.device._lib.pr_resident_render_shared_batch
        with self.assertRaisesRegex(BackendUnavailableError, 'batch ABI is unavailable'):
            self.capture()
        self.device._lib.pr_resident_render_shared_batch = native
        self.device._lib.pr_resident_shared_batch_abi_version.return_value = 2
        with self.assertRaisesRegex(BackendUnavailableError, 'batch ABI 2'):
            self.capture()
        self.device._lib.pr_resident_shared_batch_abi_version.return_value = 1
        native.side_effect = None
        native.return_value = -1
        with self.assertRaisesRegex(RuntimeError, 'native capture failed'):
            self.capture()
        self.device._render_planes_function.assert_not_called()
        self.assertEqual(self.device._shared_batch_calls, 0)
        self.assertEqual(self.session.diagnostics()['published_device_pose_batches'], 0)

    def test_result_count_and_camera_order_are_checked_before_publication(self):
        delegate = self.session._delegate
        draw = delegate.capture_shared_batch
        for change, message in ((lambda captures: captures[:1], 'wrong number'),
                                (lambda captures: captures[::-1], 'requested exposure')):
            with self.subTest(message=message):
                delegate.capture_shared_batch = lambda *args, **kwargs: change(draw(*args, **kwargs))
                with self.assertRaisesRegex(ValueError, message):
                    self.capture()
        self.assertEqual(self.session.diagnostics()['published_device_pose_batches'], 0)

    def test_late_invalid_id_or_expired_clock_rejects_whole_batch(self):
        self.payload = self.payload[:-4] + bytes((8, 0, 0, 0))
        with self.assertRaisesRegex(ValueError, 'absent from the opened scene'):
            self.capture()
        self.payload = self.payload[:-4] + bytes(4)

        def clock_check(*args):
            if self.device._shared_batch_calls == 2:
                raise ValueError('stale accepted physics frame')

        self.active.workspace.accepted_frame_time.side_effect = clock_check
        with self.assertRaisesRegex(ValueError, 'stale accepted'):
            self.capture()
        self.assertEqual(self.device._shared_batch_calls, 2)
        self.assertEqual(self.session.diagnostics()['published_device_pose_batches'], 0)
