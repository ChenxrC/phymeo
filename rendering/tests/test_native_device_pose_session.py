"""Public session ownership gates. The real raster comparison lives in examples."""
import unittest
from unittest.mock import Mock

from phymeo_rendering.backends import SessionClosedError, UnsupportedRequestError
from phymeo_rendering.backends.native import DevicePoseExposure, NativeBackend, NativeRenderSession
from phymeo_rendering.scene import FrameKey, VisualScene


class DevicePoseSessionTests(unittest.TestCase):
    def setUp(self):
        self.scene = VisualScene("fixture", (), (), ())
        self.delegate = Mock()
        self.session = NativeRenderSession(self.scene, self.delegate, shared_cuda_device=0)

    def test_exact_exposure_identity_required(self):
        key = FrameKey("fixture", "41", 0, 1, 0, "accepted_endpoint")
        token = DevicePoseExposure(key, 0.0)
        other = DevicePoseExposure(key, 0.0)
        self.session._active_device_poses = (token, object(), object(), object(), 41)
        with self.assertRaisesRegex(ValueError, "foreign or expired"):
            self.session.capture_device_poses(other, None)
        with self.assertRaisesRegex(RuntimeError, "before closing"):
            self.session.close()
        self.delegate.capture.assert_not_called()
        self.delegate.close.assert_not_called()

    def test_retry_close_after_retained_fence_timeout(self):
        self.delegate.close.side_effect = [TimeoutError("pending fence"), None]
        with self.assertRaises(TimeoutError):
            self.session.close()
        with self.assertRaises(SessionClosedError):
            self.session.capture_device_poses(None, None)
        self.session.close()
        self.session.close()
        self.assertEqual(self.delegate.close.call_count, 2)

    def test_unsupported_modes_and_host_capture(self):
        for opts in (dict(resident=False), dict(resident=True, msaa_samples=4),
                     dict(resident=True, shadow_mode="ray_query")):
            with self.subTest(opts=opts), self.assertRaises(UnsupportedRequestError):
                NativeBackend(shared_cuda_device=0, **opts)
        with self.assertRaisesRegex(UnsupportedRequestError, "capture_device_poses"):
            self.session.capture(None, None)
        with self.assertRaises(UnsupportedRequestError):
            with NativeRenderSession(self.scene, self.delegate).publish_device_poses(
                        None, None, None, producer_stream=None):
                self.fail("host-only session unexpectedly acquired shared poses")
