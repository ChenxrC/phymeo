"""Native cleanup and admission regressions, without a Vulkan device."""

import ctypes as ct
import unittest
from unittest.mock import patch

from phymeo_rendering.backends import _native_device
from phymeo_rendering.backends.native import _NativeDelegate
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import FrameKey, Pose, VisualScene


class _Library:
    def __init__(self, *, info_failure=False, destroy_failure=False):
        self.info_failure = info_failure
        self.destroy_failure = destroy_failure
        self.destroy_calls = 0

    def pr_error(self):
        return b"native diagnostic failure"

    def pr_info(self, handle, output):
        if self.info_failure:
            return -1
        value = ct.cast(output, ct.POINTER(_native_device._Info)).contents
        value.max_width = value.max_height = 4096
        value.max_vertices = 12288
        return 0

    def pr_destroy(self, handle, output=None):
        self.destroy_calls += 1
        if output is not None:
            value = ct.cast(output, ct.POINTER(_native_device._Info)).contents
            value.validation_errors = 7 if self.destroy_failure else 0
            value.validation_warnings = 2
            value.captures = 3
        return -1 if self.destroy_failure else 0

    def pr_counts(self, output):
        value = ct.cast(output, ct.POINTER(_native_device._Counts)).contents
        value.live_contexts = 0 if self.destroy_calls else 1
        value.live_frames = 0
        value.live_resources = 0 if self.destroy_calls else 8
        return 0


def device(library):
    # Inject the C-ABI test double without invoking a loader or GPU constructor.
    value = _native_device.NativeDevice.__new__(_native_device.NativeDevice)
    value._handle = 123
    value._lib = library
    value._last_info = {}
    value.validation = True
    return value


class NativeCleanup(unittest.TestCase):
    def test_info_failure_cannot_prevent_real_destroy(self):
        library = _Library(info_failure=True)
        owner = device(library)
        with self.assertRaises(RuntimeError):
            owner.info()
        owner.close()
        self.assertEqual(library.destroy_calls, 1)
        self.assertIsNone(owner._handle)
        self.assertEqual(owner.info()["captures"], 3)
        self.assertEqual(owner.info()["live_contexts"], 0)
        owner.close()
        self.assertEqual(library.destroy_calls, 1)

    def test_final_teardown_validation_errors_are_not_preclose_values(self):
        library = _Library(destroy_failure=True)
        owner = device(library)
        self.assertEqual(owner.info()["validation_errors"], 0)
        with self.assertRaises(RuntimeError):
            owner.close()
        self.assertEqual(library.destroy_calls, 1)
        final = owner.info()
        self.assertEqual(final["validation_errors"], 7)
        self.assertEqual(final["validation_warnings"], 2)
        self.assertEqual(final["live_resources"], 0)

    def test_pixel_budget_rejected_before_host_allocation(self):
        class NoAllocation:
            def __mul__(self, count):
                raise AssertionError("host image allocation reached before admission")

        owner = device(_Library())
        with patch.object(_native_device.ct, "c_uint8", NoAllocation()):
            with self.assertRaises(ValueError):
                owner.render((), _native_device.Parameters(), 4097, 4097)


class NativeFP32Admission(unittest.TestCase):
    def test_camera_fp32_collapse_and_overflow_are_not_supported(self):
        class Limits:
            def info(self):
                return dict(max_width=4096, max_height=4096, max_vertices=12288)

        scene = VisualScene("table", (), (), ())
        delegate = _NativeDelegate(scene, {}, Limits(), 1.0)
        key = FrameKey("table", "world", 0, 0, 0, "authored")
        for changes in (
            dict(near_m=1.0, far_m=1.0 + 1e-8),
            dict(fx_px=1e100), dict(fx_px=1e-100), dict(near_m=1e-100),
            dict(world_pose=Pose((1e100, 0, 0))),
        ):
            values = dict(id="front", frame_key=key, world_pose=Pose(), width=16, height=16,
                          fx_px=10.0, fy_px=10.0, cx_px=8.0, cy_px=8.0,
                          near_m=0.1, far_m=5.0)
            values.update(changes)
            request = CaptureRequest(Camera(**values), RenderPath.RASTER, (Channel.RGB,))
            with self.subTest(changes=changes):
                self.assertFalse(delegate.supports(request))
        valid = Camera("front", key, Pose(), 16, 16, 10, 10, 8, 8, 0.1, 5)
        self.assertTrue(delegate.supports(CaptureRequest(valid, RenderPath.RASTER, (Channel.RGB,))))


if __name__ == "__main__":
    unittest.main()
