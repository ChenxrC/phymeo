"""Host output contracts, checked against explicit bytes and source metadata."""

import tracemalloc
import unittest
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from itertools import combinations
from math import inf, nan
from struct import pack, unpack

from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Capture, Channel, ImagePlane
from phymeo_rendering.scene import FrameKey, Pose


def _key():
    return FrameKey("table-v1", "world-a", 2, 11, 7, "accepted_endpoint")


def _camera(width=3, height=2):
    return Camera("front", _key(), Pose(), width, height, 500.0, 500.0, 1.5, 1.0, 0.1, 5.0)


def _planes():
    # Six pixels in row-major order. Background colour is deliberately nonblack.
    return (
        ImagePlane(
            Channel.RGB,
            bytes((255, 0, 0, 0, 255, 0, 17, 23, 31, 0, 0, 255, 255, 255, 255, 0, 0, 0)),
        ),
        ImagePlane(Channel.FORWARD_DEPTH, pack("<6f", 0.25, 1.0, inf, 2.0, 3.0, 4.0)),
        ImagePlane(Channel.INSTANCE_ID, pack("<6I", 1, 256, 0, 2**32 - 1, 7, 8)),
        ImagePlane(Channel.VALID, bytes((1, 1, 0, 1, 1, 1))),
    )


_DEFAULT_PLANES = object()


def _capture(planes=_DEFAULT_PLANES, **changes):
    values = dict(
        frame_key=_key(),
        time_s=0.007,
        camera=_camera(),
        planes=_planes() if planes is _DEFAULT_PLANES else planes,
    )
    values.update(changes)
    return Capture(**values)


class ImagePlaneTests(unittest.TestCase):
    def test_channel_requires_enum_not_format_string(self):
        for channel in ("srgb8_rgb", None, 0):
            with self.subTest(channel=channel):
                with self.assertRaisesRegex(TypeError, "channel"):
                    ImagePlane(channel, b"abc")

    def test_data_requires_immutable_owned_bytes(self):
        for data in (bytearray(b"abc"), memoryview(b"abc"), (1, 2, 3), "abc", None):
            with self.subTest(data_type=type(data).__name__):
                with self.assertRaisesRegex(TypeError, "data"):
                    ImagePlane(Channel.RGB, data)

    def test_owned_copy_is_unchanged_after_source_buffer_reuse(self):
        source = bytearray((1, 2, 3, 4, 5, 6))
        plane = ImagePlane(Channel.RGB, bytes(source))
        capture = _capture((plane,), camera=_camera(2, 1))
        source[:] = b"\x00" * len(source)
        self.assertEqual(capture.planes[0].data, b"\x01\x02\x03\x04\x05\x06")
        with self.assertRaises(FrozenInstanceError):
            plane.data = b"changed"
        with self.assertRaises(FrozenInstanceError):
            capture.planes = ()

    def test_nonempty_dimension_is_checked_by_capture(self):
        plane = ImagePlane(Channel.RGB, b"")
        with self.assertRaisesRegex(ValueError, "srgb8_rgb"):
            _capture((plane,))


class CaptureTests(unittest.TestCase):
    def test_non_square_capture_preserves_pixel_bytes_and_metadata(self):
        planes = _planes()
        camera = _camera()
        capture = _capture(planes, camera=camera)
        self.assertIs(capture.camera, camera)
        self.assertEqual(capture.frame_key, _key())
        self.assertEqual(capture.time_s, 0.007)
        self.assertEqual((capture.camera.width, capture.camera.height), (3, 2))
        self.assertEqual(capture.planes, planes)
        self.assertEqual(capture.planes[0].data[:9], bytes((255, 0, 0, 0, 255, 0, 17, 23, 31)))
        self.assertEqual(capture.planes[1].data[:4], b"\x00\x00\x80\x3e")
        self.assertEqual(unpack("<6f", capture.planes[1].data), (0.25, 1.0, inf, 2.0, 3.0, 4.0))
        self.assertEqual(unpack("<6I", capture.planes[2].data), (1, 256, 0, 2**32 - 1, 7, 8))

    def test_all_nonempty_channel_subsets_are_accepted_in_supplied_order(self):
        planes = _planes()
        for size in range(1, 5):
            for subset in combinations(planes, size):
                with self.subTest(channels=tuple(p.channel for p in subset)):
                    capture = _capture(tuple(reversed(subset)))
                    self.assertEqual(capture.planes, tuple(reversed(subset)))

    def test_capture_requires_typed_frame_camera_and_tuple_planes(self):
        for field, values in (
            ("frame_key", (None, {"tick": 7}, 7)),
            ("camera", (None, {}, "front")),
            ("planes", (list(_planes()), iter(_planes()), None)),
        ):
            for value in values:
                with self.subTest(field=field, value_type=type(value).__name__):
                    with self.assertRaisesRegex(TypeError, field):
                        _capture(**{field: value})

    def test_capture_rejects_non_plane_members(self):
        for value in (None, Channel.RGB, b"abc"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "planes"):
                    _capture((value,))

    def test_capture_rejects_empty_or_duplicate_planes(self):
        with self.assertRaisesRegex(ValueError, "planes"):
            _capture(())
        for plane in _planes():
            with self.subTest(channel=plane.channel):
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    _capture((plane, plane))

    def test_capture_rejects_each_frame_key_mismatch(self):
        changes = dict(
            scene_key="other", world_id="world-b", epoch=3, revision=12, tick=8, phase="step_input"
        )
        for field, value in changes.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "frame_key"):
                    _capture(frame_key=replace(_key(), **{field: value}))

    def test_capture_accepts_equal_distinct_frame_keys(self):
        camera = _camera()
        key = _key()
        self.assertIsNot(camera.frame_key, key)
        capture = _capture(frame_key=key, camera=camera)
        self.assertEqual(capture.frame_key, capture.camera.frame_key)

    def test_capture_rejects_non_real_time(self):
        for value in (True, False, None, "0.0", 1 + 0j):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "time_s"):
                    _capture(time_s=value)

    def test_capture_rejects_invalid_or_unrepresentable_time(self):
        for value in (-0.1, nan, inf, -inf, 10**1000, Fraction(10**1000, 3)):
            with self.subTest(numeric_type=type(value).__name__):
                with self.assertRaisesRegex(ValueError, "time_s"):
                    _capture(time_s=value)

    def test_zero_and_fractional_source_times_are_preserved(self):
        for value in (0, Fraction(7, 1000)):
            with self.subTest(value=value):
                capture = _capture(time_s=value)
                self.assertIs(capture.time_s, value)

    def test_each_plane_length_must_match_camera_dimensions(self):
        for plane in _planes():
            for data in (plane.data[:-1], plane.data + b"\x00"):
                with self.subTest(channel=plane.channel, actual_bytes=len(data)):
                    with self.assertRaisesRegex(ValueError, plane.channel.value):
                        _capture((ImagePlane(plane.channel, data),))

    def test_depth_rejects_nonpositive_or_nan_samples(self):
        for value in (0.0, -0.0, -0.25, -inf, nan):
            plane = ImagePlane(Channel.FORWARD_DEPTH, pack("<6f", 1.0, value, 1, 1, 1, 1))
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "pixel 1"):
                    _capture((plane,))

    def test_depth_positive_infinity_is_the_only_miss_value(self):
        plane = ImagePlane(Channel.FORWARD_DEPTH, pack("<6f", inf, 0.25, 1, inf, 2, 3))
        capture = _capture((plane,))
        self.assertEqual(capture.planes, (plane,))

    def test_valid_rejects_non_binary_samples(self):
        for value in (2, 127, 255):
            plane = ImagePlane(Channel.VALID, bytes((1, value, 0, 1, 1, 1)))
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "pixel 1"):
                    _capture((plane,))

    def test_pairwise_background_mismatches_are_rejected_without_third_channel(self):
        depth = ImagePlane(Channel.FORWARD_DEPTH, pack("<2f", 1.0, inf))
        identity = ImagePlane(Channel.INSTANCE_ID, pack("<2I", 9, 0))
        cases = (
            ((depth, ImagePlane(Channel.VALID, b"\x00\x00")), 0),
            ((depth, ImagePlane(Channel.VALID, b"\x01\x01")), 1),
            ((identity, ImagePlane(Channel.VALID, b"\x00\x00")), 0),
            ((identity, ImagePlane(Channel.VALID, b"\x01\x01")), 1),
            ((depth, ImagePlane(Channel.INSTANCE_ID, pack("<2I", 0, 0))), 0),
            ((depth, ImagePlane(Channel.INSTANCE_ID, pack("<2I", 9, 8))), 1),
        )
        for planes, pixel in cases:
            with self.subTest(channels=tuple(p.channel for p in planes), pixel=pixel):
                with self.assertRaisesRegex(ValueError, f"pixel {pixel}"):
                    _capture(planes, camera=_camera(2, 1))

    def test_background_rgb_and_foreground_black_do_not_define_validity(self):
        capture = _capture()
        self.assertNotEqual(capture.planes[0].data[6:9], b"\x00\x00\x00")
        self.assertEqual(capture.planes[0].data[15:18], b"\x00\x00\x00")
        self.assertEqual(capture.planes[3].data, b"\x01\x01\x00\x01\x01\x01")

    def test_validation_does_not_expand_full_sensor_arrays(self):
        count = 256 * 256
        planes = (
            ImagePlane(Channel.FORWARD_DEPTH, pack("<f", 1.0) * count),
            ImagePlane(Channel.INSTANCE_ID, pack("<I", 256) * count),
            ImagePlane(Channel.VALID, b"\x01" * count),
        )
        camera = _camera(256, 256)
        tracemalloc.start()
        try:
            capture = _capture(planes, camera=camera)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(capture.planes, planes)
        # Inputs are allocated before tracing. Whole-image tuple/list expansion
        # of even one float32 channel exceeds this auxiliary-memory allowance.
        self.assertLess(peak, 256 * 1024)


if __name__ == "__main__":
    unittest.main()
