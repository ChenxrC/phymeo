"""Exact default stage sequences and request validation, without a device."""

import sys
import unittest
from dataclasses import replace

from phymeo_rendering import pipeline
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, NativePass, RenderPath
from phymeo_rendering.scene import FrameKey, Pose


def _request(**changes):
    camera = Camera(
        "front",
        FrameKey("r0-table", "world-0", 0, 1, 1, "authored"),
        Pose(),
        640,
        480,
        500.0,
        500.0,
        320.0,
        240.0,
        0.1,
        5.0,
    )
    values = dict(camera=camera, path=RenderPath.RASTER, channels=(Channel.RGB,))
    values.update(changes)
    return CaptureRequest(**values)


RGB, DEPTH, INSTANCE, VALID = Channel.RGB, Channel.FORWARD_DEPTH, Channel.INSTANCE_ID, Channel.VALID
RASTER_RGB = (
    NativePass.VISIBILITY,
    NativePass.SHADOWS,
    NativePass.LIGHTING,
    NativePass.DISPLAY_TRANSFORM,
)
RASTER_SENSOR = (NativePass.VISIBILITY, NativePass.SENSOR_EXPORT)
RASTER_MIXED = (
    NativePass.VISIBILITY,
    NativePass.SHADOWS,
    NativePass.LIGHTING,
    NativePass.DISPLAY_TRANSFORM,
    NativePass.SENSOR_EXPORT,
)
TRACE_RGB = (NativePass.TRACE, NativePass.DISPLAY_TRANSFORM)
TRACE_SENSOR = (NativePass.TRACE, NativePass.SENSOR_EXPORT)
TRACE_MIXED = (NativePass.TRACE, NativePass.DISPLAY_TRANSFORM, NativePass.SENSOR_EXPORT)

# Every nonempty subset is listed explicitly; expectations do not call a second
# implementation of the planner or derive themselves from its output.
CASES = (
    ((RGB,), RASTER_RGB, TRACE_RGB),
    ((DEPTH,), RASTER_SENSOR, TRACE_SENSOR),
    ((INSTANCE,), RASTER_SENSOR, TRACE_SENSOR),
    ((VALID,), RASTER_SENSOR, TRACE_SENSOR),
    ((DEPTH, INSTANCE), RASTER_SENSOR, TRACE_SENSOR),
    ((DEPTH, VALID), RASTER_SENSOR, TRACE_SENSOR),
    ((INSTANCE, VALID), RASTER_SENSOR, TRACE_SENSOR),
    ((DEPTH, INSTANCE, VALID), RASTER_SENSOR, TRACE_SENSOR),
    ((RGB, DEPTH), RASTER_MIXED, TRACE_MIXED),
    ((RGB, INSTANCE), RASTER_MIXED, TRACE_MIXED),
    ((RGB, VALID), RASTER_MIXED, TRACE_MIXED),
    ((RGB, DEPTH, INSTANCE), RASTER_MIXED, TRACE_MIXED),
    ((RGB, DEPTH, VALID), RASTER_MIXED, TRACE_MIXED),
    ((RGB, INSTANCE, VALID), RASTER_MIXED, TRACE_MIXED),
    ((RGB, DEPTH, INSTANCE, VALID), RASTER_MIXED, TRACE_MIXED),
)


class CaptureRequestTests(unittest.TestCase):
    def test_four_channel_request_is_accepted_without_changes(self):
        request = _request(channels=(VALID, INSTANCE, DEPTH, RGB), path=RenderPath.PATH_TRACE)
        before = replace(request)
        self.assertIsNone(pipeline.validate_capture_request(request))
        self.assertEqual(request, before)
        self.assertEqual(request.channels, (VALID, INSTANCE, DEPTH, RGB))

    def test_validator_rejects_non_request_inputs(self):
        for value in (None, {}, "raster"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "request"):
                    pipeline.validate_capture_request(value)

    def test_validator_rejects_invalid_field_types(self):
        cases = (
            ("camera", None),
            ("camera", {}),
            ("camera", "front"),
            ("path", "raster"),
            ("path", None),
            ("path", NativePass.TRACE),
            ("channels", [RGB]),
            ("channels", iter((RGB,))),
            ("channels", None),
        )
        for field, value in cases:
            with self.subTest(field=field, value_type=type(value).__name__):
                with self.assertRaisesRegex(TypeError, field):
                    pipeline.validate_capture_request(_request(**{field: value}))

    def test_validator_rejects_non_channel_members(self):
        for value in ("srgb8_rgb", None, 0, RenderPath.RASTER):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "channels"):
                    pipeline.validate_capture_request(_request(channels=(RGB, value)))

    def test_validator_rejects_empty_or_duplicate_channels(self):
        with self.assertRaisesRegex(ValueError, "channels"):
            pipeline.validate_capture_request(_request(channels=()))
        for channel in (RGB, DEPTH, INSTANCE, VALID):
            with self.subTest(channel=channel):
                with self.assertRaisesRegex(ValueError, "channels"):
                    pipeline.validate_capture_request(_request(channels=(channel, channel)))


class DefaultPassPlannerTests(unittest.TestCase):
    def test_all_thirty_combinations_match_exact_stage_table(self):
        planner = pipeline.DefaultPassPlanner()
        count = 0
        for channels, raster, trace in CASES:
            for path, expected in ((RenderPath.RASTER, raster), (RenderPath.PATH_TRACE, trace)):
                request = _request(path=path, channels=channels)
                before = replace(request)
                with self.subTest(path=path, channels=channels):
                    for _ in range(100):
                        self.assertEqual(planner.plan(request), expected)
                    self.assertEqual(request, before)
                    self.assertEqual(len(expected), len(set(expected)))
                count += 1
        self.assertEqual(count, 30)

    def test_reversed_channel_order_does_not_reorder_request_or_stages(self):
        planner = pipeline.DefaultPassPlanner()
        for channels, raster, trace in CASES:
            for path, expected in ((RenderPath.RASTER, raster), (RenderPath.PATH_TRACE, trace)):
                ordered = tuple(reversed(channels))
                request = _request(path=path, channels=ordered)
                with self.subTest(path=path, channels=ordered):
                    self.assertEqual(planner.plan(request), expected)
                    self.assertEqual(request.channels, ordered)

    def test_planner_rejects_requests_before_returning_any_plan(self):
        planner = pipeline.DefaultPassPlanner()
        cases = (
            (None, TypeError, "request"),
            (_request(camera=None), TypeError, "camera"),
            (_request(path="raster"), TypeError, "path"),
            (_request(channels=[RGB]), TypeError, "channels"),
            (_request(channels=(RGB, "depth")), TypeError, "channels"),
            (_request(channels=()), ValueError, "channels"),
            (_request(channels=(DEPTH, DEPTH)), ValueError, "channels"),
        )
        for request, exception, field in cases:
            with self.subTest(field=field, exception=exception):
                with self.assertRaisesRegex(exception, field):
                    planner.plan(request)

    def test_planning_does_not_import_a_physics_or_graphics_runtime(self):
        before = set(sys.modules)
        planner = pipeline.DefaultPassPlanner()
        for path in (RenderPath.RASTER, RenderPath.PATH_TRACE):
            request = _request(path=path, channels=(RGB, DEPTH, INSTANCE, VALID))
            self.assertIsInstance(planner.plan(request), tuple)
        imported_roots = {name.partition(".")[0] for name in set(sys.modules) - before}
        self.assertFalse(imported_roots & {"phymeo", "sapien", "vulkan", "numpy", "tyx"})


if __name__ == "__main__":
    unittest.main()
