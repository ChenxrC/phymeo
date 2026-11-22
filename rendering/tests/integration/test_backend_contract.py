"""Exercise the capture call chain using a test-only readback recorder."""

import struct
import unittest
from dataclasses import replace

from phymeo_rendering.adapters import AuthoredSceneSource
from phymeo_rendering.backends import (
    SessionClosedError,
    UnsupportedRequestError,
    ValidatedRenderSession,
)
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Capture, Channel, ImagePlane
from phymeo_rendering.pipeline import CaptureRequest, DefaultPassPlanner, NativePass, RenderPath
from phymeo_rendering.scene import compose_pose

from ._g1_table import CAMERA_MOUNT, table_frames, table_scene


class ReadbackRecorder:
    """Two artificial pixels mark ordering, frame identity and ownership."""

    def __init__(self):
        self.stages = []
        self.closed = False

    def supports(self, request):
        return request.path is RenderPath.RASTER and (
            request.camera.width,
            request.camera.height,
        ) == (2, 1)

    def capture(self, frame, request):
        self.stages.append(DefaultPassPlanner().plan(request))
        sample_id = (11, 12)[frame.key.tick]
        data = {
            Channel.RGB: bytes((sample_id, 0, 0, 0, 0, 0)),
            Channel.FORWARD_DEPTH: struct.pack("<2f", 1.0, float("inf")),
            Channel.INSTANCE_ID: struct.pack("<2I", sample_id, 0),
            Channel.VALID: bytes((1, 0)),
        }
        planes = tuple(ImagePlane(channel, data[channel]) for channel in request.channels)
        return Capture(frame.key, frame.time_s, request.camera, planes)

    def close(self):
        self.closed = True


class RecordingBackend:
    def __init__(self):
        self.reader = ReadbackRecorder()

    def open_scene(self, scene):
        return ValidatedRenderSession(scene, self.reader)


def wrist_exposure(frame):
    wrist = next(node.world_pose for node in frame.nodes if node.node_id == "wrist")
    return Camera("wrist", frame.key, compose_pose(wrist, CAMERA_MOUNT), 2, 1, 1, 1, 1, 0.5, 0.1, 5)


class TableCaptureTest(unittest.TestCase):
    def setUp(self):
        self.source = AuthoredSceneSource(table_scene(), table_frames())
        self.backend = RecordingBackend()
        self.session = self.backend.open_scene(self.source.scene())
        self.addCleanup(self.session.close)

    def test_two_wrist_exposures_publish_requested_planes_in_order(self):
        channels = (Channel.INSTANCE_ID, Channel.VALID, Channel.FORWARD_DEPTH, Channel.RGB)
        observations = []
        for frame in self.source.frames():
            request = CaptureRequest(wrist_exposure(frame), RenderPath.RASTER, channels)
            self.assertTrue(self.session.supports(request))
            observation = self.session.capture(frame, request)
            self.assertEqual((observation.frame_key, observation.time_s), (frame.key, frame.time_s))
            self.assertEqual(observation.camera, request.camera)
            self.assertEqual(tuple(p.channel for p in observation.planes), channels)
            observations.append(observation)
        self.session.close()
        self.assertTrue(self.backend.reader.closed)
        self.assertEqual(
            self.backend.reader.stages,
            [
                (
                    NativePass.VISIBILITY,
                    NativePass.SHADOWS,
                    NativePass.LIGHTING,
                    NativePass.DISPLAY_TRANSFORM,
                    NativePass.SENSOR_EXPORT,
                )
            ]
            * 2,
        )
        self.assertEqual(len(observations), 2)
        for observation, sample_id in zip(observations, (11, 12), strict=True):
            labels, mask, depth, rgb = (plane.data for plane in observation.planes)
            self.assertEqual(struct.unpack("<2I", labels), (sample_id, 0))
            self.assertEqual(tuple(mask), (1, 0))
            self.assertEqual(struct.unpack("<2f", depth), (1.0, float("inf")))
            self.assertEqual(rgb, bytes((sample_id, 0, 0, 0, 0, 0)))

    def test_rejected_exposures_never_reach_the_readback(self):
        first, second = self.source.frames()
        request = CaptureRequest(wrist_exposure(first), RenderPath.RASTER, (Channel.INSTANCE_ID,))
        with self.assertRaises(ValueError):
            self.session.capture(second, request)
        trace = replace(request, path=RenderPath.PATH_TRACE)
        self.assertFalse(self.session.supports(trace))
        with self.assertRaises(UnsupportedRequestError):
            self.session.capture(first, trace)
        result = self.session.capture(first, request)
        self.assertEqual(tuple(p.channel for p in result.planes), (Channel.INSTANCE_ID,))
        self.assertEqual(
            self.backend.reader.stages, [(NativePass.VISIBILITY, NativePass.SENSOR_EXPORT)]
        )
        self.session.close()
        with self.assertRaises(SessionClosedError):
            self.session.capture(first, request)

    def test_different_open_scenes_keep_their_own_frames_and_lifetime(self):
        first, second = self.source.frames()
        other_scene = replace(self.source.scene(), key="other-table")
        other_frame = replace(
            second, key=replace(second.key, scene_key=other_scene.key, world_id="B")
        )
        other_source = AuthoredSceneSource(other_scene, (other_frame,))
        other_backend = RecordingBackend()
        other = other_backend.open_scene(other_source.scene())
        self.addCleanup(other.close)
        request_a = CaptureRequest(wrist_exposure(first), RenderPath.RASTER, (Channel.INSTANCE_ID,))
        request_b = replace(request_a, camera=wrist_exposure(next(other_source.frames())))

        saved_a = self.session.capture(first, request_a)
        with self.assertRaises(ValueError):
            self.session.capture(other_frame, request_b)
        self.session.close()
        saved_b = other.capture(other_frame, request_b)
        self.assertEqual((saved_a.frame_key.world_id, saved_b.frame_key.world_id), ("table", "B"))
        self.assertEqual(struct.unpack("<2I", saved_a.planes[0].data), (11, 0))
        self.assertEqual(struct.unpack("<2I", saved_b.planes[0].data), (12, 0))
        self.assertEqual(len(self.backend.reader.stages), 1)
        self.assertEqual(len(other_backend.reader.stages), 1)
