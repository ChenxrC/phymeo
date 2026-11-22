"""Public checked-session contract using a test-only recording delegate."""

import math
import struct
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path

from phymeo_rendering import backends
from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Capture, Channel, ImagePlane
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene


def fixture():
    scene = VisualScene(
        "table",
        (Mesh("box", "box.obj"),),
        (Material("paint", (0.3, 0.2, 0.1), 0.0, 0.5),),
        (Instance(7, "box", "paint", "box-node"),),
    )
    key = FrameKey("table", "world-0", 0, 2, 3, "accepted_endpoint")
    frame = SceneFrame(key, 0.125, (NodePose("box-node", Pose()),))
    camera = Camera("front", key, Pose(), 2, 1, 2.0, 2.0, 1.0, 0.5, 0.1, 5.0)
    request = CaptureRequest(camera, RenderPath.RASTER, tuple(Channel))
    return scene, frame, request


def packet(frame, request, *, instance=7):
    values = {
        Channel.RGB: bytes((10, 20, 30, 40, 50, 60)),
        Channel.FORWARD_DEPTH: struct.pack("<ff", math.inf, 1.25),
        Channel.INSTANCE_ID: struct.pack("<II", 0, instance),
        Channel.VALID: bytes((0, 1)),
    }
    return Capture(
        frame.key,
        frame.time_s,
        request.camera,
        tuple(ImagePlane(channel, values[channel]) for channel in request.channels),
    )


class Recorder:
    """Never installed in the product package and never a rendering capability proof."""

    def __init__(self):
        self.requests = []
        self.captures = []
        self.close_calls = 0
        self.answer = True
        self.make_packet = packet
        self.error = None
        self.close_error = None

    def supports(self, request):
        self.requests.append(request)
        return self.answer

    def capture(self, frame, request):
        self.captures.append((frame, request))
        if self.error is not None:
            raise self.error
        return self.make_packet(frame, request)

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class SessionContracts(unittest.TestCase):
    def setUp(self):
        self.scene, self.frame, self.request = fixture()
        self.delegate = Recorder()
        self.session = backends.ValidatedRenderSession(self.scene, self.delegate)

    def test_complete_request_is_forwarded_without_rewriting(self):
        self.assertTrue(self.session.supports(self.request))
        expected = packet(self.frame, self.request)
        self.delegate.make_packet = lambda frame, request: expected
        result = self.session.capture(self.frame, self.request)
        self.assertIs(result, expected)
        self.assertEqual(len(self.delegate.requests), 2)
        self.assertTrue(all(item is self.request for item in self.delegate.requests))
        self.assertIs(self.delegate.captures[0][0], self.frame)
        self.assertIs(self.delegate.captures[0][1], self.request)
        self.assertEqual(result, packet(self.frame, self.request))

    def test_capture_rechecks_full_combination_after_supports(self):
        self.assertTrue(self.session.supports(self.request))
        self.delegate.answer = False
        with self.assertRaises(backends.UnsupportedRequestError):
            self.session.capture(self.frame, self.request)
        self.assertEqual(self.delegate.captures, [])

    def test_no_path_or_channel_fallback(self):
        self.delegate.answer = False
        request = replace(
            self.request, path=RenderPath.PATH_TRACE, channels=(Channel.FORWARD_DEPTH, Channel.RGB)
        )
        self.assertFalse(self.session.supports(request))
        with self.assertRaises(backends.UnsupportedRequestError):
            self.session.capture(self.frame, request)
        self.assertEqual(self.delegate.requests, [request, request])
        self.assertEqual(self.delegate.captures, [])

    def test_supports_must_return_bool_not_truthy_value(self):
        for value in (1, "yes", None, [], object()):
            with self.subTest(value=value):
                self.delegate.answer = value
                with self.assertRaises(TypeError):
                    self.session.supports(self.request)
                with self.assertRaises(TypeError):
                    self.session.capture(self.frame, self.request)
        self.assertEqual(self.delegate.captures, [])

    def test_request_validation_precedes_delegate(self):
        for make_request, error in (
            (lambda: object(), TypeError),
            (lambda: replace(self.request, channels=()), ValueError),
            (lambda: replace(self.request, channels=(Channel.RGB, Channel.RGB)), ValueError),
            (lambda: replace(self.request, path="raster"), TypeError),
        ):
            with self.subTest(make_request=make_request):
                with self.assertRaises(error):
                    self.session.supports(make_request())
                with self.assertRaises(error):
                    self.session.capture(self.frame, make_request())
        self.assertEqual(self.delegate.requests, [])
        self.assertEqual(self.delegate.captures, [])

    def test_supports_checks_open_scene_identity(self):
        bad_key = replace(self.frame.key, scene_key="other")
        bad = replace(self.request, camera=replace(self.request.camera, frame_key=bad_key))
        with self.assertRaisesRegex(ValueError, "scene"):
            self.session.supports(bad)
        self.assertEqual(self.delegate.requests, [])

    def test_each_exposure_key_component_is_checked(self):
        changes = dict(
            scene_key="other", world_id="world-1", epoch=1, revision=3, tick=4, phase="step_input"
        )
        for field, value in changes.items():
            request = replace(
                self.request,
                camera=replace(
                    self.request.camera, frame_key=replace(self.frame.key, **{field: value})
                ),
            )
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                self.session.capture(self.frame, request)
        self.assertEqual(self.delegate.requests, [])
        self.assertEqual(self.delegate.captures, [])

    def test_matching_wrong_scene_frame_still_rejected(self):
        frame = replace(self.frame, key=replace(self.frame.key, scene_key="other"))
        request = replace(self.request, camera=replace(self.request.camera, frame_key=frame.key))
        with self.assertRaisesRegex(ValueError, "scene"):
            self.session.capture(frame, request)
        self.assertEqual(self.delegate.captures, [])

    def test_missing_required_node_is_rejected(self):
        frame = replace(self.frame, nodes=(NodePose("camera-mount", Pose()),))
        with self.assertRaisesRegex(ValueError, "box-node"):
            self.session.capture(frame, self.request)
        self.assertEqual(self.delegate.captures, [])

    def test_extra_mount_nodes_and_replay_are_legal(self):
        for tick, epoch, time in ((8, 0, 0.5), (1, 0, 0.05), (0, 1, 0.0), (0, 1, 0.0)):
            frame = replace(
                self.frame,
                key=replace(self.frame.key, tick=tick, epoch=epoch),
                time_s=time,
                nodes=self.frame.nodes + (NodePose("mount", Pose()),),
            )
            request = replace(
                self.request, camera=replace(self.request.camera, frame_key=frame.key)
            )
            result = self.session.capture(frame, request)
            self.assertEqual((result.frame_key, result.time_s), (frame.key, time))
        self.assertEqual(len(self.delegate.captures), 4)

    def test_result_type_is_checked(self):
        self.delegate.make_packet = lambda frame, request: object()
        with self.assertRaises(TypeError):
            self.session.capture(self.frame, self.request)

    def test_result_key_and_exact_time_are_bound_to_call(self):
        valid = packet(self.frame, self.request)
        for field, value in dict(
            scene_key="other", world_id="world-1", epoch=1, revision=3, tick=4, phase="step_input"
        ).items():
            wrong_key = replace(valid.frame_key, **{field: value})
            wrong_camera = replace(valid.camera, frame_key=wrong_key)
            bad = replace(valid, frame_key=wrong_key, camera=wrong_camera)
            self.delegate.make_packet = lambda frame, request, bad=bad: bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.session.capture(self.frame, self.request)
        for time in (math.nextafter(valid.time_s, math.inf), valid.time_s + 1):
            bad = replace(valid, time_s=time)
            self.delegate.make_packet = lambda frame, request, bad=bad: bad
            with self.assertRaisesRegex(ValueError, "time"):
                self.session.capture(self.frame, self.request)

    def test_result_camera_full_value_is_bound_to_request(self):
        valid = packet(self.frame, self.request)
        for field, value in (
            ("id", "other"),
            ("world_pose", Pose((0, 0, 1))),
            ("fx_px", 3.0),
            ("fy_px", 3.0),
            ("cx_px", 1.5),
            ("cy_px", 0.75),
            ("near_m", 0.2),
            ("far_m", 6.0),
        ):
            bad = replace(valid, camera=replace(valid.camera, **{field: value}))
            self.delegate.make_packet = lambda frame, request, bad=bad: bad
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "camera"):
                self.session.capture(self.frame, self.request)

    def test_result_channels_match_requested_order_exactly(self):
        valid = packet(self.frame, self.request)
        for planes in (valid.planes[:-1], valid.planes[::-1], valid.planes + (valid.planes[0],)):
            self.delegate.make_packet = lambda frame, request, planes=planes: replace(
                valid, planes=planes
            )
            with self.subTest(planes=planes), self.assertRaises(ValueError):
                self.session.capture(self.frame, self.request)
        rgb_only = replace(self.request, channels=(Channel.RGB,))
        self.delegate.make_packet = lambda frame, request: valid
        with self.assertRaises(ValueError):
            self.session.capture(self.frame, rgb_only)

    def test_scene_instance_ids_checked_including_id_only_request(self):
        request = replace(self.request, channels=(Channel.INSTANCE_ID,))
        self.delegate.make_packet = lambda frame, request: packet(frame, request, instance=99)
        with self.assertRaisesRegex(ValueError, "99"):
            self.session.capture(self.frame, request)
        self.delegate.make_packet = packet
        self.assertEqual(
            self.session.capture(self.frame, request).planes[0].data, struct.pack("<II", 0, 7)
        )

    def test_valid_background_and_maximum_scene_id_are_allowed(self):
        other = replace(self.scene, instances=(replace(self.scene.instances[0], id=2**32 - 1),))
        delegate = Recorder()
        delegate.make_packet = lambda frame, request: packet(frame, request, instance=2**32 - 1)
        session = backends.ValidatedRenderSession(other, delegate)
        self.assertEqual(
            session.capture(self.frame, self.request).planes[2].data,
            struct.pack("<II", 0, 2**32 - 1),
        )
        session.close()

    def test_delegate_output_constructor_rejects_bad_layout_and_mutability(self):
        valid = packet(self.frame, self.request)
        for data, error in (
            (bytearray(6), TypeError),
            (memoryview(bytes(6)), TypeError),
            (bytes(5), ValueError),
        ):

            def make_bad(frame, request, data=data):
                plane = ImagePlane(Channel.RGB, data)
                return replace(valid, planes=(plane,) + valid.planes[1:])

            self.delegate.make_packet = make_bad
            before = len(self.delegate.captures)
            with self.subTest(data=data), self.assertRaises(error):
                self.session.capture(self.frame, self.request)
            self.assertEqual(len(self.delegate.captures), before + 1)
        self.delegate.make_packet = lambda frame, request: replace(valid, planes=list(valid.planes))
        with self.assertRaises(TypeError):
            self.session.capture(self.frame, self.request)

    def test_delegate_output_constructor_rejects_inconsistent_background(self):
        valid = packet(self.frame, self.request)

        def make_bad(frame, request):
            plane = ImagePlane(Channel.VALID, bytes((1, 1)))
            return replace(valid, planes=valid.planes[:3] + (plane,))

        self.delegate.make_packet = make_bad
        with self.assertRaises(ValueError):
            self.session.capture(self.frame, self.request)
        self.assertEqual(len(self.delegate.captures), 1)

    def test_delegate_errors_propagate_and_capture_may_be_retried(self):
        error = RuntimeError("device readback failed")
        self.delegate.error = error
        with self.assertRaises(RuntimeError) as found:
            self.session.capture(self.frame, self.request)
        self.assertIs(found.exception, error)
        self.delegate.error = None
        self.assertEqual(
            self.session.capture(self.frame, self.request), packet(self.frame, self.request)
        )

    def test_prior_owned_bytes_survive_later_capture_and_close(self):
        first = self.session.capture(self.frame, self.request)
        original = tuple(plane.data for plane in first.planes)
        self.session.capture(self.frame, self.request)
        self.session.close()
        self.assertEqual(tuple(plane.data for plane in first.planes), original)
        self.assertTrue(all(type(plane.data) is bytes for plane in first.planes))

    def test_close_is_idempotent_and_blocks_all_later_calls(self):
        self.session.close()
        self.session.close()
        self.assertEqual(self.delegate.close_calls, 1)
        with self.assertRaises(backends.SessionClosedError):
            self.session.supports(self.request)
        with self.assertRaises(backends.SessionClosedError):
            self.session.capture(self.frame, self.request)
        self.assertEqual(self.delegate.requests, [])

    def test_close_failure_remains_visible_without_unsafe_retry(self):
        error = RuntimeError("device release unknown")
        self.delegate.close_error = error
        for _ in range(2):
            with self.assertRaises(RuntimeError) as found:
                self.session.close()
            self.assertIs(found.exception, error)
        self.assertEqual(self.delegate.close_calls, 1)
        with self.assertRaises(backends.SessionClosedError):
            self.session.capture(self.frame, self.request)


class BackendAvailability(unittest.TestCase):
    def test_invalid_session_constructor_does_not_claim_delegate_ownership(self):
        scene, _, _ = fixture()
        delegate = Recorder()
        with self.assertRaises(TypeError):
            backends.ValidatedRenderSession(object(), delegate)
        self.assertEqual(delegate.close_calls, 0)
        for broken in (object(), None):
            with self.assertRaises(TypeError):
                backends.ValidatedRenderSession(scene, broken)

    def test_importing_backend_entry_points_and_configuring_them_needs_no_device(self):
        code = """
import sys
sys.path.insert(0, sys.argv[1])
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.backends.sapien import SapienBackend
assert callable(NativeBackend().open_scene) and callable(SapienBackend().open_scene)
assert {'sapien', 'vulkan', 'mujoco', 'numpy', 'torch', 'phymeo'}.isdisjoint(sys.modules)
"""
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                code,
                str(Path(backends.__file__).resolve().parents[2]),
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
