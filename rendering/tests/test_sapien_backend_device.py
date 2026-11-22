"""Opt-in real SAPIEN 3.0.3 device tests, never replaced by test doubles.

Set PHYMEO_SAPIEN_DEVICE=1 to opt in.
A missing runtime/device after opt-in is an error, not a skip.
"""

import json
import math
import os
import struct
import unittest
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from integration._g1_table import CAMERA_MOUNT, table_frames, table_scene
from phymeo_rendering.backends import SessionClosedError, UnsupportedRequestError
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import NodePose, Pose, compose_pose

ENABLED = os.environ.get("PHYMEO_SAPIEN_DEVICE") == "1"


def _camera(frame, *, wrist=False):
    pose = Pose((0.0, 0.0, 1.0625))
    if wrist:
        node = next(n for n in frame.nodes if n.node_id == "wrist")
        pose = compose_pose(node.world_pose, CAMERA_MOUNT)
    return Camera("g3-reference", frame.key, pose, 160, 120, 125.0, 125.0, 80.0, 60.0, 0.1, 5.0)


def _pixel(capture, channel, u, v):
    data = next(p.data for p in capture.planes if p.channel is channel)
    offset = v * capture.camera.width + u
    if channel is Channel.INSTANCE_ID:
        return struct.unpack_from("<I", data, 4 * offset)[0]
    if channel is Channel.FORWARD_DEPTH:
        return struct.unpack_from("<f", data, 4 * offset)[0]
    return data[offset]


@unittest.skipUnless(ENABLED, "explicit real SAPIEN device opt-in required")
class SapienApiSmokeTest(unittest.TestCase):
    def test_real_pure_render_paths(self):
        """Can be run before the adapter exists; actual shipped shader compile/readback."""
        import numpy as np
        import sapien

        self.assertEqual(sapien.__version__, "3.0.3")
        records = []
        for shader, target in (("default", "Lighting"), ("rt", "HdrColor")):
            with self.subTest(shader=shader):
                system = sapien.render.RenderSystem("cuda:0")
                scene = sapien.Scene([system])
                entity = sapien.Entity()
                camera = sapien.render.RenderCameraComponent(
                    16, 12, str(Path(sapien.__file__).parent / "vulkan_shader" / shader)
                )
                entity.add_component(camera)
                previous = sapien.render.get_ray_tracing_denoiser()
                try:
                    sapien.render.set_ray_tracing_denoiser("none")
                    scene.add_entity(entity)
                finally:
                    sapien.render.set_ray_tracing_denoiser(previous)
                try:
                    if shader == "rt":
                        camera.set_property("spp", 4)
                        camera.set_property("maxDepth", 2)
                    scene.update_render()
                    camera.take_picture()
                    names = camera.get_picture_names()
                    self.assertIn(target, names)
                    pixels = camera.get_picture(target)
                    self.assertEqual(pixels.shape, (12, 16, 4))
                    self.assertTrue(np.isfinite(pixels).all())
                    records.append(
                        {
                            "shader": shader,
                            "device": system.device.name,
                            "pci": system.device.pci_string,
                            "targets": names,
                        }
                    )
                finally:
                    scene.clear()
        print("SAPIEN_REAL_SMOKE=" + json.dumps(records))


@unittest.skipUnless(ENABLED, "explicit real SAPIEN device opt-in required")
class SapienBackendDeviceTest(unittest.TestCase):
    def _open(self, path):
        from phymeo_rendering.backends.sapien import SapienBackend, SapienConfig

        backend = SapienBackend(SapienConfig(path=path, device="cuda:0"))
        session = backend.open_scene(table_scene())
        self.addCleanup(session.close)
        return backend, session

    def test_raster_metric_depth_ids_background_and_owned_storage(self):
        backend, session = self._open(RenderPath.RASTER)
        first, second = table_frames()
        request = CaptureRequest(_camera(first), RenderPath.RASTER, tuple(Channel))
        self.assertTrue(session.supports(request))
        capture = session.capture(first, request)
        # Plastic block top z=.0625 and camera z=1.0625 => forward depth 1 m.
        for u, v in ((80, 60), (82, 58)):
            self.assertEqual(_pixel(capture, Channel.INSTANCE_ID, u, v), 12)
            self.assertAlmostEqual(_pixel(capture, Channel.FORWARD_DEPTH, u, v), 1.0, delta=2e-5)
            self.assertEqual(_pixel(capture, Channel.VALID, u, v), 1)
        self.assertEqual(_pixel(capture, Channel.INSTANCE_ID, 0, 0), 0)
        self.assertTrue(math.isinf(_pixel(capture, Channel.FORWARD_DEPTH, 0, 0)))
        self.assertEqual(_pixel(capture, Channel.VALID, 0, 0), 0)
        original = tuple(p.data for p in capture.planes)
        moved = replace(
            second,
            nodes=tuple(
                NodePose(n.node_id, Pose((2.0, 0.0, 0.0))) if n.node_id == "base" else n
                for n in second.nodes
            ),
        )
        other = session.capture(moved, replace(request, camera=_camera(moved)))
        self.assertEqual(_pixel(other, Channel.INSTANCE_ID, 80, 60), 0)
        replay = session.capture(first, request)
        self.assertEqual(tuple(p.data for p in replay.planes), original)
        session.close()
        self.assertEqual(tuple(p.data for p in capture.planes), original)
        with self.assertRaises(SessionClosedError):
            session.supports(request)
        self.assertEqual(backend.last_open_diagnostics["physics_systems"], [])

    def test_rotated_wrist_and_world_reset_preserve_calibration(self):
        _, session = self._open(RenderPath.RASTER)
        first, second = table_frames()
        for frame, expected_pixel in ((first, (49, 60)), (second, (80, 13))):
            request = CaptureRequest(
                _camera(frame, wrist=True),
                RenderPath.RASTER,
                (Channel.INSTANCE_ID, Channel.FORWARD_DEPTH),
            )
            capture = session.capture(frame, request)
            self.assertEqual(_pixel(capture, Channel.INSTANCE_ID, *expected_pixel), 11)
            self.assertAlmostEqual(
                _pixel(capture, Channel.FORWARD_DEPTH, *expected_pixel), 1.0, delta=2e-5
            )
        foreign = replace(first, key=replace(first.key, world_id="another-world", epoch=4))
        result = session.capture(
            foreign, CaptureRequest(_camera(foreign), RenderPath.RASTER, (Channel.INSTANCE_ID,))
        )
        self.assertEqual(_pixel(result, Channel.INSTANCE_ID, 80, 60), 12)

    def test_close_observation_is_per_session_and_counts_real_attachments(self):
        _, first = self._open(RenderPath.RASTER)
        _, second = self._open(RenderPath.RASTER)
        frame = table_frames()[0]
        request = CaptureRequest(_camera(frame), RenderPath.RASTER, (Channel.RGB,))
        first.capture(frame, request)
        before = first.diagnostics()
        self.assertEqual(before["lifecycle"]["completed_captures"], 1)
        self.assertGreater(before["lifecycle"]["attached_entities"], 0)
        self.assertEqual(before["lifecycle"]["attached_cameras"], 0)
        first.close()
        closed = first.diagnostics()
        self.assertTrue(closed["lifecycle"]["closed"])
        self.assertEqual(closed["lifecycle"]["attached_entities_after_clear"], 0)
        self.assertEqual(closed["lifecycle"]["attached_cameras_after_clear"], 0)
        self.assertEqual(closed["lifecycle"]["attached_render_bodies_after_clear"], 0)
        live = second.diagnostics()
        self.assertFalse(live["lifecycle"]["closed"])
        self.assertGreater(live["lifecycle"]["attached_entities"], 0)
        second.capture(frame, request)
        second.close()
        # The detached old dictionary and the other session remain independent.
        self.assertFalse(before["lifecycle"]["closed"])
        self.assertEqual(first.diagnostics(), closed)

    def test_pt_rejects_fp32_clip_slivers_and_large_coordinate_loss(self):
        _, session = self._open(RenderPath.PATH_TRACE)
        frame = table_frames()[0]
        request = CaptureRequest(_camera(frame), RenderPath.PATH_TRACE, (Channel.RGB,))
        # Table depth is exactly 1.125 m. This legal FP64 far plane rounds
        # back onto that plane at the renderer's FP32 API boundary.
        far = 1.125 + 2.0**-30
        self.assertGreater(far, 1.125)
        self.assertEqual(struct.unpack("<f", struct.pack("<f", far))[0], 1.125)
        with self.assertRaisesRegex(UnsupportedRequestError, "clip"):
            session.capture(frame, replace(request, camera=replace(request.camera, far_m=far)))
        # Normal public values can remain finite while FP32 destroys a metre
        # scale scene's offsets. A matched common translation must be refused,
        # not silently collapse the scene and publish a dark picture.
        shift = 2.0**30
        shifted = replace(
            frame,
            nodes=tuple(
                NodePose(n.node_id, Pose((shift, 0.0, 0.0))) if n.node_id == "base" else n
                for n in frame.nodes
            ),
        )
        huge_camera = replace(request.camera, world_pose=Pose((shift, 0.0, 1.0625)))
        with self.assertRaisesRegex(UnsupportedRequestError, "precision"):
            session.capture(shifted, replace(request, camera=huge_camera))
        nonrepresentable = replace(request.camera, world_pose=Pose((1e100, 0.0, 1.0)))
        self.assertFalse(session.supports(replace(request, camera=nonrepresentable)))
        # Rejection must not change the legal exposure or poison the session.
        capture = session.capture(frame, request)
        self.assertGreater(max(capture.planes[0].data), 10)

    def test_pt_replays_official_fixed_sampling_without_an_arbitrary_seed(self):
        backend, session = self._open(RenderPath.PATH_TRACE)
        frame, next_frame = table_frames()
        request = CaptureRequest(_camera(frame), RenderPath.PATH_TRACE, (Channel.RGB,))
        first = session.capture(frame, request)
        second = session.capture(frame, request)
        displaced = replace(
            next_frame,
            key=replace(next_frame.key, world_id="other"),
            nodes=tuple(
                NodePose(n.node_id, Pose((2.0, 0.0, 0.0))) if n.node_id == "base" else n
                for n in next_frame.nodes
            ),
        )
        different = session.capture(displaced, replace(request, camera=_camera(displaced)))
        replay = session.capture(frame, request)
        _, independent = self._open(RenderPath.PATH_TRACE)
        fresh = independent.capture(frame, request)
        data = first.planes[0].data
        for result in (second, replay, fresh):
            self.assertEqual(result.planes[0].data, data)
        self.assertNotEqual(different.planes[0].data, data)
        pattern = backend.last_open_diagnostics["sampling_pattern"]
        self.assertEqual(pattern["mode"], "official-fixed-sequence-fresh-camera")
        self.assertEqual(pattern["magic_hex"], "0x9e3779b9")
        self.assertFalse(pattern["arbitrary_seed_supported"])
        evidence = {
            "same_device_repeat_srgb8_exact": True,
            "scope": "same installed shader/device/config and authored scene; not a cross-device guarantee",
            "capture_sha256": [
                sha256(c.planes[0].data).hexdigest() for c in (first, second, replay, fresh)
            ],
            "diagnostics": backend.last_open_diagnostics,
        }
        print("SAPIEN_FIXED_SAMPLING=" + json.dumps(evidence))
        artifact = os.environ.get("PHYMEO_SAPIEN_ARTIFACT_DIR")
        if artifact:
            directory = Path(artifact)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "reference-sampling.json").write_text(
                json.dumps(evidence, indent=2), encoding="utf-8"
            )

    def test_path_trace_is_real_rgb_only_and_clip_rejection_is_explicit(self):
        backend, session = self._open(RenderPath.PATH_TRACE)
        frame = table_frames()[0]
        request = CaptureRequest(_camera(frame), RenderPath.PATH_TRACE, (Channel.RGB,))
        self.assertTrue(session.supports(request))
        self.assertFalse(session.supports(replace(request, path=RenderPath.RASTER)))
        with self.assertRaises(UnsupportedRequestError):
            session.capture(frame, replace(request, channels=(Channel.FORWARD_DEPTH,)))
        capture = session.capture(frame, request)
        self.assertGreater(max(capture.planes[0].data), 10)
        self.assertGreater(len(set(capture.planes[0].data)), 8)
        with self.assertRaisesRegex(UnsupportedRequestError, "clip"):
            session.capture(frame, replace(request, camera=replace(request.camera, far_m=0.5)))
        # A fresh camera per exposure ensures a rejected request never publishes
        # a stale picture, and no previous world or camera accumulation is reused.
        recovered = session.capture(frame, request)
        self.assertEqual(recovered.frame_key, frame.key)
        self.assertEqual(backend.last_open_diagnostics["shader_pack"], "rt")
        self.assertEqual(backend.last_open_diagnostics["seed_support"], "not-user-configurable")
        artifact = os.environ.get("PHYMEO_SAPIEN_ARTIFACT_DIR")
        if artifact:
            directory = Path(artifact)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "reference-pt.ppm").write_bytes(
                b"P6\n160 120\n255\n" + capture.planes[0].data
            )
            (directory / "reference-info.json").write_text(
                json.dumps(backend.last_open_diagnostics, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    unittest.main()
