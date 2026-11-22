"""R3 independent metrology; device execution is explicit and separately owned."""

import importlib.util
import math
import os
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

from phymeo_rendering.cameras import validate_exposure
from phymeo_rendering.outputs import Capture, Channel, ImagePlane


class CalibrationDefinitionTest(unittest.TestCase):
    def setUp(self):
        from ._r3_scene import build_calibration_fixture
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.scene, self.cases, self.surfaces = build_calibration_fixture(Path(self.folder.name))

    def test_front_planes_cover_declared_metric_distances_and_uint32_identities(self):
        from ._r3_oracle import ray_hit
        from ._r3_scene import BOARD_IDS
        front = self.cases[0]
        for camera in ("fixed", "wrist"):
            for pixel, expected_depth, expected_id in zip(
                ((170, 140), (470, 140), (170, 340), (470, 340)),
                (0.25, 0.5, 1.0, 2.0), BOARD_IDS, strict=True,
            ):
                with self.subTest(camera=camera, pixel=pixel):
                    depth, identity = ray_hit(front.frame, camera, *pixel, self.surfaces)
                    self.assertEqual(identity, expected_id)
                    self.assertAlmostEqual(depth, expected_depth, delta=1e-12)
        self.assertGreater(max(BOARD_IDS), 2**24)

    def test_depth_is_axis_distance_and_background_is_not_a_far_plane(self):
        from ._r3_oracle import ray_hit
        depth, identity = ray_hit(self.cases[0].frame, "fixed", 170, 140, self.surfaces)
        self.assertNotEqual(identity, 0)
        ray_length = math.sqrt(1+((170.5-320)/500)**2+((240-140.5)/500)**2)
        self.assertGreater(depth*ray_length-depth, 0.01)
        self.assertEqual(ray_hit(self.cases[0].frame, "fixed", 0, 0, self.surfaces), (math.inf, 0))

    def test_wrist_mount_uses_this_frame_and_reset_repeats_geometry_with_new_epoch(self):
        from ._r3_oracle import camera_axes
        front, tilted, _, _, reset = self.cases
        self.assertEqual(front.cameras[0].world_pose, tilted.cameras[0].world_pose)
        self.assertNotEqual(front.cameras[1].world_pose, tilted.cameras[1].world_pose)
        for case in self.cases:
            for camera in case.cameras:
                validate_exposure(case.frame, camera)
                origin, _ = camera_axes(case.frame, camera.id)
                for actual, expected in zip(camera.world_pose.position_m, origin, strict=True):
                    self.assertAlmostEqual(actual, expected, delta=1e-14)
        self.assertEqual(reset.frame.nodes, front.frame.nodes)
        self.assertNotEqual(reset.frame.key.epoch, front.frame.key.epoch)
        self.assertEqual(reset.frame.key.tick, 0)
        with self.assertRaises(ValueError):
            validate_exposure(reset.frame, front.cameras[1])

    def test_occlusion_and_flip_are_geometry_changes_not_image_transforms(self):
        from ._r3_oracle import ray_hit
        front, _, occluded, flipped, _ = self.cases
        a = ray_hit(front.frame, "fixed", 519, 128, self.surfaces)
        b = ray_hit(occluded.frame, "fixed", 519, 128, self.surfaces)
        self.assertNotEqual(a[1], b[1])
        self.assertLess(b[0], a[0])
        self.assertEqual(ray_hit(flipped.frame, "fixed", 469, 339, self.surfaces)[1],
                         ray_hit(front.frame, "fixed", 170, 140, self.surfaces)[1])


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy for full-image auditor unit controls")
class CalibrationAuditNegativeTest(unittest.TestCase):
    def test_auditor_rejects_depth_identity_flip_and_stale_exposure(self):
        # Synthetic oracle rasters test rejection logic only, never device quality.
        from ._r3_oracle import audit_capture, reference_image
        from ._r3_scene import build_calibration_fixture
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            _, cases, surfaces = build_calibration_fixture(Path(folder))
            case = cases[0]
            camera = case.cameras[0]
            reference = reference_image(case.frame, camera.id, surfaces)
            depth, ids = reference["depth"], reference["ids"]
            rgb = np.full((480, 640, 3), 70, dtype=np.uint8)
            for surface in surfaces:
                if surface.marker_channel is not None:
                    rgb[ids == surface.identity] = (8, 8, 8)
                    rgb[:, :, surface.marker_channel][ids == surface.identity] = 220
            def capture(d=depth, identities=ids, colors=rgb):
                return Capture(case.frame.key, case.frame.time_s, camera, (
                    ImagePlane(Channel.RGB, colors.tobytes()),
                    ImagePlane(Channel.FORWARD_DEPTH, d.astype("<f4").tobytes()),
                    ImagePlane(Channel.INSTANCE_ID, identities.astype("<u4").tobytes()),
                    ImagePlane(Channel.VALID, (identities != 0).astype("u1").tobytes()),
                ))
            audit_capture(capture(), case, surfaces)
            bad_depth = depth.copy()
            bad_depth[(ids != 0) & ~reference["edge"]] += 0.003
            bad_ids = ids.copy()
            interior = (ids != 0) & ~reference["edge"]
            bad_ids[interior] = np.uint32(7)
            for name, corrupted in (
                ("depth", capture(d=bad_depth)),
                ("ID", capture(identities=bad_ids)),
                ("flip", capture(d=depth[::-1].copy(), identities=ids[::-1].copy(), colors=rgb[::-1].copy())),
                ("RGB marker", capture(colors=np.roll(rgb, 4, axis=1))),
            ):
                with self.subTest(name=name), self.assertRaises(AssertionError):
                    audit_capture(corrupted, case, surfaces)
            with self.assertRaises(AssertionError):
                audit_capture(capture(), replace(case, frame=cases[-1].frame), surfaces)


@unittest.skipUnless(os.environ.get("PHYMEO_R3_NATIVE") == "1", "explicit R3 native device opt-in")
class NativeCalibrationTest(unittest.TestCase):
    def test_two_calibrated_cameras_five_phases(self):
        from phymeo_rendering.backends.native import NativeBackend
        from ._r3_oracle import exercise_calibration
        library = Path(os.environ["PHYMEO_R3_LIBRARY"])
        exercise_calibration(self, lambda: NativeBackend(
            library_path=library, shader_directory=library.parent,
            device_index=int(os.environ["PHYMEO_R3_DEVICE_INDEX"]), validation=True,
        ), "native-raster")


@unittest.skipUnless(os.environ.get("PHYMEO_R3_SAPIEN") == "1", "explicit R3 reference device opt-in")
class ReferenceCalibrationTest(unittest.TestCase):
    def test_two_calibrated_cameras_five_phases(self):
        from phymeo_rendering.backends.sapien import SapienBackend, SapienConfig
        from phymeo_rendering.pipeline import RenderPath
        from ._r3_oracle import exercise_calibration
        exercise_calibration(self, lambda: SapienBackend(SapienConfig(
            path=RenderPath.RASTER, device="cuda:0")), "sapien-raster")
