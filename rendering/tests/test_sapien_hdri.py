"""Directional HDRI reference contract and opt-in public-camera acceptance."""
from dataclasses import fields, FrozenInstanceError
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import tempfile
import unittest

from phymeo_rendering.backends import UnsupportedRequestError
from phymeo_rendering.backends.sapien import SapienConfig, SapienStudio
from phymeo_rendering.backends._sapien_studio import studio_recipe, uniform_hdr
from phymeo_rendering.pipeline import RenderPath


ROOT = Path(__file__).resolve().parents[1]


def probe_module():
    spec = importlib.util.spec_from_file_location("reference_hdri_probe", ROOT / "examples/reference_hdri_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SapienHDRIContractTests(unittest.TestCase):
    def test_new_environment_is_appended_and_old_neutral_fields_are_unchanged(self):
        self.assertEqual([field.name for field in fields(SapienStudio)], [
            "center_m", "scale_m", "panel_size_scale", "panel_strength", "environment_radiance", "environment_map"])
        old = SapienStudio((.18, 0., .42), 1., 1., 1., .125)
        self.assertIsNone(old.environment_map)
        self.assertEqual(studio_recipe(old)["environment_radiance"], .125)
        self.assertNotIn("environment_map", studio_recipe(old))
        raw = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 64 +X 64\n" + bytes((128, 128, 128, 126)) * 4096
        self.assertEqual(uniform_hdr(.125), raw)

    def test_map_is_a_frozen_declaration_and_explicitly_replaces_uniform_only(self):
        from phymeo_rendering.environment import RadianceEnvironment
        environment = RadianceEnvironment("not-opened.hdr", "a"*64, strength=.5, yaw_degrees=90., face_size=128)
        studio = SapienStudio((.18, 0., .42), 1., 1., .8, .125, environment)
        self.assertIs(studio.environment_map, environment)
        self.assertIsNone(studio_recipe(studio)["environment_radiance"])
        self.assertEqual(studio_recipe(studio)["environment_source"], "directional-radiance-map")
        self.assertEqual(studio_recipe(studio)["uniform_environment_radiance_ignored"], .125)
        self.assertEqual(studio_recipe(studio)["environment_map"]["sha256"], "a"*64)
        self.assertEqual(studio_recipe(studio)["environment_map"]["strength"], .5)
        self.assertEqual(studio_recipe(studio)["environment_map"]["yaw_degrees"], 90.)
        self.assertEqual(studio_recipe(studio)["panels"], studio_recipe(SapienStudio(panel_strength=.8, center_m=(.18, 0., .42)))["panels"])
        with self.assertRaises(FrozenInstanceError):
            studio.environment_map = None

    def test_map_remains_pt_only_and_unrelated_types_are_rejected(self):
        from phymeo_rendering.environment import RadianceEnvironment
        studio = SapienStudio(environment_map=RadianceEnvironment("unused.hdr", "b"*64))
        self.assertIs(SapienConfig(path=RenderPath.PATH_TRACE, studio=studio).studio, studio)
        with self.assertRaisesRegex(UnsupportedRequestError, "path_trace"):
            SapienConfig(path=RenderPath.RASTER, studio=studio)
        for value in ("environment.hdr", {}, 4):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "environment_map"):
                SapienStudio(environment_map=value)

    def test_independent_world_radiance_and_yaw_have_known_anchors(self):
        probe = probe_module()
        for direction, expected in (((1., 0., 0.), (.35, .30, .40)),
                                    ((0., 1., 0.), (.20, .50, .40)),
                                    ((0., 0., 1.), (.20, .30, .65))):
            for actual, known in zip(probe.expected_radiance(direction), expected, strict=True):
                self.assertAlmostEqual(actual, known)
        for actual, known in zip(probe.expected_radiance((0., 1., 0.), yaw_degrees=90.), (.35, .30, .40), strict=True):
            self.assertAlmostEqual(actual, known)
        # Noncentral direction, so an axis-only face permutation cannot pass.
        direction = (.6, 0., .8)
        for actual, known in zip(probe.expected_radiance(direction), (.29, .30, .60), strict=True):
            self.assertAlmostEqual(actual, known)

    def test_calibration_source_is_rgb_e_owned_bytes_and_protocol_predeclares_gates(self):
        probe = probe_module()
        raw = probe.calibration_hdr()
        header, body = raw.split(b"\n-Y 256 +X 512\n", 1)
        self.assertIn(b"FORMAT=32-bit_rle_rgbe", header)
        self.assertEqual(len(body), 512*256*4)
        self.assertEqual(raw, probe.calibration_hdr())
        protocol = probe.protocol()
        self.assertEqual(protocol["source_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(protocol["yaw_degrees"], [0., 90.])
        self.assertEqual(len(protocol["view_directions"]), 6)
        self.assertEqual(len(protocol["probe_pixels"]), 9)
        self.assertEqual(protocol["rgb8_max_absolute_error"], 3)
        self.assertEqual(protocol["sampling"]["mode"], "official")
        self.assertEqual(protocol["sampling"]["denoiser"], "none")

    def test_camera_oracle_preserves_all_six_world_axes_and_noncentral_rays(self):
        probe = probe_module()
        for direction in ((1., 0., 0.), (-1., 0., 0.), (0., 1., 0.), (0., -1., 0.), (0., 0., 1.), (0., 0., -1.)):
            camera, basis = probe.view_camera(direction)
            # The requested principal point is between pixels; passing the
            # continuous u/v below gives its exact pixel-centre ray.
            center = probe.world_ray(camera, basis, camera.cx_px - .5, camera.cy_px - .5)
            for actual, expected in zip(center, direction, strict=True):
                self.assertAlmostEqual(actual, expected)
            off_axis = probe.world_ray(camera, basis, 24, 18)
            self.assertAlmostEqual(math.hypot(*off_axis), 1.)
            self.assertGreater(sum(a*b for a, b in zip(off_axis, direction, strict=True)), .7)
            self.assertLess(sum(a*b for a, b in zip(off_axis, direction, strict=True)), .999)


@unittest.skipUnless(os.environ.get("PHYMEO_SAPIEN_HDRI_DEVICE") == "1", "explicit HDRI reference device slot")
class SapienHDRIDeviceTest(unittest.TestCase):
    def test_direction_yaw_mirror_response_and_existing_raster_sensors(self):
        output = os.environ.get("PHYMEO_SAPIEN_HDRI_OUTPUT")
        if output is None:
            self.fail("PHYMEO_SAPIEN_HDRI_OUTPUT must name a NEW persistent evidence directory")
        report = probe_module().run_probe(Path(output), device=os.environ.get("PHYMEO_SAPIEN_DEVICE_NAME", "cuda:0"))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["calibration"]["exposures"], 12)
        self.assertEqual(report["calibration"]["compared_pixels"], 108)
        self.assertLessEqual(report["calibration"]["max_channel_error"], 3)
        self.assertTrue(report["reflection"]["changed"])
        self.assertTrue(report["raster_sensors"]["unchanged"])
        self.assertEqual(report["raster_sensors"]["scope"], "independent existing raster session; PT sensors remain unsupported")


if __name__ == "__main__":
    unittest.main()
