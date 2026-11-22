"""Studio contracts plus explicitly enabled real SAPIEN PT evidence."""
from dataclasses import FrozenInstanceError, replace
import math
import os
import unittest

from phymeo_rendering.backends import UnsupportedRequestError
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import FrameKey, Pose, SceneFrame, VisualScene


class SapienStudioContractTest(unittest.TestCase):
    def test_opt_in_preserves_r0_defaults_and_rejects_unimplemented_path(self):
        from phymeo_rendering.backends.sapien import SapienConfig, SapienStudio
        self.assertIsNone(SapienConfig().studio)
        studio = SapienStudio(center_m=(0.18, 0.0, 0.42), scale_m=1.0)
        self.assertEqual(studio.environment_radiance, 0.125)
        self.assertEqual(studio.panel_size_scale, 1.0)
        with self.assertRaises(FrozenInstanceError):
            studio.scale_m = 2.0
        config = SapienConfig(path=RenderPath.PATH_TRACE, studio=studio)
        self.assertIs(config.studio, studio)
        with self.assertRaisesRegex(UnsupportedRequestError, "studio.*path_trace"):
            SapienConfig(path=RenderPath.RASTER, studio=studio)
        with self.assertRaisesRegex(TypeError, "studio"):
            SapienConfig(studio="neutral")

    def test_invalid_studio_fields_are_named_and_do_not_initialize_a_device(self):
        from phymeo_rendering.backends.sapien import SapienStudio
        cases = (
            ({"center_m": [0.0, 0.0, 0.0]}, TypeError, "center_m"),
            ({"center_m": (0.0, 0.0)}, ValueError, "center_m"),
            ({"center_m": (0.0, math.nan, 0.0)}, ValueError, "center_m"),
            ({"center_m": (0.0, 10**1000, 0.0)}, ValueError, "center_m"),
            ({"scale_m": True}, TypeError, "scale_m"),
            ({"scale_m": 0.0}, ValueError, "scale_m"),
            ({"scale_m": math.inf}, ValueError, "scale_m"),
            ({"panel_size_scale": 0.0}, ValueError, "panel_size_scale"),
            ({"panel_strength": -1.0}, ValueError, "panel_strength"),
            ({"environment_radiance": math.nan}, ValueError, "environment_radiance"),
            ({"environment_radiance": 101.0}, ValueError, "environment_radiance"),
        )
        for kwargs, error, field in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(error, field):
                SapienStudio(**kwargs)

    def test_recipe_uses_metres_and_conserves_panel_power_when_spreading_light(self):
        from phymeo_rendering.backends.sapien import SapienStudio
        from phymeo_rendering.backends._sapien_studio import studio_recipe
        center = (0.18, 0.0, 0.42)
        normal = studio_recipe(SapienStudio(center_m=center))
        doubled = studio_recipe(SapienStudio(center_m=center, scale_m=2.0))
        soft = studio_recipe(SapienStudio(center_m=center, panel_size_scale=2.0))
        self.assertEqual(len(normal["panels"]), 2)
        for a, b, c in zip(normal["panels"], doubled["panels"], soft["panels"], strict=True):
            for axis in range(3):
                self.assertAlmostEqual(b["position_m"][axis] - center[axis],
                                       2 * (a["position_m"][axis] - center[axis]))
            self.assertEqual(b["half_size_m"], [2*x for x in a["half_size_m"]])
            self.assertEqual(b["radiance_linear_rgb"], a["radiance_linear_rgb"])
            area = 4 * math.prod(a["half_size_m"])
            soft_area = 4 * math.prod(c["half_size_m"])
            self.assertAlmostEqual(area*a["radiance_linear_rgb"][0],
                                   soft_area*c["radiance_linear_rgb"][0])
            inward = [center[i] - a["position_m"][i] for i in range(3)]
            self.assertGreater(sum(x*y for x, y in zip(inward, a["emission_direction_world"])), 0.0)

    def test_uniform_hdr_is_linear_and_has_a_reproducible_lossless_anchor(self):
        from phymeo_rendering.backends._sapien_studio import uniform_hdr
        raw = uniform_hdr(0.125)
        header, pixels = raw.split(b"\n-Y 64 +X 64\n", 1)
        self.assertIn(b"FORMAT=32-bit_rle_rgbe", header)
        self.assertEqual(len(pixels), 64*64*4)
        self.assertEqual(raw, uniform_hdr(0.125))
        # Independent RGBE decode, not a product decode round trip.
        r, g, b, exponent = pixels[:4]
        decoded = tuple(v * 2.0**(exponent - 136) for v in (r, g, b))
        self.assertEqual(decoded, (0.125, 0.125, 0.125))
        self.assertEqual(pixels, pixels[:4] * (64*64))

    def test_total_upload_budget_counts_instances_not_only_unique_meshes(self):
        from phymeo_rendering.backends._sapien_studio import geometry_budget
        # Counts are metadata-only unit tests, not fake rendering evidence.
        result = geometry_budget(((30_000, 60_000),)*3, mesh_slots=3)
        self.assertEqual(result["uploaded_triangles"], 180_000)
        self.assertEqual(result["uploaded_vertices"], 90_000)
        with self.assertRaisesRegex(UnsupportedRequestError, "triangle"):
            geometry_budget(((30_000, 60_000),)*4, mesh_slots=1)
        with self.assertRaisesRegex(UnsupportedRequestError, "vert"):
            geometry_budget(((65_000, 10),)*10, mesh_slots=1)
        with self.assertRaisesRegex(UnsupportedRequestError, "instance"):
            geometry_budget(((3, 1),)*257, mesh_slots=1)
        with self.assertRaisesRegex(UnsupportedRequestError, "mesh"):
            geometry_budget((), mesh_slots=257)


@unittest.skipUnless(os.environ.get("PHYMEO_SAPIEN_DEVICE") == "1", "explicit real SAPIEN device opt-in")
class SapienStudioDeviceTest(unittest.TestCase):
    def _open(self, scene, studio):
        from phymeo_rendering.backends.sapien import SapienBackend, SapienConfig
        backend = SapienBackend(SapienConfig(path=RenderPath.PATH_TRACE, device="cuda:0",
                                            samples_per_pixel=64, studio=studio))
        session = backend.open_scene(scene)
        self.addCleanup(session.close)
        return session

    def test_real_hdr_background_and_studio_resources_are_session_owned(self):
        from phymeo_rendering.backends.sapien import SapienStudio
        key = FrameKey("studio-empty", "one", 0, 0, 0, "authored")
        scene = VisualScene(key.scene_key, (), (), ())
        frame = SceneFrame(key, 0.0, ())
        camera = Camera("studio", key, Pose((0.0, 0.0, 1.0)), 16, 12,
                        12.0, 12.0, 8.0, 6.0, 0.1, 5.0)
        request = CaptureRequest(camera, RenderPath.PATH_TRACE, (Channel.RGB,))
        plain = self._open(scene, None)
        self.assertEqual(set(plain.capture(frame, request).planes[0].data), {0})
        studio = self._open(scene, SapienStudio())
        captured = studio.capture(frame, request)
        # Uniform linear .125 -> standard sRGB8 99, independently known.
        self.assertTrue(all(abs(value - 99) <= 1 for value in captured.planes[0].data))
        info = studio.diagnostics()
        self.assertEqual(info["studio"]["environment"]["encoding"], "linear-radiance-rgbe")
        self.assertEqual(info["studio"]["environment"]["face_count"], 6)
        self.assertEqual(info["lifecycle"]["attached_lights"], 2)
        studio.close()
        closed = studio.diagnostics()
        self.assertEqual(closed["lifecycle"]["attached_entities_after_clear"], 0)
        self.assertEqual(closed["lifecycle"]["attached_lights_after_clear"], 0)
        self.assertFalse(closed["lifecycle"]["studio_temporary_files_exist"])
        self.assertFalse(plain.diagnostics()["lifecycle"]["closed"])

    def test_real_area_lights_change_diffuse_and_metal_reflection(self):
        from integration._g1_table import table_frames, table_scene
        from phymeo_rendering.backends.sapien import SapienStudio
        scene = replace(table_scene(), lights=())
        frame = table_frames()[0]
        camera = Camera("studio", frame.key, Pose((0.0, 0.0, 1.0625)), 160, 120,
                        125.0, 125.0, 80.0, 60.0, 0.1, 5.0)
        request = CaptureRequest(camera, RenderPath.PATH_TRACE, (Channel.RGB,))
        environment_only = self._open(scene, SapienStudio(panel_strength=0.0))
        panels = self._open(scene, SapienStudio())
        baseline = environment_only.capture(frame, request).planes[0].data
        image = panels.capture(frame, request).planes[0].data
        def region_mean(data, left, right):
            samples = [data[(v*160+u)*3+c] for v in range(54, 66)
                       for u in range(left, right) for c in range(3)]
            return sum(samples)/len(samples)
        # Both regions lie strictly on the known box top faces; pure metal
        # cannot be brightened by a Lambert diffuse term.
        self.assertGreater(region_mean(image, 74, 86), region_mean(baseline, 74, 86)+2.0)
        self.assertGreater(region_mean(image, 43, 55), region_mean(baseline, 43, 55)+2.0)
        self.assertEqual(panels.diagnostics()["geometry"]["uploaded_triangles"],
                         environment_only.diagnostics()["geometry"]["uploaded_triangles"])


if __name__ == "__main__":
    unittest.main()
