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
            ({"environment_radiance": 1e-100}, ValueError, "environment_radiance"),
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


    def test_real_larger_emitters_broaden_geometrically_predicted_penumbra(self):
        from integration._g1_table import table_frames, table_scene
        from phymeo_rendering.assets import Material
        from phymeo_rendering.backends.sapien import SapienBackend, SapienConfig, SapienStudio
        from phymeo_rendering.scene import Instance
        source = table_scene()
        floor = Instance(1, "table", "matte", "base")
        blocker = Instance(2, "block", "black", "base", Pose((0.0, 0.0, 0.125)))
        materials = (Material("matte", (0.4,)*3, 0.0, 1.0),
                     Material("black", (0.0,)*3, 0.0, 1.0))
        clear = replace(source, materials=materials, instances=(floor,), lights=())
        blocked = replace(clear, instances=(floor, blocker))
        frame = table_frames()[0]
        camera = Camera("shadow", frame.key, Pose((0.0, 0.0, 1.0)), 128, 128,
                        200.0, 200.0, 64.0, 64.0, 0.1, 5.0)
        request = CaptureRequest(camera, RenderPath.PATH_TRACE, (Channel.RGB,))

        def crosses_box(point, light):
            low, high = 0.0, 1.0
            for a, b, lo, hi in zip(point, light, (-0.0625, -0.0625, 0.0625),
                                   (0.0625, 0.0625, 0.1875)):
                d = b-a
                if d == 0:
                    if not lo <= a <= hi:
                        return False
                else:
                    s, t = sorted(((lo-a)/d, (hi-a)/d))
                    low, high = max(low, s), min(high, t)
            return high > low

        def key_visibility(point, size):
            # Independent finite-ray/AABB reference. These are the declared
            # key panel's metres, not image-derived edges or renderer output.
            center = (-1.2, -1.4, 1.8)
            n = tuple(-x/math.hypot(*center) for x in center)
            axis = (0.0, -n[2], n[1])
            def cross(a, b):
                return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
            def rotate(value):
                once = cross(axis, value)
                twice = cross(axis, once)
                return tuple(value[i]+once[i]+twice[i]/(1+n[0]) for i in range(3))
            # GL panel X/Y -> SAPIEN entity -Y/+Z, then shortest +X-to-n rotation.
            bx, by = rotate((0., -1., 0.)), rotate((0., 0., 1.))
            visible = 0
            for i in range(9):
                for j in range(9):
                    x, y = (2*(i+0.5)/9-1)*0.8*size, (2*(j+0.5)/9-1)*0.55*size
                    lamp = tuple(center[k]+x*bx[k]+y*by[k] for k in range(3))
                    visible += not crosses_box(point, lamp)
            return visible/81

        # Fixed camera/floor geometry, masks chosen before any image is read.
        # Both masks avoid the camera projection of the hovering 0.125 m cube.
        inner, outer = [], []
        for v in range(8, 48, 2):
            for u in range(80, 120, 2):
                point = ((u+0.5-64)/200, (64-v-0.5)/200, 0.0)
                tiny, wide = key_visibility(point, 0.05), key_visibility(point, 1.0)
                if tiny == 0 and 0.25 <= wide <= 0.75:
                    inner.append((v*128+u)*3)
                if tiny == 1 and 0.25 <= wide <= 0.75:
                    outer.append((v*128+u)*3)
        self.assertGreaterEqual(len(inner), 5, "analytic inner penumbra fixture")
        self.assertGreaterEqual(len(outer), 5, "analytic outer penumbra fixture")
        def linear(byte):
            x = byte/255.0
            return x/12.92 if x <= 0.04045 else ((x+0.055)/1.055)**2.4
        def ratios(size):
            frames = []
            for scene in (clear, blocked):
                backend = SapienBackend(SapienConfig(path=RenderPath.PATH_TRACE, device="cuda:0",
                    samples_per_pixel=128, max_bounces=0,
                    studio=SapienStudio(panel_size_scale=size, environment_radiance=0.0)))
                session = backend.open_scene(scene)
                try:
                    frames.append(session.capture(frame, request).planes[0].data)
                finally:
                    session.close()
            return [sum(linear(frames[1][p])/linear(frames[0][p]) for p in mask)/len(mask)
                    for mask in (inner, outer)]
        tiny, wide = ratios(0.05), ratios(1.0)
        # Wider emitter lightens the analytic hard-shadow interior and darkens
        # its formerly clear exterior. Brightness alone cannot satisfy both.
        self.assertGreater(wide[0], tiny[0]+0.08, (len(inner), tiny, wide))
        self.assertLess(wide[1], tiny[1]-0.08, (len(outer), tiny, wide))


if __name__ == "__main__":
    unittest.main()
