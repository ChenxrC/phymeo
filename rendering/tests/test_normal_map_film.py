"""HOST contracts for the authored, same-input normal-map film (no GPU claim)."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import importlib
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))


class NormalMapFilmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.subject = importlib.import_module("normal_map_film")

    def test_profile_freezes_one_variable_and_original_middle_four_seconds(self):
        p = self.subject.frozen_profile()
        self.assertEqual((p["frames"], p["fps"], p["size"]), (120, 30, [1280, 720]))
        self.assertEqual((p["source_start"], p["source_stop_exclusive"]), (60, 180))
        self.assertEqual(p["variants"], ["off", "on"])
        self.assertEqual((p["msaa_samples"], p["display_transform"]), (1, "pbr-neutral"))
        self.assertEqual((p["exposure"], p["cube_size"], p["area_light_grid"]), (8., 256, 2))
        self.assertEqual(p["shadow_mode"], "ray_query")
        self.assertEqual(p["crop_source_frame"], 120)
        p["size"][0] = 2
        self.assertEqual(self.subject.frozen_profile()["size"], [1280, 720])

    def test_height_derivatives_independently_predict_normal_direction(self):
        # The independently differenced analytic height uses metre-domain periods.
        for kind in ("steel", "rubber"):
            for u, v in ((.103, .217), (.317, .591), (.763, .811)):
                with self.subTest(kind=kind, u=u, v=v):
                    h, dx, dy = self.subject.height_and_slopes(kind, u, v)
                    eps = 1e-7
                    width, height = 2*math.pi*.11, math.pi*.11
                    hx = (self.subject.height_and_slopes(kind, u+eps, v)[0]
                          - self.subject.height_and_slopes(kind, u-eps, v)[0])/(2*eps*width)
                    hy = (self.subject.height_and_slopes(kind, u, v+eps)[0]
                          - self.subject.height_and_slopes(kind, u, v-eps)[0])/(2*eps*height)
                    self.assertAlmostEqual(float(dx), float(hx), delta=2e-7)
                    self.assertAlmostEqual(float(dy), float(hy), delta=2e-7)
                    expected = np.asarray((-hx, -hy, 1.))
                    expected /= np.linalg.norm(expected)
                    np.testing.assert_allclose(self.subject.tangent_normal(kind, u, v),
                                               expected, rtol=0, atol=2e-7)
                    self.assertTrue(math.isfinite(h))

    def test_normal_field_is_unit_positive_z_periodic_and_not_flat(self):
        u, v = np.meshgrid(np.arange(61)/61+.001, np.arange(61)/61+.002)
        for kind in ("steel", "rubber"):
            n = self.subject.tangent_normal(kind, u, v)
            np.testing.assert_allclose(np.linalg.norm(n, axis=-1), 1., atol=3e-15)
            self.assertTrue(np.all(n[..., 2] > .8))
            self.assertGreater(float(np.ptp(n[..., 0])), .2)
            np.testing.assert_allclose(n, self.subject.tangent_normal(kind, u+1, v+1),
                                       atol=2e-12, rtol=0)
        with self.assertRaises(ValueError):
            self.subject.tangent_normal("unregistered-material", 0., 0.)

    def test_textures_are_repeatable_rgb_linear_data_with_authored_provenance(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.subject.make_textures(root/"first", side=256)
            second = self.subject.make_textures(root/"second", side=256)
            for kind in ("steel", "rubber"):
                a, b = first[kind], second[kind]
                self.assertEqual(a["sha256"], b["sha256"])
                self.assertEqual(a["source"], "authored analytic height derivatives; not a scan")
                self.assertEqual(a["transfer"], "linear RGB8 normal data, no sRGB decode")
                path = Path(a["file"])
                self.assertEqual(sha256(path.read_bytes()).hexdigest(), a["sha256"])
                with Image.open(path) as image:
                    self.assertEqual((image.mode, image.size), ("RGB", (256, 256)))
                    rgb = np.asarray(image)
                # Independently reconstruct the chosen texel from its declared centre.
                x, y = 37, 91
                n = self.subject.tangent_normal(kind, (x+.5)/256, (y+.5)/256)
                expected = np.floor((n+1)*127.5+.5).astype(np.uint8)
                np.testing.assert_array_equal(rgb[y, x], expected)
                self.assertGreater(np.unique(rgb.reshape(-1, 3), axis=0).shape[0], 8)

    def test_uv_sphere_has_outward_nonzero_faces_and_invertible_uv_differentials(self):
        p, n, t, uv = map(np.asarray, self.subject.uv_sphere())
        self.assertEqual(uv.shape, (len(p), 2))
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1., atol=5e-15)
        edges = p[t[:, 1:]] - p[t[:, :1]]
        cross = np.cross(edges[:, 0], edges[:, 1])
        self.assertTrue(np.all(np.linalg.norm(cross, axis=1) > 1e-10))
        self.assertTrue(np.all(np.sum(cross*p[t].mean(axis=1), axis=1) > 0))
        duv = uv[t[:, 1:]] - uv[t[:, :1]]
        determinant = duv[:, 0, 0]*duv[:, 1, 1]-duv[:, 0, 1]*duv[:, 1, 0]
        self.assertTrue(np.all(np.abs(determinant) > 1e-8))
        self.assertTrue(np.all((uv >= 0) & (uv <= 1)))

    def test_pair_reuses_geometry_uv_and_keeps_original_control_materials(self):
        from phymeo_rendering.mesh_data import load_mesh
        with tempfile.TemporaryDirectory() as temporary:
            off, on, nodes, textures = self.subject.build_pair(Path(temporary))
            self.assertEqual(off.meshes, on.meshes)
            self.assertEqual(off.instances, on.instances)
            self.assertEqual(off.lights, on.lights)
            self.assertNotEqual(off.key, on.key)
            a, b = {m.id: m for m in off.materials}, {m.id: m for m in on.materials}
            for identity in a:
                if identity in ("rough-steel", "matte-rubber"):
                    for field in ("id", "base_color", "metallic", "roughness"):
                        self.assertEqual(getattr(a[identity], field), getattr(b[identity], field))
                    self.assertIsNone(b[identity].base_color_map)
                    self.assertIsNone(b[identity].roughness_map)
                    self.assertIsNotNone(b[identity].normal_map)
                else:
                    self.assertEqual(a[identity], b[identity])
            self.assertEqual(a["rough-steel"].roughness, .45)
            self.assertEqual(a["matte-rubber"].roughness, .85)
            shared = next(m for m in off.meshes if m.id == off.instances[0].mesh_id)
            data = load_mesh(shared)
            self.assertEqual(len(data.texcoords), len(data.positions))
            self.assertEqual(len(nodes), 5)

    def test_binding_rejects_an_extra_material_light_camera_or_motion_change(self):
        from phymeo_rendering.scene import Pose
        with tempfile.TemporaryDirectory() as temporary:
            off, on, nodes, _ = self.subject.build_pair(Path(temporary))
            left, right = [self.subject.exposures(s, nodes) for s in (off, on)]
            proof = self.subject.bind_pair(off, on, left, right)
            self.assertTrue(proof["matched"])
            self.assertEqual(len(left), 120)
            self.assertEqual(left[0][0].time_s, 2.)
            self.assertEqual(left[-1][0].time_s, 179/30)
            self.assertEqual(left[60][0].key.tick, 120)
            changed = replace(on, materials=(replace(on.materials[0], roughness=.2),)+on.materials[1:])
            with self.assertRaisesRegex(ValueError, "material"):
                self.subject.bind_pair(off, changed, left, right)
            # Both original scenes have no explicit directional lights; use an actual change.
            from phymeo_rendering.scene import DirectionalLight
            changed = replace(on, lights=(DirectionalLight((0., 0., -1.), (1., 1., 1.)),))
            with self.assertRaisesRegex(ValueError, "light"):
                self.subject.bind_pair(off, changed, left, right)
            bad = list(right)
            f, c = bad[10]
            bad[10] = (f, replace(c, fx_px=c.fx_px+1))
            with self.assertRaisesRegex(ValueError, "camera"):
                self.subject.bind_pair(off, on, left, bad)
            f, c = right[10]
            changed_nodes = (replace(f.nodes[0], world_pose=Pose((1., 0., 0.))),)+f.nodes[1:]
            bad[10] = (replace(f, nodes=changed_nodes), c)
            with self.assertRaisesRegex(ValueError, "frame|motion"):
                self.subject.bind_pair(off, on, left, bad)

    def test_crop_is_registered_from_geometry_before_image_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            off, on, nodes, _ = self.subject.build_pair(Path(temporary))
            sequence = self.subject.exposures(off, nodes)
            crops = self.subject.crop_rectangles(sequence[60][1])
            self.assertEqual(set(crops), {"steel", "rubber"})
            for bounds in crops.values():
                x0, y0, x1, y1 = bounds
                self.assertEqual((x1-x0, y1-y0), (192, 192))
                self.assertTrue(0 <= x0 < x1 <= 1280 and 0 <= y0 < y1 <= 720)

    def test_sensor_comparison_uses_exact_published_bytes_and_rejects_missing(self):
        from phymeo_rendering.outputs import Channel
        sensors = {Channel.FORWARD_DEPTH: b"depth", Channel.INSTANCE_ID: b"ids", Channel.VALID: b"valid"}
        self.assertTrue(self.subject.compare_sensors(sensors, deepcopy(sensors))["identical"])
        changed = dict(sensors, **{})
        changed[Channel.FORWARD_DEPTH] = b"DEPTH"
        with self.assertRaisesRegex(ValueError, "sensor"):
            self.subject.compare_sensors(sensors, changed)
        with self.assertRaisesRegex(ValueError, "sensor"):
            self.subject.compare_sensors(sensors, {})

    def test_timing_is_nearest_rank_and_rejects_nonfinite_or_incomplete(self):
        sample = list(range(1, 121))
        result = self.subject.timing(sample)
        self.assertEqual((result["p50_ms"], result["p95_ms"]), (60., 114.))
        for values in (sample[:-1], sample[:-1]+[float("nan")], sample[:-1]+[-1]):
            with self.assertRaises(ValueError):
                self.subject.timing(values)

    def test_close_check_requires_actual_zero_balanced_allocations(self):
        value = {"device": {"closed": True, "validation_errors": 0,
            "live_contexts": 0, "live_frames": 0, "live_resources": 0,
            "resident_telemetry": {"final_after_destroy": True, "device_local_bytes_current": 0,
            "host_visible_bytes_current": 0, "allocation_count": 21, "free_count": 21}}}
        self.assertTrue(self.subject.verify_close(value)["closed"])
        for field in ("device_local_bytes_current", "host_visible_bytes_current", "free_count"):
            bad = deepcopy(value)
            bad["device"]["resident_telemetry"][field] += 1
            with self.assertRaises(ValueError):
                self.subject.verify_close(bad)
        with self.assertRaises(ValueError):
            self.subject.verify_close({"device": {"closed": True}})

    def test_codec_probe_retains_exact_full_movie_contract(self):
        probe = {"streams": [{"codec_name": "h264", "codec_type": "video", "codec_tag_string": "avc1",
            "width": 2562, "height": 768, "sample_aspect_ratio": "1:1", "pix_fmt": "yuv420p",
            "color_range": "tv", "color_space": "bt709", "color_transfer": "iec61966-2-1",
            "color_primaries": "bt709", "r_frame_rate": "30/1", "avg_frame_rate": "30/1",
            "time_base": "1/15360", "duration_ts": 61440, "nb_frames": "120", "nb_read_frames": "120"}],
            "format": {"duration": "4.000000"},
            "frames": [{"best_effort_timestamp_time": f"{i/30:.6f}"} for i in range(120)]}
        self.assertEqual(self.subject.verify_movie(probe, split=True)["frames"], 120)
        for field, value in (("nb_read_frames", "119"), ("color_transfer", "bt709"),
                             ("sample_aspect_ratio", "0:1")):
            changed = deepcopy(probe)
            changed["streams"][0][field] = value
            with self.assertRaises(ValueError):
                self.subject.verify_movie(changed, split=True)


if __name__ == "__main__":
    unittest.main()

