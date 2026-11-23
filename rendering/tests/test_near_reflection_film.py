"""Independent HOST checks for the authored near-reflection optical film."""
from copy import deepcopy
from dataclasses import replace
import importlib
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))


class NearReflectionFilmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.subject = importlib.import_module("near_reflection_film")

    def test_profile_freezes_optical_inputs_and_four_channel_timing(self):
        p = self.subject.frozen_profile()
        self.assertEqual((p["frames"], p["fps"], p["size"]), (90, 30, [1280, 720]))
        self.assertEqual(p["modes"], ["environment", "single_bounce"])
        self.assertEqual((p["msaa_samples"], p["sphere_radius_m"], p["sphere_roughness"]), (1, .30, .04))
        self.assertEqual(p["crop_xyxy"], [448, 168, 832, 552])
        self.assertEqual(p["phase_frames"], [15, 75])
        self.assertEqual(len(p["channels"]), 4)
        self.assertEqual(p["cube_size_m"], .28)
        self.assertEqual(p["gain_expression"], "(R-max(G,B))_single_bounce - (R-max(G,B))_environment")
        p["size"][0] = 2
        self.assertEqual(self.subject.frozen_profile()["size"], [1280, 720])

    def test_scene_keeps_ball_and_camera_fixed_only_moves_the_real_cube(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene, sequence = self.subject.build_fixture(Path(temporary))
            self.assertEqual(len(scene.instances), 2)
            self.assertEqual({i.id for i in scene.instances}, {1, 2})
            self.assertEqual(len(sequence), 90)
            self.assertEqual(sequence[0][0].time_s, 0.)
            self.assertEqual(sequence[-1][0].time_s, 89/30)
            poses = []
            for i, (frame, camera) in enumerate(sequence):
                self.assertEqual((frame.key.tick, frame.time_s), (i, i/30))
                self.assertEqual(camera.frame_key, frame.key)
                self.assertEqual(camera.world_pose, sequence[0][1].world_pose)
                self.assertEqual(frame.nodes[0], sequence[0][0].nodes[0])
                poses.append(frame.nodes[1].world_pose.position_m)
            np.testing.assert_allclose(poses[0], (.45, -.75, .05), atol=0, rtol=0)
            np.testing.assert_allclose(poses[-1], (1.35, -.75, .05), atol=0, rtol=0)
            self.assertTrue(all(a[0] < b[0] for a, b in zip(poses, poses[1:])))
            materials = {m.id: m for m in scene.materials}
            self.assertEqual(materials["mirror-steel"].metallic, 1.)
            self.assertEqual(materials["mirror-steel"].roughness, .04)
            self.assertEqual(materials["moving-red-block"].metallic, 0.)
            self.assertEqual(materials["moving-red-block"].roughness, 1.)

    def test_registered_projection_has_visible_and_fully_offscreen_cube(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, sequence = self.subject.build_fixture(Path(temporary))
            on = self.subject.project_cube(*sequence[15])
            off = self.subject.project_cube(*sequence[75])
            self.assertGreater(on["min_x_px"], 832)
            self.assertLess(on["max_x_px"], 1280)
            self.assertGreater(off["min_x_px"], 1280+60)
            self.assertTrue(on["fully_in_view"])
            self.assertTrue(off["fully_outside_right"])
            # Independent axis-aligned camera projection for an actual cube corner.
            frame, camera = sequence[75]
            x, y, z = frame.nodes[1].world_pose.position_m
            expected = 640+1000*(x-.14)/(y+.14+2)
            self.assertAlmostEqual(off["min_x_px"], expected, delta=1e-10)

    def test_sphere_ray_has_known_axis_hit_and_reflects_back_toward_camera(self):
        point, normal, reflected, depth = self.subject.sphere_ray(640., 360.)
        np.testing.assert_allclose(point, (0., -.30, 0.), atol=1e-14)
        np.testing.assert_allclose(normal, (0., -1., 0.), atol=1e-14)
        np.testing.assert_allclose(reflected, (0., -1., 0.), atol=1e-14)
        self.assertAlmostEqual(float(depth), 1.7, delta=1e-14)
        for x, y in ((700., 350.), (550., 410.)):
            point, normal, reflected, depth = self.subject.sphere_ray(x, y)
            incoming = point-np.array((0., -2., 0.))
            incoming /= np.linalg.norm(incoming)
            self.assertAlmostEqual(float(np.linalg.norm(point)), .30, delta=1e-13)
            self.assertAlmostEqual(float(np.linalg.norm(reflected)), 1., delta=1e-13)
            self.assertAlmostEqual(float(np.dot(reflected, normal)),
                                   -float(np.dot(incoming, normal)), delta=1e-13)

    def test_ray_box_rejects_parallel_outside_and_respects_forward_direction(self):
        origin = np.array(((0., -2., 0.), (2., -2., 0.), (0., 2., 0.)))
        direction = np.array(((0., 1., 0.), (0., 1., 0.), (0., 1., 0.)))
        np.testing.assert_array_equal(self.subject.ray_box(origin, direction, (0., 0., 0.), .14),
                                      (True, False, False))

    def test_both_phase_masks_predict_real_reflection_with_conservative_miss_region(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, sequence = self.subject.build_fixture(Path(temporary))
            for index in (15, 75):
                reference = self.subject.reference_masks(*sequence[index])
                self.assertEqual(reference["hit"].shape, (384, 384))
                self.assertGreaterEqual(int(reference["hit"].sum()), 16)
                self.assertGreater(int(reference["miss"].sum()), 5000)
                self.assertFalse(np.any(reference["hit"] & reference["miss"]))
                self.assertTrue(np.all(np.isfinite(reference["depth"][reference["hit"]])))
                self.assertTrue(np.all((reference["depth"][reference["hit"]] > 1.7)
                                       & (reference["depth"][reference["hit"]] < 2.)))
                # An actual reflected ray selected by the mask enters a shrunk cube.
                yy, xx = np.argwhere(reference["hit"])[0]
                point, normal, reflected, _ = self.subject.sphere_ray(448+xx+.5, 168+yy+.5)
                centre = np.asarray(sequence[index][0].nodes[1].world_pose.position_m)
                bounds = [(centre[k]-.105-point[k])/reflected[k] for k in range(3)]
                opposite = [(centre[k]+.105-point[k])/reflected[k] for k in range(3)]
                near = max(0., *(min(a,b) for a,b in zip(bounds, opposite)))
                far = min(max(a,b) for a,b in zip(bounds, opposite))
                self.assertLessEqual(near, far)

    def test_red_gain_rejects_white_highlight_and_measures_channel_advantage(self):
        a = np.full((4, 4, 3), 100, np.uint8)
        white = np.full_like(a, 240)
        red = np.empty_like(a); red[:] = (170, 25, 20)
        mask = np.ones((4, 4), bool)
        no_red = self.subject.red_gain(a, white, mask)
        self.assertEqual(no_red["mean_delta_red_excess"], 0.)
        self.assertFalse(no_red["passed"])
        actual = self.subject.red_gain(a, red, mask)
        self.assertEqual(actual["mean_delta_red_excess"], 145.)
        self.assertEqual(actual["fraction_delta_at_least_10"], 1.)
        self.assertTrue(actual["passed"])
        with self.assertRaises(ValueError):
            self.subject.red_gain(a, red, np.zeros_like(mask))

    def test_optical_measurement_checks_actual_ids_depth_and_background(self):
        shape = (384, 384)
        hit = np.zeros(shape, bool); hit[120:124, 120:124] = True
        miss = np.zeros(shape, bool); miss[130:140, 130:140] = True
        reference = {"hit": hit, "miss": miss, "depth": np.full(shape, 1.8)}
        a = np.full((720, 1280, 3), 50, np.uint8); b = a.copy()
        b[168:552, 448:832][hit] = (180, 20, 20)
        ids = np.zeros((720, 1280), np.uint32); ids[168:552, 448:832][hit | miss] = 1
        depth = np.full((720, 1280), np.inf, np.float32); depth[ids == 1] = 1.8
        measured = self.subject.measure_frame(a, b, ids, depth, reference, direct_phase="offscreen")
        self.assertTrue(measured["optical_passed"])
        wrong_ids = ids.copy(); wrong_ids[168:552,448:832][hit] = 2
        with self.assertRaisesRegex(ValueError, "receiver|ID"):
            self.subject.measure_frame(a, b, wrong_ids, depth, reference, direct_phase="offscreen")
        wrong_depth = depth.copy(); wrong_depth[168:552,448:832][hit] += .01
        with self.assertRaisesRegex(ValueError, "depth"):
            self.subject.measure_frame(a, b, ids, wrong_depth, reference, direct_phase="offscreen")
        changed = b.copy(); changed[0, 0] = (80, 50, 50)
        with self.assertRaisesRegex(ValueError, "background"):
            self.subject.measure_frame(a, changed, ids, depth, reference, direct_phase="offscreen")
        changed = b.copy(); changed[168:552,448:832][miss] += 2
        self.assertFalse(self.subject.measure_frame(a, changed, ids, depth, reference,
                                                   direct_phase="offscreen")["optical_passed"])

    def test_geometry_channels_are_full_frame_byte_identity_not_rgb_similarity(self):
        from phymeo_rendering.outputs import Channel
        a = {Channel.FORWARD_DEPTH: b"depth", Channel.INSTANCE_ID: b"ids", Channel.VALID: b"valid"}
        self.assertTrue(self.subject.compare_sensors(a, deepcopy(a))["identical"])
        bad = dict(a); bad[Channel.INSTANCE_ID] += b"x"
        with self.assertRaises(ValueError):
            self.subject.compare_sensors(a, bad)

    def test_timing_is_ninety_actual_four_channel_frames_without_warmup(self):
        values = list(range(1, 91))
        result = self.subject.timing(values)
        self.assertEqual((result["p50_ms"], result["p95_ms"]), (45, 86))
        for bad in (values[:-1], values[:-1]+[float("nan")]):
            with self.assertRaises(ValueError):
                self.subject.timing(bad)

    def test_movie_contract_checks_ninety_pts_and_preserves_srgb_transfer(self):
        probe = {"streams": [{"codec_name": "h264", "codec_type": "video", "codec_tag_string": "avc1",
            "width": 2562, "height": 768, "sample_aspect_ratio": "1:1", "pix_fmt": "yuv420p",
            "color_range": "tv", "color_space": "bt709", "color_transfer": "iec61966-2-1",
            "color_primaries": "bt709", "r_frame_rate": "30/1", "avg_frame_rate": "30/1",
            "time_base": "1/15360", "duration_ts": 46080, "nb_frames": "90", "nb_read_frames": "90"}],
            "format": {"duration": "3.000000"},
            "frames": [{"best_effort_timestamp_time": f"{i/30:.6f}"} for i in range(90)]}
        self.assertEqual(self.subject.verify_movie(probe, split=True)["frames"], 90)
        wrong = deepcopy(probe); wrong["frames"][75]["best_effort_timestamp_time"] = "2.6"
        with self.assertRaises(ValueError):
            self.subject.verify_movie(wrong, split=True)



    def test_actual_reflection_diagnostics_bind_receiver_and_successful_calls(self):
        import struct
        fp32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]
        backend = dict(reflection=dict(mode="single_bounce", bounces=1, receiver_instances=[
            dict(instance_id=1, material_id="mirror-steel", metallic_fp32=1.,
                 authored_roughness_fp32=fp32(.04))]),
            device={"resident_reflection_info": dict(enabled=1, supported=1, shader_compatible=1,
                reflection_abi_version=1, max_bounces=1, receiver_count=1, set_calls=1,
                captures=95, rgb_captures=95, metallic_min=fp32(.99), roughness_max=fp32(.10))})
        self.assertTrue(self.subject.verify_reflection_binding(backend, mode="single_bounce"))
        for field in ("receiver_count", "captures", "enabled"):
            bad = deepcopy(backend)
            bad["device"]["resident_reflection_info"][field] = 0
            with self.assertRaises(ValueError):
                self.subject.verify_reflection_binding(bad, mode="single_bounce")
        bad = deepcopy(backend); bad["reflection"]["receiver_instances"][0]["instance_id"] = 2
        with self.assertRaises(ValueError):
            self.subject.verify_reflection_binding(bad, mode="single_bounce")
        self.assertTrue(self.subject.verify_reflection_binding(dict(
            reflection=dict(mode="environment", bounces=0, receiver_instances=[]),
            device={"resident_reflection_info": None}), mode="environment"))


if __name__ == "__main__":
    unittest.main()

