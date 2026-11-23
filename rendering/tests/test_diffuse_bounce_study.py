"""HOST contract for the real Panda geometry / single diffuse-bounce still study."""
from dataclasses import replace
import importlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))


class DiffuseBounceStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = importlib.import_module("diffuse_bounce_study")

    def test_registered_profile_is_four_stills_not_synthetic_motion(self):
        p = self.study.frozen_profile()
        self.assertEqual(p["states"], ["red", "grey", "red-blocked", "grey-blocked"])
        self.assertEqual(p["modes"], ["none", "diffuse_bounce"])
        self.assertEqual((p["size"], p["seed"], p["cosine_samples"]), ([1280, 960], 0, 16))
        self.assertEqual((p["warmup"], p["repeats"]), (3, 9))
        self.assertEqual(p["crop_xyxy"], [384, 288, 896, 672])
        self.assertEqual(p["environment"], None)
        self.assertEqual(p["receiver_ids"], [156, 158])
        self.assertEqual(p["display_transform"], "pbr-neutral")
        self.assertEqual(p["exposure"], 1.)
        self.assertEqual(p["white_reflectance"], [.7, .7, .7])
        self.assertIn("display-linear", p["linear_scope"])
        p["size"][0] = 2
        self.assertEqual(self.study.frozen_profile()["size"], [1280, 960])

    def test_grey_is_equal_linear_luminance_not_equal_encoded_srgb(self):
        red = np.asarray(self.study.RED)
        grey = self.study.equal_luminance_grey(red)
        self.assertAlmostEqual(grey, .177954, delta=1e-15)
        self.assertAlmostEqual(float(red @ np.array((.2126, .7152, .0722))), grey, delta=1e-15)
        for invalid in ((1., 0.), (float("nan"), 0., 0.), (-.1, .2, .3), (1.01, 0., 0.)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.study.equal_luminance_grey(invalid)

    def test_panda_mesh_identity_units_and_same_inputs_are_preserved(self):
        from phymeo_rendering.panda_visuals import panda_visual_assets
        _, _, original = panda_visual_assets()
        wanted = tuple(i for i in original if i.node_id in
                       ("panda_hand", "panda_leftfinger", "panda_rightfinger"))
        with tempfile.TemporaryDirectory() as temporary:
            cases = self.study.build_cases(Path(temporary))
            self.assertEqual(tuple(cases), ("red", "grey", "red-blocked", "grey-blocked"))
            base_scene, base_frame, base_camera = cases["red"]
            self.assertEqual({i.id for i in base_scene.instances if 150 <= i.id <= 158}, set(range(150, 159)))
            actual = {i.id: i for i in base_scene.instances}
            for i in wanted:
                self.assertEqual(actual[i.id], replace(i, material_id="study-white"))
            for state, (scene, frame, camera) in cases.items():
                self.assertEqual(scene.meshes, base_scene.meshes)
                self.assertEqual(scene.instances, base_scene.instances)
                self.assertEqual(scene.lights, base_scene.lights)
                self.assertEqual(frame.time_s, 0.)
                self.assertEqual(frame.key.tick, 0)
                self.assertEqual(camera.world_pose, base_camera.world_pose)
                self.assertEqual((camera.width, camera.height, camera.fx_px, camera.fy_px), (1280, 960, 1550., 1550.))
                self.assertEqual(camera.frame_key, frame.key)
                self.assertEqual(frame.nodes[:3], base_frame.nodes[:3])
                material = {m.id: m for m in scene.materials}
                self.assertEqual(material["study-white"].base_color, (.7, .7, .7))
                self.assertEqual(material["study-white"].metallic, 0.)
                self.assertEqual(material["study-white"].roughness, 1.)
                self.assertEqual(material["study-board"].base_color,
                                 tuple(self.study.RED) if state.startswith("red") else (.177954,) * 3)
            # Real URDF finger origin and 35 mm prismatic displacement, rotated as a whole hand.
            nodes = {n.node_id: n.world_pose for n in base_frame.nodes}
            np.testing.assert_allclose(nodes["panda_leftfinger"].position_m, (0., -.035, .1216), atol=1e-15)
            np.testing.assert_allclose(nodes["panda_rightfinger"].position_m, (0., .035, .1216), atol=1e-15)
            self.assertEqual(nodes["panda_hand"].orientation_xyzw, (1., 0., 0., 0.))
            self.assertEqual(cases["red"][1].nodes, cases["grey"][1].nodes)
            self.assertEqual(cases["red-blocked"][1].nodes, cases["grey-blocked"][1].nodes)
            self.assertNotEqual(cases["red"][1].nodes[-1], cases["red-blocked"][1].nodes[-1])

    def test_camera_projection_is_independently_axis_registered(self):
        # Camera basis follows its preregistered XZ plane, image right is world +Y.
        point = np.array((0., .035, .09))
        distance = np.hypot(.42, .15)
        expected = (640+1550*.035/distance, 480., distance)
        np.testing.assert_allclose(self.study.project(point), expected, atol=1e-12)
        np.testing.assert_allclose(self.study.camera_ray(640., 480.),
                                   (-.42/distance, 0., -.15/distance), atol=1e-15)
        with self.assertRaises(ValueError):
            self.study.project((.5, 0., .3))

    def test_ray_triangle_oracle_uses_nearest_positive_surface_and_barycentrics(self):
        triangles = np.array((((-1., -1., 2.), (1., -1., 2.), (0., 1., 2.)),
                              ((-1., -1., 1.), (1., -1., 1.), (0., 1., 1.))))
        result = self.study.intersect_triangles(np.array(((0., 0., 0.), (2., 0., 0.))),
                                               np.array(((0., 0., 1.), (0., 0., 1.))), triangles)
        self.assertEqual(result["triangle"].tolist(), [1, -1])
        self.assertAlmostEqual(result["distance"][0], 1.)
        np.testing.assert_allclose(result["barycentric"][0], (.25, .25, .5), atol=1e-15)
        self.assertTrue(np.isinf(result["distance"][1]))
        behind = self.study.intersect_triangles(np.zeros((1, 3)), np.array(((0., 0., -1.),)), triangles)
        self.assertEqual(behind["triangle"][0], -1)

    def test_roi_is_registered_from_original_finger_local_surface_before_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            cases = self.study.build_cases(Path(temporary))
            reference = self.study.register_reference(*cases["red"])
            self.assertEqual(reference["ids"].shape, (960, 1280))
            for identity in (156, 158):
                selected = reference["ids"] == identity
                self.assertGreaterEqual(int(selected.sum()), 128)
                self.assertTrue(np.isfinite(reference["depth"][selected]).all())
                local = reference["local_points"][selected]
                self.assertTrue(np.all((local[:, 2] >= .012) & (local[:, 2] <= .032)))
                self.assertTrue(np.all(reference["normal_x"][selected] >= .5))
            _, frame, camera = cases["red"]
            with self.assertRaises(ValueError):
                self.study.register_reference(cases["red"][0], frame, replace(camera, fx_px=1500.))

    def test_linear_values_are_inverse_srgb_display_data_not_raw_hdr(self):
        rgb = np.array(((0, 10, 11), (64, 128, 255)), np.uint8)
        measured = self.study.display_linear(rgb)
        for row in range(2):
            for channel in range(3):
                s = float(rgb[row, channel])/255
                expected = s/12.92 if s <= .04045 else ((s+.055)/1.055)**2.4
                self.assertAlmostEqual(measured[row, channel], expected, delta=1e-15)
        with self.assertRaises(ValueError):
            self.study.display_linear(rgb.astype(float))

    @staticmethod
    def samples():
        ids = np.full((20, 20), 156, np.uint32); ids[:, 10:] = 158
        images = {}
        for state in ("red", "grey", "red-blocked", "grey-blocked"):
            off = np.full((20, 20, 3), 20, np.uint8)
            on = off.copy()
            on[:] = ((110, 25, 25) if state == "red" else (60, 60, 60) if state == "grey"
                     else (30, 25, 25) if state == "red-blocked" else (27, 27, 27))
            images[state] = {"none": off, "diffuse_bounce": on}
        return images, ids

    def test_red_minus_equal_luminance_grey_and_occlusion_are_separate_gates(self):
        images, ids = self.samples()
        result = self.study.compare_appearance(images, ids)
        self.assertTrue(result["passed"])
        self.assertGreater(result["receivers"]["156"]["unblocked_linear_red_excess"], .01)
        for state in images:
            images[state]["diffuse_bounce"][:] = 80  # white increase cannot pass a chromatic test
        self.assertFalse(self.study.compare_appearance(images, ids)["passed"])

    def test_blocker_must_reduce_colour_transfer_not_only_total_brightness(self):
        images, ids = self.samples()
        images["red-blocked"]["diffuse_bounce"] = images["red"]["diffuse_bounce"].copy()
        images["grey-blocked"]["diffuse_bounce"] = images["grey"]["diffuse_bounce"].copy()
        self.assertFalse(self.study.compare_appearance(images, ids)["passed"])

    def test_direct_baseline_cannot_change_between_board_colours(self):
        images, ids = self.samples()
        images["grey"]["none"][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "direct|baseline"):
            self.study.compare_appearance(images, ids)

    def test_registered_roi_must_match_actual_primary_geometry_without_pruning(self):
        ids = np.full((20, 20), 156, np.uint32)
        reference = dict(ids=ids, depth=np.full((20, 20), .4))
        depth = np.full((20, 20), .4, np.float32)
        valid = np.ones((20, 20), np.uint8)
        self.assertLess(self.study.verify_roi(reference, ids, depth, valid)["max_depth_error_m"], .0001)
        for kind in ("ids", "depth", "valid"):
            altered = {"ids": ids.copy(), "depth": depth.copy(), "valid": valid.copy()}
            altered[kind][0, 0] = {"ids": 2, "depth": .41, "valid": 0}[kind]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.study.verify_roi(reference, **altered)

    def test_full_frame_geometry_byte_changes_fail_even_outside_foreground_roi(self):
        from phymeo_rendering.outputs import Channel
        a = {Channel.FORWARD_DEPTH: b"depth", Channel.INSTANCE_ID: b"ids", Channel.VALID: b"valid"}
        self.assertTrue(self.study.compare_sensors(a, dict(a))["identical"])
        b = dict(a); b[Channel.VALID] += b"x"
        with self.assertRaises(ValueError):
            self.study.compare_sensors(a, b)


if __name__ == "__main__":
    unittest.main()
