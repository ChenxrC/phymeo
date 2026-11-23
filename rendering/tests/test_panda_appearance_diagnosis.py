"""Q0 HOST controls. These fixtures do not claim GPU appearance validation."""
from dataclasses import replace
import importlib
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))


class AppearanceDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q0 = importlib.import_module("panda_appearance_diagnosis")

    def test_frozen_pair_changes_only_area_grid(self):
        profile = self.q0.frozen_profile()
        self.assertEqual(profile["frame_index"], 60)
        self.assertEqual(profile["size"], [1280, 720])
        self.assertEqual(profile["checkpoints"], [1, 64])
        self.assertEqual(profile["seeds"], [0, 1])
        a, b = (self.q0.case_settings(k) for k in ("baseline", "dense-panels"))
        self.assertEqual(a.pop("area_light_grid"), 2)
        self.assertEqual(b.pop("area_light_grid"), 8)
        self.assertEqual(a, b)
        self.assertEqual(a["spatial_filter"], "none")

    def test_controls_are_explicit_ablations_not_improvement_claims(self):
        profile = self.q0.frozen_profile()
        self.assertEqual(profile["candidate"], "dense-panels")
        self.assertEqual(profile["ablations"], ["no-directional", "no-panels", "no-normal"])
        with self.assertRaises(ValueError):
            self.q0.case_settings("auto-smooth")

    def test_wide_panel_candidate_preserves_position_and_total_power(self):
        from phymeo_rendering.backends.sapien import SapienStudio
        from phymeo_rendering.scene import DirectionalLight, VisualScene
        lights = (DirectionalLight((0.,0.,-1.),(.25,.25,.25)),)
        scene = VisualScene("same",(),(),(),lights)
        studio = SapienStudio(center_m=(.18,0.,.42),scale_m=.85,panel_strength=.125)
        wide, changed = self.q0._case_scene("wide-panels",(scene,scene),studio)
        self.assertIs(wide,scene)
        self.assertEqual(changed,replace(studio,panel_size_scale=2.))
        self.assertEqual(self.q0.case_settings("wide-panels"),self.q0.case_settings("dense-panels"))
        soft, soft_studio = self.q0._case_scene("soft-studio",(scene,scene),studio)
        self.assertEqual(soft.lights,())
        self.assertEqual(soft_studio,changed)
        self.assertEqual(soft.meshes,scene.meshes)
        self.assertEqual(soft.materials,scene.materials)

    def test_directional_and_panel_controls_share_dense_baseline(self):
        dense=self.q0.case_settings("dense-panels")
        for name in ("no-directional","no-panels","no-normal"):
            self.assertEqual(self.q0.case_settings(name),dense)

    def test_sensor_comparison_checks_bytes_and_all_three_planes(self):
        a = {"depth": b"1234", "id": b"5678", "valid": b"1"}
        self.assertTrue(self.q0.require_same_sensors(a, dict(a)))
        for key in a:
            bad = dict(a)
            bad[key] += b"x"
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.q0.require_same_sensors(a, bad)
        with self.assertRaises(ValueError):
            self.q0.require_same_sensors(a, {"depth": b"1234"})

    def test_perspective_normal_interpolation_not_screen_linear(self):
        normals = np.eye(3)
        actual = self.q0.interpolate_normals(normals, [0.2, 0.3, 0.5], [1, 2, 4])
        expected = np.array([0.2, 0.15, 0.125])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)

    def test_degenerate_interpolation_stays_unknown_not_nan(self):
        with self.assertRaises(ValueError):
            self.q0.interpolate_normals([[1, 0, 0], [-1, 0, 0], [0, 0, 1]],
                                        [.5, .5, 0], [1, 1, 1])

    def test_geometric_and_authored_normal_are_not_conflated(self):
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.]])
        n = np.tile([0, .6, .8], (3, 1))
        row = self.q0.normal_statistics(p, n, [[0, 1, 2]])
        self.assertAlmostEqual(row["geometric_to_authored_angle_deg_max"], 36.86989764584402)
        self.assertEqual(row["triangles"], 1)
        self.assertEqual(row["within_triangle_normal_angle_deg_max"], 0.)

    def test_preserved_split_vertices_may_legitimately_have_different_normals(self):
        p = [[0,0,0], [1,0,0], [0,1,0], [0,0,0], [0,1,0], [0,0,1]]
        n = [[0,0,1]]*3 + [[1,0,0]]*3
        row = self.q0.normal_statistics(p, n, [[0,1,2], [3,4,5]])
        self.assertEqual(row["coincident_positions_with_distinct_normals"], 2)
        self.assertEqual(row["geometric_to_authored_angle_deg_max"], 0.)

    def test_source_normal_reconstruction_uses_separate_normal_indices(self):
        xml = b'''<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
        <library_geometries><geometry id="g"><mesh>
        <source id="n"><float_array>0 0 1 0 1 0 1 0 0</float_array>
        <technique_common><accessor stride="3" count="3"/></technique_common></source>
        <triangles><input semantic="VERTEX" source="#v" offset="0"/>
        <input semantic="NORMAL" source="#n" offset="1"/><p>0 2 1 0 2 1</p></triangles>
        </mesh></geometry></library_geometries>
        <library_visual_scenes><visual_scene><node id="node"><matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix></node></visual_scene></library_visual_scenes>
        </COLLADA>'''
        row = dict(source_geometry="g", source_primitive=0, source_node="node",
                   source_triangle_runs=[[0,1]], source_triangle_count=1,
                   source_matrix_row_major=np.eye(4).reshape(-1).tolist())
        actual = self.q0.source_corner_normals(xml, row)
        np.testing.assert_array_equal(actual, [[[1,0,0],[0,0,1],[0,1,0]]])
        bad = dict(row, source_node="missing")
        with self.assertRaises(ValueError):
            self.q0.source_corner_normals(xml, bad)

    def test_crop_registration_uses_identity_only_and_requires_each_roi(self):
        ids = np.array([[150,150,0,156,156], [150,154,0,158,158], [2,2,2,0,0],
                        [112,113,0,125,149]])
        rows = self.q0.crop_regions(ids, margin=0)
        self.assertEqual(rows["white-shell"]["box"], [0,0,2,2])
        self.assertEqual(rows["fingers"]["box"], [3,0,5,2])
        self.assertEqual(rows["rubber"]["box"], [0,2,3,3])
        self.assertEqual(rows["left-arm"]["ids"], [112,113,115,117])
        self.assertEqual(rows["wrist-shell"]["ids"], [125,142,149])
        with self.assertRaises(ValueError):
            self.q0.crop_regions(np.zeros((4,4), dtype=np.uint32))

    def test_raw_attachment_requires_true_unexposed_complete_mean(self):
        raw = np.ones((2,3,4), dtype="<f4").tobytes()
        record = dict(format="RGBA32F_LE", space="scene-linear-before-exposure",
                      width=3, height=2, batches=1, primary_samples_per_pixel=16, data=raw)
        self.assertEqual(self.q0.require_linear_mean(record, 3, 2, 1).shape, (2,3,4))
        for key, value in (("batches",2),("data",raw[:-1]),("space","srgb")):
            bad = dict(record); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.q0.require_linear_mean(bad,3,2,1)


if __name__ == "__main__":
    unittest.main()
