"""Measurement and asset controls; these synthetic arrays are not render evidence."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from dataclasses import asdict


_MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "roughness_calibration.py"


def example():
    spec = importlib.util.spec_from_file_location("r2_roughness_example", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RoughnessMeasurementTest(unittest.TestCase):
    def test_half_height_is_relative_to_fixed_background_with_interpolated_crossings(self):
        measure = example().measure_profile
        samples = [0.1+0.1*min(i, 10-i) for i in range(11)]
        result = measure(list(range(11)), samples, 0.1)
        self.assertEqual(result["status"], "valid")
        self.assertAlmostEqual(result["half_level"], 0.35)
        self.assertAlmostEqual(result["left_crossing_px"], 2.5)
        self.assertAlmostEqual(result["right_crossing_px"], 7.5)
        self.assertAlmostEqual(result["fwhm_px"], 5.0)

    def test_saturation_low_contrast_multipeak_and_censoring_are_explicit_failures(self):
        measure = example().measure_profile
        cases = (([0., 0.5, 1., 0.5, 0.], "saturated"),
                 ([0., 0.01, 0.02, 0.01, 0.], "low_contrast"),
                 ([0., 0.8, 0., 0.8, 0.], "multiple_peaks"),
                 ([0.6, 0.8, 0.6], "unbounded"))
        for samples, reason in cases:
            with self.subTest(reason=reason):
                result = measure(list(range(len(samples))), samples, 0.)
                self.assertEqual(result["status"], "invalid")
                self.assertIn(reason, result["reasons"])
                self.assertNotIn("fwhm_px", result)

    def test_nonfinite_or_nonmonotonic_profile_coordinates_are_rejected(self):
        measure = example().measure_profile
        for x, y in (([0, 1, 1], [0., 1., 0.]), ([0, 1, 2], [0., float("nan"), 0.])):
            with self.assertRaises(ValueError):
                measure(x, y, 0.)

    def test_image_measurement_keeps_fixed_band_and_rejects_an_off_band_peak(self):
        module = example()
        protocol = module.make_protocol()
        pixels = bytearray(640*480*3)
        for row in range(238, 242):
            for column in range(348, 373):
                value = round(200*(1-abs(column-360)/12))
                start = (row*640+column)*3
                pixels[start:start+3] = bytes((value,)*3)
        valid = module.measure_image(bytes(pixels), protocol)
        self.assertEqual(valid["status"], "valid")
        self.assertEqual(valid["profile_rows"], [238, 239, 240, 241])
        start = (230*640+360)*3
        pixels[start:start+3] = bytes((230,)*3)
        invalid = module.measure_image(bytes(pixels), protocol)
        self.assertEqual(invalid["status"], "invalid")
        self.assertIn("off_profile", invalid["reasons"])


class RoughnessDeclarationTest(unittest.TestCase):
    def test_scene_diff_is_roughness_only_and_shared_sphere_is_valid(self):
        from phymeo_rendering.mesh_data import load_mesh
        module = example()
        with tempfile.TemporaryDirectory() as folder:
            mesh = module.write_sphere(Path(folder))
            decoded = load_mesh(mesh)
            self.assertEqual(len(decoded.triangles), 16128)
            self.assertEqual(len(decoded.positions), 8066)
            # The real mesh has outward winding; geometry admission alone does
            # not establish the orientation of its finite triangle surfaces.
            for ia, ib, ic in decoded.triangles:
                a, b, c = (decoded.positions[i] for i in (ia, ib, ic))
                u, v = tuple(b[i]-a[i] for i in range(3)), tuple(c[i]-a[i] for i in range(3))
                cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
                self.assertGreater(sum(cross[i]*(a[i]+b[i]+c[i]) for i in range(3)), 0.)
            scenes = [module.make_scene(mesh, value) for value in (0.18, 0.52)]
            first, second = (asdict(scene) for scene in scenes)
            first.pop("key")
            second.pop("key")
            self.assertEqual(first["materials"][0].pop("roughness"), 0.18)
            self.assertEqual(second["materials"][0].pop("roughness"), 0.52)
            self.assertEqual(first, second)

    def test_protocol_fixes_samples_display_and_predeclared_failure_gates(self):
        protocol = example().make_protocol()
        self.assertEqual(protocol["roughness"], [0.18, 0.52])
        self.assertEqual(protocol["sampling"]["samples_per_pixel"], 1024)
        self.assertEqual(protocol["sampling"]["denoiser"], "none")
        self.assertFalse(protocol["sampling"]["arbitrary_seed_control"])
        self.assertEqual(protocol["profile"]["columns"], [192, 448])
        self.assertEqual(protocol["profile"]["rows"], [238, 242])
        self.assertEqual(protocol["metric_domain"], "public sRGB8 / 255; equal RGB-channel mean")
