"""Frozen independent mirror geometry and public-device acceptance contracts."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
sys.path.insert(0, str(EXAMPLES))
SPEC = importlib.util.spec_from_file_location('near_reflection_chart', EXAMPLES / 'near_reflection_chart.py')
chart = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chart)


def synthetic(labels):
    colors = ((45, 45, 45), (210, 12, 12), (12, 210, 12), (12, 12, 210))
    return bytes(c for label in labels for c in colors[0 if label == 255 else label])


class MirrorOracleTests(unittest.TestCase):
    def test_virtual_projection_is_independent_closed_form(self):
        self.assertEqual(chart.project((0., 0., 1.)), (160., 120.))
        x, y = chart.project((-.45, .75, 1.8))
        self.assertAlmostEqual(x, 160 - 256 * .45 / 2.75)
        self.assertAlmostEqual(y, 120 - 256 * .8 / 2.75)
        with self.assertRaises(ValueError):
            chart.project((0., -2., 1.))

    def test_fixed_source_is_completely_offscreen_but_virtual_image_inside(self):
        red = chart.fixture('base')[0]
        real = [chart.project(p) for p in chart.corners(red)]
        virtual = [chart.project((p[0], -p[1], p[2])) for p in chart.corners(red)]
        self.assertTrue(all(y < 0 for x, y in real))
        self.assertTrue(all(32 < x < 288 and 5 < y < 235 for x, y in virtual))
        self.assertGreater(chart.expected_labels('base').count(1), 400)

    def test_ray_reflection_and_virtual_projection_agree_at_all_pixel_centers(self):
        for state in ('base', 'moved', 'occluded', 'miss'):
            expected = chart.expected_labels(state)
            for y in range(chart.HEIGHT):
                for x in range(chart.WIDTH):
                    self.assertEqual(expected[y * chart.WIDTH + x], chart.ray_identity(x, y, chart.fixture(state)))

    def test_nearest_hit_beats_submission_order_and_far_blue(self):
        targets = chart.fixture('occluded')
        pixel = (118, 45)
        self.assertEqual(chart.ray_identity(*pixel, targets), 2)
        self.assertEqual(chart.ray_identity(*pixel, list(reversed(targets))), 2)
        self.assertEqual(chart.ray_identity(*pixel, [t for t in targets if t['id'] != 2]), 1)
        self.assertEqual(chart.ray_identity(*pixel, [t for t in targets if t['id'] == 3]), 3)

    def test_motion_updates_reflection_and_true_miss_has_no_color(self):
        base, moved = chart.expected_labels('base'), chart.expected_labels('moved')
        self.assertNotEqual(base, moved)
        self.assertGreater(sum(a == 1 and b == 0 for a, b in zip(base, moved)), 400)
        self.assertEqual(set(chart.expected_labels('miss')), {0, 255})

    def test_exact_synthetic_and_one_pixel_boundary_pass(self):
        labels = chart.expected_labels('base')
        self.assertTrue(chart.evaluate_rgb(synthetic(labels), labels)['passed'])
        shifted = bytearray(labels)
        for y in range(chart.HEIGHT):
            for x in range(chart.WIDTH - 1):
                if labels[y * chart.WIDTH + x] != 255:
                    shifted[y * chart.WIDTH + x] = labels[y * chart.WIDTH + x + 1]
        self.assertTrue(chart.evaluate_rgb(synthetic(shifted), labels)['passed'])

    def test_wrong_identity_missing_offscreen_hit_and_three_pixel_shift_fail(self):
        labels = chart.expected_labels('base')
        wrong = bytes(3 if x == 1 else x for x in labels)
        empty = bytes(0 if x != 255 else 255 for x in labels)
        self.assertFalse(chart.evaluate_rgb(synthetic(wrong), labels)['passed'])
        self.assertFalse(chart.evaluate_rgb(synthetic(empty), labels)['passed'])
        shifted = bytearray(labels)
        for y in range(chart.HEIGHT):
            for x in range(chart.WIDTH - 3):
                if labels[y * chart.WIDTH + x] != 255:
                    shifted[y * chart.WIDTH + x] = labels[y * chart.WIDTH + x + 3]
        self.assertFalse(chart.evaluate_rgb(synthetic(shifted), labels)['passed'])

    def test_nearest_occluder_cannot_be_replaced_by_farther_color(self):
        labels = chart.expected_labels('occluded')
        self.assertGreater(labels.count(2), 50)
        wrong = bytes(1 if x == 2 else x for x in labels)
        self.assertFalse(chart.evaluate_rgb(synthetic(wrong), labels)['passed'])

    def test_miss_comparison_excludes_only_predeclared_boundary_band(self):
        labels = chart.expected_labels('base')
        raw = synthetic(labels)
        self.assertTrue(chart.miss_equal(raw, raw, labels))
        changed = bytearray(raw)
        changed[3 * (120 * chart.WIDTH + 160)] ^= 1
        self.assertFalse(chart.miss_equal(raw, bytes(changed), labels))
        with self.assertRaises(ValueError):
            chart.evaluate_rgb(raw[:-1], labels)

    def test_sensor_comparison_is_exact_not_numeric_tolerance(self):
        before = {name: b'1234' for name in chart.SENSORS}
        after = dict(before)
        self.assertTrue(chart.sensor_equal(before, after))
        after['FORWARD_DEPTH'] = b'1235'
        self.assertFalse(chart.sensor_equal(before, after))


@unittest.skipUnless(os.environ.get('PHYMEO_NEAR_REFLECTION_DLL'), 'actual new ray-query reflection DLL required')
class MirrorDeviceTests(unittest.TestCase):
    def test_public_nearest_moving_offscreen_miss_reset_and_sensor_contract(self):
        with tempfile.TemporaryDirectory() as root:
            report = chart.run_experiment(Path(os.environ['PHYMEO_NEAR_REFLECTION_DLL']), Path(root) / 'result',
                device_index=int(os.environ.get('PHYMEO_VULKAN_DEVICE', '1')),
                shaders=Path(os.environ['PHYMEO_NEAR_REFLECTION_DLL']).parent, validation=True)
            self.assertEqual(report['status'], 'passed', report['acceptance'])
            self.assertTrue(all(report['acceptance'].values()))
            self.assertEqual(report['physics_steps'], 0)
            self.assertEqual(report['registration']['boundary_tolerance_px'], 1)


if __name__ == '__main__':
    unittest.main()
