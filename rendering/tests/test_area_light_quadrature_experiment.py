"""Reference integration and fixed-denominator quality accounting; no GPU invoked."""
import importlib
import math
from pathlib import Path
import sys
import unittest


def experiment():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
    return importlib.import_module('area_light_quadrature')


class AreaQuadratureReference(unittest.TestCase):
    def test_rectangle_closed_form_has_independent_solid_angle_limits(self):
        module = experiment()
        full = module.rectangle_irradiance(-1., 1., -1., 1., 1.)
        # Four equal boundary terms; an infinite irradiance hemisphere gives pi.
        expected = 4 / math.sqrt(2) * math.atan(1 / math.sqrt(2))
        self.assertAlmostEqual(full, expected, places=13)
        self.assertAlmostEqual(module.rectangle_irradiance(0., 1., -1., 1., 1.), full/2, places=13)
        self.assertAlmostEqual(module.rectangle_irradiance(-1e6, 1e6, -1e6, 1e6, 1.), math.pi, places=10)
        self.assertEqual(module.rectangle_irradiance(1., 1., -1., 1., 1.), 0.)
        with self.assertRaises(ValueError): module.rectangle_irradiance(1., -1., -1., 1., 1.)
        with self.assertRaises(ValueError): module.rectangle_irradiance(-1., 1., -1., 1., 0.)

    def test_clipped_emitter_accounts_for_actual_shadow_origin_bias(self):
        module = experiment()
        eps = 1e-5
        self.assertEqual(module.visible_emitter_bounds(0., bias_m=eps), (0., .5))
        self.assertEqual(module.visible_emitter_bounds(-.6, bias_m=eps), (.5, .5))
        self.assertEqual(module.visible_emitter_bounds(.6, bias_m=eps), (-.5, .5))
        t = (.5-eps)/(1-eps)
        bound = module.visible_emitter_bounds(.2, bias_m=eps)[0]
        self.assertAlmostEqual(bound, .2*(1-1/t), places=15)
        self.assertNotEqual(bound, -.2)

    def test_uniform_midpoints_have_fixed_total_area_and_no_emitter_edge_samples(self):
        module = experiment()
        self.assertEqual(module.area_coordinates(2), ((-.5, -.5), (.5, -.5), (-.5, .5), (.5, .5)))
        for n in (2, 4, 8):
            points = module.area_coordinates(n)
            self.assertEqual(len(points), n*n)
            self.assertEqual(len(set(points)), n*n)
            self.assertEqual(sum(x for x, _ in points), 0.)
            self.assertTrue(all(-1 < x < 1 and -1 < y < 1 for x, y in points))
        with self.assertRaises(ValueError): module.area_coordinates(3)

    def test_error_denominator_is_clear_reference_peak_not_local_visibility(self):
        module = experiment()
        metrics = module.profile_metrics([0., .1, .2], [0., .2, .2], clear_peak=.5)
        self.assertAlmostEqual(metrics['normalized_mean_absolute_error'], (.1/3)/.5)
        self.assertAlmostEqual(metrics['normalized_max_absolute_error'], .2)
        self.assertAlmostEqual(metrics['normalized_max_adjacent_jump'], .2)
        with self.assertRaises(ValueError): module.profile_metrics([0.], [0.], clear_peak=0.)
        with self.assertRaises(ValueError): module.profile_metrics([0.], [0., 1.], clear_peak=1.)
        with self.assertRaises(ValueError): module.profile_metrics([math.nan], [0.], clear_peak=1.)

    def test_absolute_and_relative_acceptance_remain_separate(self):
        module = experiment()
        # Low absolute error is insufficient if it did not improve baseline by 50%.
        result = module.error_acceptance(.03, .04)
        self.assertTrue(result['absolute_error_passed'])
        self.assertFalse(result['relative_improvement_passed'])
        self.assertAlmostEqual(result['relative_improvement'], .25)
        zero = module.error_acceptance(0., 0.)
        self.assertIsNone(zero['relative_improvement'])
        self.assertFalse(zero['relative_improvement_passed'])
        self.assertTrue(module.error_acceptance(.03, .1)['relative_improvement_passed'])

    def test_full_material_reference_converges_and_is_not_a_lambert_surrogate(self):
        module = experiment()
        high, lambert = module._reference((-.3, 0., .3), 32)
        low, _ = module._reference((-.3, 0., .3), 16)
        clear, _ = module._reference((0.,), 32, clear=True)
        self.assertLess(max(abs(a-b) for a,b in zip(high,low)), 1e-8)
        self.assertAlmostEqual(high[1]*2, clear[0], places=10)
        self.assertNotAlmostEqual(high[1], lambert[1], places=6)
        self.assertTrue(all(v > 0 for v in high))

    def test_registration_pins_source_resolution_samples_and_counterexample_policy(self):
        record = experiment().load_registration()
        self.assertEqual(record['grids'], [2, 4, 8])
        self.assertEqual(record['brdf_samples_per_panel'], 8)
        self.assertEqual(record['analytic_scene']['resolution'], [641, 129])
        self.assertEqual(record['panda']['source_record_indices'], [479, 2159])
        self.assertEqual(record['cost']['timed_captures'], 20)
        self.assertEqual(record['acceptance_8']['mean_absolute_error_over_denominator_max'], .04)
        self.assertIn('counterexample', record['acceptance_8']['scope'])


if __name__ == '__main__':
    unittest.main()
