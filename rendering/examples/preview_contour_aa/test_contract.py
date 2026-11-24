import unittest

import numpy as np

from .calibrate import coverage_errors


class AnalyticCoverage(unittest.TestCase):
    def test_single_partial_pixel_is_scored_in_linear_radiance(self):
        area = np.array([[0., .375, 1.]])
        center = np.array([[0., 0., 1.]])
        solid = np.array([[1., 8., 2.]])
        resolved = np.array([[0., 2., 2.]])
        result = coverage_errors(resolved, center, solid, area)
        self.assertEqual(result["edge_pixels"], 1)
        self.assertEqual(result["mae_resolve"], .125)
        self.assertEqual(result["mae_center"], .375)

    def test_undefined_calibration_cannot_pass(self):
        with self.assertRaises(ValueError):
            coverage_errors(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), [[.5]])
