"""Regression cases for the optional spatial reconstruction study."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from progressive_first_frame_preview import reconstruct


class FirstFramePreviewTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((9, 11, 3), dtype=np.float32)
        self.depth = np.ones((9, 11), dtype=np.float32)
        self.ids = np.ones((9, 11), dtype=np.uint32)
        self.valid = np.ones((9, 11), dtype=np.uint8)
        self.guide = np.zeros_like(self.rgb, dtype=np.uint8)

    def render(self):
        return reconstruct(self.rgb, self.depth, self.ids, self.valid, self.guide)

    def test_constant_surfaces_stay_constant_including_image_borders(self):
        self.rgb[:] = (1.75, .25, .5)
        np.testing.assert_array_equal(self.render(), self.rgb)

    def test_no_transfer_across_instance_depth_or_direct_light_boundary(self):
        self.rgb[:, 5:] = (2., 1., .25)
        for field in (self.ids, self.depth, self.guide):
            with self.subTest(dtype=field.dtype):
                saved = field.copy()
                field[:, 5:] += 10
                np.testing.assert_array_equal(self.render(), self.rgb)
                field[:] = saved

    def test_background_remains_exact_and_never_illuminates_valid_neighbors(self):
        self.rgb[:, 0] = (7., 4., 2.)
        self.valid[:, 0] = 0
        self.ids[:, 0] = 0
        self.depth[:, 0] = np.inf
        np.testing.assert_array_equal(self.render(), self.rgb)

    def test_same_surface_filters_real_linear_values_without_wraparound(self):
        self.rgb[4, 0, 0] = 7.
        result = self.render()
        self.assertGreater(result[4, 0, 0], 1.)
        self.assertLess(result[4, 0, 0], 7.)
        self.assertGreater(result[4, 1, 0], 0.)
        self.assertEqual(result[4, -1, 0], 0.)
        self.assertEqual(self.rgb[4, 0, 0], 7.)
