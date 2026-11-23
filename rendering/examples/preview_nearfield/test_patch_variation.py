"""Check the spatial metric on known constant tiles and excluded boundaries."""
import unittest

import numpy as np

from .compare_patch_variation import block_errors


class PatchVariationTest(unittest.TestCase):
    def test_same_object_interiors_keep_signed_and_unsigned_error(self):
        labels = np.ones((8, 24), dtype=np.uint32)
        labels[:, 16:] = 2
        labels[0, 8] = 2
        reference = np.zeros((8, 24, 3))
        image = reference.copy()
        image[:, :8] = .25
        image[:, 8:] = .9
        value = block_errors(image, reference, labels, (1,))
        self.assertEqual(value["tiles"], 1)
        self.assertAlmostEqual(value["rms"], .25)
        self.assertAlmostEqual(value["mean_bias"], .25)
        self.assertAlmostEqual(value["spatial_std"], 0.)

    def test_opposing_errors_cannot_cancel_block_rms(self):
        labels = np.full((16, 8), 3)
        reference = np.full((16, 8, 3), .5)
        image = reference.copy()
        image[:8] -= .2
        image[8:] += .2
        value = block_errors(image, reference, labels, (3,))
        self.assertAlmostEqual(value["mean_bias"], 0.)
        self.assertAlmostEqual(value["rms"], .2)

    def test_empty_region_is_not_a_zero_error(self):
        labels = np.ones((8, 8), dtype=np.uint32)
        image = np.ones((8, 8, 3))
        with self.assertRaises(ValueError):
            block_errors(image, image, labels, (2,))
