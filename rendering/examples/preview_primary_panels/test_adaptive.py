"""Contract for adaptive area-light quadrature on the primary surface."""
import unittest

from adaptive import pilot_grid, adapt_studio_source


class AdaptivePanelTests(unittest.TestCase):
    def test_primary_refines_only_a_partially_visible_pilot(self):
        self.assertEqual([pilot_grid(mask, primary=True) for mask in range(16)], [2] + [8] * 14 + [2])

    def test_secondary_keeps_original_grid(self):
        for mask in range(16):
            self.assertEqual(pilot_grid(mask, primary=False), 2)

    def test_out_of_range_mask_and_non_boolean_receiver_rejected(self):
        for mask in (-1, 16, False):
            with self.assertRaises((TypeError, ValueError)):
                pilot_grid(mask, primary=True)
        with self.assertRaises(TypeError):
            pilot_grid(3, primary=1)

    def test_different_upstream_integrator_is_rejected(self):
        with self.assertRaises(ValueError):
            adapt_studio_source("void main() {}")
