import unittest

from model import local_weight, mix_hit


class LocalDiffuseTests(unittest.TestCase):
    def test_full_strength_contact_and_smooth_environment_boundary(self):
        self.assertEqual(local_weight(0, .12), 1.)
        self.assertAlmostEqual(local_weight(.06, .12), .5)
        self.assertEqual(local_weight(.12, .12), 0.)
        self.assertEqual(local_weight(10., .12), 0.)
        # Both sides of a ray's range boundary approach the identical sky value.
        self.assertLess(local_weight(.11999, .12), 1.e-7)

    def test_color_at_contact_and_far_field(self):
        sky=(1., 2., 3.)
        red=(4., 0., 0.)
        self.assertEqual(mix_hit(sky, red, 0., .12), red)
        self.assertEqual(mix_hit(sky, red, .12, .12), sky)
        self.assertEqual(mix_hit(sky, red, .06, .12), (2.5, 1., 1.5))

    def test_uniform_radiance_is_conserved_at_every_distance(self):
        for d in (0., .03, .06, .11, .12, 1.):
            self.assertEqual(mix_hit((2.,)*3, (2.,)*3, d, .12), (2.,)*3)

    def test_invalid_radius_and_nonfinite_inputs(self):
        for d, r in ((-.1,.12),(0.,0.),(0.,float('inf')),(float('nan'),.12)):
            with self.assertRaises(ValueError):
                local_weight(d,r)
