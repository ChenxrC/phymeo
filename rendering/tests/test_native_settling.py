"""Opt-in guided settling admission and exact registered weight schedule."""
import ctypes as ct
import unittest
from phymeo_rendering.backends import UnsupportedRequestError
from phymeo_rendering.backends.native import NativeBackend


class SettlingContract(unittest.TestCase):
    def test_explicit_public_mode_and_unchanged_default(self):
        self.assertEqual(NativeBackend().spatial_filter, 'none')
        with self.assertRaises(UnsupportedRequestError):
            NativeBackend(spatial_filter='guided_settling')
        backend = NativeBackend(resident=True, shadow_mode='ray_query', indirect_mode='diffuse_bounce',
                                progressive=True, spatial_filter='guided_settling')
        self.assertEqual(backend.spatial_filter, 'guided_settling')

    def test_one_based_schedule_and_exact_anchors(self):
        from phymeo_rendering.backends._native_spatial import settling_fraction
        import math
        self.assertEqual(settling_fraction(1), 1.)
        for n in range(2, 16):
            self.assertEqual(settling_fraction(n), 1.-math.sqrt((n-1)/15.))
        for n in (16, 17, 4096):
            self.assertEqual(settling_fraction(n), 0.)
        for n in (0, -1):
            with self.assertRaises(ValueError):
                settling_fraction(n)
        for n in (True, 1., None):
            with self.assertRaises(TypeError):
                settling_fraction(n)

    def test_optional_abi_layout(self):
        from phymeo_rendering.backends._native_spatial import SettlingInfo
        self.assertEqual(ct.sizeof(SettlingInfo), 64)
        self.assertEqual(SettlingInfo.last_filtered_fraction.offset, 16)
        self.assertEqual(SettlingInfo.guide_draws.offset, 32)


if __name__ == '__main__':
    unittest.main()
