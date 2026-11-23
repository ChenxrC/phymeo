"""Public guided-first-batch contracts; no GPU fallback or implicit filtering."""
import ctypes as ct
import unittest

from phymeo_rendering.backends import UnsupportedRequestError
from phymeo_rendering.backends.native import NativeBackend


OPTIONS = dict(resident=True, shadow_mode='ray_query', indirect_mode='diffuse_bounce', progressive=True)


class SpatialContract(unittest.TestCase):
    def test_default_and_explicit_admission(self):
        self.assertEqual(NativeBackend().spatial_filter, 'none')
        for value in (None, True, 1):
            with self.assertRaises(TypeError):
                NativeBackend(spatial_filter=value)
        for value in ('cpu', 'GUIDED', ''):
            with self.assertRaises(ValueError):
                NativeBackend(spatial_filter=value)
        with self.assertRaises(UnsupportedRequestError):
            NativeBackend(spatial_filter='guided')
        for bounces in (1, 2):
            backend = NativeBackend(spatial_filter='guided', indirect_bounces=bounces, **OPTIONS)
            self.assertEqual(backend.spatial_filter, 'guided')

    def test_optional_extension_is_untouched_by_default(self):
        from phymeo_rendering.backends._native_spatial import configure_spatial
        class Untouchable:
            def __getattr__(self, name):
                raise AssertionError(name)
        self.assertIsNone(configure_spatial(Untouchable(), 'none'))

    def test_shader_selection_is_explicit_and_vertex_is_unchanged(self):
        from phymeo_rendering.backends._native_ray_query import RayQueryResidentDevice
        device = object.__new__(RayQueryResidentDevice)
        device._area_light_grid = 4
        device._progressive_requested = True
        device._indirect_requested = True
        device._reflection_requested = True
        device._normal_maps_requested = True
        device._spatial_requested = True
        self.assertEqual(device._shader_filename('frag'), 'resident_rayquery_normal_reflection_indirect_progressive_area4.frag.spv')
        self.assertEqual(device._shader_filename('vert'), 'resident_rayquery_normal.vert.spv')
        self.assertEqual(device._spatial_guide_filename(), 'resident_rayquery_normal_area4.frag.spv')
        device._spatial_requested = False
        self.assertEqual(device._shader_filename('frag'), 'resident_rayquery_normal_reflection_indirect_progressive_area4.frag.spv')

    def test_abi_layout(self):
        from phymeo_rendering.backends._native_spatial import SpatialInfo
        self.assertEqual(ct.sizeof(SpatialInfo), 64)
        self.assertEqual(SpatialInfo.filtered_rgb_captures.offset, 32)


if __name__ == '__main__':
    unittest.main()
