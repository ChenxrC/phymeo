"""Explicit deterministic panel quadrature; recorders make no GPU quality claims."""
import ctypes as ct
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phymeo_rendering.backends import BackendUnavailableError, UnsupportedRequestError
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.backends._native_device import NativeDevice
from phymeo_rendering.backends._native_resident import ResidentDevice
from phymeo_rendering.backends._native_ray_query import RayQueryResidentDevice
from test_native_resident import fixture, Recorder


class AreaLightSelection(unittest.TestCase):
    def test_default_and_explicit_grid_contract(self):
        self.assertEqual(NativeBackend().area_light_grid, 2)
        for n in (2, 4, 8):
            self.assertEqual(NativeBackend(resident=True, area_light_grid=n).area_light_grid, n)
        for n in (0, 1, 3, 16):
            with self.subTest(n=n), self.assertRaises(ValueError):
                NativeBackend(resident=True, area_light_grid=n)
        for n in (True, 4., '4', None):
            with self.subTest(n=n), self.assertRaises(TypeError):
                NativeBackend(resident=True, area_light_grid=n)
        for n in (4, 8):
            with self.subTest(n=n), self.assertRaisesRegex(UnsupportedRequestError, 'resident'):
                NativeBackend(area_light_grid=n)

    def test_default_keeps_old_device_constructor_and_shader_names(self):
        scene, mesh = fixture()
        for mode, name, cls in (('software', 'ResidentDevice', ResidentDevice),
                                ('ray_query', 'RayQueryResidentDevice', RayQueryResidentDevice)):
            recorder = Recorder()
            with patch('phymeo_rendering.backends.native.require_library'), \
                 patch('phymeo_rendering.mesh_data.load_mesh', return_value=mesh), \
                 patch('phymeo_rendering.backends.native.' + name, return_value=recorder) as factory:
                owner = NativeBackend(resident=True, shadow_mode=mode).open_scene(scene)
                self.assertEqual(factory.call_args.kwargs, {})
                self.assertEqual(owner.diagnostics()['area_light_grid'], 2)
                owner.close()
            device = cls.__new__(cls)
            device._area_light_grid = 2
            self.assertEqual(device._shader_filename('vert'), cls._shader_prefix + '.vert.spv')
            self.assertEqual(device._shader_filename('frag'), cls._shader_prefix + '.frag.spv')

    def test_explicit_grid_selects_same_traversal_with_no_vertex_variant(self):
        scene, mesh = fixture()
        for mode, name, cls in (('software', 'ResidentDevice', ResidentDevice),
                                ('ray_query', 'RayQueryResidentDevice', RayQueryResidentDevice)):
            for n in (4, 8):
                with self.subTest(mode=mode, n=n):
                    recorder = Recorder()
                    with patch('phymeo_rendering.backends.native.require_library'), \
                         patch('phymeo_rendering.mesh_data.load_mesh', return_value=mesh), \
                         patch('phymeo_rendering.backends.native.' + name, return_value=recorder) as factory:
                        owner = NativeBackend(resident=True, shadow_mode=mode, area_light_grid=n).open_scene(scene)
                        self.assertEqual(factory.call_args.kwargs, {'area_light_grid': n})
                        info = owner.diagnostics()
                        self.assertEqual(info['area_light_grid'], n)
                        self.assertEqual(info['area_light_samples_per_panel'], n*n)
                        self.assertEqual(info['area_light_brdf_samples_per_panel'], 8)
                        self.assertEqual(info['shadow_mode'], mode)
                        owner.close()
                    device = cls.__new__(cls)
                    device._area_light_grid = n
                    self.assertEqual(device._shader_filename('vert'), cls._shader_prefix + '.vert.spv')
                    self.assertEqual(device._shader_filename('frag'), cls._shader_prefix + f'_area{n}.frag.spv')

    def test_missing_dense_fragment_does_not_load_default_or_initialize_device(self):
        # A valid-looking default SPIR-V exists deliberately: absence of the
        # explicitly selected file must fail before calling any native runtime.
        header = b'\x03\x02\x23\x07' + bytes(16)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / 'test.dll'
            library.write_bytes(b'not loaded')
            for prefix in ('resident', 'resident_rayquery'):
                (root / (prefix + '.vert.spv')).write_bytes(header)
                (root / (prefix + '.frag.spv')).write_bytes(header)
            for cls in (ResidentDevice, RayQueryResidentDevice):
                with self.subTest(cls=cls), patch.object(ct, 'CDLL') as loader:
                    with self.assertRaisesRegex(BackendUnavailableError, 'area8.frag.spv'):
                        cls(library, root, 0, False, area_light_grid=8)
                    loader.assert_not_called()

    def test_failed_variant_open_never_retries_default_grid(self):
        scene, mesh = fixture()
        with patch('phymeo_rendering.backends.native.require_library'), \
             patch('phymeo_rendering.mesh_data.load_mesh', return_value=mesh), \
             patch('phymeo_rendering.backends.native.ResidentDevice', side_effect=BackendUnavailableError('area4 missing')) as factory, \
             patch('phymeo_rendering.backends.native.NativeDevice') as legacy:
            with self.assertRaisesRegex(BackendUnavailableError, 'area4 missing'):
                NativeBackend(resident=True, area_light_grid=4).open_scene(scene)
            factory.assert_called_once()
            legacy.assert_not_called()

    def test_legacy_shader_selection_is_unchanged(self):
        device = NativeDevice.__new__(NativeDevice)
        self.assertEqual(device._shader_filename('vert'), 'native.vert.spv')
        self.assertEqual(device._shader_filename('frag'), 'native.frag.spv')


if __name__ == '__main__':
    unittest.main()
