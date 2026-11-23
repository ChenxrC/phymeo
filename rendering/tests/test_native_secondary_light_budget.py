"""Secondary-light configuration/asset contracts; recorders are not GPU evidence."""
import ctypes as ct
import itertools
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from phymeo_rendering.backends import BackendUnavailableError, UnsupportedRequestError
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.backends._native_ray_query import RayQueryResidentDevice
from test_native_resident import fixture, Recorder


def marker_spirv(value=0x0802, spec_id=73, name='PR_SECONDARY_LIGHT_BUDGET'):
    raw=name.encode()+b'\0'; raw+=b'\0'*((-len(raw))%4)
    name_words=struct.unpack('<'+'I'*(len(raw)//4),raw)
    words=[0x07230203,0x10000,0,8,0,
           ((2+len(name_words))<<16)|5,3,*name_words,
           (4<<16)|71,3,1,spec_id,
           (4<<16)|21,2,32,0,
           (4<<16)|50,2,3,value]
    return struct.pack('<'+'I'*len(words),*words)


class SecondaryLightBudget(unittest.TestCase):
    def test_inherit_and_explicit_equal_keep_original_selection(self):
        for grid in (2,4,8):
            for secondary in (None,grid):
                backend=NativeBackend(resident=True,area_light_grid=grid,secondary_area_light_grid=secondary)
                self.assertEqual(backend.secondary_area_light_grid,grid)
        self.assertEqual(NativeBackend(secondary_area_light_grid=2).secondary_area_light_grid,2)

    def test_only_new_8_2_rayquery_combination_and_strict_types(self):
        self.assertEqual(NativeBackend(resident=True,shadow_mode='ray_query',area_light_grid=8,
            secondary_area_light_grid=2).secondary_area_light_grid,2)
        for value in (True,2.,'2'):
            with self.subTest(value=value),self.assertRaises(TypeError):
                NativeBackend(resident=True,area_light_grid=8,secondary_area_light_grid=value)
        for value in (0,1,3,16):
            with self.subTest(value=value),self.assertRaises(ValueError):
                NativeBackend(resident=True,area_light_grid=8,secondary_area_light_grid=value)
        for primary,secondary,shadow in ((2,4,'ray_query'),(4,2,'ray_query'),(8,4,'ray_query'),(8,2,'software')):
            with self.subTest(primary=primary,secondary=secondary,shadow=shadow),self.assertRaises(UnsupportedRequestError):
                NativeBackend(resident=True,shadow_mode=shadow,area_light_grid=primary,secondary_area_light_grid=secondary)

    def test_public_forwarding_and_diagnostics_do_not_change_equal_default_constructor(self):
        scene,mesh=fixture()
        for secondary,expected in ((None,{'area_light_grid':8}),(8,{'area_light_grid':8}),
                                   (2,{'area_light_grid':8,'secondary_area_light_grid':2})):
            recorder=Recorder()
            with patch('phymeo_rendering.backends.native.require_library'), \
                 patch('phymeo_rendering.mesh_data.load_mesh',return_value=mesh), \
                 patch('phymeo_rendering.backends.native.RayQueryResidentDevice',return_value=recorder) as factory:
                owner=NativeBackend(resident=True,shadow_mode='ray_query',area_light_grid=8,
                    secondary_area_light_grid=secondary).open_scene(scene)
                self.assertEqual(factory.call_args.kwargs,expected)
                info=owner.diagnostics()
                self.assertEqual(info['secondary_area_light_grid'],2 if secondary==2 else 8)
                self.assertEqual(info['secondary_area_light_samples_per_panel'],4 if secondary==2 else 64)
                self.assertEqual(info['area_light_brdf_samples_per_panel'],8)
                self.assertEqual(info['area_light_budget']['primary_scope'],'camera-visible surfaces and direct guide')
                owner.close()

    def test_twelve_secondary_fragments_share_existing_vertices_and_primary_guide(self):
        for normal,mode in itertools.product((False,True),('direct','reflection','indirect','combined','progressive','progressive_combined')):
            with self.subTest(normal=normal,mode=mode):
                device=RayQueryResidentDevice.__new__(RayQueryResidentDevice)
                device._area_light_grid=8;device._secondary_area_light_grid=2
                device._normal_maps_requested=normal
                device._reflection_requested=mode in ('reflection','combined','progressive_combined')
                device._indirect_requested=mode in ('indirect','combined','progressive','progressive_combined')
                device._progressive_requested=mode.startswith('progressive')
                prefix='resident_rayquery'+('_normal' if normal else '')
                suffix=('_reflection' if device._reflection_requested else '')+('_indirect' if device._indirect_requested else '')+('_progressive' if device._progressive_requested else '')
                self.assertEqual(device._shader_filename('vert'),prefix+'.vert.spv')
                self.assertEqual(device._shader_filename('frag'),prefix+suffix+'_area8_secondary2.frag.spv')
                self.assertEqual(device._spatial_guide_filename(),prefix+'_area8.frag.spv')

    def test_missing_mixed_fragment_never_loads_default_or_dll(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);lib=root/'unloaded.dll';lib.write_bytes(b'no DLL load')
            for name in ('resident_rayquery.vert.spv','resident_rayquery_area8.frag.spv'):
                (root/name).write_bytes(marker_spirv())
            with patch.object(ct,'CDLL') as loader,self.assertRaisesRegex(BackendUnavailableError,'secondary2'):
                RayQueryResidentDevice(lib,root,0,False,area_light_grid=8,secondary_area_light_grid=2)
            loader.assert_not_called()

    def test_wrong_or_missing_compiled_budget_marker_is_rejected_before_dll(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);lib=root/'unloaded.dll';lib.write_bytes(b'no DLL load')
            (root/'resident_rayquery.vert.spv').write_bytes(marker_spirv())
            for raw in (marker_spirv(value=0x0808),marker_spirv(spec_id=72),marker_spirv(name='OTHER')):
                (root/'resident_rayquery_area8_secondary2.frag.spv').write_bytes(raw)
                with self.subTest(raw=raw),patch.object(ct,'CDLL') as loader,self.assertRaisesRegex(BackendUnavailableError,'budget'):
                    RayQueryResidentDevice(lib,root,0,False,area_light_grid=8,secondary_area_light_grid=2)
                loader.assert_not_called()

    def test_compiled_marker_reader_accepts_only_named_budget_contract(self):
        from phymeo_rendering.backends._native_secondary import validate_secondary_shader
        self.assertEqual(validate_secondary_shader(marker_spirv()),{'primary_grid':8,'secondary_grid':2,'spec_id':73})
        for raw in (b'',marker_spirv()[:-1],marker_spirv(value=0),marker_spirv(spec_id=71)):
            with self.subTest(raw=raw),self.assertRaises(BackendUnavailableError):
                validate_secondary_shader(raw)

    def test_default_build_target_retains_the_complete_shader_output_list(self):
        cmake=(Path(__file__).resolve().parents[1]/'src/phymeo_rendering/backends/_native/CMakeLists.txt').read_text()
        self.assertIn('add_custom_target(native_shaders ALL DEPENDS ${shader_outputs})',cmake)
        self.assertIn('list(LENGTH shader_outputs shader_count)',cmake)
        self.assertIn('native_shader_manifest.txt',cmake)
        self.assertIn('${CMAKE_BINARY_DIR}/${prefix}${mode_suffix}_area8_secondary2.frag.spv',cmake)
        self.assertIn('${normal_options} ${mode_options}',cmake)

    def test_generated_variants_and_shared_integrator_are_shipped_as_source(self):
        root=Path(__file__).resolve().parents[1]
        cmake=(root/'src/phymeo_rendering/backends/_native/CMakeLists.txt').read_text()
        studio=(root/'src/phymeo_rendering/backends/_shaders/resident_studio.glsl').read_text()
        self.assertIn('PR_STUDIO_SECONDARY_GRID=2',cmake)
        self.assertIn('_area8_secondary2.frag.spv',cmake)
        self.assertIn('residentStudioRadianceSecondary',studio)
        self.assertIn('PR_SECONDARY_LIGHT_BUDGET',studio)
        self.assertEqual(studio.count('vec3 brdfCos ='),1)
        for filename in ('resident_reflection.glsl','resident_indirect.glsl'):
            text=(root/'src/phymeo_rendering/backends/_shaders'/filename).read_text()
            self.assertIn('residentStudioRadianceSecondary',text)
        package=(root/'pyproject.toml').read_text()
        self.assertIn('"_shaders/*"',package)
        self.assertIn('"_native/CMakeLists.txt"',package)


if __name__=='__main__':unittest.main()
