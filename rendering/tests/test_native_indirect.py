"""Public one-diffuse-bounce admission; HOST recorders do not prove pixels."""
import ctypes as ct
from unittest.mock import patch
from types import SimpleNamespace
import unittest
from phymeo_rendering.backends import BackendUnavailableError,UnsupportedRequestError
from phymeo_rendering.backends.native import NativeBackend
from test_native_resident import fixture,Recorder


class IndirectContract(unittest.TestCase):
    def test_modes_and_uint32_seed_are_strict(self):
        self.assertEqual(NativeBackend().indirect_mode,'none')
        self.assertEqual(NativeBackend().indirect_seed,0)
        for seed in (True,False,1.,None,'1'):
            with self.assertRaises(TypeError):NativeBackend(indirect_seed=seed)
        for seed in (-1,2**32):
            with self.assertRaises(ValueError):NativeBackend(indirect_seed=seed)
        for mode in (None,True,1):
            with self.assertRaises(TypeError):NativeBackend(indirect_mode=mode)
        with self.assertRaises(ValueError):NativeBackend(indirect_mode='recursive')
        for kwargs in ({},{'resident':True}):
            with self.assertRaisesRegex(UnsupportedRequestError,'ray_query'):
                NativeBackend(indirect_mode='diffuse_bounce',**kwargs)

    def test_combination_is_explicit_and_high_seed_supported(self):
        backend=NativeBackend(resident=True,shadow_mode='ray_query',reflection_mode='single_bounce',
            indirect_mode='diffuse_bounce',indirect_seed=2**32-1)
        self.assertEqual(backend.indirect_seed,2**32-1)
        self.assertEqual(backend.reflection_mode,'single_bounce')

    def test_none_never_touches_new_optional_abi(self):
        from phymeo_rendering.backends._native_indirect import indirect_declaration,configure_indirect
        class Untouchable:
            def __getattr__(self,name):raise AssertionError(name)
        declaration=indirect_declaration('none',0)
        self.assertEqual(configure_indirect(Untouchable(),declaration),declaration)
        self.assertEqual(declaration['samples_per_pixel'],0)

    def test_estimator_declares_replacement_and_fixed_budget(self):
        from phymeo_rendering.backends._native_indirect import indirect_declaration
        value=indirect_declaration('diffuse_bounce',17)
        self.assertEqual((value['seed'],value['samples_per_pixel'],value['bounces']),(17,16,1))
        self.assertIn('replace',value['environment_policy'])
        self.assertIn('16',value['estimator'])
        self.assertIn('pixel',value['seed_scope'])

    def test_old_library_rejected_and_layout(self):
        from phymeo_rendering.backends._native_indirect import ResidentIndirectInfo,configure_indirect
        self.assertEqual(ct.sizeof(ResidentIndirectInfo),64)
        with self.assertRaisesRegex(BackendUnavailableError,'indirect'):
            configure_indirect(SimpleNamespace(_lib=SimpleNamespace()),{'mode':'diffuse_bounce','seed':0})

    def test_combined_shader_names_preserve_original_vertices(self):
        from phymeo_rendering.backends._native_resident import ResidentDevice
        value=object.__new__(ResidentDevice);value._shader_prefix='resident_rayquery'
        for normal in (False,True):
            value._normal_maps_requested=normal;prefix='resident_rayquery'+('_normal' if normal else '')
            for reflection in (False,True):
                value._reflection_requested=reflection
                for indirect in (False,True):
                    value._indirect_requested=indirect
                    for grid in (2,4,8):
                        value._area_light_grid=grid
                        self.assertEqual(value._shader_filename('vert'),prefix+'.vert.spv')
                        expected=prefix+('_reflection' if reflection else '')+('_indirect' if indirect else '')+('' if grid==2 else f'_area{grid}')+'.frag.spv'
                        self.assertEqual(value._shader_filename('frag'),expected)

    def test_public_default_constructor_compatibility_and_indirect_setup(self):
        scene,mesh=fixture()
        for mode in ('none','diffuse_bounce'):
            recorder=Recorder();events=[];recorder.upload=lambda *args:events.append('geometry')
            def configure(device,declaration):
                self.assertIs(device,recorder);self.assertEqual(events,['geometry'])
                events.append('indirect');return declaration
            with patch('phymeo_rendering.backends.native.require_library'), \
                 patch('phymeo_rendering.mesh_data.load_mesh',return_value=mesh), \
                 patch('phymeo_rendering.backends.native.RayQueryResidentDevice',return_value=recorder) as factory, \
                 patch('phymeo_rendering.backends.native.configure_indirect',side_effect=configure):
                owner=NativeBackend(resident=True,shadow_mode='ray_query',indirect_mode=mode).open_scene(scene)
                self.assertEqual(factory.call_args.kwargs,{'indirect':True} if mode!='none' else {})
                self.assertEqual(owner.diagnostics()['indirect']['mode'],mode);owner.close()

if __name__=='__main__':unittest.main()
