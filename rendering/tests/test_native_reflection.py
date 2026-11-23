"""Single-bounce reflection admission/ABI checks; no simulated image oracle."""
import ctypes as ct
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from phymeo_rendering.assets import Material, TextureMap, TextureChannel, TexturedMaterial
from phymeo_rendering.backends import BackendUnavailableError, UnsupportedRequestError
from phymeo_rendering.backends.native import NativeBackend
from test_native_resident import fixture, Recorder


class ReflectionContract(unittest.TestCase):
    def test_explicit_mode_requires_resident_hardware_and_strict_enum(self):
        self.assertEqual(NativeBackend().reflection_mode,'environment')
        for value in (None,True,1):
            with self.assertRaises(TypeError): NativeBackend(reflection_mode=value)
        with self.assertRaises(ValueError): NativeBackend(reflection_mode='recursive')
        for kwargs in ({},{'resident':True}):
            with self.assertRaisesRegex(UnsupportedRequestError,'ray_query'):
                NativeBackend(reflection_mode='single_bounce',**kwargs)
        self.assertEqual(NativeBackend(resident=True,shadow_mode='ray_query',reflection_mode='single_bounce').reflection_mode,'single_bounce')

    def test_material_gate_uses_uploaded_factor_not_roughness_image(self):
        from phymeo_rendering.backends._native_reflection import reflection_declaration
        scene,_=fixture(); source=TextureMap('unread.png','a'*64)
        for metallic,roughness,admitted in ((1.,0.,True),(.99,.10,True),(.98,.05,False),(1.,.11,False)):
            material=TexturedMaterial('paint',(.8,)*3,metallic,roughness,roughness_map=TextureChannel(source))
            current=replace(scene,materials=(material,))
            with patch('phymeo_rendering.texture_data.load_texture') as decode:
                if admitted:
                    receipt=reflection_declaration(current,'single_bounce')
                    self.assertEqual([x['instance_id'] for x in receipt['receiver_instances']],[11,12])
                    self.assertEqual(receipt['bounces'],1)
                else:
                    self.assertEqual(reflection_declaration(current,'single_bounce')['receiver_instances'],[])
                decode.assert_not_called()

    def test_unused_eligible_material_does_not_satisfy_scene_gate(self):
        from phymeo_rendering.backends._native_reflection import reflection_declaration
        scene,_=fixture(); scene=replace(scene,materials=(*scene.materials,Material('unused',(1.,)*3,1.,0.)))
        receipt=reflection_declaration(scene,'single_bounce')
        self.assertEqual(receipt['mode'],'single_bounce')
        self.assertEqual(receipt['receiver_instances'],[])

    def test_default_does_not_access_optional_abi(self):
        from phymeo_rendering.backends._native_reflection import reflection_declaration,configure_reflection
        scene,_=fixture(); declaration=reflection_declaration(scene,'environment')
        class Untouchable:
            def __getattr__(self,name): raise AssertionError(name)
        self.assertEqual(configure_reflection(Untouchable(),declaration),declaration)

    def test_old_library_rejected_and_info_layout(self):
        from phymeo_rendering.backends._native_reflection import configure_reflection,ResidentReflectionInfo
        self.assertEqual(ct.sizeof(ResidentReflectionInfo),64)
        with self.assertRaisesRegex(BackendUnavailableError,'reflection'):
            configure_reflection(SimpleNamespace(_lib=SimpleNamespace()),dict(mode='single_bounce'))

    def test_variant_names_keep_original_vertex_and_default_fragments(self):
        from phymeo_rendering.backends._native_resident import ResidentDevice
        raw=object.__new__(ResidentDevice); raw._shader_prefix='resident_rayquery'
        for normal in (False,True):
            raw._normal_maps_requested=normal
            prefix='resident_rayquery'+('_normal' if normal else '')
            for reflection in (False,True):
                raw._reflection_requested=reflection
                for grid in (2,4,8):
                    raw._area_light_grid=grid
                    self.assertEqual(raw._shader_filename('vert'),prefix+'.vert.spv')
                    self.assertEqual(raw._shader_filename('frag'),prefix+('_reflection' if reflection else '')+('' if grid==2 else f'_area{grid}')+'.frag.spv')

    def test_public_order_receipt_and_no_default_constructor_extension(self):
        scene,mesh=fixture(); scene=replace(scene,materials=(Material('paint',(.8,)*3,1.,.05),))
        for mode in ('environment','single_bounce'):
            recorder=Recorder(); events=[]; recorder.upload=lambda *args:events.append('geometry')
            def configure(device,declaration):
                self.assertIs(device,recorder)
                self.assertEqual(events,['geometry'])
                events.append('reflection');return declaration
            with patch('phymeo_rendering.backends.native.require_library'), \
                 patch('phymeo_rendering.mesh_data.load_mesh',return_value=mesh), \
                 patch('phymeo_rendering.backends.native.RayQueryResidentDevice',return_value=recorder) as factory, \
                 patch('phymeo_rendering.backends.native.configure_reflection',side_effect=configure):
                owner=NativeBackend(resident=True,shadow_mode='ray_query',reflection_mode=mode).open_scene(scene)
                self.assertEqual(factory.call_args.kwargs,{'reflection':True} if mode=='single_bounce' else {})
                self.assertEqual(owner.diagnostics()['reflection']['mode'],mode)
                owner.close()


if __name__=='__main__':unittest.main()
