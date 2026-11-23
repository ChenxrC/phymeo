"""Normal-map public admission and packing tests; no simulated GPU pixel claims."""
from dataclasses import replace
import ctypes as ct
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from phymeo_rendering.assets import Material, Mesh, TextureMap, TexturedMaterial
from phymeo_rendering.mesh_data import TriangleMesh
from phymeo_rendering.scene import VisualScene, Instance
from phymeo_rendering.texture_data import TextureData
from phymeo_rendering.backends import BackendUnavailableError, UnsupportedRequestError
from phymeo_rendering.backends.native import NativeBackend


def fixture():
    source=TextureMap('normal.png','a'*64)
    material=TexturedMaterial('mapped',(.4,.5,.6),.1,.5,normal_map=source)
    scene=VisualScene('normals',(Mesh('tri','unused.gltf'),),(material,Material('plain',(.4,.5,.6),.1,.5)),
        (Instance(1,'tri','mapped','one'),Instance(2,'tri','mapped','two'),Instance(3,'tri','plain','plain')))
    mesh=TriangleMesh(((0.,0.,0.),(1.,0.,0.),(0.,1.,0.)),((0.,0.,1.),)*3,((0,1,2),),'b'*64,
        ((0.,0.),(-1.,0.),(0.,1.)))
    image=TextureData(2,1,bytes((128,128,255,255,128,128)),'a'*64)
    return scene,mesh,image


class NativeNormalContract(unittest.TestCase):
    def test_last_optional_field_old_positional_compatibility_and_normal_only(self):
        source=TextureMap('normal.png','a'*64)
        old=TexturedMaterial('old',(1.,1.,1.),0.,.5,source)
        self.assertIsNone(old.normal_map)
        normal=TexturedMaterial('normal',(1.,1.,1.),0.,.5,None,None,source)
        self.assertEqual(normal.normal_map,source)
        for bad in (True,'normal.png',object()):
            with self.assertRaises(TypeError): TexturedMaterial('bad',(1.,)*3,0.,.5,normal_map=bad)
        with self.assertRaises(TypeError): TexturedMaterial('empty',(1.,)*3,0.,.5)

    def test_normal_only_does_not_allocate_color_or_roughness_upload(self):
        from phymeo_rendering.backends._native_texture import pack_textures
        scene,mesh,_=fixture()
        with patch('phymeo_rendering.texture_data.load_texture') as decode:
            self.assertIsNone(pack_textures(scene,{'tri':mesh}))
        decode.assert_not_called()

    def test_linear_bytes_mirrored_uv_and_shared_image_are_preserved(self):
        from phymeo_rendering.backends._native_normal import pack_normal_maps
        scene,mesh,image=fixture()
        with patch('phymeo_rendering.texture_data.load_texture',return_value=image) as decode:
            upload=pack_normal_maps(scene,{'tri':mesh})
        self.assertEqual(decode.call_count,1)
        self.assertEqual(bytes(upload.rgba),bytes((128,128,255,255,255,128,128,255)))
        self.assertEqual(tuple(upload.slots),(0,0,2**32-1))
        self.assertEqual([(p.u,p.v) for p in upload.uv],list(mesh.texcoords))
        self.assertEqual(upload.diagnostics['strength'],1.)
        self.assertEqual(upload.diagnostics['textures'][0]['source_sha256'],'a'*64)
        self.assertEqual(upload.diagnostics['encoding'],'linear RGB8 UNORM, normalize(2*c-1), glTF +Y')

    def test_missing_or_collapsed_uv_rejected_before_decode(self):
        from phymeo_rendering.backends._native_normal import pack_normal_maps
        scene,mesh,_=fixture()
        for values in (None,((0.,0.),)*3,((0.,0.),(1.,1.),(2.,2.))):
            with patch('phymeo_rendering.texture_data.load_texture') as decode:
                with self.assertRaisesRegex((ValueError,UnsupportedRequestError),'UV|texcoord'):
                    pack_normal_maps(scene,{'tri':replace(mesh,texcoords=values)})
            decode.assert_not_called()

    def test_fp32_uv_collapse_and_dimension_cap_are_explicit(self):
        from phymeo_rendering.backends._native_normal import pack_normal_maps
        scene,mesh,image=fixture()
        with patch('phymeo_rendering.texture_data.load_texture') as decode:
            with self.assertRaisesRegex((ValueError,UnsupportedRequestError),'UV|FP32'):
                pack_normal_maps(scene,{'tri':replace(mesh,texcoords=((1e8,1e8),(1e8+1,1e8),(1e8,1e8+1)))})
        decode.assert_not_called()
        image=TextureData(2049,1,bytes((128,128,255))*2049,'a'*64)
        with patch('phymeo_rendering.texture_data.load_texture',return_value=image):
            with self.assertRaisesRegex(UnsupportedRequestError,'dimension'):
                pack_normal_maps(scene,{'tri':mesh})

    def test_unmapped_scene_never_decodes_or_probes_optional_abi(self):
        from phymeo_rendering.backends._native_normal import pack_normal_maps, configure_normal_maps
        scene,mesh,_=fixture(); scene=replace(scene,materials=(scene.materials[1],),instances=(scene.instances[2],))
        class Untouchable:
            def __getattr__(self,name): raise AssertionError(name)
        with patch('phymeo_rendering.texture_data.load_texture') as decode:
            self.assertIsNone(pack_normal_maps(scene,{'tri':replace(mesh,texcoords=None)}))
            self.assertIsNone(configure_normal_maps(Untouchable(),None))
        decode.assert_not_called()

    def test_old_dll_rejects_requested_normal_and_info_layout(self):
        from phymeo_rendering.backends._native_normal import pack_normal_maps,configure_normal_maps,ResidentNormalInfo
        self.assertEqual(ct.sizeof(ResidentNormalInfo),128)
        scene,mesh,image=fixture()
        with patch('phymeo_rendering.texture_data.load_texture',return_value=image): upload=pack_normal_maps(scene,{'tri':mesh})
        with self.assertRaisesRegex(BackendUnavailableError,'normal'):
            configure_normal_maps(SimpleNamespace(_lib=SimpleNamespace()),upload)

    def test_legacy_rejects_normal_material_before_device_open(self):
        scene,mesh,_=fixture()
        with patch('phymeo_rendering.backends.native.NativeDevice') as device:
            with self.assertRaisesRegex(UnsupportedRequestError,'resident'):
                NativeBackend().open_scene(scene)
        device.assert_not_called()

    def test_declared_gltf_tangent_is_explicitly_rejected_not_ignored(self):
        import json
        from test_mesh_data import _document
        from phymeo_rendering.mesh_data import load_mesh
        with TemporaryDirectory() as directory:
            path=Path(directory)/'tangent.gltf'; document=_document()
            document['meshes'][0]['primitives'][0]['attributes']['TANGENT']=1
            path.write_text(json.dumps(document),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'attribute|TANGENT'):
                load_mesh(Mesh('tangent',str(path)))

    def test_normal_shader_variant_only_for_bound_map(self):
        from phymeo_rendering.backends._native_resident import ResidentDevice
        from test_native_resident import Recorder
        scene,mesh,_=fixture()
        for bound in (False,True):
            current=scene if bound else replace(scene,instances=(scene.instances[2],))
            with patch('phymeo_rendering.backends.native.require_library'), \
                 patch('phymeo_rendering.mesh_data.load_mesh',return_value=mesh), \
                 patch('phymeo_rendering.backends.native.ResidentDevice',return_value=Recorder()) as factory, \
                 patch('phymeo_rendering.backends.native.configure_normal_maps'), \
                 patch('phymeo_rendering.backends.native.pack_normal_maps'):
                owner=NativeBackend(resident=True).open_scene(current)
                self.assertEqual(factory.call_args.kwargs,{'normal_maps':True} if bound else {})
                owner.close()
        raw=object.__new__(ResidentDevice)
        for prefix in ('resident','resident_rayquery'):
            raw._shader_prefix=prefix
            for normal in (False,True):
                raw._normal_maps_requested=normal
                for grid in (2,4,8):
                    raw._area_light_grid=grid
                    stem=prefix+('_normal' if normal else '')
                    self.assertEqual(raw._shader_filename('vert'),stem+'.vert.spv')
                    self.assertEqual(raw._shader_filename('frag'),stem+('' if grid==2 else f'_area{grid}')+'.frag.spv')
    def test_public_upload_order_and_diagnostics(self):
        from test_native_resident import Recorder
        scene,mesh,image=fixture(); recorder=Recorder(); events=[]
        recorder.upload=lambda *args:events.append('geometry')
        def configure(device,upload):
            self.assertIs(device,recorder); self.assertIsNotNone(upload)
            events.append('normal'); return upload.diagnostics
        with patch('phymeo_rendering.backends.native.require_library'), \
             patch('phymeo_rendering.mesh_data.load_mesh',return_value=mesh), \
             patch('phymeo_rendering.texture_data.load_texture',return_value=image), \
             patch('phymeo_rendering.backends.native.ResidentDevice',return_value=recorder), \
             patch('phymeo_rendering.backends.native.configure_normal_maps',side_effect=configure):
            owner=NativeBackend(resident=True).open_scene(scene)
            self.assertEqual(events,['geometry','normal'])
            self.assertEqual(owner.diagnostics()['normal_maps']['strength'],1.)
            owner.close()
        self.assertTrue(recorder.closed)


if __name__=='__main__': unittest.main()
