"""Native texture upload/public capability contracts; no simulated pixel claims."""
import ctypes as ct
from dataclasses import replace
import unittest
from unittest.mock import patch

from phymeo_rendering.backends import BackendUnavailableError, UnsupportedRequestError
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.scene import VisualScene, Instance
from phymeo_rendering.mesh_data import TriangleMesh
from test_native_resident import Recorder


def fixture():
    from phymeo_rendering.assets import TextureMap, TexturedMaterial
    from phymeo_rendering.texture_data import TextureData
    texture=TextureMap('checked.png','a'*64)
    material=TexturedMaterial('paint',(1.,.5,.25),.2,.4,texture)
    scene=VisualScene('textured',(Mesh('quad','unused.gltf'),),
        (material,Material('plain',(.2,.3,.4),0.,.5)),
        (Instance(11,'quad','paint','first'),Instance(12,'quad','paint','second'),
         Instance(13,'quad','plain','third')))
    mesh=TriangleMesh(((0.,0.,0.),(1.,0.,0.),(0.,1.,0.)),((0.,0.,1.),)*3,((0,1,2),),'b'*64,
        ((-.25,.25),(1.25,.25),(.25,1.25)))
    # Two rows, each with a distinct orientation marker. Do not flip or gamma encode.
    data=TextureData(2,2,bytes((255,0,0, 0,255,0, 0,0,255, 255,255,255)),'a'*64)
    return scene,mesh,data


class NativeTexturePacking(unittest.TestCase):
    def test_shared_asset_uploaded_once_and_rows_uv_and_instance_slots_are_preserved(self):
        from phymeo_rendering.backends._native_texture import pack_textures
        scene,mesh,data=fixture()
        with patch('phymeo_rendering.texture_data.load_texture',return_value=data) as decode:
            upload=pack_textures(scene,{'quad':mesh})
        self.assertEqual(decode.call_count,1)
        self.assertEqual(len(upload.records),1)
        record=upload.records[0]
        self.assertEqual((record.width,record.height,record.first_byte,record.reserved),(2,2,0,0))
        self.assertEqual(bytes(upload.rgba),bytes((255,0,0,255,0,255,0,255,0,0,255,255,255,255,255,255)))
        self.assertEqual([(v.u,v.v) for v in upload.uv],list(mesh.texcoords))
        self.assertEqual(tuple(upload.slots),(0,0,2**32-1))
        self.assertEqual(upload.diagnostics['textures'][0]['source_sha256'],'a'*64)
        self.assertEqual(upload.diagnostics['padded_array_bytes'],4*4*4)

    def test_missing_uv_is_rejected_before_decode_or_device(self):
        from phymeo_rendering.backends._native_texture import pack_textures
        scene,mesh,_=fixture()
        with patch('phymeo_rendering.texture_data.load_texture') as decode:
            with self.assertRaisesRegex(UnsupportedRequestError,'UV|texcoord'):
                pack_textures(scene,{'quad':replace(mesh,texcoords=None)})
        decode.assert_not_called()

    def test_native_dimension_and_fp32_uv_limits_are_explicit(self):
        from phymeo_rendering.backends._native_texture import pack_textures
        from phymeo_rendering.texture_data import TextureData
        scene,mesh,data=fixture()
        too_wide=TextureData(2049,1,bytes(2049*3),'a'*64)
        with patch('phymeo_rendering.texture_data.load_texture',return_value=too_wide):
            with self.assertRaisesRegex(UnsupportedRequestError,'2048|dimension'):
                pack_textures(scene,{'quad':mesh})
        with patch('phymeo_rendering.texture_data.load_texture',return_value=data):
            with self.assertRaisesRegex(ValueError,'FP32'):
                pack_textures(scene,{'quad':replace(mesh,texcoords=((1e100,0.),)*3)})

    def test_device_texture_count_and_padded_array_budgets_fail_before_packing(self):
        from phymeo_rendering.backends._native_texture import pack_textures
        scene,mesh,data=fixture()
        with patch('phymeo_rendering.texture_data.load_texture') as decode:
            with self.assertRaisesRegex(UnsupportedRequestError,'count'):
                pack_textures(scene,{'quad':mesh},{'resident_texture_info':{'max_textures':0}})
        decode.assert_not_called()
        with patch('phymeo_rendering.texture_data.load_texture',return_value=data):
            with self.assertRaisesRegex(UnsupportedRequestError,'padded'):
                pack_textures(scene,{'quad':mesh},{'resident_texture_info':{'max_array_bytes':63}})
    def test_plain_scene_never_decodes_texture_or_requires_optional_exports(self):
        from phymeo_rendering.backends._native_texture import pack_textures
        scene,mesh,_=fixture()
        scene=replace(scene,materials=(scene.materials[1],),instances=(scene.instances[2],))
        with patch('phymeo_rendering.texture_data.load_texture') as decode:
            self.assertIsNone(pack_textures(scene,{'quad':replace(mesh,texcoords=None)}))
        decode.assert_not_called()

    def test_legacy_rejects_requested_texture_before_opening_a_device(self):
        scene,mesh,_=fixture()
        with patch('phymeo_rendering.backends.native.require_library'), \
             patch('phymeo_rendering.mesh_data.load_mesh',return_value=mesh), \
             patch('phymeo_rendering.backends.native.NativeDevice',return_value=Recorder()) as device:
            with self.assertRaisesRegex(UnsupportedRequestError,'resident'):
                NativeBackend().open_scene(scene)
        device.assert_not_called()

    def test_public_upload_order_is_geometry_then_texture_without_extra_geometry_upload(self):
        scene,mesh,data=fixture(); recorder=Recorder(); events=[]
        recorder.upload=lambda *args: events.append('geometry')
        recorder.set_textures=lambda value: events.append(('texture',tuple(value.slots)))
        with patch('phymeo_rendering.backends.native.require_library'), \
             patch('phymeo_rendering.mesh_data.load_mesh',return_value=mesh), \
             patch('phymeo_rendering.texture_data.load_texture',return_value=data), \
             patch('phymeo_rendering.backends.native.ResidentDevice',return_value=recorder):
            owner=NativeBackend(resident=True).open_scene(scene)
            self.assertEqual(events,['geometry',('texture',(0,0,2**32-1))])
            self.assertTrue(owner.diagnostics()['textures'])
            owner.close()
        self.assertTrue(recorder.closed)


class NativeTextureABI(unittest.TestCase):
    def owner(self):
        from phymeo_rendering.backends._native_resident import ResidentDevice
        owner=ResidentDevice.__new__(ResidentDevice)
        owner._handle=123; owner._uploaded=True; owner._has_captured=False; owner._textures_set=False
        owner._texture_function=None
        return owner

    def test_setter_requires_real_export_and_open_uploaded_unrendered_owner(self):
        from phymeo_rendering.backends._native_texture import pack_textures
        scene,mesh,data=fixture()
        with patch('phymeo_rendering.texture_data.load_texture',return_value=data):
            upload=pack_textures(scene,{'quad':mesh})
        owner=self.owner()
        with self.assertRaises(BackendUnavailableError): owner.set_textures(upload)
        calls=[]; owner._texture_function=lambda *args: calls.append(args) or 0
        owner._uploaded=False
        with self.assertRaises(RuntimeError): owner.set_textures(upload)
        owner._uploaded=True; owner._has_captured=True
        with self.assertRaises(RuntimeError): owner.set_textures(upload)
        owner._has_captured=False; owner.set_textures(upload)
        self.assertEqual(len(calls),1)
        self.assertEqual((calls[0][2],calls[0][4],calls[0][6],calls[0][8]),(1,16,3,3))
        with self.assertRaises(RuntimeError): owner.set_textures(upload)

    def test_texture_info_layout_and_semantics_are_independent(self):
        from phymeo_rendering.backends._native_texture import ResidentTextureInfo, ResidentTexture, ResidentUV
        value=ResidentTextureInfo(); value.texture_abi_version=1; value.texture_count=2
        value.image_upload_bytes=256; value.metadata_upload_bytes=80; value.rgb_captures_with_textures=3
        owner=self.owner(); owner._remember_texture_info(value)
        self.assertEqual((ct.sizeof(ResidentTexture),ct.sizeof(ResidentUV),ct.sizeof(value)),(16,8,128))
        self.assertEqual(owner._texture_info['image_upload_bytes'],256)
        self.assertIn('not visible',owner._texture_info['capture_scope'])
        value.texture_abi_version=99
        with self.assertRaisesRegex(RuntimeError,'texture ABI'): owner._remember_texture_info(value)


if __name__=='__main__': unittest.main()