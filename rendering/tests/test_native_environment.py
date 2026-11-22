"""Native directional environment seams; preprocessing is the shared implementation."""
import ctypes as ct
from dataclasses import replace
import hashlib
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from phymeo_rendering.backends import BackendUnavailableError
from phymeo_rendering.backends.native import NativeBackend
from test_native_resident import fixture, Recorder


def data():
    from phymeo_rendering.environment import NativeEnvironmentData
    side=16; levels=5; count=sum((side>>level)**2*6 for level in range(levels))
    sh=struct.pack('<36f',1.,.5,.25,0.,*([0.]*32))
    return NativeEnvironmentData(side,levels,struct.pack('<4f',.2,.3,.4,1.)*count,sh,
        {'source_sha256':'a'*64,'orientation':'world.xzy; SH world.xyz'})


class EnvironmentPacking(unittest.TestCase):
    def test_config_and_bytes_preserve_face_mip_order_and_signed_sh(self):
        from phymeo_rendering.backends._native_environment import pack_environment, ResidentEnvironment, ResidentEnvironmentInfo
        value=data(); sh=list(struct.unpack('<36f',value.diffuse_sh_f32le)); sh[4]=-.1
        value=replace(value,diffuse_sh_f32le=struct.pack('<36f',*sh))
        upload=pack_environment(value)
        self.assertEqual(ct.sizeof(ResidentEnvironment),160)
        self.assertEqual(ct.sizeof(ResidentEnvironmentInfo),128)
        self.assertEqual((upload.config.side,upload.config.mip_levels),(16,5))
        self.assertEqual(bytes(upload.rgba),value.rgba_f32le)
        self.assertAlmostEqual(upload.config.diffuse_sh[1][0],-.1)
        self.assertEqual(upload.diagnostics['rgba_sha256'],hashlib.sha256(value.rgba_f32le).hexdigest())

    def test_incomplete_chain_or_payload_and_nonfinite_values_are_rejected(self):
        from phymeo_rendering.backends._native_environment import pack_environment
        value=data()
        for changes in ({'mip_levels':4},{'rgba_f32le':value.rgba_f32le[:-4]},
                        {'diffuse_sh_f32le':value.diffuse_sh_f32le[:-4]},
                        {'rgba_f32le':struct.pack('<f',float('nan'))+value.rgba_f32le[4:]},
                        {'rgba_f32le':struct.pack('<f',101.)+value.rgba_f32le[4:]}):
            with self.subTest(changes=list(changes)):
                with self.assertRaises((TypeError,ValueError)): pack_environment(replace(value,**changes))

    def test_alpha_and_sh_padding_are_not_silently_changed(self):
        from phymeo_rendering.backends._native_environment import pack_environment
        value=data()
        with self.assertRaises(ValueError):
            pack_environment(replace(value,rgba_f32le=value.rgba_f32le[:12]+struct.pack('<f',0.)+value.rgba_f32le[16:]))
        with self.assertRaises(ValueError):
            pack_environment(replace(value,diffuse_sh_f32le=value.diffuse_sh_f32le[:12]+struct.pack('<f',1.)+value.diffuse_sh_f32le[16:]))


class EnvironmentLifecycle(unittest.TestCase):
    def owner(self,root):
        from phymeo_rendering.backends._native_resident import ResidentDevice
        owner=ResidentDevice.__new__(ResidentDevice)
        owner._handle=123; owner._uploaded=True; owner._has_captured=False; owner._environment_set=False
        owner._environment_function=None; owner.shader_directory=Path(root); owner.shader_sha256={}
        return owner

    def test_missing_export_closed_and_after_capture_are_explicit(self):
        from phymeo_rendering.backends._native_environment import pack_environment
        with TemporaryDirectory() as root:
            owner=self.owner(root); upload=pack_environment(data())
            with self.assertRaises(BackendUnavailableError): owner.set_environment(upload)
            owner._environment_function=lambda *args: self.fail('must reject before native call')
            owner._has_captured=True
            with self.assertRaises(RuntimeError): owner.set_environment(upload)
            owner._has_captured=False; owner._handle=None
            with self.assertRaises(RuntimeError): owner.set_environment(upload)

    def test_shader_bytes_are_explicit_and_hash_bound_and_set_only_once(self):
        from phymeo_rendering.backends._native_environment import pack_environment
        with TemporaryDirectory() as root:
            owner=self.owner(root); upload=pack_environment(data()); calls=[]
            owner._environment_function=lambda *args: calls.append(args) or 0
            with self.assertRaises(BackendUnavailableError): owner.set_environment(upload)
            raw=struct.pack('<5I',0x07230203,0x10000,0,1,0)
            for stage in ('vert','frag'): (Path(root)/f'resident_sky.{stage}.spv').write_bytes(raw)
            owner.set_environment(upload)
            self.assertEqual(len(calls),1)
            self.assertEqual((calls[0][3],calls[0][5],calls[0][7]),(len(data().rgba_f32le),20,20))
            self.assertEqual(owner.shader_sha256['environment_sky_vert'],hashlib.sha256(raw).hexdigest())
            with self.assertRaises(RuntimeError): owner.set_environment(upload)

    def test_public_studio_prepares_once_and_preserves_panels(self):
        from phymeo_rendering.environment import RadianceEnvironment
        from phymeo_rendering.studio import StudioLighting
        scene,mesh=fixture(); recorder=Recorder(); events=[]
        base_info=recorder.info
        recorder.info=lambda: dict(base_info(),resident_environment_available=True)
        recorder.set_studio=lambda value: events.append('studio')
        recorder.set_environment=lambda value: events.append('environment')
        declaration=RadianceEnvironment('unused.hdr','a'*64,face_size=16)
        studio=StudioLighting(environment_map=declaration,panel_strength=2.)
        with patch('phymeo_rendering.backends.native.require_library'), \
             patch('phymeo_rendering.mesh_data.load_mesh',return_value=mesh), \
             patch('phymeo_rendering.backends.native.ResidentDevice',return_value=recorder), \
             patch('phymeo_rendering.environment.prepare_native_environment',return_value=data()) as prepare:
            owner=NativeBackend(resident=True,studio=studio).open_scene(scene)
            prepare.assert_called_once_with(declaration)
            self.assertEqual(events,['studio','environment'])
            self.assertEqual(owner.diagnostics()['environment']['source_sha256'],'a'*64)
            self.assertFalse(owner.diagnostics()['studio']['environment']['active_lighting'])
            owner.close()

    def test_public_missing_capability_does_not_preprocess_or_fall_back(self):
        from phymeo_rendering.environment import RadianceEnvironment
        from phymeo_rendering.studio import StudioLighting
        scene,mesh=fixture(); recorder=Recorder()
        recorder.set_studio=lambda value: None
        studio=StudioLighting(environment_map=RadianceEnvironment('unused.hdr','a'*64,face_size=16))
        with patch('phymeo_rendering.backends.native.require_library'), \
             patch('phymeo_rendering.mesh_data.load_mesh',return_value=mesh), \
             patch('phymeo_rendering.backends.native.ResidentDevice',return_value=recorder), \
             patch('phymeo_rendering.environment.prepare_native_environment') as prepare:
            with self.assertRaises(BackendUnavailableError): NativeBackend(resident=True,studio=studio).open_scene(scene)
        prepare.assert_not_called(); self.assertTrue(recorder.closed)


if __name__=='__main__': unittest.main()