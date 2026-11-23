"""Opt-in real MSAA/center-sensor compatibility, not a quality-oracle replacement.

Set PHYMEO_NATIVE_MSAA_DLL and PHYMEO_NATIVE_MSAA_OLD_DLL (and optional
PHYMEO_NATIVE_DEVICE_INDEX). All captures use the public session. The old
library is only a compatibility and explicit-rejection control.
"""
from dataclasses import replace
import hashlib
import math
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.backends import BackendUnavailableError
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import DirectionalLight, FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene
from phymeo_rendering.studio import StudioLighting
from test_native_ray_query_device import quad

SENSORS=(Channel.FORWARD_DEPTH,Channel.INSTANCE_ID,Channel.VALID)


@unittest.skipUnless(os.environ.get('PHYMEO_NATIVE_MSAA_DLL'),'explicit MSAA device execution slot')
class NativeMsaaDevice(unittest.TestCase):
    def backend(self,samples,**extra):
        options=dict(library_path=Path(os.environ['PHYMEO_NATIVE_MSAA_DLL']),
            device_index=int(os.environ.get('PHYMEO_NATIVE_DEVICE_INDEX','1')),
            validation=True,resident=True,msaa_samples=samples)
        options.update(extra)
        options.setdefault('shader_directory',Path(options['library_path']).parent)
        return NativeBackend(**options)

    def assert_closed(self,owner):
        device=owner.diagnostics()['device']
        for name in ('validation_errors','validation_warnings','live_resources','live_contexts','live_frames'):
            self.assertEqual(device[name],0,name)
        telemetry=device['resident_telemetry']
        self.assertEqual(telemetry['device_local_bytes_current'],0)
        self.assertEqual(telemetry['host_visible_bytes_current'],0)
        self.assertEqual(telemetry['allocation_count'],telemetry['free_count'])

    @staticmethod
    def capture(owner,frame,camera,channels):
        return {p.channel:p.data for p in owner.capture(frame,CaptureRequest(camera,RenderPath.RASTER,channels)).planes}

    def test_half_coverage_linear_resolve_center_sensors_resize_and_variants(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'white.gltf'; quad(path,1.,1.)
            scene=VisualScene('msaa-edge',(Mesh('white',str(path)),),
                (Material('white',(1.,)*3,0.,1.),),(Instance(7,'white','white','white'),),
                (DirectionalLight((0.,0.,-1.),(100.,)*3),))
            key=FrameKey(scene.key,'world',0,0,0,'authored')
            frame=SceneFrame(key,0.,(NodePose('white',Pose((1.,0.,-2.))),))
            # Left edge x=0 projects to x=16.5: standard 4x sample locations
            # give exactly 2/4 white coverage, against a real black clear.
            camera=Camera('half',key,Pose(),33,25,20.,20.,16.5,12.5,.1,5.)
            baseline=self.backend(1).open_scene(scene)
            try: center=self.capture(baseline,frame,camera,tuple(Channel))
            finally: baseline.close()
            self.assert_closed(baseline)
            for mode in ('software','ray_query'):
                for grid in (2,4,8):
                    owner=self.backend(4,shadow_mode=mode,area_light_grid=grid).open_scene(scene)
                    try:
                        rgb=self.capture(owner,frame,camera,(Channel.RGB,))
                        mixed=self.capture(owner,frame,camera,tuple(Channel))
                        sensors=self.capture(owner,frame,camera,SENSORS)
                        self.assertEqual(rgb[Channel.RGB],mixed[Channel.RGB])
                        for channel in SENSORS:
                            self.assertEqual(center[channel],mixed[channel])
                            self.assertEqual(center[channel],sensors[channel])
                        at=(12*33+16)*3
                        self.assertEqual(tuple(rgb[Channel.RGB][at:at+3]),(188,188,188))
                        self.assertEqual(tuple(rgb[Channel.RGB][at+6:at+9]),(255,255,255))
                        self.assertEqual(tuple(rgb[Channel.RGB][at-6:at-3]),(0,0,0))
                        # Target recreation and return to the original extent.
                        self.capture(owner,frame,replace(camera,width=29,height=19),(Channel.RGB,))
                        again=self.capture(owner,frame,camera,(Channel.RGB,))
                        self.assertEqual(again[Channel.RGB],rgb[Channel.RGB])
                        info=owner.diagnostics()['device']['resident_msaa']
                        self.assertEqual(info['color_samples'],4); self.assertEqual(info['sensor_samples'],1)
                        self.assertEqual(info['color_passes'],4); self.assertEqual(info['center_passes'],2)
                        self.assertEqual(info['resolve_passes'],4); self.assertEqual(info['target_rebuilds'],3)
                        self.assertEqual(info['standard_sample_locations'],1)
                        # Reconfiguration after accepted captures is a real ABI negative.
                        raw=owner._delegate.device
                        self.assertNotEqual(raw._lib.pr_resident_set_msaa(raw._handle,1,None,0,None,0),0)
                    finally: owner.close()
                    self.assert_closed(owner)

    def test_old_dll_default_bytes_old_shaders_and_explicit_rejections(self):
        old=os.environ.get('PHYMEO_NATIVE_MSAA_OLD_DLL')
        self.assertTrue(old,'old DLL is required for the compatibility control')
        scene=VisualScene('old-empty',(),(),())
        key=FrameKey(scene.key,'world',0,0,0,'authored'); frame=SceneFrame(key,0.,())
        camera=Camera('odd',key,Pose(),13,7,10.,10.,6.2,3.1,.1,5.)
        results=[]
        for dll in (old,os.environ['PHYMEO_NATIVE_MSAA_DLL']):
            owner=self.backend(1,library_path=Path(dll),studio=StudioLighting(panel_strength=0.,environment_radiance=.125)).open_scene(scene)
            try: results.append(self.capture(owner,frame,camera,tuple(Channel)))
            finally: owner.close()
            self.assert_closed(owner)
        self.assertEqual(results[0],results[1])
        with self.assertRaisesRegex(BackendUnavailableError,'MSAA'):
            self.backend(4,library_path=Path(old)).open_scene(scene)
        with self.assertRaisesRegex(BackendUnavailableError,'MSAA'):
            self.backend(4,shader_directory=Path(old).parent).open_scene(scene)
        # New DLL still accepts the old shader for an explicit 1x session.
        owner=self.backend(1,shader_directory=Path(old).parent).open_scene(scene)
        try: self.capture(owner,frame,camera,tuple(Channel))
        finally: owner.close()
        self.assert_closed(owner)

    def test_default_real_geometry_bytes_match_old_dll(self):
        old=os.environ.get('PHYMEO_NATIVE_MSAA_OLD_DLL')
        self.assertTrue(old,'old DLL is required for the compatibility control')
        with TemporaryDirectory() as directory:
            path=Path(directory)/'surface.gltf'; quad(path,1.,1.)
            scene=VisualScene('default-surface',(Mesh('surface',str(path)),),
                (Material('surface',(.11,.32,.63),.4,.36),),
                (Instance(43,'surface','surface','surface'),),
                (DirectionalLight((.6,0.,-.8),(3.,2.,1.)),))
            key=FrameKey(scene.key,'world',0,0,0,'authored')
            frame=SceneFrame(key,0.,(NodePose('surface',Pose((.17,0.,-2.))),))
            camera=Camera('off-axis',key,Pose(),41,29,28.,31.,19.1,15.4,.1,5.)
            for mode in ('software','ray_query'):
                for transform in ('clip','pbr-neutral'):
                    results=[]
                    for dll in (old,os.environ['PHYMEO_NATIVE_MSAA_DLL']):
                        owner=self.backend(1,library_path=Path(dll),shadow_mode=mode,
                            display_transform=transform).open_scene(scene)
                        try: results.append(self.capture(owner,frame,camera,tuple(Channel)))
                        finally: owner.close()
                        self.assert_closed(owner)
                    self.assertEqual(results[0],results[1],(mode,transform))

    def test_uniform_background_neutral_and_real_hdri_sensor_only(self):
        from phymeo_rendering.environment import RadianceEnvironment
        scene=VisualScene('msaa-sky',(),(),())
        key=FrameKey(scene.key,'world',0,0,0,'authored'); frame=SceneFrame(key,0.,())
        camera=Camera('sky',key,Pose(),17,11,30.,30.,8.1,5.7,.1,5.)
        with TemporaryDirectory() as directory:
            hdr=Path(directory)/'uniform.hdr'
            raw=b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n'+bytes((128,64,32,131))*32
            hdr.write_bytes(raw)
            env=RadianceEnvironment(str(hdr),hashlib.sha256(raw).hexdigest(),face_size=16)
            for transform in ('clip','pbr-neutral'):
                for mapped in (False,True):
                    studio=StudioLighting(panel_strength=0.,environment_map=env) if mapped else StudioLighting(panel_strength=0.,environment_radiance=.125)
                    results=[]
                    for samples in (1,4):
                        owner=self.backend(samples,studio=studio,display_transform=transform,exposure=8.).open_scene(scene)
                        try:
                            result=self.capture(owner,frame,camera,tuple(Channel)); results.append(result)
                            sensors=self.capture(owner,frame,camera,SENSORS)
                            for channel in SENSORS: self.assertEqual(result[channel],sensors[channel])
                            self.assertEqual(result[Channel.VALID],bytes(17*11))
                            self.assertEqual(result[Channel.INSTANCE_ID],bytes(17*11*4))
                            self.assertEqual(result[Channel.FORWARD_DEPTH],struct.pack('<f',math.inf)*(17*11))
                            if mapped: self.assertEqual(owner.diagnostics()['device']['resident_environment_info']['sky_draws'],1)
                        finally: owner.close()
                        self.assert_closed(owner)
                    # FP16 display-linear storage adds bounded quantization only;
                    # this is a constant-field check, not an edge tolerance waiver.
                    self.assertLessEqual(max(abs(a-b) for a,b in zip(results[0][Channel.RGB],results[1][Channel.RGB],strict=True)),1)


if __name__=='__main__': unittest.main()
