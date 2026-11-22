"""Opt-in real public HDRI background and sensor-preservation test."""
from dataclasses import replace
import hashlib
import math
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import FrameKey, Pose, SceneFrame, VisualScene
from phymeo_rendering.studio import StudioLighting


@unittest.skipUnless(os.environ.get('PHYMEO_NATIVE_ENVIRONMENT_DLL'),'set PHYMEO_NATIVE_ENVIRONMENT_DLL for real directional environment capture')
class NativeEnvironmentDevice(unittest.TestCase):
    def test_real_panorama_yaw_changes_background_without_valid_depth_or_ids(self):
        from phymeo_rendering.environment import RadianceEnvironment
        library=Path(os.environ['PHYMEO_NATIVE_ENVIRONMENT_DLL'])
        shader_directory=Path(os.environ.get('PHYMEO_NATIVE_ENVIRONMENT_SHADERS',str(library.parent)))
        with TemporaryDirectory() as directory:
            path=Path(directory)/'direction.hdr'
            # Original authored linear RGBE panorama: red at -Y, blue at +Y.
            row=bytes((128,16,0,129))*4+bytes((0,16,128,129))*4
            raw=b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n'+row*4
            path.write_bytes(raw)
            declaration=RadianceEnvironment(str(path),hashlib.sha256(raw).hexdigest(),face_size=16)
            scene=VisualScene('directional-background',(),(),())
            key=FrameKey(scene.key,'world',0,0,0,'authored'); frame=SceneFrame(key,0.,())
            # Public camera looks along local -Z; +90 degrees about X looks +Y.
            camera=Camera('sky',key,Pose(orientation_xyzw=(math.sqrt(.5),0.,0.,math.sqrt(.5))),
                32,24,80.,80.,16.,12.,.1,5.)
            request=CaptureRequest(camera,RenderPath.RASTER,tuple(Channel)); results=[]
            for yaw in (0.,180.):
                backend=NativeBackend(library_path=library,shader_directory=shader_directory,
                    device_index=int(os.environ.get('PHYMEO_NATIVE_DEVICE_INDEX','0')),
                    validation=os.environ.get('PHYMEO_NATIVE_VALIDATION')=='1',resident=True,
                    studio=StudioLighting(panel_strength=0.,environment_map=replace(declaration,yaw_degrees=yaw)))
                owner=backend.open_scene(scene)
                try:
                    actual=owner.capture(frame,request); planes={p.channel:p.data for p in actual.planes}
                    self.assertEqual(planes[Channel.VALID],bytes(32*24))
                    self.assertEqual(planes[Channel.INSTANCE_ID],bytes(32*24*4))
                    self.assertEqual(planes[Channel.FORWARD_DEPTH],struct.pack('<f',math.inf)*(32*24))
                    self.assertTrue(any(planes[Channel.RGB]))
                    results.append(tuple(planes[Channel.RGB][(12*32+16)*3:(12*32+16)*3+3]))
                    info=owner.diagnostics()['device']['resident_environment_info']
                    self.assertEqual(info['upload_calls'],1)
                    self.assertEqual(info['sky_draws'],1)
                    self.assertEqual(info['axis_transform'],1)
                    self.assertEqual(info['diffuse_basis'],1)
                finally: owner.close()
                device=owner.diagnostics()['device']; telemetry=device['resident_telemetry']
                self.assertEqual(device['validation_errors'],0)
                self.assertEqual(telemetry['device_local_bytes_current'],0)
                self.assertEqual(telemetry['allocation_count'],telemetry['free_count'])
            self.assertGreater(results[0][2],results[0][0]+30)
            self.assertGreater(results[1][0],results[1][2]+30)


if __name__=='__main__': unittest.main()