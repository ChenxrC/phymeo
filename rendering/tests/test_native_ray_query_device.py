"""Opt-in actual hardware/software shadows with identical public raster sensors."""
import base64
from dataclasses import replace
import json
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import DirectionalLight, FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene


def quad(path,half_x,half_y):
    positions=((-half_x,half_y,0.),(half_x,half_y,0.),(half_x,-half_y,0.),(-half_x,-half_y,0.))
    raw=struct.pack('<12f',*(v for point in positions for v in point))
    raw+=struct.pack('<12f',*((0.,0.,1.)*4))+struct.pack('<6H',0,2,1,0,3,2)
    document=dict(asset={'version':'2.0'},scene=0,scenes=[{'nodes':[0]}],nodes=[{'mesh':0}],
        meshes=[{'primitives':[{'attributes':{'POSITION':0,'NORMAL':1},'indices':2,'mode':4}]}],
        buffers=[{'byteLength':len(raw),'uri':'data:application/octet-stream;base64,'+base64.b64encode(raw).decode()}],
        bufferViews=[{'buffer':0,'byteOffset':offset,'byteLength':size} for offset,size in ((0,48),(48,48),(96,12))],
        accessors=[{'bufferView':0,'componentType':5126,'count':4,'type':'VEC3',
                    'min':[-half_x,-half_y,0.],'max':[half_x,half_y,0.]},
                   {'bufferView':1,'componentType':5126,'count':4,'type':'VEC3'},
                   {'bufferView':2,'componentType':5123,'count':6,'type':'SCALAR'}])
    path.write_text(json.dumps(document),encoding='utf-8')


@unittest.skipUnless(os.environ.get('PHYMEO_NATIVE_RAY_QUERY_DLL'),'set PHYMEO_NATIVE_RAY_QUERY_DLL for actual hardware visibility')
class RayQueryDevice(unittest.TestCase):
    def test_moving_blocker_two_cameras_and_sensor_equivalence(self):
        library=Path(os.environ['PHYMEO_NATIVE_RAY_QUERY_DLL'])
        shaders=Path(os.environ.get('PHYMEO_NATIVE_RAY_QUERY_SHADERS',str(library.parent)))
        with TemporaryDirectory() as root:
            root=Path(root); receiver=root/'receiver.gltf'; blocker=root/'blocker.gltf'
            quad(receiver,1.,1.); quad(blocker,.15,.3)
            scene=VisualScene('hardware-shadow',
                (Mesh('receiver',str(receiver)),Mesh('blocker',str(blocker))),
                (Material('gray',(.65,.65,.65),0.,1.),),
                (Instance(1,'receiver','gray','receiver'),Instance(2,'blocker','gray','blocker')),
                (DirectionalLight((.6,0.,-.8),(2.,2.,2.)),))
            frames=tuple(SceneFrame(FrameKey(scene.key,'world',0,index,index,'authored'),index/60.,
                (NodePose('receiver',Pose((0.,0.,-2.))),NodePose('blocker',Pose((x,0.,-1.5)))))
                for index,x in enumerate((-.375,1.5)))
            results={}
            for mode in ('software','ray_query'):
                owner=NativeBackend(library_path=library,shader_directory=shaders,
                    device_index=int(os.environ.get('PHYMEO_NATIVE_DEVICE_INDEX','0')),
                    validation=os.environ.get('PHYMEO_NATIVE_VALIDATION')=='1',resident=True,shadow_mode=mode).open_scene(scene)
                captures=[]
                try:
                    for frame in frames:
                        for camera_index,x in enumerate((0.,.15)):
                            camera=Camera(f'camera-{camera_index}',frame.key,Pose((x,0.,0.)),128,128,80.,80.,64.,64.,.1,5.)
                            result=owner.capture(frame,CaptureRequest(camera,RenderPath.RASTER,tuple(Channel)))
                            captures.append({plane.channel:plane.data for plane in result.planes})
                    results[mode]=captures
                    def brightness(index):
                        raw=captures[index][Channel.RGB]; start=(64*128+64)*3
                        return sum(raw[start:start+3])
                    self.assertGreater(brightness(2),brightness(0)+90)
                    if mode=='ray_query':
                        info=owner.diagnostics()['device']['resident_ray_query_info']
                        self.assertEqual(info['enabled'],1); self.assertEqual(info['blas_builds'],2)
                        self.assertEqual(info['tlas_builds'],1); self.assertEqual(info['tlas_updates'],3)
                        self.assertEqual(info['instance_upload_bytes'],4*2*64)
                        self.assertEqual(info['shadow_captures'],4)
                finally: owner.close()
                device=owner.diagnostics()['device']; telemetry=device['resident_telemetry']
                self.assertTrue(all(device[key]==0 for key in ('validation_errors','live_contexts','live_frames','live_resources')))
                self.assertEqual(telemetry['device_local_bytes_current'],0)
                self.assertEqual(telemetry['host_visible_bytes_current'],0)
                self.assertEqual(telemetry['allocation_count'],telemetry['free_count'])
            for software,hardware in zip(results['software'],results['ray_query'],strict=True):
                for channel in (Channel.FORWARD_DEPTH,Channel.INSTANCE_ID,Channel.VALID):
                    self.assertEqual(software[channel],hardware[channel])
                # Compare a point strictly inside the shadow/clear region, not an edge tolerance claim.
                start=(64*128+64)*3
                self.assertTrue(all(abs(a-b)<=1 for a,b in zip(software[Channel.RGB][start:start+3],hardware[Channel.RGB][start:start+3],strict=True)))


if __name__=='__main__': unittest.main()