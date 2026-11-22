"""Opt-in real Vulkan texture/sensor check; no recorder can satisfy this test."""
import base64
import hashlib
import json
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from dataclasses import replace

from phymeo_rendering.assets import Material, Mesh, TextureMap, TexturedMaterial
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import DirectionalLight, FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene


def quad(path):
    positions=((-1.,1.,-2.),(1.,1.,-2.),(1.,-1.,-2.),(-1.,-1.,-2.))
    normals=((0.,0.,1.),)*4; uv=((0.,0.),(1.,0.),(1.,1.),(0.,1.))
    raw=struct.pack('<12f',*(v for point in positions for v in point))
    raw+=struct.pack('<12f',*(v for point in normals for v in point))
    raw+=struct.pack('<8f',*(v for point in uv for v in point))
    raw+=struct.pack('<6H',0,2,1,0,3,2)
    data=dict(asset={'version':'2.0'},scene=0,scenes=[{'nodes':[0]}],nodes=[{'mesh':0}],
        meshes=[{'primitives':[{'attributes':{'POSITION':0,'NORMAL':1,'TEXCOORD_0':2},'indices':3,'mode':4}]}],
        buffers=[{'byteLength':len(raw),'uri':'data:application/octet-stream;base64,'+base64.b64encode(raw).decode()}],
        bufferViews=[{'buffer':0,'byteOffset':offset,'byteLength':size} for offset,size in ((0,48),(48,48),(96,32),(128,12))],
        accessors=[{'bufferView':i,'componentType':5126,'count':4,'type':kind} for i,kind in enumerate(('VEC3','VEC3','VEC2'))]+
            [{'bufferView':3,'componentType':5123,'count':6,'type':'SCALAR'}])
    data['accessors'][0].update(min=[-1.,-1.,-2.],max=[1.,1.,-2.])
    path.write_text(json.dumps(data),encoding='utf-8')


@unittest.skipUnless(os.environ.get('PHYMEO_NATIVE_TEXTURE_DLL'),'set PHYMEO_NATIVE_TEXTURE_DLL for a real texture-capable Vulkan DLL')
class NativeTextureDevice(unittest.TestCase):
    def test_source_orientation_color_and_every_compact_channel_combination(self):
        from PIL import Image
        library=Path(os.environ['PHYMEO_NATIVE_TEXTURE_DLL'])
        shaders=Path(os.environ.get('PHYMEO_NATIVE_TEXTURE_SHADERS',str(library.parent)))
        backend=NativeBackend(library_path=library,shader_directory=shaders,
            device_index=int(os.environ.get('PHYMEO_NATIVE_DEVICE_INDEX','0')),
            validation=os.environ.get('PHYMEO_NATIVE_VALIDATION')=='1',resident=True)
        with TemporaryDirectory() as directory:
            directory=Path(directory); image=directory/'orientation.png'; mesh=directory/'quad.gltf'
            Image.frombytes('RGB',(2,2),bytes((255,0,0,0,255,0,0,0,255,255,255,255))).save(image)
            quad(mesh)
            material=TexturedMaterial('paint',(1.,1.,1.),0.,1.,TextureMap(str(image),hashlib.sha256(image.read_bytes()).hexdigest()))
            scene=VisualScene('texture-device',(Mesh('quad',str(mesh)),),(material,),
                (Instance(1,'quad','paint','quad'),),(DirectionalLight((0.,0.,-1.),(2.,2.,2.)),))
            key=FrameKey(scene.key,'world',0,0,0,'authored')
            frame=SceneFrame(key,0.,(NodePose('quad',Pose()),))
            camera=Camera('front',key,Pose(),128,128,64.,64.,64.,64.,.1,5.)
            request=CaptureRequest(camera,RenderPath.RASTER,tuple(Channel))
            owner=backend.open_scene(scene)
            try:
                result=owner.capture(frame,request); full={p.channel:p.data for p in result.planes}
                self.assertTrue(owner.diagnostics()['device']['resident_texture_available'])
                rgb=full[Channel.RGB]
                def color(x,y): return tuple(rgb[(y*128+x)*3:(y*128+x+1)*3])
                for x,y,index in ((48,48,0),(80,48,1),(48,80,2)):
                    value=color(x,y)
                    self.assertGreater(value[index],max(value[j] for j in range(3) if j!=index)+30,
                        (x,y,value,full[Channel.VALID][y*128+x],struct.unpack_from('<f',full[Channel.FORWARD_DEPTH],4*(y*128+x))[0],owner.diagnostics()['device']['resident_texture_info']))
                self.assertLess(max(color(80,80))-min(color(80,80)),5)
                for mask in range(1,16):
                    channels=tuple(reversed(tuple(c for i,c in enumerate(Channel) if mask&(1<<i))))
                    actual=owner.capture(frame,replace(request,channels=channels))
                    self.assertEqual(tuple(p.channel for p in actual.planes),channels)
                    self.assertTrue(all(p.data==full[p.channel] for p in actual.planes))
                self.assertEqual({p.channel:p.data for p in result.planes},full)
                info=owner.diagnostics()['device']['resident_texture_info']
                self.assertEqual(info['upload_calls'],1)
                self.assertEqual(info['texture_count'],1)
                self.assertEqual(info['rgb_captures_with_textures'],9)
            finally: owner.close()
            final=owner.diagnostics()['device']; telemetry=final['resident_telemetry']
            self.assertEqual(final['validation_errors'],0)
            self.assertEqual(telemetry['device_local_bytes_current'],0)
            self.assertEqual(telemetry['allocation_count'],telemetry['free_count'])
            plain=replace(scene,materials=(Material('paint',(1.,1.,1.),0.,1.),))
            owner=backend.open_scene(plain)
            try:
                result=owner.capture(frame,request)
                for plane in result.planes:
                    if plane.channel is not Channel.RGB: self.assertEqual(plane.data,full[plane.channel])
            finally: owner.close()


if __name__=='__main__': unittest.main()