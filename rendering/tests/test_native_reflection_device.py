"""Actual RTX nearest-hit/identity/UV/reset checks. No screen-space surrogate."""
import base64
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from phymeo_rendering.assets import Material,Mesh,TextureMap,TexturedMaterial
from phymeo_rendering.backends import BackendUnavailableError
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest,RenderPath
from phymeo_rendering.scene import DirectionalLight,FrameKey,Instance,NodePose,Pose,SceneFrame,VisualScene
from test_native_texture_device import quad

SENSORS=(Channel.FORWARD_DEPTH,Channel.INSTANCE_ID,Channel.VALID)

@unittest.skipUnless(os.environ.get('PHYMEO_NATIVE_REFLECTION_DLL'),'explicit reflection device slot')
class ReflectionDevice(unittest.TestCase):
    def backend(self,mode='single_bounce',**extra):
        library=Path(os.environ['PHYMEO_NATIVE_REFLECTION_DLL'])
        options=dict(library_path=library,shader_directory=library.parent,
            resident=True,shadow_mode='ray_query',reflection_mode=mode,
            device_index=int(os.environ.get('PHYMEO_NATIVE_DEVICE_INDEX','1')),validation=True)
        options.update(extra);return NativeBackend(**options)

    def close(self,owner):
        owner.close(); info=owner.diagnostics()['device']; telemetry=info['resident_telemetry']
        for key in ('validation_errors','validation_warnings','live_contexts','live_resources','live_frames'):
            self.assertEqual(info[key],0,key)
        self.assertEqual(telemetry['device_local_bytes_current'],0)
        self.assertEqual(telemetry['host_visible_bytes_current'],0)
        self.assertEqual(telemetry['allocation_count'],telemetry['free_count'])

    def fixture(self,root):
        mirror=root/'mirror.gltf';target=root/'target.gltf';quad(mirror);quad(target)
        document=json.loads(target.read_text()); raw=bytearray(base64.b64decode(document['buffers'][0]['uri'].split(',')[1]))
        positions=((- .25,.25,0.),(.25,.25,0.),(.25,-.25,0.),(-.25,-.25,0.))
        raw[:48]=struct.pack('<12f',*(x for p in positions for x in p));raw[48:96]=struct.pack('<12f',*((0.,0.,-1.)*4))
        document['buffers'][0]['uri']='data:application/octet-stream;base64,'+base64.b64encode(raw).decode()
        document['accessors'][0].update(min=[-.25,-.25,0.],max=[.25,.25,0.]);target.write_text(json.dumps(document))
        scene=VisualScene('reflection',(Mesh('mirror',str(mirror)),Mesh('target',str(target))),
            (Material('mirror',(.9,)*3,1.,.04),Material('red',(.8,.005,.005),0.,1.),Material('green',(.005,.8,.005),0.,1.)),
            (Instance(7,'mirror','mirror','mirror'),Instance(11,'target','red','red'),Instance(12,'target','green','green')),
            (DirectionalLight((-.8,0.,.6),(3.,)*3),DirectionalLight((.8,0.,.6),(3.,)*3)))
        return scene

    def frame(self,revision=0,x=1.,blocked=False):
        key=FrameKey('reflection','world',0,revision,revision,'authored')
        value=SceneFrame(key,float(revision),(NodePose('mirror',Pose()),NodePose('red',Pose((x,0.,-1.))),
            NodePose('green',Pose((x*2.75/3.,0.,-1.25) if blocked else (10.,0.,-1.25)))))
        camera=Camera('view',key,Pose(),128,96,80.,80.,64.,48.,.1,10.)
        return value,camera

    def shot(self,owner,frame,camera,channels=tuple(Channel)):
        return {p.channel:p.data for p in owner.capture(frame,CaptureRequest(camera,RenderPath.RASTER,channels)).planes}

    def test_directional_only_offscreen_identity_nearest_moving_and_reopen(self):
        with TemporaryDirectory() as directory:
            scene=self.fixture(Path(directory)); images={}
            for mode in ('environment','single_bounce'):
                owner=self.backend(mode).open_scene(scene); images[mode]=[]
                try:
                    for index,(x,blocked) in enumerate(((1.,False),(-1.,False),(1.,True))):
                        frame,camera=self.frame(index,x,blocked);image=self.shot(owner,frame,camera);images[mode].append(image)
                    if mode=='single_bounce':
                        info=owner.diagnostics()['device']['resident_reflection_info']
                        self.assertEqual((info['enabled'],info['receiver_count'],info['max_bounces'],info['captures'],info['rgb_captures']),(1,1,1,3,3))
                finally:self.close(owner)
            for left,right in zip(images['environment'],images['single_bounce']):
                for channel in SENSORS:self.assertEqual(left[channel],right[channel])
            # Target center is outside the original camera (x=144 or -16),
            # but its virtual image projects to 90.67 or 37.33. Interior only.
            for index,x in enumerate((90,37,90)):
                offset=(48*128+x)*3;rgb=images['single_bounce'][index][Channel.RGB][offset:offset+3]
                component=1 if index==2 else 0
                self.assertGreater(rgb[component],max(rgb[(component+1)%3],rgb[(component+2)%3])+40,rgb)
                self.assertNotEqual(rgb,images['environment'][index][Channel.RGB][offset:offset+3])
            for image,baseline in zip(images['single_bounce'],images['environment']):
                center=(48*128+64)*3
                self.assertEqual(image[Channel.RGB][center:center+3],baseline[Channel.RGB][center:center+3])
            owner=self.backend().open_scene(scene)
            try:self.assertEqual(self.shot(owner,*self.frame()),images['single_bounce'][0])
            finally:self.close(owner)

    def test_texture_uv_normal_variant_msaa_and_explicit_old_rejection(self):
        from PIL import Image
        with TemporaryDirectory() as directory:
            root=Path(directory);scene=self.fixture(root)
            path=root/'colors.png';Image.frombytes('RGB',(2,1),bytes((255,0,0,0,255,0))).save(path)
            texture=TextureMap(str(path),hashlib.sha256(path.read_bytes()).hexdigest())
            path=root/'flat.png';Image.frombytes('RGB',(1,1),bytes((128,128,255))).save(path)
            normal=TextureMap(str(path),hashlib.sha256(path.read_bytes()).hexdigest())
            scene=replace(scene,materials=(TexturedMaterial('mirror',(.9,)*3,1.,.04,normal_map=normal),
                TexturedMaterial('red',(.8,)*3,0.,1.,texture),scene.materials[2]))
            for samples in (1,4):
                outputs=[]
                for mode in ('environment','single_bounce'):
                    owner=self.backend(mode,msaa_samples=samples,area_light_grid=4).open_scene(scene)
                    try:outputs.append(self.shot(owner,*self.frame()))
                    finally:self.close(owner)
                for channel in SENSORS:self.assertEqual(outputs[0][channel],outputs[1][channel])
                # Different interior UVs on the same reflected triangle pair.
                rgb=outputs[1][Channel.RGB];left=rgb[(48*128+87)*3:(48*128+87)*3+3];right=rgb[(48*128+94)*3:(48*128+94)*3+3]
                self.assertGreater(left[0],left[1]+20,left);self.assertGreater(right[1],right[0]+20,right)
            old=Path(os.environ['PHYMEO_NATIVE_REFLECTION_OLD_DLL'])
            with self.assertRaisesRegex(BackendUnavailableError,'reflection'):
                self.backend(library_path=old).open_scene(scene)

    def test_geometric_face_guard_with_tilted_vertex_normals_and_grazing_miss(self):
        import math
        from phymeo_rendering.studio import StudioLighting
        with TemporaryDirectory() as directory:
            root=Path(directory)
            for nx in (.7,.8):
                scene=self.fixture(root); path=Path(scene.meshes[0].uri)
                document=json.loads(path.read_text());raw=bytearray(base64.b64decode(document['buffers'][0]['uri'].split(',')[1]))
                raw[48:96]=struct.pack('<12f',*((nx,0.,math.sqrt(1-nx*nx))*4))
                document['buffers'][0]['uri']='data:application/octet-stream;base64,'+base64.b64encode(raw).decode();path.write_text(json.dumps(document))
                scene=replace(scene,instances=(scene.instances[0],))
                outputs=[]
                for mode in ('environment','single_bounce'):
                    owner=self.backend(mode,studio=StudioLighting(environment_radiance=.2,panel_strength=0.)).open_scene(scene)
                    try:outputs.append(self.shot(owner,*self.frame()))
                    finally:self.close(owner)
                # nx=.7 is a near-grazing outward miss. nx=.8 reflects below
                # the true z-plane and must be rejected, not hit this same face.
                for y in range(42,54):
                    for x in range(58,70):
                        offset=(y*128+x)*3
                        self.assertEqual(outputs[0][Channel.RGB][offset:offset+3],outputs[1][Channel.RGB][offset:offset+3],(nx,x,y))
                for channel in SENSORS:self.assertEqual(outputs[0][channel],outputs[1][channel])

    def test_zero_receivers_enabled_and_boundary_factors_match_gpu_admission(self):
        with TemporaryDirectory() as directory:
            scene=self.fixture(Path(directory))
            for metallic,roughness,count in ((0.,1.,0),(.99,.10,1)):
                actual=replace(scene,materials=(replace(scene.materials[0],metallic=metallic,roughness=roughness),*scene.materials[1:]))
                images=[]
                for mode in ('environment','single_bounce'):
                    owner=self.backend(mode).open_scene(actual)
                    try:
                        images.append(self.shot(owner,*self.frame()))
                        if mode=='single_bounce':
                            info=owner.diagnostics()['device']['resident_reflection_info']
                            self.assertEqual((info['enabled'],info['receiver_count']),(1,count))
                    finally:self.close(owner)
                if count==0:self.assertEqual(images[0],images[1])
                else:
                    for y in range(44,52):
                        for x in range(87,95):
                            p=(y*128+x)*3;rgb=images[1][Channel.RGB][p:p+3]
                            self.assertGreater(rgb[0],max(rgb[1:])+30,(x,y,rgb))

    def test_remaining_grid_normal_variants_with_studio_hdr_and_msaa(self):
        from PIL import Image
        from phymeo_rendering.environment import RadianceEnvironment
        from phymeo_rendering.studio import StudioLighting
        with TemporaryDirectory() as directory:
            root=Path(directory);scene=self.fixture(root)
            hdr=root/'environment.hdr'
            raw=b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n'+bytes((128,80,40,126))*32
            hdr.write_bytes(raw);environment=RadianceEnvironment(str(hdr),hashlib.sha256(raw).hexdigest(),face_size=16)
            studio=StudioLighting(center_m=(0.,0.,-1.),panel_strength=.2,environment_map=environment)
            path=root/'flat.png';Image.frombytes('RGB',(1,1),bytes((128,128,255))).save(path)
            normal=TextureMap(str(path),hashlib.sha256(path.read_bytes()).hexdigest())
            for mapped,grid in ((False,4),(False,8),(True,2),(True,8)):
                actual=scene if not mapped else replace(scene,materials=(TexturedMaterial('mirror',(.9,)*3,1.,.04,normal_map=normal),*scene.materials[1:]))
                values=[]
                for mode in ('environment','single_bounce'):
                    owner=self.backend(mode,studio=studio,msaa_samples=4,area_light_grid=grid).open_scene(actual)
                    try:
                        values.append(self.shot(owner,*self.frame()))
                        self.assertEqual(owner.diagnostics()['device']['resident_environment_info']['sky_draws'],1)
                    finally:self.close(owner)
                for channel in SENSORS:self.assertEqual(values[0][channel],values[1][channel])
                self.assertNotEqual(values[0][Channel.RGB],values[1][Channel.RGB])

if __name__=='__main__':unittest.main()
