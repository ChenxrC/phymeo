"""Opt-in physical one-diffuse-bounce checks on the actual Vulkan device."""
import base64
from dataclasses import replace
import hashlib,json,os,struct
from pathlib import Path
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
ROI=tuple((y*64+x)*3 for y in range(20,44) for x in range(16,36))

@unittest.skipUnless(os.environ.get('PHYMEO_NATIVE_INDIRECT_DLL'),'explicit diffuse device slot')
class IndirectDevice(unittest.TestCase):
    def backend(self,mode='diffuse_bounce',seed=0,**extra):
        path=Path(os.environ['PHYMEO_NATIVE_INDIRECT_DLL'])
        options=dict(library_path=path,shader_directory=path.parent,resident=True,shadow_mode='ray_query',
            indirect_mode=mode,indirect_seed=seed,device_index=int(os.environ.get('PHYMEO_NATIVE_DEVICE_INDEX','1')),validation=True)
        options.update(extra);return NativeBackend(**options)
    def close(self,owner):
        owner.close();info=owner.diagnostics()['device'];telemetry=info['resident_telemetry']
        for key in ('validation_errors','validation_warnings','live_contexts','live_resources','live_frames'):self.assertEqual(info[key],0,key)
        self.assertEqual(telemetry['device_local_bytes_current'],0);self.assertEqual(telemetry['host_visible_bytes_current'],0)
        self.assertEqual(telemetry['allocation_count'],telemetry['free_count'])
    def fixture(self,root):
        for name,positions,normal in (
            ('floor',((-1.,1.,0.),(1.,1.,0.),(1.,-1.,0.),(-1.,-1.,0.)),(0.,0.,1.)),
            ('wall',((0.,-1.,0.),(0.,1.,0.),(0.,1.,1.3),(0.,-1.,1.3)),(-1.,0.,0.))):
            path=root/(name+'.gltf');quad(path);data=json.loads(path.read_text())
            raw=bytearray(base64.b64decode(data['buffers'][0]['uri'].split(',')[1]))
            raw[:48]=struct.pack('<12f',*(v for p in positions for v in p));raw[48:96]=struct.pack('<12f',*(normal*4))
            data['buffers'][0]['uri']='data:application/octet-stream;base64,'+base64.b64encode(raw).decode()
            data['accessors'][0].update(min=[min(p[i] for p in positions) for i in range(3)],max=[max(p[i] for p in positions) for i in range(3)])
            path.write_text(json.dumps(data))
        return VisualScene('indirect',(Mesh('floor',str(root/'floor.gltf')),Mesh('wall',str(root/'wall.gltf'))),
            (Material('white',(.7,)*3,0.,1.),Material('red',(.8,.002,.002),0.,1.),Material('black',(0.,)*3,1.,1.)),
            (Instance(1,'floor','white','floor'),Instance(2,'wall','red','wall'),Instance(3,'wall','black','block')),
            (DirectionalLight((1.,0.,0.),(3.,)*3),))
    def frame(self,index=0,wall=.6,block=10.,epoch=0):
        key=FrameKey('indirect','world',epoch,index,index,'authored')
        frame=SceneFrame(key,float(index),(NodePose('floor',Pose()),NodePose('wall',Pose((wall,0.,0.))),NodePose('block',Pose((block,0.,0.)))))
        camera=Camera('top',key,Pose((0.,0.,2.)),64,64,60.,60.,32.,32.,.1,10.)
        return frame,camera
    def shot(self,owner,frame,camera,channels=tuple(Channel)):
        return {p.channel:p.data for p in owner.capture(frame,CaptureRequest(camera,RenderPath.RASTER,channels)).planes}
    def roi_mean(self,image,channel=0):return sum(image[Channel.RGB][i+channel] for i in ROI)/len(ROI)

    def test_red_transport_seed_replay_change_occlusion_and_no_history(self):
        with TemporaryDirectory() as directory:
            scene=self.fixture(Path(directory));images=[]
            for mode,seed in (('none',0),('diffuse_bounce',0),('diffuse_bounce',1),('diffuse_bounce',2**32-1)):
                owner=self.backend(mode,seed).open_scene(scene)
                try:
                    first=self.shot(owner,*self.frame());images.append(first)
                    self.assertEqual(self.shot(owner,*self.frame()),first)
                    if mode!='none':
                        far=self.shot(owner,*self.frame(1,wall=3.))
                        self.assertLess(self.roi_mean(far),self.roi_mean(first))
                        self.assertEqual(self.shot(owner,*self.frame(2)),first)
                        self.assertEqual(self.shot(owner,*self.frame(0,epoch=1)),first)
                        info=owner.diagnostics()['device']['resident_indirect_info']
                        self.assertEqual((info['enabled'],info['seed'],info['samples_per_pixel'],info['captures']),(1,seed,16,5))
                finally:self.close(owner)
            self.assertEqual(self.roi_mean(images[0]),0.)
            self.assertGreater(self.roi_mean(images[1]),25.)
            self.assertGreater(self.roi_mean(images[1]),4*self.roi_mean(images[1],1))
            self.assertNotEqual(images[1][Channel.RGB],images[2][Channel.RGB])
            for image in images[1:]:
                for channel in SENSORS:self.assertEqual(images[0][channel],image[channel])
            owner=self.backend().open_scene(scene)
            try:
                self.assertEqual(self.shot(owner,*self.frame()),images[1])
                blocked=self.shot(owner,*self.frame(1,block=.3))
                self.assertLess(self.roi_mean(blocked),self.roi_mean(images[1])*.1)
            finally:self.close(owner)

    def test_zero_light_black_receiver_and_pure_metal_do_not_emit_diffuse(self):
        with TemporaryDirectory() as directory:
            scene=self.fixture(Path(directory))
            for actual in (replace(scene,lights=()),replace(scene,materials=(replace(scene.materials[0],base_color=(0.,)*3),*scene.materials[1:])),
                replace(scene,materials=(replace(scene.materials[0],metallic=1.),*scene.materials[1:]))):
                outputs=[]
                for mode in ('none','diffuse_bounce'):
                    owner=self.backend(mode).open_scene(actual)
                    try:outputs.append(self.shot(owner,*self.frame()))
                    finally:self.close(owner)
                for p in ROI:self.assertEqual(outputs[0][Channel.RGB][p:p+3],outputs[1][Channel.RGB][p:p+3])
                if not actual.lights:self.assertFalse(any(outputs[1][Channel.RGB]))

    def test_uniform_environment_is_replaced_once_and_old_abi_rejects(self):
        from phymeo_rendering.studio import StudioLighting
        with TemporaryDirectory() as directory:
            scene=self.fixture(Path(directory));scene=replace(scene,instances=(scene.instances[0],),lights=())
            outputs=[]
            for mode in ('none','diffuse_bounce'):
                owner=self.backend(mode,studio=StudioLighting(environment_radiance=.2,panel_strength=0.)).open_scene(scene)
                try:outputs.append(self.shot(owner,*self.frame()))
                finally:self.close(owner)
            for p in ROI:
                self.assertLessEqual(max(abs(a-b) for a,b in zip(outputs[0][Channel.RGB][p:p+3],outputs[1][Channel.RGB][p:p+3])),1)
            for channel in SENSORS:self.assertEqual(outputs[0][channel],outputs[1][channel])
            old=Path(os.environ['PHYMEO_NATIVE_INDIRECT_OLD_DLL'])
            with self.assertRaisesRegex(BackendUnavailableError,'indirect'):
                self.backend(library_path=old).open_scene(scene)

    def test_combined_lobes_and_normal_grid_msaa_variants(self):
        from phymeo_rendering.studio import StudioLighting
        from PIL import Image
        with TemporaryDirectory() as directory:
            root=Path(directory);scene=self.fixture(root)
            scene=replace(scene,materials=(*scene.materials[:2],Material('black',(.9,)*3,1.,.04)))
            path=root/'flat.png';Image.frombytes('RGB',(1,1),bytes((128,128,255))).save(path)
            normal=TextureMap(str(path),hashlib.sha256(path.read_bytes()).hexdigest())
            for mapped in (False,True):
                actual=scene if not mapped else replace(scene,materials=(TexturedMaterial('white',(.7,)*3,0.,1.,normal_map=normal),*scene.materials[1:]))
                for grid in (2,4,8):
                    images=[]
                    for reflection in ('environment','single_bounce'):
                        owner=self.backend(reflection_mode=reflection,area_light_grid=grid,msaa_samples=4,
                            studio=StudioLighting(environment_radiance=.1,panel_strength=0.)).open_scene(actual)
                        try:
                            image=self.shot(owner,*self.frame(block=-.7));images.append(image)
                            if reflection=='single_bounce':self.assertEqual(owner.diagnostics()['device']['resident_reflection_info']['receiver_count'],1)
                        finally:self.close(owner)
                    for channel in SENSORS:self.assertEqual(images[0][channel],images[1][channel])
                    self.assertNotEqual(images[0][Channel.RGB],images[1][Channel.RGB])
                    # Primary matte lobe remains the same under independent specular switch.
                    for p in ROI:self.assertEqual(images[0][Channel.RGB][p:p+3],images[1][Channel.RGB][p:p+3])

    def test_recompiled_reflection_variants_match_previous_dll(self):
        from PIL import Image
        from test_native_reflection_device import ReflectionDevice
        helper=ReflectionDevice()
        old=Path(os.environ['PHYMEO_NATIVE_INDIRECT_OLD_DLL'])
        with TemporaryDirectory() as directory:
            root=Path(directory);scene=helper.fixture(root)
            path=root/'flat.png';Image.frombytes('RGB',(1,1),bytes((128,128,255))).save(path)
            normal=TextureMap(str(path),hashlib.sha256(path.read_bytes()).hexdigest())
            for mapped in (False,True):
                actual=scene if not mapped else replace(scene,materials=(
                    TexturedMaterial('mirror',(.9,)*3,1.,.04,normal_map=normal),*scene.materials[1:]))
                for grid in (2,4,8):
                    images=[]
                    for library in (old,Path(os.environ['PHYMEO_NATIVE_INDIRECT_DLL'])):
                        owner=self.backend('none',library_path=library,shader_directory=library.parent,
                            reflection_mode='single_bounce',area_light_grid=grid).open_scene(actual)
                        try:
                            images.append([helper.shot(owner,*helper.frame(i,1.,blocked))
                                for i,blocked in enumerate((False,True))])
                        finally:self.close(owner)
                    self.assertEqual(images[0],images[1],(mapped,grid))

    def test_explicit_setup_is_atomic_and_zero_receivers_are_legal(self):
        import ctypes as ct
        from phymeo_rendering.backends._native_indirect import configure_indirect,indirect_declaration
        with TemporaryDirectory() as directory:
            scene=self.fixture(Path(directory))
            for mode in ('none','diffuse_bounce'):
                owner=self.backend(mode).open_scene(scene)
                try:
                    device=owner._delegate.device
                    # Old fragment with new DLL is rejected; already configured
                    # indirect is immutable. Neither error poisons valid capture.
                    with self.assertRaises(BackendUnavailableError):
                        configure_indirect(device,indirect_declaration('diffuse_bounce',3))
                    first=self.shot(owner,*self.frame())
                    setter=device._lib.pr_resident_set_indirect
                    setter.argtypes=[ct.c_void_p,ct.c_uint32,ct.c_uint32];setter.restype=ct.c_int
                    self.assertNotEqual(setter(device._handle,2,0),0)
                    self.assertNotEqual(setter(device._handle,1,99),0)
                    self.assertEqual(self.shot(owner,*self.frame()),first)
                    sensor=self.shot(owner,*self.frame(),channels=SENSORS)
                    self.assertEqual(sensor,{c:first[c] for c in SENSORS})
                    if mode!='none':
                        info=owner.diagnostics()['device']['resident_indirect_info']
                        self.assertEqual((info['captures'],info['rgb_captures'],info['seed']),(3,2,0))
                finally:self.close(owner)
            scene=replace(scene,materials=tuple(replace(m,metallic=1.) for m in scene.materials))
            images=[]
            for mode in ('none','diffuse_bounce'):
                owner=self.backend(mode).open_scene(scene)
                try:
                    images.append(self.shot(owner,*self.frame()))
                    if mode!='none':
                        info=owner.diagnostics()['device']['resident_indirect_info']
                        self.assertEqual((info['enabled'],info['receiver_count']),(1,0))
                finally:self.close(owner)
            self.assertEqual(images[0],images[1])
if __name__=='__main__':unittest.main()
