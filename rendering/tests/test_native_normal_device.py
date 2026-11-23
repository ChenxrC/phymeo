"""Opt-in normal-map native GPU checks, including real double-sided lighting.

PHYMEO_NATIVE_NORMAL_DLL selects the candidate; PHYMEO_NATIVE_NORMAL_OLD_DLL
selects the previously verified MSAA build for strict default-path comparison.
"""
from dataclasses import replace
import ctypes as ct
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from phymeo_rendering.assets import Material,Mesh,TextureMap,TexturedMaterial,TextureChannel
from phymeo_rendering.backends import BackendUnavailableError
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest,RenderPath
from phymeo_rendering.scene import DirectionalLight,FrameKey,Instance,NodePose,Pose,SceneFrame,VisualScene
from test_native_texture_device import quad

SENSORS=(Channel.FORWARD_DEPTH,Channel.INSTANCE_ID,Channel.VALID)


@unittest.skipUnless(os.environ.get('PHYMEO_NATIVE_NORMAL_DLL'),'explicit normal-map device execution slot')
class NativeNormalDevice(unittest.TestCase):
    def backend(self,**extra):
        options=dict(library_path=Path(os.environ['PHYMEO_NATIVE_NORMAL_DLL']),
            device_index=int(os.environ.get('PHYMEO_NATIVE_DEVICE_INDEX','1')),resident=True,validation=True)
        options.update(extra); options.setdefault('shader_directory',Path(options['library_path']).parent)
        return NativeBackend(**options)

    def close(self,owner):
        owner.close(); info=owner.diagnostics()['device']; telemetry=info['resident_telemetry']
        for key in ('validation_errors','validation_warnings','live_resources','live_contexts','live_frames'):
            self.assertEqual(info[key],0,key)
        self.assertEqual(telemetry['device_local_bytes_current'],0)
        self.assertEqual(telemetry['host_visible_bytes_current'],0)
        self.assertEqual(telemetry['allocation_count'],telemetry['free_count'])

    @staticmethod
    def image(root,name,rgb):
        from PIL import Image
        path=root/(name+'.png'); Image.frombytes('RGB',(1,1),bytes(rgb)).save(path)
        return TextureMap(str(path),hashlib.sha256(path.read_bytes()).hexdigest())

    @staticmethod
    def inputs(root):
        mesh=root/'quad.gltf'; quad(mesh)
        scene=VisualScene('normal-device',(Mesh('quad',str(mesh)),),
            (Material('surface',(.4,.4,.4),0.,1.),),(Instance(7,'quad','surface','quad'),),
            (DirectionalLight((-.6,0.,-.8),(2.,)*3),))
        key=FrameKey(scene.key,'world',0,0,0,'authored')
        frame=SceneFrame(key,0.,(NodePose('quad',Pose()),))
        camera=Camera('fixed',key,Pose(),33,25,24.,24.,16.5,12.5,.1,10.)
        return scene,frame,camera

    def shot(self,owner,frame,camera,channels=tuple(Channel)):
        return {p.channel:p.data for p in owner.capture(frame,CaptureRequest(camera,RenderPath.RASTER,channels)).planes}

    def test_flat_tilted_soft_hardware_msaa_and_sensor_invariance(self):
        with TemporaryDirectory() as root:
            root=Path(root); scene,frame,camera=self.inputs(root)
            flat=self.image(root,'flat',(128,128,255)); tilted=self.image(root,'tilted',(204,128,230))
            for mode in ('software','ray_query'):
                for samples in (1,4):
                    shots=[]
                    for source in (None,flat,tilted):
                        material=scene.materials[0] if source is None else TexturedMaterial('surface',(.4,)*3,0.,1.,normal_map=source)
                        owner=self.backend(shadow_mode=mode,msaa_samples=samples).open_scene(replace(scene,materials=(material,)))
                        try:
                            image=self.shot(owner,frame,camera); shots.append(image)
                            if source is not None:
                                rgb=self.shot(owner,frame,camera,(Channel.RGB,)); sensor=self.shot(owner,frame,camera,SENSORS)
                                self.assertEqual(rgb[Channel.RGB],image[Channel.RGB])
                                for channel in SENSORS: self.assertEqual(sensor[channel],image[channel])
                                info=owner.diagnostics()['device']['resident_normal_info']
                                self.assertEqual((info['normal_abi_version'],info['format'],info['upload_calls'],info['bound_instances']),(1,37,1,1))
                                self.assertEqual(info['rgb_captures_with_normals'],2)
                                self.assertEqual(owner.diagnostics()['normal_maps']['textures'][0]['source_sha256'],source.sha256)
                                raw=owner._delegate.device
                                self.assertNotEqual(raw._lib.pr_resident_set_normal_maps(raw._handle,None,0,None,0,None,0,None,0),0)
                        finally: self.close(owner)
                    for mapped in shots[1:]:
                        for channel in SENSORS: self.assertEqual(mapped[channel],shots[0][channel])
                    pixels=[(y*33+x)*3+c for y in range(6,19) for x in range(9,24) for c in range(3)]
                    self.assertLessEqual(max(abs(shots[1][Channel.RGB][p]-shots[0][Channel.RGB][p]) for p in pixels),1)
                    center=(12*33+16)*3
                    self.assertGreater(shots[2][Channel.RGB][center],shots[0][Channel.RGB][center]+10)

    def test_two_sided_tilted_normal_flips_complete_vector_and_bias_normal(self):
        with TemporaryDirectory() as root:
            root=Path(root); scene,frame,camera=self.inputs(root)
            tilted=self.image(root,'tilted',(204,128,230))
            scene=replace(scene,materials=(TexturedMaterial('surface',(.4,)*3,0.,1.,normal_map=tilted),))
            for mode in ('software','ray_query'):
                values=[]
                for back in (False,True):
                    actual_scene=replace(scene,lights=(DirectionalLight((.6,0.,.8) if back else (-.6,0.,-.8),(2.,)*3),))
                    actual_camera=replace(camera,world_pose=Pose((0.,0.,-4.),(0.,1.,0.,0.))) if back else camera
                    owner=self.backend(shadow_mode=mode).open_scene(actual_scene)
                    try: values.append(self.shot(owner,frame,actual_camera)[Channel.RGB][(12*33+16)*3])
                    finally: self.close(owner)
                self.assertLessEqual(abs(values[0]-values[1]),1,values)
                self.assertGreater(values[0],100)

    def test_plain_old_bytes_and_requested_normal_explicit_old_rejection(self):
        old=Path(os.environ['PHYMEO_NATIVE_NORMAL_OLD_DLL'])
        with TemporaryDirectory() as root:
            root=Path(root); scene,frame,camera=self.inputs(root); normal=self.image(root,'normal',(128,128,255))
            for samples in (1,4):
                values=[]
                for library in (old,Path(os.environ['PHYMEO_NATIVE_NORMAL_DLL'])):
                    owner=self.backend(library_path=library,msaa_samples=samples).open_scene(scene)
                    try: values.append(self.shot(owner,frame,camera))
                    finally: self.close(owner)
                self.assertEqual(values[0],values[1])
            mapped=replace(scene,materials=(TexturedMaterial('surface',(.4,)*3,0.,1.,normal_map=normal),))
            for options in ({'library_path':old},{'shader_directory':old.parent}):
                with self.assertRaisesRegex(BackendUnavailableError,'normal'):
                    self.backend(**options).open_scene(mapped)
            # Actual C rejection of a corrupted UV packet before GPU allocation.
            from phymeo_rendering.mesh_data import load_mesh
            from phymeo_rendering.backends._native_normal import pack_normal_maps,configure_normal_maps
            upload=pack_normal_maps(mapped,{'quad':load_mesh(scene.meshes[0])})
            for uv in upload.uv: uv.u=uv.v=0.
            from phymeo_rendering.backends._native_resident import ResidentDevice,pack_scene
            library=Path(os.environ['PHYMEO_NATIVE_NORMAL_DLL'])
            raw=ResidentDevice(library,library.parent,int(os.environ.get('PHYMEO_NATIVE_DEVICE_INDEX','1')),True,normal_maps=True)
            try:
                raw.upload(*pack_scene(scene,{'quad':load_mesh(scene.meshes[0])},raw.info()))
                with self.assertRaisesRegex(BackendUnavailableError,'UV'):
                    configure_normal_maps(raw,upload)
                self.assertEqual(raw.info()['resident_poisoned'],0)
            finally:
                raw.close()
                self.assertEqual(raw.info()['live_resources'],0)

    def test_color_roughness_and_normal_resources_remain_independent(self):
        with TemporaryDirectory() as root:
            root=Path(root); scene,frame,camera=self.inputs(root)
            color=self.image(root,'color',(230,100,45)); rough=self.image(root,'rough',(180,180,180))
            normal=self.image(root,'normal',(204,128,230))
            material=TexturedMaterial('surface',(.4,)*3,.2,.8,color,TextureChannel(rough),normal)
            owner=self.backend().open_scene(replace(scene,materials=(material,)))
            try:
                image=self.shot(owner,frame,camera)
                self.assertTrue(any(image[Channel.RGB]))
                info=owner.diagnostics()['device']
                self.assertEqual(info['resident_texture_info']['upload_calls'],1)
                self.assertEqual(info['resident_normal_info']['upload_calls'],1)
                self.assertEqual(info['resident_normal_info']['format'],37)
                self.assertEqual(info['resident_texture_info']['format'],43)
            finally: self.close(owner)


    def test_studio_environment_all_grid_variants(self):
        from PIL import Image
        from phymeo_rendering.studio import StudioLighting
        from phymeo_rendering.environment import RadianceEnvironment
        with TemporaryDirectory() as root:
            root=Path(root); scene,frame,camera=self.inputs(root)
            scene=replace(scene,lights=())
            tilted=self.image(root,'tilted',(204,128,230))
            raw=b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n'+(bytes((128,16,0,129))*4+bytes((0,16,128,129))*4)*4
            hdr=root/'direction.hdr'; hdr.write_bytes(raw)
            environment=RadianceEnvironment(str(hdr),hashlib.sha256(raw).hexdigest(),face_size=16)
            studio=StudioLighting(center_m=(0.,0.,-2.),environment_map=environment)
            for mode in ('software','ray_query'):
                for grid in (2,4,8):
                    shots=[]
                    for source in (None,tilted):
                        material=scene.materials[0] if source is None else TexturedMaterial('surface',(.4,)*3,0.,1.,normal_map=source)
                        owner=self.backend(shadow_mode=mode,area_light_grid=grid,msaa_samples=4,studio=studio).open_scene(replace(scene,materials=(material,)))
                        try:
                            shots.append(self.shot(owner,frame,camera))
                            self.assertEqual(owner.diagnostics()['device']['resident_environment_info']['sky_draws'],1)
                        finally: self.close(owner)
                    for channel in SENSORS: self.assertEqual(shots[0][channel],shots[1][channel])
                    self.assertNotEqual(shots[0][Channel.RGB],shots[1][Channel.RGB])
    def test_near_zero_filtered_sample_remains_visible_and_preserves_sensors(self):
        from PIL import Image
        with TemporaryDirectory() as root:
            root=Path(root); scene,frame,camera=self.inputs(root)
            # The ideal bilinear value at UV=(.5,.5) is (.5,.5,.5). Real
            # interpolation/filtering is not required to yield exactly that
            # floating-point value. Do not claim this exercises the exact-zero
            # source guard or assert equality with the geometric normal.
            path=root/'zero-center.png'; Image.frombytes('RGB',(2,1),bytes((0,0,0,255,255,255))).save(path)
            zero=TextureMap(str(path),hashlib.sha256(path.read_bytes()).hexdigest())
            scene=replace(scene,lights=(DirectionalLight((-.6,0.,-.8),(2.,)*3),))
            values=[]
            for source in (None,zero):
                material=scene.materials[0] if source is None else TexturedMaterial('surface',(.4,)*3,0.,1.,normal_map=source)
                owner=self.backend().open_scene(replace(scene,materials=(material,)))
                try: values.append(self.shot(owner,frame,camera))
                finally: self.close(owner)
            center=(12*33+16)*3
            self.assertTrue(all(0<value<255 for value in values[1][Channel.RGB][center:center+3]))
            for channel in SENSORS: self.assertEqual(values[0][channel],values[1][channel])
if __name__=='__main__': unittest.main()
