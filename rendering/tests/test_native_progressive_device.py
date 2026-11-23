"""Opt-in real GPU accumulation tests; raw linear bytes are the numeric evidence."""
from dataclasses import replace
import hashlib
import math
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

from phymeo_rendering.assets import TextureMap,TexturedMaterial
from phymeo_rendering.backends import BackendUnavailableError,UnsupportedRequestError
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.outputs import Channel
from phymeo_rendering.scene import DirectionalLight,NodePose,Pose
from phymeo_rendering.studio import StudioLighting
import test_native_indirect_device as indirect_fixture
SENSORS=indirect_fixture.SENSORS
ROI=indirect_fixture.ROI


@unittest.skipUnless(os.environ.get('PHYMEO_NATIVE_PROGRESSIVE_DLL'),'explicit progressive GPU slot')
class ProgressiveDevice(unittest.TestCase):
    def backend(self, **extra):
        path=Path(os.environ['PHYMEO_NATIVE_PROGRESSIVE_DLL'])
        values=dict(library_path=path,shader_directory=path.parent,resident=True,shadow_mode='ray_query',
                    indirect_mode='diffuse_bounce',progressive=True,validation=True,device_index=1)
        values.update(extra)
        return NativeBackend(**values)

    def setUp(self):
        self.helper=indirect_fixture.IndirectDevice()
        self.temp=TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.scene=self.helper.fixture(self.root)
        self.owners=[]
        self.addCleanup(self.close_all)

    def open(self,scene=None,**extra):
        owner=self.backend(**extra).open_scene(self.scene if scene is None else scene)
        self.owners.append(owner)
        return owner

    def close_all(self):
        for owner in reversed(self.owners):
            owner.close()
            info=owner.diagnostics()['device']
            self.assertEqual(info['validation_errors'],0)
            self.assertEqual(info['validation_warnings'],0)
            telemetry=info['resident_telemetry']
            self.assertEqual(telemetry['device_local_bytes_current'],0)
            self.assertEqual(telemetry['host_visible_bytes_current'],0)
            self.assertEqual(telemetry['allocation_count'],telemetry['free_count'])
        if self.owners:
            info=self.owners[0].diagnostics()['device']
            for name in ('live_contexts','live_frames','live_resources'):
                self.assertEqual(info[name],0,name)

    def shot(self,owner,frame_camera=None,channels=tuple(Channel)):
        return self.helper.shot(owner,*(frame_camera or self.helper.frame()),channels=channels)

    def mean(self,owner):
        before=owner.diagnostics()['progressive']['batches']
        result=owner.read_linear_mean()
        self.assertEqual(result['format'],'RGBA32F_LE')
        self.assertEqual(result['space'],'scene-linear-before-exposure')
        self.assertEqual(result['batches'],before)
        self.assertEqual(owner.diagnostics()['progressive']['batches'],before)
        self.assertIsInstance(result['data'],bytes)
        pixels=tuple(struct.iter_unpack('<4f',result['data']))
        self.assertTrue(all(math.isfinite(x) and x>=0 for pixel in pixels for x in pixel))
        self.assertTrue(all(pixel[3]==1. for pixel in pixels))
        return result,pixels

    def test_seed_batches_reset_nonrgb_failure_and_raw_display(self):
        owner=self.open()
        first=self.shot(owner)
        raw_first,_=self.mean(owner)
        self.assertEqual(owner.diagnostics()['progressive']['last_reset_reasons'],['new_session'])
        # Different sensor-only state/extent must not replace a previous RGB mean.
        frame,camera=self.helper.frame(1,wall=3.)
        self.shot(owner,(frame,replace(camera,width=63)),channels=SENSORS)
        self.assertEqual(self.mean(owner)[0]['data'],raw_first['data'])
        second=self.shot(owner,self.helper.frame(2))
        info=owner.diagnostics()['progressive']
        self.assertEqual((info['batches'],info['batch_index'],info['primary_samples_per_pixel']),(2,1,32))
        raw_second,pixels=self.mean(owner)
        self.assertNotEqual(raw_first['data'],raw_second['data'])
        for pixel in range(64*64):
            for channel in range(3):
                x=min(1.,max(0.,pixels[pixel][channel]))
                encoded=12.92*x if x<=.0031308 else 1.055*x**(1/2.4)-.055
                self.assertLessEqual(abs(round(encoded*255)-second[Channel.RGB][3*pixel+channel]),1)
        for channel in SENSORS:self.assertEqual(first[channel],second[channel])
        frame,camera=self.helper.frame(3)
        with self.assertRaises(UnsupportedRequestError):self.shot(owner,(frame,replace(camera,width=100000)))
        self.assertEqual(self.mean(owner)[0]['data'],raw_second['data'])
        owner.reset_accumulation()
        self.assertEqual(owner.diagnostics()['progressive']['batches'],0)
        self.assertEqual(self.shot(owner,self.helper.frame(4)),first)
        self.assertEqual(self.mean(owner)[0]['data'],raw_first['data'])
        self.assertEqual(owner.diagnostics()['progressive']['last_reset_reasons'],['explicit_reset'])
        reopened=self.open()
        self.assertEqual(self.shot(reopened),first)
        different=self.open(indirect_seed=1)
        self.assertNotEqual(self.shot(different)[Channel.RGB],first[Channel.RGB])

    def test_same_owner_state_and_light_changes_match_fresh(self):
        owner=self.open()
        self.shot(owner);self.shot(owner)
        frame,camera=self.helper.frame(2,wall=.9)
        cases=[(frame,camera,'transforms'),(frame,replace(camera,fx_px=59.),'camera'),
               (frame,replace(camera,width=63),'extent')]
        for frame,camera,reason in cases:
            actual=self.shot(owner,(frame,camera))
            self.assertIn(reason,owner.diagnostics()['progressive']['last_reset_reasons'])
            self.assertEqual(owner.diagnostics()['progressive']['batches'],1)
            fresh=self.open()
            self.assertEqual(actual,self.shot(fresh,(frame,camera)))
        lights=(DirectionalLight((1.,0.,0.),(1.,2.,3.)),)
        owner.set_directional_lights(lights)
        actual=self.shot(owner,(frame,camera))
        self.assertIn('lights',owner.diagnostics()['progressive']['last_reset_reasons'])
        fresh=self.open(replace(self.scene,lights=lights))
        self.assertEqual(actual,self.shot(fresh,(frame,camera)))
        count=owner.diagnostics()['progressive']['batches']
        owner.set_directional_lights((DirectionalLight((1.0000005,0.,0.),(1.,2.,3.)),))
        self.shot(owner,(frame,camera))
        self.assertEqual(owner.diagnostics()['progressive']['batches'],count+1)
        with self.assertRaises(TypeError):owner.set_directional_lights(list(lights))
        with self.assertRaises(ValueError):owner.set_directional_lights(lights*5)
        self.assertEqual(owner.diagnostics()['effective_directional_lights'][0]['illuminance'],(1.,2.,3.))

    def test_uniform_raw_hdr_is_before_exposure_and_bounces_do_not_double_environment(self):
        scene=replace(self.scene,instances=(self.scene.instances[0],),lights=())
        outputs=[]
        for bounces in (1,2):
            owner=self.open(scene,indirect_bounces=bounces,studio=StudioLighting(environment_radiance=4.,panel_strength=0.))
            first=self.shot(owner)
            raw,pixels=self.mean(owner)
            self.assertGreater(max(p[0] for p in pixels),1.)
            for _ in range(3):self.shot(owner)
            self.assertEqual(self.mean(owner)[0]['data'],raw['data'])
            outputs.append(first)
        self.assertEqual(outputs[0],outputs[1])
        dark=self.open(replace(self.scene,lights=()),indirect_bounces=2)
        self.assertFalse(any(self.shot(dark)[Channel.RGB]))

    def test_two_bounces_normal_grid_combination_and_old_rejection(self):
        from PIL import Image
        path=self.root/'normal.png';Image.frombytes('RGB',(1,1),bytes((128,128,255))).save(path)
        normal=TextureMap(str(path),hashlib.sha256(path.read_bytes()).hexdigest())
        for mapped in (False,True):
            scene=self.scene if not mapped else replace(self.scene,materials=(
                TexturedMaterial('white',(.7,)*3,0.,1.,normal_map=normal),*self.scene.materials[1:]))
            for grid in (2,4,8):
                for reflection in ('environment','single_bounce'):
                    owner=self.open(scene,indirect_bounces=2,area_light_grid=grid,reflection_mode=reflection)
                    self.shot(owner);self.mean(owner)
                    info=owner.diagnostics()['device']['resident_progressive_info']
                    self.assertEqual((info['enabled'],info['bounces'],info['batches'],info['rgb_captures']),(1,2,1,1))
                    owner.close()
        old=Path(os.environ['PHYMEO_NATIVE_PROGRESSIVE_OLD_DLL'])
        with self.assertRaisesRegex(BackendUnavailableError,'progressive'):
            self.backend(library_path=old).open_scene(self.scene)


    def test_failed_publication_discards_unpublished_gpu_mean(self):
        from unittest.mock import patch
        owner=self.open()
        first=self.shot(owner)
        with patch('phymeo_rendering.backends.native.Capture',side_effect=ValueError('injected publication rejection')):
            with self.assertRaisesRegex(ValueError,'injected publication'):
                self.shot(owner)
        self.assertEqual(owner.diagnostics()['progressive']['batches'],0)
        self.assertEqual(owner.diagnostics()['progressive']['last_reset_reasons'],['failed_capture'])
        with self.assertRaises(UnsupportedRequestError):owner.read_linear_mean()
        self.assertEqual(self.shot(owner),first)

    def test_hdr_sky_raw_linear_and_previous_nonprogressive_bytes(self):
        from phymeo_rendering.environment import RadianceEnvironment
        import shutil
        hdr=self.root/'constant.hdr'
        raw=b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n'+bytes((128,80,40,130))*32
        hdr.write_bytes(raw)
        environment=RadianceEnvironment(str(hdr),hashlib.sha256(raw).hexdigest(),face_size=16)
        studio=StudioLighting(environment_map=environment,panel_strength=0.)
        empty=replace(self.scene,instances=(),lights=())
        means=[]
        for transform,exposure in (('clip',.25),('pbr-neutral',2.)):
            owner=self.open(empty,studio=studio,display_transform=transform,exposure=exposure)
            image=self.shot(owner)
            mean,pixels=self.mean(owner);means.append(mean['data'])
            self.assertEqual(pixels[0],(2.,1.25,.625,1.))
            self.assertEqual(owner.diagnostics()['device']['resident_environment_info']['sky_draws'],1)
            self.assertFalse(any(image[Channel.VALID]))
        self.assertEqual(means[0],means[1])
        old=Path(os.environ['PHYMEO_NATIVE_PROGRESSIVE_OLD_DLL'])
        for mode in ('none','diffuse_bounce'):
            values=[]
            for path in (old,Path(os.environ['PHYMEO_NATIVE_PROGRESSIVE_DLL'])):
                owner=self.open(progressive=False,indirect_mode=mode,library_path=path,shader_directory=path.parent)
                values.append(self.shot(owner))
            self.assertEqual(values[0],values[1])
        mismatched=self.root/'mixed';mismatched.mkdir()
        source=Path(os.environ['PHYMEO_NATIVE_PROGRESSIVE_DLL']).parent
        for name in ('resident_rayquery.vert.spv','resident_rayquery_indirect_progressive.frag.spv','resident_sky.vert.spv','resident_progressive.comp.spv'):
            shutil.copyfile(source/name,mismatched/name)
        shutil.copyfile(source/'resident_sky.frag.spv',mismatched/'resident_sky_progressive.frag.spv')
        with self.assertRaisesRegex(RuntimeError,'resident sky progressive radiance ABI mismatch'):
            self.backend(shader_directory=mismatched,studio=studio).open_scene(empty)
if __name__=='__main__':unittest.main()
