"""Actual GPU controls for the retained-guide settling schedule and raw anchors."""
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from phymeo_rendering.backends import BackendUnavailableError, UnsupportedRequestError
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.outputs import Channel
from phymeo_rendering.scene import DirectionalLight
from phymeo_rendering.studio import StudioLighting
import test_native_indirect_device as fixture


@unittest.skipUnless(os.environ.get('PHYMEO_NATIVE_SETTLING_DLL'), 'explicit settling GPU slot')
class SettlingDevice(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.helper = fixture.IndirectDevice()
        self.scene = self.helper.fixture(self.root)
        self.owners = []
        self.addCleanup(self.close_all)

    def open(self, scene=None, **extra):
        path = Path(os.environ['PHYMEO_NATIVE_SETTLING_DLL'])
        options = dict(library_path=path, shader_directory=path.parent, device_index=1,
                       validation=True, resident=True, shadow_mode='ray_query',
                       indirect_mode='diffuse_bounce', progressive=True, spatial_filter='guided_settling')
        options.update(extra)
        owner = NativeBackend(**options).open_scene(self.scene if scene is None else scene)
        self.owners.append(owner)
        return owner

    def shot(self, owner, state=None, channels=tuple(Channel)):
        return self.helper.shot(owner, *(state or self.helper.frame()), channels=channels)

    def close_all(self):
        for owner in reversed(self.owners):
            owner.close()
            info = owner.diagnostics()['device']
            self.assertEqual(info['validation_errors'], 0)
            self.assertEqual(info['validation_warnings'], 0)
            telemetry = info['resident_telemetry']
            self.assertEqual(telemetry['allocation_count'], telemetry['free_count'])
            self.assertEqual(telemetry['device_local_bytes_current'], 0)
            self.assertEqual(telemetry['host_visible_bytes_current'], 0)
        if self.owners:
            for key in ('live_contexts', 'live_frames', 'live_resources'):
                self.assertEqual(self.owners[0].diagnostics()['device'][key], 0)

    def test_all_seventeen_batches_frozen_kernel_and_byte_exact_anchors(self):
        path = Path(__file__).parents[1] / 'examples/progressive_first_frame_preview.py'
        spec = importlib.util.spec_from_file_location('frozen_preview', path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        for display in ('clip', 'pbr-neutral'):
            options = dict(display_transform=display, studio=StudioLighting(environment_radiance=.03, panel_strength=.3))
            settling = self.open(**options)
            guided = self.open(spatial_filter='guided', **options)
            plain = self.open(spatial_filter='none', **options)
            guide_bytes = None
            for n in range(1, 18):
                actual = self.shot(settling)
                old_guided = self.shot(guided)
                unfiltered = self.shot(plain)
                raw = settling.read_linear_mean()['data']
                self.assertEqual(raw, guided.read_linear_mean()['data'])
                self.assertEqual(raw, plain.read_linear_mean()['data'])
                for channel in fixture.SENSORS:
                    self.assertEqual(actual[channel], unfiltered[channel])
                if n == 1:
                    self.assertEqual(actual, old_guided)
                if n >= 16:
                    self.assertEqual(actual, unfiltered)
                    with self.assertRaises(UnsupportedRequestError):
                        settling.read_spatial_guide()
                else:
                    read = settling.read_spatial_guide()['data']
                    if guide_bytes is None:
                        guide_bytes = read
                    self.assertEqual(read, guide_bytes)
                    valid = np.frombuffer(actual[Channel.VALID], np.uint8).reshape(64, 64)
                    ids = np.frombuffer(actual[Channel.INSTANCE_ID], '<u4').reshape(64, 64)
                    depth = np.frombuffer(actual[Channel.FORWARD_DEPTH], '<f4').reshape(64, 64)
                    linear = np.frombuffer(raw, '<f4').reshape(64, 64, 4)[..., :3]
                    guide = np.frombuffer(read, np.uint8).reshape(64, 64, 4)[..., :3]
                    filtered = reference.reconstruct(linear, depth, ids, valid, guide)
                    info = settling.diagnostics()['device']['resident_settling_info']
                    w = info['last_filtered_fraction']
                    self.assertAlmostEqual(w, 1.-np.sqrt((n-1)/15.), delta=1e-7)
                    shown = filtered if n == 1 else linear.astype(np.float64)*(1.-w)+filtered.astype(np.float64)*w
                    if display == 'pbr-neutral':
                        encoded = np.frombuffer(reference.encode_neutral(shown, 1.), np.uint8).reshape(64, 64, 3)
                    else:
                        x = np.clip(shown.astype(np.float64), 0., 1.)
                        encoded = np.rint(np.where(x <= .0031308, 12.92*x, 1.055*x**(1/2.4)-.055)*255).astype(np.uint8)
                    pixels = np.frombuffer(actual[Channel.RGB], np.uint8).reshape(64, 64, 3)
                    self.assertLessEqual(int(np.max(np.abs(encoded.astype(int)-pixels))), 1)
                    self.assertTrue(np.array_equal(pixels[valid == 0], np.frombuffer(unfiltered[Channel.RGB], np.uint8).reshape(64,64,3)[valid == 0]))
                info = settling.diagnostics()['device']['resident_settling_info']
                self.assertEqual(info['guide_draws'], 1)
                self.assertEqual(info['last_batch'], n)
                self.assertEqual(info['filter_dispatches'], min(n, 15))
                self.assertEqual(info['filtered_primary_samples'], 16*sum(range(1, min(n, 15)+1)))
                self.assertEqual(settling.diagnostics()['spatial_filter']['last_capture_filtered'], n < 16)

    def test_retained_guide_survives_sensor_target_rebuild_and_all_resets(self):
        owner = self.open()
        first = self.shot(owner)
        guide = owner.read_spatial_guide()['data']
        raw = owner.read_linear_mean()['data']
        frame, camera = self.helper.frame(1)
        self.shot(owner, (frame, replace(camera, width=63, height=65)), channels=fixture.SENSORS)
        self.assertEqual(owner.read_linear_mean()['data'], raw)
        self.assertEqual(owner.diagnostics()['device']['resident_settling_info']['guide_draws'], 1)
        with self.assertRaises(UnsupportedRequestError):
            owner.read_spatial_guide()
        self.shot(owner)
        self.assertEqual(owner.read_spatial_guide()['data'], guide)
        self.assertEqual(owner.diagnostics()['device']['resident_settling_info']['guide_draws'], 1)
        for state in (self.helper.frame(2, wall=.8), self.helper.frame(3, epoch=1),
                      (frame, replace(camera, width=63)), (frame, replace(camera, fx_px=59.))):
            actual = self.shot(owner, state)
            fresh = self.open()
            self.assertEqual(actual, self.shot(fresh, state))
            self.assertEqual(owner.read_linear_mean()['data'], fresh.read_linear_mean()['data'])
            self.assertEqual(owner.read_spatial_guide()['data'], fresh.read_spatial_guide()['data'])
        lights = (DirectionalLight((1.,0.,0.), (1.,2.,3.)),)
        owner.set_directional_lights(lights)
        fresh = self.open(replace(self.scene, lights=lights))
        self.assertEqual(self.shot(owner), self.shot(fresh))
        owner.set_directional_lights(self.scene.lights)
        owner.reset_accumulation()
        self.assertEqual(self.shot(owner), first)
        with patch('phymeo_rendering.backends.native.Capture', side_effect=ValueError('publication control')):
            with self.assertRaisesRegex(ValueError, 'publication control'):
                self.shot(owner)
        self.assertEqual(owner.diagnostics()['device']['resident_settling_info']['guide_valid'], 0)
        with self.assertRaises(UnsupportedRequestError):
            owner.read_spatial_guide()
        self.assertEqual(self.shot(owner), first)
        reopened = self.open()
        self.assertEqual(self.shot(reopened), first)

    def test_two_bounces_normal_reflection_dense_lighting_and_hdr_background(self):
        import hashlib
        from PIL import Image
        from phymeo_rendering.assets import TextureMap, TexturedMaterial
        from phymeo_rendering.environment import RadianceEnvironment
        path = self.root / 'normal.png'
        Image.frombytes('RGB', (1,1), bytes((128,128,255))).save(path)
        normal = TextureMap(str(path), hashlib.sha256(path.read_bytes()).hexdigest())
        mapped = replace(self.scene, materials=(
            TexturedMaterial('white', (.7,)*3, 0., 1., normal_map=normal), *self.scene.materials[1:]))
        options = dict(indirect_bounces=2, reflection_mode='single_bounce', area_light_grid=8,
                       studio=StudioLighting(environment_radiance=.03, panel_strength=.3))
        settling = self.open(mapped, **options)
        guided = self.open(mapped, spatial_filter='guided', **options)
        plain = self.open(mapped, spatial_filter='none', **options)
        for n in range(1,17):
            actual, anchor, raw = self.shot(settling), self.shot(guided), self.shot(plain)
            self.assertEqual(settling.read_linear_mean()['data'], plain.read_linear_mean()['data'])
            for channel in fixture.SENSORS:
                self.assertEqual(actual[channel], raw[channel])
            if n == 1:
                self.assertEqual(actual, anchor)
            elif n == 16:
                self.assertEqual(actual, raw)
        hdr = self.root / 'constant.hdr'
        data = b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n' + bytes((128,80,40,130))*32
        hdr.write_bytes(data)
        environment = RadianceEnvironment(str(hdr), hashlib.sha256(data).hexdigest(), face_size=16)
        empty = replace(self.scene, instances=(), lights=())
        options = dict(studio=StudioLighting(environment_map=environment, panel_strength=0.), display_transform='pbr-neutral')
        a = self.open(empty, **options)
        b = self.open(empty, spatial_filter='none', **options)
        for _ in range(2):
            self.assertEqual(self.shot(a), self.shot(b))
            self.assertEqual(a.read_linear_mean()['data'], b.read_linear_mean()['data'])

    def test_failed_post_gpu_diagnostic_does_not_commit_unpublished_history(self):
        owner = self.open()
        first = self.shot(owner)
        with patch('phymeo_rendering.backends.native.query_settling', side_effect=RuntimeError('diagnostic control')):
            with self.assertRaisesRegex(RuntimeError, 'diagnostic control'):
                self.shot(owner)
        self.assertEqual(owner.diagnostics()['progressive']['batches'], 0)
        self.assertFalse(owner.diagnostics()['spatial_filter']['last_capture_filtered'])
        self.assertEqual(self.shot(owner), first)

    def test_old_library_and_shader_rejection_and_existing_modes(self):
        import shutil
        old = Path(os.environ['PHYMEO_NATIVE_SETTLING_OLD_DLL'])
        with self.assertRaisesRegex(BackendUnavailableError, 'settling'):
            self.open(library_path=old)
        for mode in ('none', 'guided'):
            previous = self.open(library_path=old, shader_directory=old.parent, spatial_filter=mode)
            current = self.open(spatial_filter=mode)
            for _ in range(2):
                self.assertEqual(self.shot(previous), self.shot(current))
                self.assertEqual(previous.read_linear_mean()['data'], current.read_linear_mean()['data'])
        source = Path(os.environ['PHYMEO_NATIVE_SETTLING_DLL']).parent
        mixed = self.root / 'mixed'
        mixed.mkdir()
        for name in ('resident_rayquery.vert.spv', 'resident_rayquery_indirect_progressive.frag.spv',
                     'resident_progressive.comp.spv', 'resident_rayquery.frag.spv'):
            shutil.copyfile(source/name, mixed/name)
        shutil.copyfile(source/'resident_spatial.comp.spv', mixed/'resident_settling.comp.spv')
        with self.assertRaisesRegex(BackendUnavailableError, 'settling display compute ABI'):
            self.open(shader_directory=mixed)


if __name__ == '__main__':
    unittest.main()
