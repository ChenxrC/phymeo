"""Focused actual-device tests for same-exposure first-batch guided display."""
from dataclasses import replace
import hashlib
import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from phymeo_rendering.assets import TextureMap, TexturedMaterial
from phymeo_rendering.backends import BackendUnavailableError, UnsupportedRequestError
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.outputs import Channel
from phymeo_rendering.scene import DirectionalLight
from phymeo_rendering.studio import StudioLighting
import test_native_indirect_device as fixture


@unittest.skipUnless(os.environ.get('PHYMEO_NATIVE_SPATIAL_DLL'), 'explicit spatial GPU slot')
class SpatialDevice(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.helper = fixture.IndirectDevice()
        self.scene = self.helper.fixture(self.root)
        self.owners = []
        self.addCleanup(self.close_all)

    def open(self, scene=None, **extra):
        path = Path(os.environ['PHYMEO_NATIVE_SPATIAL_DLL'])
        options = dict(library_path=path, shader_directory=path.parent, device_index=1,
                       validation=True, resident=True, shadow_mode='ray_query',
                       indirect_mode='diffuse_bounce', progressive=True, spatial_filter='guided')
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

    def test_frozen_cpu_kernel_same_exposure_guide_raw_and_later_batches(self):
        path = Path(__file__).parents[1] / 'examples/progressive_first_frame_preview.py'
        spec = importlib.util.spec_from_file_location('frozen_preview', path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        for display in ('clip', 'pbr-neutral'):
            guided = self.open(display_transform=display, studio=StudioLighting(environment_radiance=.03, panel_strength=.3))
            plain = self.open(spatial_filter='none', display_transform=display, studio=StudioLighting(environment_radiance=.03, panel_strength=.3))
            direct = self.open(spatial_filter='none', progressive=False, indirect_mode='none', display_transform=display,
                               studio=StudioLighting(environment_radiance=.03, panel_strength=.3))
            a, b, d = self.shot(guided), self.shot(plain), self.shot(direct)
            raw = guided.read_linear_mean()
            self.assertEqual(raw['data'], plain.read_linear_mean()['data'])
            for channel in fixture.SENSORS:
                self.assertEqual(a[channel], b[channel])
                self.assertEqual(a[channel], d[channel])
            guide = guided.read_spatial_guide()
            self.assertEqual(guide['format'], 'RGBA8_UNORM')
            self.assertEqual(guide['space'], 'display-srgb-direct')
            self.assertIsInstance(guide['data'], bytes)
            valid = np.frombuffer(a[Channel.VALID], np.uint8).reshape(64, 64)
            ids = np.frombuffer(a[Channel.INSTANCE_ID], '<u4').reshape(64, 64)
            depth = np.frombuffer(a[Channel.FORWARD_DEPTH], '<f4').reshape(64, 64)
            linear = np.frombuffer(raw['data'], '<f4').reshape(64, 64, 4)[..., :3]
            g = np.frombuffer(guide['data'], np.uint8).reshape(64, 64, 4)[..., :3]
            direct_rgb = np.frombuffer(d[Channel.RGB], np.uint8).reshape(64, 64, 3)
            self.assertTrue(np.array_equal(g[valid != 0], direct_rgb[valid != 0]))
            filtered = reference.reconstruct(linear, depth, ids, valid, g)
            if display == 'pbr-neutral':
                encoded = np.frombuffer(reference.encode_neutral(filtered, 1.), np.uint8).reshape(64, 64, 3)
            else:
                x = np.clip(filtered.astype(np.float64), 0., 1.)
                encoded = np.rint(np.where(x <= .0031308, 12.92*x, 1.055*x**(1/2.4)-.055)*255).astype(np.uint8)
            actual = np.frombuffer(a[Channel.RGB], np.uint8).reshape(64, 64, 3)
            self.assertLessEqual(int(np.max(np.abs(encoded.astype(int)-actual))), 1)
            plain_rgb = np.frombuffer(b[Channel.RGB], np.uint8).reshape(64, 64, 3)
            self.assertTrue(np.array_equal(actual[valid == 0], plain_rgb[valid == 0]))
            self.assertNotEqual(a[Channel.RGB], b[Channel.RGB])
            self.assertEqual(guided.read_linear_mean()['data'], raw['data'])
            for _ in range(3):
                self.assertEqual(self.shot(guided), self.shot(plain))
                self.assertEqual(guided.read_linear_mean()['data'], plain.read_linear_mean()['data'])
            self.assertFalse(guided.diagnostics()['spatial_filter']['last_capture_filtered'])
            with self.assertRaises(UnsupportedRequestError):
                guided.read_spatial_guide()
            info = guided.diagnostics()['device']['resident_spatial_info']
            self.assertEqual((info['filtered_rgb_captures'], info['rgb_captures'], info['guide_readbacks']), (1, 4, 1))
            self.assertEqual(info['guide_readback_bytes'], 64*64*4)

    def test_sensor_only_rebuild_keeps_previous_rgb_history(self):
        guided = self.open()
        plain = self.open(spatial_filter='none')
        self.shot(guided); self.shot(plain)
        before = guided.read_linear_mean()['data']
        frame, camera = self.helper.frame(1)
        self.shot(guided, (frame, replace(camera, width=63, height=65)), channels=fixture.SENSORS)
        self.assertEqual(guided.read_linear_mean()['data'], before)
        self.assertEqual(self.shot(guided), self.shot(plain))
        self.assertEqual(guided.read_linear_mean()['data'], plain.read_linear_mean()['data'])
        self.assertEqual(guided.diagnostics()['progressive']['batches'], 2)
        self.assertFalse(guided.diagnostics()['spatial_filter']['last_capture_filtered'])

    def test_reset_motion_sensor_and_failed_publication(self):
        guided = self.open()
        first = self.shot(guided)
        raw = guided.read_linear_mean()['data']
        self.shot(guided, channels=fixture.SENSORS)
        self.assertEqual(guided.read_linear_mean()['data'], raw)
        with self.assertRaises(UnsupportedRequestError):
            guided.read_spatial_guide()
        for state in (self.helper.frame(1, wall=.8), self.helper.frame(2, epoch=1)):
            actual = self.shot(guided, state)
            fresh = self.open()
            self.assertEqual(actual, self.shot(fresh, state))
            self.assertTrue(guided.diagnostics()['spatial_filter']['last_capture_filtered'])
        guided.reset_accumulation()
        with self.assertRaises(UnsupportedRequestError):
            guided.read_spatial_guide()
        self.assertEqual(self.shot(guided), first)
        with patch('phymeo_rendering.backends.native.Capture', side_effect=ValueError('publication control')):
            with self.assertRaisesRegex(ValueError, 'publication control'):
                self.shot(guided)
        with self.assertRaises(UnsupportedRequestError):
            guided.read_spatial_guide()
        self.assertEqual(self.shot(guided), first)
        lights = (DirectionalLight((1., 0., 0.), (1., 2., 3.)),)
        guided.set_directional_lights(lights)
        fresh = self.open(replace(self.scene, lights=lights))
        self.assertEqual(self.shot(guided), self.shot(fresh))

    def test_odd_small_extents_background_and_all_variants(self):
        owner = self.open()
        plain = self.open(spatial_filter='none')
        frame, camera = self.helper.frame()
        for width, height in ((1, 1), (3, 2), (63, 65)):
            state = (frame, replace(camera, width=width, height=height))
            a, b = self.shot(owner, state), self.shot(plain, state)
            for channel in fixture.SENSORS:
                self.assertEqual(a[channel], b[channel])
            self.assertEqual(owner.read_linear_mean()['data'], plain.read_linear_mean()['data'])
            self.assertEqual(len(owner.read_spatial_guide()['data']), width*height*4)
        empty = replace(self.scene, instances=())
        for display in ('clip', 'pbr-neutral'):
            a = self.open(empty, display_transform=display, studio=StudioLighting(environment_radiance=.2, panel_strength=0.))
            b = self.open(empty, spatial_filter='none', display_transform=display, studio=StudioLighting(environment_radiance=.2, panel_strength=0.))
            self.assertEqual(self.shot(a), self.shot(b))
        from PIL import Image
        path = self.root / 'normal.png'
        Image.frombytes('RGB', (1, 1), bytes((128, 128, 255))).save(path)
        normal = TextureMap(str(path), hashlib.sha256(path.read_bytes()).hexdigest())
        for mapped in (False, True):
            scene = self.scene if not mapped else replace(self.scene, materials=(
                TexturedMaterial('white', (.7,)*3, 0., 1., normal_map=normal), *self.scene.materials[1:]))
            for grid in (2, 4, 8):
                for reflection in ('environment', 'single_bounce'):
                    a = self.open(scene, indirect_bounces=2, area_light_grid=grid, reflection_mode=reflection)
                    b = self.open(scene, spatial_filter='none', indirect_bounces=2, area_light_grid=grid, reflection_mode=reflection)
                    self.shot(a); self.shot(b)
                    self.assertEqual(a.read_linear_mean()['data'], b.read_linear_mean()['data'])
                    self.assertEqual(self.shot(a), self.shot(b))
                    a.close(); b.close()

    def test_hdr_background_large_finite_values_and_shader_rejection(self):
        from phymeo_rendering.environment import RadianceEnvironment
        import shutil
        hdr = self.root / 'constant.hdr'
        content = b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n' + bytes((128, 80, 40, 130))*32
        hdr.write_bytes(content)
        environment = RadianceEnvironment(str(hdr), hashlib.sha256(content).hexdigest(), face_size=16)
        studio = StudioLighting(environment_map=environment, panel_strength=0.)
        empty = replace(self.scene, instances=(), lights=())
        a = self.open(empty, studio=studio, display_transform='pbr-neutral')
        b = self.open(empty, spatial_filter='none', studio=studio, display_transform='pbr-neutral')
        self.assertEqual(self.shot(a), self.shot(b))
        self.assertEqual(a.read_linear_mean()['data'], b.read_linear_mean()['data'])
        bright = replace(self.scene, instances=(self.scene.instances[0],),
                         lights=(DirectionalLight((0., 0., -1.), (1e38,)*3),))
        high = self.open(bright)
        self.shot(high)
        values = np.frombuffer(high.read_linear_mean()['data'], '<f4')
        self.assertTrue(np.isfinite(values).all())
        self.assertGreater(float(values.max()), 1e37)
        source = Path(os.environ['PHYMEO_NATIVE_SPATIAL_DLL']).parent
        names = ('resident_rayquery.vert.spv', 'resident_rayquery_indirect_progressive.frag.spv',
                 'resident_spatial.comp.spv', 'resident_rayquery.frag.spv')
        for target, replacement, reason in (
            ('resident_spatial.comp.spv', 'resident_progressive.comp.spv', 'compute shader ABI'),
            ('resident_rayquery.frag.spv', 'resident_rayquery_indirect_progressive.frag.spv', 'direct/environment-only'),
        ):
            mixed = self.root / target
            mixed.mkdir()
            for name in names:
                shutil.copyfile(source / name, mixed / name)
            shutil.copyfile(source / replacement, mixed / target)
            with self.assertRaisesRegex(BackendUnavailableError, reason):
                self.open(shader_directory=mixed)

    def test_old_abi_shader_rejection_and_unchanged_default(self):
        old = Path(os.environ['PHYMEO_NATIVE_SPATIAL_OLD_DLL'])
        with self.assertRaisesRegex(BackendUnavailableError, 'spatial'):
            self.open(library_path=old)
        legacy = self.open(library_path=old, shader_directory=old.parent, spatial_filter='none')
        current = self.open(spatial_filter='none')
        for _ in range(2):
            self.assertEqual(self.shot(legacy), self.shot(current))
            self.assertEqual(legacy.read_linear_mean()['data'], current.read_linear_mean()['data'])


if __name__ == '__main__':
    unittest.main()
