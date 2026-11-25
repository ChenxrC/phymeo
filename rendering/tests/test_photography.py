"""Public photographic defaults and explicit overrides without a graphics device."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import tomllib
import unittest
from unittest.mock import patch

from phymeo_rendering import StudioRenderer, photography
from phymeo_rendering.mesh_data import _local_path
from phymeo_rendering.photography import studio_lighting
from phymeo_rendering.scene import FrameKey, Pose


class PhotographyTests(unittest.TestCase):
    def test_default_lights_quality_and_camera(self):
        self.assertIs(StudioRenderer, photography.StudioRenderer)
        renderer = StudioRenderer()
        self.assertTrue(renderer.resident)
        self.assertEqual((renderer.area_light_grid**2, renderer.msaa_samples, renderer.shadow_mode),
                         (16, 4, 'ray_query'))
        self.assertEqual((renderer.display_transform, renderer.exposure), ('pbr-neutral', 7.))
        env = renderer.studio.environment_map
        self.assertEqual(sha256(_local_path(env.uri, suffixes=('.hdr',)).read_bytes()).hexdigest(), env.sha256)
        self.assertEqual((env.strength, env.yaw_degrees, renderer.studio.panel_strength), (.075, 35., .135))
        key = FrameKey('scene','a',0,1,1,'accepted_endpoint')
        camera = renderer.camera('main', key, Pose(), vertical_fov_degrees=90.)
        self.assertEqual(renderer.size, (1440,810))
        self.assertEqual((camera.width, camera.height, camera.cx_px, camera.cy_px), (1440,810,720.,405.))
        self.assertAlmostEqual(camera.fy_px, 405.)
        self.assertEqual(camera.fx_px, camera.fy_px)
        self.assertIs(camera.frame_key, key)

    def test_explicit_options_reach_the_native_backend(self):
        lighting = replace(studio_lighting(), scale_m=2.65)
        renderer = StudioRenderer(size=(1281,721), panel_samples=64, studio=lighting, exposure=2.,
                                  display_transform='clip', msaa_samples=1, shadow_mode='software')
        self.assertIs(renderer.studio, lighting)
        self.assertEqual((renderer.area_light_grid, renderer.msaa_samples, renderer.shadow_mode), (8,1,'software'))
        self.assertEqual((renderer.display_transform, renderer.exposure), ('clip',2.))
        key = FrameKey('scene','b',0,0,0,'authored')
        camera = renderer.camera('tool',key,Pose(),size=(320,240),near_m=.03,far_m=4.)
        self.assertEqual((camera.width,camera.height,camera.near_m,camera.far_m), (320,240,.03,4.))
        self.assertEqual(renderer.size, (1281,721))
        self.assertIsNone(StudioRenderer(studio=None).studio)
        for grid in (2,4,8):
            self.assertEqual(StudioRenderer(area_light_grid=grid).area_light_grid,grid)
            self.assertEqual(StudioRenderer(panel_samples=grid**2,area_light_grid=grid).area_light_grid,grid)

    def test_invalid_or_conflicting_settings_fail(self):
        for kwargs, error in (({'panel_samples':8},ValueError), ({'panel_samples':True},TypeError),
                              ({'panel_samples':64,'area_light_grid':4},ValueError), ({'size':(True,720)},ValueError)):
            with self.subTest(kwargs=kwargs), self.assertRaises(error):
                StudioRenderer(**kwargs)

    def test_wheel_maps_the_pinned_hdr_and_attribution(self):
        config = tomllib.loads((Path(__file__).resolve().parents[1]/'pyproject.toml').read_text(encoding='utf8'))
        package = 'phymeo_rendering._studio_assets'
        setup = config['tool']['setuptools']
        self.assertIn(package,setup['packages'])
        self.assertEqual(setup['package-dir'][package],'examples/materials')
        self.assertEqual(set(setup['package-data'][package]),{'university_workshop_1k.hdr','NOTICE.md'})
        installed = Path(__file__).resolve().parent/'site-packages'/'phymeo_rendering'
        with patch.object(photography,'__file__',str(installed/'photography.py')):
            self.assertEqual(_local_path(studio_lighting().environment_map.uri, suffixes=('.hdr',)),
                             installed/'_studio_assets'/'university_workshop_1k.hdr')
