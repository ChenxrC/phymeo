"""Real PNGs and the native packing functions exercise texture admission without a GPU."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from phymeo_rendering import _prepare_cache
from phymeo_rendering._prepare_textures import fit_textures
from phymeo_rendering.assets import Material, Mesh, TextureChannel, TextureMap, TexturedMaterial
from phymeo_rendering.backends import UnsupportedRequestError
from phymeo_rendering.backends._native_normal import pack_normal_maps
from phymeo_rendering.backends._native_texture import pack_textures
from phymeo_rendering.mesh_data import TriangleMesh
from phymeo_rendering.scene import Instance, VisualScene
from phymeo_rendering.texture_data import load_texture
from test_texture_data import png_bytes


CAPS = dict(max_textures=8, max_layers=8, max_width=16, max_height=16, max_array_bytes=4096)
ZERO = dict.fromkeys(CAPS, 0)


class TexturePreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache = self.root/'cache'

    def image(self, name, w, h, pixels=None):
        raw = png_bytes(w, h, pixels if pixels is not None else bytes((23,67,223))*(w*h))
        path = self.root/(name+'.png')
        path.write_bytes(raw)
        return TextureMap(path.as_uri(), sha256(raw).hexdigest())

    def scene(self, materials, ids):
        return VisualScene('source', (Mesh('mesh', 'not-read.gltf'),), tuple(materials),
                           tuple(Instance(i+1, 'mesh', key, 'node') for i, key in enumerate(ids)))

    def fit(self, scene, color=CAPS, normal=CAPS, **kwargs):
        return fit_textures(scene, color_limits=color, normal_limits=normal, cache=self.cache, **kwargs)

    def test_zero_caps_skip_unbound_files_and_keep_scene_identity(self):
        plain = Material('plain', (.4,.3,.2), .1, .6)
        missing = TexturedMaterial('unused',(1.,1.,1.),0.,1., TextureMap('missing.png','0'*64))
        scene = self.scene((plain, missing), ['plain'])
        actual, report = self.fit(scene, ZERO, ZERO)
        self.assertIs(actual, scene)
        self.assertFalse(self.cache.exists())
        self.assertEqual(report['color']['after']['padded_array_bytes'], 0)
        self.assertEqual(report['normal']['after']['layers'], 0)
        json.dumps(report, allow_nan=False)

    def test_crossed_extent_budget_preserves_shared_protected_layer_and_matches_native(self):
        wide = self.image('wide', 8, 2)
        tall = self.image('tall', 2, 8)
        normal = self.image('normal', 6, 3)
        first = TexturedMaterial('print', (1.,1.,1.), 0.,.6, wide, normal_map=normal)
        alias = replace(first, id='alias')
        third = replace(first, id='table', base_color_map=tall)
        scene = self.scene((first, alias, third), ['print','alias','print','table'])
        caps = dict(CAPS, max_array_bytes=480)
        with self.assertRaisesRegex(UnsupportedRequestError, 'strict'):
            self.fit(scene, caps, allow_adapt=False)
        with self.assertRaisesRegex(UnsupportedRequestError, 'protected'):
            self.fit(scene, caps, protected_materials=('print','table'))
        self.assertFalse(self.cache.exists())
        prepared, report = self.fit(scene, caps, protected_materials=('print',))
        self.assertIs(prepared.materials[0], first)
        self.assertIs(prepared.materials[1], alias)
        self.assertIs(prepared.materials[2].normal_map, normal)
        self.assertIs(prepared.instances, scene.instances)
        self.assertNotEqual(prepared.key, scene.key)
        self.assertEqual(report['color']['before']['padded_array_bytes'], 800)
        self.assertEqual(report['color']['after']['padded_array_bytes'], 480)
        mesh = TriangleMesh(((0.,0.,0.),(1.,0.,0.),(0.,1.,0.)), ((0.,0.,1.),)*3,
                            ((0,1,2),), 'a'*64, ((0.,0.),(1.,0.),(0.,1.)))
        for kind, pack, limits in [('color',pack_textures,caps), ('normal',pack_normal_maps,CAPS)]:
            field = 'resident_texture_info' if kind == 'color' else 'resident_normal_info'
            upload = pack(prepared, {'mesh':mesh}, {field:limits})
            after = report[kind]['after']
            self.assertEqual(after['layers'], len(upload.records))
            for key in ('source_rgba_bytes', 'padded_array_bytes'):
                self.assertEqual(after[key], upload.diagnostics[key])
        again, info = self.fit(prepared, caps, protected_materials=('print',))
        self.assertIs(again, prepared)
        self.assertEqual(info['reductions'], [])

    def test_rough_channels_count_as_distinct_layers_and_normals_have_own_capacity(self):
        img = self.image('shared', 3, 2)
        red = TexturedMaterial('red', (1.,1.,1.), 0.,1., img, TextureChannel(img, 'r'), img)
        green = replace(red, id='green', roughness_map=TextureChannel(img, 'g'))
        scene = self.scene((red,green), ('red','green','red'))
        for field in ('max_layers','max_textures'):
            with self.assertRaisesRegex(UnsupportedRequestError, '2 unique layers'):
                self.fit(scene, dict(CAPS, **{field:1}))
        with self.assertRaisesRegex(UnsupportedRequestError, 'normal'):
            self.fit(scene, CAPS, ZERO)
        actual, info = self.fit(scene, normal=dict(CAPS,max_textures=1))
        self.assertIs(actual, scene)
        self.assertEqual(info['color']['after']['layers'], 2)
        self.assertEqual(info['normal']['after']['layers'], 1)

    def test_mismatched_pair_and_pinned_bounded_source_fail_before_derivation(self):
        wide = self.image('wide', 4, 2)
        tall = self.image('tall', 2, 4)
        paint = TexturedMaterial('paint', (1.,1.,1.), 0.,1., wide, TextureChannel(tall, 'g'))
        scene = self.scene((paint,), ('paint',))
        for opts in (dict(allow_adapt=False), dict(protected_materials=('paint',))):
            with self.assertRaisesRegex(UnsupportedRequestError, 'equal dimensions'):
                self.fit(scene, **opts)
        prepared, info = self.fit(scene)
        self.assertEqual(info['color']['paired_size_mismatches'], ['paint'])
        for src in [prepared.materials[0].base_color_map, prepared.materials[0].roughness_map.image]:
            data = load_texture(src)
            self.assertEqual((data.width, data.height), (2,2))
        self.assertEqual(prepared.materials[0].roughness_map.channel, 'g')
        for bad, error in ((replace(wide, sha256='f'*64), 'SHA256'),
                           (self.image('oversize', 4097,1), 'width')):
            bad_scene = self.scene((replace(paint, base_color_map=bad),), ('paint',))
            with self.assertRaisesRegex(ValueError, error):
                self.fit(bad_scene)

    def test_role_specific_filtering_and_atomic_concurrent_cache_publication(self):
        pattern = self.image('bw', 2, 1, bytes((0,0,0,255,255,255)))
        tilted = self.image('tilted', 2, 1, bytes((255,128,128,128,255,128)))
        paint = TexturedMaterial('paint', (1.,1.,1.), 0.,1., pattern, TextureChannel(pattern,'b'), tilted)
        scene = self.scene((paint,), ['paint'])
        one = dict(CAPS, max_width=1)
        def prepare(_):
            return self.fit(scene, one, one)
        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(prepare, range(2)))
        self.assertEqual(outputs[0], outputs[1])
        result, info = outputs[0]
        m = result.materials[0]
        self.assertEqual(load_texture(m.base_color_map).rgb8, bytes((188,188,188)))
        self.assertEqual(load_texture(m.roughness_map.image).rgb8, bytes((128,128,128)))
        self.assertEqual(load_texture(m.normal_map).rgb8, bytes((218,218,128)))
        self.assertEqual({r['role'] for r in info['reductions']}, {'color','roughness','normal'})
        self.assertEqual(len(list(self.cache.iterdir())), 3)
        self.assertEqual(m.roughness_map.channel, 'b')

    def test_repairs_corrupt_cache_and_retries_transient_windows_sharing_denials(self):
        target = self.cache/'texture.png'
        self.cache.mkdir()
        target.write_bytes(b'damaged')
        original = _prepare_cache.os.replace
        def denied():
            error = PermissionError('file is temporarily in use')
            error.winerror = 32
            return error
        tries = []
        def replace_once_unlocked(src, dest):
            tries.append(src)
            if len(tries) == 1:
                raise denied()
            return original(src, dest)
        with patch.object(_prepare_cache.os, 'replace', side_effect=replace_once_unlocked):
            _prepare_cache.publish_bytes(target, b'correct')
        self.assertEqual(target.read_bytes(), b'correct')
        self.assertEqual(len(tries), 2)
        # A reader can also encounter the transient denial while validating an existing winner.
        with patch.object(_prepare_cache, '_matches', side_effect=[denied(), True]) as check:
            _prepare_cache.publish_bytes(target, b'correct')
            self.assertEqual(check.call_count, 2)
        self.assertEqual(list(self.cache.iterdir()), [target])
