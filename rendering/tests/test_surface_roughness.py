"""A data-channel texture describes linear roughness independently of color."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from phymeo_rendering import assets
from phymeo_rendering.backends._native_texture import pack_textures
from test_sapien_textures import fixture
from test_texture_data import png_bytes


class SurfaceRoughness(TestCase):
    def test_linear_channel_declaration_can_be_used_without_color_texture(self):
        image = assets.TextureMap("map.png", "a"*64)
        rough = assets.TextureChannel(image, "g")
        material = assets.TexturedMaterial("coat", (.3,.5,.8), .25, .7, roughness_map=rough)
        self.assertIsNone(material.base_color_map)
        self.assertEqual(material.roughness_map.image, image)
        with self.assertRaises(ValueError):
            assets.TextureChannel(image, "a")
        with self.assertRaises(TypeError):
            replace(material, roughness_map=image)

    def test_native_color_and_linear_roughness_share_one_uv_lookup(self):
        from phymeo_rendering.texture_data import load_texture

        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "color.png"
            source = png_bytes()
            path.write_bytes(source)
            image = assets.TextureMap(path.as_uri(), sha256(source).hexdigest())
            scene, decoded = fixture(image)
            material = replace(scene.materials[0], roughness_map=assets.TextureChannel(image,"g"))
            scene = replace(scene, materials=(material,))
            upload = pack_textures(scene, decoded)
            rgb = load_texture(image).rgb8
            self.assertEqual(bytes(upload.rgba)[3::4], rgb[1::3])
            self.assertEqual(bytes(upload.rgba)[0::4], rgb[0::3])
            self.assertEqual(upload.records[0].reserved, 1)
            self.assertTrue(upload.uses_roughness)
            plain = pack_textures(replace(scene, materials=(replace(material, roughness_map=None),)), decoded)
            self.assertFalse(plain.uses_roughness)
            self.assertEqual(set(bytes(plain.rgba)[3::4]), {255})
            data_only = pack_textures(replace(scene, materials=(replace(material, base_color_map=None),)), decoded)
            self.assertEqual(set(bytes(data_only.rgba)[0::4]), {255})
            self.assertEqual(bytes(data_only.rgba)[3::4], rgb[1::3])

    def test_reference_roughness_uses_linear_channel_and_factor_once(self):
        import numpy as np
        from phymeo_rendering.backends._sapien_session import _roughness_pixels
        from phymeo_rendering.texture_data import TextureData

        source = TextureData(2,1,bytes((255,128,0,0,64,255)),"a"*64)
        values, gpu_format = _roughness_pixels(source,"g",.5,np)
        self.assertEqual(gpu_format, "R32Sfloat")
        np.testing.assert_array_equal(values.reshape(-1), np.array([64/255,32/255],dtype=np.float32))

    def test_pair_capacity_and_mismatched_dimensions_fail_before_upload(self):
        from phymeo_rendering.backends import UnsupportedRequestError

        with TemporaryDirectory() as temporary:
            def descriptor(name,content):
                path=Path(temporary)/name
                path.write_bytes(content)
                return assets.TextureMap(path.as_uri(),sha256(content).hexdigest())
            color=descriptor("color.png",png_bytes())
            small=descriptor("rough.png",png_bytes(1,1,bytes((30,60,90))))
            scene,decoded=fixture(color,((1.,1.,1.),(.5,.5,.5)))
            second=replace(scene.materials[1],roughness_map=assets.TextureChannel(small))
            scene=replace(scene,materials=(scene.materials[0],second))
            with self.assertRaisesRegex(UnsupportedRequestError,"count"):
                pack_textures(scene,decoded,{"resident_texture_info":{"max_layers":1}})
            with self.assertRaisesRegex(UnsupportedRequestError,"equal.*dimensions"):
                pack_textures(scene,decoded)
