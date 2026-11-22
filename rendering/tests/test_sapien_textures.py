"""Host texture upload semantics; real SAPIEN images are separate acceptance."""
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from phymeo_rendering import assets
from phymeo_rendering.backends import _sapien_session as reference
from phymeo_rendering.mesh_data import TriangleMesh
from phymeo_rendering.scene import Instance, VisualScene
from phymeo_rendering import texture_data
from test_texture_data import png_bytes

POSITIONS = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
UV = ((0., 0.), (1., 0.), (0., 1.))


def fixture(texture, factors=((1., 1., 1.),), *, uv=UV):
    materials = tuple(assets.TexturedMaterial(str(i), factor, 0., .5, texture) for i, factor in enumerate(factors))
    source = VisualScene("textures", (assets.Mesh("shared", "unused.gltf"),), materials,
                         tuple(Instance(i + 1, "shared", material.id, "body") for i, material in enumerate(materials)))
    decoded = {"shared": TriangleMesh(POSITIONS, ((0., 0., 1.),)*3, ((0, 1, 2),), "a"*64, uv)}
    return source, decoded


class SapienTextureHostTests(unittest.TestCase):
    def test_missing_uv_is_rejected_before_image_loading_or_device_creation(self):
        source, decoded = fixture(assets.TextureMap("not-opened.png", "a"*64), uv=None)
        with patch.object(texture_data, "load_texture", side_effect=AssertionError("image opened before missing-UV rejection")):
            with self.assertRaisesRegex(ValueError, "TEXCOORD_0|texcoords|UV"):
                reference._prepare_textures(source, decoded)

    def test_one_image_reused_across_instances_but_factor_variants_are_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "corners.png"
            raw = png_bytes()
            path.write_bytes(raw)
            image = assets.TextureMap(path.as_uri(), hashlib.sha256(raw).hexdigest())
            source, decoded = fixture(image, ((1., 1., 1.), (.5, .25, .75), (.5, .25, .75)))
            with patch.object(texture_data, "load_texture", wraps=texture_data.load_texture) as load:
                images, evidence = reference._prepare_textures(source, decoded)
            self.assertEqual(load.call_count, 1)
            self.assertIs(images["0"], images["1"])
            self.assertIs(images["1"], images["2"])
            self.assertEqual(len(evidence["uploads"]), 2)
            self.assertEqual(evidence["decoded_rgb8_bytes"], 12)
            self.assertEqual(evidence["upload_bytes"], 2*2*4 + 2*2*16)
            self.assertEqual([row["materials"] for row in evidence["uploads"]], [["0"], ["1", "2"]])
            with patch.object(reference, "_MAX_TEXTURE_UPLOAD_BYTES", 79):
                with self.assertRaisesRegex(ValueError, "texture.*budget|texture.*limit"):
                    reference._prepare_textures(source, decoded)

    def test_white_factor_keeps_srgb8_and_nonwhite_is_exact_linear_factor(self):
        import numpy as np
        raw = bytes((0, 10, 128, 255, 128, 10, 128, 255, 0, 10, 0, 255))
        image = texture_data.TextureData(2, 2, raw, "a"*64)
        white, format_name, srgb = reference._texture_pixels(image, (1., 1., 1.), np)
        self.assertEqual((format_name, srgb, white.dtype, white.shape), ("R8G8B8A8Unorm", True, np.dtype("uint8"), (2, 2, 4)))
        self.assertEqual(white[..., :3].tobytes(), raw)
        self.assertTrue((white[..., 3] == 255).all())
        factor = (.25, .5, .75)
        linear, format_name, srgb = reference._texture_pixels(image, factor, np)
        self.assertEqual((format_name, srgb, linear.dtype, linear.shape), ("R32G32B32A32Sfloat", False, np.dtype("float32"), (2, 2, 4)))
        # Independent standard-sRGB anchors; these distinguish gamma 2.2, sRGB
        # byte multiplication, applying the factor twice, and quantizing again.
        anchors = {0: 0., 10: 0.003035269835488375, 128: 0.21586050011389926, 255: 1.}
        for i, value in enumerate(raw):
            expected = anchors[value] * factor[i % 3]
            actual = float(linear.reshape(-1, 4)[i // 3, i % 3])
            self.assertEqual(struct.pack("<f", actual), struct.pack("<f", expected))
        self.assertTrue((linear[..., 3] == 1.).all())
        other, _, _ = reference._texture_pixels(image, (.5, .25, .125), np)
        self.assertNotEqual(linear.tobytes(), other.tobytes())

    def test_plain_material_needs_no_image_dependency_or_texture_upload(self):
        material = assets.Material("plain", (.3, .4, .5), .1, .7)
        source = VisualScene("plain", (assets.Mesh("mesh", "unused.gltf"),), (material,), (Instance(1, "mesh", "plain", "body"),))
        decoded = {"mesh": TriangleMesh(POSITIONS, ((0., 0., 1.),)*3, ((0, 1, 2),), "a"*64)}
        with patch.object(texture_data, "load_texture", side_effect=AssertionError("plain material decoded an image")):
            images, evidence = reference._prepare_textures(source, decoded)
        self.assertEqual(images, {})
        self.assertEqual((evidence["uploads"], evidence["upload_bytes"], evidence["decoded_rgb8_bytes"]), ([], 0, 0))


if __name__ == "__main__":
    unittest.main()
